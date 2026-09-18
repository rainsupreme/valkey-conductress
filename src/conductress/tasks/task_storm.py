"""Connection-storm benchmark task.

Measures how a server behaves under a connection storm: a burst of clients
connect, each issues one command, and any client that times out or is refused
reconnects according to a back-off policy. A stall injector makes the server's
main thread unavailable for a bounded window partway through the run, which is
what turns a burst into a storm.

Three mechanisms combine to make the storm costly, and the task measures all
three:

1. Kernel listen-queue overflow. While the main thread is stalled it stops
   calling ``accept()``; completed connections fill the kernel accept backlog
   and, once it is full, are dropped. Counted from ``/proc/net/netstat``
   (``ListenOverflows``/``ListenDrops``) as a before/after delta.
2. First reply needs the main thread. A newly accepted connection's first
   command is served on the main thread, so while it is stalled every fresh
   client blocks and eventually times out and reconnects -- the offered
   connection load amplifies far above the client population.
3. Post-recovery tail. Clients that already gave up leave work (half-open
   connections, queued accepts) the server must still drain after the stall
   clears, so throughput and connect latency recover over a measurable window
   rather than instantly.

Primary score: ``recovery_seconds`` -- wall time from the stall clearing until
the storm is quiescent (every client connected, no attempt in flight).
Lower is better. The task records the number as the score directly and does
not modify any sweep coordinator; storm results are their own series and are
not sweep-comparable.

Phases per run:
    1. start the server (single instance, loopback)
    2. prefill the one key the storm's first command reads
    3. start an INFO sampler on a persistent connection (per tick:
       connected_clients, total_connections_received, rejected_connections;
       gaps recorded when the sampler is itself stalled)
    4. take the kernel listen-overflow baseline
    5. run the storm generator subprocess with the stall injector
    6. read the generator's JSON, take the listen-overflow delta
    7. assemble results, stop the server

Extending: additional stall injectors (busy Lua, large scans), a background
closed-loop load, and cluster mode are natural follow-ups; see docs.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import List

from conductress.config import ServerInfo
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.stormgen.policy import parse_policy
from conductress.stormgen.stall import parse_stall
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.utility import RealtimeCommand

logger = logging.getLogger(__name__)

METHOD = "storm"

# The single key the storm's default first command (GET) reads. Prefilled so a
# healthy first command is a hit, not a miss -- the storm measures connection
# behaviour, not key population.
STORM_KEY = "stormkey"
DEFAULT_FIRST_COMMAND = f"GET {STORM_KEY}"
# INFO fields sampled each tick; names are stable across Redis and Valkey.
SAMPLED_INFO_FIELDS = ("connected_clients", "total_connections_received", "rejected_connections")


def parse_handshake(spec: str) -> List[str]:
    """Split a ``;``-separated handshake spec into commands.

    ``"HELLO 3"`` -> ``["HELLO 3"]``; ``""`` -> ``[]`` (no handshake);
    ``"HELLO 3;AUTH u p"`` -> ``["HELLO 3", "AUTH u p"]``.
    """
    return [part.strip() for part in (spec or "").split(";") if part.strip()]


@dataclass
class StormTaskData(BaseTaskData):
    """Task data for the connection-storm benchmark. See module docstring.

    Every field carries a default: the queue serializes via ``asdict`` and a
    field without a default would be dropped from a persisted task, so the
    round-trip test in the unit suite guards this invariant.
    """

    io_threads: int = 1
    server_args: str = ""
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""
    # Storm shape
    clients: int = 2000
    burst_ms: int = 200
    connect_timeout_ms: float = 1000.0
    reply_timeout_ms: float = 500.0
    policy: str = "fixed:200"
    handshake: str = "HELLO 3"  # ';'-separated; '' = no handshake
    first_command: str = DEFAULT_FIRST_COMMAND
    duration_s: float = 20.0
    stall: str = "none"  # none | debug-sleep:<seconds>
    stall_after_s: float = 5.0
    workers: int = 1
    bind_addrs: str = ""  # ','-separated loopback source addresses; '' = none
    tick_ms: int = 100
    # Background closed-loop load is a v1 non-goal; kept as an explicit field so
    # the queue round-trip is stable when it is added.
    background_load: str = "none"

    def __post_init__(self):
        super().__post_init__()
        self.duration_s = float(self.duration_s)
        self.stall_after_s = float(self.stall_after_s)
        if self.clients < 1:
            raise ValueError(f"clients must be >= 1, got {self.clients}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        if self.burst_ms < 0:
            raise ValueError(f"burst_ms must be >= 0, got {self.burst_ms}")
        if self.connect_timeout_ms <= 0 or self.reply_timeout_ms <= 0:
            raise ValueError("connect_timeout_ms and reply_timeout_ms must be > 0")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.tick_ms <= 0:
            raise ValueError(f"tick_ms must be > 0, got {self.tick_ms}")
        if self.stall_after_s < 0 or self.stall_after_s >= self.duration_s:
            raise ValueError(
                f"stall_after_s ({self.stall_after_s}) must be in [0, duration_s={self.duration_s}); "
                "the stall has to fire and clear inside the run"
            )
        if self.background_load != "none":
            raise ValueError("background_load other than 'none' is not implemented yet")
        # Validate the policy and stall specs at submission so a typo fails
        # before a runner ever claims the task.
        parse_policy(self.policy)
        parse_stall(self.stall)

    def handshake_commands(self) -> List[str]:
        """Handshake commands as a list (empty means no handshake)."""
        return parse_handshake(self.handshake)

    def bind_addr_list(self) -> List[str]:
        """Loopback source addresses to spread ephemeral ports across (may be empty)."""
        return [a.strip() for a in self.bind_addrs.split(",") if a.strip()]

    def short_description(self) -> str:
        stall_desc = parse_stall(self.stall).describe()
        return (
            f"storm {self.clients} clients, burst {self.burst_ms}ms, {self.policy} reconnect, "
            f"stall {stall_desc}, io-threads {self.io_threads}, {self.duration_s:g}s"
        )

    def prepare_task_runner(self, server_infos: List[ServerInfo]) -> "StormTaskRunner":
        return StormTaskRunner(self, server_infos)


@dataclass
class _StormOutcome:
    """The generator's parsed document plus the sampler and netstat series."""

    document: dict
    info_samples: List[dict]
    sampler_gaps: List[dict]


class StormTaskRunner(BaseTaskRunner):
    """Run the connection-storm benchmark. See module docstring for the phases."""

    def __init__(self, task: StormTaskData, server_infos: List[ServerInfo]):
        super().__init__(task.task_id)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not server_infos:
            raise ValueError("storm task needs a server host")
        self.task = task
        self.server_infos = server_infos
        self.commit_hash = ""
        self.title = (
            f"storm, {task.source}:{task.specifier}, {task.clients} clients, "
            f"burst {task.burst_ms}ms, {task.policy} reconnect, "
            f"stall {parse_stall(task.stall).describe()}, io-threads={task.io_threads}, {task.duration_s:g}s"
        )
        self.status = BenchmarkStatus(steps_total=max(1, int(task.duration_s)), task_type=METHOD)

    # ------------------------------------------------------------------ sampler

    async def _sample_loop(self, server: "Server", samples: List[dict], gaps: List[dict], stop: asyncio.Event) -> None:
        """Poll INFO clients+stats each tick until ``stop`` is set.

        Each sample stamps a monotonic time and the sampled fields. When a poll
        takes longer than one tick (the sampler's own connection blocked behind
        the stalled main thread) the overrun is recorded as a gap, so a reader
        can tell "server quiet" from "sampler could not observe it".
        """
        interval = self.task.tick_ms / 1000.0
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                clients = await server.info("clients")
                stats = await server.info("stats")
                merged = {**clients, **stats}
                sample: dict = {"t": time.monotonic()}
                for field in SAMPLED_INFO_FIELDS:
                    if field in merged:
                        try:
                            sample[field] = int(merged[field])
                        except ValueError:
                            sample[field] = merged[field]
                samples.append(sample)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("INFO sample failed: %s", exc)
            elapsed = time.monotonic() - t0
            if elapsed > interval * 2:
                gaps.append({"t": t0, "elapsed_s": round(elapsed, 4)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval - elapsed))
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------ generator subprocess

    def _stormgen_command(self, host: str, port: int, json_path: str) -> str:
        """Build the ``python -m conductress.stormgen`` command line for this task.

        The generator runs as a subprocess (its own asyncio/multiprocessing
        lifecycle) rather than in the runner's loop, matching how other tasks
        shell out to their load generators.
        """
        task = self.task
        parts = [
            sys.executable,
            "-m",
            "conductress.stormgen",
            "--host",
            host,
            "--port",
            str(port),
            "--clients",
            str(task.clients),
            "--burst-ms",
            str(task.burst_ms),
            "--connect-timeout-ms",
            _fmt(task.connect_timeout_ms),
            "--reply-timeout-ms",
            _fmt(task.reply_timeout_ms),
            "--policy",
            task.policy,
            "--first-command",
            task.first_command,
            "--duration-s",
            _fmt(task.duration_s),
            "--stall",
            task.stall,
            "--stall-after-s",
            _fmt(task.stall_after_s),
            "--workers",
            str(task.workers),
            "--bucket-ms",
            str(task.tick_ms),
            "--json",
            json_path,
        ]
        handshake = task.handshake_commands()
        if handshake:
            for command in handshake:
                parts += ["--handshake", command]
        else:
            parts += ["--handshake", ""]
        for addr in task.bind_addr_list():
            parts += ["--bind-addr", addr]
        return " ".join(_shell_quote(p) for p in parts)

    async def _run_generator(self, server: "Server") -> _StormOutcome:
        """Phases 3-6: sampler + netstat baseline, run the generator, collect its JSON and the delta."""
        host = server.ip
        port = server.port or 6379
        json_path = str(self.file_protocol.work_dir / "storm_result.json")

        samples: List[dict] = []
        gaps: List[dict] = []
        stop = asyncio.Event()
        sampler = asyncio.ensure_future(self._sample_loop(server, samples, gaps, stop))

        command_string = self._stormgen_command(host, port, json_path)
        self.logger.info("launching storm generator: %s", command_string)
        self.status.state = "running"
        self.file_protocol.write_status(self.status)

        output_lines: List[str] = []
        command = RealtimeCommand(command_string)
        try:
            command.start()
            while command.is_running():
                line, _ = command.poll_output()
                while line:
                    output_lines.append(line)
                    line, _ = command.poll_output()
                await asyncio.sleep(0.5)
            line, _ = command.poll_output()
            while line:
                output_lines.append(line)
                line, _ = command.poll_output()
        finally:
            stop.set()
            try:
                await sampler
            except asyncio.CancelledError:
                pass

        proc = command.p
        exit_code = getattr(proc, "returncode", None)
        if exit_code != 0:
            joined = "\n".join(output_lines)
            raise RuntimeError(f"storm generator exited with code {exit_code}. Output:\n{joined[-2000:]}")

        document = self._read_document(json_path, output_lines)
        return _StormOutcome(document=document, info_samples=samples, sampler_gaps=gaps)

    @staticmethod
    def _read_document(json_path: str, output_lines: List[str]) -> dict:
        """Load the generator's JSON document from its file, or fall back to stdout."""
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            joined = "\n".join(output_lines)
            start = joined.find("{")
            if start >= 0:
                try:
                    return json.loads(joined[start:])
                except json.JSONDecodeError as exc:
                    raise RuntimeError("storm generator produced no parseable JSON result") from exc
            raise RuntimeError("storm generator produced no JSON result")

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        """Start the server, prefill, run the storm, record once, stop the server."""
        self.logger.info("preparing: %s", self.title)
        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos,
            self.task.source,
            self.task.specifier,
            self.task.io_threads,
            self.task.make_args,
            server_cpu_override=self.task.server_cpu_override,
            server_args=self.task.server_args,
        )
        try:
            await replication_group.kill_all_valkey_instances()
            await replication_group.start()
            if not replication_group.primary:
                raise RuntimeError("Server failed to start: no primary")
            server = replication_group.primary
            self.commit_hash = server.get_build_hash() or ""

            await self._prefill_key(server)
            outcome = await self._run_generator(server)
            await self._record_result(server, outcome)

            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = self.status.steps_total
            self.file_protocol.write_status(self.status)
        finally:
            await replication_group.stop_all_servers()

    async def _prefill_key(self, server: "Server") -> None:
        """Set the single key the storm's default first command reads.

        Only meaningful for the default GET-style first command; a custom first
        command is the operator's responsibility.
        """
        await server.run_valkey_command(f"SET {STORM_KEY} stormvalue")

    async def _record_result(self, server: "Server", outcome: _StormOutcome) -> None:
        """Assemble the result row; score is recovery_seconds (lower is better)."""
        completion_time = datetime.datetime.now()
        lscpu_output, _ = await server.run_host_command("lscpu")

        doc = outcome.document
        reduced = doc.get("metrics", {})
        totals = reduced.get("totals", {})
        quiescence = reduced.get("time_to_quiescence", {})
        recovery_seconds = quiescence.get("from_stall_end")
        # When there is no stall, recovery-from-stall is undefined; fall back to
        # quiescence from the storm start so the score is always a number.
        score = recovery_seconds if recovery_seconds is not None else quiescence.get("from_start")
        if score is None:
            score = float(self.task.duration_s)  # never reached quiescence within the run

        detailed = {
            "io-threads": self.task.io_threads,
            "clients": self.task.clients,
            "burst_ms": self.task.burst_ms,
            "connect_timeout_ms": self.task.connect_timeout_ms,
            "reply_timeout_ms": self.task.reply_timeout_ms,
            "policy": self.task.policy,
            "handshake": self.task.handshake_commands(),
            "first_command": self.task.first_command,
            "duration_s": self.task.duration_s,
            "stall": parse_stall(self.task.stall).describe(),
            "stall_after_s": self.task.stall_after_s,
            "workers": self.task.workers,
            "bind_addrs": self.task.bind_addr_list(),
            "tick_ms": self.task.tick_ms,
            "server_args": self.task.server_args,
            "recovery_seconds": recovery_seconds,
            "recovery_lower_is_better": True,
            "amplification": totals.get("amplification"),
            "attempts": totals.get("attempts"),
            "connected_clients": totals.get("connected_clients"),
            "outcomes": reduced.get("outcomes"),
            "attempt_latency": reduced.get("attempt_latency"),
            "time_to_quiescence": quiescence,
            "timeline": reduced.get("timeline"),
            "stall_record": doc.get("stall"),
            "listen_overflow_delta": doc.get("listen_overflow_delta"),
            "info_timeline": outcome.info_samples,
            "info_sampler_gaps": outcome.sampler_gaps,
            "schema_version": doc.get("schema_version"),
            "generator_config": doc.get("config"),
            "server_cpus": server.server_cpus,
            "lscpu": lscpu_output,
        }

        results = BenchmarkResults(
            method=METHOD,
            source=self.task.source,
            specifier=self.task.specifier,
            commit_hash=self.commit_hash,
            score=float(score),
            end_time=completion_time,
            data=detailed,
            make_args=self.task.make_args,
            note=self.task.note,
        )
        self.file_protocol.write_results(results)


def _fmt(value: float) -> str:
    """Render a float without a trailing ``.0`` for whole numbers."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _shell_quote(token: str) -> str:
    """Single-quote a token for a shell command line if it needs quoting."""
    if token and all(c.isalnum() or c in "._-/=:" for c in token):
        return token
    return "'" + token.replace("'", "'\\''") + "'"
