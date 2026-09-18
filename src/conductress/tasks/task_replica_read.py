"""Replica-read benchmark task.

Measures read throughput served by a REPLICA while its primary ingests writes
at a fixed rate over a real replication link. On a replica the only writer is
the replication stream, executed on the main thread, so no client connection
ever interleaves writes with reads. That makes it the cleanest topology for
measuring any change to how a server serves reads while writes are arriving.

Topology (one host, loopback, all instances pinned by the CPU allocator):

    primary   127.0.0.1:<base>      <- writer:  cachecannon SET, rate_limit=W
    replica   127.0.0.1:<base+1>    <- reader:  cachecannon GET, closed loop
    [replica  127.0.0.1:<base+2> ...]  extra replicas add fan-out load only

Phases per repetition:
    1. kill stray servers, start topology, REPLICAOF, wait for link up
    2. preload: cachecannon prefill of the keyspace against the primary,
       then wait until every replica's offset equals the primary's
    3. measure: start the fixed-rate writer at the primary, then the reader
       at the first replica (warmup + duration); sample INFO and CPU counters
       from every instance and both generators on a fixed cadence for the
       whole reader window
    4. guards: reader 0% errors and >= 99% hit rate; writer 0% errors and
       achieved rate within tolerance of the target (an under-delivering
       writer silently turns the run into a lower-write-rate run)
    5. bottleneck verdict: was anything other than the server the limit?
       Recorded on the result, never a failure (see cpu_sampling)

Score = mean reader throughput (rps). Results carry the writer's achieved
rate, replication-lag statistics (primary minus replica offset, bytes), the
per-instance INFO series, the verdict with its evidence and both TOML
configs, so a run can be audited without repeating it.

Independent variables the task is designed around: replica io-threads, write
rate, and replica-only server arguments. An A/B is two queued tasks that
differ in --specifier (two builds) or --replica-args (one build, two configs).

Cluster mode: the TopologySpec can describe it but the bootstrap is a
follow-up (see topology.py).
"""

import asyncio
import datetime
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional

from conductress import cpu_sampling
from conductress.cachecannon import DEFAULT_CACHECANNON_BINARY, generate_toml_config, parse_json_results
from conductress.config import (
    DEFAULT_DURATION,
    DEFAULT_REPETITIONS,
    DEFAULT_VAL_SIZE,
    DEFAULT_WARMUP,
    PERF_BENCH_KEYSPACE,
    ServerInfo,
)
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.topology import DEFAULT_BASE_PORT, TopologyGroup, TopologySpec, replication_lag_stats
from conductress.utility import HumanByte, HumanTime, RealtimeCommand, parse_cpulist

logger = logging.getLogger(__name__)

METHOD = "replica-read"

# Writer runs longer than the reader so the reader's whole window sees the
# write stream, and the writer still exits on its own (its JSON result is how
# we learn the achieved rate).
WRITER_SLACK_SECONDS = 5
# Fraction of the target write rate the writer must achieve for a valid cell.
WRITE_RATE_TOLERANCE = 0.10
# Preload: a short cachecannon run whose only job is the prefill.
PRELOAD_DURATION_SECONDS = 1
# Reader must hit: every key was written to the primary and replicated.
READER_MIN_HIT_RATE_PCT = 99.0
# Cores the runner process confines itself to while a task runs. Its steady
# work (log and status writes, the per-second sampler) measured ~0.35 of a
# core on a 192-CPU runner and otherwise lands on an unallocated core, where
# the bottleneck verdict correctly reports it as foreign interference.
RUNNER_MANAGEMENT_CPUS = 2


def current_affinity_cpulist() -> str:
    """This process's CPU mask as a compact cpulist (e.g. ``0-191``)."""
    return format_cpulist(sorted(os.sched_getaffinity(0)))


def pin_current_process(cpulist: str) -> None:
    """Confine every thread of this process to ``cpulist`` (``taskset -a``)."""
    subprocess.run(["taskset", "-acp", cpulist, str(os.getpid())], check=True, capture_output=True)


def format_cpulist(cpus: list) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    ranges: list = []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def _io_threads_of(inst) -> int:
    if inst.io_threads is None:
        raise ValueError(f"instance on port {inst.port} has no io_threads")
    return inst.io_threads


@dataclass
class ReplicaReadTaskData(BaseTaskData):
    """Task data for the replica-read benchmark. See module docstring."""

    # Reader (measured) side -- aimed at the first replica
    val_size: int = DEFAULT_VAL_SIZE
    pipelining: int = 1
    connections: int = 400
    threads: int = 8
    keyspace_count: int = PERF_BENCH_KEYSPACE
    warmup: int = DEFAULT_WARMUP
    duration: int = DEFAULT_DURATION
    repetitions: int = DEFAULT_REPETITIONS
    # Writer side -- fixed rate at the primary
    write_rate: int = 50_000
    write_connections: int = 16
    write_threads: int = 4
    write_pipelining: int = 1
    # Topology: the inherited ``topology`` field, built with
    # ``TopologySpec.replica_read`` (primary + replicas on consecutive ports of
    # the runner host, each with its own working directory). The properties
    # below read the levers back out of it.
    # Sampling
    sample_interval: float = 1.0
    info_fields: str = ""  # comma-separated extra INFO fields to sample (e.g. counters a build under test exposes)
    cachecannon_binary: str = DEFAULT_CACHECANNON_BINARY
    benchmark_cpu_override: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.warmup = int(self.warmup)
        self.duration = int(self.duration)
        if not self.topology.replicas:
            raise ValueError("replica-read needs at least one replica in its topology")
        if self.topology.host_count() != 1:
            raise ValueError(
                "replica-read runs every instance on the runner host; the topology must not name other hosts"
            )
        if any(i.io_threads is None for i in self.topology.instances):
            raise ValueError("every replica-read instance must set io_threads")
        if self.write_rate < 1:
            raise ValueError(f"write_rate must be >= 1 req/s, got {self.write_rate}")
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.duration < 1:
            raise ValueError(f"duration must be >= 1s, got {self.duration}")
        if self.sample_interval <= 0:
            raise ValueError(f"sample_interval must be > 0, got {self.sample_interval}")
        for name in ("connections", "threads", "write_connections", "write_threads", "pipelining", "write_pipelining"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    @property
    def replica_count(self) -> int:
        """Replica instances on the runner host; reads are measured at the first."""
        return len(self.topology.replicas)

    @property
    def io_threads(self) -> int:
        """Replica io-threads (the measured instance)."""
        return _io_threads_of(self.topology.replicas[0])

    @property
    def primary_io_threads(self) -> int:
        return _io_threads_of(self.topology.primary)

    @property
    def base_port(self) -> int:
        return self.topology.primary.port

    def topology_spec(self) -> TopologySpec:
        """The instance layout (the inherited ``topology`` field)."""
        return self.topology

    def extra_info_fields(self) -> list:
        """Extra INFO field names to sample from every instance."""
        return [f.strip() for f in self.info_fields.split(",") if f.strip()]

    def short_description(self) -> str:
        return (
            f"replica-read {self.replica_count}r io{self.io_threads}, writes {self.write_rate}/s, "
            f"{HumanByte.to_human(self.val_size)} values, P{self.pipelining}, {self.connections}c, "
            f"{self.threads}t, {HumanTime.to_human(self.duration)} x{self.repetitions}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "ReplicaReadTaskRunner":
        return ReplicaReadTaskRunner(self, server_infos)


@dataclass(frozen=True)
class _Placement:
    """Where the generators and the runner itself run (cpulists; empty means unpinned).

    ``runner_cpus`` are the management cores this process (its log/status
    writes and the per-second sampler) is confined to for the task's duration,
    so the bottleneck verdict can count them as claimed rather than foreign.
    ``launch_cpus`` is the mask this process had before pinning; generators are
    launched under it because a child inherits its parent's mask at fork and
    must not be dragged onto the management cores.
    """

    reader_cpus: str
    writer_cpus: str
    runner_cpus: str = ""
    launch_cpus: str = ""


@dataclass(frozen=True)
class _Measurement:
    """Everything one measure phase produced, handed to the judge phase."""

    reader: dict
    writer: dict
    samples: list
    reader_started: float  # monotonic; the verdict window opens at reader_started + warmup
    reader_toml: str
    writer_toml: str


def _primary_and_replica(group: TopologyGroup) -> tuple:
    """The primary and the measured (first) replica of a started topology."""
    if group.primary is None or not group.replicas:
        raise RuntimeError("topology not started")
    return group.primary, group.replicas[0]


def _endpoint(server: Server) -> str:
    return f"{server.ip}:{server.port}"


class ReplicaReadTaskRunner(BaseTaskRunner):
    """Run the replica-read benchmark. See module docstring for the phases."""

    def __init__(self, task: ReplicaReadTaskData, server_infos: list[ServerInfo]):
        super().__init__(task.task_id)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task = task
        if not server_infos:
            raise ValueError("replica-read needs a server host")
        self.host = server_infos[0]
        self.spec = task.topology_spec()
        self.commit_hash = ""
        # Generator placement: allocated on the first rep, released when the run ends.
        self._client: Optional[Server] = None
        self._reader_tag: Optional[AllocationTag] = None
        self._writer_tag: Optional[AllocationTag] = None
        # Management cores for this process; ``_launch_cpus`` is the mask to restore.
        self._runner_tag: Optional[AllocationTag] = None
        self._launch_cpus: str = ""
        self.title = (
            f"replica-read, {task.source}:{task.specifier}, {task.replica_count} replica(s), "
            f"replica io-threads={task.io_threads}, writes {task.write_rate}/s, "
            f"P{task.pipelining}, {task.connections}c, {task.threads}t, "
            f"{HumanTime.to_human(task.duration)} x{task.repetitions}"
        )
        self.status = BenchmarkStatus(
            steps_total=(task.warmup + task.duration + WRITER_SLACK_SECONDS) * task.repetitions,
            task_type=METHOD,
        )

    # ------------------------------------------------------------------ CPU placement

    def _allocate_client_cpus(self, client: Server, purpose: str, count: int) -> Optional[AllocationTag]:
        """Allocate ``count`` CPUs for a generator, away from every server instance (None under --client-cpus)."""
        if self.task.benchmark_cpu_override:
            return None
        return self._allocate(client, purpose, count)

    def _allocate(self, client: Server, purpose: str, count: int) -> AllocationTag:
        """Allocate ``count`` client-side CPUs tagged ``purpose``, away from every server instance."""
        server_tags = [
            AllocationTag(task_id=f"server_{self.host.ip}_{i.port}", purpose="server") for i in self.spec.instances
        ]
        tag = AllocationTag(task_id=f"{self.task_name}_{purpose}", purpose="benchmark")
        cpus = client.allocate_client_cpus(tag, count, avoid_tags=server_tags)
        self.logger.info("Allocated CPUs %s for %s", cpus, purpose)
        return tag

    def _cpu_list(self, client: Server, tag: Optional[AllocationTag]) -> str:
        if self.task.benchmark_cpu_override:
            return self.task.benchmark_cpu_override
        return client.allocated_cpu_list(tag) if tag else ""

    # ------------------------------------------------------------------ generators

    def _toml(self, name: str, content: str) -> str:
        path = self.file_protocol.work_dir / name
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(path)

    def _launch(self, toml_path: str) -> RealtimeCommand:
        prefix = f"taskset -c {self._launch_cpus} " if self._launch_cpus else ""
        command = RealtimeCommand(f"{prefix}{self.task.cachecannon_binary} {toml_path}")
        command.start()
        return command

    @staticmethod
    def _pid(command: Optional[RealtimeCommand]) -> Optional[int]:
        """OS pid of a launched generator, or None if it has not started."""
        if command is None:
            return None
        return getattr(command.p, "pid", None)

    @staticmethod
    def _drain(command: RealtimeCommand, lines: list) -> None:
        line, _ = command.poll_output()
        while line:
            lines.append(line)
            line, _ = command.poll_output()

    async def _wait_and_parse(self, command: RealtimeCommand, label: str) -> dict:
        """Wait for a generator to exit, then parse its JSON result."""
        lines: list = []
        while command.is_running():
            self._drain(command, lines)
            await asyncio.sleep(0.5)
        self._drain(command, lines)
        output = "\n".join(lines)
        exit_code = getattr(command.p, "returncode", None)
        if exit_code != 0:
            raise RuntimeError(f"{label} cachecannon exited with code {exit_code}. Output:\n{output[-2000:]}")
        parsed = parse_json_results(output)
        if parsed["error_pct"] > 0:
            raise RuntimeError(f"{label} cachecannon reported {parsed['error_pct']}% errors. Output:\n{output[-1000:]}")
        return parsed

    # ------------------------------------------------------------------ run

    async def run(self):
        """Bring up, preload, measure, judge, repeat; record once at the end."""
        task = self.task
        self.logger.info("preparing: %s", self.title)
        self.file_protocol.write_status(self.status)

        group = TopologyGroup(self.host, self.spec, task.source, task.specifier, task.make_args)
        reps: list = []
        last: Optional[_Measurement] = None
        try:
            for rep in range(1, task.repetitions + 1):
                await self._bring_up(group, first=rep == 1)
                placement = await self._place_generators()
                await self._preload(group, rep, placement)
                last = await self._measure(group, rep, placement)
                reps.append(self._judge(group, rep, placement, last))
            assert last is not None
            await self._record_result(group, reps, last.reader_toml, last.writer_toml)
            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = self.status.steps_total
            self.file_protocol.write_status(self.status)
        finally:
            await group.stop_all_servers()
            self._release_generators()

    # ------------------------------------------------------------------ phases

    async def _bring_up(self, group: TopologyGroup, first: bool) -> None:
        """Phase 1: fresh instances, replication wired and synced; records the build hash."""
        if not first:
            await group.stop_all_servers()
            await Server(self.host.ip, username=self.host.username).run_host_command(
                "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False
            )
        await group.kill_all_valkey_instances()
        await group.start()
        await group.begin_replication()
        await group.wait_for_repl_sync()
        assert group.primary is not None
        self.commit_hash = group.primary.get_build_hash() or ""

    async def _place_generators(self) -> "_Placement":
        """Allocate generator and management CPUs once (first call) and return their cpulists.

        The allocation outlives the per-rep topology so that every rep's
        generators sit on the same cores; ``_release_generators`` undoes it.
        The runner process pins itself to the management cores for the whole
        task: its own work (log and status writes, the per-second sampler)
        otherwise lands on arbitrary cores and reads as foreign interference
        in the bottleneck verdict.
        """
        if self._client is None:
            self._client = Server("127.0.0.1")
            await self._client.ensure_host_cpu_allocation()
            self._reader_tag = self._allocate_client_cpus(self._client, "reader", self.task.threads)
            self._writer_tag = self._allocate_client_cpus(self._client, "writer", self.task.write_threads)
            self._runner_tag = self._allocate(self._client, "runner", RUNNER_MANAGEMENT_CPUS)
            runner_cpus = self._client.allocated_cpu_list(self._runner_tag) if self._runner_tag else ""
            if runner_cpus:
                self._launch_cpus = current_affinity_cpulist()
                pin_current_process(runner_cpus)
                self.logger.info("Pinned runner to management CPUs %s (was %s)", runner_cpus, self._launch_cpus)
        return _Placement(
            reader_cpus=self._cpu_list(self._client, self._reader_tag),
            writer_cpus=self._cpu_list(self._client, self._writer_tag),
            runner_cpus=self._client.allocated_cpu_list(self._runner_tag) if self._runner_tag else "",
            launch_cpus=self._launch_cpus,
        )

    def _release_generators(self) -> None:
        if self._client is None:
            return
        if self._launch_cpus:
            pin_current_process(self._launch_cpus)
            self._launch_cpus = ""
        for tag in (self._reader_tag, self._writer_tag, self._runner_tag):
            if tag:
                self._client.release_cpus(tag)

    async def _preload(self, group: TopologyGroup, rep: int, placement: "_Placement") -> None:
        """Phase 2: fill the keyspace at the primary and wait until every replica has all of it."""
        task = self.task
        primary, replica = _primary_and_replica(group)
        self.status.state = "running"
        self.file_protocol.write_status(self.status)
        preload_toml = self._toml(
            f"preload_rep{rep}.toml",
            generate_toml_config(
                duration=PRELOAD_DURATION_SECONDS,
                warmup=0,
                threads=task.write_threads,
                cpu_list=placement.writer_cpus,
                endpoint=_endpoint(primary),
                connections=max(task.write_connections, 32),
                pipeline_depth=16,
                keyspace_count=task.keyspace_count,
                val_size=task.val_size,
                test="get",
                prefill=True,
            ),
        )
        await self._wait_and_parse(self._launch(preload_toml), "preload")
        await group.wait_for_offsets_caught_up()
        keys_primary = (await primary.count_items_expires())[0]
        keys_replica = (await replica.count_items_expires())[0]
        if keys_replica != keys_primary or keys_primary < task.keyspace_count:
            raise RuntimeError(
                f"preload mismatch: primary {keys_primary} keys, replica {keys_replica}, "
                f"expected >= {task.keyspace_count}"
            )

    def _generator_configs(self, placement: "_Placement", primary: Server, replica: Server) -> tuple:
        """(writer_toml, reader_toml) for the measure phase.

        The writer outlives the reader by ``WRITER_SLACK_SECONDS`` so the whole
        reader window sees the write stream, and still exits on its own.
        """
        task = self.task
        writer_toml = generate_toml_config(
            duration=task.warmup + task.duration + WRITER_SLACK_SECONDS,
            warmup=0,
            threads=task.write_threads,
            cpu_list=placement.writer_cpus,
            endpoint=_endpoint(primary),
            connections=task.write_connections,
            pipeline_depth=task.write_pipelining,
            keyspace_count=task.keyspace_count,
            val_size=task.val_size,
            test="set",
            rate_limit=task.write_rate,
            prefill=False,
        )
        reader_toml = generate_toml_config(
            duration=task.duration,
            warmup=task.warmup,
            threads=task.threads,
            cpu_list=placement.reader_cpus,
            endpoint=_endpoint(replica),
            connections=task.connections,
            pipeline_depth=task.pipelining,
            keyspace_count=task.keyspace_count,
            val_size=task.val_size,
            test="get",
            prefill=False,
        )
        return writer_toml, reader_toml

    async def _measure(self, group: TopologyGroup, rep: int, placement: "_Placement") -> "_Measurement":
        """Phase 3: fixed-rate writer at the primary, closed-loop reader at the replica, sampled throughout."""
        task = self.task
        primary, replica = _primary_and_replica(group)
        writer_toml, reader_toml = self._generator_configs(placement, primary, replica)
        writer_path = self._toml(f"writer_rep{rep}.toml", writer_toml)
        reader_path = self._toml(f"reader_rep{rep}.toml", reader_toml)
        self.logger.info(
            "rep %d/%d: writer %d/s at %s, reader at %s",
            rep,
            task.repetitions,
            task.write_rate,
            _endpoint(primary),
            _endpoint(replica),
        )

        samples: list = []
        writer_cmd = self._launch(writer_path)
        await asyncio.sleep(1.0)  # let the write stream reach steady state before reads start
        reader_cmd = self._launch(reader_path)
        reader_started = time.monotonic()
        sampler = asyncio.create_task(
            self._sample_loop(group, samples, reader_cmd, writer_cmd, pin_cpus=placement.runner_cpus)
        )
        try:
            reader = await self._wait_and_parse(reader_cmd, "reader")
        finally:
            sampler.cancel()
            try:
                await sampler
            except asyncio.CancelledError:
                pass
        writer = await self._wait_and_parse(writer_cmd, "writer")
        return _Measurement(
            reader=reader,
            writer=writer,
            samples=samples,
            reader_started=reader_started,
            reader_toml=reader_toml,
            writer_toml=writer_toml,
        )

    def _check_guards(self, reader: dict, writer: dict) -> None:
        """Phase 4: conditions under which the rep is not the experiment it claims to be."""
        hit = reader["hit_rate"]["percent"] if reader["hit_rate"] else 0.0
        if hit < READER_MIN_HIT_RATE_PCT:
            raise RuntimeError(f"reader hit rate {hit}% < {READER_MIN_HIT_RATE_PCT}%: replica dataset incomplete")
        achieved = writer["throughput_rps"]
        target = self.task.write_rate
        if achieved < target * (1 - WRITE_RATE_TOLERANCE):
            raise RuntimeError(
                f"writer achieved {achieved:.0f}/s, below {1 - WRITE_RATE_TOLERANCE:.0%} of target "
                f"{target}/s: primary or writer could not sustain the rate"
            )

    def _judge(self, group: TopologyGroup, rep: int, placement: "_Placement", m: "_Measurement") -> dict:
        """Phases 4-5: apply the guards, then reduce the rep to its result row entry with a verdict."""
        self._check_guards(m.reader, m.writer)
        primary, replica = _primary_and_replica(group)
        lag = replication_lag_stats(m.samples, primary.port, replica.port)
        parties = cpu_sampling.Parties(
            replica=cpu_sampling.ServerIdentity(replica.port, replica.valkey_pid),
            primary=cpu_sampling.ServerIdentity(primary.port, primary.valkey_pid),
            allocated_cpus=self._allocated_cpus(group, placement),
        )
        bottleneck = cpu_sampling.bottleneck_verdict(
            m.samples, parties, window_start=m.reader_started + self.task.warmup
        )
        log = self.logger.info if bottleneck["valid"] else self.logger.warning
        log(
            "rep %d/%d: reader %.0f rps, writer %.0f rps (target %d), lag max %s bytes, bottleneck=%s (%s)",
            rep,
            self.task.repetitions,
            m.reader["throughput_rps"],
            m.writer["throughput_rps"],
            self.task.write_rate,
            lag.get("max_bytes"),
            bottleneck["verdict"],
            bottleneck["reason"],
        )
        return {
            "reader_rps": m.reader["throughput_rps"],
            "reader_latency": m.reader["latency"],
            "reader_hit_rate": m.reader["hit_rate"],
            "writer_rps": m.writer["throughput_rps"],
            "writer_latency": m.writer["latency"],
            "lag": lag,
            "bottleneck": bottleneck,
            "samples": self._slim_samples(m.samples),
        }

    @staticmethod
    def _allocated_cpus(group: TopologyGroup, placement: "_Placement") -> dict:
        """Label -> CPU list for every pinned party; cores in none of them are foreign."""
        primary, replica = _primary_and_replica(group)
        allocated = {
            "primary": list(primary.server_cpus),
            "replica": list(replica.server_cpus),
            "reader": parse_cpulist(placement.reader_cpus) if placement.reader_cpus else [],
            "writer": parse_cpulist(placement.writer_cpus) if placement.writer_cpus else [],
            "runner": parse_cpulist(placement.runner_cpus) if placement.runner_cpus else [],
        }
        for extra in group.replicas[1:]:
            allocated[f"replica:{extra.port}"] = list(extra.server_cpus)
        return allocated

    async def _sample_loop(
        self,
        group: TopologyGroup,
        samples: list,
        reader_cmd: RealtimeCommand,
        writer_cmd: Optional[RealtimeCommand] = None,
        pin_cpus: str = "",
    ) -> None:
        """Sample the topology (remote shell) and the two local generators once per interval.

        The generators run on THIS host, so their thread ticks are read straight
        from ``/proc`` rather than through the remote shell; ``t_local`` stamps
        that read so the two clocks are never mixed.
        """
        fields = self.task.extra_info_fields()
        reader_pid = self._pid(reader_cmd)
        writer_pid = self._pid(writer_cmd)
        while reader_cmd.is_running():
            try:
                sample = await group.sample(fields, pin_cpus=pin_cpus)
                sample["t_local"] = time.monotonic()
                sample["generators"] = {
                    "reader": cpu_sampling.read_local_thread_stats(reader_pid),
                    "writer": cpu_sampling.read_local_thread_stats(writer_pid),
                }
                samples.append(sample)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.warning("INFO sample failed: %s", exc)
            await asyncio.sleep(self.task.sample_interval)

    @staticmethod
    def _slim_samples(samples: list) -> list:
        """Drop the cumulative counters from stored samples; the verdict already summarises them.

        Per-core and per-thread jiffies for a 96-core host at 1 Hz over a
        60 s rep would add ~1 MB to the result row; the INFO series (offsets,
        ops/sec, any extra INFO fields) is what later analysis reads back.
        """
        return [{k: v for k, v in s.items() if k not in ("cores", "threads", "generators")} for s in samples]

    # ------------------------------------------------------------------ results

    @staticmethod
    def _aggregate_bottleneck(reps: list) -> dict:
        """One verdict for the row: the score is only a server number if EVERY rep was.

        Anything else names the worst rep so a reader of the row does not have
        to open ``per_rep_results`` to find out why ``valid`` is false.
        """
        verdicts = [r["bottleneck"]["verdict"] for r in reps]
        valid = all(r["bottleneck"]["valid"] for r in reps)
        if valid:
            verdict = cpu_sampling.VERDICT_SERVER
        else:
            verdict = next(v for v in verdicts if v != cpu_sampling.VERDICT_SERVER)
        worst = next((r["bottleneck"] for r in reps if r["bottleneck"]["verdict"] == verdict), reps[0]["bottleneck"])
        duties = [v for r in reps if (v := (r["bottleneck"].get("replica") or {}).get("loop_duty")) is not None]
        reader_peaks = [
            v for r in reps if (v := (r["bottleneck"].get("reader") or {}).get("max_thread_util")) is not None
        ]
        return {
            "verdict": verdict,
            "valid": valid,
            "per_rep": verdicts,
            "reason": worst["reason"],
            "replica_loop_duty_mean": mean(duties) if duties else None,
            "reader_max_thread_util": max(reader_peaks, default=None),
        }

    async def _record_result(self, group: TopologyGroup, reps: list, reader_toml: str, writer_toml: str) -> None:
        task = self.task
        completion_time = datetime.datetime.now()
        assert group.primary is not None
        lscpu_output, _ = await group.primary.run_host_command("lscpu")

        reader_rps = [r["reader_rps"] for r in reps]
        mean_rps = mean(reader_rps)
        cv = (stdev(reader_rps) / mean_rps) * 100 if len(reader_rps) >= 2 and mean_rps else None

        detailed = {
            "topology": self.spec.to_dict(),
            "server_cpus": {s.port: s.server_cpus for s in group.servers},
            "warmup": task.warmup,
            "duration": task.duration,
            "io-threads": task.io_threads,
            "primary_io_threads": task.primary_io_threads,
            "pipeline": task.pipelining,
            "connections": task.connections,
            "threads": task.threads,
            "size": task.val_size,
            "keyspace_count": task.keyspace_count,
            "write_rate_target": task.write_rate,
            "write_connections": task.write_connections,
            "write_threads": task.write_threads,
            "write_pipeline": task.write_pipelining,
            "write_rate_achieved_mean": mean(r["writer_rps"] for r in reps),
            "replication_lag": {
                "max_bytes": max((r["lag"].get("max_bytes", 0) or 0) for r in reps),
                "mean_bytes": mean((r["lag"].get("mean_bytes", 0.0) or 0.0) for r in reps),
            },
            "per_run_rps": reader_rps,
            "mean_rps": mean_rps,
            "latency": reps[-1]["reader_latency"],
            "bottleneck": self._aggregate_bottleneck(reps),
            "per_rep_results": reps,
            "sample_interval": task.sample_interval,
            "info_fields": task.extra_info_fields(),
            "cachecannon_binary": task.cachecannon_binary,
            "reader_toml": reader_toml,
            "writer_toml": writer_toml,
            "lscpu": lscpu_output,
        }

        results = BenchmarkResults(
            method=METHOD,
            source=task.source,
            specifier=task.specifier,
            commit_hash=self.commit_hash,
            score=mean_rps,
            end_time=completion_time,
            data=detailed,
            make_args=task.make_args,
            note=task.note,
            cv=cv,
            reps=len(reps),
        )
        self.file_protocol.write_results(results)
