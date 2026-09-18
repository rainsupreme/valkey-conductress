"""Pathological-workload scenario benchmark task.

Measures how badly a named pathology disturbs concurrent GET throughput.
Architecture: steady background GET load via memtier (same prefill/keyspace as
task_mixed), PLUS an overlay driver that exercises the pathological pattern.

Reports:
  - Sustained baseline throughput (per-interval RPS timeseries)
  - Scenario-specific metrics (overlay ops/s, dip depth, recovery time)
"""

import asyncio
import datetime
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import t as t_dist

from conductress.config import PERF_BENCH_KEYSPACE, ServerInfo, get_sweep_engine, should_profile_internals
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus, FileProtocol, MetricData
from conductress.server import Server
from conductress.stormgen import netstat
from conductress.stormgen.policy import parse_policy
from conductress.stormgen.resp import ReplyParser, encode_command
from conductress.stormgen.stall import parse_stall
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.tasks.task_mixed import (
    MIXED_CLIENTS,
    MIXED_KEYSPACE,
    MIXED_THREADS,
    parse_memtier_total_rps,
    set_ratio_to_memtier_ratio,
)
from conductress.topology import TopologyGroup, TopologySpec
from conductress.utility import RealtimeCommand

logger = logging.getLogger(__name__)

# Valid scenario names
SCENARIO_CHOICES = (
    "eval-storm",
    "scan-churn",
    "multi-exec",
    "flushall-spike",
    "expiry-heavy",
    "bgsave",
    "large-value-reader",
    "connection-storm",
)

# The scenario whose overlay is the standalone connection-storm generator.
CONNECTION_STORM = "connection-storm"

# Default value size for the large-value-reader overlay's dedicated keyset (bytes)
LARGE_VALUE_READER_DEFAULT_SIZE = 10240  # 10 KB
# Dedicated keyset size for the large-value-reader overlay (separate from background keyspace)
LARGE_VALUE_READER_KEYSPACE = 50000

# Overlay thread/connection counts (kept small to avoid dominating the load)
OVERLAY_THREADS = 2
OVERLAY_CLIENTS = 4  # per-thread


def validate_scenario(name: str) -> bool:
    """Check whether a scenario name is recognized."""
    return name in SCENARIO_CHOICES


def encode_resp_array(*args: str) -> bytes:
    """Encode a single RESP command as an array of bulk strings.

    Example: encode_resp_array("SET", "key:1", "val") ->
        b"*3\r\n$3\r\nSET\r\n$5\r\nkey:1\r\n$3\r\nval\r\n"
    """
    parts = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        encoded = arg.encode()
        parts.append(f"${len(encoded)}\r\n".encode())
        parts.append(encoded)
        parts.append(b"\r\n")
    return b"".join(parts)


def generate_multi_exec_resp_payload(num_transactions: int, keyspace: int, val_size: int) -> bytes:
    """Generate a RESP payload of real MULTI/GET/SET/EXEC transactions.

    Each transaction is:
        MULTI
        GET memtier-<rand>
        SET memtier-<rand> <value>
        EXEC

    Keys use memtier's default prefix format (memtier-<N>) to hit the same
    prefilled keyspace as the background GET load.

    Returns raw RESP bytes ready for `valkey-cli --pipe`.
    """
    value = "x" * val_size
    payload_parts: List[bytes] = []
    for _ in range(num_transactions):
        key = f"memtier-{random.randint(1, keyspace)}"
        payload_parts.append(encode_resp_array("MULTI"))
        payload_parts.append(encode_resp_array("GET", key))
        payload_parts.append(encode_resp_array("SET", key, value))
        payload_parts.append(encode_resp_array("EXEC"))
    return b"".join(payload_parts)


def parse_memtier_json_intervals(json_path: str, json_content: str) -> List[float]:
    """Parse memtier --json-out-file for per-second ops/sec timeseries.

    Real memtier JSON (pinned build) structure:
        {"ALL STATS": {"Totals": {"Time-Serie": {"0": {"Count": N}, "1": {"Count": N}, ...}}}}
    Each Time-Serie key is a 0-indexed second; Count is total ops in that interval.
    The last entry is often a partial second (run teardown) and is excluded.

    Returns per-second RPS values (one per full second) for timeseries/dip analysis.
    """
    try:
        data = json.loads(json_content)
    except (json.JSONDecodeError, ValueError):
        return []

    all_stats = data.get("ALL STATS", {})
    totals = all_stats.get("Totals", {})
    time_serie: Dict[str, Any] = totals.get("Time-Serie", {})

    if not time_serie:
        return []

    # Sort by numeric key, exclude last entry (partial second from run teardown)
    sorted_keys = sorted(time_serie.keys(), key=lambda k: int(k))
    if len(sorted_keys) <= 1:
        # Only one interval — can't distinguish partial from full; return as-is
        entry = time_serie[sorted_keys[0]]
        count = entry.get("Count", 0)
        return [float(count)] if count > 0 else []

    # Exclude last entry (partial second) — its Count is typically much lower
    full_keys = sorted_keys[:-1]
    rps_values: List[float] = []
    for key in full_keys:
        entry = time_serie[key]
        count = entry.get("Count", 0)
        rps_values.append(float(count))

    return rps_values


def parse_memtier_stdout_intervals(stdout: str) -> List[float]:
    """Parse memtier stdout progress lines for per-interval instantaneous ops/sec.

    Memtier prints periodic progress to stdout (even with --hide-histogram):
        [RUN #1 20%,   1 secs]  2 threads  8 conns:  180540 ops,  90271 (avg:  90271) ops/sec, ...

    The first number before '(avg:' is the instantaneous ops/sec for that reporting interval.
    Lines use \\r for in-place overwrite on a TTY; when captured to a string they appear
    concatenated or \\r-separated.

    Returns per-interval RPS values (one per progress report). Used as fallback when
    JSON Time-Serie is unavailable.
    """
    rps_values: List[float] = []
    # Split on both \n and \r to handle TTY-style output
    lines = re.split(r"[\r\n]+", stdout)
    # Pattern: "NNN (avg: NNN) ops/sec" — capture the first NNN (instantaneous)
    pattern = re.compile(r"\[RUN #\d+.*?\]\s+.*?(\d+)\s+\(avg:\s*\d+\)\s+ops/sec")
    for line in lines:
        match = pattern.search(line)
        if match:
            rps_values.append(float(match.group(1)))
    return rps_values


def compute_dip_metrics(baseline_rps: float, interval_rps: List[float]) -> Dict[str, Any]:
    """Compute throughput dip depth and duration from interval timeseries.

    Returns:
        dip_depth_pct: maximum instantaneous drop from baseline (0-100%)
        dip_duration_seconds: number of seconds where RPS < 80% of baseline
        min_rps: lowest observed interval RPS
        recovery_seconds: time from dip start to return to 90% of baseline
    """
    if not interval_rps or baseline_rps <= 0:
        return {"dip_depth_pct": 0.0, "dip_duration_seconds": 0, "min_rps": 0.0, "recovery_seconds": 0}

    min_rps = min(interval_rps)
    dip_depth_pct = max(0.0, (1.0 - min_rps / baseline_rps) * 100.0)

    # Duration where RPS < 80% of baseline
    threshold_80 = baseline_rps * 0.80
    dip_seconds = sum(1 for r in interval_rps if r < threshold_80)

    # Recovery: find first dip below 80%, then count until back above 90%
    threshold_90 = baseline_rps * 0.90
    recovery = 0
    in_dip = False
    for r in interval_rps:
        if r < threshold_80:
            in_dip = True
        if in_dip:
            recovery += 1
            if r >= threshold_90:
                break
    if not in_dip:
        recovery = 0

    return {
        "dip_depth_pct": round(dip_depth_pct, 2),
        "dip_duration_seconds": dip_seconds,
        "min_rps": round(min_rps, 1),
        "recovery_seconds": recovery,
    }


def build_overlay_command(
    scenario: str,
    server_ip: str,
    port: int,
    duration: int,
    keyspace: int,
    val_size: int,
    overlay_value_size: int = 0,
) -> str:
    """Build the overlay command string for a given scenario.

    Returns a shell command that drives the pathological workload.
    Uses valkey-benchmark for repeatable load generation with arbitrary commands.
    Fallback to valkey-cli loops only where multi-step transactions require it.

    Args:
        overlay_value_size: For large-value-reader, the value size of the dedicated
            keyset (bytes). Ignored for other scenarios.
    """
    bench = "~/conductress/valkey-benchmark"
    cli = "~/conductress/valkey-cli"
    conns = OVERLAY_THREADS * OVERLAY_CLIENTS
    # Total requests: enough to fill the duration at moderate rate
    # Use -t for time-based where available, otherwise estimate
    n_requests = duration * 50000  # 50K ops/sec target per overlay

    if scenario == "eval-storm":
        # Small read+write Lua script at max rate on a few connections.
        # EVAL "redis.call('GET',KEYS[1]) redis.call('SET',KEYS[1],ARGV[1]) return 1" 1 key val
        lua_script = "redis.call('GET',KEYS[1]) redis.call('SET',KEYS[1],ARGV[1]) return 1"
        return (
            f"{bench} -h {server_ip} -p {port} "
            f"-c {conns} -n {n_requests} --threads {OVERLAY_THREADS} "
            f'EVAL "{lua_script}" 1 __rand_int__ value_payload -r {keyspace}'
        )

    elif scenario == "scan-churn":
        # Continuous SCAN cursor iteration from few connections.
        # valkey-benchmark doesn't support stateful SCAN (needs cursor chaining),
        # so we use a bash loop with valkey-cli.
        return (
            f"bash -c '"
            f"end=$((SECONDS + {duration})); "
            f"while [ $SECONDS -lt $end ]; do "
            f'{cli} -h {server_ip} -p {port} --scan --pattern "*" > /dev/null 2>&1; '
            f"done'"
        )

    elif scenario == "multi-exec":
        # Real MULTI/GET/SET/EXEC transactions via valkey-cli --pipe.
        # Pre-generate a RESP payload file of ~1000 transactions, then loop-replay
        # it until the overlay is terminated. This exercises true transaction
        # serialization (Tier-3 exclusive-mode pathology).
        payload_file = "/tmp/multi_exec_payload.resp"
        return (
            f"bash -c '"
            f"end=$((SECONDS + {duration})); "
            f"while [ $SECONDS -lt $end ]; do "
            f"{cli} -h {server_ip} -p {port} --pipe < {payload_file} > /dev/null 2>&1; "
            f"done; "
            f"rm -f {payload_file}'"
        )

    elif scenario == "flushall-spike":
        # Single FLUSHALL (ASYNC if supported) mid-measurement, then re-prefill.
        # We'll schedule FLUSHALL at ~40% through duration, then re-prefill.
        flush_delay = max(2, duration * 2 // 5)
        prefill_n = keyspace // (MIXED_THREADS * MIXED_CLIENTS)
        memtier = "~/conductress/memtier_benchmark"
        return (
            f"bash -c '"
            f"sleep {flush_delay}; "
            f"{cli} -h {server_ip} -p {port} FLUSHALL ASYNC; "
            f"sleep 1; "
            f"{memtier} --server {server_ip} --port {port} --protocol redis "
            f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
            f"--ratio 1:0 --key-pattern P:P "
            f"--key-minimum 1 --key-maximum {keyspace} "
            f"--data-size {val_size} "
            f"--requests {prefill_n} --hide-histogram > /dev/null 2>&1'"
        )

    elif scenario == "expiry-heavy":
        # Overlay SETs with short random TTLs so expiry churns during reads.
        # valkey-benchmark: SET with EX via custom command.
        # TTL 1-5 seconds for aggressive expiry pressure.
        return (
            f"{bench} -h {server_ip} -p {port} "
            f"-c {conns} -n {n_requests} --threads {OVERLAY_THREADS} "
            f"-r {keyspace} SET __rand_int__ value_payload EX 3"
        )

    elif scenario == "bgsave":
        # Single BGSAVE mid-measurement (at ~40% of duration).
        # The fork()+COW impact shows up in the interval RPS timeseries as a dip.
        # Dataset size drives the fork cost, so prefill size matters.
        # Server runs with --save '' but explicit BGSAVE command works regardless.
        bgsave_delay = max(2, duration * 2 // 5)
        return f"bash -c 'sleep {bgsave_delay}; {cli} -h {server_ip} -p {port} BGSAVE'"

    elif scenario == "large-value-reader":
        # Continuous GETs against a DEDICATED large-value keyset (separate key prefix).
        # Measures how a legitimate heavyweight-read neighbor (large values) degrades
        # the background workload's throughput and latency.
        # MUST use memtier with the same --key-prefix/--key-minimum/--key-maximum
        # as the prefill: valkey-benchmark's __rand_int__ zero-pads keys
        # (lvr:000000000042) while memtier writes lvr:42 — mixing the two tools
        # makes every overlay GET a miss and the neighbor weighs nothing.
        return (
            f"~/conductress/memtier_benchmark "
            f"--server {server_ip} --port {port} --protocol redis "
            f"--threads {OVERLAY_THREADS} --clients {OVERLAY_CLIENTS} "
            f"--ratio 0:1 --key-pattern R:R "
            f"--key-prefix lvr: "
            f"--key-minimum 1 --key-maximum {LARGE_VALUE_READER_KEYSPACE} "
            f"--requests {n_requests} --hide-histogram"
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# --------------------------------------------------------------------------- storm overlay spec

# Keys the connection-storm overlay_spec JSON may carry, with their defaults.
# These mirror the standalone generator's CLI so a scenario spec is just the
# generator's parameters in JSON form.
STORM_SPEC_DEFAULTS: Dict[str, Any] = {
    "clients": 2000,
    "burst_ms": 200,
    "connect_timeout_ms": 1000.0,
    "reply_timeout_ms": 500.0,
    "policy": "fixed:200",
    "handshake": ["HELLO 3"],
    "first_command": "GET storm:key",
    "stall": "none",
    "burst_after_stall_ms": 200,
    "burst_first": False,
    # Launch the generator this long after the overlay starts, so the background
    # series carries an undisturbed baseline before the stall and the burst.
    "start_delay_s": 5.0,
    "prewarm_connections": None,  # None -> generator default (= clients)
    "workers": 0,  # 0 -> generator auto (min(8, cpu_count))
    "bind_addrs": [],
}
# The single key the connection-storm overlay's default first command reads.
STORM_KEY = "storm:key"


def parse_storm_spec(spec: str) -> Dict[str, Any]:
    """Parse and validate a connection-storm ``overlay_spec`` JSON string.

    Pure function (no I/O): returns a fully-defaulted, validated spec dict, or
    raises ``ValueError`` with a clear message. An empty string yields all
    defaults so ``--scenario connection-storm`` works with no ``--storm-*``
    flags. Unknown keys, the wrong JSON shape, or an invalid policy/stall spec
    are rejected here, at submission time, rather than on a runner.
    """
    text = (spec or "").strip()
    if not text:
        parsed: Dict[str, Any] = {}
    else:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"connection-storm overlay_spec is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("connection-storm overlay_spec must be a JSON object")

    unknown = set(parsed) - set(STORM_SPEC_DEFAULTS)
    if unknown:
        raise ValueError(f"connection-storm overlay_spec has unknown keys: {', '.join(sorted(unknown))}")

    out: Dict[str, Any] = dict(STORM_SPEC_DEFAULTS)
    out["handshake"] = list(STORM_SPEC_DEFAULTS["handshake"])
    out["bind_addrs"] = list(STORM_SPEC_DEFAULTS["bind_addrs"])
    out.update(parsed)

    # Validate the sub-specs eagerly so a typo fails at submission.
    parse_policy(str(out["policy"]))
    parse_stall(str(out["stall"]))
    if int(out["clients"]) < 1:
        raise ValueError(f"connection-storm clients must be >= 1, got {out['clients']}")
    if int(out["burst_ms"]) < 0:
        raise ValueError(f"connection-storm burst_ms must be >= 0, got {out['burst_ms']}")
    if float(out["connect_timeout_ms"]) <= 0 or float(out["reply_timeout_ms"]) <= 0:
        raise ValueError("connection-storm connect_timeout_ms and reply_timeout_ms must be > 0")
    if int(out["workers"]) < 0:
        raise ValueError(f"connection-storm workers must be >= 0 (0 = auto), got {out['workers']}")
    if int(out["burst_after_stall_ms"]) < 0:
        raise ValueError(f"connection-storm burst_after_stall_ms must be >= 0, got {out['burst_after_stall_ms']}")
    if float(out["start_delay_s"]) < 0:
        raise ValueError(f"connection-storm start_delay_s must be >= 0, got {out['start_delay_s']}")
    if out["prewarm_connections"] is not None and int(out["prewarm_connections"]) < 0:
        raise ValueError(f"connection-storm prewarm_connections must be >= 0 or null, got {out['prewarm_connections']}")
    if not isinstance(out["handshake"], list):
        raise ValueError("connection-storm handshake must be a JSON array of command strings")
    if not isinstance(out["bind_addrs"], list):
        raise ValueError("connection-storm bind_addrs must be a JSON array of source addresses")
    return out


def storm_uses_debug_sleep(spec: Dict[str, Any]) -> bool:
    """True when the storm spec's stall injector is a debug-sleep (needs enable-debug-command)."""
    return parse_stall(str(spec.get("stall", "none"))).kind == "debug-sleep"


# --------------------------------------------------------------------------- server-side sampler

# Fields pulled from the INFO clients + INFO stats replies each tick. All are
# O(1) on the server, so the sampler is one extra idle client.
_SAMPLED_INT_FIELDS = (
    "connected_clients",
    "total_connections_received",
    "rejected_connections",
    "total_commands_processed",
)
# A reply that arrives later than this multiple of the tick is a gap: the main
# thread was busy (stalled) and could not answer INFO on time.
GAP_TICK_MULTIPLE = 5
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def parse_info_fields(blob: str) -> Dict[str, int]:
    """Extract the sampled integer fields from one or more INFO reply blobs.

    INFO replies are ``key:value`` lines. Only the fields in
    ``_SAMPLED_INT_FIELDS`` are kept, and only when they parse as integers;
    everything else (human-readable config, non-numeric values) is dropped.
    """
    out: Dict[str, int] = {}
    for line in blob.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in _SAMPLED_INT_FIELDS:
            try:
                out[key] = int(value.strip())
            except ValueError:
                continue
    return out


def sampler_gaps(rows: List[Dict[str, Any]], tick_ms: int) -> List[Dict[str, float]]:
    """Return the intervals where a sample's reply came back late (a gap).

    Pure function. A gap is a row whose ``t_reply - t_sent`` exceeds
    ``GAP_TICK_MULTIPLE`` times the tick: the server's main thread was busy
    (typically the stall) and could not answer ``INFO`` on time. Each gap is
    ``{"t_sent": ..., "t_reply": ...}`` (wall-clock), which the plot uses to
    break lines and draw a grey marker. The reply is never retried; the late
    ``t_reply`` is recorded as-is, so a gap is data, not an error.
    """
    threshold = GAP_TICK_MULTIPLE * (tick_ms / 1000.0)
    gaps: List[Dict[str, float]] = []
    for row in rows:
        t_sent = row.get("t_sent")
        t_reply = row.get("t_reply")
        if t_sent is None or t_reply is None:
            continue
        if (t_reply - t_sent) > threshold:
            gaps.append({"t_sent": float(t_sent), "t_reply": float(t_reply)})
    return gaps


class ServerSampler:
    """Polls one server's INFO counters on a fixed cadence, over one connection.

    Overlay-agnostic: started just before the overlay and stopped after the
    background measurement ends, for any scenario that opts in
    (``server_sample_ms > 0``). Each tick sends ``INFO clients`` and
    ``INFO stats`` in a single write and reads both replies on a persistent
    connection, stamping wall-clock ``t_sent`` / ``t_reply`` so the row shares
    the same clock as the storm's ``origin_wall`` and the stall record.

    When the runner and the server share a host (loopback -- the fleet today),
    each tick also reads ``/proc/net/netstat`` TcpExt ``ListenOverflows`` /
    ``ListenDrops`` as deltas from the first tick, reusing
    :mod:`conductress.stormgen.netstat`. When the server is remote those two
    fields are ``None`` (logged once), because the counters are the runner
    host's, not the server's.

    Gaps are never retried inside a tick: while the main thread is stalled the
    reply simply arrives late and the real ``t_reply`` is recorded.
    """

    def __init__(self, host: str, port: int, tick_ms: int, connect_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.tick_ms = tick_ms
        self.connect_timeout = connect_timeout
        self.rows: List[Dict[str, Any]] = []
        self._local = host in _LOOPBACK_HOSTS
        self._baseline: Optional[netstat.ListenCounters] = None
        # Created lazily in run(): asyncio.Event() needs a running loop on 3.9,
        # and the sampler is constructed from sync code (and sync tests).
        self._stop: Optional[asyncio.Event] = None
        self._stop_requested = False
        self._remote_logged = False
        # One RESP parser per connection: leftover bytes from a read that
        # carried more than one reply belong to the next reply, not the bin.
        self._parser = ReplyParser()

    def stop(self) -> None:
        self._stop_requested = True
        if self._stop is not None:
            self._stop.set()

    async def run(self) -> None:
        """Sample until :meth:`stop` is set. Never raises out; a failed tick is skipped."""
        self._stop = asyncio.Event()
        if self._stop_requested:
            self._stop.set()
        interval = self.tick_ms / 1000.0
        reader = writer = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=self.connect_timeout
            )
            self._parser = ReplyParser()
            if self._local:
                self._baseline = netstat.read_counters()
            else:
                logger.info("ServerSampler: server %s is remote; listen-overflow counters recorded as null", self.host)
            while not self._stop.is_set():
                await self._one_tick(reader, writer)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("ServerSampler could not connect to %s:%s: %s", self.host, self.port, exc)
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, asyncio.TimeoutError):
                    pass

    async def _one_tick(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One INFO clients + INFO stats round-trip; append a row (best effort)."""
        t_sent = time.time()
        try:
            writer.write(encode_command("INFO", "clients") + encode_command("INFO", "stats"))
            await writer.drain()
            first = await self._read_reply(reader)
            second = await self._read_reply(reader)
        except (OSError, asyncio.TimeoutError, ConnectionError) as exc:
            self.stop()
            logger.warning("ServerSampler read failed: %s", exc)
            return
        t_reply = time.time()
        fields = parse_info_fields(f"{first}\n{second}")
        row: Dict[str, Any] = {"t_sent": t_sent, "t_reply": t_reply}
        for name in _SAMPLED_INT_FIELDS:
            row[name] = fields.get(name)
        row["listen_overflows"], row["listen_drops"] = self._netstat_deltas()
        self.rows.append(row)

    def _netstat_deltas(self) -> Tuple[Optional[int], Optional[int]]:
        """(ListenOverflows, ListenDrops) delta from the first tick; (None, None) if remote/unreadable."""
        if not self._local or self._baseline is None:
            return None, None
        current = netstat.read_counters()
        if current is None:
            return None, None
        return (
            current.listen_overflows - self._baseline.listen_overflows,
            current.listen_drops - self._baseline.listen_drops,
        )

    async def _read_reply(self, reader: asyncio.StreamReader) -> str:
        """Read exactly one RESP reply (a bulk string for INFO) and return it as text.

        The parser is per connection, not per reply: the two INFO replies of a
        tick normally arrive in ONE TCP read, and a per-reply parser would
        return the first reply and drop the bytes of the second along with
        itself, leaving the next read waiting forever for a reply the server
        has already sent.
        """
        parser = self._parser
        while True:
            complete, value = parser.try_parse()
            if complete:
                return value if isinstance(value, str) else ("" if value is None else str(value))
            chunk = await reader.read(65536)
            if not chunk:
                # Connection closed mid-reply; return whatever parsed (usually "").
                complete, value = parser.try_parse()
                return value if (complete and isinstance(value, str)) else ""
            parser.feed(chunk)


# --------------------------------------------------------------------------- overlay drivers


@dataclass
class OverlayResult:
    """What an overlay produced: an optional ops/s rate plus arbitrary metrics.

    ``rate`` is the overlay's own ops/s where that is meaningful (the
    command-string overlays parse it from valkey-benchmark output); it is
    ``None`` for an overlay that has no single-rate summary (the storm).
    ``metrics`` is a free-form dict merged into the scenario metrics.
    """

    rate: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)


class Overlay:
    """A pathology overlay run concurrently with the background GET load.

    The lifecycle mirrors the runner's existing flow so command overlays are
    byte-for-byte unchanged: ``start`` launches the overlay (non-blocking) and
    returns an opaque handle; the runner then measures the background load;
    ``finish`` stops the overlay and returns its :class:`OverlayResult`. A
    driver may also declare extra server arguments it needs via
    ``extra_server_args``.
    """

    def extra_server_args(self) -> str:
        """Extra raw server args this overlay requires (appended after the task's)."""
        return ""

    async def start(self, server: "Server", work_dir: "Any") -> Any:
        """Launch the overlay; return an opaque handle passed to ``finish``."""
        raise NotImplementedError

    async def finish(self, server: "Server", handle: Any) -> OverlayResult:
        """Stop the overlay and return its result."""
        raise NotImplementedError

    async def abort(self, server: "Server", handle: Any) -> None:
        """Best-effort teardown when a rep fails before ``finish`` ran."""
        raise NotImplementedError


class CommandOverlay(Overlay):
    """Adapter around the existing shell-command overlays.

    Produces the IDENTICAL command string ``build_overlay_command`` did and runs
    it exactly as the runner did before (background process on the server host,
    killed after the measurement, ops/s parsed from its output), so every
    pre-existing scenario behaves unchanged.
    """

    def __init__(self, runner: "ScenarioTaskRunner"):
        self._runner = runner

    def command(self, server: "Server") -> str:
        return build_overlay_command(
            scenario=self._runner.scenario,
            server_ip=server.ip,
            port=server.port,
            duration=self._runner.duration,
            keyspace=MIXED_KEYSPACE,
            val_size=self._runner.val_size,
            overlay_value_size=self._runner.overlay_value_size,
        )

    async def start(self, server: "Server", work_dir: Any) -> Any:
        if self._runner.scenario == "multi-exec":
            await self._runner._write_multi_exec_payload(server)
        return await self._runner._run_overlay(server, self.command(server))

    async def finish(self, server: "Server", handle: Any) -> OverlayResult:
        output = await self._runner._kill_overlay(server, handle)
        rate = self._runner._parse_overlay_rps(output) if output else None
        return OverlayResult(rate=rate, metrics={})

    async def abort(self, server: "Server", handle: Any) -> None:
        await self._runner._kill_overlay(server, handle)


class StormOverlay(Overlay):
    """Runs the standalone connection-storm generator as the overlay.

    Launches ``python -m conductress.stormgen --json`` locally (its own
    asyncio/multiprocessing lifecycle, prewarm and stall injector included),
    against the same server the background load hits. Produces no single ops/s
    rate (``rate=None``); its full JSON document is returned as metrics for the
    runner to fold into the ``storm.*`` namespace.
    """

    def __init__(self, spec: Dict[str, Any], work_dir: Any):
        self.spec = spec
        self._work_dir = work_dir

    def extra_server_args(self) -> str:
        # DEBUG SLEEP is rejected unless the server enables it. Append the flag
        # only for a debug-sleep stall; never for any other overlay.
        return "--enable-debug-command local" if storm_uses_debug_sleep(self.spec) else ""

    def _command(self, server: "Server", json_path: str) -> str:
        spec = self.spec
        parts: List[str] = [
            sys.executable,
            "-m",
            "conductress.stormgen",
            "--host",
            server.ip,
            "--port",
            str(server.port or 6379),
            "--clients",
            str(spec["clients"]),
            "--burst-ms",
            str(spec["burst_ms"]),
            "--connect-timeout-ms",
            _num(spec["connect_timeout_ms"]),
            "--reply-timeout-ms",
            _num(spec["reply_timeout_ms"]),
            "--policy",
            str(spec["policy"]),
            "--first-command",
            str(spec["first_command"]),
            "--duration-s",
            _num(self._duration_s()),
            "--stall",
            str(spec["stall"]),
            "--burst-after-stall-ms",
            str(spec["burst_after_stall_ms"]),
            "--workers",
            str(spec["workers"]),
            "--json",
            json_path,
        ]
        if spec["burst_first"]:
            parts.append("--burst-first")
        if spec["prewarm_connections"] is not None:
            parts += ["--prewarm-connections", str(spec["prewarm_connections"])]
        handshake = list(spec["handshake"])
        if handshake:
            for command in handshake:
                parts += ["--handshake", str(command)]
        else:
            parts += ["--handshake", ""]
        for addr in spec["bind_addrs"]:
            parts += ["--bind-addr", str(addr)]
        return " ".join(_shell_quote(p) for p in parts)

    # Seconds kept back from the storm's duration so it ends before the
    # background measurement does (the generator's prewarm and the runner's
    # settle sleep both precede the storm's own clock).
    STORM_END_MARGIN_S = 2.0
    STORM_MIN_DURATION_S = 5.0

    def _duration_s(self) -> float:
        """The storm's own duration: the measurement window minus the start delay and a margin."""
        window = float(self.spec["_duration_s"])
        remaining = window - float(self.spec["start_delay_s"]) - self.STORM_END_MARGIN_S
        if remaining < self.STORM_MIN_DURATION_S:
            logger.warning(
                "connection-storm: start_delay_s %.1f leaves %.1fs of a %.1fs window; clamping the storm to %.1fs",
                float(self.spec["start_delay_s"]),
                remaining,
                window,
                self.STORM_MIN_DURATION_S,
            )
            return self.STORM_MIN_DURATION_S
        return remaining

    async def start(self, server: "Server", work_dir: Any) -> Any:
        json_path = str(work_dir / "storm_result.json")
        command_string = self._command(server, json_path)
        handle: Dict[str, Any] = {"cmd": None, "json_path": json_path, "lines": [], "launch": None}
        delay = float(self.spec["start_delay_s"])

        async def _launch() -> None:
            if delay > 0:
                await asyncio.sleep(delay)
            logger.info("connection-storm overlay: %s", command_string)
            cmd = RealtimeCommand(command_string)
            cmd.start()
            handle["cmd"] = cmd
            handle["launched_wall"] = time.time()

        # Returning immediately keeps the runner's flow unchanged: the background
        # measurement starts on schedule and the storm lands start_delay_s later.
        handle["launch"] = asyncio.ensure_future(_launch())
        return handle

    async def finish(self, server: "Server", handle: Any) -> OverlayResult:
        launch = handle.get("launch")
        if launch is not None:
            await launch
        cmd = handle["cmd"]
        lines: List[str] = handle["lines"]
        # Drain to completion (the background measurement has ended by now; the
        # storm's own duration bounds it).
        while cmd.is_running():
            line, _ = cmd.poll_output()
            while line:
                lines.append(line)
                line, _ = cmd.poll_output()
            await asyncio.sleep(0.5)
        line, _ = cmd.poll_output()
        while line:
            lines.append(line)
            line, _ = cmd.poll_output()
        exit_code = getattr(cmd.p, "returncode", None)
        if exit_code != 0:
            joined = "\n".join(lines)
            raise RuntimeError(f"connection-storm generator exited with code {exit_code}. Output:\n{joined[-2000:]}")
        document = _read_storm_document(handle["json_path"], lines)
        metrics = storm_metrics_namespace(document)
        metrics["storm"]["launched_wall"] = handle.get("launched_wall")
        return OverlayResult(rate=None, metrics=metrics)

    async def abort(self, server: "Server", handle: Any) -> None:
        launch = handle.get("launch") if isinstance(handle, dict) else None
        if launch is not None and not launch.done():
            launch.cancel()
        cmd = handle.get("cmd") if isinstance(handle, dict) else None
        if cmd is not None:
            try:
                cmd.kill()
            except Exception:  # pylint: disable=broad-except
                pass


def storm_metrics_namespace(document: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a stormgen JSON document into the scenario's ``storm.*`` metric keys."""
    metrics = document.get("metrics", {})
    totals = metrics.get("totals", {})
    quiescence = metrics.get("time_to_quiescence", {})
    return {
        "storm": {
            "quiescence_from_stall_end_s": quiescence.get("from_stall_end"),
            "quiescence_from_start_s": quiescence.get("from_start"),
            "amplification": totals.get("amplification"),
            "attempts": totals.get("attempts"),
            "connected_clients": totals.get("connected_clients"),
            "outcomes": metrics.get("outcomes"),
            "attempt_latency": metrics.get("attempt_latency"),
            "listen_overflow_delta": document.get("listen_overflow_delta"),
            "prewarm_listen_overflow_delta": document.get("prewarm_listen_overflow_delta"),
            "burst_actual_ms": metrics.get("burst_actual_ms"),
            "stall": document.get("stall"),
            "timeline": metrics.get("timeline"),
            "origin_wall": document.get("origin_wall"),
            "schema_version": document.get("schema_version"),
            "generator_config": document.get("config"),
        }
    }


def _read_storm_document(json_path: str, output_lines: List[str]) -> Dict[str, Any]:
    """Load the stormgen JSON document from its file, or fall back to stdout."""
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
                raise RuntimeError("connection-storm generator produced no parseable JSON result") from exc
        raise RuntimeError("connection-storm generator produced no JSON result")


def _num(value: float) -> str:
    """Render a float without a trailing ``.0`` for whole numbers."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _shell_quote(token: str) -> str:
    """Single-quote a token for a shell command line if it needs quoting."""
    if token and all(c.isalnum() or c in "._-/=:" for c in token):
        return token
    return "'" + token.replace("'", "'\\''") + "'"


@dataclass
class ScenarioTaskData(BaseTaskData):
    """Task data for pathological-workload scenario measurement."""

    scenario: str
    val_size: int
    io_threads: int
    pipelining: int
    duration: int
    warmup: int = 5
    perf_stat_enabled: bool = False
    repetitions: int = 3
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""
    background_set_ratio: int = 0
    server_args: str = ""  # extra raw args appended to the server command line (override defaults)
    overlay_value_size: int = 0  # value size for the large-value-reader overlay keyset (0 = default 10KB)
    overlay_spec: str = ""  # JSON object string parameterising an overlay (connection-storm only, for now)
    server_sample_ms: int = 0  # server INFO-sampler cadence in ms (0 = off); connection-storm defaults to 100 via CLI

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "ScenarioTaskData"
        if not validate_scenario(self.scenario):
            raise ValueError(f"Unknown scenario '{self.scenario}'. Valid: {', '.join(SCENARIO_CHOICES)}")
        if not (0 <= self.background_set_ratio <= 100):
            raise ValueError(f"background_set_ratio must be 0-100, got {self.background_set_ratio}")
        if self.overlay_value_size < 0:
            raise ValueError(f"overlay_value_size must be >= 0, got {self.overlay_value_size}")
        # 0 = sampler off; otherwise >= 20 ms so the row count stays bounded
        # (100 ms over a 60 s measurement is 600 rows).
        if self.server_sample_ms != 0 and self.server_sample_ms < 20:
            raise ValueError(f"server_sample_ms must be 0 (off) or >= 20, got {self.server_sample_ms}")
        if self.scenario == CONNECTION_STORM:
            # Parse eagerly so a bad spec fails at submission, not on a runner.
            parse_storm_spec(self.overlay_spec)
        elif self.overlay_spec:
            raise ValueError(f"overlay_spec is only valid for scenario '{CONNECTION_STORM}', not '{self.scenario}'")

    def storm_spec(self) -> Dict[str, Any]:
        """The validated connection-storm overlay spec (call only for that scenario)."""
        return parse_storm_spec(self.overlay_spec)

    def short_description(self) -> str:
        from conductress.utility import HumanByte, HumanTime

        ratio_str = f" SET={self.background_set_ratio}%" if self.background_set_ratio > 0 else ""
        lvr_str = ""
        if self.scenario == "large-value-reader":
            effective_size = self.overlay_value_size if self.overlay_value_size > 0 else LARGE_VALUE_READER_DEFAULT_SIZE
            lvr_str = f" ovl={HumanByte.to_human(effective_size)}"
        return (
            f"scenario:{self.scenario} "
            f"v={HumanByte.to_human(self.val_size)} "
            f"io={self.io_threads} P={self.pipelining}"
            f"{ratio_str}{lvr_str} "
            f"{HumanTime.to_human(self.duration)}"
            f"{' perf-stat' if self.perf_stat_enabled else ''}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "ScenarioTaskRunner":
        return ScenarioTaskRunner(
            task_name=self.task_id,
            server_infos=server_infos,
            topology=self.topology,
            source=self.source,
            specifier=self.specifier,
            make_args=self.make_args,
            io_threads=self.io_threads,
            val_size=self.val_size,
            pipelining=self.pipelining,
            scenario=self.scenario,
            warmup=self.warmup,
            duration=self.duration,
            repetitions=self.repetitions,
            perf_stat_enabled=self.perf_stat_enabled,
            note=self.note,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
            background_set_ratio=self.background_set_ratio,
            server_args=self.server_args,
            overlay_value_size=self.overlay_value_size,
            overlay_spec=self.overlay_spec,
            server_sample_ms=self.server_sample_ms,
        )


class ScenarioTaskRunner(BaseTaskRunner):
    """Runs a pathological-workload scenario: background GET + overlay driver."""

    def __init__(
        self,
        task_name: str,
        server_infos: list[ServerInfo],
        source: str,
        specifier: str,
        make_args: str,
        io_threads: int,
        val_size: int,
        pipelining: int,
        scenario: str,
        warmup: int,
        duration: int,
        repetitions: int,
        perf_stat_enabled: bool = False,
        note: str = "",
        server_cpu_override: str = "",
        benchmark_cpu_override: str = "",
        background_set_ratio: int = 0,
        server_args: str = "",
        overlay_value_size: int = 0,
        overlay_spec: str = "",
        server_sample_ms: int = 0,
        topology: Optional[TopologySpec] = None,
    ):
        super().__init__(task_name)
        self.server_infos = server_infos
        self.topology = topology if topology is not None else TopologySpec.standalone()
        self.source = source
        self.specifier = specifier
        self.make_args = make_args
        self.io_threads = io_threads
        self.val_size = val_size
        self.pipelining = pipelining
        self.scenario = scenario
        self.warmup = warmup
        self.duration = duration
        self.repetitions = repetitions
        self.perf_stat_enabled = perf_stat_enabled
        self._perf_stat_scope: Optional[str] = None
        self.note = note
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override
        self.background_set_ratio = background_set_ratio
        self.server_args = server_args
        self.overlay_value_size = overlay_value_size
        self.overlay_spec = overlay_spec
        self.server_sample_ms = server_sample_ms

        # Build the overlay driver once. connection-storm runs the standalone
        # generator; every other scenario uses the unchanged shell-command path.
        self._storm_spec: Optional[Dict[str, Any]] = None
        if scenario == CONNECTION_STORM:
            self._storm_spec = parse_storm_spec(overlay_spec)
            self._storm_spec["_duration_s"] = float(duration)
            self._overlay: Overlay = StormOverlay(self._storm_spec, self.file_protocol.work_dir)
        else:
            self._overlay = CommandOverlay(self)

        # An overlay may require extra server args (connection-storm + debug-sleep
        # appends --enable-debug-command local). Record the effective args so a
        # result reader can see exactly what the server ran with.
        extra = self._overlay.extra_server_args()
        self.server_args_effective = f"{server_args} {extra}".strip() if extra else server_args
        if extra:
            logger.info(
                "connection-storm with a debug-sleep stall: appended %r to server args (effective: %r)",
                extra,
                self.server_args_effective,
            )

        self.commit_hash = ""
        self._profile_internals = should_profile_internals(get_sweep_engine(source))

        self.status = BenchmarkStatus(
            steps_total=repetitions * 3,  # prefill + baseline + scenario per rep
            task_type=f"scenario-{scenario}",
        )

    async def _run_overlay(self, server: Server, overlay_cmd: str) -> Optional[str]:
        """Start overlay process and return its PID for cleanup.

        Returns the remote PID as a string so we can kill it on failure.
        """
        # Run overlay in background, capture PID
        bg_cmd = f"nohup {overlay_cmd} > /tmp/overlay_out.txt 2>&1 & echo $!"
        stdout, _ = await server.run_host_command(bg_cmd, check=False)
        pid = stdout.strip().split("\n")[-1].strip()
        if pid.isdigit():
            return pid
        return None

    async def _kill_overlay(self, server: Server, pid: Optional[str]) -> Optional[str]:
        """Kill overlay process and return its stdout."""
        overlay_output = None
        if pid:
            # Read output before killing
            cat_stdout, _ = await server.run_host_command("cat /tmp/overlay_out.txt 2>/dev/null || true", check=False)
            overlay_output = cat_stdout
            # Kill the process tree
            await server.run_host_command(
                f"kill -TERM {pid} 2>/dev/null; sleep 0.5; kill -9 {pid} 2>/dev/null", check=False
            )
            # Also kill any child processes (bash loops spawn children)
            await server.run_host_command(
                f"pkill -TERM -P {pid} 2>/dev/null; sleep 0.2; pkill -9 -P {pid} 2>/dev/null", check=False
            )
        return overlay_output

    async def _write_multi_exec_payload(self, server: Server) -> None:
        """Generate and write MULTI/EXEC RESP payload to /tmp on the remote host.

        Writes ~1000 transactions worth of RESP-encoded MULTI/GET/SET/EXEC
        sequences to /tmp/multi_exec_payload.resp for valkey-cli --pipe replay.
        """
        payload = generate_multi_exec_resp_payload(
            num_transactions=1000,
            keyspace=MIXED_KEYSPACE,
            val_size=self.val_size,
        )
        # Write payload via base64 to avoid shell quoting issues with binary RESP
        import base64

        encoded = base64.b64encode(payload).decode()
        # Split into chunks to avoid argument-too-long (payload ~200KB base64)
        chunk_size = 65536
        chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
        # Write first chunk (truncate), append rest
        await server.run_host_command(f"echo -n '{chunks[0]}' > /tmp/multi_exec_payload.b64", check=True)
        for chunk in chunks[1:]:
            await server.run_host_command(f"echo -n '{chunk}' >> /tmp/multi_exec_payload.b64", check=True)
        await server.run_host_command(
            "base64 -d /tmp/multi_exec_payload.b64 > /tmp/multi_exec_payload.resp && rm -f /tmp/multi_exec_payload.b64",
            check=True,
        )

    async def run(self) -> None:
        """Execute N repetitions of scenario benchmark."""
        logger.info(
            "Scenario benchmark: %s, v=%d, io=%d, P=%d, %ds x %d reps",
            self.scenario,
            self.val_size,
            self.io_threads,
            self.pipelining,
            self.duration,
            self.repetitions,
        )
        self.file_protocol.write_status(self.status)

        topology_group = TopologyGroup.for_task(
            self.server_infos,
            self.topology,
            self.source,
            self.specifier,
            io_threads=self.io_threads,
            make_args=self.make_args,
            server_args=self.server_args_effective,
            cpu_override=self.server_cpu_override,
        )

        per_run_rps: List[float] = []
        per_run_scenario_metrics: List[Dict[str, Any]] = []
        perf_counters: Optional[dict] = None

        try:
            for rep in range(self.repetitions):
                overlay_handle: Any = None
                sampler: Optional[ServerSampler] = None
                sampler_task: Optional["asyncio.Task"] = None

                # Between-rep housekeeping
                if rep > 0:
                    await topology_group.stop_all_servers()
                    server = topology_group.primary or Server(self.server_infos[0].ip)
                    await server.run_host_command("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False)

                # Start server
                await topology_group.kill_all_valkey_instances()
                await topology_group.start()
                if not topology_group.primary:
                    raise RuntimeError("Server failed to start")
                await topology_group.begin_replication()
                await topology_group.wait_for_repl_sync()
                server = topology_group.primary
                self.commit_hash = server.get_build_hash() or ""

                # Prefill keyspace
                total_conns = MIXED_THREADS * MIXED_CLIENTS
                prefill_cmd = (
                    f"~/conductress/memtier_benchmark "
                    f"--server {server.ip} --port {server.port} --protocol redis "
                    f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
                    f"--ratio 1:0 --key-pattern P:P "
                    f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                    f"--data-size {self.val_size} "
                    f"--requests {MIXED_KEYSPACE // total_conns} "
                    f"--hide-histogram"
                )
                await server.run_host_command(prefill_cmd)

                # Prefill dedicated large-value keyset for large-value-reader overlay
                if self.scenario == "large-value-reader":
                    effective_lvr_size = (
                        self.overlay_value_size if self.overlay_value_size > 0 else LARGE_VALUE_READER_DEFAULT_SIZE
                    )
                    lvr_prefill_cmd = (
                        f"~/conductress/memtier_benchmark "
                        f"--server {server.ip} --port {server.port} --protocol redis "
                        f"--threads {OVERLAY_THREADS} --clients {OVERLAY_CLIENTS} "
                        f"--ratio 1:0 --key-pattern P:P "
                        f"--key-prefix lvr: "
                        f"--key-minimum 1 --key-maximum {LARGE_VALUE_READER_KEYSPACE} "
                        f"--data-size {effective_lvr_size} "
                        f"--requests {LARGE_VALUE_READER_KEYSPACE // (OVERLAY_THREADS * OVERLAY_CLIENTS)} "
                        f"--hide-histogram"
                    )
                    await server.run_host_command(lvr_prefill_cmd)

                self.status.steps_completed = rep * 3 + 1
                self.file_protocol.write_status(self.status)

                try:
                    # Perf stat: start before measurement
                    if self.perf_stat_enabled:
                        await server.perf_stat_start()

                    # Server-side sampler (opt-in via server_sample_ms). Started
                    # just before the overlay so it captures the pre-overlay
                    # baseline, stopped after the background measurement ends.
                    if self.server_sample_ms > 0:
                        sampler = ServerSampler(server.ip, server.port, self.server_sample_ms)
                        sampler_task = asyncio.ensure_future(sampler.run())

                    # Start overlay driver (CommandOverlay for the shell-command
                    # scenarios; StormOverlay for connection-storm). The offset
                    # of the overlay start relative to the background RPS series
                    # is recorded so a reader can align the memtier dip with the
                    # stall/storm timeline (see how bgsave's ~40% offset works).
                    measure_start = time.monotonic()
                    measure_start_wall = time.time()
                    overlay_handle = await self._overlay.start(server, self.file_protocol.work_dir)

                    # Brief delay to let overlay establish connections
                    await asyncio.sleep(1)
                    overlay_start_offset_s = time.monotonic() - measure_start

                    # Run background load measurement (under pathology)
                    json_out = f"/tmp/memtier_scenario_rep{rep}.json"
                    bg_ratio = set_ratio_to_memtier_ratio(self.background_set_ratio)
                    measure_cmd = (
                        f"~/conductress/memtier_benchmark "
                        f"--server {server.ip} --port {server.port} --protocol redis "
                        f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
                        f"--ratio {bg_ratio} --key-pattern R:R "
                        f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                        f"--data-size {self.val_size} "
                        f"--pipeline {self.pipelining} "
                        f"--test-time {self.duration} "
                        f"--json-out-file {json_out} "
                        f"--hide-histogram"
                    )
                    background_start_wall = time.time()
                    stdout, _ = await server.run_host_command(measure_cmd)
                    background_end_wall = time.time()

                    # Stop the sampler (background measurement is over) and collect rows.
                    server_timeline: List[Dict[str, Any]] = []
                    if sampler is not None and sampler_task is not None:
                        sampler.stop()
                        try:
                            await sampler_task
                        except asyncio.CancelledError:
                            pass
                        server_timeline = sampler.rows

                    # Stop the overlay and collect its result
                    overlay_result = await self._overlay.finish(server, overlay_handle)
                    overlay_handle = None  # cleared

                    # Perf stat: stop and collect
                    if self.perf_stat_enabled:
                        await server.perf_stat_stop()

                    # Parse baseline GET RPS
                    total_rps = parse_memtier_total_rps(stdout)
                    if total_rps is None:
                        raise RuntimeError(f"Failed to parse memtier output for rep {rep + 1}")

                    # Try to get interval RPS from JSON output (Time-Serie)
                    interval_rps: List[float] = []
                    try:
                        json_stdout, _ = await server.run_host_command(f"cat {json_out}", check=False)
                        if json_stdout.strip():
                            interval_rps = parse_memtier_json_intervals(json_out, json_stdout)
                    except Exception:
                        pass  # interval data is best-effort

                    # Fallback: parse stdout progress lines if JSON had no timeseries
                    if not interval_rps and stdout:
                        interval_rps = parse_memtier_stdout_intervals(stdout)

                    per_run_rps.append(total_rps)
                    logger.info(
                        "Rep %d/%d: %.1f baseline ops/sec under %s", rep + 1, self.repetitions, total_rps, self.scenario
                    )

                    # Compute scenario-specific metrics
                    scenario_metrics: Dict[str, Any] = {"scenario": self.scenario}
                    scenario_metrics["overlay_start_offset_s"] = round(overlay_start_offset_s, 3)
                    # Wall-clock anchors: one shared axis for every series. A plot
                    # aligns the memtier dip (background_start_wall), the server
                    # sampler rows (their own t_sent), and the storm timeline
                    # (storm.origin_wall) against these.
                    scenario_metrics["measure_start_wall"] = measure_start_wall
                    scenario_metrics["background_start_wall"] = background_start_wall
                    scenario_metrics["background_end_wall"] = background_end_wall
                    if self.server_sample_ms > 0:
                        scenario_metrics["server_timeline"] = server_timeline
                        scenario_metrics["server_timeline_tick_ms"] = self.server_sample_ms

                    # Overlay's own ops/s rate, where it has one (command overlays
                    # parse it; the storm has none and returns rate=None).
                    if overlay_result.rate is not None:
                        scenario_metrics["overlay_ops_per_sec"] = round(overlay_result.rate, 1)
                    # Overlay-specific metrics (e.g. the storm.* namespace).
                    scenario_metrics.update(overlay_result.metrics)

                    # Dip metrics from interval timeseries
                    if interval_rps:
                        # Use early intervals as "undisturbed" baseline reference
                        # (first 2 seconds before overlay fully ramps)
                        ref_rps = total_rps  # use aggregate as reference
                        dip = compute_dip_metrics(ref_rps, interval_rps)
                        scenario_metrics.update(dip)
                        scenario_metrics["interval_rps"] = [round(r, 1) for r in interval_rps]

                    per_run_scenario_metrics.append(scenario_metrics)

                    # Write per-rep metric
                    metric = MetricData(
                        metrics={"rps": total_rps, "scenario_metrics": scenario_metrics},
                        rep=rep + 1,
                    )
                    self.file_protocol.append_metric(metric)

                    self.status.steps_completed = rep * 3 + 3
                    self.file_protocol.write_status(self.status)

                    # Collect perf stat report
                    if self.perf_stat_enabled:
                        server.perf_stat_wait()
                        result_dir = self.file_protocol.get_result_dir()
                        rep_counters = await server.perf_stat_report(result_dir)
                        self._perf_stat_scope = server.perf_stat_scope or self._perf_stat_scope
                        if rep_counters:
                            if perf_counters is None:
                                perf_counters = rep_counters
                            else:
                                for bucket, events in rep_counters.items():
                                    acc = perf_counters.setdefault(bucket, {})
                                    for k, v in events.items():
                                        acc[k] = acc.get(k, 0) + v

                finally:
                    # Stop the sampler if a rep failed before it was stopped.
                    if sampler is not None and sampler_task is not None and not sampler_task.done():
                        sampler.stop()
                        try:
                            await sampler_task
                        except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
                            pass
                    # Ensure the overlay is always torn down, even on failure.
                    if overlay_handle is not None:
                        await self._overlay.abort(server, overlay_handle)
                    # Clean up multi-exec payload file if present
                    if self.scenario == "multi-exec":
                        await server.run_host_command("rm -f /tmp/multi_exec_payload.resp", check=False)

        finally:
            await topology_group.stop_all_servers()

        if not per_run_rps:
            raise RuntimeError("No successful repetitions")

        # Compute aggregated stats
        mean_rps = mean(per_run_rps)
        ci_95 = 0.0
        if len(per_run_rps) >= 2:
            ci_95 = t_dist.ppf(0.975, len(per_run_rps) - 1) * (stdev(per_run_rps) / sqrt(len(per_run_rps)))

        # Aggregate scenario metrics across reps
        agg_scenario = {"scenario": self.scenario, "per_rep": per_run_scenario_metrics}
        if per_run_scenario_metrics:
            overlay_rates = [
                m.get("overlay_ops_per_sec", 0) for m in per_run_scenario_metrics if "overlay_ops_per_sec" in m
            ]
            if overlay_rates:
                agg_scenario["mean_overlay_ops_per_sec"] = round(mean(overlay_rates), 1)
            dip_depths = [m.get("dip_depth_pct", 0) for m in per_run_scenario_metrics if "dip_depth_pct" in m]
            if dip_depths:
                agg_scenario["mean_dip_depth_pct"] = round(mean(dip_depths), 2)

        # Record results
        detailed_data = {
            "topology": self.topology.to_dict(),
            "management_cpus": self.management_cpus,
            "scenario": self.scenario,
            "duration": self.duration,
            "warmup": self.warmup,
            "io_threads": self.io_threads,
            "pipeline": self.pipelining,
            "size": self.val_size,
            "keyspace": MIXED_KEYSPACE,
            "threads": MIXED_THREADS,
            "clients": MIXED_CLIENTS,
            "repetitions": self.repetitions,
            "background_set_ratio": self.background_set_ratio,
            "server_args": self.server_args,
            "server_args_effective": self.server_args_effective,
            "overlay_spec": self.overlay_spec,
            "per_run_rps": per_run_rps,
            "mean_rps": mean_rps,
            "ci_95": ci_95,
            "scenario_metrics": agg_scenario,
        }
        if self.scenario == "large-value-reader":
            effective_lvr_size = (
                self.overlay_value_size if self.overlay_value_size > 0 else LARGE_VALUE_READER_DEFAULT_SIZE
            )
            detailed_data["overlay_value_size"] = effective_lvr_size
            detailed_data["overlay_keyspace"] = LARGE_VALUE_READER_KEYSPACE
        if perf_counters:
            detailed_data["perf_counters"] = perf_counters
            if self._perf_stat_scope:
                detailed_data["perf_counters_scope"] = self._perf_stat_scope
            detailed_data["perf_duration_seconds"] = float(self.duration)
            detailed_data["perf_rep_count"] = len(per_run_rps)

        results = BenchmarkResults(
            method=f"scenario-{self.scenario}",
            source=self.source,
            specifier=self.specifier,
            commit_hash=self.commit_hash,
            score=mean_rps,
            end_time=datetime.datetime.now(),
            data=detailed_data,
            make_args=self.make_args,
            note=self.note,
        )
        self.file_protocol.write_results(results)

        self.status.state = "completed"
        self.status.end_time = time.time()
        self.status.steps_completed = self.repetitions * 3
        self.file_protocol.write_status(self.status)

    def _parse_overlay_rps(self, output: str) -> Optional[float]:
        """Parse valkey-benchmark output for total requests/sec.

        valkey-benchmark summary line format:
          <N> requests completed in <T> seconds
        or the throughput summary line.
        """
        # Look for "requests per second" line
        for line in output.splitlines():
            # Pattern: "NNNN.NN requests per second"
            match = re.search(r"([\d.]+)\s+requests per second", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
