"""Mixed GET/SET throughput benchmark using memtier_benchmark.

Drives a configurable ratio of GET and SET commands at maximum throughput,
measuring combined RPS across N repetitions with server restarts between reps.
"""

import datetime
import logging
import re
import time
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import List, Optional

from scipy.stats import t as t_dist

from conductress.config import PERF_BENCH_KEYSPACE, ServerInfo, get_sweep_engine, should_profile_internals
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus, MetricData
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.utility import CLIENT_CPU_SATURATION_THRESHOLD, count_cpu_list

logger = logging.getLogger(__name__)

# memtier defaults — preserved exactly for backward compatibility.
MIXED_THREADS = 8
MIXED_CLIENTS = 50  # 400 total connections (8*50)
MIXED_KEYSPACE = PERF_BENCH_KEYSPACE  # 3M keys — same as perf task

# Upper bounds — defensible limits preventing accidental resource exhaustion.
# memtier_benchmark threads are OS threads with epoll loops; 256 threads is
# already extreme (typical hosts have <= 192 vCPUs).  Per-thread client count
# at 1000 already implies 256K connections at max threads; beyond that is an
# operational hazard, not a benchmark.
MAX_MEMTIER_THREADS = 256
MAX_MEMTIER_CLIENTS = 1000

# Minimum total connections to ensure prefill requests > 0.
# prefill_requests = keyspace // total_connections; with keyspace = 3M,
# total_connections must be <= 3M.  Practical minimum is 1 connection.
MIN_TOTAL_CONNECTIONS = 1


def set_ratio_to_memtier_ratio(set_pct: int) -> str:
    """Convert a SET percentage (0-100) to memtier --ratio=SET:GET string.

    memtier --ratio is SET:GET. E.g. set_pct=20 -> ratio 1:4 (1 SET per 4 GETs).
    Special cases: 0 -> 0:1 (pure GET), 100 -> 1:0 (pure SET).
    """
    if set_pct == 0:
        return "0:1"
    if set_pct == 100:
        return "1:0"
    from math import gcd

    get_pct = 100 - set_pct
    g = gcd(set_pct, get_pct)
    return f"{set_pct // g}:{get_pct // g}"


def parse_memtier_total_rps(output: str) -> Optional[float]:
    """Parse memtier_benchmark stdout for the Totals line ops/sec.

    Expected format (from memtier summary table):
    Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency  ...
    ...
    Totals      1234567.89      ...
    """
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Totals":
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def parse_memtier_interval_lines(output: str) -> List[float]:
    """Parse memtier interval output for per-second RPS.

    memtier interval lines: "[RUN #N NN%, ...]  NNN.NN KB - Gets: NNN, Sets: NNN, Ops: NNN ..."
    """
    rps_values: List[float] = []
    for line in output.splitlines():
        match = re.search(r"Ops:\s+(\d+)", line)
        if match:
            rps_values.append(float(match.group(1)))
    return rps_values


def _effective_memtier_threads(override: int) -> int:
    """Return the memtier thread count: explicit override or the legacy default."""
    return override if override > 0 else MIXED_THREADS


def _effective_memtier_clients(override: int) -> int:
    """Return the memtier per-thread client count: explicit override or the legacy default."""
    return override if override > 0 else MIXED_CLIENTS


def _validate_memtier_bounds(threads: int, clients: int) -> None:
    """Validate memtier thread/client values against named bounds.

    Raises ValueError with a descriptive message on violation.
    """
    if threads < 0:
        raise ValueError(f"memtier_threads must be >= 0 (0 = default), got {threads}")
    if clients < 0:
        raise ValueError(f"memtier_clients must be >= 0 (0 = default), got {clients}")
    if threads > MAX_MEMTIER_THREADS:
        raise ValueError(f"memtier_threads must be <= {MAX_MEMTIER_THREADS}, got {threads}")
    if clients > MAX_MEMTIER_CLIENTS:
        raise ValueError(f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}, got {clients}")
    # Check effective total connections ensure prefill requests > 0
    eff_threads = _effective_memtier_threads(threads)
    eff_clients = _effective_memtier_clients(clients)
    total = eff_threads * eff_clients
    if total < MIN_TOTAL_CONNECTIONS:
        raise ValueError(
            f"Effective total connections ({eff_threads} × {eff_clients} = {total}) "
            f"must be >= {MIN_TOTAL_CONNECTIONS}"
        )
    if total > MIXED_KEYSPACE:
        raise ValueError(
            f"Effective total connections ({total}) exceeds keyspace ({MIXED_KEYSPACE}); "
            f"prefill requests per connection would be zero"
        )


# --- GNU /usr/bin/time CPU parsing ---

# GNU time -v outputs lines like:
#   User time (seconds): 12.34
#   System time (seconds): 5.67
#   Elapsed (wall clock) time (h:mm:ss or m:ss): 0:30.12
_GNU_TIME_USER_RE = re.compile(r"User time \(seconds\):\s+([\d.]+)")
_GNU_TIME_SYS_RE = re.compile(r"System time \(seconds\):\s+([\d.]+)")
_GNU_TIME_WALL_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+(?:(\d+):)?(\d+):([\d.]+)")


def parse_gnu_time_stderr(stderr: str) -> Optional[dict]:
    """Parse GNU /usr/bin/time -v output from stderr.

    Returns a dict with keys: user_seconds, system_seconds, wall_seconds,
    cpu_seconds (user+system), cores_busy (cpu_seconds / wall_seconds).
    Returns None if parsing fails (tool missing or output malformed).
    """
    user_match = _GNU_TIME_USER_RE.search(stderr)
    sys_match = _GNU_TIME_SYS_RE.search(stderr)
    wall_match = _GNU_TIME_WALL_RE.search(stderr)

    if not (user_match and sys_match and wall_match):
        return None

    user_s = float(user_match.group(1))
    sys_s = float(sys_match.group(1))

    # Wall time: h:mm:ss.ff or m:ss.ff
    hours = int(wall_match.group(1)) if wall_match.group(1) else 0
    minutes = int(wall_match.group(2))
    seconds = float(wall_match.group(3))
    wall_s = hours * 3600 + minutes * 60 + seconds

    cpu_s = user_s + sys_s
    cores_busy = cpu_s / wall_s if wall_s > 0 else 0.0

    return {
        "user_seconds": round(user_s, 3),
        "system_seconds": round(sys_s, 3),
        "wall_seconds": round(wall_s, 3),
        "cpu_seconds": round(cpu_s, 3),
        "cores_busy": round(cores_busy, 3),
    }


@dataclass
class MixedTaskData(BaseTaskData):
    """Task data for mixed GET/SET throughput measurement."""

    set_ratio: int  # percentage of SET commands (0-100)
    val_size: int
    io_threads: int
    pipelining: int
    duration: int
    warmup: int = 5
    perf_stat_enabled: bool = False
    key_size: int = 0  # 0 = standard 16B keys
    repetitions: int = 3
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""
    server_args: str = ""  # extra raw args appended to the server command line (override defaults)
    memtier_threads: int = 0  # 0 = MIXED_THREADS default (8)
    memtier_clients: int = 0  # 0 = MIXED_CLIENTS default (50 per thread)

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "MixedTaskData"
        if not (0 <= self.set_ratio <= 100):
            raise ValueError(f"set_ratio must be 0-100, got {self.set_ratio}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")
        _validate_memtier_bounds(self.memtier_threads, self.memtier_clients)

    @property
    def effective_threads(self) -> int:
        """Memtier thread count that will actually be used."""
        return _effective_memtier_threads(self.memtier_threads)

    @property
    def effective_clients(self) -> int:
        """Memtier per-thread client count that will actually be used."""
        return _effective_memtier_clients(self.memtier_clients)

    @property
    def total_connections(self) -> int:
        """Total TCP connections = threads × clients."""
        return self.effective_threads * self.effective_clients

    def short_description(self) -> str:
        from conductress.utility import HumanByte, HumanTime

        desc = (
            f"mixed {self.set_ratio}%SET/{100-self.set_ratio}%GET "
            f"v={HumanByte.to_human(self.val_size)} "
            f"io={self.io_threads} P={self.pipelining} "
            f"{HumanTime.to_human(self.duration)}"
        )
        if self.memtier_threads or self.memtier_clients:
            desc += f" {self.total_connections}c({self.effective_threads}t×{self.effective_clients}c)"
        if self.perf_stat_enabled:
            desc += " perf-stat"
        return desc

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "MixedTaskRunner":
        return MixedTaskRunner(
            task_name=self.task_id,
            server_infos=server_infos,
            source=self.source,
            specifier=self.specifier,
            make_args=self.make_args,
            io_threads=self.io_threads,
            val_size=self.val_size,
            pipelining=self.pipelining,
            set_ratio=self.set_ratio,
            warmup=self.warmup,
            duration=self.duration,
            repetitions=self.repetitions,
            perf_stat_enabled=self.perf_stat_enabled,
            key_size=self.key_size,
            note=self.note,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
            server_args=self.server_args,
            memtier_threads=self.memtier_threads,
            memtier_clients=self.memtier_clients,
        )


class MixedTaskRunner(BaseTaskRunner):
    """Runs memtier_benchmark with a configurable GET/SET ratio at max throughput."""

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
        set_ratio: int,
        warmup: int,
        duration: int,
        repetitions: int,
        perf_stat_enabled: bool = False,
        key_size: int = 0,
        note: str = "",
        server_cpu_override: str = "",
        benchmark_cpu_override: str = "",
        server_args: str = "",
        memtier_threads: int = 0,
        memtier_clients: int = 0,
    ):
        super().__init__(task_name)
        self.server_infos = server_infos
        self.source = source
        self.specifier = specifier
        self.make_args = make_args
        self.io_threads = io_threads
        self.val_size = val_size
        self.pipelining = pipelining
        self.set_ratio = set_ratio
        self.warmup = warmup
        self.duration = duration
        self.repetitions = repetitions
        self.perf_stat_enabled = perf_stat_enabled
        self.key_size = key_size
        self.note = note
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override
        self.server_args = server_args

        # Effective memtier concurrency
        self.memtier_threads = _effective_memtier_threads(memtier_threads)
        self.memtier_clients = _effective_memtier_clients(memtier_clients)
        self.total_connections = self.memtier_threads * self.memtier_clients

        self.commit_hash = ""
        self._profile_internals = should_profile_internals(get_sweep_engine(source))

        # Client CPU telemetry — populated per-rep via GNU /usr/bin/time
        self._client_cores_busy_per_rep: list[float] = []
        self._gnu_time_available: Optional[bool] = None  # probed once per run

        self.status = BenchmarkStatus(
            steps_total=repetitions * 2,
            task_type=f"perf-mixed-{set_ratio}pct-set",
        )

    async def _probe_gnu_time(self, server: Server) -> bool:
        """Check whether /usr/bin/time exists on the remote host. Probed once."""
        if self._gnu_time_available is not None:
            return self._gnu_time_available
        try:
            stdout, _ = await server.run_host_command("/usr/bin/time --version 2>&1 || true", check=False)
            # GNU time --version prints to stderr (some versions) or stdout
            self._gnu_time_available = "GNU" in stdout or "time" in stdout.lower()
        except Exception:
            self._gnu_time_available = False
        if not self._gnu_time_available:
            logger.warning("GNU /usr/bin/time not found on remote host; " "client CPU measurement will be unavailable")
        return self._gnu_time_available

    def _build_taskset_prefix(self) -> str:
        """Build a taskset prefix if benchmark_cpu_override is set.

        Returns 'taskset -c <cpulist> ' (trailing space) or '' if no pinning.
        taskset -c accepts the same range/list notation as the cpulist field
        (e.g. '0-7,16-23').  This pins both prefill and measurement memtier
        invocations to the declared cores, making the result's capacity_cores
        field truthful.
        """
        if self.benchmark_cpu_override:
            return f"taskset -c {self.benchmark_cpu_override} "
        return ""

    def _build_warmup_arg(self) -> str:
        """Return memtier's warmup option, or an empty string when disabled."""
        return f"--warmup-period {self.warmup} " if self.warmup > 0 else ""

    def _build_timed_command(self, cmd: str) -> str:
        """Wrap a command with /usr/bin/time -v for CPU accounting.

        GNU time writes its report to stderr; the wrapped command's stdout
        is unaffected.  The time report is parsed from the combined stderr
        after the command completes.

        Composition with taskset: taskset is already embedded in ``cmd`` via
        _build_taskset_prefix, so time wraps the entire pinned invocation.
        This is correct — time measures the process tree including any
        scheduling constraints.
        """
        return f"/usr/bin/time -v {cmd}"

    def _compute_client_cpu_meta(self) -> dict:
        """Build the mixed-task-specific client CPU metadata block.

        Two capacity models:
        - **Pinned** (benchmark_cpu_override set): capacity_cores is the lower
          of memtier worker-thread count and cpulist core count.  basis =
          'min(memtier_thread_count,taskset_cpulist)'.  This is the hard CPU
          ceiling because neither extra workers nor extra pinned cores can
          increase parallelism alone.  Denominator for utilization.
        - **Unpinned** (no override): capacity_cores = memtier_threads.  Each
          memtier worker thread runs a single epoll loop that can consume at
          most one core; with OS-level scheduling the thread count is the
          practical throughput ceiling.  basis = 'memtier_thread_count'.
          Denominator for utilization.

        Both models use the shared 0.90 CLIENT_CPU_SATURATION_THRESHOLD.
        When empirical cores_busy data is absent, utilization/saturated are
        omitted (provenance = 'unavailable').
        """
        if self.benchmark_cpu_override:
            pinned_count = count_cpu_list(self.benchmark_cpu_override)
            capacity_cores = min(self.memtier_threads, pinned_count) if pinned_count else self.memtier_threads
            capacity_basis = "min(memtier_thread_count,taskset_cpulist)"
        else:
            capacity_cores = self.memtier_threads
            capacity_basis = "memtier_thread_count"

        meta: dict = {
            "capacity_cores": capacity_cores,
            "capacity_basis": capacity_basis,
            "memtier_threads": self.memtier_threads,
            "memtier_clients": self.memtier_clients,
            "total_connections": self.total_connections,
        }

        if self.benchmark_cpu_override:
            meta["benchmark_cpu_override"] = self.benchmark_cpu_override

        if self._client_cores_busy_per_rep:
            cores_busy = [round(v, 3) for v in self._client_cores_busy_per_rep]
            meta["cores_busy_per_rep"] = cores_busy
            meta["measurement_method"] = "gnu_time"
            if capacity_cores and capacity_cores > 0:
                utilization = max(self._client_cores_busy_per_rep) / capacity_cores
                meta["utilization"] = round(utilization, 3)
                meta["saturated"] = utilization >= CLIENT_CPU_SATURATION_THRESHOLD
        else:
            meta["measurement_method"] = "unavailable"
            meta["note"] = (
                "GNU /usr/bin/time not available on remote host or all reps "
                "failed to parse; no empirical client CPU data collected."
            )

        return meta

    async def run(self) -> None:
        """Execute N repetitions of mixed GET/SET benchmark with memtier."""
        logger.info(
            "Mixed benchmark: %d%%SET/%d%%GET, v=%d, io=%d, P=%d, %ds x %d reps, "
            "%d threads x %d clients = %d connections",
            self.set_ratio,
            100 - self.set_ratio,
            self.val_size,
            self.io_threads,
            self.pipelining,
            self.duration,
            self.repetitions,
            self.memtier_threads,
            self.memtier_clients,
            self.total_connections,
        )
        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos,
            self.source,
            self.specifier,
            self.io_threads,
            self.make_args,
            server_cpu_override=self.server_cpu_override,
            server_args=self.server_args,
        )

        per_run_rps: List[float] = []
        perf_counters: Optional[dict] = None

        try:
            for rep in range(self.repetitions):
                # Between-rep housekeeping
                if rep > 0:
                    await replication_group.stop_all_servers()
                    server = replication_group.primary or Server(self.server_infos[0].ip)
                    await server.run_host_command("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False)

                # Start server
                await replication_group.kill_all_valkey_instances()
                await replication_group.start()
                if not replication_group.primary:
                    raise RuntimeError("Server failed to start")
                server = replication_group.primary
                self.commit_hash = server.get_build_hash() or ""

                # Set client capacity model (once)
                if rep == 0:
                    # Probe GNU time availability on the benchmark host
                    await self._probe_gnu_time(server)

                # Build taskset prefix — applied to both prefill and measure
                taskset_pfx = self._build_taskset_prefix()

                # Prefill keyspace so GETs hit (same approach as perf task)
                prefill_cmd = (
                    f"{taskset_pfx}~/conductress/memtier_benchmark "
                    f"--server {server.ip} --port {server.port} --protocol redis "
                    f"--threads {self.memtier_threads} --clients {self.memtier_clients} "
                    f"--ratio 1:0 --key-pattern P:P "
                    f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                    f"--data-size {self.val_size} "
                    f"--requests {MIXED_KEYSPACE // self.total_connections} "
                    f"--hide-histogram"
                )
                await server.run_host_command(prefill_cmd)
                self.status.steps_completed = rep * 2 + 1
                self.file_protocol.write_status(self.status)

                # Perf stat: start before measurement
                if self.perf_stat_enabled:
                    await server.perf_stat_start()

                # Run measurement phase
                ratio_str = set_ratio_to_memtier_ratio(self.set_ratio)
                measure_cmd = (
                    f"{taskset_pfx}~/conductress/memtier_benchmark "
                    f"--server {server.ip} --port {server.port} --protocol redis "
                    f"--threads {self.memtier_threads} --clients {self.memtier_clients} "
                    f"--ratio {ratio_str} --key-pattern R:R "
                    f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                    f"--data-size {self.val_size} "
                    f"--pipeline {self.pipelining} "
                    f"{self._build_warmup_arg()}"
                    f"--test-time {self.duration} "
                    f"--hide-histogram"
                )

                # Wrap with GNU time for empirical client CPU measurement
                use_gnu_time = self._gnu_time_available is True
                if use_gnu_time:
                    timed_cmd = self._build_timed_command(measure_cmd)
                    stdout, stderr = await server.run_host_command(timed_cmd)
                else:
                    stdout, _ = await server.run_host_command(measure_cmd)
                    stderr = ""

                # Extract client CPU from GNU time output
                if use_gnu_time and stderr:
                    time_data = parse_gnu_time_stderr(stderr)
                    if time_data and time_data["wall_seconds"] > 0:
                        self._client_cores_busy_per_rep.append(time_data["cores_busy"])
                        logger.info(
                            "Rep %d client CPU: %.3f cores busy " "(%.1fs user + %.1fs sys / %.1fs wall)",
                            rep + 1,
                            time_data["cores_busy"],
                            time_data["user_seconds"],
                            time_data["system_seconds"],
                            time_data["wall_seconds"],
                        )
                    else:
                        logger.warning(
                            "Rep %d: GNU time output present but unparseable; " "client CPU unavailable this rep",
                            rep + 1,
                        )

                # Perf stat: stop and collect
                if self.perf_stat_enabled:
                    await server.perf_stat_stop()

                # Parse total RPS from memtier output
                total_rps = parse_memtier_total_rps(stdout)
                if total_rps is None:
                    raise RuntimeError(f"Failed to parse memtier output for rep {rep + 1}")

                per_run_rps.append(total_rps)
                logger.info("Rep %d/%d: %.1f ops/sec", rep + 1, self.repetitions, total_rps)

                # Write per-rep metric
                metric = MetricData(metrics={"rps": total_rps}, rep=rep + 1)
                self.file_protocol.append_metric(metric)

                self.status.steps_completed = rep * 2 + 2
                self.file_protocol.write_status(self.status)

                # Collect perf stat report
                if self.perf_stat_enabled:
                    server.perf_stat_wait()
                    result_dir = self.file_protocol.get_result_dir()
                    rep_counters = await server.perf_stat_report(result_dir)
                    if rep_counters:
                        if perf_counters is None:
                            perf_counters = rep_counters
                        else:
                            for bucket, events in rep_counters.items():
                                acc = perf_counters.setdefault(bucket, {})
                                for k, v in events.items():
                                    acc[k] = acc.get(k, 0) + v

        finally:
            await replication_group.stop_all_servers()

        if not per_run_rps:
            raise RuntimeError("No successful repetitions")

        # Compute aggregated stats
        mean_rps = mean(per_run_rps)
        ci_95 = 0.0
        if len(per_run_rps) >= 2:
            ci_95 = t_dist.ppf(0.975, len(per_run_rps) - 1) * (stdev(per_run_rps) / sqrt(len(per_run_rps)))

        # Record results
        detailed_data = {
            "set_ratio": self.set_ratio,
            "memtier_ratio": set_ratio_to_memtier_ratio(self.set_ratio),
            "duration": self.duration,
            "warmup": self.warmup,
            "io_threads": self.io_threads,
            "pipeline": self.pipelining,
            "size": self.val_size,
            "key_size": self.key_size,
            "keyspace": MIXED_KEYSPACE,
            "threads": self.memtier_threads,
            "clients": self.memtier_clients,
            "total_connections": self.total_connections,
            "repetitions": self.repetitions,
            "per_run_rps": per_run_rps,
            "mean_rps": mean_rps,
            "ci_95": ci_95,
        }

        # Client CPU provenance — mixed-task-specific metadata block with
        # capacity model, empirical measurement, and saturation detection.
        detailed_data["client_cpu"] = self._compute_client_cpu_meta()

        if perf_counters:
            detailed_data["perf_counters"] = perf_counters
            detailed_data["perf_duration_seconds"] = float(self.duration)
            detailed_data["perf_rep_count"] = len(per_run_rps)

        results = BenchmarkResults(
            method=f"perf-mixed-{self.set_ratio}set",
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
        self.status.steps_completed = self.repetitions * 2
        self.file_protocol.write_status(self.status)
