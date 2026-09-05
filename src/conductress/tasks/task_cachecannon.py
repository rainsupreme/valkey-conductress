"""Cachecannon benchmark task -- second-opinion generator instrument.

cachecannon is a Rust load generator using io_uring (ringline framework).
Its workload definition differs from valkey-benchmark -- results are a SEPARATE
series, never sweep-comparable. Use for generator-wall cross-checks and
absolute ceiling validation.

Binary: /home/ec2-user/cachecannon/target/release/cachecannon (on bench hosts).
Config: TOML file generated per-run in the result directory.
"""

import datetime
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from conductress.config import (
    DEFAULT_DURATION,
    DEFAULT_MAKE_ARGS,
    DEFAULT_PIPELINING,
    DEFAULT_REPETITIONS,
    DEFAULT_VAL_SIZE,
    DEFAULT_WARMUP,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
    ServerInfo,
)
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.utility import (
    HumanByte,
    HumanTime,
    RealtimeCommand,
    count_cpu_list,
    sample_process_tree_cpu,
    summarize_client_cpu,
)

logger = logging.getLogger(__name__)

# Default path to cachecannon binary on bench hosts
DEFAULT_CACHECANNON_BINARY = "/home/ec2-user/cachecannon/target/release/cachecannon"


def _compute_aggregated_stats(per_run_rps: list) -> tuple:
    """Compute mean and 95% CI. Deferred import to avoid circular dependency."""
    from conductress.tasks.task_perf_benchmark import compute_aggregated_stats

    return compute_aggregated_stats(per_run_rps)


def _should_stop_adaptive(per_run_rps: list, rep: int, min_reps: int, target_cv: float) -> bool:
    """Adaptive stopping test. Deferred import to avoid circular dependency.

    Shares the perf-benchmark implementation so both generators converge on
    precision identically rather than drifting apart.
    """
    from conductress.tasks.task_perf_benchmark import should_stop_adaptive

    return should_stop_adaptive(per_run_rps, rep, min_reps, target_cv)


def _latency_from_json(command: str, block: dict) -> dict:
    """Convert a cachecannon JSON latency object to the recorded ms schema.

    cachecannon reports latency percentiles as integer microseconds, so no unit
    inference is required.
    """
    required = ("p50_us", "p90_us", "p99_us", "p999_us", "p9999_us", "max_us")
    missing = [field for field in required if field not in block]
    if missing:
        raise ValueError(f"cachecannon JSON latency object missing {missing}: {block!r}")
    return {
        "command": command,
        "p50_ms": block["p50_us"] / 1000.0,
        "p90_ms": block["p90_us"] / 1000.0,
        "p99_ms": block["p99_us"] / 1000.0,
        "p999_ms": block["p999_us"] / 1000.0,
        "p9999_ms": block["p9999_us"] / 1000.0,
        "max_ms": block["max_us"] / 1000.0,
        "count": block.get("count", 0),
    }


def parse_json_results(output: str) -> dict:
    """Parse cachecannon's NDJSON output and return EXACT result values.

    cachecannon emits newline-delimited JSON when ``[admin] format = "json"`` is
    set. Unstructured progress lines (``[precheck]``, ringline diagnostics) are
    interleaved on the same streams and are skipped.

    Exactness is the entire point of this parser. The human ``clean`` formatter
    renders throughput through an abbreviating helper, so 3,104,882 ops/sec
    prints as ``3.10M`` -- three significant figures, a quantization step of
    roughly 0.3-0.5% at multi-million ops/sec. That is at or above the sweep
    target CV, which makes variance unmeasurable and collapses independent
    repetitions onto a single representable value. The JSON ``result`` message
    carries the exact integer instead.

    Returns dict with keys: throughput_rps, error_pct, hit_rate, latency,
    latency_get, latency_set, requests, responses, errors, duration_secs.
    """
    result_obj = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "result":
            result_obj = candidate

    if result_obj is None:
        raise ValueError(
            'No cachecannon JSON result message found. Is [admin] format = "json" set? '
            f"Output tail:\n{output[-1000:]}"
        )

    for field in ("throughput", "err_pct", "hits", "misses", "hit_pct"):
        if field not in result_obj:
            raise ValueError(f"cachecannon JSON result missing {field!r}: {result_obj!r}")

    latency_get = _latency_from_json("GET", result_obj["get"]) if "get" in result_obj else None
    latency_set = _latency_from_json("SET", result_obj["set"]) if "set" in result_obj else None

    # Primary latency is whichever command actually carried the workload. The
    # previous clean-output parser recorded the LAST latency row it matched,
    # which for a GET-only run could be an all-zero SET row.
    primary = latency_get
    if latency_set and (latency_get is None or latency_set["count"] > latency_get["count"]):
        primary = latency_set

    return {
        "throughput_rps": float(result_obj["throughput"]),
        "error_pct": float(result_obj["err_pct"]),
        "hit_rate": {
            "percent": float(result_obj["hit_pct"]),
            "hits": float(result_obj["hits"]),
            "misses": float(result_obj["misses"]),
        },
        "latency": primary,
        "latency_get": latency_get,
        "latency_set": latency_set,
        "requests": result_obj.get("requests"),
        "responses": result_obj.get("responses"),
        "errors": result_obj.get("errors"),
        "duration_secs": result_obj.get("duration_secs"),
    }


def workload_issues_gets(test: str, set_ratio: int) -> bool:
    """Whether the effective workload sends any GET commands.

    Mirrors generate_toml_config's weight mapping: set_ratio > 0 overrides
    'test' with a mixed workload (get_weight = 100 - set_ratio); set_ratio == 0
    means a pure workload per 'test'. The hit-rate prefill guard is only
    meaningful when this returns True -- pure-SET runs have hit rate 0.0 by
    definition (SETs cannot hit)."""
    if set_ratio > 0:
        return set_ratio < 100
    return test == "get"


def generate_toml_config(
    *,
    duration: int,
    warmup: int,
    threads: int,
    cpu_list: str,
    endpoint: str,
    connections: int,
    pipeline_depth: int,
    keyspace_count: int,
    val_size: int,
    test: str = "get",
    set_ratio: int = 0,
    distribution: str = "uniform",
) -> str:
    """Generate a cachecannon TOML configuration file.

    Args:
        duration: Test duration in seconds.
        warmup: Warmup duration in seconds.
        threads: Number of worker threads.
        cpu_list: Comma-separated CPU list (e.g. '8,9,10,11').
        endpoint: Server endpoint (e.g. '127.0.0.1:6379').
        connections: Total connections.
        pipeline_depth: Pipeline depth.
        keyspace_count: Number of keys in the keyspace.
        val_size: Value size in bytes.
        test: Command to bench ('get' or 'set'). Ignored when set_ratio > 0.
        set_ratio: Percentage of SET commands (0-100). When > 0, overrides
            'test' with a mixed GET/SET workload (get = 100 - set_ratio).
        distribution: Key distribution ('uniform' or 'zipf').

    Returns:
        TOML configuration string.
    """
    if not 0 <= set_ratio <= 100:
        raise ValueError(f"set_ratio must be 0-100, got {set_ratio}")
    if distribution not in ("uniform", "zipf"):
        raise ValueError(f"distribution must be 'uniform' or 'zipf', got '{distribution}'")

    # Map test name / set_ratio to cachecannon command weights. ALWAYS write
    # all three weights explicitly: cachecannon applies serde per-field
    # defaults (get=80, set=20, delete=0) to any weight omitted from the
    # TOML, so a config with only 'set = 100' silently runs 80% GET.
    if set_ratio > 0:
        get_weight, set_weight = 100 - set_ratio, set_ratio
    elif test == "set":
        get_weight, set_weight = 0, 100
    else:  # 'get' (default)
        get_weight, set_weight = 100, 0
    commands_section = f"get = {get_weight}\nset = {set_weight}\ndelete = 0"

    toml = f"""[general]
duration = "{duration}s"
warmup = "{warmup}s"
threads = {threads}
cpu_list = "{cpu_list}"
io_engine = "uring"

[target]
endpoints = ["{endpoint}"]
protocol = "resp"

[connection]
connections = {connections}
pipeline_depth = {pipeline_depth}

[workload]
prefill = true

[workload.keyspace]
length = 16
count = {keyspace_count}
distribution = "{distribution}"

[workload.commands]
{commands_section}

[workload.values]
length = {val_size}

[timestamps]
enabled = true
mode = "userspace"

[admin]
format = "json"
"""
    return toml


@dataclass
class CachecannonTaskData(BaseTaskData):
    """Data class for cachecannon benchmark task.

    cachecannon is a second-opinion generator instrument -- its workload
    definition differs from valkey-benchmark. Results are a SEPARATE series,
    never directly comparable with the valkey-benchmark sweep history.
    """

    test: str = "get"
    val_size: int = DEFAULT_VAL_SIZE
    pipelining: int = DEFAULT_PIPELINING
    connections: int = 1200
    threads: int = PERF_BENCH_THREADS
    warmup: int = DEFAULT_WARMUP
    duration: int = DEFAULT_DURATION
    repetitions: int = DEFAULT_REPETITIONS
    keyspace_count: int = PERF_BENCH_KEYSPACE
    io_threads: int = 9  # server io-threads
    cachecannon_binary: str = DEFAULT_CACHECANNON_BINARY
    server_args: str = ""
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""
    set_ratio: int = 0  # 0 = pure workload per 'test'; >0 = mixed GET/SET
    distribution: str = "uniform"  # key distribution: 'uniform' or 'zipf'
    max_reps: int = 0  # 0 = fixed reps; >0 = adaptive mode upper limit
    target_cv: float = 0.0  # adaptive: stop early when 95% CI half-width (% of mean) <= this; 0 = disabled
    sweep_commit: str = ""  # non-empty marks this as a sweep task

    def __post_init__(self):
        super().__post_init__()
        self.warmup = int(self.warmup)
        self.duration = int(self.duration)
        # These must be REAL controls. An accepted-but-ignored value is a fake
        # lever: it makes the operator believe they configured a precision
        # target that never took effect. Adaptive stopping can only ever fire
        # between `repetitions` and `max_reps`, so a target with no headroom
        # above the minimum is rejected rather than silently doing nothing.
        if self.max_reps and self.max_reps < self.repetitions:
            raise ValueError(
                f"max_reps ({self.max_reps}) must be >= repetitions ({self.repetitions}); "
                "repetitions is the adaptive minimum and max_reps the ceiling"
            )
        if self.target_cv < 0:
            raise ValueError(f"target_cv must be >= 0, got {self.target_cv}")
        if self.target_cv > 0 and self.max_reps <= self.repetitions:
            raise ValueError(
                f"target_cv={self.target_cv} requires max_reps > repetitions "
                f"(got max_reps={self.max_reps}, repetitions={self.repetitions}); "
                "otherwise adaptive stopping has no reps to skip and the target is a no-op"
            )

    def workload_label(self) -> str:
        """Human label for the workload: 'get', 'set', or 'mixed s<N>'."""
        label = f"mixed s{self.set_ratio}" if self.set_ratio > 0 else self.test
        if self.distribution != "uniform":
            label += f" {self.distribution}"
        return label

    def short_description(self) -> str:
        return (
            f"cachecannon {self.workload_label()}, {HumanByte.to_human(self.val_size)} values, "
            f"P{self.pipelining}, {self.connections}c, {self.threads}t, "
            f"{HumanTime.to_human(self.duration)} x{self.repetitions}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "CachecannonTaskRunner":
        return CachecannonTaskRunner(
            task_id=self.task_id,
            server_infos=server_infos,
            source=self.source,
            specifier=self.specifier,
            make_args=self.make_args,
            io_threads=self.io_threads,
            test=self.test,
            val_size=self.val_size,
            pipelining=self.pipelining,
            connections=self.connections,
            threads=self.threads,
            warmup=self.warmup,
            duration=self.duration,
            repetitions=self.repetitions,
            max_reps=self.max_reps,
            target_cv=self.target_cv,
            keyspace_count=self.keyspace_count,
            cachecannon_binary=self.cachecannon_binary,
            server_args=self.server_args,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
            set_ratio=self.set_ratio,
            distribution=self.distribution,
            note=self.note,
        )


class CachecannonTaskRunner(BaseTaskRunner):
    """Run a cachecannon benchmark against a Valkey server.

    Builds and starts the server using the standard ReplicationGroup machinery,
    allocates client CPUs, generates a TOML config, launches cachecannon, and
    parses results.
    """

    def __init__(
        self,
        task_id: str,
        server_infos: list[ServerInfo],
        source: str,
        specifier: str,
        make_args: str,
        io_threads: int,
        test: str,
        val_size: int,
        pipelining: int,
        connections: int,
        threads: int,
        warmup: int,
        duration: int,
        repetitions: int,
        keyspace_count: int,
        cachecannon_binary: str,
        server_args: str,
        server_cpu_override: str,
        benchmark_cpu_override: str,
        note: str,
        set_ratio: int = 0,
        distribution: str = "uniform",
        max_reps: int = 0,
        target_cv: float = 0.0,
    ):
        super().__init__(task_id)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{test}")

        self.server_infos = server_infos
        self.source = source
        self.specifier = specifier
        self.make_args = make_args
        self.io_threads = io_threads
        self.test = test
        self.val_size = val_size
        self.pipelining = pipelining
        self.connections = connections
        self.threads = threads
        self.warmup = warmup
        self.duration = duration
        self.repetitions = repetitions
        self.max_reps = max_reps
        self.target_cv = target_cv
        self.keyspace_count = keyspace_count
        self.cachecannon_binary = cachecannon_binary
        self.server_args = server_args
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override
        self.set_ratio = set_ratio
        self.distribution = distribution
        self.note = note

        self.commit_hash = ""
        self._client_cores_busy_per_rep: list[float] = []
        self._client_allocated_cores: Optional[int] = None

        workload = f"mixed s{set_ratio}" if set_ratio > 0 else test
        if distribution != "uniform":
            workload += f" {distribution}"
        self.workload = workload
        effective_reps = max_reps if max_reps > 0 else repetitions
        rep_label = f"x{repetitions}" if effective_reps == repetitions else f"x{repetitions}-{effective_reps}"
        self.title = (
            f"cachecannon {workload}, {source}:{specifier}, io-threads={io_threads}, "
            f"P{pipelining}, {connections}c, {threads}t, "
            f"{HumanTime.to_human(duration)} {rep_label}"
        )

        # Status tracking. Budget for the adaptive ceiling so progress never
        # exceeds 100% when extra reps are needed to hit the precision target.
        self.status = BenchmarkStatus(
            steps_total=(warmup + duration) * effective_reps,
            task_type=f"cachecannon-{workload.replace(' ', '-')}",
        )

    def _allocate_benchmark_cpus(self, client: "Server", server: "Server") -> Optional[AllocationTag]:
        """Allocate CPUs for the cachecannon client. Returns the tag or None."""
        if self.benchmark_cpu_override:
            self.logger.info("Using explicit benchmark CPU override: %s", self.benchmark_cpu_override)
            return None

        target_ip = server.ip
        if target_ip not in {"127.0.0.1", "localhost", "::1"}:
            return None

        self.logger.info("Local benchmark detected - allocating client CPUs")
        server_tag = AllocationTag(task_id=f"server_{server.ip}_{server.port}", purpose="server")
        benchmark_alloc_tag = AllocationTag(task_id=self.task_name, purpose="benchmark")
        net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
        platform = getattr(server, "_platform_info", None)
        is_chiplet = platform is not None and platform.needs_single_cache_pinning
        benchmark_cpus = client._cpu_allocator.allocate(
            client.ip,
            benchmark_alloc_tag,
            count=self.threads,
            require_numa=net_numa,
            avoid_tags=[server_tag],
            prefer_different_cache=True,
            minimize_cache_groups=is_chiplet,
        )
        self.logger.info(
            "Allocated CPUs %s for cachecannon (NUMA node %d)",
            benchmark_cpus,
            net_numa,
        )
        return benchmark_alloc_tag

    def _get_cpu_list(self, client: "Server", benchmark_alloc_tag: Optional[AllocationTag]) -> str:
        """Get the comma-separated CPU list for cachecannon's cpu_list config."""
        if self.benchmark_cpu_override:
            return self.benchmark_cpu_override
        if benchmark_alloc_tag:
            allocated = client._cpu_allocator.get_allocation(client.ip, benchmark_alloc_tag)
            if allocated:
                return ",".join(map(str, allocated))
        # Fallback: let the OS schedule
        return ""

    def _build_command(self, toml_path: str, cpu_list: str) -> str:
        """Build the cachecannon launch command."""
        if cpu_list:
            # Use numactl for memory binding even though cachecannon pins its own threads
            from conductress.utility import parse_cpulist

            cpus = parse_cpulist(cpu_list)
            # Determine NUMA nodes for membind
            numa_node = 0  # sensible default
            return f"numactl --membind={numa_node} {self.cachecannon_binary} {toml_path}"
        return f"{self.cachecannon_binary} {toml_path}"

    async def run(self):
        """Run the cachecannon benchmark with repetitions."""
        self.logger.info("preparing: %s", self.title)
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

        benchmark_alloc_tag = None
        client = None
        server = None
        per_run_rps: list[float] = []
        all_results: list[dict] = []

        try:
            effective_reps = self.max_reps if self.max_reps > 0 else self.repetitions
            for rep in range(effective_reps):
                # Between-rep housekeeping
                if rep > 0:
                    await replication_group.stop_all_servers()
                    primary_server = replication_group.primary or Server(self.server_infos[0].ip)
                    platform = getattr(primary_server, "_platform_info", None)
                    if platform is None or platform.needs_drop_caches:
                        await primary_server.run_host_command(
                            "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'",
                            check=False,
                        )

                # Start server
                await replication_group.kill_all_valkey_instances()
                await replication_group.start()
                if not replication_group.primary:
                    raise RuntimeError("Replication group failed to start: no primary")

                await replication_group.begin_replication()
                await replication_group.wait_for_repl_sync()
                server = replication_group.primary
                self.commit_hash = server.get_build_hash() or ""

                # Setup client CPU allocation (once)
                if client is None:
                    client = Server("127.0.0.1")
                    await client.ensure_host_cpu_allocation()
                    benchmark_alloc_tag = self._allocate_benchmark_cpus(client, server)
                    if self.benchmark_cpu_override:
                        self._client_allocated_cores = count_cpu_list(self.benchmark_cpu_override)
                    elif benchmark_alloc_tag is not None:
                        self._client_allocated_cores = self.threads

                # Get CPU list for TOML config
                cpu_list = self._get_cpu_list(client, benchmark_alloc_tag)

                # Generate TOML config
                server_port = server.port or 6379
                endpoint = f"{server.ip}:{server_port}"

                toml_content = generate_toml_config(
                    duration=self.duration,
                    warmup=self.warmup,
                    threads=self.threads,
                    cpu_list=cpu_list,
                    endpoint=endpoint,
                    connections=self.connections,
                    pipeline_depth=self.pipelining,
                    keyspace_count=self.keyspace_count,
                    val_size=self.val_size,
                    test=self.test,
                    set_ratio=self.set_ratio,
                    distribution=self.distribution,
                )

                # Write TOML to result directory
                toml_path = str(self.file_protocol.work_dir / f"cachecannon_rep{rep+1}.toml")
                with open(toml_path, "w") as f:
                    f.write(toml_content)

                # Build and execute cachecannon command
                command_string = self._build_command(toml_path, cpu_list)
                self.logger.info("Starting cachecannon (rep %d/%d): %s", rep + 1, effective_reps, command_string)

                self.status.state = "running"
                self.file_protocol.write_status(self.status)

                # Launch cachecannon -- it handles its own warmup internally
                command = RealtimeCommand(command_string)
                command.start()

                # Collect output (cachecannon runs to completion).
                # CPU telemetry: sample_process_tree_cpu returns None once the
                # root exits, so refresh the last-known sample each poll cycle.
                output_lines: list[str] = []
                client_cpu_t0 = time.monotonic()
                client_cpu_s0 = sample_process_tree_cpu(command.p.pid) if command.p else None
                client_cpu_t1: Optional[float] = None
                client_cpu_s1: Optional[float] = None
                while command.is_running():
                    line, _ = command.poll_output()
                    while line is not None and line != "":
                        output_lines.append(line)
                        line, _ = command.poll_output()
                    if command.p:
                        sample = sample_process_tree_cpu(command.p.pid)
                        if sample is not None:
                            client_cpu_s1 = sample
                            client_cpu_t1 = time.monotonic()
                    time.sleep(1)

                # Drain remaining output
                line, _ = command.poll_output()
                while line is not None and line != "":
                    output_lines.append(line)
                    line, _ = command.poll_output()

                # Check exit code (is_running() returned False, so p.poll() has run)
                exit_code = command.p.returncode if command.p else None
                if exit_code != 0:
                    full_output = "\n".join(output_lines)
                    raise RuntimeError(f"cachecannon exited with code {exit_code}. Output:\n{full_output[-2000:]}")

                # Parse results (exact values from cachecannon's JSON output)
                full_output = "\n".join(output_lines)
                parsed = parse_json_results(full_output)

                # Fail loudly on errors or low hit rate
                if parsed["error_pct"] > 0:
                    raise RuntimeError(
                        f"cachecannon reported {parsed['error_pct']}% errors "
                        f"(threshold: 0%). Output:\n{full_output[-1000:]}"
                    )
                # Prefill sanity guard -- only meaningful when the workload
                # issues GETs (see workload_issues_gets): a pure-SET run has
                # hit rate 0.0 by definition, and the unconditional check
                # failed every valid pure-SET task.
                if (
                    workload_issues_gets(self.test, self.set_ratio)
                    and parsed["hit_rate"]
                    and parsed["hit_rate"]["percent"] < 99.0
                ):
                    raise RuntimeError(
                        f"cachecannon hit rate {parsed['hit_rate']['percent']}% "
                        f"(threshold: 99%). Prefill may have failed."
                    )

                per_run_rps.append(parsed["throughput_rps"])
                all_results.append(parsed)
                self.logger.info(
                    "Rep %d/%d: %.0f rps, %.2f%% errors, hit rate %.1f%%",
                    rep + 1,
                    effective_reps,
                    parsed["throughput_rps"],
                    parsed["error_pct"],
                    parsed["hit_rate"]["percent"] if parsed["hit_rate"] else 0,
                )

                # Client CPU telemetry (approximate -- spans cachecannon's
                # internal warmup as well as the measurement window)
                if (
                    self._client_allocated_cores
                    and client_cpu_s0 is not None
                    and client_cpu_s1 is not None
                    and client_cpu_t1 is not None
                    and client_cpu_t1 > client_cpu_t0
                ):
                    cores_busy = (client_cpu_s1 - client_cpu_s0) / (client_cpu_t1 - client_cpu_t0)
                    self._client_cores_busy_per_rep.append(cores_busy)

                # Adaptive stop: once the 95% CI half-width is inside the
                # precision target there is nothing to gain from more reps, and
                # each one costs a full server restart plus a 3M-key prefill.
                if _should_stop_adaptive(per_run_rps, rep, self.repetitions, self.target_cv):
                    mean_rps, ci_95 = _compute_aggregated_stats(per_run_rps)
                    self.logger.info(
                        "Adaptive stop after %d reps: 95%% CI half-width %.3f%% <= target %.3f%%",
                        rep + 1,
                        (ci_95 / mean_rps) * 100 if mean_rps else 0.0,
                        self.target_cv,
                    )
                    self.status.steps_total = (self.warmup + self.duration) * (rep + 1)
                    break

            # Record aggregated results
            if server is None:
                raise RuntimeError("No server available for recording results")
            await self._record_result(server, per_run_rps, all_results, toml_content)

            # Final status
            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = self.status.steps_total
            self.file_protocol.write_status(self.status)

        finally:
            await replication_group.stop_all_servers()
            if benchmark_alloc_tag and client:
                client._cpu_allocator.release(client.ip, benchmark_alloc_tag)

    async def _record_result(
        self,
        server: "Server",
        per_run_rps: list[float],
        all_results: list[dict],
        toml_content: str,
    ):
        """Record the final benchmark result."""
        completion_time = datetime.datetime.now()
        lscpu_output, _ = await server.run_host_command("lscpu")

        # Compute aggregated stats
        if len(per_run_rps) >= 2:
            mean_rps, ci_95 = _compute_aggregated_stats(per_run_rps)
        else:
            mean_rps = per_run_rps[0] if per_run_rps else 0
            ci_95 = 0.0

        # Build detailed data
        detailed_data = {
            "warmup": self.warmup,
            "duration": self.duration,
            "io-threads": self.io_threads,
            "pipeline": self.pipelining,
            "connections": self.connections,
            "threads": self.threads,
            "size": self.val_size,
            "keyspace_count": self.keyspace_count,
            "set_ratio": self.set_ratio,
            "distribution": self.distribution,
            "cachecannon_binary": self.cachecannon_binary,
            "toml_config": toml_content,
            "lscpu": lscpu_output,
            "server_cpus": server.server_cpus,
            "repetitions": self.repetitions,
            "per_run_rps": per_run_rps,
            "mean_rps": mean_rps,
            "ci_95": ci_95,
        }

        # Latency from last rep (most representative after warmup effects)
        if all_results and all_results[-1].get("latency"):
            detailed_data["latency"] = all_results[-1]["latency"]
            for per_command in ("latency_get", "latency_set"):
                if all_results[-1].get(per_command):
                    detailed_data[per_command] = all_results[-1][per_command]

        # Hit rate from last rep
        if all_results and all_results[-1].get("hit_rate"):
            detailed_data["hit_rate"] = all_results[-1]["hit_rate"]

        # Per-rep results for full transparency
        detailed_data["per_rep_results"] = [
            {
                "throughput_rps": r["throughput_rps"],
                "error_pct": r["error_pct"],
                "hit_rate": r.get("hit_rate"),
                "latency": r.get("latency"),
            }
            for r in all_results
        ]

        if self._client_cores_busy_per_rep:
            detailed_data["client_cpu"] = summarize_client_cpu(
                self._client_cores_busy_per_rep, self._client_allocated_cores
            )

        results = BenchmarkResults(
            method=f"cachecannon-{self.workload.replace(' ', '-')}",
            source=self.source,
            specifier=self.specifier,
            commit_hash=self.commit_hash,
            score=mean_rps,
            end_time=completion_time,
            data=detailed_data,
            make_args=self.make_args,
            note=self.note,
        )

        self.file_protocol.write_results(results)
