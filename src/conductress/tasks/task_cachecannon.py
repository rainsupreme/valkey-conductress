"""Cachecannon benchmark task -- second-opinion generator instrument.

cachecannon is a Rust load generator using io_uring (ringline framework).
Its workload definition differs from valkey-benchmark -- results are a SEPARATE
series, never sweep-comparable. Use for generator-wall cross-checks and
absolute ceiling validation.

Binary: /home/ec2-user/cachecannon/target/release/cachecannon (on bench hosts).
Config: TOML file generated per-run in the result directory.
"""

import datetime
import logging
import re
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


def parse_throughput(text: str) -> float:
    """Parse a throughput string like '1.7M req/s', '850K req/s', '1234567 req/s'.

    Returns requests per second as a float.
    """
    match = re.search(r"([\d.]+)\s*([MKk])?\s*req/s", text)
    if not match:
        raise ValueError(f"Cannot parse throughput from: {text!r}")
    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "M":
        return value * 1_000_000
    elif suffix in ("K", "k"):
        return value * 1_000
    return value


def parse_error_rate(text: str) -> float:
    """Parse error percentage from line like 'throughput   1.7M req/s, 0.00% errors'.

    Returns the error percentage as a float (e.g. 0.0, 1.5).
    """
    match = re.search(r"([\d.]+)%\s*errors", text)
    if not match:
        raise ValueError(f"Cannot parse error rate from: {text!r}")
    return float(match.group(1))


def parse_hit_rate(text: str) -> dict:
    """Parse hit rate line like 'hit rate     100% (50.4M hit, 0 miss)'.

    Returns dict with 'percent', 'hits', 'misses'.
    """
    match = re.search(r"([\d.]+)%\s*\(([\d.]+)([MKk])?\s*hit,\s*([\d.]+)([MKk])?\s*miss\)", text)
    if not match:
        raise ValueError(f"Cannot parse hit rate from: {text!r}")
    percent = float(match.group(1))

    def _parse_count(val_str, suffix):
        val = float(val_str)
        if suffix == "M":
            return val * 1_000_000
        elif suffix in ("K", "k"):
            return val * 1_000
        return val

    hits = _parse_count(match.group(2), match.group(3))
    misses = _parse_count(match.group(4), match.group(5))
    return {"percent": percent, "hits": hits, "misses": misses}


def parse_latency_row(text: str) -> dict:
    """Parse a latency percentile row like:
    'GET          7.24 ms   7.34 ms   8.12 ms   9.45 ms   12.3 ms   15.6 ms'

    Returns dict with keys: command, p50, p90, p99, p999, p9999, max (all in ms).
    """
    # The row format is: COMMAND  p50  p90  p99  p999  p9999  max (all in ms or us)
    parts = text.split()
    if len(parts) < 7:
        raise ValueError(f"Cannot parse latency row (too few fields): {text!r}")

    command = parts[0]
    # Extract numeric values - skip 'ms' or 'us' unit tokens
    values = []
    for part in parts[1:]:
        try:
            values.append(float(part))
        except ValueError:
            continue  # skip unit labels like 'ms', 'us'

    if len(values) < 6:
        raise ValueError(f"Cannot parse latency row (found {len(values)} numeric values, need 6): {text!r}")

    return {
        "command": command,
        "p50_ms": values[0],
        "p90_ms": values[1],
        "p99_ms": values[2],
        "p999_ms": values[3],
        "p9999_ms": values[4],
        "max_ms": values[5],
    }


def parse_results_block(output: str) -> dict:
    """Parse the final RESULTS block from cachecannon output.

    Expects a block starting with 'RESULTS (Xs)' containing throughput, hit rate,
    and latency lines.

    Returns dict with keys: throughput_rps, error_pct, hit_rate, latency.
    """
    lines = output.splitlines()
    # Find the RESULTS block
    results_start = None
    for i, line in enumerate(lines):
        if re.match(r"RESULTS\s*\(\d+", line):
            results_start = i
            break

    if results_start is None:
        raise ValueError("No RESULTS block found in cachecannon output")

    # Parse lines after the RESULTS header
    throughput_rps = None
    error_pct = None
    hit_rate = None
    latency = None

    for line in lines[results_start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "req/s" in stripped:
            throughput_rps = parse_throughput(stripped)
            error_pct = parse_error_rate(stripped)
        elif "hit rate" in stripped.lower():
            hit_rate = parse_hit_rate(stripped)
        elif stripped and stripped.split()[0].isupper() and "ms" in stripped:
            # Latency row (command name in uppercase followed by ms values)
            try:
                latency = parse_latency_row(stripped)
            except ValueError:
                pass

    if throughput_rps is None:
        raise ValueError("Could not parse throughput from RESULTS block")
    if error_pct is None:
        raise ValueError("Could not parse error rate from RESULTS block")

    return {
        "throughput_rps": throughput_rps,
        "error_pct": error_pct,
        "hit_rate": hit_rate,
        "latency": latency,
    }


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
        test: Command to bench ('get' for now).

    Returns:
        TOML configuration string.
    """
    # Map test name to cachecannon command weights
    if test == "get":
        commands_section = "get = 100"
    elif test == "set":
        commands_section = "set = 100"
    else:
        commands_section = f"{test} = 100"

    toml = f"""[general]
duration = {duration}
warmup = {warmup}
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
distribution = "uniform"

[workload.commands]
{commands_section}

[workload.values]
length = {val_size}

[timestamps]
userspace = true
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

    def __post_init__(self):
        super().__post_init__()
        self.warmup = int(self.warmup)
        self.duration = int(self.duration)

    def short_description(self) -> str:
        return (
            f"cachecannon {self.test}, {HumanByte.to_human(self.val_size)} values, "
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
            keyspace_count=self.keyspace_count,
            cachecannon_binary=self.cachecannon_binary,
            server_args=self.server_args,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
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
        self.keyspace_count = keyspace_count
        self.cachecannon_binary = cachecannon_binary
        self.server_args = server_args
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override
        self.note = note

        self.commit_hash = ""
        self._client_cores_busy_per_rep: list[float] = []
        self._client_allocated_cores: Optional[int] = None

        self.title = (
            f"cachecannon {test}, {source}:{specifier}, io-threads={io_threads}, "
            f"P{pipelining}, {connections}c, {threads}t, "
            f"{HumanTime.to_human(duration)} x{repetitions}"
        )

        # Status tracking
        self.status = BenchmarkStatus(
            steps_total=(warmup + duration) * repetitions,
            task_type=f"cachecannon-{test}",
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
            for rep in range(self.repetitions):
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
                )

                # Write TOML to result directory
                toml_path = str(self.file_protocol.work_dir / f"cachecannon_rep{rep+1}.toml")
                with open(toml_path, "w") as f:
                    f.write(toml_content)

                # Build and execute cachecannon command
                command_string = self._build_command(toml_path, cpu_list)
                self.logger.info("Starting cachecannon (rep %d/%d): %s", rep + 1, self.repetitions, command_string)

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

                # Parse results
                full_output = "\n".join(output_lines)
                parsed = parse_results_block(full_output)

                # Fail loudly on errors or low hit rate
                if parsed["error_pct"] > 0:
                    raise RuntimeError(
                        f"cachecannon reported {parsed['error_pct']}% errors "
                        f"(threshold: 0%). Output:\n{full_output[-1000:]}"
                    )
                if parsed["hit_rate"] and parsed["hit_rate"]["percent"] < 99.0:
                    raise RuntimeError(
                        f"cachecannon hit rate {parsed['hit_rate']['percent']}% "
                        f"(threshold: 99%). Prefill may have failed."
                    )

                per_run_rps.append(parsed["throughput_rps"])
                all_results.append(parsed)
                self.logger.info(
                    "Rep %d/%d: %.0f rps, %.2f%% errors, hit rate %.1f%%",
                    rep + 1,
                    self.repetitions,
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
            method=f"cachecannon-{self.test}",
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
