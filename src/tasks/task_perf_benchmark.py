"""Throughput benchmark"""

import datetime
import logging
import time
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import List, Optional, Sequence, Union

from scipy.stats import t as t_dist

from src.base_task_visualizer import PlotTaskVisualizer
from src.config import (
    PERF_BENCH_CLIENTS,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
    PROJECT_ROOT,
    VALKEY_BENCHMARK,
    ServerInfo,
)
from src.file_protocol import (
    BenchmarkResults,
    BenchmarkStatus,
    FileProtocol,
    MetricData,
)
from src.server import Server
from src.replication_group import ReplicationGroup
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import BILLION, HumanByte, HumanNumber, HumanTime, RealtimeCommand
from src.cpu_allocator import AllocationTag


BASE_KEY_PATTERN = "key:__rand_int__"
BASE_KEY_SIZE = len(BASE_KEY_PATTERN)  # 16 bytes


def generate_padded_key(key_size: int) -> str:
    """Generate a padded key pattern that reaches the target byte size.

    Appends deterministic ASCII characters to the base key pattern.
    Returns the base pattern unmodified if key_size <= BASE_KEY_SIZE.
    """
    if key_size <= BASE_KEY_SIZE:
        return BASE_KEY_PATTERN
    padding_needed = key_size - BASE_KEY_SIZE
    padding = "A" * padding_needed
    return BASE_KEY_PATTERN + padding


def compute_aggregated_stats(per_run_rps: list[float]) -> tuple[float, float]:
    """Compute mean RPS and 95% confidence interval from per-run RPS values.

    Args:
        per_run_rps: List of average RPS values from each repetition run.
                     Must contain at least 2 values.

    Returns:
        Tuple of (mean_rps, ci_95) where ci_95 is the half-width of the
        95% confidence interval.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    n = len(per_run_rps)
    if n < 2:
        raise ValueError(f"Need at least 2 values for aggregation, got {n}")
    mean_rps = mean(per_run_rps)
    ci_95 = t_dist.ppf(0.975, n - 1) * (stdev(per_run_rps) / sqrt(n))
    return mean_rps, ci_95


@dataclass
class PerfTaskData(BaseTaskData):
    """data class for performance benchmark task"""

    test: str
    val_size: int
    io_threads: int
    pipelining: int
    warmup: int
    duration: int
    profiling_sample_rate: int
    perf_stat_enabled: bool
    has_expire: bool
    preload_keys: bool
    key_size: int = 0  # target key size in bytes, 0 = standard keys
    repetitions: int = 1  # number of independent benchmark runs
    sweep_commit: str = ""  # non-empty marks this as a sweep task

    def __post_init__(self):
        super().__post_init__()
        self.warmup = int(self.warmup)
        self.duration = int(self.duration)

    def short_description(self) -> str:
        profiling = self.profiling_sample_rate > 0
        return (
            f"{HumanByte.to_human(self.val_size)} {self.test} items for "
            f"{HumanTime.to_human(self.duration)}, {self.io_threads} threads"
            f", {self.pipelining} pipelined"
            f"{', profiling' if profiling else ''}"
            f"{', perf-stat' if self.perf_stat_enabled else ''}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "PerfTaskRunner":
        """Return the task runner for this task."""
        return PerfTaskRunner(
            self.task_id,
            server_infos,
            self.source,
            self.specifier,
            io_threads=self.io_threads,
            valsize=self.val_size,
            pipelining=self.pipelining,
            test=self.test,
            warmup=self.warmup,
            duration=self.duration,
            preload_keys=self.preload_keys,
            has_expire=self.has_expire,
            make_args=self.make_args,
            sample_rate=self.profiling_sample_rate,
            perf_stat_enabled=self.perf_stat_enabled,
            note=self.note,
            key_size=self.key_size,
            repetitions=self.repetitions,
        )


class PerfTaskRunner(BaseTaskRunner):
    """Benchmark the throughput of a Valkey server."""

    @dataclass
    class Test:
        """Defines an available test"""

        name: str
        preload_command: Optional[str]
        test_command: str
        expire_command: Optional[str] = None

    tests: dict[str, Test] = {
        "set": Test(
            name="set",
            preload_command="-t set",
            test_command="-t set",
            expire_command=f"EXPIRE key:__rand_int__ {7*24*60*60}",
        ),
        "get": Test(
            name="get",
            preload_command="-t set",
            test_command="-t get",
            expire_command=f"EXPIRE key:__rand_int__ {7*24*60*60}",
        ),
        "sadd": Test(name="sadd", preload_command="-t sadd", test_command="-t sadd"),
        "hset": Test(name="hset", preload_command="-t hset", test_command="-t hset"),
        "zadd": Test(name="zadd", preload_command="-t zadd", test_command="-t zadd"),
        "zrank": Test(
            name="zrank",
            preload_command="-t zadd",
            test_command=" -- ZRANK myzset element:__rand_int__",
        ),
        "zcount": Test(
            name="zcount",
            preload_command="-t zadd",
            test_command=" -- ZCOUNT myzset __rand_int__ __rand_int__",
        ),
        "zscore": Test(
            name="zscore",
            preload_command="-t zadd",
            test_command=" -- ZSCORE myzset element:__rand_int__",
        ),
        "sismember": Test(
            name="sismember",
            preload_command="-t sadd",
            test_command=" -- SISMEMBER myset element:__rand_int__",
        ),
        "ping": Test(
            name="ping",
            preload_command=None,
            test_command="-t ping",
        ),
        "mget": Test(
            name="mget",
            preload_command="-t set",
            test_command=" -- MGET key:__rand_int__ key:__rand_int__ key:__rand_int__ key:__rand_int__",
        ),
    }

    def __init__(
        self,
        task_name: str,
        server_infos: list[ServerInfo],
        binary_source: str,
        specifier: str,
        io_threads: int,
        valsize: int,
        pipelining: int,
        test: str,
        warmup: int,
        duration: int,
        preload_keys: bool,
        has_expire: bool,
        make_args: str,
        sample_rate: int = -1,
        perf_stat_enabled: bool = False,
        note: str = "",
        key_size: int = 0,
        repetitions: int = 1,
    ):
        super().__init__(task_name)

        self.logger = logging.getLogger(self.__class__.__name__ + "." + test)

        # settings
        self.task_name = task_name
        self.server_infos = server_infos
        self.binary_source = binary_source
        self.specifier = specifier
        self.io_threads = io_threads
        self.valsize = valsize
        self.pipelining = pipelining
        self.test: PerfTaskRunner.Test = PerfTaskRunner.tests[test]
        self.warmup = warmup  # seconds
        self.duration = duration  # seconds
        self.preload_keys = preload_keys
        self.has_expire = has_expire
        self.sample_rate = sample_rate
        self.note = note
        self.make_args = make_args
        self.key_size = key_size
        self.repetitions = repetitions

        self.profiling_thread = None
        self.profiling = self.sample_rate > 0
        self.perf_stat_enabled = perf_stat_enabled

        # Build custom commands when key_size > 0
        if self.key_size > 0:
            padded_key = generate_padded_key(self.key_size)
            preload_custom = self._build_custom_command(self.test, padded_key, is_preload=True)
            test_custom = self._build_custom_command(self.test, padded_key, is_preload=False)
            self.preload_command: Optional[str] = preload_custom
            self.test_command: Optional[str] = test_custom
        else:
            self.preload_command = self.test.preload_command
            self.test_command = self.test.test_command

        self.title = (
            f"{test} throughput, {binary_source}:{specifier}, io-threads={io_threads}, "
            f"pipelining={pipelining}, size={HumanByte.to_human(valsize)}, "
            f"warmup={HumanTime.to_human(warmup)}, "
            f"duration={HumanTime.to_human(duration)}"
        )
        if self.key_size > 0:
            self.title += f", key-size={HumanByte.to_human(self.key_size)}"
        if self.profiling:
            self.title += f", profiling={self.sample_rate}Hz"
        if self.perf_stat_enabled:
            self.title += ", perf-stat"

        # statistics
        self.rps_data: list[float] = []

        self.commit_hash = ""

        # Initialize status
        self.status = BenchmarkStatus(steps_total=self.warmup + self.duration, task_type=f"perf-{test}")

    def _build_custom_command(self, test: "PerfTaskRunner.Test", padded_key: str, is_preload: bool) -> Optional[str]:
        """Build a custom command string for the given test type using the padded key.

        Returns the custom command string, or None if the test has no preload
        and is_preload is True.
        """
        name = test.name

        if is_preload:
            preload_map: dict[str, Optional[str]] = {
                "set": f" -- SET {padded_key} __rand_field__",
                "get": f" -- SET {padded_key} __rand_field__",
                "sadd": f" -- SADD {padded_key} element:__rand_int__",
                "sismember": f" -- SADD {padded_key} element:__rand_int__",
                "hset": f" -- HSET {padded_key} field:__rand_int__ __rand_field__",
                "zadd": f" -- ZADD {padded_key} __rand_int__ element:__rand_int__",
                "mget": f" -- SET {padded_key} __rand_field__",
                "ping": None,
                "zrank": f" -- ZADD {padded_key} __rand_int__ element:__rand_int__",
                "zcount": f" -- ZADD {padded_key} __rand_int__ element:__rand_int__",
                "zscore": f" -- ZADD {padded_key} __rand_int__ element:__rand_int__",
            }
            return preload_map[name]
        else:
            test_map: dict[str, str] = {
                "set": f" -- SET {padded_key} __rand_field__",
                "get": f" -- GET {padded_key}",
                "sadd": f" -- SADD {padded_key} element:__rand_int__",
                "sismember": f" -- SISMEMBER {padded_key} element:__rand_int__",
                "hset": f" -- HSET {padded_key} field:__rand_int__ __rand_field__",
                "zadd": f" -- ZADD {padded_key} __rand_int__ element:__rand_int__",
                "mget": f" -- MGET {padded_key} {padded_key} {padded_key} {padded_key}",
                "ping": "-t ping",
                "zrank": f" -- ZRANK {padded_key} element:__rand_int__",
                "zcount": f" -- ZCOUNT {padded_key} __rand_int__ __rand_int__",
                "zscore": f" -- ZSCORE {padded_key} element:__rand_int__",
            }
            return test_map[name]

    async def __collect_metrics(self, command: RealtimeCommand):
        line, _ = command.poll_output()
        while line is not None and line != "" and not line.isspace():
            if "overall" not in line:
                line, _ = command.poll_output()
                continue
            # line looks like this:
            # "GET: rps=140328.0 (overall: 141165.2) avg_msec=0.193 (overall: 0.191)"
            # or this:
            # ZRANK myzset ele__rand_int__: rps=442912.0 (overall: 436252.6) avg_msec=5.868 (overall: 5.948)
            rps = float(line.split("rps=")[1].split()[0])
            self.rps_data.append(rps)

            # Write metric to file protocol
            metric = MetricData(metrics={"rps": rps})
            self.file_protocol.append_metric(metric)

            line, _ = command.poll_output()

    async def __record_result(self, server, per_run_rps: Optional[list[float]] = None):
        completion_time = datetime.datetime.now()

        if len(self.rps_data) == 0 and not per_run_rps:
            raise RuntimeError("No results recorded")

        # Get system information
        lscpu_output, _ = await server.run_host_command("lscpu")

        if per_run_rps is not None and len(per_run_rps) > 1:
            # Aggregated result for repetitions > 1
            mean_rps, ci_95 = compute_aggregated_stats(per_run_rps)

            detailed_data = {
                "warmup": self.warmup,
                "duration": self.duration,
                "io-threads": self.io_threads,
                "pipeline": self.pipelining,
                "has_expire": self.has_expire,
                "size": self.valsize,
                "key_size": self.key_size,
                "preload_keys": self.preload_keys,
                "profiling_enabled": self.profiling,
                "perf_stat_enabled": self.perf_stat_enabled,
                "lscpu": lscpu_output,
                "server_cpus": server.server_cpus,
                "repetitions": self.repetitions,
                "per_run_rps": per_run_rps,
                "mean_rps": mean_rps,
                "ci_95": ci_95,
            }

            results = BenchmarkResults(
                method=f"perf-{self.test.name}",
                source=self.binary_source,
                specifier=self.specifier,
                commit_hash=self.commit_hash,
                score=mean_rps,
                end_time=completion_time,
                data=detailed_data,
                make_args=self.make_args,
                note=self.note,
            )
        else:
            # Single-run result (repetitions == 1 or legacy behavior)
            avg_rps = sum(self.rps_data) / len(self.rps_data)

            detailed_data = {
                "warmup": self.warmup,
                "duration": self.duration,
                "io-threads": self.io_threads,
                "pipeline": self.pipelining,
                "has_expire": self.has_expire,
                "size": self.valsize,
                "key_size": self.key_size,
                "preload_keys": self.preload_keys,
                "profiling_enabled": self.profiling,
                "perf_stat_enabled": self.perf_stat_enabled,
                "avg_rps": avg_rps,
                "lscpu": lscpu_output,
                "server_cpus": server.server_cpus,
            }

            results = BenchmarkResults(
                method=f"perf-{self.test.name}",
                source=self.binary_source,
                specifier=self.specifier,
                commit_hash=self.commit_hash,
                score=avg_rps,
                end_time=completion_time,
                data=detailed_data,
                make_args=self.make_args,
                note=self.note,
            )

        self.file_protocol.write_results(results)

    async def run(self):
        """Run the benchmark.

        When repetitions > 1, executes the benchmark loop N times sequentially.
        Between each repetition: stop the replication group, restart it,
        re-preload data, run the benchmark, and collect avg RPS.
        After all runs, compute aggregated statistics and write a single result.

        When repetitions == 1, preserves existing single-run behavior unchanged.
        """
        if self.repetitions > 1:
            await self._run_with_repetitions()
        else:
            await self._run_single()

    async def _run_single(self):
        """Run a single benchmark (original behavior for repetitions == 1)."""
        benchmark_update_interval = 0.1  # s

        # setup
        print("preparing:", self.title)

        # Write initial status
        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos, self.binary_source, self.specifier, self.io_threads, self.make_args
        )
        await replication_group.kill_all_valkey_instances()

        await replication_group.start()
        if not replication_group.primary:
            raise RuntimeError("Replication group failed to start: no primary server available")

        benchmark_alloc_tag = None
        client = None
        try:
            await replication_group.begin_replication()
            await replication_group.wait_for_repl_sync()
            server = replication_group.primary
            target_ip = server.ip
            self.commit_hash = server.get_build_hash() or ""
            if self.preload_keys and self.preload_command is not None:
                await server.run_valkey_command_over_keyspace(
                    PERF_BENCH_KEYSPACE, f"-d {self.valsize} {self.preload_command}"
                )
                if self.has_expire:
                    if not self.test.expire_command:
                        self.logger.warning("Expire command not available, skipping expiration")
                    else:
                        await server.run_valkey_command_over_keyspace(
                            PERF_BENCH_KEYSPACE, self.test.expire_command
                        )

            # Setup client CPU allocation and pinning
            client = Server("127.0.0.1")
            await client.ensure_host_cpu_allocation()

            # Detect if this is a local benchmark (server and client on same host)
            is_local_benchmark = self._is_local_benchmark(target_ip)

            if is_local_benchmark:
                self.logger.info("Local benchmark detected - optimizing CPU allocation")

                # Get server allocation tag to avoid its cache
                server_tag = AllocationTag(task_id=f"server_{server.ip}_{server.port}", purpose="server")

                # Allocate CPUs for benchmark client, avoiding server cache
                # On chiplet architectures, minimize cache groups to prevent
                # non-deterministic thread placement across CCDs
                is_chiplet = getattr(server, "_platform_info", None) is not None and \
                    server._platform_info.needs_single_cache_pinning
                benchmark_alloc_tag = AllocationTag(task_id=self.task_name, purpose="benchmark")
                net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
                benchmark_cpus = client._cpu_allocator.allocate(
                    client.ip,
                    benchmark_alloc_tag,
                    count=PERF_BENCH_THREADS,
                    require_numa=net_numa,
                    avoid_tags=[server_tag],
                    prefer_different_cache=True,
                    minimize_cache_groups=is_chiplet,
                )
                benchmark_cpu_list = ",".join(map(str, benchmark_cpus))

                self.logger.info(
                    "Allocated CPUs %s for benchmark client (NUMA node %d)",
                    benchmark_cpus,
                    net_numa,
                )

                # Pin benchmark to allocated CPUs and bind memory to NUMA node
                command_string = (
                    f"numactl --physcpubind={benchmark_cpu_list} --membind={net_numa} "
                    f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                    f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                    f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION} {self.test_command}"
                )
            else:
                # Remote benchmark - use NUMA node binding only
                net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
                command_string = (
                    f"numactl --cpunodebind={net_numa} --membind={net_numa} "
                    f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                    f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                    f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION} {self.test_command}"
                )
            command = RealtimeCommand(command_string)

            # run benchmark
            self.logger.info("Starting realtime command: %s", command_string)
            command.start()
            start_time = time.monotonic()
            test_start_time = start_time + self.warmup
            end_time = test_start_time + self.duration
            warming_up = True

            # Update status to running
            self.status.state = "running"
            self.file_protocol.write_status(self.status)

            print("started rt cmd")
            last_heartbeat = time.time()
            while command.is_running():
                await self.__collect_metrics(command)
                time.sleep(benchmark_update_interval)
                now = time.monotonic()

                # Update heartbeat and progress every 5 seconds
                if time.time() - last_heartbeat > 5.0:
                    # Update progress based on total elapsed time (warmup + test)
                    elapsed_total_time = now - start_time
                    self.status.steps_completed = min(int(elapsed_total_time), self.warmup + self.duration)

                    self.file_protocol.write_status(self.status)
                    last_heartbeat = time.time()

                if now > end_time:
                    if self.profiling:
                        await server.profiling_stop()
                    if self.perf_stat_enabled:
                        await server.perf_stat_stop()
                    command.kill()
                elif warming_up and now >= test_start_time:
                    self.rps_data = []
                    warming_up = False
                    if self.profiling:
                        server.profiling_start(self.sample_rate)
                    if self.perf_stat_enabled:
                        await server.perf_stat_start()

            await self.__collect_metrics(command)
            await self.__record_result(server)

            # Write final status
            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = self.warmup + self.duration  # 100% complete
            self.file_protocol.write_status(self.status)

            if self.profiling:
                server.profiling_wait()
                result_dir = self.file_protocol.get_result_dir()
                await server.profiling_report(result_dir)

            if self.perf_stat_enabled:
                server.perf_stat_wait()
                result_dir = self.file_protocol.get_result_dir()
                await server.perf_stat_report(result_dir)
        finally:
            # Clean up all servers and release CPUs
            await replication_group.stop_all_servers()

            # Release benchmark client CPUs if allocated
            if benchmark_alloc_tag and client:
                client._cpu_allocator.release(client.ip, benchmark_alloc_tag)

    async def _run_with_repetitions(self):
        """Run the benchmark multiple times and aggregate results.

        Executes the benchmark loop N times sequentially. Between each
        repetition: stop the replication group, restart it, re-preload
        data, run the benchmark, and collect avg RPS per run.
        After all runs, compute mean RPS and 95% CI and write a single
        aggregated result.
        """
        benchmark_update_interval = 0.1  # s
        per_run_rps: list[float] = []

        # setup
        print("preparing:", self.title, f"({self.repetitions} repetitions)")

        # Update status for total steps across all repetitions
        total_steps = (self.warmup + self.duration) * self.repetitions
        self.status.steps_total = total_steps
        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos, self.binary_source, self.specifier, self.io_threads, self.make_args
        )

        benchmark_alloc_tag = None
        client = None
        server = None

        try:
            for rep in range(self.repetitions):
                self.logger.info("Starting repetition %d/%d", rep + 1, self.repetitions)

                # Kill any existing instances and start fresh
                await replication_group.kill_all_valkey_instances()

                # Drop page caches between reps to prevent accumulated memory state
                # from causing drift. Skip on Intel (large monolithic L3 stays warm).
                if rep > 0:
                    primary_server = replication_group.primary or Server(self.server_infos[0].ip)
                    platform = getattr(primary_server, "_platform_info", None)
                    if platform is None or platform.needs_drop_caches:
                        await primary_server.run_host_command(
                            "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False
                        )

                await replication_group.start()
                if not replication_group.primary:
                    raise RuntimeError("Replication group failed to start: no primary server available")

                await replication_group.begin_replication()
                await replication_group.wait_for_repl_sync()
                server = replication_group.primary
                target_ip = server.ip
                self.commit_hash = server.get_build_hash() or ""

                is_chiplet = getattr(server, "_platform_info", None) is not None and \
                    server._platform_info.needs_single_cache_pinning

                # Preload data
                if self.preload_keys and self.preload_command is not None:
                    await server.run_valkey_command_over_keyspace(
                        PERF_BENCH_KEYSPACE, f"-d {self.valsize} {self.preload_command}"
                    )
                    if self.has_expire:
                        if not self.test.expire_command:
                            self.logger.warning("Expire command not available, skipping expiration")
                        else:
                            await server.run_valkey_command_over_keyspace(
                                PERF_BENCH_KEYSPACE, self.test.expire_command
                            )

                # Setup client CPU allocation (only on first repetition)
                if client is None:
                    client = Server("127.0.0.1")
                    await client.ensure_host_cpu_allocation()

                is_local_benchmark = self._is_local_benchmark(target_ip)

                if is_local_benchmark:
                    if benchmark_alloc_tag is None:
                        self.logger.info("Local benchmark detected - optimizing CPU allocation")
                        server_tag = AllocationTag(task_id=f"server_{server.ip}_{server.port}", purpose="server")
                        benchmark_alloc_tag = AllocationTag(task_id=self.task_name, purpose="benchmark")
                        net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
                        benchmark_cpus = client._cpu_allocator.allocate(
                            client.ip,
                            benchmark_alloc_tag,
                            count=PERF_BENCH_THREADS,
                            require_numa=net_numa,
                            avoid_tags=[server_tag],
                            prefer_different_cache=True,
                            minimize_cache_groups=is_chiplet,
                        )
                        benchmark_cpu_list = ",".join(map(str, benchmark_cpus))
                        self.logger.info(
                            "Allocated CPUs %s for benchmark client (NUMA node %d)",
                            benchmark_cpus,
                            net_numa,
                        )
                    else:
                        net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
                        allocated = client._cpu_allocator.get_allocation(client.ip, benchmark_alloc_tag)
                        benchmark_cpu_list = ",".join(map(str, allocated)) if allocated else ""

                    command_string = (
                        f"numactl --physcpubind={benchmark_cpu_list} --membind={net_numa} "
                        f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                        f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                        f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION} {self.test_command}"
                    )
                else:
                    net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)
                    command_string = (
                        f"numactl --cpunodebind={net_numa} --membind={net_numa} "
                        f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                        f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                        f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION} {self.test_command}"
                    )

                # Reset per-run metrics
                self.rps_data = []

                command = RealtimeCommand(command_string)
                self.logger.info("Starting realtime command (rep %d/%d): %s", rep + 1, self.repetitions, command_string)
                command.start()
                start_time = time.monotonic()
                test_start_time = start_time + self.warmup
                end_time = test_start_time + self.duration
                warming_up = True

                # Update status to running
                self.status.state = "running"
                self.file_protocol.write_status(self.status)

                print(f"started rt cmd (rep {rep + 1}/{self.repetitions})")
                last_heartbeat = time.time()
                while command.is_running():
                    await self.__collect_metrics(command)
                    time.sleep(benchmark_update_interval)
                    now = time.monotonic()

                    # Update heartbeat and progress every 5 seconds
                    if time.time() - last_heartbeat > 5.0:
                        elapsed_total_time = now - start_time
                        steps_this_rep = min(int(elapsed_total_time), self.warmup + self.duration)
                        self.status.steps_completed = rep * (self.warmup + self.duration) + steps_this_rep
                        self.file_protocol.write_status(self.status)
                        last_heartbeat = time.time()

                    if now > end_time:
                        if self.profiling:
                            await server.profiling_stop()
                        if self.perf_stat_enabled:
                            await server.perf_stat_stop()
                        command.kill()
                    elif warming_up and now >= test_start_time:
                        self.rps_data = []
                        warming_up = False
                        if self.profiling:
                            server.profiling_start(self.sample_rate)
                        if self.perf_stat_enabled:
                            await server.perf_stat_start()

                await self.__collect_metrics(command)

                # Collect avg RPS for this run
                if len(self.rps_data) == 0:
                    raise RuntimeError(f"No results recorded for repetition {rep + 1}")
                run_avg_rps = sum(self.rps_data) / len(self.rps_data)
                per_run_rps.append(run_avg_rps)
                self.logger.info("Repetition %d/%d avg RPS: %.1f", rep + 1, self.repetitions, run_avg_rps)

                # Handle profiling/perf_stat reports per run
                if self.profiling:
                    server.profiling_wait()
                    result_dir = self.file_protocol.get_result_dir()
                    await server.profiling_report(result_dir)

                if self.perf_stat_enabled:
                    server.perf_stat_wait()
                    result_dir = self.file_protocol.get_result_dir()
                    await server.perf_stat_report(result_dir)

                # Stop the replication group between repetitions (except after the last one)
                if rep < self.repetitions - 1:
                    await replication_group.stop_all_servers()

            # All repetitions complete — write aggregated result
            if server is None:
                raise RuntimeError("No server available for recording results")
            await self.__record_result(server, per_run_rps=per_run_rps)

            # Write final status
            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = total_steps
            self.file_protocol.write_status(self.status)

        finally:
            # Clean up all servers and release CPUs
            await replication_group.stop_all_servers()

            # Release benchmark client CPUs if allocated
            if benchmark_alloc_tag and client:
                client._cpu_allocator.release(client.ip, benchmark_alloc_tag)

    def _is_local_benchmark(self, target_ip: str) -> bool:
        """Check if benchmark is running locally (server and client on same host)."""
        # Normalize localhost variations
        local_ips = {"127.0.0.1", "localhost", "::1"}
        return target_ip in local_ips


class PerfTaskVisualizer(PlotTaskVisualizer):
    """Visualizer for performance benchmark tasks."""

    def __init__(self, task_id: str, file_protocol: FileProtocol, *args, **kwargs):
        super().__init__(task_id, *args, **kwargs)
        self.file_protocol = file_protocol

    def format_x_tick(self, value: float) -> str:
        return HumanTime.to_human(value / 4)

    def format_y_tick(self, value: float) -> str:
        return HumanNumber.to_human(value, 3)

    def get_plot_data(self) -> "List[Optional[float]]":
        datapoints = self.file_protocol.read_metrics()
        data = [dp.metrics.get("rps", 0.0) for dp in datapoints]

        if len(data) < 4:
            return data  # type: ignore[return-value]

        sorted_data = sorted(data)
        q1_idx: int = len(sorted_data) // 4
        q3_idx: int = 3 * len(sorted_data) // 4
        q1, q3 = sorted_data[q1_idx], sorted_data[q3_idx]
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr

        return [x if lower <= x <= upper else None for x in data]
