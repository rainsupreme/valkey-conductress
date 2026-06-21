"""Throughput benchmark"""

import datetime
import logging
import time
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import List, Optional, Sequence, Union

from scipy.stats import t as t_dist

from conductress.base_task_visualizer import PlotTaskVisualizer
from conductress.config import (
    BENCHMARK_MAX_ITERATIONS,
    BENCHMARK_UPDATE_INTERVAL,
    HEARTBEAT_INTERVAL,
    PERF_BENCH_CLIENTS,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
    PROJECT_ROOT,
    VALKEY_BENCHMARK,
    ServerInfo,
)
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus, FileProtocol, MetricData
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.utility import HumanByte, HumanNumber, HumanTime, RealtimeCommand

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


def should_stop_adaptive(per_run_rps: list[float], rep: int, min_reps: int, target_cv: float) -> bool:
    """Return True if adaptive precision target is met and we can stop early.

    Args:
        per_run_rps: RPS values collected so far.
        rep: Current repetition index (0-based).
        min_reps: Minimum number of reps before early exit is allowed.
        target_cv: Target CV% threshold. 0 = disabled.
    """
    if target_cv <= 0 or rep < min_reps - 1 or len(per_run_rps) < 2:
        return False
    cv = stdev(per_run_rps) / mean(per_run_rps) * 100
    return cv <= target_cv


@dataclass
class PerfTaskData(BaseTaskData):
    """data class for performance benchmark task"""

    test: str
    val_size: int
    io_threads: int
    pipelining: int
    warmup: int
    duration: int
    perf_stat_enabled: bool
    has_expire: bool
    preload_keys: bool
    key_size: int = 0  # target key size in bytes, 0 = standard keys
    repetitions: int = 1  # number of independent benchmark runs (min reps in adaptive mode)
    max_reps: int = 0  # 0 = fixed reps; >0 = adaptive mode upper limit
    target_cv: float = 0.0  # adaptive: stop early when CV% <= this; 0 = disabled
    sweep_commit: str = ""  # non-empty marks this as a sweep task

    def __post_init__(self):
        super().__post_init__()
        self.warmup = int(self.warmup)
        self.duration = int(self.duration)

    def short_description(self) -> str:
        return (
            f"{HumanByte.to_human(self.val_size)} {self.test} items for "
            f"{HumanTime.to_human(self.duration)}, {self.io_threads} threads"
            f", {self.pipelining} pipelined"
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
            perf_stat_enabled=self.perf_stat_enabled,
            note=self.note,
            key_size=self.key_size,
            repetitions=self.repetitions,
            max_reps=self.max_reps,
            target_cv=self.target_cv,
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
        perf_stat_enabled: bool = False,
        note: str = "",
        key_size: int = 0,
        repetitions: int = 1,
        max_reps: int = 0,
        target_cv: float = 0.0,
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
        self.note = note
        self.make_args = make_args
        self.key_size = key_size
        self.repetitions = repetitions
        self.max_reps = max_reps
        self.target_cv = target_cv

        self.perf_stat_enabled = perf_stat_enabled
        self._is_last_rep = False
        self._cpu_stacks_main: list[list] = []
        self._cpu_stacks_io: list[list] = []

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

    def _store_perf_counters(self, detailed_data: dict, perf_counters: dict) -> None:
        """Write perf counters into detailed_data, splitting per-thread buckets.

        ``perf_counters`` is the bucketed structure produced by the collection path:
        ``{"all": {...}, "main": {...}, "io": {...}}``. The process-wide total goes
        to ``perf_counters`` (preserving the historical key/shape so existing export
        and dashboard code is unaffected); the per-thread groups go to
        ``perf_counters_main`` / ``perf_counters_io`` when non-empty. A legacy flat
        dict (no bucket keys) is treated as the process-wide total.
        """
        if any(k in perf_counters for k in ("all", "main", "io")):
            all_counters = perf_counters.get("all") or {}
            main_counters = perf_counters.get("main") or {}
            io_counters = perf_counters.get("io") or {}
        else:
            all_counters, main_counters, io_counters = perf_counters, {}, {}

        if all_counters:
            detailed_data["perf_counters"] = all_counters
            detailed_data["perf_duration_seconds"] = float(self.duration)
        if main_counters:
            detailed_data["perf_counters_main"] = main_counters
        if io_counters:
            detailed_data["perf_counters_io"] = io_counters

    async def __record_result(
        self, server, per_run_rps: Optional[list[float]] = None, perf_counters: Optional[dict] = None
    ):
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
                "perf_stat_enabled": self.perf_stat_enabled,
                "lscpu": lscpu_output,
                "server_cpus": server.server_cpus,
                "repetitions": self.repetitions,
                "per_run_rps": per_run_rps,
                "mean_rps": mean_rps,
                "ci_95": ci_95,
            }
            if perf_counters:
                self._store_perf_counters(detailed_data, perf_counters)
            if self._cpu_stacks_main:
                detailed_data["cpu_stacks_main"] = self._cpu_stacks_main
                detailed_data["cpu_stacks_io"] = self._cpu_stacks_io

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
                "perf_stat_enabled": self.perf_stat_enabled,
                "avg_rps": avg_rps,
                "lscpu": lscpu_output,
                "server_cpus": server.server_cpus,
            }
            if perf_counters:
                self._store_perf_counters(detailed_data, perf_counters)
            if self._cpu_stacks_main:
                detailed_data["cpu_stacks_main"] = self._cpu_stacks_main
                detailed_data["cpu_stacks_io"] = self._cpu_stacks_io

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

        When repetitions > 1 or adaptive mode is enabled, executes the benchmark
        loop N times sequentially with server restarts between reps.
        Otherwise, runs a single benchmark pass.
        """
        effective_reps = self.max_reps if self.max_reps > 0 else self.repetitions
        if effective_reps > 1:
            total_steps = (self.warmup + self.duration) * effective_reps
            self.status.steps_total = total_steps
            self.logger.info(
                "preparing: %s %s",
                self.title,
                (
                    f"({effective_reps} max repetitions, target CV {self.target_cv}%)"
                    if self.target_cv > 0
                    else f"({self.repetitions} repetitions)"
                ),
            )
        else:
            total_steps = self.warmup + self.duration
            self.logger.info("preparing: %s", self.title)

        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos,
            self.binary_source,
            self.specifier,
            self.io_threads,
            self.make_args,
        )

        benchmark_alloc_tag = None
        client = None
        server = None
        per_run_rps: list[float] = []
        perf_counters: Optional[dict] = None

        try:
            for rep in range(effective_reps):
                # Between-rep housekeeping (skip on first rep)
                if rep > 0:
                    await replication_group.stop_all_servers()
                    # Drop page caches between reps to prevent drift.
                    # Skip on Intel (large monolithic L3 stays warm).
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
                    raise RuntimeError("Replication group failed to start: no primary server available")

                await replication_group.begin_replication()
                await replication_group.wait_for_repl_sync()
                server = replication_group.primary
                self.commit_hash = server.get_build_hash() or ""

                # Preload data
                if self.preload_keys and self.preload_command is not None:
                    await server.run_valkey_command_over_keyspace(
                        PERF_BENCH_KEYSPACE, f"-d {self.valsize} {self.preload_command}"
                    )
                    if self.has_expire:
                        if not self.test.expire_command:
                            self.logger.warning("Expire command not available, skipping expiration")
                        else:
                            await server.run_valkey_command_over_keyspace(PERF_BENCH_KEYSPACE, self.test.expire_command)

                # Setup client CPU allocation (once)
                if client is None:
                    client = Server("127.0.0.1")
                    await client.ensure_host_cpu_allocation()
                    benchmark_alloc_tag = self._allocate_benchmark_cpus(client, server)

                # Build and execute benchmark command
                command_string = self._build_benchmark_command(client, server.ip, benchmark_alloc_tag)
                self._is_last_rep = rep == effective_reps - 1
                avg_rps = await self._execute_benchmark_loop(command_string, server, rep, effective_reps)
                per_run_rps.append(avg_rps)
                self.logger.info("Repetition %d/%d avg RPS: %.1f", rep + 1, effective_reps, avg_rps)

                # Update last-rep flag after we have this rep's data
                if not self._is_last_rep:
                    self._is_last_rep = should_stop_adaptive(per_run_rps, rep, self.repetitions, self.target_cv)

                # Collect profiling reports
                rep_counters = await self._collect_profiling_reports(server)
                if rep_counters:
                    # Sum raw counters across all reps for better statistical robustness.
                    # rep_counters is bucketed: {"all": {...}, "main": {...}, "io": {...}}.
                    if perf_counters is None:
                        perf_counters = rep_counters
                    else:
                        for bucket, events in rep_counters.items():
                            acc = perf_counters.setdefault(bucket, {})
                            for k, v in events.items():
                                acc[k] = acc.get(k, 0) + v

                # Collect CPU profile stacks on last rep
                if self._is_last_rep and self.perf_stat_enabled:
                    try:
                        cpu_main, cpu_io = await server.cpu_profile_collect()
                        if cpu_main:
                            self._cpu_stacks_main = cpu_main
                            self._cpu_stacks_io = cpu_io
                    except Exception as e:
                        self.logger.warning("CPU profile collection failed: %s", e)

                # Adaptive early exit
                if should_stop_adaptive(per_run_rps, rep, self.repetitions, self.target_cv):
                    self.logger.info(
                        "Target CV reached: %.2f%% <= %.2f%% after %d reps",
                        stdev(per_run_rps) / mean(per_run_rps) * 100,
                        self.target_cv,
                        rep + 1,
                    )
                    break

            # Record results
            if server is None:
                raise RuntimeError("No server available for recording results")
            if effective_reps > 1:
                await self.__record_result(server, per_run_rps=per_run_rps, perf_counters=perf_counters)
            else:
                await self.__record_result(server, perf_counters=perf_counters)

            # Write final status
            self.status.state = "completed"
            self.status.end_time = time.time()
            self.status.steps_completed = total_steps
            self.file_protocol.write_status(self.status)

        finally:
            await replication_group.stop_all_servers()
            if benchmark_alloc_tag and client:
                client._cpu_allocator.release(client.ip, benchmark_alloc_tag)

    def _allocate_benchmark_cpus(self, client: "Server", server: "Server") -> Optional[AllocationTag]:
        """Allocate CPUs for the benchmark client. Returns the tag or None."""
        target_ip = server.ip
        if not self._is_local_benchmark(target_ip):
            return None

        self.logger.info("Local benchmark detected - optimizing CPU allocation")
        server_tag = AllocationTag(task_id=f"server_{server.ip}_{server.port}", purpose="server")
        platform = getattr(server, "_platform_info", None)
        is_chiplet = platform is not None and platform.needs_single_cache_pinning
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
        self.logger.info(
            "Allocated CPUs %s for benchmark client (NUMA node %d)",
            benchmark_cpus,
            net_numa,
        )
        return benchmark_alloc_tag

    def _build_benchmark_command(
        self,
        client: "Server",
        target_ip: str,
        benchmark_alloc_tag: Optional[AllocationTag],
    ) -> str:
        """Build the numactl + valkey-benchmark command string."""
        net_numa = client._cpu_allocator.get_net_interface_numa(client.ip)

        if benchmark_alloc_tag and self._is_local_benchmark(target_ip):
            allocated = client._cpu_allocator.get_allocation(client.ip, benchmark_alloc_tag)
            benchmark_cpu_list = ",".join(map(str, allocated)) if allocated else ""
            return (
                f"numactl --physcpubind={benchmark_cpu_list} --membind={net_numa} "
                f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                f"--threads {PERF_BENCH_THREADS} -q -l -n {BENCHMARK_MAX_ITERATIONS} {self.test_command}"
            )
        else:
            return (
                f"numactl --cpunodebind={net_numa} --membind={net_numa} "
                f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
                f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
                f"--threads {PERF_BENCH_THREADS} -q -l -n {BENCHMARK_MAX_ITERATIONS} {self.test_command}"
            )

    async def _execute_benchmark_loop(self, command_string: str, server: "Server", rep: int, total_reps: int) -> float:
        """Execute one benchmark run (warmup + measurement). Returns avg RPS."""
        benchmark_update_interval = BENCHMARK_UPDATE_INTERVAL
        self.rps_data = []

        command = RealtimeCommand(command_string)
        self.logger.info(
            "Starting realtime command (rep %d/%d): %s",
            rep + 1,
            total_reps,
            command_string,
        )
        command.start()
        start_time = time.monotonic()
        test_start_time = start_time + self.warmup
        end_time = test_start_time + self.duration
        warming_up = True

        self.status.state = "running"
        self.file_protocol.write_status(self.status)

        self.logger.info(f"started rt cmd (rep {rep + 1}/{total_reps})")
        last_heartbeat = time.time()
        while command.is_running():
            await self.__collect_metrics(command)
            time.sleep(benchmark_update_interval)
            now = time.monotonic()

            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                elapsed_total_time = now - start_time
                steps_this_rep = min(int(elapsed_total_time), self.warmup + self.duration)
                self.status.steps_completed = rep * (self.warmup + self.duration) + steps_this_rep
                self.file_protocol.write_status(self.status)
                last_heartbeat = time.time()

            if now > end_time:
                if self.perf_stat_enabled:
                    await server.perf_stat_stop()
                command.kill()
            elif warming_up and now >= test_start_time:
                self.rps_data = []
                warming_up = False
                if self.perf_stat_enabled:
                    await server.perf_stat_start()
                if self.perf_stat_enabled:
                    server.cpu_profile_start(self.duration)

        await self.__collect_metrics(command)

        if len(self.rps_data) == 0:
            raise RuntimeError(f"No results recorded for repetition {rep + 1}")
        return sum(self.rps_data) / len(self.rps_data)

    async def _collect_profiling_reports(self, server: "Server") -> Optional[dict]:
        """Collect perf stat and CPU profile reports. Returns perf counters dict or None."""
        if self.perf_stat_enabled:
            server.perf_stat_wait()
            result_dir = self.file_protocol.get_result_dir()
            return await server.perf_stat_report(result_dir)
        return None

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
