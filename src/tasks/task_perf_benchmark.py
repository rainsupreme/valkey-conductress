"""Throughput benchmark"""

import datetime
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union

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
from src.replication_group import ReplicationGroup
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import BILLION, HumanByte, HumanNumber, HumanTime, RealtimeCommand


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
    has_expire: bool
    preload_keys: bool

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
            note=self.note,
        )


class PerfTaskRunner(BaseTaskRunner):
    """Benchmark the throughput of a Valkey server."""

    @dataclass
    class Test:
        """Defines an available test"""

        name: str
        preload_command: str
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
        note: str = "",
    ):
        super().__init__(task_name)

        self.logger = logging.getLogger(self.__class__.__name__ + "." + test)

        self.title = (
            f"{test} throughput, {binary_source}:{specifier}, io-threads={io_threads}, "
            f"pipelining={pipelining}, size={HumanByte.to_human(valsize)}, "
            f"warmup={HumanTime.to_human(warmup)}, "
            f"duration={HumanTime.to_human(duration)}"
        )

        # settings
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

        self.profiling_thread = None
        self.profiling = self.sample_rate > 0
        if self.profiling:
            self.title += f", profiling={self.sample_rate}Hz"

        # statistics
        self.rps_data: list[float] = []

        self.commit_hash = ""

        # Initialize status
        self.status = BenchmarkStatus(steps_total=self.warmup + self.duration, task_type=f"perf-{test}")

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

    def __record_result(self):
        completion_time = datetime.datetime.now()

        assert len(self.rps_data) > 0, "No results recorded"
        avg_rps = sum(self.rps_data) / len(self.rps_data)

        # Write results to file protocol
        detailed_data = {
            "warmup": self.warmup,
            "duration": self.duration,
            "io-threads": self.io_threads,
            "pipeline": self.pipelining,
            "has_expire": self.has_expire,
            "size": self.valsize,
            "preload_keys": self.preload_keys,
            "avg_rps": avg_rps,
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
        """Run the benchmark."""
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
        assert replication_group.primary
        await replication_group.begin_replication()
        await replication_group.wait_for_repl_sync()
        server = replication_group.primary
        target_ip = server.ip
        self.commit_hash = server.get_build_hash() or ""
        if self.preload_keys:
            await server.run_valkey_command_over_keyspace(
                PERF_BENCH_KEYSPACE, f"-d {self.valsize} {self.test.preload_command}"
            )
            if self.has_expire:
                if not self.test.expire_command:
                    self.logger.warning("Expire command not available, skipping expiration")
                else:
                    await server.run_valkey_command_over_keyspace(
                        PERF_BENCH_KEYSPACE, self.test.expire_command
                    )

        command_string = (
            f"{PROJECT_ROOT / VALKEY_BENCHMARK} -h {target_ip} -d {self.valsize} "
            f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
            f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION} {self.test.test_command}"
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
                command.kill()
            elif warming_up and now >= test_start_time:
                self.rps_data = []
                warming_up = False
                if self.profiling:
                    server.profiling_start(self.sample_rate)

        await self.__collect_metrics(command)
        self.__record_result()

        # Write final status
        self.status.state = "completed"
        self.status.end_time = time.time()
        self.status.steps_completed = self.warmup + self.duration  # 100% complete
        self.file_protocol.write_status(self.status)

        if self.profiling:  # TODO: fix up profiling
            await server.profiling_report(
                self.task_name, "primary"
            )  # TODO: fix up profiling - don't have task name any more

        # Clean up all servers and release CPUs
        await replication_group.stop_all_servers()


class PerfTaskVisualizer(PlotTaskVisualizer):
    """Visualizer for performance benchmark tasks."""

    def __init__(self, task_id: str, file_protocol: FileProtocol, *args, **kwargs):
        super().__init__(task_id, *args, **kwargs)
        self.file_protocol = file_protocol

    def format_x_tick(self, value: float) -> str:
        return HumanTime.to_human(value / 4)

    def format_y_tick(self, value: float) -> str:
        return HumanNumber.to_human(value, 3)

    def get_plot_data(self) -> Sequence[Union[float, None]]:
        datapoints = self.file_protocol.read_metrics()
        data: list[float] = [dp.metrics.get("rps", 0.0) for dp in datapoints]

        if len(data) < 4:
            return data

        sorted_data = sorted(data)
        q1_idx: int = len(sorted_data) // 4
        q3_idx: int = 3 * len(sorted_data) // 4
        q1, q3 = sorted_data[q1_idx], sorted_data[q3_idx]
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr

        return [x if lower <= x <= upper else None for x in data]
