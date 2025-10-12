"""Throughput benchmark"""

import datetime
import logging
import time
from dataclasses import dataclass
from typing import Optional

import plotext as plt

from src.config import (
    PERF_BENCH_CLIENTS,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
    PROJECT_ROOT,
    VALKEY_BENCHMARK,
    ServerInfo,
)
from src.replication_group import ReplicationGroup
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import (
    BILLION,
    MINUTE,
    HumanByte,
    HumanNumber,
    HumanTime,
    RealtimeCommand,
    calc_percentile_averages,
    dump_task_data,
    record_task_result,
)


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

    def short_description(self) -> str:
        profiling = self.profiling_sample_rate > 0
        return (
            f"{HumanByte.to_human(self.val_size)} {self.test} items for "
            f"{HumanTime.to_human(self.duration * MINUTE)}, {self.io_threads} threads"
            f", {self.pipelining} pipelined"
            f"{', profiling' if profiling else ''}"
        )

    def prepare_task_runner(self, server_infos) -> "PerfTaskRunner":
        """Return the task runner for this task."""
        return PerfTaskRunner(
            f"{self.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{self.test}_perf",
            server_infos,
            self.source,
            self.specifier,
            io_threads=self.io_threads,
            valsize=self.val_size,
            pipelining=self.pipelining,
            test=self.test,
            warmup=self.warmup * MINUTE,
            duration=self.duration * MINUTE,
            preload_keys=self.preload_keys,
            has_expire=self.has_expire,
            sample_rate=self.profiling_sample_rate,
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
            name="zrank", preload_command="-t zadd", test_command=" -- ZRANK myzset element:__rand_int__"
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
        sample_rate: int = -1,
    ):

        self.logger = logging.getLogger(self.__class__.__name__ + "." + test)

        self.title = (
            f"{test} throughput, {binary_source}:{specifier}, io-threads={io_threads}, "
            f"pipelining={pipelining}, size={HumanByte.to_human(valsize)}, "
            f"warmup={HumanTime.to_human(warmup)}, "
            f"duration={HumanTime.to_human(duration)}"
        )

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

        self.profiling_thread = None
        self.profiling = self.sample_rate > 0
        if self.profiling:
            self.title += f", profiling={self.sample_rate}Hz"

        # statistics
        self.next_metric_update = time.monotonic()  # now
        self.avg_rps = -1.0
        self.rps_data: list[float] = []
        self.lat_data: list[float] = []

        self.commit_hash = ""

    async def __read_updates(self, command: RealtimeCommand):
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
            avg_msec = float(line.split("avg_msec=")[1].split()[0])
            self.rps_data.append(rps)
            self.lat_data.append(avg_msec)

            line, _ = command.poll_output()

    def __update_graph(self, command: RealtimeCommand):
        graph_update_interval = 10.0
        now = time.monotonic()
        if now > self.next_metric_update or not command.is_running():
            self.next_metric_update = now + graph_update_interval
            if len(self.rps_data) == 0:
                return

            # update metrics
            self.avg_rps = sum(self.rps_data) / len(self.rps_data)

            # update graph
            plt.clear_terminal()
            plt.clear_figure()
            plt.canvas_color("black")
            plt.axes_color("black")
            plt.ticks_color("orange")

            plt.scatter(self.rps_data, marker="braille", color="orange+")
            plt.horizontal_line(self.avg_rps, color="white")
            plt.title(self.title)
            plt.frame(False)

            xlabel_intervals = 4
            point_count = self.duration * 4
            xticks = range(0, point_count + 1, point_count // xlabel_intervals)
            xticks_labels = [f"{HumanTime.to_human(i//4)}" for i in xticks]
            plt.xticks(xticks, xticks_labels)
            plt.xlim(left=0, right=point_count)

            ylabel_intervals = 8
            rps_min = min(self.rps_data)
            rps_max = max(self.rps_data) + 999
            inverval = int(rps_max - rps_min) // ylabel_intervals
            yticks = range(int(rps_min), int(rps_max) + inverval, inverval)
            ytick_labels = [f"{HumanNumber.to_human(tick, 3)}" for tick in yticks]
            plt.yticks(yticks, ytick_labels)

            plt.show()

    def __record_result(self):
        completion_time = datetime.datetime.now()
        name = f"perf-{self.test.name}"

        avg_rps = calc_percentile_averages(self.rps_data, (100, 99, 95))
        avg_lat = calc_percentile_averages(self.lat_data, (100, 99, 95), lowest_vals=True)

        result = {
            "warmup": self.warmup,
            "duration": self.duration,
            "io-threads": self.io_threads,
            "pipeline": self.pipelining,
            "has_expire": self.has_expire,
            "size": self.valsize,
            "preload_keys": self.preload_keys,
            "rps": avg_rps,
            "latency": avg_lat,
        }
        record_task_result(
            name,
            self.binary_source,
            self.specifier,
            self.commit_hash,
            avg_rps[0],
            completion_time,
            result,
        )

        dump_data = {
            "rps_data": self.rps_data,
        }
        dump_task_data(name, self.commit_hash, completion_time, dump_data)

    async def run(self):
        """Run the benchmark."""
        benchmark_update_interval = 0.1  # s

        # setup
        print("preparing:", self.title)

        replication_group = ReplicationGroup(
            self.server_infos, self.binary_source, self.specifier, self.io_threads
        )
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

        print("started rt cmd")
        while command.is_running():
            await self.__read_updates(command)
            self.__update_graph(command)
            time.sleep(benchmark_update_interval)
            now = time.monotonic()
            if now > end_time:
                if self.profiling:
                    await server.profiling_stop()
                command.kill()
            elif warming_up and now >= test_start_time:
                self.rps_data = []
                self.lat_data = []
                warming_up = False
                if self.profiling:
                    server.profiling_start(self.sample_rate)  # TODO port thread to async

        await self.__read_updates(command)
        self.__record_result()
        if self.profiling:
            await server.profiling_report(self.task_name, "primary")
