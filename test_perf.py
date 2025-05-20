"""Throughput benchmark"""

import datetime
import logging
import time
from threading import Thread

import plotext as plt

from config import (
    CONDUCTRESS_DATA_DUMP,
    CONDUCTRESS_OUTPUT,
    PERF_BENCH_CLIENTS,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
)
from replication_group import ReplicationGroup
from utility import (
    BILLION,
    RealtimeCommand,
    calc_percentile_averages,
    human,
    human_time,
)

logger = logging.getLogger(__name__)


class TestPerf:
    """Benchmark the throughput of a Valkey server."""

    def __init__(
        self,
        task_name: str,
        server_ips: list,
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
        self.title = (
            f"{test} throughput, {binary_source}:{specifier}, io-threads={io_threads}, "
            f"pipelining={pipelining}, size={valsize}, warmup={human_time(warmup)}, "
            f"duration={human_time(duration)}"
        )

        # settings
        self.task_name = task_name
        self.server_ips = server_ips
        self.binary_source = binary_source
        self.specifier = specifier
        self.io_threads = io_threads
        self.valsize = valsize
        self.pipelining = pipelining
        self.test = test
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

        print("preparing:", self.title)

        self.replication_group = ReplicationGroup(server_ips, binary_source, specifier, io_threads, [])
        self.replication_group.begin_replication()
        self.replication_group.wait_for_repl_sync()

        self.server = self.replication_group.primary
        self.target_ip = self.server.ip
        self.commit_hash = self.server.get_build_hash()
        if self.preload_keys:
            self.server.fill_keyspace(self.valsize, PERF_BENCH_KEYSPACE, self.test)
            if self.has_expire:
                self.server.expire_keyspace(PERF_BENCH_KEYSPACE, self.test)

        self.command_string = (
            f"./valkey-benchmark -h {self.target_ip} -d {self.valsize} "
            f"-r {PERF_BENCH_KEYSPACE} -c {PERF_BENCH_CLIENTS} -P {self.pipelining} "
            f"--threads {PERF_BENCH_THREADS} -q -l -n {2 * BILLION}"
        )
        if self.test == "zrank":
            self.command_string += " -- ZRANK myzset element:__rand_int__"
        else:
            self.command_string += f" -t {self.test}"
        print(repr(self.command_string))
        self.command = RealtimeCommand(self.command_string)

    def __read_updates(self):
        line, _ = self.command.poll_output()
        while line is not None and line != "" and not line.isspace():
            if "overall" not in line:
                line, _ = self.command.poll_output()
                continue
            # line looks like this:
            # "GET: rps=140328.0 (overall: 141165.2) avg_msec=0.193 (overall: 0.191)"
            # or this:
            # ZRANK myzset ele__rand_int__: rps=442912.0 (overall: 436252.6) avg_msec=5.868 (overall: 5.948)
            rps = float(line.split("rps=")[1].split()[0])
            avg_msec = float(line.split("avg_msec=")[1].split()[0])
            self.rps_data.append(rps)
            self.lat_data.append(avg_msec)

            line, _ = self.command.poll_output()

    def __update_graph(self):
        graph_update_interval = 10.0
        now = time.monotonic()
        if now > self.next_metric_update or not self.command.is_running():
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
            xticks_labels = [f"{human_time(i//4)}" for i in xticks]
            plt.xticks(xticks, xticks_labels)
            plt.xlim(left=0, right=point_count)

            ylabel_intervals = 8
            rps_min = min(self.rps_data)
            rps_max = max(self.rps_data) + 999
            inverval = int(rps_max - rps_min) // ylabel_intervals
            yticks = range(int(rps_min), int(rps_max) + inverval, inverval)
            ytick_labels = [f"{human(tick, 0)}" for tick in yticks]
            plt.yticks(yticks, ytick_labels)

            plt.show()

    def __record_result(self):
        logger.debug(
            "%dB %s perf on %s:%s\trps_data=%s\tlat_data=%s",
            self.valsize,
            self.test,
            self.binary_source,
            self.specifier,
            repr(self.rps_data),
            repr(self.lat_data),
        )

        avg_rps = calc_percentile_averages(self.rps_data, (100, 99, 95))
        avg_lat = calc_percentile_averages(self.lat_data, (100, 99, 95), lowest_vals=True)

        result = {
            "method": "perf",
            "source": self.binary_source,
            "specifier": self.specifier,
            "commit_hash": self.commit_hash,
            "test": self.test,
            "warmup": self.warmup,
            "duration": self.duration,
            "endtime": datetime.datetime.now(),
            "io-threads": self.io_threads,
            "pipeline": self.pipelining,
            "has_expire": self.has_expire,
            "size": self.valsize,
            "preload_keys": self.preload_keys,
            "avg_rps": avg_rps[0],
            "avg_99_rps": avg_rps[1],
            "avg_95_rps": avg_rps[2],
            "avg_lat": avg_lat[0],
            "avg_99_lat": avg_lat[1],
            "avg_95_lat": avg_lat[2],
        }

        field_order = (
            "method source specifier commit_hash test warmup duration endtime "
            "io-threads pipeline has_expire size preload_keys avg_rps avg_99_rps "
            "avg_95_rps avg_lat avg_99_lat avg_95_lat"
        )

        result_string = [f"{field}:{result[field]}" for field in field_order.split()]
        result_string = "\t".join(result_string) + "\n"
        with open(CONDUCTRESS_OUTPUT, "a", encoding="utf-8") as f:
            f.write(result_string)

        dump_field_order = (
            "method source specifier commit_hash test warmup duration endtime io-threads pipeline size"
        )
        dump_string = [f"{field}:{result[field]}" for field in dump_field_order.split()]
        dump_string = "\t".join(dump_string)
        with open(CONDUCTRESS_DATA_DUMP, "a", encoding="utf-8") as f:
            f.write(dump_string)
            f.write(f"\trps_data={repr(self.rps_data)}\tlat_data={repr(self.lat_data)}\n")

    def run(self):
        """Run the benchmark."""
        benchmark_update_interval = 0.1  # s

        self.command.start()
        start_time = time.monotonic()
        test_start_time = start_time + self.warmup
        end_time = test_start_time + self.duration
        warming_up = True

        while self.command.is_running():
            self.__read_updates()
            self.__update_graph()
            time.sleep(benchmark_update_interval)
            now = time.monotonic()
            if now > end_time:
                self.command.kill()
                if self.profiling:
                    self.server.profiling_end()
            elif warming_up and now >= test_start_time:
                self.rps_data = []
                self.lat_data = []
                warming_up = False
                if self.profiling:
                    self.server.profiling_start(self.sample_rate)

        self.__read_updates()
        self.__record_result()
        if self.profiling:
            self.server.profiling_report(self.task_name)
