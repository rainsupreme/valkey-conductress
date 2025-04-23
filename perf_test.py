import datetime
import logging
import plotext as plt
import time
from threading import Thread

from config import perf_bench_keyspace, conductress_output, conductress_data_dump
from utility import *
from server import Server
from pathlib import Path

logger = logging.getLogger(__name__)

class PerfBench:
    def __init__(self, task_name: str, server_ip: str, repo: str, specifier: str, io_threads: int, valsize: int, pipelining: int, test: str, warmup: int, duration: int, preload_keys: bool, has_expire: bool, sample_rate: int = -1):
        self.title = f'{test} throughput, {repo}:{specifier}, io-threads={io_threads}, pipelining={pipelining}, size={valsize}, warmup={human_time(warmup)}, duration={human_time(duration)}'

        # settings
        self.task_name = task_name
        self.server_ip = server_ip
        self.repo = repo
        self.specifier = specifier
        self.io_threads = io_threads
        self.valsize = valsize
        self.pipelining = pipelining
        self.test = test
        self.warmup = warmup # seconds
        self.duration = duration # seconds
        self.preload_keys = preload_keys
        self.has_expire = has_expire
        self.sample_rate = sample_rate

        self.profiling = self.sample_rate > 0
        if (self.profiling):
            self.title = f'==PROFILING==   {self.title}, profiling={self.sample_rate}Hz   ==PROFILING=='

        # statistics
        self.avg_rps = -1.0

        clients = 650
        threads = 64
        self.command_string = f'./valkey-benchmark -h {server_ip} -d {valsize} -r {perf_bench_keyspace} -c {clients} -P {pipelining} --threads {threads} -t {test} -q -l -n {2000 * million}'
        self.rps_data = []
        self.lat_data = []

        print("preparing:", self.title)

        server_args = [] if io_threads == 1 else ['--io-threads', str(io_threads)]
        self.server = Server(server_ip, repo, specifier, server_args)
        self.commit_hash = self.server.get_commit_hash()
        if self.preload_keys:
            self.server.fill_keyspace(self.valsize, perf_bench_keyspace, self.test)
            if self.has_expire:
                self.server.expire_keyspace(perf_bench_keyspace)


    def __read_updates(self):
        line = self.command.poll_output()
        while line != None and line != '' and not line.isspace():
            if 'overall' not in line:
                line = self.command.poll_output()
                continue
            # line looks like this: "GET: rps=140328.0 (overall: 141165.2) avg_msec=0.193 (overall: 0.191)"
            line = line.strip().split()
            rps = float(line[1][4:])
            msec = float(line[-3][9:])
            self.rps_data.append(rps)
            self.lat_data.append(msec)
            
            line = self.command.poll_output()

    def __update_graph(self):
        graph_update_interval = 10.0
        now = time.monotonic()
        if now > self.next_metric_update or not self.command.is_running():
            self.next_metric_update += graph_update_interval
            if len(self.rps_data) == 0:
                return

            # update metrics
            self.avg_rps = sum(self.rps_data) / len(self.rps_data)

            # update graph
            plt.clear_terminal()
            plt.clear_figure()
            plt.canvas_color('black')
            plt.axes_color('black')
            plt.ticks_color('orange')

            plt.scatter(self.rps_data, marker='braille', color='orange+')
            plt.horizontal_line(self.avg_rps, color='white')
            plt.title(self.title)
            plt.frame(False)

            xticks = range(1, self.duration*4+1, 4*60*10)
            xticks_labels = [f'{i//4//60}m' for i in xticks]
            plt.xticks(xticks, xticks_labels)
            plt.xlim(left=1, right=self.duration*4+1)

            rps_min = min(self.rps_data)
            rps_max = max(self.rps_data) + 999
            inverval = int(rps_max-rps_min)//8
            yticks = range(int(rps_min), int(rps_max) + inverval, inverval)
            ytick_labels = [f'{tick//1000}k' for tick in yticks]
            plt.yticks(yticks, ytick_labels)

            plt.show()

    def __record_result(self):
        logger.debug(f'{self.valsize}B {self.test} perf on {self.repo}:{self.specifier}\trps_data={repr(self.rps_data)}\tlat_data={repr(self.lat_data)}')

        (avg_rps, avg_99_rps, avg_95_rps) = calc_percentile_averages(self.rps_data, (100, 99, 95))
        (avg_lat, avg_99_lat, avg_95_lat) = calc_percentile_averages(self.lat_data, (100, 99, 95), lowestVals=True)

        result = {
            'method': 'perf',
            'repo': self.repo,
            'specifier': self.specifier,
            'commit_hash': self.commit_hash,
            'test': self.test,
            'warmup': self.warmup,
            'duration': self.duration,
            'endtime': datetime.datetime.now(),
            'io-threads': self.io_threads,
            'pipeline': self.pipelining,
            'has_expire': self.has_expire,
            'size': self.valsize,
            'preload_keys': self.preload_keys,
            'avg_rps': avg_rps,
            'avg_99_rps': avg_99_rps,
            'avg_95_rps': avg_95_rps,
            'avg_lat': avg_lat,
            'avg_99_lat': avg_99_lat,
            'avg_95_lat': avg_95_lat,
        }

        result_fields = 'method repo specifier commit_hash test warmup duration endtime io-threads pipeline has_expire size preload_keys avg_rps avg_99_rps avg_95_rps avg_lat avg_99_lat avg_95_lat'.split()

        result_string = [f'{field}:{result[field]}' for field in result_fields]
        result_string = '\t'.join(result_string) + '\n'
        with open(conductress_output,'a') as f:
            f.write(result_string)

        dump_fields = 'method repo specifier commit_hash test warmup duration endtime io-threads pipeline size'.split()
        dump_string = [f'{field}:{result[field]}' for field in dump_fields]
        dump_string = '\t'.join(dump_string)
        with open(conductress_data_dump, 'a') as f:
            f.write(dump_string)
            f.write(f'\trps_data={repr(self.rps_data)}\tlat_data={repr(self.lat_data)}\n')

    def __run_profiling(self):
        print("Beginning Profile...")
        run_server_command(f'sudo perf record -F {self.sample_rate} -a -g -o perf.data -- sleep {self.duration}'.split())

        print("Profile complete, generating flamegraph...")
        self.command.kill() # end the benchmark

        test_dir = Path("results") / Path(self.task_name)
        test_dir.mkdir(parents=True, exist_ok=False)

        run_server_command("sudo chown $(whoami):$(whoami) perf.data".split())
        perf_data = test_dir / "perf.data"
        scp_file_from_server("perf.data", perf_data)
        run_command(f"perf script report flamegraph".split(), cwd=perf_data.parent)
        print("Done generating flamegraph")

    def run(self):
        benchmark_update_interval = 0.1 # s
        self.start_time = time.monotonic()
        self.test_start_time = self.start_time + self.warmup
        self.end_time = self.test_start_time + self.duration
        warming_up = True
        self.next_metric_update = self.start_time
        self.command = RealtimeCommand(self.command_string.split())
        while self.command.is_running():
            self.__read_updates()
            self.__update_graph()
            time.sleep(benchmark_update_interval)
            now = time.monotonic()
            if not self.profiling and now > self.end_time:
                self.command.kill()
            elif warming_up and now >= self.test_start_time:
                self.rps_data = []
                self.lat_data = []
                warming_up = False
                if self.profiling:
                    self.profiling_thread = Thread(target=self.__run_profiling)
                    self.profiling_thread.start()
        self.__read_updates()
        self.__record_result()
        if self.profiling:
            self.profiling_thread.join()
