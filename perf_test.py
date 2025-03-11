import datetime
import logging
import plotext as plt
import time
from collections import Counter, defaultdict

from config import perf_bench_keyspace, conductress_output, conductress_data_dump
from utility import *
from server import Server

logger = logging.getLogger(__name__)

def preload_keys_for_perf_tests(valsize: int, test_list: list[str]) -> None:
    load_type_for_test = {
        'set': 'set',
        'get': 'set',
        'sadd': 'sadd',
        'hset': 'hset',
        'zadd': 'zadd',
        'zrange': 'zadd',
        }
    
    load_types = [load_type_for_test[x] for x in test_list]
    load_types = set(load_types)
    
    print("preloading keys:", repr(load_types))
    for type in load_types:
        load_sequential_keys(valsize, perf_bench_keyspace, type)

class PerfBench:
    def __init__(self, server: str, repo: str, commit_id: str, io_threads: int, valsize: int, pipelining: int, test: str, warmup: int, duration: int):
        self.title = f'{test} throughput, {repo}:{commit_id}, io-threads={io_threads}, pipelining={pipelining}, size={valsize}, warmup={human_time(warmup)}, duration={human_time(duration)}'

        # settings
        self.server = server
        self.repo = repo
        self.commit_id = commit_id
        self.io_threads = io_threads
        self.valsize = valsize
        self.pipelining = pipelining
        self.test = test
        self.warmup = warmup # seconds
        self.duration = duration # seconds

        # statistics
        self.bucket_size = 1000
        self.count_buckets = Counter()
        self.val_buckets = defaultdict(list)
        self.max_bucket = -1
        self.mode_rps = -1.0
        self.avg_rps = -1.0

        clients = 650
        threads = 64
        self.command_string = f'./valkey-benchmark -h {server} -d {valsize} -r {perf_bench_keyspace} -c {clients} -P {pipelining} --threads {threads} -t {test} -q -l -n {2000 * million}'
        self.rps_data = []
        self.lat_data = []

        print("preparing:", self.title)

        server_args = [] if io_threads == 1 else ['--io-threads', str(io_threads)]
        self.server = Server(repo, commit_id, server_args)
        self.commit_hash = self.server.get_commit_hash()
        preload_keys_for_perf_tests(valsize, [test])

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
            
            bucket_key = int(rps - (rps % self.bucket_size))
            self.count_buckets[bucket_key] += 1
            self.val_buckets[bucket_key].append(rps)
            if self.count_buckets[bucket_key] > self.count_buckets[self.max_bucket]:
                self.max_bucket = bucket_key
            line = self.command.poll_output()

    def __update_metrics(self):
        graph_update_interval = 10.0
        now = time.monotonic()
        if now > self.next_metric_update or not self.command.is_running():
            self.next_metric_update += graph_update_interval
            if len(self.rps_data) == 0:
                return

            # update metrics
            self.avg_rps = sum(self.rps_data) / len(self.rps_data)
            mode_band_center = self.max_bucket + self.bucket_size/2
            mode_band_radius = 5000 # diameter is double this value
            mode_list = [x for x in self.rps_data if x >= mode_band_center - mode_band_radius and x <= mode_band_center + mode_band_radius]
            if (len(mode_list)) > 0:
                self.mode_rps = sum(mode_list) / len(mode_list)

            # update graph
            plt.clear_terminal()
            plt.clear_figure()
            plt.theme('matrix')
            plt.subplots(2,1)

            plt.subplot(1,1)
            plt.scatter(self.rps_data)
            plt.title(self.title)
            plt.xlim(left=1, right=self.duration*4)
            plt.horizontal_line(self.avg_rps, color='orange+')
            plt.horizontal_line(self.mode_rps, color='white')
            plt.frame(False)

            plt.subplot(2,1)
            plt.hist(self.rps_data, width=1, bins=40)
            plt.vertical_line(self.avg_rps, color='orange+')
            plt.vertical_line(self.mode_rps, color='white')
            plt.frame(False)

            plt.show()

    def __record_result(self):
        logger.debug(f'{self.valsize}B {self.test} perf on {self.repo}:{self.commit_id}\trps_data={repr(self.rps_data)}\tlat_data={repr(self.lat_data)}')

        (avg_rps, avg_99_rps, avg_95_rps) = calc_percentile_averages(self.rps_data, (100, 99, 95))
        (avg_lat, avg_99_lat, avg_95_lat) = calc_percentile_averages(self.lat_data, (100, 99, 95), lowestVals=True)

        result = {
            'method': 'perf',
            'repo': self.repo,
            'commit': self.commit_id,
            'commit_hash': self.commit_hash,
            'test': self.test,
            'warmup': self.warmup,
            'duration': self.duration,
            'endtime': datetime.datetime.now(),
            'io-threads': self.io_threads,
            'pipeline': self.pipelining,
            'size': self.valsize,
            'mode_rps': self.mode_rps,
            'avg_rps': avg_rps,
            'avg_99_rps': avg_99_rps,
            'avg_95_rps': avg_95_rps,
            'avg_lat': avg_lat,
            'avg_99_lat': avg_99_lat,
            'avg_95_lat': avg_95_lat,
        }

        result_fields = 'method repo commit commit_hash test warmup duration endtime io-threads pipeline size mode_rps avg_rps avg_99_rps avg_95_rps avg_lat avg_99_lat avg_95_lat'.split()

        result_string = [f'{field}:{result[field]}' for field in result_fields]
        result_string = '\t'.join(result_string) + '\n'
        with open(conductress_output,'a') as f:
            f.write(result_string)

        dump_fields = 'method repo commit commit_hash test warmup duration endtime io-threads pipeline size'.split()
        dump_string = [f'{field}:{result[field]}' for field in dump_fields]
        dump_string = '\t'.join(dump_string)
        with open(conductress_data_dump, 'a') as f:
            f.write(dump_string)
            f.write(f'\trps_data={repr(self.rps_data)}\tlat_data={repr(self.lat_data)}\n')

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
            self.__update_metrics()
            time.sleep(benchmark_update_interval)
            now = time.monotonic()
            if now > self.end_time:
                self.command.kill()
            elif warming_up and now >= self.test_start_time:
                self.rps_data = []
                self.lat_data = []
                warming_up = False
        self.__read_updates()
        self.__record_result()