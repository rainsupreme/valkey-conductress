import datetime
import os
import logging
import plotext as plt
import subprocess
import time
from collections import Counter, defaultdict
from numerize import numerize
# from typing import Optional, TextIO

from config import *

plt.theme('pro')
logging.basicConfig(filename=conductress_log, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_cached_build_path(repo: str, hash_id: str) -> str:
    return os.path.join('~', 'build_cache', repo, hash_id)

def get_repo_binary_path(repo: str) -> str:
    return os.path.join('~',repo,'src')

def human(number: float) -> str:
    return numerize.numerize(number)

def human_byte(number: float) -> str:
    number = float(number)
    units = ('B','KB','MB','GB','TB','PB')
    unit_index = 0
    while number >= 512 and unit_index+1 < len(units):
        number /= 1024
        unit_index += 1
    if number.is_integer():
        return f'{number:,g}{units[unit_index]}'
    else:
        return f'{number:,.1f}{units[unit_index]}'

def human_time(number: float) -> str:
    number = float(number)
    divisors = [1, 60, 60, 24]
    units = 'smhd'
    unit_index = 0
    while unit_index+1 < len(units) and number >= divisors[unit_index+1]:
        unit_index += 1
        number /= divisors[unit_index]
    if number.is_integer():
        return f'{number:,g}{units[unit_index]}'
    else:
        return f'{number:,.1f}{units[unit_index]}'

def pretty_header(text: str):
    margin = 1
    text = ' '*margin + text + ' '*margin

    endcap = '•'
    center = '•°•♥•°•'
    fill = '─'
    fillsize = len(text) - len(center) - len(endcap) * 2
    fillsize = fillsize//2 + 1
    divider = '\n' + endcap + fill*fillsize + center + fill*fillsize + endcap

    print(divider)
    print(text)

def run_command(command: list):
    result = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
    return result.stdout

def run_server_command(command: list):
    command = ['ssh', '-i', sshkeyfile, server] + command
    return run_command(command)

class RealtimeCommand:
    def __init__(self, command: list):
        self.p = subprocess.Popen(command, stdout=subprocess.PIPE)
        os.set_blocking(self.p.stdout.fileno(), False)
    def poll_output(self):
        output = self.p.stdout.read()
        if output != None:
            output = output.decode('utf-8')
        return output
    def is_running(self):
        return self.p.poll() == None
    def kill(self):
        self.p.kill()

def get_current_commit_hash(repo: str) -> str:
    repo_path = get_repo_binary_path(repo)
    return run_server_command(f'cd {repo_path}; git rev-parse HEAD'.split()).strip()

def check_server_file_exists(path: str) -> bool:
    command = f'[[ -f {path} ]] && echo 1 || echo 0;'.split()
    result = run_server_command(command)
    return result.strip() == '1'

def is_build_cached(repo: str, hash_id: str) -> bool:
    return check_server_file_exists(os.path.join(get_cached_build_path(repo, hash_id), valkey_binary))

def ensure_server_stopped():
    run_server_command(['pkill', '-f', valkey_binary])

def ensure_build_cached(repo: str, commit_id: str) -> str:
    repo_path = get_repo_binary_path(repo)
    run_server_command(f'cd {repo_path}; git reset --hard && git fetch && git switch {commit_id} && git pull'.split())
    hash_id = get_current_commit_hash(repo)
    
    cached_build_path = get_cached_build_path(repo, hash_id)
    cached_binary_path = os.path.join(cached_build_path, valkey_binary)

    if not is_build_cached(repo, hash_id):
        print(f"building {commit_id}... (no cached build)")

        run_server_command(f'cd {repo_path}; make distclean && make -j USE_FAST_FLOAT=yes'.split())
        run_server_command(f'mkdir -p {cached_build_path}'.split())
        build_binary = os.path.join(repo_path, valkey_binary)
        run_server_command(['cp', build_binary, cached_binary_path])

    return cached_binary_path

def start_server_with_args(repo: str, commit_id: str, args: list) -> str:
    ensure_server_stopped()
    cached_binary_path = ensure_build_cached(repo, commit_id)
    command = [cached_binary_path, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
    run_server_command(command)
    return get_current_commit_hash(repo)

def load_keys(valsize: int, count: int, pipelining: int, test: str) -> None:
    run_command(f'./amz_valkey-benchmark -h {server} -d {valsize} --sequential {count} -c 650 -P {pipelining} --threads 50 -t {test} -q'.split())

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
        load_keys(valsize, perf_bench_keyspace, 4, type)

def calc_percentile_averages(data: list, percentages, lowestVals=False) -> tuple:
    copy = data.copy()
    copy.sort()
    if lowestVals:
        copy.reverse()

    result = []
    for percent in percentages:
        start_index = len(copy) * (100 - percent) // 100
        slice = copy[start_index:]
        slice_avg = sum(slice) / len(slice)
        result.append(slice_avg)
    return tuple(result)

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

        server_args = [] if io_threads == 1 else ['--io-threads', str(io_threads)]
        self.commit_hash = start_server_with_args(repo, commit_id, server_args)
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

        fieldOrder = 'method repo commit commit_hash test warmup duration endtime io-threads pipeline size mode_rps avg_rps avg_99_rps avg_95_rps avg_lat avg_99_lat avg_95_lat'.split()

        result_string = [f'{field}:{result[field]}' for field in fieldOrder]
        result_string = '\t'.join(result_string) + '\n'
        with open(conductress_output,'a') as f:
            f.write(result_string)

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

def info_command(section: str) -> dict:
    result = run_command(f'./valkey-cli -h {server} info {section}'.split())
    result = result.strip().split('\n')
    keypairs = {}
    for item in result:
        if ':' in item:
            (key, value) = item.strip().split(':')
            keypairs[key.strip()] = value.strip()
    return keypairs

def measure_used_memory() -> int:
    info = info_command('memory')
    return int(info['used_memory'])

def mem_test(repo: str, commit_id: str, valsize: int, test: str) -> None:
    count = 5 * million
    pretty_header(f'Memory efficiency testing {repo}:{commit_id} with {human(count)} {human_byte(valsize)} {test} elements')

    args = ['--io-threads', '9']
    start_server_with_args(repo, commit_id, args)

    before_usage = measure_used_memory()

    print(f'loading {human(count)} {human_byte(valsize)} {test} elements')
    load_keys(valsize, count, 1, test)

    after_usage = measure_used_memory()

    # output result
    total_usage = after_usage - before_usage
    per_key = float(total_usage) / count
    per_key_overhead = per_key - valsize
    print(f'done testing {human_byte(valsize)} {test} elements: {per_key_overhead:.2f} overhead per key')
    with open(conductress_output,'a') as f:
        f.write(f'mem, {datetime.datetime.now()}, {test}, {repo}:{commit_id}, {repr(args)}, size={valsize}, total_usage={total_usage}, key_mem={per_key:.2f}, key_overhead={per_key_overhead:.2f}\n')

def parse_lazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

def run_script():
    # sizelist = list(range(24, 96, 8)) + list(range(23, 95, 8))
    # sizelist.sort()
    # print(len(sizelist), 'sizes', sizelist)
    # tests = ['get','set']
    # for specifier in ['valkey:unstable', 'zuiderkwast:embed-128']:
    #     (repo, branch) = parseLazy(specifier)
    #     # perfTest(repo, branch, ['--io-threads', '9'], sizelist, 1, tests)
    #     memEfficiencyTest(repo, branch, sizelist, 'set', 5 * million)

    # repolist = ['valkey', 'SoftlyRaining', 'zuiderkwast']

    # sizes = [512]
    # configs = [(True, 4), (True, 1), (False, 4)]
    # tests = ['set','get','sadd','hset','zadd','zrange']
    # versions = ['valkey:7.2', 'valkey:8.0', 'valkey:unstable']
    sizes = [512, 87, 8]
    configs = [(9, 1)] # (io-threads, pipelining)
    tests = ['set'] # ['set','get']
    versions = ['valkey:unstable', 'valkey:8.0', 'valkey:7.2']

    for version in versions:
        (repo, branch) = parse_lazy(version)
        for size in sizes:
            for (io_threads, pipelining) in configs:
                for test in tests:
                    test_runner = PerfBench(server, repo, branch, io_threads=io_threads, valsize=size, pipelining=pipelining, test=test, warmup=5*minute, duration=30*minute)
                    test_runner.run()

    print('\n\nAll done 🌧 ♥')

if __name__ == "__main__":
    run_script()

# TODO finish job queue
# TODO can I get ZRANGE test to work?
# TODO improve mode calculation?
# TODO log thread/irq cpu affinity over time