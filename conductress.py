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

def getCachedBuildPath(repo: str, hash_id: str) -> str:
    return os.path.join('~', 'build_cache', repo, hash_id)

def getRepoBinaryPath(repo: str) -> str:
    return os.path.join('~',repo,'src')

def human(number: float) -> str:
    return numerize.numerize(number)

def humanByte(number: float) -> str:
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

def humanTime(number: float) -> str:
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

def prettyHeader(text: str):
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

def runCommand(command: list):
    result = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
    return result.stdout

def runServerCommand(command: list):
    command = ['ssh', '-i', sshkeyfile, server] + command
    return runCommand(command)

class RealtimeCommand:
    def __init__(self, command: list):
        self.p = subprocess.Popen(command, stdout=subprocess.PIPE)
        os.set_blocking(self.p.stdout.fileno(), False)
    def pollOutput(self):
        output = self.p.stdout.read()
        if output != None:
            output = output.decode('utf-8')
        return output
    def isRunning(self):
        return self.p.poll() == None
    def kill(self):
        self.p.kill()

def getCurrentCommitHash(repo: str) -> str:
    repo_path = getRepoBinaryPath(repo)
    return runServerCommand(f'cd {repo_path}; git rev-parse HEAD'.split()).strip()

def checkServerFileExists(path: str) -> bool:
    command = f'[[ -f {path} ]] && echo 1 || echo 0;'.split()
    result = runServerCommand(command)
    return result.strip() == '1'

def isBuildCached(repo: str, hash_id: str) -> bool:
    return checkServerFileExists(os.path.join(getCachedBuildPath(repo, hash_id), valkey_binary))

def ensureServerStopped():
    runServerCommand(['pkill', '-f', valkey_binary])

def ensureBuildCached(repo: str, commit_id: str) -> str:
    repo_path = getRepoBinaryPath(repo)
    runServerCommand(f'cd {repo_path}; git reset --hard && git fetch && git switch {commit_id} && git pull'.split())
    hash_id = getCurrentCommitHash(repo)
    
    cached_build_path = getCachedBuildPath(repo, hash_id)
    cached_binary_path = os.path.join(cached_build_path, valkey_binary)

    if not isBuildCached(repo, hash_id):
        print(f"building {commit_id}... (no cached build)")

        runServerCommand(f'cd {repo_path}; make distclean && make -j USE_FAST_FLOAT=yes'.split())
        runServerCommand(f'mkdir -p {cached_build_path}'.split())
        build_binary = os.path.join(repo_path, valkey_binary)
        runServerCommand(['cp', build_binary, cached_binary_path])

    return cached_binary_path

def startServerWithArgs(repo: str, commit_id: str, args: list) -> None:
    ensureServerStopped()
    cached_binary_path = ensureBuildCached(repo, commit_id)
    command = [cached_binary_path, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
    runServerCommand(command)

def loadKeys(valsize: int, count: int, pipelining: int, test: str) -> None:
    runCommand(f'./amz_valkey-benchmark -h {server} -d {valsize} --sequential {count} -c 650 -P {pipelining} --threads 50 -t {test} -q'.split())

def preloadKeysForPerfTests(valsize: int, test_list: list[str]) -> None:
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
        loadKeys(valsize, perf_bench_keyspace, 4, type)

class PerfBench:
    def __init__(self, server: str, repo: str, commit_id: str, server_args: list, valsize: int, pipelining: int, test: str, warmup: int, duration: int):
        self.title = f'{test} throughput, {repo}:{commit_id}, {repr(server_args)}, pipelining={pipelining}, size={valsize}, warmup={humanTime(warmup)}, duration={humanTime(duration)}'

        # settings
        self.server = server
        self.repo = repo
        self.commit_id = commit_id
        self.server_args = server_args
        self.valsize = valsize
        self.pipelining = pipelining
        self.test = test
        self.warmup = warmup # seconds
        self.duration = duration # seconds

        # statistics
        self.bucket_size = 5000
        self.count_buckets = Counter()
        self.val_buckets = defaultdict(list)
        self.max_bucket = -1
        self.mode_value = -1

        clients = 650
        threads = 64
        self.command_string = f'./valkey-benchmark -h {server} -d {valsize} -r {perf_bench_keyspace} -c {clients} -P {pipelining} --threads {threads} -t {test} -q -l -n {2000 * million}'
        self.data = []
        self.p90 = -1.0

        startServerWithArgs(repo, commit_id, server_args)
        preloadKeysForPerfTests(valsize, [test])

    def __readUpdates(self):
        line = self.command.pollOutput().strip()
        while line != None and line != '':
            if 'overall' not in line:
                line = self.command.pollOutput()
                continue
            rps = float(line.split()[1][4:])
            self.data.append(rps)
            
            bucket_key = int(rps - (rps % self.bucket_size))
            self.count_buckets[bucket_key] += 1
            self.val_buckets[bucket_key].append(rps)
            if self.count_buckets[bucket_key] > self.count_buckets[self.max_bucket]:
                self.max_bucket = bucket_key
            line = self.command.pollOutput()

    def __updateMetrics(self):
        graph_update_interval = 10.0
        now = time.monotonic()
        if now > self.next_metric_update or not self.command.isRunning():
            self.next_metric_update += graph_update_interval
            if len(self.data) == 0:
                return

            # update metrics
            avg = sum(self.data) / len(self.data)
            mode_list = self.val_buckets[self.max_bucket] + self.val_buckets[self.max_bucket - self.bucket_size] + self.val_buckets[self.max_bucket + self.bucket_size]
            if (len(mode_list)) > 0:
                self.mode_value = sum(mode_list) / len(mode_list)

            # update graph
            plt.clear_terminal()
            plt.clear_figure()
            plt.theme('matrix')
            plt.subplots(2,1)

            plt.subplot(1,1)
            plt.scatter(self.data)
            plt.title(self.title)
            plt.xlim(left=1, right=self.duration*4)
            plt.horizontal_line(avg, color='orange+')
            plt.horizontal_line(self.mode_value, color='white')
            plt.frame(False)

            plt.subplot(2,1)
            plt.hist(self.data, width=1, bins=40)
            plt.vertical_line(avg, color='orange+')
            plt.vertical_line(self.mode_value, color='white')
            plt.frame(False)

            plt.show()

    def __recordResult(self):
        with open(conductress_output,'a') as f:
            f.write(f'perf, {datetime.datetime.now()}, {self.test}, mode_rps={self.mode_value}, warmup={self.warmup}, duration={self.duration}, {self.repo}:{self.commit_id}, {repr(self.server_args)}, pipelining={self.pipelining}, size={self.valsize}, data={repr(self.data)}\n')

    def run(self):
        benchmark_update_interval = 0.1 # s
        self.start_time = time.monotonic()
        self.test_start_time = self.start_time + self.warmup
        self.end_time = self.test_start_time + self.duration
        warming_up = True
        self.next_metric_update = self.start_time
        self.command = RealtimeCommand(self.command_string.split())
        while self.command.isRunning():
            self.__readUpdates()
            self.__updateMetrics()
            time.sleep(benchmark_update_interval)
            now = time.monotonic()
            if now > self.end_time:
                self.command.kill()
            elif warming_up and now >= self.test_start_time:
                self.data = []
                warming_up = False
        self.__readUpdates()
        self.__recordResult()

def infoCommand(section: str) -> dict:
    result = runCommand(f'./valkey-cli -h {server} info {section}'.split())
    result = result.strip().split('\n')
    keypairs = {}
    for item in result:
        if ':' in item:
            (key, value) = item.strip().split(':')
            keypairs[key.strip()] = value.strip()
    return keypairs

def measureUsedMemory() -> int:
    info = infoCommand('memory')
    return int(info['used_memory'])

def memTest(repo: str, commit_id: str, valsize: int, test: str) -> None:
    count = 5 * million
    prettyHeader(f'Memory efficiency testing {repo}:{commit_id} with {human(count)} {humanByte(valsize)} {test} elements')

    args = ['--io-threads', '9']
    startServerWithArgs(repo, commit_id, args)

    before_usage = measureUsedMemory()

    print(f'loading {human(count)} {humanByte(valsize)} {test} elements')
    loadKeys(valsize, count, 1, test)

    after_usage = measureUsedMemory()

    # output result
    total_usage = after_usage - before_usage
    per_key = float(total_usage) / count
    per_key_overhead = per_key - valsize
    print(f'done testing {humanByte(valsize)} {test} elements: {per_key_overhead:.2f} overhead per key')
    with open(conductress_output,'a') as f:
        f.write(f'mem, {datetime.datetime.now()}, {test}, {repo}:{commit_id}, {repr(args)}, size={valsize}, total_usage={total_usage}, key_mem={per_key:.2f}, key_overhead={per_key_overhead:.2f}\n')

def parseLazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

class BenchTask:
    def __init__(self, test: str, repo: str, commit_id: str, iothreads: int, pipelining: int, data_size: int, data_type: str):
        self.test = test
        self.repo = repo
        self.commit_id = commit_id
        self.iothreads = iothreads
        self.pipelining = pipelining
        self.data_size = data_size
        self.data_type = data_type

    def from_string(string: str):
        bits = [x.strip() for x in string.split(',')]
        assert(len(bits) == 7)
        return BenchTask(bits[0], bits[1], bits[2], int(bits[3]), int(bits[4]), int(bits[5]), bits[6])

    def runTest(self):
        args = []
        if (self.iothreads > 1):
            args = ['--io-threads', str(self.iothreads)]
        if (self.test == 'perf'):
            perfTest(self.repo, self.commit_id, args, self.pipelining, self.data_type, self.data_size)
        elif (self.test == 'mem'):
            memTest(self.repo, self.commit_id, self.data_size, self.data_type)
        else:
            print(f'ERROR unknown testing method {self.test}')
            logger.error(f'ERROR unknown testing method {self.test}')
            exit()

def tryPopTaskFromQueueFile():
    # TODO load queue file and try to pop file from it
    return None

def watchQueue() -> None:
    # pick off tasks from queue file and run them. Wait idly when file is empty.
    while True:
        print('getting task from queue', end='', flush=True)
        task = None
        while True:
            task = tryPopTaskFromQueueFile()
            if task != None:
                break
            time.sleep(60) # 1 minute
            print('.', end='', flush=True)
        task.runTest()


# sizelist = list(range(24, 96, 8)) + list(range(23, 95, 8))
# sizelist.sort()
# print(len(sizelist), 'sizes', sizelist)
# tests = ['get','set']
# for specifier in ['valkey:unstable', 'zuiderkwast:embed-128']:
#     (repo, branch) = parseLazy(specifier)
#     # perfTest(repo, branch, ['--io-threads', '9'], sizelist, 1, tests)
#     memEfficiencyTest(repo, branch, sizelist, 'set', 5 * million)

# repolist = ['valkey', 'SoftlyRaining', 'zuiderkwast']
configs = [(True, 4), (True, 1), (False, 4)]
tests = ['set','get','sadd','hset','zadd','zrange']
versions = ['valkey:7.2', 'valkey:8.0', 'valkey:unstable']

for (io_threads, pipelining) in configs:
    args = ['--io-threads', '9'] if io_threads else []
    for test in tests:
        for version in versions:
            (repo, branch) = parseLazy(version)
            test_runner = PerfBench(server, repo, branch, server_args=args, valsize=512, pipelining=pipelining, test=test, warmup=5*minute, duration=60*minute)
            test_runner.run()

print('\n\nAll done 🌧 ♥')
