import datetime
import os
import subprocess
from numerize import numerize
from typing import Optional, TextIO
from collections import Counter

import plotext as plt
plt.theme('pro')

from config import *

import logging
logging.basicConfig(filename=conductress_log, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)
valkey_binary = 'valkey-server'
million = 1000000

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
    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    return output

def runServerCommand(command: list):
    command = ['ssh', '-i', sshkeyfile, server] + command
    return runCommand(command)

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
    runServerCommand(f'cd {repo_path}; git reset --hard && git fetch && git switch {branch} && git pull'.split())
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

class PerfResult:
    def __init__(self, string: str, count: int, duration: Optional[int], time: datetime):
        values = [x[1:-1] for x in string.split(',')]
        self.test = values[0]
        self.rps = float(values[1])
        self.avg_latency = float(values[2])
        self.p50_latency = float(values[3])

        self.duration = duration
        if (duration != None):
            self.count = -1
        else:
            self.count = count
        self.time = time

    def __str__(self) -> str:
        if self.duration:
            return f'{self.test} perf: {human(self.rps)} rps for {human(self.duration)} seconds'
        else:
            return f'{self.test} perf: {human(self.rps)} rps for {human(self.count)} requests'

    def log(self) -> str:
        return f'perf bench: time={self.time},test={self.test},rps={self.rps},avg_latency={self.avg_latency},p50_latency={self.p50_latency},count={self.count},duration={self.duration}'

def runBench(server: str, valsize: int, keyspace: int, pipelining: int, clients: int, threads: int, count: int, test: str, duration: Optional[int]) -> PerfResult:
    command = f'./valkey-benchmark -h {server} -d {valsize} -r {keyspace} -c {clients} -P {pipelining} --threads {threads} -t {test} --csv'
    if duration != None:
        command += f' -n {1000 * million} --test-duration {duration}'
    else:
        command += f' -n {count}'
    start = datetime.datetime.now()
    result = runCommand(command.split())

    result = result.strip().split('\n')
    assert(len(result) == 2)
    assert(result[0] == '"test","rps","avg_latency_ms","min_latency_ms","p50_latency_ms","p95_latency_ms","p99_latency_ms","max_latency_ms"')
    result = PerfResult(result[1], count, duration, start)
    logger.info(result.log())
    return result

def runBenchByCount(valsize, pipelining, count, test) -> float:
    result = runBench(server, valsize, perf_bench_keyspace, pipelining, 650, 64, count, test, None)
    return result.rps

def runBenchByMinutes(valsize, pipelining, minutes, test) -> float:
    result = runBench(server, valsize, perf_bench_keyspace, pipelining, 650, 64, 10, test, minutes*60)
    return result.rps

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

def preloadKeysForPerfTests(valsize: int, test_list: list[str]) -> None:
    load_type_for_test = {
        'set': 'set',
        'get': 'set',
        'sadd': 'sadd',
        }
    
    load_types = [load_type_for_test[x] for x in test_list]
    load_types = set(load_types)
    
    for type in load_types:
        loadKeys(valsize, perf_bench_keyspace, 4, type)

def perfTest(repo: str, commit_id: str, args: list[str], pipelining: int, test: str, valsize: int) -> None:
    minutes = 120

    title = f'{test} throughput, {repo}:{commit_id}, {repr(args)}, pipelining={pipelining} , size={valsize}'
    prettyHeader(title + f', {minutes} minutes')

    startServerWithArgs(repo, commit_id, args)
    print("preloading keys")
    preloadKeysForPerfTests(valsize, [test])
    print("starting test")

    bucket_size = 5000
    buckets = Counter()
    max_bucket = -1
    max_bucket_count = -1
    bench_value = -1
    
    data = []
    for i in range(0,minutes):
        # rps = runBenchByCount(valsize, pipelining, 2 * million, test)
        rps = runBenchByMinutes(valsize, pipelining, 1, test)
        data.append(rps)
        bucket_val = rps - (rps % bucket_size)
        buckets[bucket_val] += 1
        if buckets[bucket_val] > max_bucket_count:
            max_bucket = bucket_val
            max_bucket_count = buckets[bucket_val]
            bench_value = [x for x in data if x >= bucket_val-bucket_size and x < bucket_val+bucket_size]
            bench_value = sum(bench_value) / len(bench_value)

        # plt.clear_terminal()
        # plt.clear_data()
        plt.clear_terminal()
        plt.clear_figure()
        plt.theme('matrix')
        plt.title(title + f', {i}/{minutes} minutes')
        plt.subplots(2,1)

        plt.subplot(1,1)
        plt.plot(data)
        plt.horizontal_line(bench_value, color='white')
        plt.xlim(left=0, right=minutes)
        plt.frame(False)

        plt.subplot(2,1)
        plt.hist(data, bins=40)
        plt.frame(False)
        plt.show()

    print(f'done testing {test}: {bench_value} rps')
    with open(conductress_output,'a') as f:
        f.write(f'{datetime.datetime.now()}, {test}, {repo}:{commit_id}, {repr(args)}, pipelining={pipelining}, size={valsize}, rps={bench_value}, data={repr(data)}\n')

def memEfficiencyTest(repo: str, commit_id: str, sizelist: list[int], test: str, count: int) -> None:
    prettyHeader(f'Memory efficiency testing {repo}:{commit_id} with {human(count)} {test} elements')
    args = ['--io-threads', '9']

    print ('size, total, per key, overhead')
    for valsize in sizelist:
        startServerWithArgs(repo, commit_id, args)

        before_usage = measureUsedMemory()

        print(f'loading {human(count)} {test} {humanByte(valsize)} elements', end='', flush=True)
        loadKeys(valsize, count, 1, test)

        after_usage = measureUsedMemory()

        # output result
        total_usage = after_usage - before_usage
        per_key = float(total_usage) / count
        per_key_overhead = per_key - valsize
        print(f'                  \r{valsize}, {total_usage}, {per_key:.2f}, {per_key_overhead:.2f}')

def parseLazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

# repolist = ['valkey', 'SoftlyRaining', 'zuiderkwast']

# versions = ['origin/7.2', 'origin/8.0', 'unstable']
# sizelist = list(range(8, 128, 8)) + list(range(128, 544, 32))
# print(sizelist)
# for version in versions:
#     perfTest(version, [], 512, 1)
#     perfTest(version, [], 512, 4)
#     perfTest(version, ['--io-threads', '9'], 512, 1)
#     perfTest(version, ['--io-threads', '9'], 512, 4)
#     memEfficiencyTest(version, sizelist, 'set', 10 * million)

# sizelist = list(range(24, 96, 8)) + list(range(23, 95, 8))
# sizelist.sort()
# print(len(sizelist), 'sizes', sizelist)
# tests = ['get','set']
# for specifier in ['valkey:unstable', 'zuiderkwast:embed-128']:
#     (repo, branch) = parseLazy(specifier)
#     # perfTest(repo, branch, ['--io-threads', '9'], sizelist, 1, tests)
#     memEfficiencyTest(repo, branch, sizelist, 'set', 5 * million)

perf_bench_keyspace = 3 * million
warmup = 2 * million
count = 2 * million

configs = [(True, 4)] # [(True, 4),(True, 1), (False, 4)]
tests = ['get','sadd']
versions = ['valkey:7.2', 'valkey:8.0', 'valkey:unstable']
for (io_threads, pipelining) in configs:
    args = ['--io-threads', '9'] if io_threads else []
    for test in tests:
        for version in versions:
            (repo, branch) = parseLazy(version)
            perfTest(repo, branch, args, pipelining, test, 512)
