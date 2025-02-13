import os
import subprocess
from numerize import numerize

from config import *
valkey_binary = 'valkey-server'
million = 1000000
perf_bench_keyspace = 3 * million
warmup = 250 * million
count = 250 * million


def getCachedBuildPath(repo, hash_id):
    return os.path.join('~', 'build_cache', repo, hash_id)

def getRepoBinaryPath(repo):
    return os.path.join('~',repo,'src')

def human(number):
    return numerize.numerize(number)

def humanByte(number):
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
    
def prettyHeader(text):
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

def runCommand(command):
    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    return output

def runServerCommand(command):
    command = ['ssh', '-i', sshkeyfile, server] + command
    return runCommand(command)

def getServerHash(repo, commit_id):
    repo_path = getRepoBinaryPath(repo)
    return runServerCommand(['cd', repo_path, '&&', 'git', 'rev-parse', commit_id]).strip()

# def getCurrentServerHash(repo):
#     return getServerHash(repo, 'HEAD')

def checkServerFileExists(path):
    command = f'[[ -f {path} ]] && echo 1 || echo 0;'.split()
    result = runServerCommand(command)
    return result.strip() == '1'

def isBuildCached(repo, hash_id):
    return checkServerFileExists(os.path.join(getCachedBuildPath(repo, hash_id), valkey_binary))

def ensureServerStopped():
    runServerCommand(['pkill', '-f', valkey_binary])

# def doBuildCurrentState(repo):
#     repo_path = getRepoBinaryPath(repo)
#     runServerCommand(f'cd {repo_path}; make distclean && make -j'.split())

def buildAndCache(repo, hash_id):
    src_path = getRepoBinaryPath(repo)
    build_binary = os.path.join(src_path, valkey_binary)
    cached_build_path = getCachedBuildPath(repo, hash_id)
    cached_binary_path = os.path.join(cached_build_path, valkey_binary)
    runServerCommand(f'cd {src_path}; git checkout {hash_id} && make distclean && make -j'.split())
    runServerCommand(f'mkdir -p {cached_build_path}'.split())
    runServerCommand(['cp', build_binary, cached_binary_path])

# def startCurrentStateWithArgs(repo, args):
#     ensureServerStopped()

#     src_path = getRepoBinaryPath(repo)
#     build_binary = os.path.join(src_path, valkey_binary)
#     command = [build_binary, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
#     runServerCommand(command)

def startServerWithArgs(repo, commit_id, args):
    ensureServerStopped()

    hash_id = getServerHash(repo, commit_id)
    # if hash_id != commit_id:
    #     print(f"{commit_id} maps to hash {hash_id}")
    if not isBuildCached(repo, hash_id):
        print(f"building... (no cached build of {commit_id})")
        buildAndCache(repo, hash_id)

    cached_build_path = getCachedBuildPath(repo, hash_id)
    cached_binary_path = os.path.join(cached_build_path, valkey_binary)
    command = [cached_binary_path, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
    runServerCommand(command)

def loadKeys(valsize, count, pipelining, test):
    runCommand(f'./amz_valkey-benchmark -h {server} -d {valsize} --sequential {count} -c 650 -P {pipelining} --threads 50 -t {test} -q'.split())

def runBench(valsize, pipelining, count, test):
    result = runCommand(f'./valkey-benchmark -h {server} -d {valsize} -r {perf_bench_keyspace} -c 650 -P {pipelining} --threads 50 -n {count} -t {test} -q'.split())
    result = result.strip().split('\r')[-1]
    rps = float(result.split()[1])
    return rps

def infoCommand(section):
    result = runCommand(f'./valkey-cli -h {server} info {section}'.split())
    result = result.strip().split('\n')
    keypairs = {}
    for item in result:
        if ':' in item:
            (key, value) = item.strip().split(':')
            keypairs[key.strip()] = value.strip()
    return keypairs

def measureUsedMemory():
    info = infoCommand('memory')
    return int(info['used_memory'])

def preloadKeysForPerfTests(valsize, test_list):
    load_type_for_test = {
        'set': 'set',
        'get': 'set',
        'sadd': 'sadd',
        }
    
    load_types = [load_type_for_test[x] for x in test_list]
    load_types = set(load_types)
    
    for type in load_types:
        loadKeys(valsize, perf_bench_keyspace, 4, type)

def perfTest(repo, commit_id, args, sizelist, pipelining, tests):
    prettyHeader(f'Performance testing {repo}:{commit_id}, {repr(args)}, pipelining={pipelining} warmup={human(warmup)} count={human(count)}')

    print('size,test,rps')

    for valsize in sizelist:
        startServerWithArgs(repo, commit_id, args)
        preloadKeysForPerfTests(valsize, tests)
        
        for test in tests:
            print(f'{test} warmup...', end='', flush=True)
            rps = runBench(valsize, pipelining, warmup, test)
            
            print(f'        \r{human(rps)} rps warmup. testing {test}...', end='', flush=True)
            rps = runBench(valsize, pipelining, count, test)
            print(f'        \r{valsize},{test},{rps}                             ')

def memEfficiencyTest(repo, commit_id, sizelist, test, count):
    prettyHeader(f'Memory efficiency testing {repo}:{commit_id} with {human(count)} {test} elements')
    args = ['--io-threads', '9']

    print ('size, total, per key, overhead')
    for valsize in sizelist:
        startServerWithArgs(repo, commit_id, args)

        before_usage = measureUsedMemory()
        # print(f'Valkey is using {humanByte(before_usage)} before')
        # load keys
        print(f'loading {human(count)} {test} {humanByte(valsize)} elements', end='', flush=True)
        loadKeys(valsize, count, 1, test)

        after_usage = measureUsedMemory()
        # print(f'Valkey is using {humanByte(after_usage)} after')

        # output result
        total_usage = after_usage - before_usage
        per_key = float(total_usage) / count
        per_key_overhead = per_key - valsize
        print(f'                  \r{valsize}, {total_usage}, {per_key:.2f}, {per_key_overhead:.2f}')

def parseLazy(lazySpecifier):
    (repo, branch) = lazySpecifier.split(':')
    branch = 'origin/' + branch
    return (repo, branch)

# versions = ['origin/7.2', 'origin/8.0', 'unstable']
# sizelist = list(range(8, 128, 8)) + list(range(128, 544, 32))
# print(sizelist)
# for version in versions:
#     perfTest(version, [], 512, 1)
#     perfTest(version, [], 512, 4)
#     perfTest(version, ['--io-threads', '9'], 512, 1)
#     perfTest(version, ['--io-threads', '9'], 512, 4)
#     memEfficiencyTest(version, sizelist, 'set', 10 * million)

# repolist = ['valkey', 'SoftlyRaining', 'zuiderkwast']
# sizelist = list(range(8, 128, 4)) + [128] + [23, 39, 55, 63, 87]
sizelist = [85]
tests = ['get','set']
sizelist.sort()
for specifier in ['valkey:unstable', 'zuiderkwast:embed-128']:
    (repo, branch) = parseLazy(specifier)
    perfTest(repo, branch, ['--io-threads', '9'], sizelist, 1, tests)
    # memEfficiencyTest(repo, branch, sizelist, 'set', 5 * million)
