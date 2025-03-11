import datetime
import os
import logging
import time

from config import conductress_log, conductress_output
from utility import *
from server import Server
from task_queue import BenchmarkTask, TaskQueue
from perf_test import PerfBench

logger = logging.getLogger(__name__)

def measure_used_memory(s: Server) -> int:
    info = s.info_command('memory')
    return int(info['used_memory'])

def mem_test(repo: str, commit_id: str, valsize: int, test: str) -> float:
    """Tests memory efficiency for 5 million keys of the specified type. Returns bytes of overhead per item."""
    pretty_header(f'Memory efficiency testing {repo}:{commit_id} with {human_byte(valsize)} {test} elements')

    args = ['--io-threads', '9']
    s = Server(repo, commit_id, args)
    commit_hash = s.get_commit_hash()

    before_usage = measure_used_memory()

    count = 5 * million
    print(f'loading {human(count)} {human_byte(valsize)} {test} elements')
    load_sequential_keys(valsize, count, test)

    after_usage = measure_used_memory()

    # output result
    total_usage = after_usage - before_usage
    per_key = float(total_usage) / count
    per_key_overhead = per_key - valsize
    print(f'done testing {human_byte(valsize)} {test} elements: {per_key_overhead:.2f} overhead per key')

    result = {
        'method': 'mem',
        'repo': repo,
        'commit': commit_id,
        'commit_hash': commit_hash,
        'test': test,
        'count': count,
        'endtime': datetime.datetime.now(),
        'size': valsize,
        'per_key_size': per_key,
        'per_key_overhead': per_key_overhead,
    }

    result_fields = 'method repo commit commit_hash test count endtime size per_key_size per_key_overhead'.split()
    result_string = [f'{field}:{result[field]}' for field in result_fields]
    result_string = '\t'.join(result_string) + '\n'
    with open(conductress_output,'a') as f:
        f.write(result_string)
    return per_key_overhead

def parse_lazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

def run_task(task: BenchmarkTask) -> None:
    if task.bench_type == 'perf':
        test_runner = PerfBench(server, task.repo, task.commit_id, io_threads=task.io_threads, valsize=task.val_size, pipelining=task.pipelining, test=task.test, warmup=task.warmup*minute, duration=task.duration*minute)
        test_runner.run()
    if task.bench_type == 'mem':
        # ignored for memory useage test: warmup, duration, threading, pipelining
        mem_test(task.repo, task.commit_id, valsize=task.val_size, test=task.test)
    else:
        logger.error(f'unrecognized benchmark type {task}')

def run_script():
    queue = TaskQueue()

    task = None
    while True:
        while task:
            run_task(task)
            task = queue.get_next_task()
        print("waiting for new jobs in queue")
        while not task:
            time.sleep(4)
            task = queue.get_next_task()

if __name__ == "__main__":
    logging.basicConfig(filename=conductress_log, encoding='utf-8', level=logging.DEBUG)
    run_script()

# TODO perf profiling tests - end goal is flame graphs
# TODO log thread/irq cpu affinity over time
# TODO calculate some error bar metric (std dev.? variance? P95?)
# TODO more instances, more machines, faster results, etc
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration