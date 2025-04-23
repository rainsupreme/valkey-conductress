import os
import logging
import time

from config import conductress_log
from utility import server, minute
from task_queue import Task, TaskQueue
from perf_test import PerfBench
from mem_test import MemBench

logger = logging.getLogger(__name__)

def parse_lazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

def run_task(task: Task) -> None:
    if task.type == 'perf':
        perf_test_runner = PerfBench(
            f'{task.timestamp}_{task.test}_{task.type}',
            server,
            task.repo,
            task.specifier,
            io_threads=task.io_threads,
            valsize=task.val_size,
            pipelining=task.pipelining,
            test=task.test,
            warmup=task.warmup*minute,
            duration=task.duration*minute,
            preload_keys=task.preload_keys,
            has_expire=task.has_expire,
            sample_rate=task.profiling_sample_rate,
        )
        perf_test_runner.run()
    elif task.type == 'mem':
        # ignored for memory usage test: warmup, duration, threading, pipelining, preload, profiling_sample_rate
        mem_tester = MemBench(
            server,
            task.repo,
            task.specifier,
            task.test,
            task.has_expire,
        )
        mem_tester.test_single_size(task.val_size)
    else:
        logger.error(f'unrecognized benchmark type {task.type}')

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
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration