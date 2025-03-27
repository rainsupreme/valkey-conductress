import os
import logging
import time

from config import conductress_log
from utility import server, minute
from task_queue import BenchmarkTask, TaskQueue
from perf_test import PerfBench
from mem_test import MemBench

logger = logging.getLogger(__name__)



def parse_lazy(lazySpecifier: str) -> tuple[str, str]:
    (repo, branch) = lazySpecifier.split(':')
    return (repo, branch)

def run_task(task: BenchmarkTask) -> None:
    if task.bench_type == 'perf':
        test_runner = PerfBench(
            server,
            task.repo,
            task.commit_id,
            io_threads=task.io_threads,
            valsize=task.val_size,
            pipelining=task.pipelining,
            test=task.test,
            warmup=task.warmup*minute,
            duration=task.duration*minute,
            preload_keys=task.preload_keys,
            has_expire=task.has_expire,
            )
        test_runner.run()
    if task.bench_type == 'mem':
        # ignored for memory useage test: warmup, duration, threading, pipelining, preload
        test_runner = MemBench(
            server,
            task.repo,
            task.commit_id,
            task.test,
            task.has_expire,
            )
        test_runner.test_single_size(task.val_size)
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
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration