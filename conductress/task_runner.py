"""Main entry point for Conductress, the benchmarking framework for Valkey.
This script runs tasks from a queue, executing performance and memory tests"""

import logging
import time
from threading import Thread
from typing import Optional

from .config import CONDUCTRESS_LOG, SERVERS
from .task_full_sync import TestFullSync
from .task_mem_efficiency import TestMem
from .task_perf_benchmark import TestPerf
from .task_queue import Task, TaskQueue
from .utility import MINUTE

logger = logging.getLogger(__name__)


class TaskRunner:
    """Takes tasks from queue and runs them"""

    def __init__(self) -> None:
        self.task: Optional[Task] = None
        self.thread = Thread(target=self.__run)
        self.thread.start()

    def __run_task(self, task: Task) -> None:
        """Run a task"""
        server_count = task.replicas + 1 if task.replicas > 0 else 1
        assert (
            len(SERVERS) >= server_count
        ), f"Not enough servers for {task.replicas} replicas. Found {len(SERVERS)} servers."

        if task.task_type == "perf":
            perf_test_runner = TestPerf(
                f"{task.timestamp}_{task.test}_{task.task_type}",
                SERVERS[:server_count],
                task.source,
                task.specifier,
                io_threads=task.io_threads,
                valsize=task.val_size,
                pipelining=task.pipelining,
                test=task.test,
                warmup=task.warmup * MINUTE,
                duration=task.duration * MINUTE,
                preload_keys=task.preload_keys,
                has_expire=task.has_expire,
                sample_rate=task.profiling_sample_rate,
            )
            perf_test_runner.run()
        elif task.task_type == "mem":
            # ignored for memory usage test:
            # warmup, duration, threading, pipelining, preload, profiling_sample_rate
            mem_tester = TestMem(
                SERVERS[0],
                task.source,
                task.specifier,
                task.test,
                task.has_expire,
            )
            mem_tester.test_single_size(task.val_size)
        elif task.task_type == "sync":
            # ignored for full sync test:
            # warmup, duration, threading, pipelining, preload, profiling_sample_rate
            full_sync_tester = TestFullSync(
                f"{task.timestamp}_{task.test}_{task.task_type}",
                SERVERS[:server_count],
                task.source,
                task.specifier,
                io_threads=task.io_threads,
                valsize=task.val_size,
                valcount=task.keyspace,
                profiling_sample_rate=task.profiling_sample_rate,
            )
            full_sync_tester.run()
        else:
            logger.error("unrecognized benchmark type %s", task.task_type)

    def __run(self):
        """Main function - execute tasks from the queue."""
        queue = TaskQueue()
        self.task = queue.get_next_task()
        while True:
            while self.task:
                self.__run_task(self.task)
                queue.finish_task(self.task)
                self.task = queue.get_next_task()
            print("waiting for new jobs in queue")
            while not self.task:
                time.sleep(4)
                self.task = queue.get_next_task()


if __name__ == "__main__":
    logging.basicConfig(filename=CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    runner = TaskRunner()
    runner.thread.join()  # wait for thread to exit

# TODO log thread/irq cpu affinity over time
# TODO calculate some error bar metric (std dev.? variance? P95?)
#  - 2 std deviations is +/- 1% for single threaded server! Need to figure out multiple threads now.
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration
# TODO print or log - choose only one
# TODO add setup.py?
