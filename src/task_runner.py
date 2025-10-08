"""Main entry point for Conductress, the benchmarking framework for Valkey.
This script runs tasks from a queue, executing performance and memory tests"""

import asyncio
import logging
import time
from threading import Thread
from typing import Optional

from .config import CONDUCTRESS_LOG, SERVERS
from .task_queue import BaseTaskData, TaskQueue

logger = logging.getLogger(__name__)


class TaskRunner:
    """Takes tasks from queue and runs them"""

    def __init__(self) -> None:
        self.task: Optional[BaseTaskData] = None

    async def __run_task(self, task_data: BaseTaskData) -> None:
        """Run a task"""
        server_count = task_data.replicas + 1 if task_data.replicas > 0 else 1
        assert (
            len(SERVERS) >= server_count
        ), f"Not enough servers for {task_data.replicas} replicas. Found {len(SERVERS)} servers."

        task_runner = task_data.prepare_task_runner(SERVERS[:server_count])
        await task_runner.run()

    async def run(self):
        """Main function - execute tasks from the queue."""
        queue = TaskQueue()
        self.task = queue.get_next_task()
        while True:
            while self.task:
                await self.__run_task(self.task)
                queue.finish_task(self.task)
                self.task = queue.get_next_task()
            print("waiting for new jobs in queue")
            while not self.task:
                time.sleep(4)
                self.task = queue.get_next_task()


if __name__ == "__main__":
    logging.basicConfig(filename=CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    runner = TaskRunner()
    asyncio.run(runner.run())

# TODO calculate some error bar metric (std dev.? variance? P95?)
#  - 2 std deviations is +/- 1% for single threaded server! Need to figure out multiple threads now.
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration
# TODO print or log - choose only one

# TODO make tasks run as separate processes/scripts called by conductress
# TODO log status and data as files
# TODO write tests for tasks
