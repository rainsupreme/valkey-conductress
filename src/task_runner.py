"""Main entry point for Conductress, the benchmarking framework for Valkey.
This script runs tasks from a queue, executing performance and memory tests"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Optional

from src.task_queue import BaseTaskRunner

from .config import CONDUCTRESS_LOG, PROJECT_ROOT, get_servers
from .file_protocol import FileProtocol
from .server import Server
from .task_queue import BaseTaskData, TaskQueue

logger = logging.getLogger(__name__)

FAILED_TASKS_LOG = PROJECT_ROOT / "failed_tasks.jsonl"
FAILED_TASKS_DIR = PROJECT_ROOT / "failed"


class TaskRunner:
    """Takes tasks from queue and runs them"""

    def __init__(self) -> None:
        self.task: Optional[BaseTaskData] = None

    async def __run_task(self, task_data: BaseTaskData) -> None:
        """Run a task"""
        servers = get_servers()
        server_count = task_data.replicas + 1 if task_data.replicas > 0 else 1
        if len(servers) < server_count:
            raise RuntimeError(
                f"Not enough servers for {task_data.replicas} replicas. Found {len(servers)} servers."
            )

        task_runner: BaseTaskRunner = task_data.prepare_task_runner(servers[:server_count])
        try:
            await task_runner.run()
            # Task completed successfully, mark as completed and clean up immediately
            task_runner.file_protocol.mark_completed_and_cleanup()
        except Exception:
            # Task failed, just clean up without marking as completed
            task_runner.file_protocol.cleanup()
            raise

    async def run(self):
        """Main function - execute tasks from the queue."""
        # Clean up any orphaned benchmark directories on startup
        cleaned_count = FileProtocol.cleanup_orphaned_tasks()
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} orphaned benchmark directories on startup")
            print(f"Cleaned up {cleaned_count} orphaned benchmark directories on startup")

        # Kill all valkey instances on all servers
        await asyncio.gather(*[Server(server.ip).kill_all_valkey_instances_on_host() for server in get_servers()])

        queue = TaskQueue()
        self.task = queue.get_next_task()
        while True:
            while self.task:
                try:
                    await self.__run_task(self.task)
                except Exception as exc:
                    self._record_failure(self.task, exc)
                queue.finish_task(self.task)
                self.task = queue.get_next_task()
            print("waiting for new jobs in queue")
            while not self.task:
                time.sleep(4)
                self.task = queue.get_next_task()

    def _record_failure(self, task: BaseTaskData, exc: Exception) -> None:
        """Log a task failure to failed_tasks.jsonl and move task file to failed/."""
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        error_msg = f"{type(exc).__name__}: {exc}"

        logger.error("Task failed (note=%s, task_id=%s): %s", task.note, task.task_id, error_msg)

        # Append to failed_tasks.jsonl
        entry = {
            "task_id": task.task_id,
            "note": task.note,
            "source": task.source,
            "specifier": task.specifier,
            "error": error_msg,
            "traceback": "".join(tb),
            "timestamp": datetime.now().isoformat(),
        }
        with open(FAILED_TASKS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Copy task file to failed/ directory for inspection
        FAILED_TASKS_DIR.mkdir(exist_ok=True)
        task_file = FAILED_TASKS_DIR / f"task_{task.task_id}.json"
        task.save_to_file(task_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename=CONDUCTRESS_LOG,
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("asyncssh").setLevel(logging.WARNING)
    runner = TaskRunner()
    asyncio.run(runner.run())

# TODO calculate some error bar metric (std dev.? variance? P95?)
#  - 2 std deviations is +/- 1% for single threaded server! Need to figure out multiple threads now.
# TODO store results in some database?
# TODO fill in perf timeline of specified branch (unstable)
# TODO github action integration
# TODO print or log - choose only one

# TODO make tasks run as separate processes/scripts called by conductress
# TODO write tests for tasks
