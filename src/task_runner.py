"""Main entry point for Conductress, the benchmarking framework for Valkey.
This script runs tasks from a queue, executing performance and memory tests"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.task_queue import BaseTaskRunner

from .config import CONDUCTRESS_FAILED_DIR, CONDUCTRESS_FAILED_LOG, CONDUCTRESS_LOG, get_servers
from .file_protocol import FileProtocol
from .server import Server
from .task_queue import BaseTaskData, TaskQueue

logger = logging.getLogger(__name__)


class TaskRunner:
    """Takes tasks from queue and runs them"""

    def __init__(self, sweep: bool = False, repo_path: Optional[Path] = None) -> None:
        self.task: Optional[BaseTaskData] = None
        self.sweep_enabled = sweep
        self.sweep_runner: Optional["SweepRunner"] = None  # type: ignore[name-defined]
        self._sweep_commit: Optional[str] = None  # tracks current sweep task's commit
        if sweep:
            from src.sweep.runner import SweepRunner
            if repo_path is None:
                repo_path = Path.home() / "valkey"
            self.sweep_runner = SweepRunner(repo_path)
            self.sweep_runner.initialize()

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

    def _get_sweep_task(self) -> Optional[BaseTaskData]:
        """Ask the sweep planner for the next task when queue is empty."""
        if not self.sweep_runner:
            return None
        task = self.sweep_runner.get_next_sweep_task()
        if task:
            self._sweep_commit = task.specifier  # commit hash
            print(f"[sweep] Next: {task.specifier[:8]} - {task.note}")
        return task

    def _handle_sweep_completion(self, task_data: BaseTaskData) -> None:
        """After a sweep task completes, record results and clean up."""
        if not self.sweep_runner or not self._sweep_commit:
            return

        commit = self._sweep_commit
        self._sweep_commit = None

        # Read the result from the file protocol output
        rps, cv = self._extract_result(task_data)
        if rps is not None and cv is not None:
            from src.tasks.task_perf_benchmark import PerfTaskData
            assert isinstance(task_data, PerfTaskData)
            self.sweep_runner.record_result(commit, rps, cv, task_data.repetitions)
            self.sweep_runner.delete_cached_binary(commit)
        else:
            logger.warning("Could not extract result for sweep commit %s", commit[:8])

    def _handle_sweep_failure(self) -> None:
        """Record a build/run failure for the current sweep task."""
        if not self.sweep_runner or not self._sweep_commit:
            return
        self.sweep_runner.record_build_failure(self._sweep_commit)
        self.sweep_runner.delete_cached_binary(self._sweep_commit)
        self._sweep_commit = None

    def _extract_result(self, task_data: BaseTaskData) -> tuple:
        """Extract RPS and CV from the most recent completed task's output."""
        from src.config import CONDUCTRESS_RESULTS
        from src.file_protocol import FileProtocol
        import json

        # Find the most recent result file for this task
        results_dir = CONDUCTRESS_RESULTS
        if not results_dir.exists():
            return None, None

        # Look for the task's output in the JSONL file
        output_file = results_dir / "output.jsonl"
        if not output_file.exists():
            return None, None

        # Read the last entry that matches our task
        last_rps = None
        last_cv = None
        try:
            lines = output_file.read_text().strip().splitlines()
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    if data.get("task_id") == task_data.task_id:
                        last_rps = data.get("mean_rps") or data.get("rps")
                        # CV from stdev/mean
                        if "stdev_rps" in data and last_rps:
                            last_cv = (data["stdev_rps"] / last_rps) * 100
                        elif "cv_pct" in data:
                            last_cv = data["cv_pct"]
                        else:
                            last_cv = 0.0
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            logger.warning("Error reading results: %s", e)

        return last_rps, last_cv

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
                is_sweep = self._sweep_commit is not None
                try:
                    await self.__run_task(self.task)
                except Exception as exc:
                    if is_sweep:
                        logger.error("Sweep task failed: %s", exc)
                        self._handle_sweep_failure()
                        self.task = queue.get_next_task()
                        continue
                    self._record_failure(self.task, exc)

                if is_sweep:
                    self._handle_sweep_completion(self.task)
                else:
                    queue.finish_task(self.task)
                self.task = queue.get_next_task()

            # Queue empty — try sweep if enabled
            if self.sweep_enabled:
                self.task = self._get_sweep_task()
                if self.task:
                    continue

            print("waiting for new jobs in queue")
            while not self.task:
                time.sleep(4)
                self.task = queue.get_next_task()
                if not self.task and self.sweep_enabled:
                    self.task = self._get_sweep_task()

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
        with open(CONDUCTRESS_FAILED_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Copy task file to failed/ directory for inspection
        CONDUCTRESS_FAILED_DIR.mkdir(exist_ok=True)
        task_file = CONDUCTRESS_FAILED_DIR / f"task_{task.task_id}.json"
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
