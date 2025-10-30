"""Data service layer for TUI to prevent blocking I/O operations."""

from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from src.file_protocol import FileProtocol
from src.task_queue import BaseTaskData, TaskQueue


class TUIDataService:
    """Non-blocking data service for TUI operations with caching."""

    def __init__(self, work_dir: Path = Path("/tmp")):
        self._queue = TaskQueue()
        self._work_dir = work_dir
        self._cached_tasks: list[BaseTaskData] = []
        self._cached_active_tasks: dict = {}
        self._cached_task_statuses: dict = {}
        self._cache_lock = Lock()

    def refresh_all(self) -> tuple[list[BaseTaskData], dict]:
        """Refresh all cached data from disk."""
        tasks = self._queue.get_all_tasks()
        active_tasks = FileProtocol.get_active_task_ids()

        # Refresh statuses for all active tasks
        task_statuses = {}
        for task_id in active_tasks.keys():
            protocol = FileProtocol(task_id, self._work_dir)
            task_statuses[task_id] = protocol.read_status()

        with self._cache_lock:
            self._cached_tasks = tasks
            self._cached_active_tasks = active_tasks
            self._cached_task_statuses = task_statuses

        return tasks, active_tasks

    def get_queue_data(self) -> list[BaseTaskData]:
        """Get cached queue tasks."""
        with self._cache_lock:
            return self._cached_tasks

    def get_active_tasks(self) -> dict:
        """Get cached active task statuses."""
        with self._cache_lock:
            return self._cached_active_tasks

    def get_task_status(self, task_id: str):
        """Get cached task status, re-fetch if stale."""
        with self._cache_lock:
            status = self._cached_task_statuses.get(task_id)

        if status and hasattr(status, "heartbeat") and status.heartbeat:
            heartbeat_dt = datetime.fromtimestamp(status.heartbeat)
            age = datetime.now() - heartbeat_dt
            if age < timedelta(seconds=30):
                return status

        protocol = FileProtocol(task_id, self._work_dir)
        new_status = protocol.read_status()

        with self._cache_lock:
            self._cached_task_statuses[task_id] = new_status

        return new_status

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue."""
        return self._queue.remove_task(task_id)

    def submit_task(self, task: BaseTaskData) -> None:
        """Submit a task to the queue."""
        self._queue.submit_task(task)
