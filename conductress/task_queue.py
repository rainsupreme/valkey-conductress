"""Task queue for benchmark tasks. If run provides a cli for queueing tasks."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task for benchmarking"""

    timestamp: str
    task_type: str  # 'perf', 'mem', 'sync'
    test: str
    source: str
    specifier: str
    val_size: int
    io_threads: int
    pipelining: int
    warmup: int
    duration: int
    profiling_sample_rate: int
    has_expire: bool
    preload_keys: bool
    replicas: int

    def to_json(self) -> str:
        """Convert the task to a JSON string."""
        return json.dumps(asdict(self))

    def __init__(
        self,
        task_type: str,
        timestamp: str,
        test: str,
        source: str,
        specifier: str,
        val_size: int,
        io_threads: int,
        pipelining: int,
        warmup: int,
        duration: int,
        profiling_sample_rate: int,
        has_expire: bool,
        preload_keys: bool,
        replicas: int,
        keyspace: int,
    ) -> None:
        self.task_type = task_type
        self.timestamp = timestamp
        self.test = test
        self.source = source
        self.specifier = specifier
        self.val_size = val_size
        self.io_threads = io_threads
        self.pipelining = pipelining
        self.warmup = warmup
        self.duration = duration
        self.profiling_sample_rate = profiling_sample_rate
        self.has_expire = has_expire
        self.preload_keys = preload_keys
        self.replicas = replicas
        self.keyspace = keyspace

        assert self.source == config.MANUALLY_UPLOADED or self.source in config.REPO_NAMES

    @staticmethod
    def perf_task(
        test: str,
        source: str,
        specifier: str,
        val_size: int,
        io_threads: int,
        pipelining: int,
        warmup: int,
        duration: int,
        profiling_sample_rate: int,
        has_expire: bool,
        preload_keys: bool,
        replicas: int,
    ) -> "Task":
        """Create a performance task"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Task(
            "perf",
            timestamp,
            test,
            source,
            specifier,
            val_size,
            io_threads,
            pipelining,
            warmup,
            duration,
            profiling_sample_rate,
            has_expire,
            preload_keys,
            replicas,
            -1,  # keyspace not used for perf tasks
        )

    @staticmethod
    def mem_task(source: str, specifier: str, val_size: int, test: str, has_expire: bool) -> "Task":
        """Create a memory efficiency task"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Task(
            "mem",
            timestamp,
            test,
            source,
            specifier,
            val_size,
            -1,  # io_threads not used for mem tasks
            -1,  # pipelining not used for mem tasks
            -1,  # warmup not used for mem tasks
            -1,  # duration not used for mem tasks
            -1,  # profiling not used for mem tasks
            has_expire,
            True,
            -1,  # replicas not used for mem tasks
            -1,  # keyspace not used for mem tasks
        )

    @staticmethod
    def sync_task(
        test: str,
        source: str,
        specifier: str,
        val_size: int,
        val_count: int,
        io_threads: int,
        replicas: int,
        profiling_sample_rate: int,
    ) -> "Task":
        """Create a full sync task"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Task(
            "sync",
            timestamp,
            test,
            source,
            specifier,
            val_size,
            io_threads,
            -1,  # pipelining not used for sync
            -1,  # warmup not used for sync
            -1,  # duration not used for sync
            profiling_sample_rate,
            False,  # expire not used for sync
            True,  # preload always true for sync
            replicas,
            val_count,  # use val_count as keyspace
        )

    @classmethod
    def from_file(cls, filepath: Path) -> "Task":
        """Load a task from a JSON file"""
        try:
            with filepath.open("r") as f:
                data = json.load(f)
            task = cls(**data)
            if f"task_{task.timestamp}" != filepath.stem:
                raise ValueError(f"Invalid task file name: {filepath.stem}")
            return cls(**data)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Task file not found: {filepath}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in file: {filepath}") from exc

    def save_to_file(self, filepath: Path):
        """Save the task to a JSON file"""
        with filepath.open("w") as f:
            json.dump(self.__dict__, f, indent=2)


class TaskQueue:
    """Task queue for benchmark tasks"""

    def __init__(self, queue_dir="./benchmark_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def submit_task(self, task: Task) -> None:
        """Add a new task to the queue"""
        task_file = self.queue_dir / f"task_{task.timestamp}.json"
        task.save_to_file(task_file)

    def get_next_task(self) -> Optional[Task]:
        """Get the next task from the queue"""
        tasks = sorted(self.queue_dir.glob("task_*.json"))
        if not tasks:
            return None

        task_file = tasks[0]
        try:
            task = Task.from_file(task_file)
            return task
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted task files
            if task_file.exists():
                logger.error("unable to read - skipping %s", task_file)
                task_file.unlink()
            return None

    def finish_task(self, task: Task) -> None:
        """Delete a task from the queue, indicating it has been completed"""
        task_file = self.queue_dir / f"task_{task.timestamp}.json"
        if task_file.exists():
            task_file.unlink()
        else:
            print(f"Unable to delete task {task_file}")
            logger.error("Task file not found: %s", task_file)

    def get_all_tasks(self) -> list[Task]:
        """Returns list of (timestamp, task) tuples, sorted by timestamp"""
        tasks = []
        for task_file in self.queue_dir.glob("task_*.json"):
            try:
                task = Task.from_file(task_file)
                tasks.append(task)
            except (ValueError, json.JSONDecodeError, FileNotFoundError):
                continue

        return sorted(tasks, key=lambda x: x.timestamp)

    def get_queue_length(self) -> int:
        """Get the number of tasks in the queue"""
        return len(list(self.queue_dir.glob("task_*.json")))
