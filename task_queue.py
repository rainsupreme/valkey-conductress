"""Task queue for benchmark tasks. If run provides a cli for queueing tasks."""

import argparse
import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

import config
from server import Server
from utility import GB

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task for benchmarking"""

    timestamp: str
    task_type: str  # 'perf' or 'mem'
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
            -1,  # profiling not used for sync
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


class QueueStatusApp(App):
    """Textual app to display queue status."""

    # Define keyboard bindings
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the application"),
        # You can also use ctrl+c as an alternative
        Binding(key="ctrl+c", action="quit", description="Quit the application"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Last update: Never")
        yield DataTable()
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.set_interval(5, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh the table data."""
        queue = TaskQueue()
        tasks = queue.get_all_tasks()

        table = self.query_one(DataTable)
        status = self.query_one(Static)

        table.clear()

        if not table.columns:
            table.add_columns(
                "Timestamp",
                "Type",
                "Test",
                "Source:Specifier",
                "Threads",
                "Pipeline",
                "ValSize",
                "Expire",
                "Profiling",
            )

        for task in tasks:
            table.add_row(
                task.timestamp,
                task.task_type,
                task.test,
                f"{task.source}:{task.specifier}",
                str(task.io_threads),
                str(task.pipelining),
                str(task.val_size),
                str(task.has_expire),
                str(task.profiling_sample_rate > 0),
            )

        # Update the status message
        status.update(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    async def action_quit(self):
        """Quit the application."""
        self.exit()


def create_parser() -> ArgumentParser:
    """Sets up command line argument parser"""
    parser = argparse.ArgumentParser(description="Queue benchmark tasks")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Perf command
    perf_parser = subparsers.add_parser("perf", help="Queue performance benchmark task")
    perf_parser.add_argument("--test", type=str, required=True, help="Test name")
    perf_parser.add_argument(
        "--source",
        choices=config.REPO_NAMES + [config.MANUALLY_UPLOADED],
        default="valkey",
        help=f"one of {str(config.REPO_NAMES)} or {config.MANUALLY_UPLOADED} to indicate manual upload",
    )
    perf_parser.add_argument(
        "--specifier",
        type=str,
        required=True,
        help="git specifier or local path if manual upload specified",
    )
    perf_parser.add_argument("--size", type=int, required=True, help="Value size")
    perf_parser.add_argument("--threads", type=int, default=1, help="IO threads")
    perf_parser.add_argument("--pipe", type=int, required=True, help="Pipeline depth")
    perf_parser.add_argument("--warmup", type=int, default=5, help="Warmup duration (minutes)")
    perf_parser.add_argument("--duration", type=int, default=60, help="Test duration (minutes)")
    perf_parser.add_argument(
        "--sample_rate",
        type=int,
        default=-1,
        help="Profiling sample rate (-1 for no profiling)",
    )
    perf_parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload keys before running the test",
    )
    perf_parser.add_argument(
        "--expire",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add expiry data before test",
    )
    perf_parser.add_argument(
        "--replicas",
        type=int,
        default=-1,
        help="Number of replicas (-1 for no replicas)",
    )

    # Mem command
    mem_parser = subparsers.add_parser("mem", help="Queue memory benchmark task")
    mem_parser.add_argument("--test", type=str, required=True, help="Test name")
    mem_parser.add_argument(
        "--source",
        choices=config.REPO_NAMES + [config.MANUALLY_UPLOADED],
        default="valkey",
        help=f"one of {str(config.REPO_NAMES)} or {config.MANUALLY_UPLOADED} to indicate manual upload",
    )
    mem_parser.add_argument(
        "--specifier",
        type=str,
        required=True,
        help="git specifier or local path if manual upload specified",
    )
    mem_parser.add_argument("--size", type=int, required=True, help="Value size")
    mem_parser.add_argument(
        "--expire",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add expiry data before test",
    )

    # Full sync command
    sync_parser = subparsers.add_parser("sync", help="Queue full sync benchmark task")
    sync_parser.add_argument(
        "--source",
        choices=config.REPO_NAMES + [config.MANUALLY_UPLOADED],
        default="valkey",
        help=f"one of {str(config.REPO_NAMES)} or {config.MANUALLY_UPLOADED} to indicate manual upload",
    )
    sync_parser.add_argument(
        "--specifier",
        type=str,
        required=True,
        help="git specifier or local path if manual upload specified",
    )
    sync_parser.add_argument("--threads", type=int, default=1, help="Number of IO threads")
    sync_parser.add_argument("--valsize", type=int, required=True, help="Size of each value in bytes")
    sync_parser.add_argument("--valcount", type=int, required=True, help="Number of values to sync")
    sync_parser.add_argument("--replicas", type=int, default=1, help="Number of replica servers to sync to")

    subparsers.add_parser("status", help="Show queue status")
    subparsers.add_parser("delete_build_cache", help="Delete cached Valkey builds on all servers")
    subparsers.add_parser("rain", help="Rain nonsense, who knows")
    return parser


def main():
    """Main function - parse cli commands"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "status":
        QueueStatusApp().run()
        return
    if args.command == "delete_build_cache":
        print("Deleting all cached builds")
        Server.delete_entire_build_cache(config.SERVERS)
        return
    if args.command == "rain":
        rain()
        return

    if args.command == "perf":
        task = Task.perf_task(
            test=args.test,
            source=args.source,
            specifier=args.specifier,
            val_size=args.size,
            io_threads=args.threads,
            pipelining=args.pipe,
            warmup=args.warmup,
            duration=args.duration,
            profiling_sample_rate=args.sample_rate,
            has_expire=args.expire,
            preload_keys=args.preload,
            replicas=args.replicas,
        )
    elif args.command == "mem":
        task = Task.mem_task(
            test=args.test,
            source=args.source,
            specifier=args.specifier,
            val_size=args.size,
            has_expire=args.expire,
        )
    elif args.command == "sync":
        task = Task.sync_task(
            test="set",
            source=args.source,
            specifier=args.specifier,
            val_size=args.valsize,
            val_count=args.valcount,
            io_threads=args.threads,
            replicas=args.replicas,
        )
    else:
        print(f"Unknown command: {args.command}")
        return

    TaskQueue().submit_task(task)
    print(f"Task queued successfully: {task}")


def rain():
    """Rain nonsense, who knows. I'm lazy"""
    queue = TaskQueue()

    task = Task.sync_task(
        test="set",
        source="valkey",
        specifier="unstable",
        val_size=512,
        val_count=5 * GB // 512 * 2,  # should be enough data for about 2 minutes of syncing
        io_threads=9,
        replicas=1,
    )
    queue.submit_task(task)

    # sources = ["valkey"]
    # preload_keys = [True]
    # # versions = ['7.2','8.0','8.1']
    # specifiers = ["add716b7ddce48d4e13ebffe65401c7d0e26b91a"]
    # pipelining = [4]
    # io_threads = [9]
    # # sizes = [512, 87, 8]
    # sizes = [512]
    # tests = ["set"]

    # # pipelining = [1, 4]
    # # io_threads = [1, 9]
    # # tests = ['get', 'set']

    # # sizes = list(range(8, 256, 8)) + list(range(16, 512+16, 16))
    # # sizes = list(set(sizes))
    # # tests = ['set']
    # # expire_keys = [True, False]
    # expire_keys = [False]

    # all_tests = list(
    #     product(
    #         sizes,
    #         pipelining,
    #         io_threads,
    #         tests,
    #         specifiers,
    #         sources,
    #         preload_keys,
    #         expire_keys,
    #     )
    # )
    # for _ in range(100):
    #     for size, pipe, thread, test, specifier, source, preload, expire in all_tests:
    #         task = Task.perf_task(
    #             test=test,
    #             source=source,
    #             specifier=specifier,
    #             val_size=size,
    #             io_threads=thread,
    #             pipelining=pipe,
    #             warmup=5,
    #             duration=60,
    #             profiling_sample_rate=-1,
    #             has_expire=expire,
    #             preload_keys=preload,
    #             replicas=-1,
    #         )
    #         # task = BenchmarkTask.mem_task(
    #         #     test=test,
    #         #     source=source,
    #         #     specifier=specifier,
    #         #     val_size=size,
    #         #     has_expire=expire,
    #         # )
    #         queue.submit_task(task)
    print("All done 🌧 ♥")


if __name__ == "__main__":
    logging.basicConfig(filename=config.CONDUCTRESS_LOG, encoding="utf-8", level=logging.DEBUG)
    main()
