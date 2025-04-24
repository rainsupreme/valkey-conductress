import argparse
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path

from config import conductress_log

logger = logging.getLogger(__name__)

@dataclass
class Task:
    timestamp: str
    type: str  # 'perf' or 'mem'
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

    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    def __init__(self, type: str, timestamp: str, test: str, source: str, specifier: str, val_size: int, io_threads: int, pipelining: int, warmup: int, duration: int, profiling_sample_rate: int, has_expire: bool, preload_keys: bool) -> 'Task':
        self.type = type
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

        assert self.source in ['manually_uploaded', 'valkey', 'SoftlyRaining', 'zuiderkwast', 'JimB123']

    @staticmethod
    def perf_task(test: str, source: str, specifier: str, val_size: int, io_threads: int, pipelining: int, warmup: int, duration: int, profiling_sample_rate: int, has_expire: bool, preload_keys: bool) -> 'Task':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Task('perf', timestamp, test, source, specifier, val_size, io_threads, pipelining, warmup, duration, profiling_sample_rate, has_expire, preload_keys)

    @staticmethod
    def mem_task(source: str, specifier: str, val_size: int, test: str, has_expire: bool) -> 'Task':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Task('mem', timestamp, test, source, specifier, val_size, -1, -1, -1, -1, -1, has_expire, True)

    @classmethod
    def from_file(cls, filepath: Path) -> 'Task':
        try:
            with filepath.open('r') as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Task file not found: {filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {filepath}")

    def save_to_file(self, filepath: Path):
        with filepath.open('w') as f:
            json.dump(self.__dict__, f, indent=2)

class TaskQueue:
    def __init__(self, queue_dir="./benchmark_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
    def submit_task(self, task: Task):
        """Add a new task to the queue"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_file = self.queue_dir / f"task_{timestamp}.json"
        
        task.save_to_file(task_file)

    def get_next_task(self) -> Task:
        """Get the next task from the queue"""
        tasks = sorted(self.queue_dir.glob("task_*.json"))
        if not tasks:
            return None
            
        task_file = tasks[0]
        try:
            task = Task.from_file(task_file)
            task_file.unlink()  # Remove the task file after reading
            return task
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted or disappeared task files
            if task_file.exists():
                logger.error(f'unable to read - skipping {task_file}')
                task_file.unlink()
            return None

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
        return len(list(self.queue_dir.glob("task_*.json")))

def create_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(description='Queue benchmark tasks')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Perf command
    perf_parser = subparsers.add_parser('perf', help='Queue performance benchmark task')
    perf_parser.add_argument('--test', type=str, required=True, help='Test name')
    perf_parser.add_argument('--source', type=str, default="valkey", help='Repository or "manually_uploaded"')
    perf_parser.add_argument('--specifier', type=str, required=True, help='git specifier or local path if manual upload specified')
    perf_parser.add_argument('--size', type=int, required=True, help='Value size')
    perf_parser.add_argument('--threads', type=int, required=True, help='IO threads')
    perf_parser.add_argument('--pipe', type=int, required=True, help='Pipeline depth')
    perf_parser.add_argument('--warmup', type=int, default=5, help='Warmup duration (minutes)')
    perf_parser.add_argument('--duration', type=int, default=60, help='Test duration (minutes)')
    perf_parser.add_argument('--sample_rate', type=int, default=-1, help='Profiling sample rate (-1 for no profiling)')
    perf_parser.add_argument('--preload', action=argparse.BooleanOptionalAction, default=True, help='Preload keys before running the test')
    perf_parser.add_argument('--expire', action=argparse.BooleanOptionalAction, default=False, help='Add expiry data before test')

    # Mem command
    mem_parser = subparsers.add_parser('mem', help='Queue memory benchmark task')
    mem_parser.add_argument('--test', type=str, required=True, help='Test name')
    mem_parser.add_argument('--source', type=str, default="valkey", help='Repository or "manually_uploaded"')
    mem_parser.add_argument('--specifier', type=str, required=True, help='git specifier or local path if manual upload specified')
    mem_parser.add_argument('--size', type=int, required=True, help='Value size')
    mem_parser.add_argument('--expire', action=argparse.BooleanOptionalAction, default=False, help='Add expiry data before test')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show queue status')

    # Add a temp command just for me
    rain_parser = subparsers.add_parser('rain', help='Rain nonsense, who knows')
    return parser

def show_status(queue: TaskQueue):
    tasks = queue.get_all_tasks()
    
    if not tasks:
        print("Queue is empty")
        return

    print(f"\nQueue Status: {len(tasks)} tasks pending\n")
    print("Pending Tasks:")
    print(f"{'Timestamp':<25}{'Type':<10}{'Test':<10}{'Source:Specifier':<20}{'Threads':<10}{'Pipeline':<10}{'ValSize':<10}{'Expire':<10}{'Profiling':<10}")
    print("-" * 114)
    for task in tasks:
        print(f"{task.timestamp:<25}{task.type:<10}{task.test:<10}{f'{task.source}:{task.specifier}':<20}{task.io_threads:<10}{task.pipelining:<10}{task.val_size:<10}{str(task.has_expire):<10}{str(task.profiling_sample_rate>0):<10}")

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    queue = TaskQueue()

    if args.command == 'status':
        show_status(queue)
        return
    if args.command == 'rain':
        rain()
        return

    if args.command == 'perf':
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
        )
    elif args.command == 'mem':
        task = Task.mem_task(
            test=args.test,
            source=args.source,
            specifier=args.specifier,
            val_size=args.size,
            has_expire=args.expire,
        )
    else:
        print(f"Unknown command: {args.command}")
        return

    queue.submit_task(task)
    print(f"Task queued successfully: {task}")

from itertools import product
def rain():
    queue = TaskQueue()

    # available_repos = ['valkey', 'SoftlyRaining', 'zuiderkwast', 'JimB123']

    sources = ['valkey']
    preload_keys = [True]
    # versions = ['7.2','8.0','8.1']
    specifiers = ['add716b7ddce48d4e13ebffe65401c7d0e26b91a']
    pipelining = [4]
    io_threads = [9]
    # sizes = [512, 87, 8]
    sizes = [512]
    tests = ['set']

    # pipelining = [1, 4]
    # io_threads = [1, 9]
    # tests = ['get', 'set']

    # sizes = list(range(8, 256, 8)) + list(range(16, 512+16, 16))
    # sizes = list(set(sizes))
    # tests = ['set']
    # expire_keys = [True, False]
    expire_keys = [False]

    all_tests = list(product(sizes, pipelining, io_threads, tests, specifiers, sources, preload_keys, expire_keys))
    for i in range(100):
        for (size, pipe, thread, test, specifier, source, preload, expire) in all_tests:
            task = Task.perf_task(
                test=test,
                source=source,
                specifier=specifier,
                val_size=size,
                io_threads=thread,
                pipelining=pipe,
                warmup=5,
                duration=60,
                profiling_sample_rate=-1,
                has_expire=expire,
                preload_keys=preload,
            )
            # task = BenchmarkTask.mem_task(
            #     test=test,
            #     source=source,
            #     specifier=specifier,
            #     val_size=size,
            #     has_expire=expire,
            # )
            queue.submit_task(task)
    print('\n\nAll done 🌧 ♥')

if __name__ == "__main__":
    logging.basicConfig(filename=conductress_log, encoding='utf-8', level=logging.DEBUG)
    main()
