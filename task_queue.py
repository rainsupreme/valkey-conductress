import argparse
from argparse import ArgumentParser
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkTask:
    bench_type: str # 'perf' or 'mem'
    test: str
    repo: str
    commit_id: str
    val_size: int
    io_threads: int
    pipelining: int
    warmup: int
    duration: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    def __init__(self, bench_type, test, repo, commit_id, val_size, io_threads, pipelining, warmup, duration) -> 'BenchmarkTask':
        self.bench_type = bench_type
        self.test = test
        self.repo = repo
        self.commit_id = commit_id
        self.val_size = val_size
        self.io_threads = io_threads
        self.pipelining = pipelining
        self.warmup = warmup
        self.duration = duration

    @staticmethod
    def mem_task(repo: str, commit_id: str, val_size: int, test: str) -> 'BenchmarkTask':
        return BenchmarkTask('mem', test, repo, commit_id, val_size, -1, -1, -1, -1)

    @classmethod
    def from_file(cls, filepath: Path) -> 'BenchmarkTask':
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
        
    def submit_task(self, task: BenchmarkTask):
        """Add a new task to the queue"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_file = self.queue_dir / f"task_{timestamp}.json"
        
        task.save_to_file(task_file)

    def get_next_task(self) -> BenchmarkTask:
        """Get the next task from the queue"""
        tasks = sorted(self.queue_dir.glob("task_*.json"))
        if not tasks:
            return None
            
        task_file = tasks[0]
        try:
            task = BenchmarkTask.from_file(task_file)
            task_file.unlink()  # Remove the task file after reading
            return task
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted or disappeared task files
            if task_file.exists():
                logger.error(f'unable to read - skipping {task_file}')
                task_file.unlink()
            return None

    def get_all_tasks(self) -> list[tuple[datetime, BenchmarkTask]]:
        """Returns list of (timestamp, task) tuples, sorted by timestamp"""
        tasks = []
        for task_file in self.queue_dir.glob("task_*.json"):
            try:
                # Parse timestamp from filename
                timestamp_str = task_file.name[5:-5]  # Remove "task_" and ".json"
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
                
                # Load task
                task = BenchmarkTask.from_file(task_file)
                tasks.append((timestamp, task))
            except (ValueError, json.JSONDecodeError, FileNotFoundError):
                continue
        
        return sorted(tasks, key=lambda x: x[0])

    def get_queue_length(self) -> int:
        return len(list(self.queue_dir.glob("task_*.json")))

def create_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(description='Queue benchmark tasks')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Add perf command
    perf_parser = subparsers.add_parser('perf', help='Queue performance benchmark task')
    perf_parser.add_argument('--test', required=True, help='Test name')
    perf_parser.add_argument('--repo', required=True, help='Repository')
    perf_parser.add_argument('--commit', required=True, help='Commit ID')
    perf_parser.add_argument('--size', type=int, required=True, help='Value size')
    perf_parser.add_argument('--threads', type=int, required=True, help='IO threads')
    perf_parser.add_argument('--pipelining', type=int, required=True, help='Pipelining factor')
    perf_parser.add_argument('--warmup', type=int, required=True, help='Warmup duration')
    perf_parser.add_argument('--duration', type=int, required=True, help='Test duration')

    # Add mem command
    mem_parser = subparsers.add_parser('mem', help='Queue memory benchmark task')
    mem_parser.add_argument('--test', required=True, help='Test name')
    mem_parser.add_argument('--repo', required=True, help='Repository')
    mem_parser.add_argument('--commit', required=True, help='Commit ID')
    mem_parser.add_argument('--size', type=int, required=True, help='Value size')

    # Add status command
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
    print("Timestamp\t\t\ttype\ttest\trepo:commit\tthreads\tpipel.\tval size")
    for (timestamp, task) in tasks:
        print(f'{timestamp}\t{task.bench_type}\t{task.test}\t{task.repo}:{task.commit_id}\t{task.io_threads}\t{task.pipelining}\t{task.val_size}')

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
        task = BenchmarkTask(
            bench_type='perf',
            test=args.test,
            repo=args.repo,
            commit_id=args.commit,
            val_size=args.size,
            io_threads=args.threads,
            pipelining=args.pipelining,
            warmup=args.warmup,
            duration=args.duration
        )
    elif args.command == 'mem':
        task = BenchmarkTask.mem_task(
            repo=args.repo,
            commit_id=args.commit,
            val_size=args.size,
            test=args.test
        )

    queue.submit_task(task)
    print(f"Task queued successfully: {task}")

from itertools import product
def rain():
    # sizelist = list(range(24, 96, 8)) + list(range(23, 95, 8))
    # sizelist.sort()
    # print(len(sizelist), 'sizes', sizelist)
    # tests = ['get','set']
    # for specifier in ['valkey:unstable', 'zuiderkwast:embed-128']:
    #     (repo, branch) = parseLazy(specifier)
    #     # perfTest(repo, branch, ['--io-threads', '9'], sizelist, 1, tests)
    #     memEfficiencyTest(repo, branch, sizelist, 'set', 5 * million)

    # repolist = ['valkey', 'SoftlyRaining', 'zuiderkwast']

    # sizes = [512]
    # configs = [(True, 4), (True, 1), (False, 4)]
    # tests = ['set','get','sadd','hset','zadd','zrange']
    # versions = ['valkey:7.2', 'valkey:8.0', 'valkey:unstable']

    queue = TaskQueue()

    sizes = [512, 87, 8]
    pipelining = [1, 4]
    io_threads = [1, 9]
    tests = ['set','get']
    versions = ['8.0','unstable']
    repos = ['valkey']

    all_tests = list(product(sizes, pipelining, io_threads, tests, versions, repos))
    for (size, pipe, thread, test, version, repo) in all_tests:
        if pipe==1 and thread==9 and test=='get':
            task = BenchmarkTask(
                bench_type='perf',
                test=test,
                repo=repo,
                commit_id=version,
                val_size=size,
                io_threads=thread,
                pipelining=pipe,
                warmup=5,
                duration=60
            )
            queue.submit_task(task)
    print('\n\nAll done 🌧 ♥')

if __name__ == "__main__":
    main()
