"""Non-interactive CLI for queuing and managing benchmark tasks."""

import argparse
import itertools
import logging
import sys
from typing import List, Optional, Tuple

from . import config
from .task_queue import TaskQueue
from .tasks.task_perf_benchmark import PerfTaskData
from .utility import HumanByte, HumanTime

logger = logging.getLogger(__name__)


def validate_source(source: str) -> bool:
    """Check whether a source string is a recognized repository name or manually uploaded."""
    return source in config.REPO_NAMES or source == config.MANUALLY_UPLOADED


def generate_task_combinations(
    tests: List[str],
    sizes: List[int],
    io_threads: List[int],
    pipelining: List[int],
    key_sizes: List[int],
) -> List[Tuple[str, int, int, int, int]]:
    """Compute the Cartesian product of multi-valued benchmark parameters."""
    return list(itertools.product(tests, sizes, io_threads, pipelining, key_sizes))


def _parse_comma_separated_ints(value: str, name: str) -> List[int]:
    """Parse a comma-separated string of integers."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--{name} cannot be empty")
    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid integer in --{name}: '{part}'")
    return result


def _parse_comma_separated_bytes(value: str, name: str) -> List[int]:
    """Parse a comma-separated string of human-readable byte values."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--{name} cannot be empty")
    result = []
    for part in parts:
        try:
            val = HumanByte.from_human(part)
            result.append(int(val))
        except ValueError:
            raise ValueError(f"Invalid byte value in --{name}: '{part}'")
    return result


def _parse_human_time(value: str, name: str) -> int:
    """Parse a human-readable time value."""
    try:
        return int(HumanTime.from_human(value))
    except ValueError:
        raise ValueError(f"Invalid time value for --{name}: '{value}'")


def _parse_tests(value: str) -> List[str]:
    """Parse a comma-separated list of test names."""
    tests = [t.strip() for t in value.split(",") if t.strip()]
    if not tests:
        raise ValueError("No tests specified")
    return tests


def _add_perf_args(parser: argparse.ArgumentParser) -> None:
    """Add performance benchmark arguments to a parser."""
    parser.add_argument("--source", default="valkey", help="Repository source name (default: valkey)")
    parser.add_argument("--specifier", default="unstable", help="Branch, tag, or commit (default: unstable)")
    parser.add_argument(
        "--tests", required=True, help="Comma-separated test names (e.g., get,set)"
    )
    parser.add_argument(
        "--sizes", default=str(config.DEFAULT_VAL_SIZE),
        help=f"Comma-separated value sizes (e.g., 16,512,1KB). Default: {config.DEFAULT_VAL_SIZE}"
    )
    parser.add_argument(
        "--io-threads", default=str(config.DEFAULT_IO_THREADS),
        help=f"Comma-separated IO thread counts (e.g., 1,9). Default: {config.DEFAULT_IO_THREADS}"
    )
    parser.add_argument(
        "--pipelining", default=str(config.DEFAULT_PIPELINING),
        help=f"Comma-separated pipelining values (e.g., 1,4,10). Default: {config.DEFAULT_PIPELINING}"
    )
    parser.add_argument(
        "--warmup", default=f"{config.DEFAULT_WARMUP}s",
        help=f"Warmup duration (e.g., 30s, 1m). Default: {config.DEFAULT_WARMUP}s"
    )
    parser.add_argument(
        "--duration", default=f"{config.DEFAULT_DURATION}s",
        help=f"Test duration (e.g., 5m, 15m). Default: {config.DEFAULT_DURATION}s"
    )
    parser.add_argument(
        "--repetitions", type=int, default=config.DEFAULT_REPETITIONS,
        help=f"Number of repetitions per config. Default: {config.DEFAULT_REPETITIONS}"
    )
    parser.add_argument(
        "--key-sizes", default=str(config.DEFAULT_KEY_SIZE),
        help=f"Comma-separated key sizes in bytes (0=standard). Default: {config.DEFAULT_KEY_SIZE}"
    )
    parser.add_argument("--note", default="", help="Optional note for the tasks")
    parser.add_argument(
        "--make-args", default=config.DEFAULT_MAKE_ARGS,
        help=f"Build arguments. Default: '{config.DEFAULT_MAKE_ARGS}'"
    )
    parser.add_argument(
        "--perf-stat", action="store_true", help="Enable perf stat hardware counter collection"
    )
    parser.add_argument(
        "--no-preload", action="store_true", help="Disable key preloading"
    )
    perf_parser.add_argument(
        "--perf-stat", action="store_true", default=False,
        help="Collect hardware performance counters (perf stat) during benchmark"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="conductress",
        description="Conductress CLI for queuing and managing benchmark tasks",
    )
    subparsers = parser.add_subparsers(dest="command")

    # queue subcommand with its own subcommands
    queue_parser = subparsers.add_parser("queue", help="Manage the task queue")
    queue_sub = queue_parser.add_subparsers(dest="queue_command")

    # queue list
    queue_sub.add_parser("list", help="List all pending tasks")

    # queue add
    add_parser = queue_sub.add_parser("add", help="Add performance benchmark tasks to the queue")
    _add_perf_args(add_parser)

    # queue remove
    remove_parser = queue_sub.add_parser("remove", help="Remove a task from the queue")
    remove_parser.add_argument("task_id", help="Task ID to remove (from 'queue list' output)")

    # queue clear
    queue_sub.add_parser("clear", help="Remove all pending tasks from the queue")

    return parser


def handle_queue_add(args: argparse.Namespace) -> int:
    """Handle 'queue add': validate inputs, generate tasks, and submit them."""
    if not validate_source(args.source):
        valid_sources = config.REPO_NAMES + [config.MANUALLY_UPLOADED]
        print(
            f"Error: Invalid source '{args.source}'. "
            f"Valid sources: {', '.join(valid_sources)}",
            file=sys.stderr,
        )
        return 1

    try:
        tests = _parse_tests(args.tests)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        sizes = _parse_comma_separated_bytes(args.sizes, "sizes")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        io_threads = _parse_comma_separated_ints(args.io_threads, "io-threads")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        pipelining = _parse_comma_separated_ints(args.pipelining, "pipelining")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        key_sizes = _parse_comma_separated_bytes(args.key_sizes, "key-sizes")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        warmup = _parse_human_time(args.warmup, "warmup")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        duration = _parse_human_time(args.duration, "duration")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.repetitions < 1:
        print("Error: Repetitions must be at least 1", file=sys.stderr)
        return 1

    combinations = generate_task_combinations(tests, sizes, io_threads, pipelining, key_sizes)

    queue = TaskQueue()
    for test, size, io_thread, pipeline, key_size in combinations:
        task = PerfTaskData(
            source=args.source,
            specifier=args.specifier,
            make_args=args.make_args,
            replicas=0,
            note=args.note,
            requirements={},
            test=test,
            val_size=size,
            io_threads=io_thread,
            pipelining=pipeline,
            warmup=warmup,
            duration=duration,
            profiling_sample_rate=0,
            perf_stat_enabled=args.perf_stat,
            has_expire=False,
            preload_keys=not args.no_preload,
            key_size=key_size,
            repetitions=args.repetitions,
        )
        queue.submit_task(task)

    print(f"Queued {len(combinations)} task(s):")
    print(f"  source={args.source} specifier={args.specifier}")
    print(f"  tests={tests} sizes={sizes} io-threads={io_threads} pipeline={pipelining}")
    print(f"  duration={duration}s warmup={warmup}s reps={args.repetitions}")
    if args.make_args:
        print(f"  make-args: {args.make_args}")
    if args.note:
        print(f"  note: {args.note}")
    return 0


def handle_queue_list(args: argparse.Namespace) -> int:
    """Handle 'queue list': show all pending tasks."""
    queue = TaskQueue()
    tasks = queue.get_all_tasks()

    if not tasks:
        print("No pending tasks in the queue.")
        return 0

    print(f"{'#':<4} {'Task ID':<30} {'Description':<50} {'Note'}")
    print("-" * 110)
    for i, task in enumerate(tasks, 1):
        print(f"{i:<4} {task.task_id:<30} {task.short_description():<50} {task.note}")

    print(f"\nTotal: {len(tasks)} task(s)")
    return 0


def handle_queue_remove(args: argparse.Namespace) -> int:
    """Handle 'queue remove': remove a task by ID."""
    queue = TaskQueue()
    if queue.remove_task(args.task_id):
        print(f"Removed task: {args.task_id}")
        return 0
    else:
        print(f"Error: Task not found: {args.task_id}", file=sys.stderr)
        return 1


def handle_queue_clear(args: argparse.Namespace) -> int:
    """Handle 'queue clear': remove all pending tasks."""
    queue = TaskQueue()
    tasks = queue.get_all_tasks()
    if not tasks:
        print("Queue is already empty.")
        return 0

    for task in tasks:
        queue.remove_task(task.task_id)

    print(f"Cleared {len(tasks)} task(s) from the queue.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the CLI module."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage()
        return 1

    if args.command == "queue":
        if args.queue_command is None:
            # Default: list
            return handle_queue_list(args)
        elif args.queue_command == "list":
            return handle_queue_list(args)
        elif args.queue_command == "add":
            return handle_queue_add(args)
        elif args.queue_command == "remove":
            return handle_queue_remove(args)
        elif args.queue_command == "clear":
            return handle_queue_clear(args)

    parser.print_usage()
    return 1


if __name__ == "__main__":
    sys.exit(main())
