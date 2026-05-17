"""Non-interactive CLI for queuing benchmark tasks."""

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
    """Check whether a source string is a recognized repository name or manually uploaded.

    Args:
        source: The source string to validate.

    Returns:
        True if the source is valid, False otherwise.
    """
    return source in config.REPO_NAMES or source == config.MANUALLY_UPLOADED


def generate_task_combinations(
    tests: List[str],
    sizes: List[int],
    io_threads: List[int],
    pipelining: List[int],
    key_sizes: List[int],
) -> List[Tuple[str, int, int, int, int]]:
    """Compute the Cartesian product of multi-valued benchmark parameters.

    Args:
        tests: List of test names (e.g., ["get", "set"]).
        sizes: List of value sizes in bytes.
        io_threads: List of IO thread counts.
        pipelining: List of pipelining values.
        key_sizes: List of key sizes in bytes.

    Returns:
        A list of (test, size, io_thread, pipeline, key_size) tuples,
        one for each unique combination.
    """
    return list(itertools.product(tests, sizes, io_threads, pipelining, key_sizes))


def _parse_comma_separated_ints(value: str, name: str) -> List[int]:
    """Parse a comma-separated string of integers.

    Args:
        value: Comma-separated string (e.g., "1,9").
        name: Parameter name for error messages.

    Returns:
        List of parsed integers.

    Raises:
        ValueError: If any element is not a valid integer.
    """
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
    """Parse a comma-separated string of human-readable byte values.

    Args:
        value: Comma-separated string (e.g., "512,1KB").
        name: Parameter name for error messages.

    Returns:
        List of parsed byte values as integers.

    Raises:
        ValueError: If any element is not a valid byte value.
    """
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
    """Parse a human-readable time value.

    Args:
        value: Human-readable time string (e.g., "1m", "15m").
        name: Parameter name for error messages.

    Returns:
        Time value in seconds as an integer.

    Raises:
        ValueError: If the value is not a valid time string.
    """
    try:
        return int(HumanTime.from_human(value))
    except ValueError:
        raise ValueError(f"Invalid time value for --{name}: '{value}'")


def _parse_tests(value: str) -> List[str]:
    """Parse a comma-separated list of test names.

    Args:
        value: Comma-separated test names (e.g., "get,set").

    Returns:
        List of test name strings.

    Raises:
        ValueError: If the list is empty.
    """
    tests = [t.strip() for t in value.split(",") if t.strip()]
    if not tests:
        raise ValueError("No tests specified")
    return tests


def build_perf_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'perf' subcommand to the argument parser.

    Args:
        subparsers: The subparsers action from the parent parser.
    """
    perf_parser = subparsers.add_parser("perf", help="Queue performance benchmark tasks")
    perf_parser.add_argument("--source", required=True, help="Repository source name")
    perf_parser.add_argument("--specifier", required=True, help="Branch, tag, or commit hash")
    perf_parser.add_argument(
        "--tests", required=True, help="Comma-separated test names (e.g., get,set)"
    )
    perf_parser.add_argument(
        "--sizes", required=True, help="Comma-separated value sizes (e.g., 512,1KB)"
    )
    perf_parser.add_argument(
        "--io-threads", required=True, help="Comma-separated IO thread counts (e.g., 1,9)"
    )
    perf_parser.add_argument(
        "--pipelining", required=True, help="Comma-separated pipelining values (e.g., 1,4)"
    )
    perf_parser.add_argument(
        "--warmup", default="1m", help="Warmup duration (e.g., 1m, 30s). Default: 1m"
    )
    perf_parser.add_argument(
        "--duration", default="15m", help="Test duration (e.g., 15m, 1h). Default: 15m"
    )
    perf_parser.add_argument(
        "--repetitions", type=int, default=1, help="Number of repetitions per configuration. Default: 1"
    )
    perf_parser.add_argument(
        "--key-sizes", default="0", help="Comma-separated key sizes (e.g., 0,64). Default: 0"
    )
    perf_parser.add_argument("--note", default="", help="Optional note for the tasks")
    perf_parser.add_argument(
        "--make-args", default=config.DEFAULT_MAKE_ARGS, help="Build arguments"
    )
    perf_parser.add_argument(
        "--perf-stat", action="store_true", default=False,
        help="Collect hardware performance counters (perf stat) during benchmark"
    )


def build_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'queue' subcommand to the argument parser.

    Args:
        subparsers: The subparsers action from the parent parser.
    """
    subparsers.add_parser("queue", help="List all pending tasks in the queue")


def handle_perf(args: argparse.Namespace) -> int:
    """Handle the 'perf' subcommand: validate inputs, generate tasks, and submit them.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Validate source
    if not validate_source(args.source):
        valid_sources = config.REPO_NAMES + [config.MANUALLY_UPLOADED]
        print(
            f"Error: Invalid source '{args.source}'. "
            f"Valid sources: {', '.join(valid_sources)}",
            file=sys.stderr,
        )
        return 1

    # Parse multi-valued parameters
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

    # Parse time values
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

    # Validate repetitions
    if args.repetitions < 1:
        print("Error: Repetitions must be at least 1", file=sys.stderr)
        return 1

    # Generate Cartesian product of combinations
    combinations = generate_task_combinations(tests, sizes, io_threads, pipelining, key_sizes)

    # Submit tasks to queue
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
            preload_keys=True,
            key_size=key_size,
            repetitions=args.repetitions,
        )
        queue.submit_task(task)

    print(f"Queued {len(combinations)} task(s) for source={args.source} specifier={args.specifier}")
    return 0


def handle_queue(args: argparse.Namespace) -> int:
    """Handle the 'queue' subcommand: list all pending tasks.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    queue = TaskQueue()
    tasks = queue.get_all_tasks()

    if not tasks:
        print("No pending tasks in the queue.")
        return 0

    print(f"{'Task ID':<30} {'Type':<12} {'Source':<20} {'Specifier':<20} {'Description'}")
    print("-" * 110)
    for task in tasks:
        task_type = task.task_type.removesuffix("TaskData")
        print(
            f"{task.task_id:<30} {task_type:<12} {task.source:<20} "
            f"{task.specifier:<20} {task.short_description()}"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="conductress-cli",
        description="Conductress CLI for queuing and managing benchmark tasks",
    )
    subparsers = parser.add_subparsers(dest="command")
    build_perf_parser(subparsers)
    build_queue_parser(subparsers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the CLI module.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:] if None.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage()
        return 1

    if args.command == "perf":
        return handle_perf(args)
    elif args.command == "queue":
        return handle_queue(args)
    else:
        parser.print_usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
