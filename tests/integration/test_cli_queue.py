"""Integration tests for CLI task queuing.

Tests verify end-to-end CLI flow without mocking the TaskQueue:
- Invoke CLI perf subcommand, verify task JSON files appear in queue directory
  with correct parameters.
- Invoke CLI queue subcommand, verify it lists the queued tasks.

Requirements: 9.8
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src import config
from src.cli import main
from src.task_queue import TaskQueue


@pytest.fixture
def queue_dir(tmp_path):
    """Provide a temporary queue directory for task files."""
    queue_path = tmp_path / "benchmark_queue"
    queue_path.mkdir()
    return queue_path


@pytest.fixture(autouse=True)
def isolate_queue(queue_dir):
    """Patch TaskQueue so the CLI uses the temporary queue directory.

    The default argument in TaskQueue.__init__ is captured at class definition
    time, so we patch the class used by the CLI module to always use our temp dir.
    """
    _OriginalTaskQueue = TaskQueue

    class _IsolatedTaskQueue(_OriginalTaskQueue):
        def __init__(self, queue_dir_override=None):
            super().__init__(queue_dir=queue_dir)

    with patch("src.cli.TaskQueue", _IsolatedTaskQueue):
        yield


@pytest.fixture(autouse=True)
def patch_sources():
    """Patch REPO_NAMES and MANUALLY_UPLOADED for all tests in this module."""
    with patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]), \
         patch("src.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]), \
         patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"), \
         patch("src.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"):
        yield


class TestCliPerfQueuing:
    """End-to-end tests: invoke CLI queue add subcommand, verify task JSON files."""

    def test_queue_add_creates_task_files_in_queue_directory(self, queue_dir):
        """Invoking queue add with a single combination creates exactly one task JSON file."""
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "unstable",
            "--tests", "get",
            "--sizes", "512",
            "--io-threads", "1",
            "--pipelining", "1",
        ])

        assert exit_code == 0

        task_files = list(queue_dir.glob("task_*.json"))
        assert len(task_files) == 1

    def test_queue_add_task_json_has_correct_parameters(self, queue_dir):
        """The queued task JSON file contains the correct benchmark parameters."""
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "v8.0",
            "--tests", "set",
            "--sizes", "1024",
            "--io-threads", "9",
            "--pipelining", "4",
            "--warmup", "30s",
            "--duration", "5m",
            "--repetitions", "3",
            "--key-sizes", "64",
            "--note", "integration test",
        ])

        assert exit_code == 0

        task_files = list(queue_dir.glob("task_*.json"))
        assert len(task_files) == 1

        with open(task_files[0]) as f:
            data = json.load(f)

        assert data["source"] == "valkey"
        assert data["specifier"] == "v8.0"
        assert data["test"] == "set"
        assert data["val_size"] == 1024
        assert data["io_threads"] == 9
        assert data["pipelining"] == 4
        assert data["warmup"] == 30
        assert data["duration"] == 300
        assert data["repetitions"] == 3
        assert data["key_size"] == 64
        assert data["note"] == "integration test"
        assert data["task_type"] == "PerfTaskData"

    def test_queue_add_cartesian_product_creates_correct_number_of_files(self, queue_dir):
        """Multiple values for tests, sizes, io-threads, pipelining, key-sizes
        produce the full Cartesian product of task files."""
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "unstable",
            "--tests", "get,set",
            "--sizes", "512,1024",
            "--io-threads", "1,9",
            "--pipelining", "1",
            "--key-sizes", "0,64",
        ])

        assert exit_code == 0

        task_files = list(queue_dir.glob("task_*.json"))
        # 2 tests * 2 sizes * 2 io-threads * 1 pipelining * 2 key-sizes = 16
        assert len(task_files) == 16

    def test_queue_add_cartesian_product_covers_all_combinations(self, queue_dir):
        """Every unique (test, val_size, io_threads, pipelining, key_size) combination
        appears exactly once across the queued task files."""
        exit_code = main([
            "queue", "add",
            "--source", "testrepo",
            "--specifier", "main",
            "--tests", "get,set",
            "--sizes", "512,1024",
            "--io-threads", "1,9",
            "--pipelining", "1,4",
            "--key-sizes", "0",
        ])

        assert exit_code == 0

        task_files = list(queue_dir.glob("task_*.json"))
        # 2 * 2 * 2 * 2 * 1 = 16
        assert len(task_files) == 16

        combos = set()
        for tf in task_files:
            with open(tf) as f:
                data = json.load(f)
            combo = (
                data["test"],
                data["val_size"],
                data["io_threads"],
                data["pipelining"],
                data["key_size"],
            )
            combos.add(combo)

        assert len(combos) == 16, "Every combination should be unique"

    def test_queue_add_invalid_source_creates_no_files(self, queue_dir):
        """An invalid source should produce no task files and return exit code 1."""
        exit_code = main([
            "queue", "add",
            "--source", "nonexistent_repo",
            "--specifier", "unstable",
            "--tests", "get",
            "--sizes", "512",
            "--io-threads", "1",
            "--pipelining", "1",
        ])

        assert exit_code == 1
        task_files = list(queue_dir.glob("task_*.json"))
        assert len(task_files) == 0

    def test_queue_add_manually_uploaded_source_is_accepted(self, queue_dir):
        """The manually_uploaded source should be accepted and create task files."""
        exit_code = main([
            "queue", "add",
            "--source", "manually_uploaded",
            "--specifier", "custom-build",
            "--tests", "get",
            "--sizes", "512",
            "--io-threads", "1",
            "--pipelining", "1",
        ])

        assert exit_code == 0
        task_files = list(queue_dir.glob("task_*.json"))
        assert len(task_files) == 1

        with open(task_files[0]) as f:
            data = json.load(f)
        assert data["source"] == "manually_uploaded"

    def test_queue_add_default_values_applied(self, queue_dir):
        """Default values for warmup, duration, repetitions, key-sizes are applied."""
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "unstable",
            "--tests", "get",
        ])

        assert exit_code == 0

        task_files = list(queue_dir.glob("task_*.json"))
        assert len(task_files) == 1

        with open(task_files[0]) as f:
            data = json.load(f)

        assert data["warmup"] == 30
        assert data["duration"] == 300
        assert data["repetitions"] == 5
        assert data["key_size"] == 0
        assert data["io_threads"] == 9
        assert data["pipelining"] == 10
        assert data["val_size"] == 512
        assert data["make_args"] == "USE_FAST_FLOAT=yes"


class TestCliQueueListing:
    """End-to-end tests: invoke CLI queue subcommand, verify it lists queued tasks."""

    def test_queue_lists_queued_tasks(self, queue_dir, capsys):
        """After queuing tasks via perf, the queue subcommand should list them."""
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "unstable",
            "--tests", "get,set",
            "--sizes", "512",
            "--io-threads", "1",
            "--pipelining", "1",
        ])
        assert exit_code == 0
        capsys.readouterr()  # clear perf output

        # Now list the queue
        exit_code = main(["queue"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Total: 2 task(s)" in captured.out

    def test_queue_empty_shows_no_pending(self, queue_dir, capsys):
        """When no tasks are queued, the queue subcommand shows a no-pending message."""
        exit_code = main(["queue"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No pending tasks" in captured.out

    def test_queue_shows_correct_task_count(self, queue_dir, capsys):
        """The queue listing should show the same number of tasks that were queued."""
        # Queue 4 tasks: 2 tests * 2 sizes
        exit_code = main([
            "queue", "add",
            "--source", "valkey",
            "--specifier", "unstable",
            "--tests", "get,set",
            "--sizes", "512,1024",
            "--io-threads", "1",
            "--pipelining", "1",
        ])
        assert exit_code == 0
        capsys.readouterr()

        exit_code = main(["queue"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Total: 4 task(s)" in captured.out

    def test_queue_shows_task_details(self, queue_dir, capsys):
        """The queue listing should show task ID, description, and note."""
        exit_code = main([
            "queue", "add",
            "--source", "testrepo",
            "--specifier", "feature-branch",
            "--tests", "set",
            "--sizes", "512",
            "--io-threads", "1",
            "--pipelining", "1",
            "--note", "detail check",
        ])
        assert exit_code == 0
        capsys.readouterr()

        exit_code = main(["queue"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "detail check" in captured.out
        assert "Task ID" in captured.out
