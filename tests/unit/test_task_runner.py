"""Tests for TaskRunner resilience — failed tasks don't crash the runner."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.task_runner import TaskRunner


@pytest.fixture
def temp_project_root(tmp_path):
    """Patch FAILED_TASKS paths to a temp directory for test isolation."""
    with patch("src.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed_tasks.jsonl"), \
         patch("src.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"):
        yield tmp_path


def _make_failed_task():
    """Create a minimal task for failure testing."""
    from unittest.mock import MagicMock
    task = MagicMock()
    task.task_id = "2026.05.20_12.00.00.000000"
    task.note = "test-task"
    task.source = "valkey"
    task.specifier = "abc123"
    return task


class TestRecordFailure:
    def test_records_failure_to_jsonl(self, temp_project_root):
        runner = TaskRunner()
        task = _make_failed_task()
        exc = RuntimeError("server crashed")

        runner._record_failure(task, exc)

        log_file = temp_project_root / "failed_tasks.jsonl"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry["note"] == "test-task"
        assert "server crashed" in entry["error"]
        assert "RuntimeError" in entry["error"]
        assert entry["traceback"]  # non-empty

    def test_saves_task_to_failed_dir(self, temp_project_root):
        runner = TaskRunner()
        task = _make_failed_task()
        exc = ValueError("bad config")

        runner._record_failure(task, exc)

        failed_dir = temp_project_root / "failed"
        assert failed_dir.exists()
        task.save_to_file.assert_called_once()
        # Verify the path passed to save_to_file is in the failed dir
        call_path = task.save_to_file.call_args[0][0]
        assert str(failed_dir) in str(call_path)


class TestSubscribers:
    """Tests for the pub/sub task completion pattern."""

    def test_subscriber_registered_when_sweep_enabled(self):
        with patch("src.sweep.coordinator.SweepCoordinator") as MockCoord:
            MockCoord.return_value.initialize = MagicMock()
            runner = TaskRunner(sweep=True)
            assert len(runner._subscribers) == 1

    def test_no_subscribers_without_sweep(self):
        runner = TaskRunner(sweep=False)
        assert len(runner._subscribers) == 0
