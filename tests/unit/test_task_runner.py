"""Tests for TaskRunner resilience — failed tasks don't crash the runner."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


class TestExtractResult:
    """Tests for _extract_result reading sweep results from output.jsonl."""

    def test_extracts_score_from_last_entry(self, tmp_path):
        output_file = tmp_path / "results" / "output.jsonl"
        output_file.parent.mkdir(parents=True)
        entries = [
            json.dumps({"score": 1000000, "task_id": "old_task"}),
            json.dumps({"score": 1952516.22, "task_id": "latest_task"}),
        ]
        output_file.write_text("\n".join(entries))

        with patch("src.config.CONDUCTRESS_RESULTS", tmp_path / "results"):
            runner = TaskRunner()
            task = _make_failed_task()
            rps, cv = runner._extract_result(task)

        assert rps == pytest.approx(1952516.22)
        assert cv == 0.0  # no cv_pct or stdev_rps in entry

    def test_returns_none_when_no_output_file(self, tmp_path):
        with patch("src.config.CONDUCTRESS_RESULTS", tmp_path / "nonexistent"):
            runner = TaskRunner()
            task = _make_failed_task()
            rps, cv = runner._extract_result(task)

        assert rps is None
        assert cv is None

    def test_extracts_cv_from_stdev(self, tmp_path):
        output_file = tmp_path / "results" / "output.jsonl"
        output_file.parent.mkdir(parents=True)
        output_file.write_text(json.dumps({
            "score": 2000000, "stdev_rps": 10000, "task_id": "x"
        }))

        with patch("src.config.CONDUCTRESS_RESULTS", tmp_path / "results"):
            runner = TaskRunner()
            rps, cv = runner._extract_result(_make_failed_task())

        assert rps == 2000000
        assert cv == pytest.approx(0.5)  # 10000/2000000 * 100
