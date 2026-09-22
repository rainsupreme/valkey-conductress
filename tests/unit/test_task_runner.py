"""Tests for TaskRunner resilience — failed tasks don't crash the runner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.task_runner import TaskRunner


@pytest.fixture
def temp_project_root(tmp_path):
    """Patch FAILED_TASKS paths to a temp directory for test isolation."""
    with (
        patch("conductress.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed_tasks.jsonl"),
        patch("conductress.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"),
    ):
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
        """With v1 archived (default), sweep builds no v1 coordinator but still
        registers the generator-independent memory subscribers."""
        with (
            patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord,
            patch("conductress.sweep.latency_coordinator.LatencySweepCoordinator") as MockLatency,
            patch("conductress.sweep.memory_coordinator.create_memory_coordinators") as mock_factory,
            patch("conductress.platform.get_local_platform_tag", return_value="amd64"),
        ):
            MockCoord.return_value.initialize = MagicMock()
            MockLatency.return_value.initialize = MagicMock()
            mock_factory.return_value = [MagicMock(initialize=MagicMock()) for _ in range(5)]
            runner = TaskRunner(sweep=True)
            # Memory coordinators register; the archived v1 throughput and
            # latency coordinators are never constructed.
            assert len(runner._subscribers) == 5
            MockCoord.assert_not_called()
            MockLatency.assert_not_called()

    def test_no_v1_coordinator_built_when_archived(self):
        """The falsifiable core of the shutdown: an archived v1 epoch builds no
        v1 SweepCoordinator or LatencySweepCoordinator at all. This fails
        against main, where both are constructed unconditionally."""
        with (
            patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord,
            patch("conductress.sweep.latency_coordinator.LatencySweepCoordinator") as MockLatency,
            patch("conductress.sweep.memory_coordinator.create_memory_coordinators", return_value=[]),
            patch("conductress.platform.get_local_platform_tag", return_value="amd64"),
            patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ("v1",)),
        ):
            TaskRunner(sweep=True)
            MockCoord.assert_not_called()
            MockLatency.assert_not_called()

    def test_v1_coordinator_built_when_not_archived(self):
        """When v1 is NOT archived, the v1 throughput and latency coordinators
        are constructed as before, confirming the gate is the only change."""
        with (
            patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord,
            patch("conductress.sweep.latency_coordinator.LatencySweepCoordinator") as MockLatency,
            patch("conductress.sweep.memory_coordinator.create_memory_coordinators", return_value=[]),
            patch("conductress.platform.get_local_platform_tag", return_value="amd64"),
            patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ()),
        ):
            MockCoord.return_value.initialize = MagicMock()
            MockLatency.return_value.initialize = MagicMock()
            TaskRunner(sweep=True)
            assert MockCoord.called
            MockLatency.assert_called_once()

    def test_no_subscribers_without_sweep(self):
        runner = TaskRunner(sweep=False)
        assert len(runner._subscribers) == 0
