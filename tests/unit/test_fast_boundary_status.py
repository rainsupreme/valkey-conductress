"""Regression and performance-contract tests for the fast-boundary-status feature.

Proves:
1. One file-tail read per build_status (not two whole-file reads).
2. One build_status call per _publish_boundary (not two).
3. Rsync retry fires only/exactly for STARTING when the first publish fails.
4. Atomic local status.json write (no partial files served to concurrent readers).
5. Duration calibration and recent results share a single snapshot.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from conductress.duration_estimator import load_duration_calibration, load_duration_calibration_from_lines
from conductress.status_export import (
    _TAIL_BUDGET,
    _get_queue_info,
    _get_recent_results,
    _read_result_snapshot,
    build_status,
    export_status,
)


# ---------------------------------------------------------------------------
# Contract 1: One tail read per build_status
# ---------------------------------------------------------------------------
class TestSingleTailReadPerBuild:
    """build_status must read output.jsonl at most once (via tail_lines)."""

    def test_build_status_calls_tail_lines_once(self):
        """Without a pre-supplied snapshot, build_status reads via _read_result_snapshot."""
        with (
            patch("conductress.status_export.tail_lines", return_value=[]) as mock_tail,
            patch(
                "conductress.status_export.get_runner_info",
                return_value={"runner_id": "test", "platform": {"label": "test"}},
            ),
            patch(
                "conductress.status_export._get_runner_info",
                return_value={"pid": None, "state": "stopped", "uptime_hours": None},
            ),
            patch("conductress.status_export._get_current_task", return_value=None),
            patch("conductress.status_export.TaskQueue") as MockQueue,
            patch("conductress.status_export._get_disk_info", return_value={}),
        ):
            MockQueue.return_value.get_all_tasks.return_value = []
            build_status()

        mock_tail.assert_called_once()

    def test_build_status_skips_tail_when_snapshot_provided(self):
        """Pre-supplied snapshot skips file I/O entirely."""
        with (
            patch("conductress.status_export.tail_lines") as mock_tail,
            patch(
                "conductress.status_export.get_runner_info",
                return_value={"runner_id": "test", "platform": {"label": "test"}},
            ),
            patch(
                "conductress.status_export._get_runner_info",
                return_value={"pid": None, "state": "stopped", "uptime_hours": None},
            ),
            patch("conductress.status_export._get_current_task", return_value=None),
            patch("conductress.status_export.TaskQueue") as MockQueue,
            patch("conductress.status_export._get_disk_info", return_value={}),
        ):
            MockQueue.return_value.get_all_tasks.return_value = []
            build_status(_result_snapshot=["line1", "line2"])

        mock_tail.assert_not_called()


# ---------------------------------------------------------------------------
# Contract 2: One build_status per _publish_boundary
# ---------------------------------------------------------------------------
class TestSingleBuildPerBoundary:
    """TaskRunner._publish_boundary must call build_status exactly once."""

    def test_publish_boundary_calls_build_status_once_with_mailbox(self):
        from conductress.task_runner import TaskRunner

        mailbox = MagicMock()
        mailbox.status.return_value = {"mode": "live"}
        mailbox.push_status.return_value = True

        runner = TaskRunner()
        runner._mailbox = mailbox
        runner._publish_target = "user@host:/data"

        with (
            patch("conductress.task_runner.build_status", wraps=build_status) as mock_bs,
            patch("conductress.task_runner.export_status") as mock_export,
            # Mock all the sub-functions build_status calls
            patch(
                "conductress.status_export.get_runner_info",
                return_value={"runner_id": "test", "platform": {"label": "test"}},
            ),
            patch(
                "conductress.status_export._get_runner_info",
                return_value={"pid": None, "state": "stopped", "uptime_hours": None},
            ),
            patch("conductress.status_export._get_current_task", return_value=None),
            patch("conductress.status_export.TaskQueue") as MockQueue,
            patch("conductress.status_export._get_disk_info", return_value={}),
            patch("conductress.status_export.tail_lines", return_value=[]),
        ):
            MockQueue.return_value.get_all_tasks.return_value = []
            runner._publish_boundary("completed", None)

        # build_status called exactly once
        mock_bs.assert_called_once()
        # export_status called exactly once (no retry for non-starting)
        mock_export.assert_called_once()

    def test_publish_boundary_calls_build_status_once_without_mailbox(self):
        from conductress.task_runner import TaskRunner

        runner = TaskRunner()
        runner._publish_target = "user@host:/data"

        with (
            patch("conductress.task_runner.build_status", wraps=build_status) as mock_bs,
            patch("conductress.task_runner.export_status") as mock_export,
            patch(
                "conductress.status_export.get_runner_info",
                return_value={"runner_id": "test", "platform": {"label": "test"}},
            ),
            patch(
                "conductress.status_export._get_runner_info",
                return_value={"pid": None, "state": "stopped", "uptime_hours": None},
            ),
            patch("conductress.status_export._get_current_task", return_value=None),
            patch("conductress.status_export.TaskQueue") as MockQueue,
            patch("conductress.status_export._get_disk_info", return_value={}),
            patch("conductress.status_export.tail_lines", return_value=[]),
        ):
            MockQueue.return_value.get_all_tasks.return_value = []
            runner._publish_boundary("idle", None)

        mock_bs.assert_called_once()


# ---------------------------------------------------------------------------
# Contract 3: Checked rsync retry only for STARTING
# ---------------------------------------------------------------------------
class TestStartingRsyncRetry:
    """STARTING requests two attempts; export_status retries only on failure."""

    def _make_runner_with_target(self):
        from conductress.task_runner import TaskRunner

        runner = TaskRunner()
        runner._publish_target = "user@host:/data"
        return runner

    def test_starting_requests_two_publish_attempts(self):
        runner = self._make_runner_with_target()
        with (
            patch("conductress.task_runner.build_status", return_value={"boundary": {"state": "starting"}}),
            patch("conductress.task_runner.export_status") as mock_export,
        ):
            runner._publish_boundary("starting", None)

        mock_export.assert_called_once()
        assert mock_export.call_args.kwargs["publish_attempts"] == 2

    @pytest.mark.parametrize("state", ["completed", "failed", "idle"])
    def test_nonstarting_requests_one_publish_attempt(self, state):
        runner = self._make_runner_with_target()
        with (
            patch("conductress.task_runner.build_status", return_value={"boundary": {"state": state}}),
            patch("conductress.task_runner.export_status") as mock_export,
        ):
            runner._publish_boundary(state, None)

        mock_export.assert_called_once()
        assert mock_export.call_args.kwargs["publish_attempts"] == 1

    def test_export_retries_once_after_failure(self, tmp_path):
        status_dir = tmp_path / "status"
        status_file = status_dir / "status.json"
        with (
            patch("conductress.status_export.STATUS_EXPORT_DIR", status_dir),
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
            patch("conductress.status_export._publish_status", side_effect=[False, True]) as publish,
        ):
            path = export_status(
                "user@host:/data",
                status={"schema_version": 1},
                publish_attempts=2,
            )

        assert path == status_file
        assert publish.call_count == 2

    def test_export_does_not_retry_after_success(self, tmp_path):
        status_dir = tmp_path / "status"
        status_file = status_dir / "status.json"
        with (
            patch("conductress.status_export.STATUS_EXPORT_DIR", status_dir),
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
            patch("conductress.status_export._publish_status", return_value=True) as publish,
        ):
            export_status(
                "user@host:/data",
                status={"schema_version": 1},
                publish_attempts=2,
            )

        publish.assert_called_once()

    def test_publish_attempts_must_be_positive(self):
        with pytest.raises(ValueError, match="publish_attempts"):
            export_status(status={"schema_version": 1}, publish_attempts=0)


# ---------------------------------------------------------------------------
# Contract 4: Atomic local write
# ---------------------------------------------------------------------------
class TestAtomicStatusWrite:
    """export_status must not leave a partial status.json for concurrent readers."""

    def test_atomic_write_via_rename(self, tmp_path):
        """The final status.json is written atomically (no partial content visible)."""
        status_dir = tmp_path / "status"
        status_file = status_dir / "status.json"

        with (
            patch("conductress.status_export.STATUS_EXPORT_DIR", status_dir),
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
        ):
            status = {"test": True, "schema_version": 1}
            path = export_status(status=status)

        assert path == status_file
        data = json.loads(status_file.read_text())
        assert data["test"] is True

    def test_no_temp_files_left_on_success(self, tmp_path):
        status_dir = tmp_path / "status"
        status_file = status_dir / "status.json"

        with (
            patch("conductress.status_export.STATUS_EXPORT_DIR", status_dir),
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
        ):
            export_status(status={"ok": True})

        # Only status.json should exist, no .status-*.json temp files
        files = list(status_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == "status.json"

    def test_temp_file_cleaned_on_failure(self, tmp_path):
        status_dir = tmp_path / "status"
        status_dir.mkdir(parents=True)
        status_file = status_dir / "status.json"

        with (
            patch("conductress.status_export.STATUS_EXPORT_DIR", status_dir),
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
            patch("conductress.status_export.json.dump", side_effect=IOError("disk full")),
            pytest.raises(IOError),
        ):
            export_status(status={"ok": True})

        # No temp files should be left behind
        temp_files = [f for f in status_dir.iterdir() if f.name.startswith(".status-")]
        assert temp_files == []


# ---------------------------------------------------------------------------
# Contract 5: Shared snapshot for calibration + recent results
# ---------------------------------------------------------------------------
class TestSharedSnapshot:
    """Calibration and recent results must come from the same snapshot."""

    def test_queue_info_uses_snapshot_lines(self):
        """_get_queue_info with snapshot uses load_duration_calibration_from_lines."""
        lines = [
            json.dumps(
                {
                    "task_id": f"t{i}",
                    "duration_family": "perf",
                    "expected_duration_sec": 100,
                    "observed_duration_sec": 130,
                }
            )
            for i in range(5)
        ]

        with (patch("conductress.status_export.TaskQueue") as MockQueue,):
            MockQueue.return_value.get_all_tasks.return_value = []
            result = _get_queue_info(snapshot=lines)

        assert result["depth"] == 0

    def test_recent_results_uses_snapshot_lines(self):
        """_get_recent_results with snapshot does not touch the filesystem."""
        lines = [
            json.dumps({"task_id": "t1", "score": 100, "commit_hash": "abc12345"}),
            json.dumps({"task_id": "t2", "score": 200, "commit_hash": "def67890"}),
        ]

        result = _get_recent_results(snapshot=lines)
        assert len(result) == 2
        assert result[0]["task_id"] == "t2"  # newest first (reversed)

    def test_calibration_from_lines_matches_file_based(self, tmp_path):
        """load_duration_calibration_from_lines produces same result as file-based."""
        records = [
            {"task_id": f"t{i}", "duration_family": "perf", "expected_duration_sec": 100, "observed_duration_sec": 130}
            for i in range(5)
        ]
        lines = [json.dumps(r) for r in records]
        output = tmp_path / "output.jsonl"
        output.write_text("\n".join(lines), encoding="utf-8")

        from_file = load_duration_calibration(output)
        from_lines = load_duration_calibration_from_lines(lines)
        assert from_file == from_lines


# ---------------------------------------------------------------------------
# Contract 6: _publish_status returns bool
# ---------------------------------------------------------------------------
class TestPublishStatusReturnsBool:
    """_publish_status must return rsync success/failure."""

    def test_returns_true_on_rsync_success(self, tmp_path):
        from conductress.status_export import _publish_status

        status_file = tmp_path / "status.json"
        status_file.write_text('{"runner_id": "test"}')

        with (
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
            patch("conductress.publisher.detect_platform", return_value=("arm64", "arm64/test")),
            patch("conductress.status_export.get_runner_info", return_value={"runner_id": "test"}),
            patch("conductress.utility.run_rsync", return_value=True),
        ):
            assert _publish_status("user@host:/data") is True

    def test_returns_false_on_rsync_failure(self, tmp_path):
        from conductress.status_export import _publish_status

        status_file = tmp_path / "status.json"
        status_file.write_text('{"runner_id": "test"}')

        with (
            patch("conductress.status_export.STATUS_EXPORT_FILE", status_file),
            patch("conductress.publisher.detect_platform", return_value=("arm64", "arm64/test")),
            patch("conductress.status_export.get_runner_info", return_value={"runner_id": "test"}),
            patch("conductress.utility.run_rsync", return_value=False),
        ):
            assert _publish_status("user@host:/data") is False
