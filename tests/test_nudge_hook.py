"""Tests for the NudgeHook HTTP webhook subscriber."""

import json
import os
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress.config import CONDUCTRESS_OUTPUT
from conductress.nudge_hook import NudgeHook
from conductress.tasks.task_perf_benchmark import PerfTaskData


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    """Ensure test_cli.py pollution (direct config.REPO_NAMES modification) doesn't affect our tests."""
    import conductress.config as cfg

    monkeypatch.setattr(cfg, "REPO_NAMES", ["valkey", "rainsupreme", "zuiderkwast", "JimB123"])
    monkeypatch.setattr(cfg, "MANUALLY_UPLOADED", "manually_uploaded")
    Path(CONDUCTRESS_OUTPUT).unlink(missing_ok=True)
    yield
    Path(CONDUCTRESS_OUTPUT).unlink(missing_ok=True)


def _make_perf_task() -> PerfTaskData:
    """Create a minimal PerfTaskData for testing."""
    return PerfTaskData(
        source="valkey",
        specifier="unstable",
        make_args="",
        replicas=0,
        note="test note",
        requirements={},
        test="get",
        val_size=16,
        io_threads=9,
        pipelining=10,
        warmup=30,
        duration=30,
        perf_stat_enabled=False,
        has_expire=False,
        preload_keys=True,
        repetitions=1,
    )


class TestNudgeHookInit:
    """Test NudgeHook initialization."""

    def test_default_events(self):
        hook = NudgeHook("http://example.com/nudge")
        assert hook._endpoint_url == "http://example.com/nudge"
        assert hook._events == {"completed", "failed", "empty"}

    def test_custom_events(self):
        events = {"completed", "empty"}
        hook = NudgeHook("http://example.com/nudge", events=events)
        assert hook._events == events

    def test_no_nudge_for_unset_events(self):
        """When an event type is not in the events set, no request is sent."""
        hook = NudgeHook("http://example.com/nudge", events=set())
        task = _make_perf_task()

        with patch.object(hook, "_send") as mock_send:
            hook.on_task_completed(task)
            mock_send.assert_not_called()


class TestNudgeHookPayload:
    """Test nudge payload construction."""

    @patch("conductress.nudge_hook.logger")
    def test_payload_includes_task_metadata(self, mock_logger):
        hook = NudgeHook("http://example.com/nudge")
        task = _make_perf_task()

        with patch.object(hook, "_send") as mock_send:
            hook.on_task_completed(task)

        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["event"] == "completed"
        assert payload["task_id"] == task.task_id
        assert payload["source"] == "valkey"
        assert payload["specifier"] == "unstable"
        assert payload["note"] == "test note"
        assert payload["task_type"] == "PerfTaskData"
        assert payload["test"] == "get"
        assert payload["val_size"] == 16
        assert payload["io_threads"] == 9
        assert payload["pipelining"] == 10

    @patch("conductress.nudge_hook.logger")
    def test_payload_includes_results_from_output_log(self, mock_logger, tmp_path):
        """When a result exists in the output log, score and data are included."""
        hook = NudgeHook("http://example.com/nudge")
        task = _make_perf_task()

        # Write a result line to the output log
        os.makedirs(os.path.dirname(CONDUCTRESS_OUTPUT), exist_ok=True)
        result_entry = {
            "task_id": task.task_id,
            "score": 3295847.0,
            "commit_hash": "abc12345",
            "data": {"mean_rps": 3295847.0, "ci_95": 50000.0},
            "source": "valkey",
            "specifier": "unstable",
            "make_args": "",
            "note": "test note",
            "method": "perf-get",
            "end_time": "2026-07-24T15:00:00",
            "features": {},
            "task_type": "perf_runner",
        }
        with open(CONDUCTRESS_OUTPUT, "a") as f:
            f.write(json.dumps(result_entry) + "\n")

        with patch.object(hook, "_send") as mock_send:
            hook.on_task_completed(task)

        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["score"] == 3295847.0
        assert payload["commit_hash"] == "abc12345"
        assert payload["data"]["mean_rps"] == 3295847.0

    def test_queue_empty_payload(self):
        hook = NudgeHook("http://example.com/nudge", events={"empty"})
        with patch.object(hook, "_send") as mock_send:
            hook.on_queue_empty()
        mock_send.assert_called_once()
        sent_payload = mock_send.call_args[0][0]
        assert sent_payload["event"] == "empty"

    def test_task_failed_payload(self):
        hook = NudgeHook("http://example.com/nudge", events={"failed"})
        task = _make_perf_task()
        with patch.object(hook, "_send") as mock_send:
            hook.on_task_failed(task)
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["event"] == "failed"


class TestNudgeHookReadResults:
    """Test the _read_latest_result helper."""

    def test_reads_matching_result(self):
        task = _make_perf_task()
        entry = {
            "task_id": task.task_id,
            "score": 3000000.0,
            "commit_hash": "def67890",
            "data": {},
            "source": "valkey",
            "specifier": "unstable",
            "make_args": "",
            "note": "",
            "method": "perf-get",
            "end_time": "2026-07-24T15:00:00",
            "features": {},
            "task_type": "perf_runner",
        }
        os.makedirs(os.path.dirname(CONDUCTRESS_OUTPUT), exist_ok=True)
        with open(CONDUCTRESS_OUTPUT, "w") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": "other",
                        "score": 100,
                        "source": "valkey",
                        "specifier": "x",
                        "make_args": "",
                        "note": "",
                        "method": "m",
                        "end_time": "t",
                        "features": {},
                        "task_type": "t",
                    }
                )
                + "\n"
            )
            f.write(json.dumps(entry) + "\n")

        result = NudgeHook._read_latest_result(task.task_id)
        assert result["score"] == 3000000.0
        assert result["commit_hash"] == "def67890"

    def test_returns_none_when_no_match(self):
        os.makedirs(os.path.dirname(CONDUCTRESS_OUTPUT), exist_ok=True)
        with open(CONDUCTRESS_OUTPUT, "w") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": "other",
                        "score": 100,
                        "source": "valkey",
                        "specifier": "x",
                        "make_args": "",
                        "note": "",
                        "method": "m",
                        "end_time": "t",
                        "features": {},
                        "task_type": "t",
                    }
                )
                + "\n"
            )

        result = NudgeHook._read_latest_result("nonexistent_task")
        assert result is None

    def test_returns_none_when_file_not_found(self):
        result = NudgeHook._read_latest_result("any_task_id")
        assert result is None


class TestNudgeHookSend:
    """Test the _send HTTP method."""

    @patch("conductress.nudge_hook.logger")
    def test_successful_http_post(self, mock_logger):
        hook = NudgeHook("http://example.com/nudge")
        payload = {"event": "completed", "task_id": "test_task"}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            hook._send(payload)

        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        assert isinstance(request, type(request))
        assert request.full_url == "http://example.com/nudge"

    @patch("conductress.nudge_hook.logger")
    def test_handles_http_error(self, mock_logger):
        """HTTP errors are logged but do not raise."""
        hook = NudgeHook("http://example.com/nudge")
        payload = {"event": "completed", "task_id": "test_task"}

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("http://example.com/nudge", 500, "Internal Server Error", {}, None),
        ):
            hook._send(payload)

        mock_logger.warning.assert_called()


class TestPerfTaskDataCreation:
    """Smoke test: PerfTaskData can be instantiated with realistic fields."""

    def test_perf_task_data(self):
        task = _make_perf_task()
        assert task.source == "valkey"
        assert task.specifier == "unstable"
        assert task.test == "get"
        assert task.val_size == 16
        assert task.io_threads == 9
        assert task.pipelining == 10
        assert task.warmup == 30
        assert task.duration == 30
        assert task.task_type == "PerfTaskData"
        assert task.task_id != ""
