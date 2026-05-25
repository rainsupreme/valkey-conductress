"""Tests for the dashboard publisher."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress.publisher import DashboardPublisher, detect_platform


class TestDetectPlatform:
    def test_aarch64(self):
        with patch("conductress.publisher.platform.machine", return_value="aarch64"):
            pid, label = detect_platform()
            assert pid == "arm64"
            assert "graviton" in label

    def test_x86_64_amd(self):
        with patch("conductress.publisher.platform.machine", return_value="x86_64"):
            with patch("pathlib.Path.read_text", return_value="model name : AMD EPYC 9R14"):
                pid, label = detect_platform()
                assert pid == "amd64"

    def test_x86_64_intel(self):
        with patch("conductress.publisher.platform.machine", return_value="x86_64"):
            with patch("pathlib.Path.read_text", return_value="model name : Intel(R) Xeon(R) Platinum 8488C"):
                pid, label = detect_platform()
                assert pid == "intel"
                assert "sapphire" in label


class TestDashboardPublisher:
    def test_init(self, tmp_path):
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        pub = DashboardPublisher("user@host:/path", [coord])
        assert pub.target == "user@host:/path"
        assert pub.coordinators == [coord]

    def test_on_task_failed_is_noop(self):
        pub = DashboardPublisher("user@host:/path", [])
        pub.on_task_failed(MagicMock())  # should not raise

    def test_on_queue_empty_is_noop(self):
        pub = DashboardPublisher("user@host:/path", [])
        pub.on_queue_empty()  # should not raise

    @patch("conductress.publisher.subprocess.run")
    def test_publish_calls_rsync(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        coord.state = MagicMock()
        coord.export.return_value = 5

        pub = DashboardPublisher("user@host:/path", [coord])

        with patch("conductress.sweep.exporter.export_perf_metrics", return_value={}):
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())

        # rsync was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "rsync"
        assert "user@host:/path" in call_args[-1]

    @patch("conductress.publisher.subprocess.run")
    def test_publish_failure_does_not_raise(self, mock_run):
        """Publish failures are non-fatal."""
        mock_run.side_effect = Exception("network error")
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        coord.state = MagicMock()
        coord.export.return_value = 5

        pub = DashboardPublisher("user@host:/path", [coord])

        with patch("conductress.sweep.exporter.export_perf_metrics", return_value={}):
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())  # should not raise
