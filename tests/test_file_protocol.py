"""Tests for file-based communication protocol."""

import json
import time
from threading import Thread

import pytest

import src.file_protocol
from src.file_protocol import (
    BenchmarkResults,
    BenchmarkStatus,
    FileProtocol,
    MetricData,
)


class TestFileProtocol:
    """Test file protocol operations."""

    @pytest.fixture(autouse=True)
    def setup_config_mock(self, tmp_path):
        """Mock config paths for all tests in this class."""
        results_dir = tmp_path / "results"
        output_file = tmp_path / "output.txt"

        # Store originals
        original_results = src.file_protocol.CONDUCTRESS_RESULTS
        original_output = src.file_protocol.CONDUCTRESS_OUTPUT

        # Apply mocks
        src.file_protocol.CONDUCTRESS_RESULTS = results_dir
        src.file_protocol.CONDUCTRESS_OUTPUT = output_file

        # Make tmp_path available to test methods
        self.tmp_path = tmp_path

        yield

        # Restore originals
        src.file_protocol.CONDUCTRESS_RESULTS = original_results
        src.file_protocol.CONDUCTRESS_OUTPUT = original_output

    def test_status_write_read(self):
        """Test status file operations."""
        protocol = FileProtocol("test_task", self.tmp_path)

        status = BenchmarkStatus(steps_total=100)
        status.state = "running"
        status.pid = 12345

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.state == "running"
        assert read_status.pid == 12345
        assert read_status.steps_total == 100

    def test_metrics_append_read(self):
        """Test metrics file operations."""
        protocol = FileProtocol("test_task", self.tmp_path)

        metrics = [
            MetricData(metrics={"rps": 1000.0, "latency_ms": 2.5}),
            MetricData(metrics={"rps": 1100.0, "latency_ms": 2.3}),
            MetricData(metrics={"rps": 1050.0, "latency_ms": 2.7}),
        ]

        for metric in metrics:
            protocol.append_metric(metric)

        read_metrics = list(protocol.read_metrics())
        assert len(read_metrics) == 3
        assert read_metrics[0].metrics["rps"] == 1000.0
        assert read_metrics[1].metrics["rps"] == 1100.0

    def test_results_write_read(self):
        """Test results file operations."""
        protocol = FileProtocol("test_task", self.tmp_path)

        results = BenchmarkResults(
            method="perf-set",
            source="valkey",
            specifier="unstable",
            commit_hash="abc123",
            score=1000.0,
            end_time="2024-01-01T12:00:00",
            data={"rps": [1000.0, 950.0, 900.0], "latency": [2.5, 2.3, 2.1]},
        )

        protocol.write_results(results)
        read_results = protocol.read_results()

        assert read_results is not None
        assert read_results.method == "perf-set"
        assert read_results.score == 1000.0

    def test_multiple_readers(self):
        """Test multiple readers can access metrics simultaneously."""
        protocol = FileProtocol("test_task", self.tmp_path)

        # Write some metrics first
        for i in range(10):
            metric = MetricData(metrics={"rps": 1000.0 + i, "latency_ms": 2.0})
            protocol.append_metric(metric)

        # Multiple readers should all get the same data
        def read_metrics():
            return list(protocol.read_metrics())

        threads = [Thread(target=read_metrics) for _ in range(3)]
        results = []

        def store_result(func):
            results.append(func())

        threads = [Thread(target=store_result, args=(read_metrics,)) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All readers should get the same 10 metrics
        assert len(results) == 3
        for result in results:
            assert len(result) == 10

    def test_cleanup(self):
        """Test file cleanup."""
        protocol = FileProtocol("test_task", self.tmp_path)

        # Create some files
        protocol.write_status(BenchmarkStatus(steps_total=100))
        protocol.append_metric(MetricData(metrics={"rps": 1000.0, "latency_ms": 2.0}))

        assert protocol.status_file.exists()
        assert protocol.metrics_file.exists()

        protocol.cleanup()

        assert not protocol.status_file.exists()
        assert not protocol.metrics_file.exists()
        assert not protocol.work_dir.exists()

    def test_metrics_file_format(self):
        """Test that metrics file format is valid JSONL."""
        protocol = FileProtocol("format_test", self.tmp_path)

        # Write several metrics
        for i in range(5):
            metric = MetricData(
                metrics={"rps": 1000.0 + i, "latency_ms": 2.0 + (i * 0.1)}
            )
            protocol.append_metric(metric)

        # Verify file format by reading raw lines
        with open(protocol.metrics_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 5

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "timestamp" in data
            assert "metrics" in data
            assert "rps" in data["metrics"]
            assert "latency_ms" in data["metrics"]

    def test_status_heartbeat_updates(self):
        """Test frequent status updates don't corrupt file."""
        protocol = FileProtocol("heartbeat_test", self.tmp_path)

        status = BenchmarkStatus(steps_total=100)
        status.state = "running"
        status.pid = 12345

        # Rapid heartbeat updates
        for _ in range(20):
            protocol.write_status(status)
            time.sleep(0.01)  # 100Hz updates

        # Should still be readable
        final_status = protocol.read_status()
        assert final_status is not None
        assert final_status.state == "running"
        assert final_status.pid == 12345

    def test_automatic_heartbeat(self):
        """Test that heartbeat is automatically updated on write_status."""
        protocol = FileProtocol("auto_heartbeat_test", self.tmp_path)

        status = BenchmarkStatus(steps_total=100)
        status.state = "running"
        status.pid = 12345

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.heartbeat is not None
        assert read_status.heartbeat > 0

    def test_metrics_copy_to_results(self):
        """Test that metrics file is copied to results folder."""
        protocol = FileProtocol("copy_test", self.tmp_path)

        # Write some metrics
        for i in range(3):
            metric = MetricData(metrics={"rps": 1000.0 + i, "latency_ms": 2.0})
            protocol.append_metric(metric)

        results = BenchmarkResults(
            method="perf-set",
            source="valkey",
            specifier="unstable",
            commit_hash="abc123",
            score=1000.0,
            end_time="2024-01-01T12:00:00",
            data={"test": "data"},
        )

        protocol.write_results(results)

        # Check if metrics file was copied (using mocked results dir)
        results_dir = src.file_protocol.CONDUCTRESS_RESULTS
        copied_files = list(results_dir.glob("*.jsonl"))
        assert len(copied_files) == 1
        assert "perf-set_valkey_unstable" in copied_files[0].name

        # Verify content
        with open(copied_files[0], encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_progress_tracking(self):
        """Test progress tracking in status."""
        protocol = FileProtocol("progress_test", self.tmp_path)

        status = BenchmarkStatus(steps_total=10)
        status.state = "running"
        status.pid = 12345
        status.steps_completed = 3

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.steps_completed == 3
        assert read_status.steps_total == 10
        assert read_status.heartbeat is not None  # Should be auto-added
