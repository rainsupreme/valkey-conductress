"""Tests for file-based communication protocol."""

import json
import time
from datetime import datetime
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
        protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)

        status = BenchmarkStatus(steps_total=100, task_type="test")
        status.state = "running"
        status.pid = 12345

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.state == "running"
        assert read_status.pid == 12345
        assert read_status.steps_total == 100
        assert read_status.task_type == "test"

    def test_metrics_append_read(self):
        """Test metrics file operations."""
        protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)

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

    def test_results_write(self):
        """Test results file operations."""
        protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)

        results = BenchmarkResults(
            method="perf-set",
            source="valkey",
            specifier="unstable",
            commit_hash="abc123",
            score=1000.0,
            end_time=datetime(2024, 1, 1, 12, 0, 0),
            data={"rps": [1000.0, 950.0, 900.0], "latency": [2.5, 2.3, 2.1]},
            make_args="-O2 -g",
            note="test note",
        )

        protocol.write_results(results)

        # Verify legacy output was written
        output_file = src.file_protocol.CONDUCTRESS_OUTPUT
        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["method"] == "perf-set"
            assert data["score"] == 1000.0
            assert data["note"] == "test note"
            assert data["make_args"] == "-O2 -g"

    def test_multiple_readers(self):
        """Test multiple readers can access metrics simultaneously."""
        protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)

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
        protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)

        # Create some files
        protocol.write_status(BenchmarkStatus(steps_total=100, task_type="test"))
        protocol.append_metric(MetricData(metrics={"rps": 1000.0, "latency_ms": 2.0}))

        assert protocol.status_file.exists()
        assert protocol.metrics_file.exists()

        protocol.cleanup()

        assert not protocol.status_file.exists()
        assert not protocol.metrics_file.exists()
        assert not protocol.work_dir.exists()

    def test_metrics_file_format(self):
        """Test that metrics file format is valid JSONL."""
        protocol = FileProtocol("format_test", role_id="client", base_dir=self.tmp_path)

        # Write several metrics
        for i in range(5):
            metric = MetricData(metrics={"rps": 1000.0 + i, "latency_ms": 2.0 + (i * 0.1)})
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
        protocol = FileProtocol("heartbeat_test", role_id="client", base_dir=self.tmp_path)

        status = BenchmarkStatus(steps_total=100, task_type="test")
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
        protocol = FileProtocol("auto_heartbeat_test", role_id="client", base_dir=self.tmp_path)

        status = BenchmarkStatus(steps_total=100, task_type="test")
        status.state = "running"
        status.pid = 12345

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.heartbeat is not None
        assert read_status.heartbeat > 0

    def test_metrics_copy_to_results(self):
        """Test that metrics file is copied to results folder."""
        protocol = FileProtocol("copy_test", role_id="client", base_dir=self.tmp_path)

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
            end_time=datetime(2024, 1, 1, 12, 0, 0),
            data={"test": "data"},
            make_args="-O2 -g",
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
        protocol = FileProtocol("progress_test", role_id="client", base_dir=self.tmp_path)

        status = BenchmarkStatus(steps_total=10, task_type="test")
        status.state = "running"
        status.pid = 12345
        status.steps_completed = 3

        protocol.write_status(status)
        read_status = protocol.read_status()

        assert read_status is not None
        assert read_status.steps_completed == 3
        assert read_status.steps_total == 10

    def test_read_metrics_caching(self):
        """Test that read_metrics caches results."""
        protocol = FileProtocol("cache_test", role_id="client", base_dir=self.tmp_path)

        protocol.append_metric(MetricData(metrics={"rps": 1000.0}))
        metrics1 = protocol.read_metrics()
        assert len(metrics1) == 1

        protocol.append_metric(MetricData(metrics={"rps": 2000.0}))
        metrics2 = protocol.read_metrics()
        assert len(metrics2) == 2
        assert metrics2[0].metrics["rps"] == 1000.0
        assert metrics2[1].metrics["rps"] == 2000.0

    def test_read_metrics_tail_behavior(self):
        """Test that read_metrics only reads new lines."""
        protocol = FileProtocol("tail_test", role_id="client", base_dir=self.tmp_path)

        protocol.append_metric(MetricData(metrics={"rps": 1000.0}))
        protocol.read_metrics()
        initial_position = protocol._last_read_position

        protocol.append_metric(MetricData(metrics={"rps": 2000.0}))
        protocol.read_metrics()

        assert protocol._last_read_position > initial_position

    def test_read_metrics_returns_same_reference(self):
        """Test that read_metrics returns reference to cache."""
        protocol = FileProtocol("ref_test", role_id="client", base_dir=self.tmp_path)

        protocol.append_metric(MetricData(metrics={"rps": 1000.0}))
        metrics1 = protocol.read_metrics()
        metrics2 = protocol.read_metrics()

        assert metrics1 is metrics2

    def test_get_active_task_ids_empty(self):
        """Test get_active_task_ids with no active tasks."""
        active_tasks = FileProtocol.get_active_task_ids(self.tmp_path)
        assert active_tasks == {}

    def test_get_active_task_ids_single_task(self):
        """Test get_active_task_ids with one active task."""
        protocol = FileProtocol("task1", role_id="client", base_dir=self.tmp_path)
        status = BenchmarkStatus(steps_total=100, task_type="perf-get", state="running", pid=12345)
        status.steps_completed = 50
        protocol.write_status(status)

        active_tasks = FileProtocol.get_active_task_ids(self.tmp_path)
        assert len(active_tasks) == 1
        assert "task1" in active_tasks
        assert active_tasks["task1"].state == "running"
        assert active_tasks["task1"].pid == 12345
        assert active_tasks["task1"].steps_completed == 50
        assert active_tasks["task1"].steps_total == 100
        assert active_tasks["task1"].task_type == "perf-get"

    def test_get_active_task_ids_multiple_tasks(self):
        """Test get_active_task_ids with multiple active tasks."""
        protocol1 = FileProtocol("task1", role_id="client", base_dir=self.tmp_path)
        status1 = BenchmarkStatus(steps_total=100, task_type="perf-get", state="running", pid=12345)
        status1.steps_completed = 50
        protocol1.write_status(status1)

        protocol2 = FileProtocol("task2", role_id="server", base_dir=self.tmp_path)
        status2 = BenchmarkStatus(steps_total=200, task_type="mem-set", state="starting", pid=67890)
        status2.steps_completed = 10
        protocol2.write_status(status2)

        active_tasks = FileProtocol.get_active_task_ids(self.tmp_path)
        assert len(active_tasks) == 2
        assert "task1" in active_tasks
        assert "task2" in active_tasks
        assert active_tasks["task1"].state == "running"
        assert active_tasks["task1"].task_type == "perf-get"
        assert active_tasks["task2"].state == "starting"
        assert active_tasks["task2"].task_type == "mem-set"

    def test_get_active_task_ids_ignores_no_status(self):
        """Test get_active_task_ids ignores directories without status files."""
        # Create directory without status file
        empty_dir = self.tmp_path / "benchmark_empty"
        empty_dir.mkdir()

        protocol = FileProtocol("task1", role_id="client", base_dir=self.tmp_path)
        status = BenchmarkStatus(steps_total=100, task_type="test", state="running", pid=12345)
        protocol.write_status(status)

        active_tasks = FileProtocol.get_active_task_ids(self.tmp_path)
        assert len(active_tasks) == 1
        assert "task1" in active_tasks
        assert "empty" not in active_tasks

    def test_role_id_in_filename(self):
        """Test that role_id is included in metrics filename."""
        protocol = FileProtocol("test_task", role_id="server_10_0_1_5_p9000", base_dir=self.tmp_path)
        
        protocol.append_metric(MetricData(metrics={"rps": 1000.0}))
        
        assert protocol.metrics_file.name == "metrics_server_10_0_1_5_p9000.jsonl"
        assert protocol.metrics_file.exists()

    def test_multiple_roles_same_task(self):
        """Test multiple roles can write to same task directory."""
        client_protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)
        server_protocol = FileProtocol("test_task", role_id="server", base_dir=self.tmp_path)
        
        client_protocol.append_metric(MetricData(metrics={"client_rps": 1000.0}, source="client"))
        server_protocol.append_metric(MetricData(metrics={"server_cpu": 85.0}, source="server"))
        
        # Both files should exist in same work directory
        assert client_protocol.work_dir == server_protocol.work_dir
        assert client_protocol.metrics_file.exists()
        assert server_protocol.metrics_file.exists()
        assert client_protocol.metrics_file != server_protocol.metrics_file

    def test_role_specific_results_copy(self):
        """Test that role_id is included in copied results filename."""
        protocol = FileProtocol("copy_test", role_id="primary_10_0_1_5_p9000", base_dir=self.tmp_path)
        
        protocol.append_metric(MetricData(metrics={"rps": 1000.0}))
        
        results = BenchmarkResults(
            method="perf-set",
            source="valkey",
            specifier="unstable",
            commit_hash="abc123",
            score=1000.0,
            end_time=datetime(2024, 1, 1, 12, 0, 0),
            data={"test": "data"},
            make_args="-O2 -g",
        )
        
        protocol.write_results(results)
        
        results_dir = src.file_protocol.CONDUCTRESS_RESULTS
        copied_files = list(results_dir.glob("*.jsonl"))
        assert len(copied_files) == 1
        assert "primary_10_0_1_5_p9000" in copied_files[0].name

    def test_different_roles_independent_caches(self):
        """Test that different role protocols have independent caches."""
        client_protocol = FileProtocol("test_task", role_id="client", base_dir=self.tmp_path)
        server_protocol = FileProtocol("test_task", role_id="server", base_dir=self.tmp_path)
        
        client_protocol.append_metric(MetricData(metrics={"client_rps": 1000.0}))
        server_protocol.append_metric(MetricData(metrics={"server_cpu": 85.0}))
        
        client_metrics = client_protocol.read_metrics()
        server_metrics = server_protocol.read_metrics()
        
        assert len(client_metrics) == 1
        assert len(server_metrics) == 1
        assert client_metrics[0].metrics["client_rps"] == 1000.0
        assert server_metrics[0].metrics["server_cpu"] == 85.0
