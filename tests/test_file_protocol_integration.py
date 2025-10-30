"""Tests for file protocol integration with task runners."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ServerInfo
from src.file_protocol import BenchmarkStatus, FileProtocol, MetricData
from src.task_runner import TaskRunner
from src.tasks.task_perf_benchmark import PerfTaskRunner


class TestPerfTaskRunnerIntegration:
    """Test file protocol integration with PerfTaskRunner."""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        """Provide temporary directory for tests."""
        return tmp_path

    def test_task_runner_creates_file_protocol(self):
        """Test that PerfTaskRunner creates file protocol instance."""
        task_runner = PerfTaskRunner(
            task_name="test_task",
            server_infos=[ServerInfo(ip="127.0.0.1")],
            binary_source="valkey",
            specifier="unstable",
            io_threads=1,
            valsize=256,
            pipelining=1,
            test="set",
            warmup=5,
            duration=10,
            preload_keys=False,
            has_expire=False,
            sample_rate=-1,
        )

        assert hasattr(task_runner, "file_protocol")
        assert isinstance(task_runner.file_protocol, FileProtocol)
        assert task_runner.file_protocol.task_id == "test_task"

    def test_task_runner_file_protocol_methods(self, tmp_dir):
        """Test that task runner has file protocol methods available."""
        task_runner = PerfTaskRunner(
            task_name="method_test",
            server_infos=[ServerInfo(ip="127.0.0.1")],
            binary_source="valkey",
            specifier="unstable",
            io_threads=1,
            valsize=256,
            pipelining=1,
            test="set",
            warmup=0,
            duration=1,
            preload_keys=False,
            has_expire=False,
            sample_rate=-1,
        )

        # Override to use temp directory
        task_runner.file_protocol = FileProtocol("method_test", tmp_dir)

        # Test status writing
        status = BenchmarkStatus(steps_total=100, task_type="test")
        status.state = "running"
        status.pid = 12345
        task_runner.file_protocol.write_status(status)

        read_status = task_runner.file_protocol.read_status()
        assert read_status
        assert read_status.state == "running"
        assert read_status.pid == 12345

        # Test metric writing
        metric = MetricData(metrics={"rps": 1000.0, "latency_ms": 2.5})
        task_runner.file_protocol.append_metric(metric)

        metrics = task_runner.file_protocol.read_metrics()
        assert len(metrics) == 1
        assert metrics[0].metrics["rps"] == 1000.0

    def test_metric_data_format(self, tmp_dir):
        """Test that metric data is written in correct format."""
        protocol = FileProtocol("format_test", tmp_dir)

        # Simulate what PerfTaskRunner writes
        metric = MetricData(metrics={"rps": 1500.75, "latency_ms": 2.345})
        protocol.append_metric(metric)

        # Read back and verify format
        metrics = list(protocol.read_metrics())
        assert len(metrics) == 1
        assert metrics[0].metrics["rps"] == 1500.75
        assert metrics[0].metrics["latency_ms"] == 2.345


class TestTaskRunnerCleanup:
    """Test task runner cleanup functionality."""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        """Provide temporary directory for tests."""
        return tmp_path

    @pytest.mark.asyncio
    @patch("src.task_runner.SERVERS", [ServerInfo(ip="127.0.0.1")])
    async def test_task_runner_cleans_up_files(self, tmp_dir):
        """Test that TaskRunner cleans up file protocol files after task completion."""

        # Create a mock task that uses file protocol
        mock_task_data = MagicMock()
        mock_task_data.replicas = 0

        # Create a mock task runner with file protocol
        mock_task_runner = MagicMock()
        mock_task_runner.run = AsyncMock()
        mock_task_runner.file_protocol = FileProtocol("cleanup_test", tmp_dir)

        # Write some test files
        status = BenchmarkStatus(steps_total=100, task_type="test")
        status.state = "running"
        status.pid = 12345
        mock_task_runner.file_protocol.write_status(status)
        mock_task_runner.file_protocol.append_metric(MetricData(metrics={"rps": 1000.0, "latency_ms": 2.0}))

        # Verify files exist
        assert mock_task_runner.file_protocol.status_file.exists()
        assert mock_task_runner.file_protocol.metrics_file.exists()

        mock_task_data.prepare_task_runner.return_value = mock_task_runner

        # Run task through TaskRunner
        task_runner = TaskRunner()
        await task_runner._TaskRunner__run_task(mock_task_data)

        # Verify cleanup was called and files are gone
        assert not mock_task_runner.file_protocol.status_file.exists()
        assert not mock_task_runner.file_protocol.metrics_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, tmp_dir):
        """Test that cleanup happens even when task runner raises exception."""

        # Create a mock task that raises an exception
        mock_task_data = MagicMock()
        mock_task_data.replicas = 0

        mock_task_runner = MagicMock()
        mock_task_runner.run = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_task_runner.file_protocol = FileProtocol("exception_test", tmp_dir)

        # Write test files
        status = BenchmarkStatus(steps_total=100, task_type="test")
        status.state = "running"
        status.pid = 12345
        mock_task_runner.file_protocol.write_status(status)

        assert mock_task_runner.file_protocol.status_file.exists()

        mock_task_data.prepare_task_runner.return_value = mock_task_runner

        # Run task and expect exception
        task_runner = TaskRunner()
        with pytest.raises(RuntimeError, match="Test error"):
            await task_runner._TaskRunner__run_task(mock_task_data)

        # Verify cleanup still happened
        assert not mock_task_runner.file_protocol.status_file.exists()


class TestTUIStatusIntegration:
    """Test TUI status tab integration with file protocol."""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        """Provide temporary directory for tests."""
        return tmp_path

    def test_tui_status_data_processing(self, tmp_dir):
        """Test that TUI can process status data from file protocol."""
        # Create test data
        task_id = "2024.01.01_12.00.00.000000_set_perf"
        protocol = FileProtocol(task_id, tmp_dir)

        # Write status and metrics
        status = BenchmarkStatus(steps_total=10, task_type="test")
        status.state = "running"
        status.pid = 12345
        status.steps_completed = 3
        protocol.write_status(status)

        for i in range(3):
            metric = MetricData(metrics={"rps": 1000.0 + i * 10, "latency_ms": 2.0 + i * 0.1})
            protocol.append_metric(metric)

        # Test that we can read the data back
        read_status = protocol.read_status()
        assert read_status
        assert read_status.state == "running"
        assert read_status.pid == 12345

        metrics = list(protocol.read_metrics())
        assert len(metrics) == 3
        assert metrics[-1].metrics["rps"] == 1020.0  # Last metric should have highest RPS

        # Verify the data format matches what TUI expects
        progress = "N/A"
        if read_status.steps_total and read_status.steps_completed is not None:
            pct = (read_status.steps_completed / read_status.steps_total) * 100
            progress = f"{pct:.0f}% ({read_status.steps_completed}/{read_status.steps_total})"

        task_data = {
            "task_id": task_id,
            "state": read_status.state,
            "pid": read_status.pid,
            "latest_rps": f"{metrics[-1].metrics['rps']:.0f}" if metrics else "N/A",
            "metrics_count": len(metrics),
            "progress": progress,
        }

        assert task_data["state"] == "running"
        assert task_data["latest_rps"] == "1020"
        assert task_data["metrics_count"] == 3
        assert task_data["progress"] == "30% (3/10)"
