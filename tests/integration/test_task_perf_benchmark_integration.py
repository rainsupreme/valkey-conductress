import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import ServerInfo
from src.file_protocol import FileProtocol
from src.tasks.task_perf_benchmark import PerfTaskData


class TestPerfTaskIntegration:
    """Integration tests for PerfTaskRunner with real server instances."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("src.file_protocol.CONDUCTRESS_OUTPUT", tmp_path / "output.jsonl"), patch(
                "src.file_protocol.CONDUCTRESS_RESULTS", tmp_path / "results"
            ):
                yield tmp_path

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, temp_dir):
        """Test complete PerfTaskRunner workflow with real valkey binary."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="integration test",
            requirements={},
            make_args="",
            test="set",
            val_size=64,
            io_threads=4,
            pipelining=16,
            warmup=1,
            duration=2,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # Verify results were written
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        # Verify results structure
        assert "data" in results
        assert "avg_rps" in results["data"]
        assert results["data"]["io-threads"] == 4
        assert results["data"]["pipeline"] == 16
        assert results["data"]["size"] == 64
        assert isinstance(results["data"]["avg_rps"], (int, float))
        assert results["data"]["avg_rps"] > 0

        # Verify lscpu data is collected
        assert "lscpu" in results["data"]
        assert isinstance(results["data"]["lscpu"], str)
        assert len(results["data"]["lscpu"]) > 0

        # Verify feature toggles are present
        assert "features" in results
        assert isinstance(results["features"], dict)
        assert "PIN_VALKEY_THREADS" in results["features"]
        assert "ENABLE_CPU_CONSISTENCY_MODE" in results["features"]

        # Verify status shows completion
        status_file = runner.file_protocol.status_file
        assert status_file.exists()

        with open(status_file) as f:
            status = json.load(f)
        assert status["state"] == "completed"

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.parametrize("command_name", ["set", "get", "sadd", "hset", "zadd", "zrank", "zcount"])
    @pytest.mark.asyncio
    async def test_all_commands_integration(self, temp_dir, command_name):
        """Test PerfTaskRunner with all command types."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note=f"{command_name} test",
            requirements={},
            make_args="",
            test=command_name,
            val_size=64,
            io_threads=2,
            pipelining=16,
            warmup=1,
            duration=2,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        assert results["method"] == f"perf-{command_name}"
        assert results["data"]["avg_rps"] > 0

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_no_preload_integration(self, temp_dir):
        """Test PerfTaskRunner with GET command without preloading."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="no preload test",
            requirements={},
            make_args="",
            test="get",
            val_size=64,
            io_threads=2,
            pipelining=16,
            warmup=1,
            duration=2,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        assert results["method"] == "perf-get"
        assert results["data"]["preload_keys"] is False
        assert results["data"]["avg_rps"] > 0

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_expiration_workflow_integration(self, temp_dir):
        """Test PerfTaskRunner with expiration enabled."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="expiration test",
            requirements={},
            make_args="",
            test="set",
            val_size=64,
            io_threads=2,
            pipelining=16,
            warmup=1,
            duration=2,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=True,
            preload_keys=True,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        assert results["data"]["has_expire"] is True
        assert results["data"]["avg_rps"] > 0

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_heartbeat_and_progress_integration(self, temp_dir):
        """Test that heartbeat and progress steps are tracked correctly."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="heartbeat test",
            requirements={},
            make_args="",
            test="set",
            val_size=64,
            io_threads=2,
            pipelining=16,
            warmup=1,
            duration=2,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        with open(runner.file_protocol.status_file) as f:
            final_status = json.load(f)

        assert final_status["state"] == "completed"
        assert final_status["heartbeat"] is not None
        assert final_status["start_time"] is not None
        assert final_status["end_time"] is not None
        assert final_status["steps_total"] == 3
        assert final_status["steps_completed"] == 3
        assert final_status["end_time"] > final_status["start_time"]

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_dir):
        """Test PerfTaskRunner error handling with invalid configuration."""
        with patch("src.config.REPO_NAMES", ["valkey"]), patch(
            "src.task_queue.config.REPO_NAMES", ["valkey"]
        ):
            task_data = PerfTaskData(
                source="valkey",
                specifier="8.0",
                replicas=0,
                note="error test",
                requirements={},
                make_args="",
                test="set",
                val_size=64,
                io_threads=2,
                pipelining=16,
                warmup=1,
                duration=1,
                profiling_sample_rate=0,
                perf_stat_enabled=False,
                has_expire=False,
                preload_keys=False,
            )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.test}_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        with patch("src.server.Server.ensure_binary_cached", side_effect=Exception("Binary not found")):
            with pytest.raises(Exception):
                await runner.run()

        status_file = runner.file_protocol.status_file
        assert status_file.exists()

        with open(status_file) as f:
            status = json.load(f)
        assert "state" in status
        assert "start_time" in status

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    def test_task_data_serialization_integration(self, temp_dir):
        """Test PerfTaskData serialization/deserialization."""
        original_task = PerfTaskData(
            source="valkey",
            specifier="v7.2.0",
            replicas=1,
            note="serialization test",
            requirements={"memory": "4GB"},
            make_args="-O3 -march=native",
            test="get",
            val_size=256,
            io_threads=8,
            pipelining=32,
            warmup=10,
            duration=60,
            profiling_sample_rate=99,
            perf_stat_enabled=False,
            has_expire=True,
            preload_keys=True,
        )

        task_file = temp_dir / "test_task.json"
        original_task.save_to_file(task_file)
        loaded_task = PerfTaskData.from_file(task_file)
        assert isinstance(loaded_task, PerfTaskData)

        assert loaded_task.source == original_task.source
        assert loaded_task.specifier == original_task.specifier
        assert loaded_task.test == original_task.test
        assert loaded_task.val_size == original_task.val_size
        assert loaded_task.io_threads == original_task.io_threads
        assert loaded_task.pipelining == original_task.pipelining
        assert loaded_task.warmup == original_task.warmup
        assert loaded_task.duration == original_task.duration
        assert loaded_task.profiling_sample_rate == original_task.profiling_sample_rate
        assert loaded_task.has_expire == original_task.has_expire
        assert loaded_task.preload_keys == original_task.preload_keys

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = loaded_task.prepare_task_runner([server_info])
        assert runner.binary_source == "valkey"
        assert runner.test.name == "get"
        assert runner.valsize == 256
