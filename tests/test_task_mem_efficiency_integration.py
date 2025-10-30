import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import ServerInfo
from src.file_protocol import FileProtocol
from src.tasks.task_mem_efficiency import MemTaskData


class TestMemTaskIntegration:
    """Integration tests for MemTaskRunner with real server instances."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("src.file_protocol.CONDUCTRESS_OUTPUT", tmp_path / "output.jsonl"), \
                 patch("src.file_protocol.CONDUCTRESS_RESULTS", tmp_path / "results"):
                yield tmp_path

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @patch("src.tasks.task_mem_efficiency.MEM_TEST_ITEM_COUNT", 50_000)
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, temp_dir):
        """Test complete MemTaskRunner workflow with real valkey binary."""
        task_data = MemTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="integration test",
            requirements={},
            type="set",
            val_sizes=[32, 64],
            has_expire=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.type}_mem"
        runner.file_protocol = FileProtocol(task_name, temp_dir)

        with patch("src.tasks.task_mem_efficiency.plt"):
            await runner.run()

        # Verify results were written to legacy output
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        # Verify results structure
        assert "data" in results
        assert len(results["data"]) == 2  # Two value sizes

        for result in results["data"]:
            assert "val_size" in result
            assert "per_item_overhead" in result
            assert result["val_size"] in [32, 64]
            assert isinstance(result["per_item_overhead"], (int, float))

        # Verify status shows completion
        status_file = runner.file_protocol.status_file
        assert status_file.exists()

        with open(status_file) as f:
            status = json.load(f)
        assert status["state"] == "completed"

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @patch("src.tasks.task_mem_efficiency.MEM_TEST_ITEM_COUNT", 50_000)
    @pytest.mark.asyncio
    async def test_expiration_workflow_integration(self, temp_dir):
        """Test MemTaskRunner with expiration enabled."""
        task_data = MemTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="expiration test",
            requirements={},
            type="set",
            val_sizes=[32],
            has_expire=True,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.type}_mem"
        runner.file_protocol = FileProtocol(task_name, temp_dir)

        with patch("src.tasks.task_mem_efficiency.plt"):
            await runner.run()

        # Verify results include expiration data
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        result = results["data"][0]
        assert result["has_expire"] is True

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @patch("src.tasks.task_mem_efficiency.MEM_TEST_ITEM_COUNT", 50_000)
    @pytest.mark.asyncio
    async def test_concurrent_size_testing(self, temp_dir):
        """Test MemTaskRunner handles multiple sizes concurrently."""
        task_data = MemTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="concurrent test",
            requirements={},
            type="set",
            val_sizes=[16, 32, 64, 128],
            has_expire=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.type}_mem"
        runner.file_protocol = FileProtocol(task_name, temp_dir)

        with patch("src.tasks.task_mem_efficiency.plt"):
            await runner.run()

        # Verify all sizes were tested
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])

        assert len(results["data"]) == 4
        tested_sizes = {r["val_size"] for r in results["data"]}
        assert tested_sizes == {16, 32, 64, 128}

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_dir):
        """Test MemTaskRunner error handling with invalid configuration."""
        with patch("src.config.REPO_NAMES", ["valkey"]), patch("src.task_queue.config.REPO_NAMES", ["valkey"]):
            task_data = MemTaskData(
                source="valkey",
                specifier="8.0",
                replicas=0,
                note="error test",
                requirements={},
                type="set",
                val_sizes=[32],
                has_expire=False,
            )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.type}_mem"
        runner.file_protocol = FileProtocol(task_name, temp_dir)

        # Mock server to raise exception during binary caching
        with patch("src.server.Server.ensure_binary_cached", side_effect=Exception("Binary not found")):
            with pytest.raises(Exception):
                await runner.run()

        # Verify status file was created (even if error occurred early)
        status_file = runner.file_protocol.status_file
        assert status_file.exists(), "Status file should exist even after error"

        with open(status_file) as f:
            status = json.load(f)
        # The status might still be "starting" or "running" since the error occurs early
        # The important thing is that the exception was raised and caught
        assert "state" in status
        assert "start_time" in status

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    def test_task_data_serialization_integration(self, temp_dir):
        """Test MemTaskData serialization/deserialization."""
        original_task = MemTaskData(
            source="valkey",
            specifier="v7.2.0",
            replicas=1,
            note="serialization test",
            requirements={"memory": "4GB"},
            type="zadd",
            val_sizes=[64, 128, 256],
            has_expire=True,
        )

        # Test round-trip serialization
        task_file = temp_dir / "test_task.json"
        original_task.save_to_file(task_file)
        loaded_task = MemTaskData.from_file(task_file)

        # Verify all fields preserved
        assert loaded_task.source == original_task.source
        assert loaded_task.specifier == original_task.specifier
        assert loaded_task.type == original_task.type
        assert loaded_task.val_sizes == original_task.val_sizes
        assert loaded_task.has_expire == original_task.has_expire
        assert loaded_task.replicas == original_task.replicas
        assert loaded_task.note == original_task.note
        assert loaded_task.requirements == original_task.requirements

        # Verify loaded task can create runner
        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = loaded_task.prepare_task_runner([server_info])
        assert runner.source == "valkey"
        assert runner.test == "zadd"
        assert runner.val_sizes == [64, 128, 256]

    @patch("src.config.REPO_NAMES", ["valkey"])
    @patch("src.task_queue.config.REPO_NAMES", ["valkey"])
    @patch("src.tasks.task_mem_efficiency.MEM_TEST_ITEM_COUNT", 50_000)
    @pytest.mark.asyncio
    async def test_file_protocol_integration(self, temp_dir):
        """Test FileProtocol integration with MemTaskRunner."""
        task_data = MemTaskData(
            source="valkey",
            specifier="8.0",
            replicas=0,
            note="file protocol test",
            requirements={},
            type="set",
            val_sizes=[32],
            has_expire=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_{task_data.type}_mem"
        runner.file_protocol = FileProtocol(task_name, temp_dir)

        with patch("src.tasks.task_mem_efficiency.plt"):
            await runner.run()

        # Verify expected files were created
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists(), "Output file was not created"
        assert runner.file_protocol.status_file.exists(), "Status file was not created"

        # Verify file contents are valid
        import json

        # Output file
        with open(output_file) as f:
            results = json.loads(f.readlines()[-1])
        assert "data" in results
        assert "method" in results

        # Status file
        with open(runner.file_protocol.status_file) as f:
            status = json.load(f)
        assert "state" in status
        assert "start_time" in status
