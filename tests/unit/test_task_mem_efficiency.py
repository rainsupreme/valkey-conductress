"""Tests for memory efficiency task module."""

import asyncio
import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ServerInfo
from src.file_protocol import BenchmarkResults
from src.server import Server
from src.tasks.task_mem_efficiency import MemTaskData, MemTaskRunner


class TestMemTaskData:
    """Test MemTaskData class."""

    @patch('src.config.REPO_NAMES', ["valkey", "test_repo"])
    @patch('src.task_queue.config.REPO_NAMES', ["valkey", "test_repo"])
    def test_init_valid_data(self):
        """Test initialization with valid data."""
        task = MemTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="test",
            requirements={},
            make_args="-O2 -g",
            type="set",
            val_sizes=[64, 128, 256],
            has_expire=False,
        )
        assert task.type == "set"
        assert task.val_sizes == [64, 128, 256]
        assert task.has_expire is False

    @patch('src.config.REPO_NAMES', ["valkey", "test_repo"])
    @patch('src.task_queue.config.REPO_NAMES', ["valkey", "test_repo"])
    def test_short_description_single_size(self):
        """Test short description with single value size."""
        task = MemTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="test",
            requirements={},
            make_args="-O2 -g",
            type="set",
            val_sizes=[1024],
            has_expire=False,
        )
        desc = task.short_description()
        assert "set" in desc
        assert "1KB" in desc
        assert "expiration" not in desc

    @patch('src.config.REPO_NAMES', ["valkey", "test_repo"])
    @patch('src.task_queue.config.REPO_NAMES', ["valkey", "test_repo"])
    def test_short_description_multiple_sizes(self):
        """Test short description with multiple value sizes."""
        task = MemTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="test",
            requirements={},
            make_args="-O2 -g",
            type="zadd",
            val_sizes=[64, 128, 256],
            has_expire=True,
        )
        desc = task.short_description()
        assert "zadd" in desc
        assert "3 sizes" in desc
        assert "with expiration" in desc

    @patch('src.config.REPO_NAMES', ["valkey", "test_repo"])
    @patch('src.task_queue.config.REPO_NAMES', ["valkey", "test_repo"])
    def test_prepare_task_runner(self):
        """Test task runner preparation."""
        task = MemTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="test",
            requirements={},
            make_args="-O2 -g",
            type="set",
            val_sizes=[64],
            has_expire=False,
        )
        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task.prepare_task_runner([server_info])
        assert isinstance(runner, MemTaskRunner)
        assert runner.server_ip == "127.0.0.1"
        assert runner.test == "set"


class TestMemTaskRunner:
    """Test MemTaskRunner class."""

    @pytest.fixture
    def mock_server(self):
        """Mock server for testing."""
        server = MagicMock(spec=Server)
        server.get_available_cpu_count = AsyncMock(return_value=4)
        server.kill_all_valkey_instances_on_host = AsyncMock()
        server.ensure_binary_cached = AsyncMock(return_value=Path("/mock/binary"))
        server.get_build_hash = MagicMock(return_value="abc123")
        server.info = AsyncMock(return_value={"used_memory": "1000000"})
        server.run_valkey_command_over_keyspace = AsyncMock()
        server.count_items_expires = AsyncMock(return_value=(5000000, 0))
        return server

    @pytest.fixture
    def runner(self):
        """Create a test runner."""
        return MemTaskRunner(
            task_name="test_task",
            server_ip="127.0.0.1",
            source="valkey",
            specifier="unstable",
            test="set",
            val_sizes=[64, 128],
            has_expire=False,
            make_args="-O2 -g",
            note="test note",
        )

    def test_init_valid_test(self, runner):
        """Test initialization with valid test type."""
        assert runner.test == "set"
        assert runner.val_sizes == [64, 128]
        assert runner.has_expire is False

    def test_init_invalid_test(self):
        """Test initialization with invalid test type."""
        with pytest.raises(AssertionError, match="Test invalid is not supported"):
            MemTaskRunner(
                task_name="test_task",
                server_ip="127.0.0.1",
                source="valkey",
                specifier="unstable",
                test="invalid",
                val_sizes=[64],
                has_expire=False,
                make_args="-O2 -g",
                note="test note",
            )

    @patch("src.tasks.task_mem_efficiency.Server")
    @patch("src.tasks.task_mem_efficiency.print_pretty_header")
    @pytest.mark.asyncio
    async def test_run_basic_flow(self, mock_print, mock_server_class, runner, mock_server):
        """Test basic run flow."""
        # Ensure Server() instantiation returns mock_server
        mock_server_class.return_value = mock_server
        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        mock_server_class.getNumCPUs = MagicMock(return_value=1)

        # Mock file protocol
        runner.file_protocol = MagicMock()

        # Mock test_single_size_overhead to return simple results
        async def mock_test_overhead(val_size, port, semaphore):
            return {
                "val_size": val_size,
                "per_item_overhead": 10.0 + val_size * 0.1,
                "before_memory": {"used_memory": "1000000"},
                "after_memory": {"used_memory": "2000000"},
                "has_expire": False,
                "key_size": 16,
                "per_key_size": 20.0,
            }

        runner.test_single_size_overhead = mock_test_overhead
        runner.plot = MagicMock()

        await runner.run()

        # Verify server interactions
        mock_server.get_available_cpu_count.assert_called_once()
        assert mock_server.kill_all_valkey_instances_on_host.call_count == 2
        mock_server.ensure_binary_cached.assert_called_once()

        # Verify file protocol calls
        assert runner.file_protocol.write_status.call_count >= 2
        runner.file_protocol.write_results.assert_called_once()

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_test_single_size_overhead(self, mock_server_class, runner, mock_server):
        """Test single size overhead calculation."""
        mock_server_class.with_path = AsyncMock(return_value=mock_server)

        # Setup memory info responses
        mock_server.info.side_effect = [
            {"used_memory": "1000000"},  # before
            {"used_memory": "6000000"},  # after (5MB increase for 5M items)
        ]

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        semaphore = asyncio.Semaphore(1)
        result = await runner.test_single_size_overhead(64, 6379, semaphore)

        assert result["val_size"] == 64
        assert result["key_size"] == 16
        assert "per_item_overhead" in result
        assert "per_key_size" in result

        # Verify server calls
        mock_server.info.assert_called()
        mock_server.run_valkey_command_over_keyspace.assert_called()
        mock_server.count_items_expires.assert_called_once()

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_expiration_with_non_set_command(self, mock_server_class, runner, mock_server):
        """Test that expiration with non-set command logs error."""
        runner.test = "zadd"
        runner.has_expire = True

        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        mock_server.info.side_effect = [
            {"used_memory": "1000000"},
            {"used_memory": "2000000"},
        ]
        mock_server.count_items_expires = AsyncMock(return_value=(5000000, 0))  # No expiration for non-set

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        with patch("src.tasks.task_mem_efficiency.logger") as mock_logger:
            semaphore = asyncio.Semaphore(1)
            with pytest.raises(AssertionError):  # Should fail assertion since expire_count != count
                await runner.test_single_size_overhead(64, 6379, semaphore)

            mock_logger.error.assert_called_once_with(
                "Expiration is only supported for sets, skipping expiration test."
            )

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_expiration_with_set_command(self, mock_server_class, runner, mock_server):
        """Test expiration works with set command."""
        runner.has_expire = True

        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        mock_server.info.side_effect = [
            {"used_memory": "1000000"},
            {"used_memory": "2000000"},
        ]
        mock_server.count_items_expires.return_value = (
            5000000,
            5000000,
        )  # All items have expiry

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        semaphore = asyncio.Semaphore(1)
        _ = await runner.test_single_size_overhead(64, 6379, semaphore)

        # Should call expire command
        assert mock_server.run_valkey_command_over_keyspace.call_count == 2
        expire_call = mock_server.run_valkey_command_over_keyspace.call_args_list[1]
        assert "EXPIRE" in expire_call[0][1]

    def test_plot_basic(self, runner):
        """Test basic plotting functionality."""
        with patch("src.tasks.task_mem_efficiency.plt") as mock_plt:
            efficiency_map = {64: 10.5, 128: 12.3}
            runner.plot(efficiency_map)

            mock_plt.clear_terminal.assert_called_once()
            mock_plt.clear_figure.assert_called_once()
            mock_plt.plot.assert_called_once()
            mock_plt.show.assert_called_once()

    def test_plot_with_missing_data(self, runner):
        """Test plotting with missing data points."""
        with patch("src.tasks.task_mem_efficiency.plt") as mock_plt:
            # Only one size has data
            efficiency_map = {64: 10.5}
            runner.plot(efficiency_map)

            # Should still plot (with None values)
            mock_plt.plot.assert_called_once()
            plot_args = mock_plt.plot.call_args[0]
            overheads = plot_args[1]
            assert None in overheads  # Missing data should be None

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_memory_calculation_accuracy(self, mock_server_class, runner, mock_server):
        """Test memory calculation accuracy."""
        mock_server_class.with_path = AsyncMock(return_value=mock_server)

        # Precise memory values for calculation testing
        before_mem = 1000000  # 1MB
        after_mem = 6000000  # 6MB (5MB increase)
        count = 5000000  # 5M items
        val_size = 64
        key_size = 16

        mock_server.info.side_effect = [
            {"used_memory": str(before_mem)},
            {"used_memory": str(after_mem)},
        ]

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        semaphore = asyncio.Semaphore(1)
        result = await runner.test_single_size_overhead(val_size, 6379, semaphore)

        expected_per_key = (after_mem - before_mem) / count  # 1.0 bytes per key
        expected_overhead = expected_per_key - val_size - key_size  # Should be negative

        assert result["per_key_size"] == expected_per_key
        assert result["per_item_overhead"] == expected_overhead

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_assertion_failures(self, mock_server_class, runner, mock_server):
        """Test assertion failures in count verification."""
        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        mock_server.info.side_effect = [
            {"used_memory": "1000000"},
            {"used_memory": "2000000"},
        ]

        # Wrong item count
        mock_server.count_items_expires.return_value = (4000000, 0)  # Should be 5M

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        semaphore = asyncio.Semaphore(1)
        with pytest.raises(AssertionError):
            await runner.test_single_size_overhead(64, 6379, semaphore)

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_expire_count_assertion(self, mock_server_class, runner, mock_server):
        """Test expire count assertion."""
        runner.has_expire = True

        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        mock_server.info.side_effect = [
            {"used_memory": "1000000"},
            {"used_memory": "2000000"},
        ]

        # Wrong expire count when expiration is enabled
        mock_server.count_items_expires.return_value = (
            5000000,
            0,
        )  # Should have expires

        # Set cached_binary_path as it would be set in run()
        runner.cached_binary_path = Path("/mock/binary")

        semaphore = asyncio.Semaphore(1)
        with pytest.raises(AssertionError):
            await runner.test_single_size_overhead(64, 6379, semaphore)

    def test_status_initialization(self, runner):
        """Test status initialization."""
        assert runner.status.steps_total == len(runner.val_sizes) + 2
        assert runner.status.state == "starting"
        assert runner.status.steps_completed == 0

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_cpu_limit_handling(self, mock_server_class, runner, mock_server):
        """Test CPU limit handling."""
        # Test with high CPU count
        mock_server.get_available_cpu_count.return_value = 20
        mock_server_class.return_value = mock_server
        mock_server_class.getNumCPUs = MagicMock(return_value=1)

        runner.file_protocol = MagicMock()
        runner.test_single_size_overhead = AsyncMock(return_value={"val_size": 64, "per_item_overhead": 10.0})
        runner.plot = MagicMock()

        await runner.run()

        # Should be limited to 9 CPUs
        # Verify semaphore was created (indirectly through successful execution)
        assert runner.test_single_size_overhead.call_count == len(runner.val_sizes)

    def test_results_data_structure(self, runner):
        """Test results data structure format."""
        # Mock the completion flow
        runner.commit_hash = "test_hash"
        results = [
            {"val_size": 64, "per_item_overhead": 10.0},
            {"val_size": 128, "per_item_overhead": 12.0},
        ]

        # Simulate results creation
        completion_time = datetime.datetime.now()
        results_data = BenchmarkResults(
            method=f"mem-{runner.test}",
            source=runner.source,
            specifier=runner.specifier,
            commit_hash=runner.commit_hash,
            score=results[0]["per_item_overhead"],  # Single result score
            end_time=completion_time,
            data=results,
            make_args="-O2 -g",
        )

        assert results_data.method == "mem-set"
        assert results_data.source == "valkey"
        assert results_data.score == 10.0
        assert len(results_data.data) == 2

    @patch("src.tasks.task_mem_efficiency.Server")
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, mock_server_class, runner, mock_server):
        """Test concurrent execution of multiple size tests."""
        mock_server_class.with_path = AsyncMock(return_value=mock_server)
        
        # Mock Server constructor to return mock_server
        mock_server_instance = MagicMock()
        mock_server_instance.get_available_cpu_count = AsyncMock(return_value=2)
        mock_server_instance.kill_all_valkey_instances_on_host = AsyncMock()
        mock_server_instance.ensure_binary_cached = AsyncMock(return_value=Path("/mock/binary"))
        mock_server_class.return_value = mock_server_instance
        mock_server_class.getNumCPUs = MagicMock(return_value=1)

        call_order = []

        async def track_calls(val_size, _, semaphore):
            async with semaphore:
                call_order.append(f"start_{val_size}")
                await asyncio.sleep(0.01)  # Simulate work
                call_order.append(f"end_{val_size}")
                return {"val_size": val_size, "per_item_overhead": 10.0}

        runner.test_single_size_overhead = track_calls
        runner.file_protocol = MagicMock()
        runner.plot = MagicMock()

        # Use small semaphore to test concurrency
        with patch("asyncio.Semaphore") as mock_semaphore:
            mock_semaphore.return_value = asyncio.Semaphore(1)
            await runner.run()

        # Verify both sizes were processed
        assert len([x for x in call_order if x.startswith("start_")]) == 2
        assert len([x for x in call_order if x.startswith("end_")]) == 2
