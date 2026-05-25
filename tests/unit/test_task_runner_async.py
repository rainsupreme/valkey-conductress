"""Async tests for TaskRunner main loop behavior."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.task_runner import TaskRunner


class _ExitLoop(Exception):
    """Raised to break out of the infinite task runner loop in tests."""

    pass


@pytest.fixture
def mock_servers():
    """Patch get_servers to return a mock server list."""
    with patch("conductress.task_runner.get_servers") as mock:
        mock.return_value = [MagicMock(ip="127.0.0.1")]
        yield mock


@pytest.fixture
def mock_queue(tmp_path):
    """Patch TaskQueue to use a controllable mock."""
    with patch("conductress.task_runner.TaskQueue") as MockQueue:
        queue = MagicMock()
        MockQueue.return_value = queue
        yield queue


@pytest.fixture
def mock_cleanup():
    """Patch orphan cleanup and server kill."""
    with (
        patch("conductress.task_runner.FileProtocol.cleanup_orphaned_tasks", return_value=0),
        patch("conductress.task_runner.Server") as MockServer,
    ):
        mock_server_instance = MagicMock()
        mock_server_instance.kill_all_valkey_instances_on_host = AsyncMock()
        MockServer.return_value = mock_server_instance
        yield


class TestTaskRunnerInit:
    def test_no_subscribers_by_default(self):
        runner = TaskRunner()
        assert len(runner._subscribers) == 0

    def test_sweep_registers_subscriber(self):
        with patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord:
            MockCoord.return_value.initialize = MagicMock()
            runner = TaskRunner(sweep=True)
            assert len(runner._subscribers) == 1

    def test_memory_sweep_registers_subscriber(self):
        with patch("conductress.sweep.memory_coordinator.create_memory_coordinators") as mock_factory:
            mock_coord = MagicMock()
            mock_coord.initialize = MagicMock()
            mock_factory.return_value = [mock_coord]
            runner = TaskRunner(memory_sweep=True)
            assert len(runner._subscribers) == 1

    def test_both_sweeps_register_two_subscribers(self):
        with (
            patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord,
            patch("conductress.sweep.memory_coordinator.create_memory_coordinators") as mock_factory,
        ):
            MockCoord.return_value.initialize = MagicMock()
            mock_mem = MagicMock()
            mock_mem.initialize = MagicMock()
            mock_factory.return_value = [mock_mem]
            runner = TaskRunner(sweep=True, memory_sweep=True)
            assert len(runner._subscribers) == 2


class TestTaskRunnerLoop:
    @pytest.mark.asyncio
    async def test_successful_task_notifies_subscribers(self, mock_servers, mock_queue, mock_cleanup):
        """A successful task triggers on_task_completed for all subscribers."""
        runner = TaskRunner()
        subscriber = MagicMock()
        runner._subscribers.append(subscriber)

        task = MagicMock()
        task.replicas = 0
        mock_task_runner = MagicMock()
        mock_task_runner.run = AsyncMock()
        mock_task_runner.file_protocol = MagicMock()
        task.prepare_task_runner.return_value = mock_task_runner

        # Return task, then raise to exit loop
        mock_queue.get_next_task.side_effect = [task, _ExitLoop()]

        with pytest.raises(_ExitLoop):
            await runner.run()

        subscriber.on_task_completed.assert_called_once_with(task)
        mock_queue.finish_task.assert_called_with(task)

    @pytest.mark.asyncio
    async def test_failed_task_notifies_subscribers_and_continues(
        self, mock_servers, mock_queue, mock_cleanup, tmp_path
    ):
        """A failed task triggers on_task_failed, records failure, and continues."""
        with (
            patch("conductress.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed.jsonl"),
            patch("conductress.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"),
        ):
            runner = TaskRunner()
            subscriber = MagicMock()
            runner._subscribers.append(subscriber)

            failing_task = MagicMock()
            failing_task.replicas = 0
            failing_task.task_id = "2026.01.01_00.00.00.000000"
            failing_task.note = "test"
            failing_task.source = "valkey"
            failing_task.specifier = "main"
            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(side_effect=RuntimeError("build failed"))
            mock_runner.file_protocol = MagicMock()
            failing_task.prepare_task_runner.return_value = mock_runner

            mock_queue.get_next_task.side_effect = [failing_task, _ExitLoop()]

            with pytest.raises(_ExitLoop):
                await runner.run()

            subscriber.on_task_failed.assert_called_once_with(failing_task)
            assert (tmp_path / "failed.jsonl").exists()

    @pytest.mark.asyncio
    async def test_empty_queue_notifies_subscribers(self, mock_servers, mock_queue, mock_cleanup):
        """When queue is empty, on_queue_empty is called for all subscribers."""
        runner = TaskRunner()
        subscriber = MagicMock()
        runner._subscribers.append(subscriber)

        # Empty queue, then raise on poll
        mock_queue.get_next_task.side_effect = [None, _ExitLoop()]

        with pytest.raises(_ExitLoop):
            await runner.run()

        assert subscriber.on_queue_empty.call_count >= 1

    @pytest.mark.asyncio
    async def test_insufficient_servers_records_failure(self, mock_servers, mock_queue, mock_cleanup, tmp_path):
        """Task requiring more replicas than available servers is recorded as failure."""
        with (
            patch("conductress.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed.jsonl"),
            patch("conductress.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"),
        ):
            runner = TaskRunner()

            task = MagicMock()
            task.replicas = 5  # Need 6 servers, only have 1
            task.task_id = "2026.01.01_00.00.00.000000"
            task.note = "test"
            task.source = "valkey"
            task.specifier = "main"

            mock_queue.get_next_task.side_effect = [task, _ExitLoop()]

            with pytest.raises(_ExitLoop):
                await runner.run()

            assert (tmp_path / "failed.jsonl").exists()

    @pytest.mark.asyncio
    async def test_task_failure_releases_cpu_allocations(self, mock_servers, mock_queue, mock_cleanup, tmp_path):
        """When a task fails, kill_all_valkey_instances_on_host is called to release CPU allocations."""
        with (
            patch("conductress.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed.jsonl"),
            patch("conductress.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"),
            patch("conductress.task_runner.Server") as MockServer,
        ):
            mock_server_instance = MagicMock()
            mock_server_instance.kill_all_valkey_instances_on_host = AsyncMock()
            MockServer.return_value = mock_server_instance

            runner = TaskRunner()

            task = MagicMock()
            task.replicas = 0
            task.task_id = "2026.01.01_00.00.00.000000"
            task.note = "test"
            task.source = "valkey"
            task.specifier = "main"

            # Task runner raises during run()
            mock_task_runner = MagicMock()
            mock_task_runner.run = AsyncMock(side_effect=RuntimeError("server crashed"))
            mock_task_runner.file_protocol = MagicMock()
            task.prepare_task_runner.return_value = mock_task_runner

            mock_queue.get_next_task.side_effect = [task, _ExitLoop()]

            with pytest.raises(_ExitLoop):
                await runner.run()

            # Verify kill_all was called to release leaked allocations
            mock_server_instance.kill_all_valkey_instances_on_host.assert_called()
