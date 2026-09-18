"""Async tests for TaskRunner main loop behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.task_runner import TaskRunner
from conductress.topology import TopologySpec


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
        with (
            patch("conductress.sweep.coordinator.SweepCoordinator") as MockCoord,
            patch("conductress.sweep.latency_coordinator.LatencySweepCoordinator") as MockLatency,
            patch("conductress.sweep.memory_coordinator.create_memory_coordinators") as mock_factory,
            patch("conductress.platform.get_local_platform_tag", return_value="amd64"),
        ):
            MockCoord.return_value.initialize = MagicMock()
            MockLatency.return_value.initialize = MagicMock()
            mock_mem = MagicMock()
            mock_mem.initialize = MagicMock()
            mock_factory.return_value = [mock_mem]
            runner = TaskRunner(sweep=True)
            # Should register at least throughput + latency + memory subscribers
            assert len(runner._subscribers) >= 3

    def test_memory_sweep_standalone(self):
        """--memory-sweep without --sweep still works (backward compat)."""
        with patch("conductress.sweep.memory_coordinator.create_memory_coordinators") as mock_factory:
            mock_coord = MagicMock()
            mock_coord.initialize = MagicMock()
            mock_factory.return_value = [mock_coord]
            runner = TaskRunner(sweep=False, memory_sweep=True)
            assert len(runner._subscribers) == 1


class TestTaskRunnerLoop:
    @pytest.mark.asyncio
    async def test_successful_task_notifies_subscribers(self, mock_servers, mock_queue, mock_cleanup):
        """A successful task triggers on_task_completed for all subscribers."""
        runner = TaskRunner()
        subscriber = MagicMock()
        runner._subscribers.append(subscriber)

        task = MagicMock()
        task.topology = TopologySpec.standalone()
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
            failing_task.topology = TopologySpec.standalone()
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
            task.topology = TopologySpec.standalone()
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


def test_sweep_v2_flag_registers_get_and_mixed_coordinators():
    with (
        patch("conductress.config.SWEEP_V2_ENABLED", True),
        patch("conductress.sweep.coordinator.SweepCoordinator") as legacy,
        patch("conductress.sweep.coordinator_v2.ThroughputSweepCoordinatorV2") as get_v2,
        patch("conductress.sweep.coordinator_v2.MixedSweepCoordinatorV2") as mixed_v2,
        patch("conductress.sweep.latency_coordinator.LatencySweepCoordinator") as latency,
        patch("conductress.sweep.memory_coordinator.create_memory_coordinators", return_value=[]),
        patch("conductress.platform.get_local_platform_tag", return_value="amd64"),
    ):
        legacy.return_value.initialize = MagicMock()
        get_v2.return_value.initialize = MagicMock()
        mixed_v2.return_value.initialize = MagicMock()
        latency.return_value.initialize = MagicMock()
        runner = TaskRunner(sweep=True)

    get_v2.assert_called_once()
    mixed_v2.assert_called_once()
    get_v2.return_value.initialize.assert_called_once()
    mixed_v2.return_value.initialize.assert_called_once()
    assert get_v2.return_value in runner._subscribers
    assert mixed_v2.return_value in runner._subscribers


class _FakeManagement:
    """Stands in for ManagementCores: records acquire/release relative to the task's run()."""

    events: list = []
    instances: list = []

    def __init__(self, server, task_id, count=None):  # pylint: disable=unused-argument
        self.task_id = task_id
        _FakeManagement.instances.append(self)

    async def acquire(self):
        _FakeManagement.events.append("acquire")
        return "15"

    def release(self):
        _FakeManagement.events.append("release")


@pytest.fixture
def fake_management(monkeypatch):
    _FakeManagement.events = []
    _FakeManagement.instances = []
    monkeypatch.setattr("conductress.task_runner.ManagementCores", _FakeManagement)
    return _FakeManagement


def _task_with_runner(run_side_effect=None):
    task = MagicMock()
    task.topology = TopologySpec.standalone()
    task.task_id = "t1"
    task.note, task.source, task.specifier = "test", "valkey", "main"
    task_runner = MagicMock()
    task_runner.management_cpus = ""
    task_runner.run = AsyncMock(side_effect=run_side_effect)
    task_runner.file_protocol = MagicMock()

    async def run_and_record():
        _FakeManagement.events.append("run")
        if run_side_effect:
            raise run_side_effect

    task_runner.run = run_and_record
    task.prepare_task_runner.return_value = task_runner
    return task, task_runner


class TestManagementCores:
    @pytest.mark.asyncio
    async def test_runner_pins_before_the_task_and_releases_after(
        self, mock_servers, mock_queue, mock_cleanup, fake_management
    ):
        runner = TaskRunner()
        task, task_runner = _task_with_runner()
        mock_queue.get_next_task.side_effect = [task, _ExitLoop()]
        with pytest.raises(_ExitLoop):
            await runner.run()
        assert fake_management.events == ["acquire", "run", "release"]
        assert task_runner.management_cpus == "15"  # handed to the task before run()
        assert fake_management.instances[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_runner_releases_management_cores_when_the_task_fails(
        self, mock_servers, mock_queue, mock_cleanup, fake_management, tmp_path
    ):
        runner = TaskRunner()
        task, _ = _task_with_runner(run_side_effect=RuntimeError("boom"))
        mock_queue.get_next_task.side_effect = [task, _ExitLoop()]
        with (
            patch("conductress.task_runner.CONDUCTRESS_FAILED_LOG", tmp_path / "failed.jsonl"),
            patch("conductress.task_runner.CONDUCTRESS_FAILED_DIR", tmp_path / "failed"),
            pytest.raises(_ExitLoop),
        ):
            await runner.run()
        assert fake_management.events == ["acquire", "run", "release"]

    @pytest.mark.asyncio
    async def test_remote_benchmark_host_does_not_pin_the_runner(
        self, mock_servers, mock_queue, mock_cleanup, fake_management
    ):
        mock_servers.return_value = [MagicMock(ip="10.0.0.9")]
        runner = TaskRunner()
        task, task_runner = _task_with_runner()
        mock_queue.get_next_task.side_effect = [task, _ExitLoop()]
        with pytest.raises(_ExitLoop):
            await runner.run()
        assert fake_management.events == ["run"]
        assert task_runner.management_cpus == ""
