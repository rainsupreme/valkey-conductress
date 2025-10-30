import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ServerInfo
from src.file_protocol import BenchmarkStatus
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.tui_data_service import TUIDataService


@dataclass
class MockTaskData(BaseTaskData):
    """Mock task data for testing."""

    def short_description(self) -> str:
        return "Mock task"

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> BaseTaskRunner:
        return MagicMock()


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def service(temp_dir):
    return TUIDataService(work_dir=temp_dir)


def test_init(temp_dir):
    service = TUIDataService(work_dir=temp_dir)
    assert service._work_dir == temp_dir
    assert service._cached_tasks == []
    assert service._cached_active_tasks == {}
    assert service._cached_task_statuses == {}


@patch("src.tui_data_service.TaskQueue")
@patch("src.tui_data_service.FileProtocol")
def test_refresh_all(mock_file_protocol_class, mock_task_queue_class, service):
    mock_tasks = [MagicMock(), MagicMock()]
    mock_active = {"task1": MagicMock(), "task2": MagicMock()}
    mock_status = BenchmarkStatus(steps_total=10, task_type="test")

    service._queue.get_all_tasks = MagicMock(return_value=mock_tasks)
    mock_file_protocol_class.get_active_task_ids = MagicMock(return_value=mock_active)
    mock_protocol_instance = MagicMock()
    mock_protocol_instance.read_status.return_value = mock_status
    mock_file_protocol_class.return_value = mock_protocol_instance

    tasks, active = service.refresh_all()

    assert tasks == mock_tasks
    assert active == mock_active
    assert service._cached_tasks == mock_tasks
    assert service._cached_active_tasks == mock_active
    assert len(service._cached_task_statuses) == 2


def test_get_queue_data(service):
    mock_tasks = [MagicMock(), MagicMock()]
    service._cached_tasks = mock_tasks

    result = service.get_queue_data()

    assert result is mock_tasks


def test_get_active_tasks(service):
    mock_active = {"task1": MagicMock()}
    service._cached_active_tasks = mock_active

    result = service.get_active_tasks()

    assert result is mock_active


@patch("src.tui_data_service.FileProtocol")
def test_get_task_status_fresh(mock_file_protocol_class, service):
    task_id = "task1"
    mock_status = BenchmarkStatus(steps_total=10, task_type="test")
    mock_status.heartbeat = time.time()
    service._cached_task_statuses[task_id] = mock_status

    result = service.get_task_status(task_id)

    assert result == mock_status
    mock_file_protocol_class.assert_not_called()


@patch("src.tui_data_service.FileProtocol")
def test_get_task_status_stale(mock_file_protocol_class, service):
    task_id = "task1"
    old_status = BenchmarkStatus(steps_total=10, task_type="test")
    old_status.heartbeat = time.time() - 60
    service._cached_task_statuses[task_id] = old_status

    new_status = BenchmarkStatus(steps_total=10, task_type="test")
    new_status.heartbeat = time.time()
    mock_protocol_instance = MagicMock()
    mock_protocol_instance.read_status.return_value = new_status
    mock_file_protocol_class.return_value = mock_protocol_instance

    result = service.get_task_status(task_id)

    assert result == new_status
    assert service._cached_task_statuses[task_id] == new_status


@patch("src.tui_data_service.FileProtocol")
def test_get_task_status_not_cached(mock_file_protocol_class, service):
    task_id = "task1"
    new_status = BenchmarkStatus(steps_total=10, task_type="test")
    mock_protocol_instance = MagicMock()
    mock_protocol_instance.read_status.return_value = new_status
    mock_file_protocol_class.return_value = mock_protocol_instance

    result = service.get_task_status(task_id)

    assert result == new_status
    assert service._cached_task_statuses[task_id] == new_status


def test_remove_task(service):
    task_id = "task1"
    service._queue.remove_task = MagicMock(return_value=True)

    result = service.remove_task(task_id)

    assert result is True
    service._queue.remove_task.assert_called_once_with(task_id)


def test_submit_task(service):
    mock_task = MagicMock()
    service._queue.submit_task = MagicMock()

    service.submit_task(mock_task)

    service._queue.submit_task.assert_called_once_with(mock_task)
