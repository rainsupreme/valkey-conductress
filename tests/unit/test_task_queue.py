import logging
import shutil
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from conductress import task_queue
from conductress.config import ServerInfo
from conductress.topology import TopologySpec


class DummyConfig(types.ModuleType):
    MANUALLY_UPLOADED = "manual"
    REPO_NAMES = ["repo1", "repo2"]


@pytest.fixture(autouse=True)
def _patch_task_queue_config(monkeypatch):
    """Patch config for task_queue tests without leaking to other modules."""
    monkeypatch.setattr(task_queue, "config", DummyConfig("dummy_config"))


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@dataclass
class MockTaskData(task_queue.BaseTaskData):
    """Mock task data for testing purposes."""

    extra_data: str

    def short_description(self) -> str:
        return "Mock task"

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "task_queue.BaseTaskRunner":
        return MockTaskRunner(server_infos)


class MockTaskRunner(task_queue.BaseTaskRunner):
    """Mock task runner for testing purposes."""

    def __init__(self, server_infos: list[ServerInfo]):
        super().__init__("mock_task")
        self.server_infos = server_infos

    async def run(self):
        print(f"Running mock task on servers: {self.server_infos}")


def make_task():
    return MockTaskData(
        source="manual",
        specifier="spec",
        make_args="",
        topology=TopologySpec.replication_hosts(1),
        note="test",
        requirements={},
        extra_data="extra_info",
    )


def test_task_task_type():
    task = make_task()
    assert isinstance(task, MockTaskData)
    assert isinstance(task.timestamp, datetime)
    assert task.task_type == "MockTaskData"


def test_task_save_and_load(temp_dir):
    task = make_task()
    task.save_to_file(temp_dir / "task.json")
    loaded = task_queue.BaseTaskData.from_file(temp_dir / "task.json")
    assert isinstance(loaded, MockTaskData)
    assert loaded.extra_data == task.extra_data
    assert loaded.timestamp == task.timestamp


def test_invalid_repo_fails():
    with pytest.raises(ValueError):
        MockTaskData(
            source="invalid_repo",
            specifier="spec",
            make_args="",
            topology=TopologySpec.replication_hosts(1),
            note="test",
            requirements={},
            extra_data="mock_info",
        )


def test_task_queue_submit_and_get_next_task(temp_dir) -> None:
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    task: MockTaskData = make_task()
    queue.submit_task(task)
    next_task = queue.get_next_task()
    assert next_task is not None
    assert isinstance(next_task, MockTaskData)
    assert next_task.timestamp == task.timestamp


def test_task_queue_finish_task(temp_dir):
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    task = make_task()
    queue.submit_task(task)
    assert queue.get_queue_length() == 1

    next_task = queue.get_next_task()
    assert next_task is not None
    queue.finish_task(next_task)
    assert queue.get_queue_length() == 0

    next_task = queue.get_next_task()
    assert next_task is None


def test_task_queue_get_all_tasks(temp_dir) -> None:
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    task1: MockTaskData = make_task()
    task2: MockTaskData = make_task()
    task2.timestamp = datetime.now()
    queue.submit_task(task1)
    queue.submit_task(task2)
    all_tasks = queue.get_all_tasks()
    assert len(all_tasks) == 2
    assert all(isinstance(t, MockTaskData) for t in all_tasks)


def test_task_queue_get_queue_length(temp_dir):
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    assert queue.get_queue_length() == 0
    queue.submit_task(make_task())
    assert queue.get_queue_length() == 1


def test_invalid_json_file_skipped(temp_dir):
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    # Write invalid JSON
    bad_file = temp_dir / "task_20220101T000000.json"
    with bad_file.open("w") as f:
        f.write("{not valid json")

    with pytest.raises(ValueError) as excinfo:
        queue.get_next_task()
    assert "Invalid JSON in file" in str(excinfo.value)


def test_finish_task_logs_error_when_file_missing(temp_dir, caplog):
    """finish_task() should log an error but not crash when the task file doesn't exist."""
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    task = make_task()
    # Don't submit the task — the file won't exist on disk
    with caplog.at_level(logging.ERROR):
        queue.finish_task(task)  # Should not raise
    assert "Task file not found" in caplog.text
    assert "This is a bug" in caplog.text


def test_finish_task_does_not_crash_when_file_missing(temp_dir):
    """finish_task() should not raise or call exit() when file is missing."""
    queue = task_queue.TaskQueue(queue_dir=temp_dir)
    task = make_task()
    # Should complete without raising
    queue.finish_task(task)


def test_invalid_source_raises_value_error_with_message():
    """BaseTaskData.__post_init__ should raise ValueError with a descriptive message for invalid sources."""
    with pytest.raises(ValueError, match="Unknown source: bad_source"):
        MockTaskData(
            source="bad_source",
            specifier="spec",
            make_args="",
            topology=TopologySpec.replication_hosts(1),
            note="test",
            requirements={},
            extra_data="mock_info",
        )


def test_valid_source_from_repo_names():
    """BaseTaskData.__post_init__ should accept sources from REPO_NAMES."""
    task = MockTaskData(
        source="repo1",
        specifier="spec",
        make_args="",
        topology=TopologySpec.replication_hosts(1),
        note="test",
        requirements={},
        extra_data="mock_info",
    )
    assert task.source == "repo1"


def test_valid_source_manually_uploaded():
    """BaseTaskData.__post_init__ should accept MANUALLY_UPLOADED as a valid source."""
    task = MockTaskData(
        source="manual",
        specifier="spec",
        make_args="",
        topology=TopologySpec.replication_hosts(1),
        note="test",
        requirements={},
        extra_data="mock_info",
    )
    assert task.source == "manual"


# --------------------------------------------------------------------------- topology field


def _perf_document(**overrides) -> dict:
    from conductress.tasks.task_perf_benchmark import PerfTaskData  # noqa: F401  (registers the type)

    doc = {
        "source": "repo1",
        "specifier": "abc",
        "note": "",
        "requirements": {},
        "make_args": "",
        "task_type": "PerfTaskData",
        "timestamp": "2026-01-01T00:00:00",
        "test": "get",
        "val_size": 16,
        "io_threads": 7,
        "pipelining": 10,
        "warmup": 5,
        "duration": 30,
        "perf_stat_enabled": False,
        "has_expire": False,
        "preload_keys": True,
    }
    doc.update(overrides)
    return doc


class TestTopologyField:
    def test_legacy_replicas_zero_loads_as_standalone(self, _patch_task_queue_config):
        task = task_queue.BaseTaskData.from_dict(_perf_document(replicas=0))
        assert task.topology == TopologySpec.standalone()
        assert not hasattr(task, "replicas")

    def test_legacy_replicas_n_loads_as_replication_hosts(self, _patch_task_queue_config):
        task = task_queue.BaseTaskData.from_dict(_perf_document(replicas=2))
        assert task.topology == TopologySpec.replication_hosts(2)
        assert task.topology.host_count() == 3

    def test_legacy_negative_replicas_is_standalone(self, _patch_task_queue_config):
        # tui.py used to send -1 as a placeholder; the runner treated it as 0.
        task = task_queue.BaseTaskData.from_dict(_perf_document(replicas=-1))
        assert task.topology == TopologySpec.standalone()

    def test_document_with_both_keys_keeps_topology(self, _patch_task_queue_config):
        doc = _perf_document(replicas=3, topology=TopologySpec.standalone().to_dict())
        task = task_queue.BaseTaskData.from_dict(doc)
        assert task.topology == TopologySpec.standalone()

    def test_non_integer_replicas_is_invalid(self, _patch_task_queue_config):
        with pytest.raises(ValueError, match="replicas must be an integer"):
            task_queue.BaseTaskData.from_dict(_perf_document(replicas="two"))

    def test_topology_round_trips_through_save_and_load(self, _patch_task_queue_config, temp_dir):
        task = task_queue.BaseTaskData.from_dict(_perf_document(replicas=1))
        path = Path(temp_dir) / "t.json"
        task.save_to_file(path)
        raw = path.read_text(encoding="utf-8")
        assert '"replicas"' not in raw and '"topology"' in raw
        loaded = task_queue.BaseTaskData.from_file(path)
        assert loaded.topology == task.topology

    def test_constructor_accepts_a_dict_topology(self, _patch_task_queue_config):
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        doc = _perf_document()
        doc.pop("task_type")
        doc.pop("timestamp")
        task = PerfTaskData(topology=TopologySpec.replication_hosts(1).to_dict(), **doc)
        assert task.topology == TopologySpec.replication_hosts(1)

    def test_document_with_neither_key_is_standalone(self, _patch_task_queue_config):
        """The layout key is optional on the wire: the canary scheduler's document omits it."""
        task = task_queue.BaseTaskData.from_dict(_perf_document())
        assert task.topology == TopologySpec.standalone()
