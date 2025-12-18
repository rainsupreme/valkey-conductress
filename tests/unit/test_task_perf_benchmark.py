import json
import shutil
import tempfile
import types
from pathlib import Path

import pytest

from src import config
from src.tasks.task_perf_benchmark import PerfTaskData


class DummyConfig(types.ModuleType):
    MANUALLY_UPLOADED = "manual"
    REPO_NAMES = ["repo1", "repo2"]


# Patch config for tests
config.MANUALLY_UPLOADED = "manual"
config.REPO_NAMES = ["repo1", "repo2"]


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestPerfTaskData:
    """Tests for PerfTaskData class."""

    def test_converts_float_to_int(self):
        """Test that warmup and duration floats are converted to ints."""
        task = PerfTaskData(
            source="manual",
            specifier="test",
            make_args="",
            replicas=1,
            note="test",
            requirements={},
            test="set",
            val_size=1024,
            io_threads=4,
            pipelining=1,
            warmup=5.5,
            duration=10.7,
            profiling_sample_rate=0,
            has_expire=False,
            preload_keys=True,
        )
        assert isinstance(task.warmup, int)
        assert isinstance(task.duration, int)
        assert task.warmup == 5
        assert task.duration == 10

    def test_loads_floats_from_json(self, temp_dir):
        """Test that loading from JSON with float values converts them to ints."""
        task_file = temp_dir / "task.json"
        
        # Write JSON with float values for warmup and duration
        task_data = {
            "source": "manual",
            "specifier": "test",
            "make_args": "",
            "replicas": 1,
            "note": "test",
            "requirements": {},
            "test": "set",
            "val_size": 1024,
            "io_threads": 4,
            "pipelining": 1,
            "warmup": 5.5,
            "duration": 10.7,
            "profiling_sample_rate": 0,
            "has_expire": False,
            "preload_keys": True,
            "task_type": "PerfTaskData",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        with task_file.open("w") as f:
            json.dump(task_data, f)
        
        # Load the task
        from src.task_queue import BaseTaskData
        loaded_task = BaseTaskData.from_file(task_file)
        
        assert isinstance(loaded_task, PerfTaskData)
        assert isinstance(loaded_task.warmup, int)
        assert isinstance(loaded_task.duration, int)
        assert loaded_task.warmup == 5
        assert loaded_task.duration == 10
