"""Tests for MemorySweepCoordinator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress.sweep.memory_coordinator import MemorySweepCoordinator
from conductress.sweep.planner import SweepState
from conductress.tasks.task_mem_efficiency import MemTaskData
from conductress.tasks.task_perf_benchmark import PerfTaskData


@pytest.fixture
def tmp_state(tmp_path):
    state_file = tmp_path / "memory_state.json"
    state = SweepState(
        merge_commits=["aaa", "bbb", "ccc"],
        commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
    )
    state.save(state_file)
    return state_file


@pytest.fixture
def coordinator(tmp_state, tmp_path, monkeypatch):
    import conductress.config as config

    monkeypatch.setattr(config, "REPO_NAMES", ["valkey", "rainsupreme"])
    with patch("conductress.sweep.memory_coordinator.MEMORY_STATE_FILE", tmp_state):
        coord = MemorySweepCoordinator(tmp_path / "repo")
        coord.initialize()
        return coord


class TestMemoryCoordinatorInit:
    def test_metric_id(self, coordinator):
        assert coordinator.metric_id == "memory"

    def test_metric_unit(self, coordinator):
        assert coordinator.metric_unit == "bytes/item"

    def test_lower_is_better(self, coordinator):
        assert coordinator.lower_is_better is True


class TestTaskCreation:
    def test_creates_mem_task_data(self, coordinator):
        from conductress.sweep.planner import SweepTask, TaskPriority

        sweep_task = SweepTask(commit="aaa", date="2024-01-01", priority=TaskPriority.BACKFILL, reason="test")
        task = coordinator._create_task(sweep_task)

        assert isinstance(task, MemTaskData)
        assert task.source == "valkey"
        assert task.specifier == "aaa"
        assert task.note == "[memory-sweep] test"


class TestTaskFiltering:
    def test_accepts_mem_task_with_sweep_commit(self, coordinator):
        task = MagicMock(spec=MemTaskData)
        task.sweep_commit = "aaa"
        assert coordinator._is_my_task(task) is True

    def test_rejects_mem_task_without_sweep_commit(self, coordinator):
        task = MagicMock(spec=MemTaskData)
        task.sweep_commit = ""
        assert coordinator._is_my_task(task) is False

    def test_rejects_perf_task(self, coordinator):
        task = MagicMock(spec=PerfTaskData)
        task.sweep_commit = "aaa"
        assert coordinator._is_my_task(task) is False


class TestResultExtraction:
    def test_extracts_score_from_output(self, coordinator, tmp_path):
        output_file = tmp_path / "results" / "output.jsonl"
        output_file.parent.mkdir(parents=True)
        entry = {"task_id": "2026.01.01_00.00.00.000000", "score": 50.26}
        output_file.write_text(json.dumps(entry) + "\n")

        task = MagicMock()
        task.task_id = "2026.01.01_00.00.00.000000"

        with patch("conductress.sweep.memory_coordinator.CONDUCTRESS_RESULTS", tmp_path / "results"):
            result = coordinator._extract_result(task)

        assert result == (50.26, 0.0, 1)

    def test_returns_none_when_no_output(self, coordinator, tmp_path):
        task = MagicMock()
        task.task_id = "nonexistent"

        with patch("conductress.sweep.memory_coordinator.CONDUCTRESS_RESULTS", tmp_path / "empty"):
            result = coordinator._extract_result(task)

        assert result is None
