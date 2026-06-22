"""Tests for perf stat integration in sweep coordinator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.sweep.coordinator import SweepCoordinator
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepPlanner, SweepState
from conductress.tasks.task_perf_benchmark import PerfTaskData


def _make_task():
    """Create a PerfTaskData instance for testing."""
    return PerfTaskData(
        source="valkey",
        specifier="aaa",
        make_args="",
        replicas=0,
        note="",
        requirements={},
        test="get",
        val_size=16,
        io_threads=7,
        pipelining=10,
        warmup=5,
        duration=30,
        perf_stat_enabled=True,
        has_expire=False,
        preload_keys=True,
    )


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    """Create a SweepCoordinator with mocked paths."""
    state_file = tmp_path / "state_get-k16-v16-t7-p10.json"
    monkeypatch.setattr("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path)

    state = SweepState(
        merge_commits=["aaa", "bbb", "ccc"],
        commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
    )
    state.save(state_file)

    with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
        coord = SweepCoordinator.__new__(SweepCoordinator)
        coord.repo_path = tmp_path
        coord.state_file = state_file
        coord.state = state
        coord.planner = SweepPlanner(state)
    return coord


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    """Ensure config values are set for test environment."""
    import conductress.config as config_mod
    import conductress.sweep.coordinator as coord_mod

    monkeypatch.setattr(config_mod, "REPO_NAMES", ["valkey"])


@pytest.fixture
def output_jsonl(tmp_path):
    """Create a mock output.jsonl and patch CONDUCTRESS_RESULTS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir / "output.jsonl"


@pytest.fixture(autouse=True)
def _patch_results(output_jsonl, monkeypatch):
    """Ensure CONDUCTRESS_RESULTS points to our temp dir for all tests in this module."""
    import conductress.sweep.coordinator as coord_mod

    monkeypatch.setattr(coord_mod, "CONDUCTRESS_RESULTS", output_jsonl.parent)


class TestRecordPerfCounters:
    def test_stores_counters_on_existing_point(self, coordinator):
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=2000000, cv=0.5, reps=5, status=PointStatus.COMPLETED
        )
        counters = {"instructions": 900000000000, "cycles": 300000000000}
        coordinator.record_perf_counters("aaa", counters, 30.0, 2000000.0)

        point = coordinator.state.points["aaa"]
        assert point.perf_counters == counters
        assert point.perf_duration_seconds == 30.0
        assert point.perf_rps == 2000000.0

    def test_persists_to_disk(self, coordinator):
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=2000000, cv=0.5, reps=5, status=PointStatus.COMPLETED
        )
        counters = {"instructions": 100, "cycles": 50}
        coordinator.record_perf_counters("aaa", counters, 10.0, 1000000.0)

        loaded = SweepState.load(coordinator.state_file)
        assert loaded.points["aaa"].perf_counters == counters
        assert loaded.points["aaa"].perf_duration_seconds == 10.0

    def test_warns_on_missing_commit(self, coordinator, caplog):
        coordinator.record_perf_counters("nonexistent", {"instructions": 1}, 10.0, 100.0)
        assert "not in state" in caplog.text


class TestFindTaskEntry:
    def test_finds_matching_entry(self, coordinator, output_jsonl):
        task = _make_task()
        entry = {"task_id": task.task_id, "score": 2000000, "data": {"per_run_rps": [2000000, 2010000]}}
        output_jsonl.write_text(json.dumps(entry) + "\n")

        result = coordinator._find_task_entry(task)
        assert result is not None
        assert result["score"] == 2000000

    def test_returns_none_for_missing_file(self, coordinator, output_jsonl):
        task = _make_task()
        assert coordinator._find_task_entry(task) is None

    def test_returns_none_for_no_match(self, coordinator, output_jsonl):
        entry = {"task_id": "other-task", "score": 1000000, "data": {}}
        output_jsonl.write_text(json.dumps(entry) + "\n")

        task = _make_task()
        assert coordinator._find_task_entry(task) is None


class TestExtractPerfCounters:
    def test_extracts_counters(self, coordinator, output_jsonl):
        task = _make_task()
        entry = {
            "task_id": task.task_id,
            "score": 2000000,
            "data": {
                "per_run_rps": [2000000],
                "perf_counters": {"instructions": 900000000000, "cycles": 300000000000},
                "perf_duration_seconds": 30.0,
                "perf_rep_count": 3,
            },
        }
        output_jsonl.write_text(json.dumps(entry) + "\n")

        result = coordinator._extract_perf_counters(task)
        assert result is not None
        counters, duration, rps, counters_main, counters_io, rep_count = result
        assert counters["instructions"] == 900000000000
        assert duration == 30.0
        assert rps == 2000000
        assert rep_count == 3
        # No per-thread data in this fixture
        assert counters_main is None
        assert counters_io is None

    def test_returns_none_when_no_counters(self, coordinator, output_jsonl):
        task = _make_task()
        entry = {"task_id": task.task_id, "score": 2000000, "data": {"per_run_rps": [2000000]}}
        output_jsonl.write_text(json.dumps(entry) + "\n")

        assert coordinator._extract_perf_counters(task) is None


class TestStateRoundTrip:
    def test_perf_counters_survive_save_load(self, tmp_path):
        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=2000000,
            cv=0.5,
            reps=5,
            status=PointStatus.COMPLETED,
            perf_counters={"instructions": 900000000000, "cycles": 300000000000, "LLC-load-misses": 9000000},
            perf_duration_seconds=30.0,
            perf_rps=2000000.0,
            perf_rep_count=5,
        )

        path = tmp_path / "state.json"
        state.save(path)
        loaded = SweepState.load(path)

        assert loaded.points["aaa"].perf_counters == {
            "instructions": 900000000000,
            "cycles": 300000000000,
            "LLC-load-misses": 9000000,
        }
        assert loaded.points["aaa"].perf_duration_seconds == 30.0
        assert loaded.points["aaa"].perf_rps == 2000000.0
        assert loaded.points["aaa"].perf_rep_count == 5

    def test_none_counters_survive_round_trip(self, tmp_path):
        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=2000000, status=PointStatus.COMPLETED
        )

        path = tmp_path / "state.json"
        state.save(path)
        loaded = SweepState.load(path)

        assert loaded.points["aaa"].perf_counters is None
        assert loaded.points["aaa"].perf_duration_seconds is None
        assert loaded.points["aaa"].perf_rps is None
        assert loaded.points["aaa"].perf_rep_count is None
