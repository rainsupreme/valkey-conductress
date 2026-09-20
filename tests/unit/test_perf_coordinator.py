"""Tests for perf stat integration in sweep coordinator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.sweep.coordinator import SweepCoordinator, perf_counters_from_entry
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepPlanner, SweepState
from conductress.tasks.task_perf_benchmark import PerfTaskData
from conductress.topology import TopologySpec


def _make_task():
    """Create a PerfTaskData instance for testing."""
    return PerfTaskData(
        source="valkey",
        specifier="aaa",
        make_args="",
        topology=TopologySpec.standalone(),
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
        assert result.counters["instructions"] == 900000000000
        assert result.duration == 30.0
        assert result.rps == 2000000
        assert result.rep_count == 3
        # No per-thread data or scope in this fixture
        assert result.counters_main is None
        assert result.counters_io is None
        assert result.scope is None

    def test_returns_none_when_no_counters(self, coordinator, output_jsonl):
        task = _make_task()
        entry = {"task_id": task.task_id, "score": 2000000, "data": {"per_run_rps": [2000000]}}
        output_jsonl.write_text(json.dumps(entry) + "\n")

        assert coordinator._extract_perf_counters(task) is None


class TestPerfCountersFromEntry:
    """Both result-row shapes normalise to the same record."""

    def test_flat_shape_with_per_thread_siblings_and_scope(self):
        entry = {
            "score": 2_000_000,
            "data": {
                "perf_counters": {"instructions": 900, "cycles": 300},
                "perf_counters_main": {"instructions": 500, "cycles": 200},
                "perf_counters_io": {"instructions": 400, "cycles": 100},
                "perf_duration_seconds": 30.0,
                "perf_rep_count": 5,
                "perf_counters_scope": "user+kernel",
            },
        }
        rec = perf_counters_from_entry(entry)
        assert rec is not None
        assert rec.counters == {"instructions": 900, "cycles": 300}
        assert rec.counters_main == {"instructions": 500, "cycles": 200}
        assert rec.counters_io == {"instructions": 400, "cycles": 100}
        assert (rec.duration, rec.rps, rec.rep_count, rec.scope) == (30.0, 2_000_000, 5, "user+kernel")

    def test_nested_shape_as_written_by_cachecannon_and_mixed_tasks(self):
        entry = {
            "score": 1_990_000,
            "data": {
                "perf_counters": {
                    "all": {"instructions": 900, "cycles": 300, "raw_syscalls:sys_enter": 12},
                    "main": {"instructions": 500, "cycles": 200, "raw_syscalls:sys_enter": 6},
                    "io": {"instructions": 400, "cycles": 100, "raw_syscalls:sys_enter": 6},
                },
                "perf_counters_scope": "user+kernel",
                "perf_duration_seconds": 29.2,
                "perf_rep_count": 2,
            },
        }
        rec = perf_counters_from_entry(entry)
        assert rec is not None
        assert rec.counters["raw_syscalls:sys_enter"] == 12
        assert rec.counters_main["instructions"] == 500  # type: ignore[index]
        assert rec.counters_io["cycles"] == 100  # type: ignore[index]
        assert (rec.duration, rec.rep_count, rec.scope) == (29.2, 2, "user+kernel")

    def test_nested_shape_without_per_thread_buckets(self):
        entry = {"score": 1, "data": {"perf_counters": {"all": {"instructions": 9, "cycles": 3}}}}
        rec = perf_counters_from_entry(entry)
        assert rec is not None
        assert rec.counters == {"instructions": 9, "cycles": 3}
        assert rec.counters_main is None and rec.counters_io is None

    def test_empty_all_bucket_is_no_data(self):
        assert perf_counters_from_entry({"score": 1, "data": {"perf_counters": {"all": {}, "main": {}}}}) is None
        assert perf_counters_from_entry({"score": 1, "data": {}}) is None

    def test_rate_limited_latency_row_uses_the_achieved_rate_not_the_p99_score(self):
        # A latency cell scores p99 in microseconds; dividing counters by that
        # would inflate every per-request metric ~1000x. The row records the
        # rate the counters were collected at.
        entry = {
            "score": 92.2,
            "data": {
                "perf_counters": {"all": {"instructions": 900, "cycles": 300}},
                "mean_rps": 99_999.6,
                "score_metric": "p99",
                "perf_duration_seconds": 28.7,
                "perf_rep_count": 10,
            },
        }
        rec = perf_counters_from_entry(entry)
        assert rec is not None
        assert rec.rps == 99_999.6

    def test_throughput_row_rate_is_unchanged_by_the_recorded_mean(self):
        # cachecannon throughput rows record mean_rps equal to the score; rows
        # from tasks that do not record it keep using the score.
        with_mean = {"score": 1_990_000.0, "data": {"perf_counters": {"all": {"cycles": 3}}, "mean_rps": 1_990_000.0}}
        without = {"score": 2_000_000.0, "data": {"perf_counters": {"cycles": 3}}}
        assert perf_counters_from_entry(with_mean).rps == 1_990_000.0  # type: ignore[union-attr]
        assert perf_counters_from_entry(without).rps == 2_000_000.0  # type: ignore[union-attr]


class TestPerfScopeRecording:
    def test_scope_is_stored_and_persisted(self, coordinator):
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=1.0, status=PointStatus.COMPLETED
        )
        coordinator.record_perf_counters("aaa", {"instructions": 1}, 30.0, 1.0, scope="user+kernel")
        assert coordinator.state.points["aaa"].perf_counters_scope == "user+kernel"
        assert SweepState.load(coordinator.state_file).points["aaa"].perf_counters_scope == "user+kernel"

    def test_none_scope_keeps_existing_value(self, coordinator):
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=1.0, status=PointStatus.COMPLETED, perf_counters_scope="user"
        )
        coordinator.record_perf_counters("aaa", {"instructions": 1}, 30.0, 1.0)
        assert coordinator.state.points["aaa"].perf_counters_scope == "user"

    def test_on_task_completed_carries_scope_to_point(self, coordinator, output_jsonl):
        task = _make_task()
        task.sweep_commit = "aaa"
        coordinator.state.points["aaa"] = BenchmarkPoint(commit="aaa", date="2024-01-01")
        entry = {
            "task_id": task.task_id,
            "score": 2_000_000,
            "data": {
                "per_run_rps": [2_000_000, 2_010_000],
                "perf_counters": {"instructions": 9, "cycles": 3},
                "perf_counters_scope": "user+kernel",
                "perf_duration_seconds": 30.0,
                "perf_rep_count": 2,
            },
        }
        output_jsonl.write_text(json.dumps(entry) + "\n")
        with (
            patch.object(coordinator, "_is_my_task", return_value=True),
            patch("conductress.sweep.coordinator.get_head", return_value="zzz"),
        ):
            coordinator.on_task_completed(task)
        point = coordinator.state.points["aaa"]
        assert point.perf_counters == {"instructions": 9, "cycles": 3}
        assert point.perf_counters_scope == "user+kernel"
        assert point.perf_rep_count == 2


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
            perf_counters_scope="user+kernel",
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
        assert loaded.points["aaa"].perf_counters_scope == "user+kernel"

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
        assert loaded.points["aaa"].perf_counters_scope is None
