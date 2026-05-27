"""Tests for LatencySweepCoordinator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.sweep.latency_coordinator import LATENCY_LOAD_FRACTION, LATENCY_STATE_FILE, LatencySweepCoordinator
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState
from conductress.tasks.task_latency import LatencyTaskData


@pytest.fixture(autouse=True)
def patch_repo_names(monkeypatch):
    """Allow 'valkey' as a valid source in task creation."""
    monkeypatch.setattr("conductress.task_queue.config.REPO_NAMES", ["valkey"])


@pytest.fixture
def throughput_state_file(tmp_path):
    """Create a throughput state file with some completed points."""
    state = SweepState()
    state.merge_commits = ["aaa", "bbb", "ccc", "ddd", "eee"]
    # Simulate throughput results for 3 commits
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa", date="2026-01-01", value=2000000, cv=0.5, reps=3, status=PointStatus.COMPLETED
    )
    state.points["ccc"] = BenchmarkPoint(
        commit="ccc", date="2026-03-01", value=2100000, cv=0.4, reps=3, status=PointStatus.COMPLETED
    )
    state.points["eee"] = BenchmarkPoint(
        commit="eee", date="2026-05-01", value=1800000, cv=0.6, reps=3, status=PointStatus.COMPLETED
    )
    state_file = tmp_path / "throughput_state.json"
    state.save(state_file)
    return state_file


@pytest.fixture
def latency_state_file(tmp_path, monkeypatch):
    """Patch the latency state file to use tmp_path."""
    state_file = tmp_path / "latency_state.json"
    monkeypatch.setattr("conductress.sweep.latency_coordinator.LATENCY_STATE_FILE", state_file)
    return state_file


@pytest.fixture
def repo_path(tmp_path):
    """Create a fake repo path."""
    repo = tmp_path / "valkey"
    repo.mkdir()
    return repo


@pytest.fixture
def coordinator(repo_path, throughput_state_file, latency_state_file):
    """Create a LatencySweepCoordinator with test fixtures."""
    coord = LatencySweepCoordinator(repo_path, throughput_state_file)
    # Manually set merge commits (normally done by initialize())
    coord.state.merge_commits = ["aaa", "bbb", "ccc", "ddd", "eee"]
    # Recreate planner with updated state
    from conductress.sweep.planner import SweepPlanner

    coord.planner = SweepPlanner(coord.state)
    return coord


class TestLatencyCoordinatorProperties:
    def test_metric_id(self, coordinator):
        assert coordinator.metric_id == "latency"

    def test_lower_is_better(self, coordinator):
        assert coordinator.lower_is_better is True

    def test_workload_id(self, coordinator):
        assert "get" in coordinator.workload_id
        assert "p10" in coordinator.workload_id


class TestCandidateCommits:
    def test_returns_only_throughput_measured_commits(self, coordinator):
        candidates = coordinator._get_candidate_commits()
        assert set(candidates) == {"aaa", "ccc", "eee"}

    def test_sorted_by_commit_index(self, coordinator):
        candidates = coordinator._get_candidate_commits()
        assert candidates == ["aaa", "ccc", "eee"]

    def test_empty_when_no_throughput_data(self, repo_path, tmp_path, monkeypatch):
        empty_state = tmp_path / "empty_throughput.json"
        SweepState().save(empty_state)
        monkeypatch.setattr(
            "conductress.sweep.latency_coordinator.LATENCY_STATE_FILE",
            tmp_path / "lat.json",
        )
        coord = LatencySweepCoordinator(repo_path, empty_state)
        assert coord._get_candidate_commits() == []


class TestUrgencyScore:
    def test_zero_when_insufficient_throughput_data(self, repo_path, tmp_path, monkeypatch):
        """Need at least 2 throughput points to run."""
        state_file = tmp_path / "one_point.json"
        state = SweepState()
        state.merge_commits = ["aaa"]
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=2000000, cv=0.5, reps=3, status=PointStatus.COMPLETED
        )
        state.save(state_file)
        monkeypatch.setattr(
            "conductress.sweep.latency_coordinator.LATENCY_STATE_FILE",
            tmp_path / "lat.json",
        )
        coord = LatencySweepCoordinator(repo_path, state_file)
        assert coord.get_urgency_score() == 0.0

    def test_infinity_for_new_series(self, coordinator):
        """New series with <2 latency points gets infinity."""
        assert coordinator.get_urgency_score() == float("inf")

    def test_dampened_after_initial_points(self, coordinator):
        """After 2+ points, urgency is dampened by 0.5x."""
        from conductress.sweep.planner import PointStatus, SweepPlanner

        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["eee"] = BenchmarkPoint(
            commit="eee", date="2026-05-01", value=900, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.planner = SweepPlanner(coordinator.state)
        score = coordinator.get_urgency_score()
        # Should be finite and positive (there's a gap to bisect)
        assert 0 < score < float("inf")


class TestCreateTask:
    def test_creates_latency_task_with_correct_target(self, coordinator):
        from conductress.sweep.planner import SweepTask, TaskPriority

        task = coordinator._create_task(
            SweepTask(commit="ccc", date="2026-03-01", priority=TaskPriority.BISECTION, reason="test")
        )
        assert isinstance(task, LatencyTaskData)
        assert task.target_rps == int(2100000 * LATENCY_LOAD_FRACTION)
        assert task.load_fraction == LATENCY_LOAD_FRACTION
        assert task.source == "valkey"
        assert task.specifier == "ccc"

    def test_raises_when_no_throughput(self, coordinator):
        from conductress.sweep.planner import SweepTask, TaskPriority

        with pytest.raises(ValueError, match="No throughput data"):
            coordinator._create_task(
                SweepTask(commit="bbb", date="2026-02-01", priority=TaskPriority.BISECTION, reason="test")
            )


class TestIsMyTask:
    def test_matches_latency_task_with_sweep_commit(self, coordinator):
        task = LatencyTaskData(
            source="valkey",
            specifier="ccc",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            target_rps=1000000,
        )
        task.sweep_commit = "ccc"
        assert coordinator._is_my_task(task) is True

    def test_rejects_latency_task_without_sweep_commit(self, coordinator):
        task = LatencyTaskData(
            source="valkey",
            specifier="ccc",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            target_rps=1000000,
        )
        assert coordinator._is_my_task(task) is False

    def test_rejects_non_latency_task(self, coordinator):
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        task = PerfTaskData(
            source="valkey",
            specifier="ccc",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )
        task.sweep_commit = "ccc"
        assert coordinator._is_my_task(task) is False


class TestExtractResult:
    def test_extracts_p99_from_output(self, coordinator, tmp_path, monkeypatch):
        output_file = tmp_path / "output.jsonl"
        entry = {
            "task_id": "2026-05-27_01-00-00",
            "score": 892.0,
            "data": {"reps": 3, "p99_us": 892.0},
        }
        output_file.write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr("conductress.sweep.latency_coordinator.CONDUCTRESS_RESULTS", tmp_path)

        task = LatencyTaskData(
            source="valkey",
            specifier="ccc",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            target_rps=1000000,
        )
        # Mock the task_id to match
        from unittest.mock import PropertyMock

        type(task).task_id = PropertyMock(return_value="2026-05-27_01-00-00")

        result = coordinator._extract_result(task)
        assert result == (892.0, 0.0, 3)
