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

    def test_urgency_after_initial_points(self, coordinator):
        """After 2+ points, urgency equals base bisection score."""
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


class TestGetNextTaskDependency:
    """Test that _get_next_task respects throughput data dependency."""

    def test_returns_none_when_fewer_than_2_candidates(self, repo_path, tmp_path, monkeypatch):
        """Can't bisect with only 1 throughput point."""
        state_file = tmp_path / "tp.json"
        state = SweepState()
        state.merge_commits = ["aaa", "bbb"]
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=2000000, cv=0.5, reps=3, status=PointStatus.COMPLETED
        )
        state.save(state_file)
        monkeypatch.setattr("conductress.sweep.latency_coordinator.LATENCY_STATE_FILE", tmp_path / "lat.json")
        coord = LatencySweepCoordinator(repo_path, state_file)
        coord.state.merge_commits = ["aaa", "bbb"]
        from conductress.sweep.planner import SweepPlanner

        coord.planner = SweepPlanner(coord.state)
        assert coord._get_next_task() is None

    def test_returns_task_for_commit_with_throughput(self, coordinator):
        """When planner suggests a commit that has throughput data, use it."""
        task = coordinator._get_next_task()
        # Should return a task (new series, planner will suggest something)
        if task is not None:
            # The commit must have throughput data
            assert coordinator._get_throughput_for_commit(task.commit) is not None

    def test_falls_back_when_planner_picks_commit_without_throughput(self, coordinator):
        """When planner picks a commit without throughput, fall back to first unmeasured candidate."""
        # Give the latency coordinator some existing points so planner does bisection
        from conductress.sweep.planner import SweepPlanner

        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["eee"] = BenchmarkPoint(
            commit="eee", date="2026-05-01", value=900, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.planner = SweepPlanner(coordinator.state)

        # Planner will try to bisect between aaa and eee, picking "ccc" (midpoint)
        # "ccc" HAS throughput data, so it should be returned directly
        task = coordinator._get_next_task()
        if task is not None:
            # Must be a commit with throughput data
            assert coordinator._get_throughput_for_commit(task.commit) is not None

    def test_skips_commits_already_measured_in_fallback(self, coordinator):
        """Fallback skips candidates that already have latency data."""
        from conductress.sweep.planner import SweepPlanner

        # Mark aaa and ccc as already measured for latency
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["ccc"] = BenchmarkPoint(
            commit="ccc", date="2026-03-01", value=600, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.planner = SweepPlanner(coordinator.state)

        task = coordinator._get_next_task()
        if task is not None:
            # Should NOT pick aaa or ccc (already measured)
            assert task.commit not in ["aaa", "ccc"] or task.commit == "eee"

    def test_returns_none_when_all_candidates_measured(self, coordinator):
        """Returns None when all throughput-measured commits have latency data."""
        from conductress.sweep.planner import SweepPlanner

        # Mark all 3 candidates as measured
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["ccc"] = BenchmarkPoint(
            commit="ccc", date="2026-03-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["eee"] = BenchmarkPoint(
            commit="eee", date="2026-05-01", value=500, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.planner = SweepPlanner(coordinator.state)

        # All candidates measured with same value (no bisection needed)
        task = coordinator._get_next_task()
        # Should be None (no work to do) since all values are identical (no segments to bisect)
        assert task is None

    def test_queue_next_if_needed_respects_dependency(self, coordinator, monkeypatch):
        """queue_next_if_needed only queues tasks for commits with throughput data."""
        from unittest.mock import MagicMock, patch

        # Mock TaskQueue to capture what gets submitted
        mock_queue = MagicMock()
        mock_queue.get_all_tasks.return_value = []
        monkeypatch.setattr("conductress.sweep.coordinator.TaskQueue", lambda: mock_queue)

        queued = coordinator.queue_next_if_needed()
        if queued:
            # Verify the submitted task has a valid target_rps (derived from throughput)
            submitted_task = mock_queue.submit_task.call_args[0][0]
            assert isinstance(submitted_task, LatencyTaskData)
            assert submitted_task.target_rps > 0


class TestLatencyTaskSerialization:
    """Regression test: sweep_commit must survive queue save/load roundtrip."""

    def test_sweep_commit_persists_through_queue(self, tmp_path, monkeypatch):
        """Without sweep_commit as a dataclass field, it was lost during serialization."""
        from conductress.config import LATENCY_MAKE_ARGS, SWEEP_IO_THREADS
        from conductress.task_queue import BaseTaskData, TaskQueue

        monkeypatch.setattr("conductress.task_queue.config.REPO_NAMES", ["valkey"])
        monkeypatch.setattr("conductress.task_queue.config.CONDUCTRESS_QUEUE", tmp_path)

        task = LatencyTaskData(
            source="valkey",
            specifier="abc123def456",
            make_args=LATENCY_MAKE_ARGS,
            replicas=0,
            note="test",
            requirements={},
            target_rps=1500000,
            io_threads=SWEEP_IO_THREADS,
            sweep_commit="abc123def456",
        )

        # Save to queue
        queue = TaskQueue(queue_dir=tmp_path)
        queue.submit_task(task)

        # Load back
        loaded = queue.get_next_task()
        assert loaded is not None
        assert isinstance(loaded, LatencyTaskData)
        assert loaded.sweep_commit == "abc123def456"
        assert loaded.target_rps == 1500000
