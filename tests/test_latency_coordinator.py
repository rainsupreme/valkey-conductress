"""Tests for LatencySweepCoordinator (flat 100K rps, P=1, no throughput dependency)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.config import LATENCY_TARGET_RPS
from conductress.sweep.latency_coordinator import LatencySweepCoordinator
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepPlanner, SweepState
from conductress.tasks.task_latency import LatencyTaskData


@pytest.fixture(autouse=True)
def patch_repo_names(monkeypatch):
    """Allow 'valkey' as a valid source in task creation."""
    monkeypatch.setattr("conductress.task_queue.config.REPO_NAMES", ["valkey"])


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
def coordinator(repo_path, latency_state_file):
    """Create a LatencySweepCoordinator with test fixtures."""
    coord = LatencySweepCoordinator(repo_path)
    # Manually set merge commits (normally done by initialize())
    coord.state.merge_commits = ["aaa", "bbb", "ccc", "ddd", "eee"]
    coord.planner = SweepPlanner(coord.state)
    return coord


class TestLatencyCoordinatorProperties:
    def test_metric_id(self, coordinator):
        assert coordinator.metric_id == "latency"

    def test_lower_is_better(self, coordinator):
        assert coordinator.lower_is_better is True

    def test_workload_id(self, coordinator):
        assert coordinator.workload_id == "get-k16-v16"

    def test_export_filename_no_double_suffix(self, coordinator):
        """Publisher uses {workload_id}-{metric_id}.json — workload_id must not contain metric_id."""
        filename = f"series-arm64-{coordinator.workload_id}-{coordinator.metric_id}.json"
        assert filename == "series-arm64-get-k16-v16-latency.json"
        assert "latency-latency" not in filename


class TestUrgencyScore:
    def test_infinity_for_new_series(self, coordinator):
        """New series with <2 latency points gets infinity."""
        assert coordinator.get_urgency_score() == float("inf")

    def test_finite_after_two_points(self, coordinator):
        """After 2+ points, urgency is dampened by 0.5x."""
        coordinator.state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=50, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.state.points["eee"] = BenchmarkPoint(
            commit="eee", date="2026-05-01", value=90, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        coordinator.planner = SweepPlanner(coordinator.state)
        score = coordinator.get_urgency_score()
        assert 0 < score < float("inf")


class TestCreateTask:
    def test_creates_latency_task_with_flat_rate(self, coordinator):
        from conductress.sweep.planner import SweepTask, TaskPriority

        task = coordinator._create_task(
            SweepTask(commit="ccc", date="2026-03-01", priority=TaskPriority.BISECTION, reason="test")
        )
        assert isinstance(task, LatencyTaskData)
        assert task.target_rps == LATENCY_TARGET_RPS
        assert task.source == "valkey"
        assert task.specifier == "ccc"

    def test_task_note_mentions_flat_rate(self, coordinator):
        from conductress.sweep.planner import SweepTask, TaskPriority

        task = coordinator._create_task(
            SweepTask(commit="aaa", date="2026-01-01", priority=TaskPriority.LANDMARK, reason="landmark")
        )
        assert "100K rps flat" in task.note


class TestIsMyTask:
    def test_matches_latency_task_with_sweep_commit(self, coordinator):
        task = LatencyTaskData(
            source="valkey",
            specifier="ccc",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            target_rps=100000,
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
            target_rps=100000,
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
            "score": 42.0,
            "data": {"reps": 3, "p99_us": 42.0},
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
            target_rps=100000,
        )
        from unittest.mock import PropertyMock

        type(task).task_id = PropertyMock(return_value="2026-05-27_01-00-00")

        result = coordinator._extract_result(task)
        assert result == (42.0, 0.0, 3)


class TestNoThroughputDependency:
    """Verify the coordinator operates independently of throughput data."""

    def test_no_throughput_state_file_parameter(self):
        """Constructor takes only repo_path — no throughput_state_file."""
        import inspect

        sig = inspect.signature(LatencySweepCoordinator.__init__)
        params = list(sig.parameters.keys())
        assert params == ["self", "repo_path"]

    def test_operates_on_full_commit_history(self, coordinator):
        """Uses all merge commits, not a throughput-measured subset."""
        # The coordinator's state has all 5 commits
        assert len(coordinator.state.merge_commits) == 5

    def test_all_commits_eligible_for_tasks(self, coordinator):
        """Any commit can get a latency task (no throughput gate)."""
        from conductress.sweep.planner import SweepTask, TaskPriority

        # Even commits without throughput data can get tasks
        for commit in ["aaa", "bbb", "ccc", "ddd", "eee"]:
            task = coordinator._create_task(
                SweepTask(commit=commit, date="2026-01-01", priority=TaskPriority.BACKFILL, reason="test")
            )
            assert task.target_rps == LATENCY_TARGET_RPS


class TestLatencyTaskSerialization:
    """Regression test: sweep_commit must survive queue save/load roundtrip."""

    def test_sweep_commit_persists_through_queue(self, tmp_path, monkeypatch):
        from conductress.task_queue import TaskQueue

        monkeypatch.setattr("conductress.task_queue.config.REPO_NAMES", ["valkey"])
        monkeypatch.setattr("conductress.task_queue.config.CONDUCTRESS_QUEUE", tmp_path)

        task = LatencyTaskData(
            source="valkey",
            specifier="abc123def456",
            make_args="",
            replicas=0,
            note="test",
            requirements={},
            target_rps=LATENCY_TARGET_RPS,
            io_threads=7,
            sweep_commit="abc123def456",
        )

        queue = TaskQueue(queue_dir=tmp_path)
        queue.submit_task(task)

        loaded = queue.get_next_task()
        assert loaded is not None
        assert isinstance(loaded, LatencyTaskData)
        assert loaded.sweep_commit == "abc123def456"
        assert loaded.target_rps == LATENCY_TARGET_RPS
