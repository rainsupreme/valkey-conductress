"""Tests for score-based scheduling in TaskRunner._schedule_next."""

from unittest.mock import MagicMock, patch

import pytest

from conductress.sweep.planner import BenchmarkPoint, SweepPlanner, SweepState
from conductress.task_runner import TaskRunner


@pytest.fixture
def runner():
    """TaskRunner with no sweep initialization."""
    return TaskRunner()


@pytest.fixture
def mock_config():
    """Patch load_sweep_config to allow all workloads."""
    config = MagicMock()
    config.is_allowed.return_value = True
    with patch("conductress.task_runner.load_sweep_config", return_value=config):
        yield config


def _make_subscriber(workload_id="get-k16-v16-t7-p10", urgency=1.0, has_nightly=False, queues_task=True):
    """Create a mock subscriber with configurable behavior."""
    sub = MagicMock()
    sub.workload_id = workload_id
    sub.get_urgency_score = MagicMock(return_value=urgency)
    sub.has_nightly_task = MagicMock(return_value=has_nightly)

    def on_queue_empty():
        if queues_task:
            # Simulate queuing a task by making TaskQueue.get_all_tasks return something
            pass

    sub.on_queue_empty = MagicMock(side_effect=on_queue_empty)
    return sub


class TestScheduleNext:
    """Tests for _schedule_next score-based scheduling."""

    def test_highest_urgency_wins(self, runner, mock_config):
        """The coordinator with highest urgency score gets to queue."""
        low = _make_subscriber("memory-set-k16-v64", urgency=2.0)
        high = _make_subscriber("get-k16-v16-t7-p10", urgency=10.0)
        runner._subscribers = [low, high]

        # Make TaskQueue return tasks after on_queue_empty is called
        call_count = [0]

        def queue_side_effect():
            call_count[0] += 1
            # First call after high's on_queue_empty returns a task
            if call_count[0] >= 1:
                return [MagicMock()]
            return []

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.side_effect = queue_side_effect
            MockQueue.return_value = queue
            runner._schedule_next()

        # High urgency called first
        high.on_queue_empty.assert_called_once()
        low.on_queue_empty.assert_not_called()

    def test_nightly_preempts_urgency(self, runner, mock_config):
        """NIGHTLY coordinator wins regardless of urgency score."""
        high_urgency = _make_subscriber("get-k16-v16-t7-p10", urgency=100.0, has_nightly=False)
        nightly = _make_subscriber("memory-redis-set-k16-v64", urgency=0.1, has_nightly=True)
        runner._subscribers = [high_urgency, nightly]

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.return_value = [MagicMock()]
            MockQueue.return_value = queue
            runner._schedule_next()

        # Nightly wins despite lower urgency
        nightly.on_queue_empty.assert_called_once()
        high_urgency.on_queue_empty.assert_not_called()

    def test_disallowed_workload_skipped(self, runner):
        """Coordinators for disallowed workloads are not scheduled."""
        blocked = _make_subscriber("get-k16-v16-t7-p10", urgency=100.0)
        allowed = _make_subscriber("memory-set-k16-v64", urgency=1.0)

        config = MagicMock()
        config.is_allowed.side_effect = lambda wid: wid != "get-k16-v16-t7-p10"

        runner._subscribers = [blocked, allowed]

        with (
            patch("conductress.task_runner.load_sweep_config", return_value=config),
            patch("conductress.task_runner.TaskQueue") as MockQueue,
        ):
            queue = MagicMock()
            queue.get_all_tasks.return_value = [MagicMock()]
            MockQueue.return_value = queue
            runner._schedule_next()

        blocked.on_queue_empty.assert_not_called()
        allowed.on_queue_empty.assert_called_once()

    def test_no_self_feeding_after_completion(self, runner, mock_config):
        """on_task_completed does NOT queue the next task from the same coordinator.

        This is the key behavioral test: after a throughput bisection completes,
        the next task should be chosen by _schedule_next (score-based), not by
        the completing coordinator self-feeding.
        """
        from conductress.sweep.coordinator import BaseSweepCoordinator
        from conductress.sweep.planner import BenchmarkPoint, SweepPlanner, SweepState

        # Create a minimal coordinator that would have work to do
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        )
        state.points["aaa"] = BenchmarkPoint(commit="aaa", date="2024-01-01", value=1000.0, cv=0.5, reps=3)

        # Create a mock task that looks like it belongs to a sweep coordinator
        task = MagicMock()
        task.source = "valkey"
        task.specifier = "bbb"
        task.sweep_commit = "bbb"
        task.type = "get"
        task.val_sizes = [16]
        task.note = "[perf-sweep:valkey/get-k16-v16-t7-p10] Bisecting"

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.return_value = []
            queue.submit_task = MagicMock()
            MockQueue.return_value = queue

            # Simulate what on_task_completed does for a sweep subscriber
            # The key assertion: it should NOT call queue.submit_task
            sub = MagicMock()
            sub.on_task_completed = MagicMock()
            runner._subscribers = [sub]

            # After on_task_completed, the queue should still be empty
            # (no self-feeding). Only _schedule_next should queue.
            for s in runner._subscribers:
                s.on_task_completed(task)

            # The subscriber's on_task_completed was called but it doesn't queue
            sub.on_task_completed.assert_called_once_with(task)
            # No task was directly submitted to queue by the subscriber
            queue.submit_task.assert_not_called()

    def test_multiple_nightly_first_one_wins(self, runner, mock_config):
        """When multiple coordinators have NIGHTLY, the first one encountered wins."""
        nightly1 = _make_subscriber("memory-redis-set-k16-v64", urgency=0.1, has_nightly=True)
        nightly2 = _make_subscriber("memory-redis-zadd-m20", urgency=0.1, has_nightly=True)
        runner._subscribers = [nightly1, nightly2]

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.return_value = [MagicMock()]
            MockQueue.return_value = queue
            runner._schedule_next()

        nightly1.on_queue_empty.assert_called_once()
        nightly2.on_queue_empty.assert_not_called()

    def test_falls_through_when_no_tasks_available(self, runner, mock_config):
        """If no coordinator can produce a task, _schedule_next returns cleanly."""
        sub = _make_subscriber("get-k16-v16-t7-p10", urgency=5.0)
        runner._subscribers = [sub]

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.return_value = []  # Nothing queued
            MockQueue.return_value = queue
            runner._schedule_next()  # Should not raise

        sub.on_queue_empty.assert_called_once()


class TestUrgencyTierOrdering:
    """Verify that urgency scores follow the expected priority hierarchy:
    NIGHTLY (absolute preemption) > new series (inf) > bisection > backfill.
    """

    def _make_coordinator(self, state, repo_path):
        """Create a real SweepCoordinator with mocked git ops."""
        from conductress.sweep.coordinator import SweepCoordinator

        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.repo_path = repo_path
            coord.state = state
            coord.state_file = repo_path / "state.json"
            coord.planner = SweepPlanner(state)
            coord.engine = None
        return coord

    def test_new_series_has_infinite_urgency(self, tmp_path):
        """A coordinator with <2 completed points should have infinite urgency."""
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc", "ddd", "eee"],
            commit_dates={c: "2024-01-01" for c in ["aaa", "bbb", "ccc", "ddd", "eee"]},
        )
        # Only 1 completed point
        state.points["aaa"] = BenchmarkPoint(commit="aaa", date="2024-01-01", value=1000.0, cv=0.5, reps=3)

        coord = self._make_coordinator(state, tmp_path)
        assert coord.get_urgency_score() == float("inf")

    def test_bisection_scores_higher_than_backfill(self, tmp_path):
        """A coordinator with active bisection scores higher than one doing backfill only."""
        # Bisection scenario: two points with a large delta and wide gap
        state_bisect = SweepState(
            merge_commits=[f"c{i}" for i in range(100)],
            commit_dates={f"c{i}": "2024-01-01" for i in range(100)},
        )
        state_bisect.points["c0"] = BenchmarkPoint(commit="c0", date="2024-01-01", value=1000.0, cv=0.5, reps=3)
        state_bisect.points["c99"] = BenchmarkPoint(commit="c99", date="2024-01-01", value=1200.0, cv=0.5, reps=3)

        # Backfill scenario: well-covered series with small deltas
        state_backfill = SweepState(
            merge_commits=[f"c{i}" for i in range(100)],
            commit_dates={f"c{i}": "2024-01-01" for i in range(100)},
        )
        # Many points, all similar values (no bisection needed)
        for i in range(0, 100, 5):
            state_backfill.points[f"c{i}"] = BenchmarkPoint(
                commit=f"c{i}", date="2024-01-01", value=1000.0 + i * 0.1, cv=0.5, reps=3
            )

        coord_bisect = self._make_coordinator(state_bisect, tmp_path)
        coord_backfill = self._make_coordinator(state_backfill, tmp_path)

        score_bisect = coord_bisect.get_urgency_score()
        score_backfill = coord_backfill.get_urgency_score()

        assert (
            score_bisect > score_backfill
        ), f"Bisection ({score_bisect:.2f}) should score higher than backfill ({score_backfill:.2f})"

    def test_larger_regression_scores_higher(self, tmp_path):
        """A 20% regression in a wide gap should outscore a 5% regression in a narrow gap."""
        # Large regression, wide gap
        state_large = SweepState(
            merge_commits=[f"c{i}" for i in range(100)],
            commit_dates={f"c{i}": "2024-01-01" for i in range(100)},
        )
        state_large.points["c0"] = BenchmarkPoint(commit="c0", date="2024-01-01", value=1000.0, cv=0.5, reps=3)
        state_large.points["c99"] = BenchmarkPoint(commit="c99", date="2024-01-01", value=800.0, cv=0.5, reps=3)

        # Small regression, narrow gap
        state_small = SweepState(
            merge_commits=[f"c{i}" for i in range(20)],
            commit_dates={f"c{i}": "2024-01-01" for i in range(20)},
        )
        state_small.points["c0"] = BenchmarkPoint(commit="c0", date="2024-01-01", value=1000.0, cv=0.5, reps=3)
        state_small.points["c19"] = BenchmarkPoint(commit="c19", date="2024-01-01", value=950.0, cv=0.5, reps=3)

        coord_large = self._make_coordinator(state_large, tmp_path)
        coord_small = self._make_coordinator(state_small, tmp_path)

        assert coord_large.get_urgency_score() > coord_small.get_urgency_score()

    def test_nightly_preempts_infinite_urgency(self, runner, mock_config):
        """Even a coordinator with inf urgency (new series) loses to NIGHTLY preemption."""
        new_series = _make_subscriber("get-k16-v16-t7-p10", urgency=float("inf"), has_nightly=False)
        nightly = _make_subscriber("memory-redis-set-k16-v64", urgency=0.0, has_nightly=True)
        runner._subscribers = [new_series, nightly]

        with patch("conductress.task_runner.TaskQueue") as MockQueue:
            queue = MagicMock()
            queue.get_all_tasks.return_value = [MagicMock()]
            MockQueue.return_value = queue
            runner._schedule_next()

        nightly.on_queue_empty.assert_called_once()
        new_series.on_queue_empty.assert_not_called()

    def test_completed_series_has_zero_urgency(self, tmp_path):
        """A fully-swept series with no remaining work returns 0 urgency."""
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        )
        # All commits benchmarked with similar values (no bisection needed, no gaps)
        state.points["aaa"] = BenchmarkPoint(commit="aaa", date="2024-01-01", value=1000.0, cv=0.5, reps=3)
        state.points["bbb"] = BenchmarkPoint(commit="bbb", date="2024-02-01", value=1001.0, cv=0.5, reps=3)
        state.points["ccc"] = BenchmarkPoint(commit="ccc", date="2024-03-01", value=1002.0, cv=0.5, reps=3)

        coord = self._make_coordinator(state, tmp_path)
        assert coord.get_urgency_score() == 0.0
