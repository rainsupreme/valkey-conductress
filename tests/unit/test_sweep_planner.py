"""Unit tests for the sweep planner module."""

from typing import Dict, List, Optional

import pytest

from conductress.sweep.planner import (
    BenchmarkPoint,
    Landmark,
    PointStatus,
    Segment,
    SweepPlanner,
    SweepState,
    SweepTask,
    TaskPriority,
)


def make_state(
    commits: List[str],
    points: Optional[Dict[str, float]] = None,
    threshold: float = 0.01,
    landmarks: Optional[List[Landmark]] = None,
    last_head: Optional[str] = None,
) -> SweepState:
    """Helper to create a SweepState with minimal boilerplate."""
    state = SweepState(
        merge_commits=commits,
        commit_dates={c: f"2024-{i+1:02d}-01" for i, c in enumerate(commits)},
        threshold=threshold,
        last_benchmarked_head=last_head,
        landmarks=landmarks or [],
    )
    if points:
        for commit, rps in points.items():
            state.points[commit] = BenchmarkPoint(
                commit=commit,
                date=state.commit_dates.get(commit, ""),
                value=rps,
                cv=0.2,
                status=PointStatus.COMPLETED,
            )
    return state


class TestSweepPlannerNightly:
    """Tests for nightly HEAD tracking (Priority 1)."""

    def test_new_head_generates_task(self):
        state = make_state(["a", "b", "c"], points={"a": 100000})
        planner = SweepPlanner(state)
        task = planner.get_next_task(current_head="c")
        assert task is not None
        assert task.commit == "c"
        assert task.priority == TaskPriority.NIGHTLY

    def test_already_benchmarked_head_skipped(self):
        state = make_state(["a", "b", "c"], points={"a": 100000, "c": 100000})
        planner = SweepPlanner(state)
        task = planner.get_next_task(current_head="c")
        # Should not return nightly task, may return bisection/backfill
        assert task is None or task.priority != TaskPriority.NIGHTLY

    def test_same_as_last_benchmarked_head_skipped(self):
        state = make_state(["a", "b", "c"], points={"a": 100000}, last_head="c")
        planner = SweepPlanner(state)
        task = planner.get_next_task(current_head="c")
        assert task is None or task.priority != TaskPriority.NIGHTLY

    def test_no_head_provided_skips_nightly(self):
        state = make_state(["a", "b", "c"])
        planner = SweepPlanner(state)
        task = planner.get_next_task(current_head=None)
        # Should still return something (backfill) but not nightly
        if task:
            assert task.priority != TaskPriority.NIGHTLY


class TestSweepPlannerBisection:
    """Tests for bisection logic (Priority 2)."""

    def test_large_delta_triggers_bisection(self):
        # 10% regression between a and c
        state = make_state(["a", "b", "c"], points={"a": 100000, "c": 90000})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.commit == "b"
        assert task.priority == TaskPriority.BISECTION
        assert "regression" in task.reason

    def test_small_delta_no_bisection(self):
        # 0.5% change — below 1% threshold
        state = make_state(["a", "b", "c"], points={"a": 100000, "c": 99500})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        # No bisection needed, might get backfill for "b"
        if task:
            assert task.priority != TaskPriority.BISECTION

    def test_bisection_picks_largest_segment(self):
        # Two segments: a->c has 5% delta, c->e has 2% delta
        state = make_state(
            ["a", "b", "c", "d", "e"],
            points={"a": 100000, "c": 95000, "e": 93000},
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.commit == "b"  # midpoint of larger segment a->c

    def test_bisection_skips_build_failures(self):
        state = make_state(["a", "b", "c", "d", "e"], points={"a": 100000, "e": 90000})
        # Mark midpoint as build failure
        state.points["c"] = BenchmarkPoint(commit="c", date="2024-03-01", status=PointStatus.BUILD_FAILED)
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        # Should pick b or d, not c
        assert task.commit in ("b", "d")

    def test_fully_bisected_segment_no_task(self):
        # Adjacent commits, no room to bisect
        state = make_state(["a", "b"], points={"a": 100000, "b": 90000})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        # No commits between a and b to bisect
        assert task is None

    def test_improvement_also_bisected(self):
        # 5% improvement
        state = make_state(["a", "b", "c"], points={"a": 100000, "c": 105000})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.BISECTION
        assert "improvement" in task.reason


class TestSweepPlannerLandmarks:
    """Tests for release commit landmarks (Priority 3)."""

    def test_unbenchmarked_landmark_generates_task(self):
        landmarks = [Landmark(commit="b", date="2024-06-15", label="8.0.0")]
        state = make_state(
            ["a", "b", "c"],
            points={"a": 100000, "c": 100000},  # small delta, no bisection
            landmarks=landmarks,
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.commit == "b"
        assert task.priority == TaskPriority.LANDMARK
        assert "8.0.0" in task.reason

    def test_benchmarked_landmark_skipped(self):
        landmarks = [Landmark(commit="b", date="2024-06-15", label="8.0.0")]
        state = make_state(
            ["a", "b", "c"],
            points={"a": 100000, "b": 100500, "c": 100000},
            landmarks=landmarks,
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        # Landmark already done, should be None or backfill
        assert task is None or task.priority == TaskPriority.BACKFILL

    def test_landmark_not_in_commit_list_skipped(self):
        landmarks = [Landmark(commit="z", date="2024-06-15", label="8.0.0")]
        state = make_state(
            ["a", "b", "c"],
            points={"a": 100000, "c": 100000},
            landmarks=landmarks,
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        # "z" not in merge_commits, should skip
        assert task is None or task.commit != "z"


class TestSweepPlannerBackfill:
    """Tests for backfill / skeleton pass (Priority 4)."""

    def test_empty_state_starts_skeleton(self):
        state = make_state(["a", "b", "c", "d", "e"])
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.BACKFILL
        assert "Skeleton" in task.reason

    def test_backfill_picks_largest_gap(self):
        # Points at positions 0 and 9, gap of 8 in between
        commits = [f"c{i}" for i in range(10)]
        state = make_state(commits, points={"c0": 100000, "c9": 100000})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.BACKFILL
        # Should pick somewhere in the middle
        idx = commits.index(task.commit)
        assert 1 <= idx <= 8

    def test_all_benchmarked_returns_none(self):
        state = make_state(["a", "b", "c"], points={"a": 100000, "b": 100000, "c": 100000})
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is None

    def test_skeleton_evenly_spaces_samples(self):
        # 32 commits, skeleton should target every 2nd
        commits = [f"c{i:02d}" for i in range(32)]
        state = make_state(commits)
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.commit == "c00"  # First skeleton point


class TestSweepPlannerPriority:
    """Tests for priority ordering."""

    def test_nightly_beats_bisection(self):
        # Has both a regression to bisect AND a new HEAD
        state = make_state(
            ["a", "b", "c", "d"],
            points={"a": 100000, "c": 90000},
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task(current_head="d")
        assert task is not None
        assert task.priority == TaskPriority.NIGHTLY
        assert task.commit == "d"

    def test_bisection_beats_landmark(self):
        landmarks = [Landmark(commit="d", date="2024-04-01", label="8.0.0")]
        state = make_state(
            ["a", "b", "c", "d", "e"],
            points={"a": 100000, "e": 90000},
            landmarks=landmarks,
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.BISECTION

    def test_landmark_beats_backfill(self):
        landmarks = [Landmark(commit="c", date="2024-03-01", label="7.2.5")]
        # All points close together (no bisection needed), but landmark unbenchmarked
        state = make_state(
            ["a", "b", "c", "d", "e"],
            points={"a": 100000, "e": 100050},  # 0.05% delta, below threshold
            landmarks=landmarks,
        )
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.priority == TaskPriority.LANDMARK


class TestRecordResult:
    """Tests for recording benchmark results."""

    def test_record_new_result(self):
        state = make_state(["a", "b", "c"])
        planner = SweepPlanner(state)
        planner.record_result("b", value=150000, cv=0.19, pr=1234, pr_title="Optimize dict")
        assert "b" in state.points
        assert state.points["b"].value == 150000
        assert state.points["b"].pr == 1234
        assert state.points["b"].is_complete

    def test_record_updates_existing(self):
        state = make_state(["a", "b"], points={"a": 100000})
        planner = SweepPlanner(state)
        planner.record_result("a", value=101000, cv=0.15)
        assert state.points["a"].value == 101000
        assert state.points["a"].cv == 0.15

    def test_record_build_failure(self):
        state = make_state(["a", "b", "c"])
        planner = SweepPlanner(state)
        planner.record_build_failure("b")
        assert state.points["b"].status == PointStatus.BUILD_FAILED
        assert not state.points["b"].is_complete


class TestSegments:
    """Tests for segment computation."""

    def test_segments_sorted_by_delta(self):
        # Points at a, d, h with gaps between them
        commits = ["a", "b", "c", "d", "e", "f", "g", "h"]
        state = make_state(
            commits,
            points={"a": 100000, "d": 95000, "h": 90000},
        )
        planner = SweepPlanner(state)
        segments = planner.get_segments()
        # a->d has 5% delta (3 commits between), d->h has ~5.3% delta (3 commits between)
        assert len(segments) == 2
        deltas = [s.abs_delta for s in segments]
        assert deltas == sorted(deltas, reverse=True)

    def test_no_segments_with_single_point(self):
        state = make_state(["a", "b", "c"], points={"b": 100000})
        planner = SweepPlanner(state)
        assert planner.get_segments() == []

    def test_segment_delta_calculation(self):
        state = make_state(["a", "b", "c"], points={"a": 100000, "c": 90000})
        planner = SweepPlanner(state)
        segments = planner.get_segments()
        assert len(segments) == 1
        assert segments[0].delta == pytest.approx(-0.1, abs=0.001)

    def test_unresolved_segments_respects_threshold(self):
        state = make_state(
            ["a", "b", "c", "d", "e"],
            points={"a": 100000, "c": 99500, "e": 90000},  # a->c: 0.5%, c->e: 10%
            threshold=0.01,
        )
        planner = SweepPlanner(state)
        unresolved = planner.get_unresolved_segments()
        assert len(unresolved) == 1
        assert unresolved[0].left_commit == "c"
        assert unresolved[0].right_commit == "e"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_commit_list(self):
        state = SweepState()
        planner = SweepPlanner(state)
        assert planner.get_next_task() is None

    def test_single_commit(self):
        state = make_state(["a"])
        planner = SweepPlanner(state)
        task = planner.get_next_task()
        assert task is not None
        assert task.commit == "a"

    def test_all_build_failures_between(self):
        state = make_state(["a", "b", "c", "d", "e"], points={"a": 100000, "e": 90000})
        for c in ["b", "c", "d"]:
            state.points[c] = BenchmarkPoint(commit=c, date="", status=PointStatus.BUILD_FAILED)
        planner = SweepPlanner(state)
        # All midpoints are build failures — no bisection possible
        task = planner.get_next_task()
        # Should return None for bisection (no valid midpoints)
        assert task is None

    def test_zero_rps_left_point(self):
        state = make_state(["a", "b", "c"], points={"a": 0, "c": 100000})
        planner = SweepPlanner(state)
        segments = planner.get_segments()
        # Should handle division by zero gracefully
        assert len(segments) == 1
        assert segments[0].delta == 0.0  # Our defined behavior for zero base

    def test_many_commits_performance(self):
        """Ensure planner handles large commit lists efficiently."""
        n = 5000
        commits = [f"c{i:05d}" for i in range(n)]
        state = make_state(commits, points={"c00000": 100000, f"c{n-1:05d}": 95000})
        planner = SweepPlanner(state)
        # Should not hang or be unreasonably slow
        task = planner.get_next_task()
        assert task is not None


class TestBreakdownSerialization:
    """Regression test: breakdown field must survive save/load round-trip."""

    def test_breakdown_persists_through_save_load(self, tmp_path):
        from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState

        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=50.0,
            cv=0.0,
            reps=1,
            status=PointStatus.COMPLETED,
            breakdown={"embedded_obj": 40.0, "hashtable": 10.0, "sds": 0, "other": 0.1},
        )

        state_file = tmp_path / "state.json"
        state.save(state_file)

        loaded = SweepState.load(state_file)
        assert loaded.points["aaa"].breakdown == {"embedded_obj": 40.0, "hashtable": 10.0, "sds": 0, "other": 0.1}

    def test_none_breakdown_survives_round_trip(self, tmp_path):
        from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState

        state = SweepState(merge_commits=["bbb"], commit_dates={"bbb": "2024-02-01"})
        state.points["bbb"] = BenchmarkPoint(
            commit="bbb",
            date="2024-02-01",
            value=30.0,
            cv=0.0,
            reps=1,
            status=PointStatus.COMPLETED,
        )

        state_file = tmp_path / "state.json"
        state.save(state_file)

        loaded = SweepState.load(state_file)
        assert loaded.points["bbb"].breakdown is None

    def test_all_dataclass_fields_survive_round_trip(self, tmp_path):
        """Generic test: every BenchmarkPoint field must be serialized.

        If you add a field to BenchmarkPoint and this test fails,
        you need to add it to SweepState.save() and .load().
        """
        from dataclasses import fields

        from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState

        # Create a point with ALL fields set to non-default values
        all_fields_point = BenchmarkPoint(
            commit="test123",
            date="2024-06-15",
            value=42.5,
            cv=1.23,
            reps=5,
            pr=999,
            pr_title="Test PR title",
            perf_counters={"cycles": 1000, "instructions": 2000},
            perf_duration_seconds=30.0,
            perf_rps=1500000.0,
            breakdown={"embedded_obj": 30.0, "sds": 10.0},
            status=PointStatus.COMPLETED,
        )

        state = SweepState(merge_commits=["test123"], commit_dates={"test123": "2024-06-15"})
        state.points["test123"] = all_fields_point

        state_file = tmp_path / "state.json"
        state.save(state_file)
        loaded = SweepState.load(state_file)
        loaded_point = loaded.points["test123"]

        # Check every field
        for field in fields(BenchmarkPoint):
            original = getattr(all_fields_point, field.name)
            restored = getattr(loaded_point, field.name)
            assert restored == original, (
                f"Field '{field.name}' not preserved: saved={original!r}, loaded={restored!r}. "
                f"Did you forget to add it to SweepState.save() and .load()?"
            )
