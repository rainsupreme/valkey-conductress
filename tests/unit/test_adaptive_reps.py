"""Tests for adaptive repetition logic."""

import pytest

from conductress.tasks.task_perf_benchmark import should_stop_adaptive


class TestShouldStopAdaptive:
    """Tests for the adaptive early-exit decision function."""

    def test_stops_when_cv_below_target(self):
        # CV of [1000, 1001, 999] is ~0.1% — well below 0.5% target
        assert should_stop_adaptive([1000, 1001, 999], rep=2, min_reps=3, target_cv=0.5) is True

    def test_continues_when_cv_above_target(self):
        # CV of [1000, 1050, 950] is ~5% — above 0.5% target
        assert should_stop_adaptive([1000, 1050, 950], rep=2, min_reps=3, target_cv=0.5) is False

    def test_never_stops_before_min_reps(self):
        # Even with perfect CV, don't stop before min_reps
        assert should_stop_adaptive([1000, 1000], rep=1, min_reps=3, target_cv=0.5) is False

    def test_disabled_when_target_zero(self):
        assert should_stop_adaptive([1000, 1001, 999], rep=2, min_reps=3, target_cv=0.0) is False

    def test_needs_at_least_two_values(self):
        assert should_stop_adaptive([1000], rep=2, min_reps=1, target_cv=0.5) is False

    def test_stops_at_min_reps_boundary(self):
        # rep=2 (0-indexed) is the 3rd rep — exactly at min_reps=3
        assert should_stop_adaptive([1000, 1001, 1000], rep=2, min_reps=3, target_cv=0.5) is True

    def test_allows_stop_after_min_reps(self):
        # rep=4 (5th rep) with min_reps=3 — allowed
        assert should_stop_adaptive([1000, 1001, 999, 1000, 1001], rep=4, min_reps=3, target_cv=0.5) is True
