"""Tests for adaptive repetition logic."""

import pytest

from conductress.tasks.task_perf_benchmark import should_stop_adaptive


class TestShouldStopAdaptive:
    """Tests for the adaptive early-exit decision function."""

    def test_stops_when_ci_below_target(self):
        # [1000, 1001, 999]: stdev=1.0, 95% CI half-width = 4.303*1.0/sqrt(3)
        # = ±0.248% of mean — below 0.5% target
        assert should_stop_adaptive([1000, 1001, 999], rep=2, min_reps=3, target_cv=0.5) is True

    def test_continues_when_ci_above_target(self):
        # CV of [1000, 1050, 950] is ~5% — way above target
        assert should_stop_adaptive([1000, 1050, 950], rep=2, min_reps=3, target_cv=0.5) is False

    def test_low_cv_but_wide_ci_does_not_stop(self):
        # [1000, 1004, 996]: CV = 0.4% (below target — the old raw-CV rule
        # would stop), but at n=3 the t multiplier is 4.303 so the 95% CI
        # half-width is ±0.99% of mean — above the 0.5% target.
        assert should_stop_adaptive([1000, 1004, 996], rep=2, min_reps=3, target_cv=0.5) is False

    def test_same_spread_converges_with_more_reps(self):
        # Same ±4 spread over 6 reps: t(5)=2.571, CI half-width = ±0.38% — stops.
        runs = [1000, 1004, 996, 1000, 1004, 996]
        assert should_stop_adaptive(runs, rep=5, min_reps=3, target_cv=0.5) is True

    def test_min_reps_guards_against_bimodal_lucky_streak(self):
        # Three very tight reps (e.g. all landed on one mode of a bimodal
        # platform) satisfy the CI criterion, but min_reps=5 forces more
        # sampling before an early stop is allowed.
        one_mode = [1_000_000, 1_000_500, 999_500]
        assert should_stop_adaptive(one_mode, rep=2, min_reps=5, target_cv=0.5) is False

    def test_never_stops_before_min_reps(self):
        # Even with perfect precision, don't stop before min_reps
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
