"""Tests for the climb-and-bisect rate search (a pure state machine)."""

import pytest

from conductress.rate_search import Done, Probe, RateSearch


def _run(search: RateSearch, knee: int) -> tuple:
    """Drive a search against an oracle where every rate <= knee passes; returns (Done, probed rates)."""
    step = search.first()
    rates = []
    while isinstance(step, Probe):
        rates.append(step.rate)
        assert len(rates) <= search.max_probes, "the probe bound must hold"
        step = search.advance(step.rate <= knee)
    assert isinstance(step, Done)
    return step, rates


def test_climbs_geometrically_then_bisects_to_within_tolerance():
    done, rates = _run(RateSearch(1000, 2.0, 1_000_000, 0.05, 8), knee=30_000)
    assert rates[:6] == [1000, 2000, 4000, 8000, 16000, 32000]  # climb ends at the first failure
    assert done.knee is not None and done.knee <= 30_000, "the reported knee must have passed"
    assert done.hi is not None and done.hi > 30_000, "the bracket's upper end must have failed"
    assert (done.hi - done.knee) / done.hi <= 0.05
    assert not done.ceiling_reached


def test_knee_is_always_a_rate_that_passed():
    for knee in (0, 500, 1000, 1500, 7777, 30_000, 999_999, 5_000_000):
        done, _ = _run(RateSearch(1000, 2.0, 1_000_000, 0.05, 8), knee=knee)
        if done.knee is not None:
            assert done.knee <= knee


def test_reaching_max_rate_without_a_failure_reports_the_ceiling():
    done, rates = _run(RateSearch(1000, 2.0, 1_000_000, 0.05, 8), knee=10_000_000)
    assert rates[-1] == 1_000_000, "the last climb step lands exactly on max_rate"
    assert done.knee == 1_000_000 and done.hi is None and done.ceiling_reached


def test_start_rate_failing_bisects_downward_instead_of_giving_up():
    done, rates = _run(RateSearch(100_000, 1.5, 1_000_000, 0.05, 8), knee=30_000)
    assert rates[0] == 100_000 and all(r < 100_000 for r in rates[1:])
    assert done.knee is not None and done.knee <= 30_000


def test_everything_failing_reports_no_knee():
    done, _ = _run(RateSearch(1000, 2.0, 1_000_000, 0.05, 8), knee=0)
    assert done.knee is None and done.hi is not None


def test_bisection_is_capped_by_max_bisect_steps():
    search = RateSearch(1000, 2.0, 1_000_000, 0.0, 3)
    done, rates = _run(search, knee=12_345)
    assert done.knee is not None and search.bisect_steps <= 3
    assert len(rates) <= search.max_probes


def test_zero_tolerance_and_zero_bisect_steps_still_terminate():
    done, rates = _run(RateSearch(1000, 2.0, 1_000_000, 0.0, 0), knee=12_345)
    assert done.knee == 8000 and done.hi == 16_000  # the climb's own bracket, no bisection
    assert len(rates) == 5


def test_max_probes_bounds_every_search_on_a_grid():
    for start, mult, ceiling, knee in ((250_000, 1.5, 20_000_000, 6_400_000), (500_000, 1.5, 20_000_000, 1_900_000)):
        search = RateSearch(start, mult, ceiling, 0.05, 8)
        _, rates = _run(search, knee=knee)
        assert len(rates) <= search.max_probes


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(start_rate=0), "start_rate"),
        (dict(max_rate=10), "max_rate"),
        (dict(step_multiplier=1.0), "step_multiplier"),
        (dict(tolerance=1.0), "tolerance"),
        (dict(max_bisect_steps=-1), "max_bisect_steps"),
    ],
)
def test_parameters_are_validated(kwargs, match):
    params = dict(start_rate=1000, step_multiplier=1.5, max_rate=1_000_000, tolerance=0.05, max_bisect_steps=8)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RateSearch(**params)
