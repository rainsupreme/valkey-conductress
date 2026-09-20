"""
Climb-and-bisect search for the highest offered rate that still passes a check.

The search is a pure state machine: the caller measures a rate, reports
whether it passed, and is told the next rate to measure or that the search is
over. Nothing in here knows what "passed" means; the replica-read task uses
replication lag and reader delivery, another task could use a latency SLO.

Two phases:

    climb   multiply the rate by ``step_multiplier`` after every pass, until a
            probe fails or ``max_rate`` passes
    bisect  halve the interval between the last passing rate and the first
            failing one until it is narrower than ``tolerance`` of its upper
            end, or ``max_bisect_steps`` probes have been spent

The answer is the highest rate that passed. It is reported with the bracket
``(lo, hi)`` it was found in: ``lo`` passed, ``hi`` failed (or ``hi`` is
``None`` when nothing failed because ``max_rate`` passed). The bracket is the
search's resolution, so a reader of the result can see what was actually
measured rather than trusting a single number.

Shape borrowed from cachecannon's saturation search (``saturation.rs``), which
runs the same climb-then-bisect on its own latency SLO.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Probe:
    """Measure this rate next."""

    rate: int


@dataclass(frozen=True)
class Done:
    """Search finished.

    ``knee`` is the highest rate that passed (``None`` when even the start
    rate failed). ``hi`` is the lowest rate that failed above it, ``None``
    when the search climbed to ``max_rate`` without a failure.
    """

    knee: Optional[int]
    hi: Optional[int]

    @property
    def ceiling_reached(self) -> bool:
        return self.knee is not None and self.hi is None


class RateSearch:
    """Climb geometrically to the first failure, then bisect to the knee."""

    def __init__(
        self,
        start_rate: int,
        step_multiplier: float,
        max_rate: int,
        tolerance: float,
        max_bisect_steps: int,
    ):
        if start_rate < 1:
            raise ValueError(f"start_rate must be >= 1, got {start_rate}")
        if max_rate < start_rate:
            raise ValueError(f"max_rate {max_rate} must be >= start_rate {start_rate}")
        if step_multiplier <= 1.0:
            raise ValueError(f"step_multiplier must be > 1, got {step_multiplier}")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError(f"tolerance must be in [0, 1), got {tolerance}")
        if max_bisect_steps < 0:
            raise ValueError(f"max_bisect_steps must be >= 0, got {max_bisect_steps}")
        self.start_rate = start_rate
        self.step_multiplier = step_multiplier
        self.max_rate = max_rate
        self.tolerance = tolerance
        self.max_bisect_steps = max_bisect_steps
        self.current_rate = start_rate
        self.climbing = True
        self.lo: Optional[int] = None  # highest rate that passed
        self.hi: Optional[int] = None  # lowest rate that failed
        self.bisect_steps = 0
        self.probes = 0

    @property
    def max_probes(self) -> int:
        """Upper bound on probes the search can take, for progress estimates."""
        climb = int(math.ceil(math.log(self.max_rate / self.start_rate) / math.log(self.step_multiplier))) + 1
        return climb + self.max_bisect_steps

    def first(self) -> Probe:
        """The first rate to measure."""
        return Probe(self.start_rate)

    def advance(self, passed: bool):
        """Record the outcome for ``current_rate`` and decide the next move."""
        self.probes += 1
        if passed:
            self.lo = self.current_rate
        else:
            self.hi = self.current_rate
        if self.climbing:
            if not passed:
                self.climbing = False
                return self._bisect_or_done()
            if self.current_rate >= self.max_rate:
                return Done(knee=self.lo, hi=None)
            nxt = int(self.current_rate * self.step_multiplier)
            self.current_rate = min(max(nxt, self.current_rate + 1), self.max_rate)
            return Probe(self.current_rate)
        self.bisect_steps += 1
        return self._bisect_or_done()

    def _bisect_or_done(self):
        assert self.hi is not None
        lo = self.lo or 0
        width = (self.hi - lo) / self.hi
        if width <= self.tolerance or self.bisect_steps >= self.max_bisect_steps or self.hi - lo <= 1:
            return Done(knee=self.lo, hi=self.hi)
        mid = lo + (self.hi - lo) // 2
        self.current_rate = min(max(mid, lo + 1), self.hi - 1)
        return Probe(self.current_rate)
