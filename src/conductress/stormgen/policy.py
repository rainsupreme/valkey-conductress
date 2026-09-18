"""Reconnect back-off policies for storm clients.

A policy maps a zero-based retry attempt index to a delay in seconds. Policies
are parsed from a compact string so they can travel through a CLI flag or a
task field:

* ``immediate``                 -- always retry with no delay
* ``fixed:<delay_ms>``          -- constant delay
* ``exp:<base_ms>:<max_ms>``    -- exponential back-off, capped at max
* ``exp:<base_ms>:<max_ms>:jitter`` -- same, with full jitter in [0, delay]

The interface is deliberately small (:meth:`ReconnectPolicy.next_delay`) so a
new policy is one class plus one line in :func:`parse_policy`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class ReconnectPolicy(ABC):
    """Maps a retry attempt index to a wait before the next connect attempt."""

    @abstractmethod
    def next_delay(self, attempt_index: int) -> float:
        """Return the delay in seconds before retry ``attempt_index`` (0-based)."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Return the canonical spec string for this policy."""
        raise NotImplementedError


class ImmediatePolicy(ReconnectPolicy):
    """Retry with no delay."""

    def next_delay(self, attempt_index: int) -> float:
        return 0.0

    def describe(self) -> str:
        return "immediate"


class FixedPolicy(ReconnectPolicy):
    """Constant delay between every attempt."""

    def __init__(self, delay_ms: float) -> None:
        if delay_ms < 0:
            raise ValueError(f"fixed delay must be >= 0 ms, got {delay_ms}")
        self.delay_ms = delay_ms

    def next_delay(self, attempt_index: int) -> float:
        return self.delay_ms / 1000.0

    def describe(self) -> str:
        return f"fixed:{_fmt(self.delay_ms)}"


class ExponentialPolicy(ReconnectPolicy):
    """Exponential back-off ``base * 2**attempt``, capped at ``max``.

    With ``jitter`` the returned delay is drawn uniformly from ``[0, capped]``
    (full jitter), the standard way to keep a reconnecting herd from retrying
    in lockstep. ``rng`` is injectable so tests are deterministic.
    """

    def __init__(self, base_ms: float, max_ms: float, jitter: bool = False, rng: random.Random | None = None) -> None:
        if base_ms <= 0:
            raise ValueError(f"exp base must be > 0 ms, got {base_ms}")
        if max_ms < base_ms:
            raise ValueError(f"exp max ({max_ms} ms) must be >= base ({base_ms} ms)")
        self.base_ms = base_ms
        self.max_ms = max_ms
        self.jitter = jitter
        self._rng = rng or random.Random()

    def next_delay(self, attempt_index: int) -> float:
        # Cap the shift so 2**attempt cannot overflow for a pathological index.
        shift = min(attempt_index, 30)
        capped_ms = min(self.max_ms, self.base_ms * (2**shift))
        if self.jitter:
            capped_ms = self._rng.uniform(0.0, capped_ms)
        return capped_ms / 1000.0

    def describe(self) -> str:
        suffix = ":jitter" if self.jitter else ""
        return f"exp:{_fmt(self.base_ms)}:{_fmt(self.max_ms)}{suffix}"


def _fmt(ms: float) -> str:
    """Render a millisecond value without a trailing ``.0`` for whole numbers."""
    return str(int(ms)) if float(ms).is_integer() else str(ms)


def _parse_ms(token: str, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"policy {field} must be a number of milliseconds, got {token!r}") from exc
    return value


def parse_policy(spec: str, rng: random.Random | None = None) -> ReconnectPolicy:
    """Parse a policy spec string into a :class:`ReconnectPolicy`.

    Raises :class:`ValueError` with a message that names the accepted forms
    for any spec that does not match, so a typo fails at submission rather
    than silently picking a default.
    """
    text = (spec or "").strip()
    if not text:
        raise ValueError(
            "empty reconnect policy; expected one of: immediate, fixed:<ms>, exp:<base_ms>:<max_ms>[:jitter]"
        )
    head, _, tail = text.partition(":")
    head = head.lower()

    if head == "immediate":
        if tail:
            raise ValueError(f"'immediate' takes no parameters, got {text!r}")
        return ImmediatePolicy()

    if head == "fixed":
        if not tail:
            raise ValueError("'fixed' requires a delay: fixed:<delay_ms>")
        return FixedPolicy(_parse_ms(tail, "fixed delay"))

    if head == "exp":
        parts = tail.split(":") if tail else []
        if len(parts) not in (2, 3):
            raise ValueError("'exp' requires exp:<base_ms>:<max_ms>[:jitter]")
        base_ms = _parse_ms(parts[0], "exp base")
        max_ms = _parse_ms(parts[1], "exp max")
        jitter = False
        if len(parts) == 3:
            if parts[2].lower() != "jitter":
                raise ValueError(f"unknown exp option {parts[2]!r}; the only option is 'jitter'")
            jitter = True
        return ExponentialPolicy(base_ms, max_ms, jitter=jitter, rng=rng)

    raise ValueError(
        f"unknown reconnect policy {text!r}; expected one of: "
        "immediate, fixed:<delay_ms>, exp:<base_ms>:<max_ms>[:jitter]"
    )
