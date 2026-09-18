"""Pure reductions from storm events to the reported numbers.

Every function here takes plain data (the attempt events a run produced, plus
the stall record) and returns plain data, with no I/O and no clock of its own.
That keeps them unit-testable on synthetic events with known answers, and it
keeps the definition of each metric in one readable place.

Event shape (one dict per connect attempt), all times in seconds relative to
the storm's start (a shared monotonic origin):

    {
        "client_id": int,
        "attempt": int,          # 0-based attempt index for that client
        "t_start": float,        # when the connect attempt began
        "t_end": float,          # when it resolved (connected or failed)
        "outcome": str,          # one of OUTCOMES
    }

``connected`` is terminal for a client: after it, the client holds the
connection open until the run ends and produces no more events. Any other
outcome is a failure that is followed by a back-off and another attempt.
"""

from __future__ import annotations

from typing import Dict, List, Optional

OUTCOME_CONNECTED = "connected"
OUTCOMES = (
    OUTCOME_CONNECTED,
    "connect_timeout",
    "reply_timeout",
    "refused",
    "reset",
    "error",
)

DEFAULT_BUCKET_MS = 100


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, min(len(sorted_values), round(pct / 100.0 * len(sorted_values))))
    return sorted_values[rank - 1]


def outcome_counts(events: List[dict]) -> Dict[str, int]:
    """Count attempts per outcome; every known outcome present as a key (0 if none)."""
    counts = {outcome: 0 for outcome in OUTCOMES}
    for event in events:
        outcome = str(event.get("outcome"))
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def totals(events: List[dict], client_count: int) -> Dict[str, float]:
    """Client population, attempt count, and amplification (attempts / clients).

    Amplification is the headline: 1.0 means every client connected on its
    first try; a stall that forces reconnects drives it above 1.0.
    """
    attempts = len(events)
    amplification = (attempts / client_count) if client_count else 0.0
    connected_clients = len({e["client_id"] for e in events if e.get("outcome") == OUTCOME_CONNECTED})
    return {
        "clients": client_count,
        "attempts": attempts,
        "amplification": amplification,
        "connected_clients": connected_clients,
    }


def attempt_latency_percentiles(events: List[dict]) -> Dict[str, float]:
    """Percentiles of per-attempt duration (t_end - t_start) in milliseconds."""
    durations = sorted((e["t_end"] - e["t_start"]) * 1000.0 for e in events if e["t_end"] >= e["t_start"])
    return {
        "p50_ms": _percentile(durations, 50),
        "p90_ms": _percentile(durations, 90),
        "p99_ms": _percentile(durations, 99),
        "max_ms": durations[-1] if durations else 0.0,
    }


def time_to_quiescence(
    events: List[dict],
    client_count: int,
    *,
    storm_start: float = 0.0,
    stall_end: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Seconds until the storm is quiet, measured from two origins.

    The storm is quiet once every client has a terminal ``connected`` event
    and no attempt is still in flight. Quiescence time is the last relevant
    ``t_end`` minus the origin:

    * ``from_start`` -- from the storm's start (``storm_start``)
    * ``from_stall_end`` -- from the moment the stall cleared (``stall_end``);
      ``None`` when there was no stall

    Either value is ``None`` when the run never reached quiescence (some client
    never connected), so a caller never mistakes "never recovered" for "instant".
    """
    connected = {e["client_id"] for e in events if e.get("outcome") == OUTCOME_CONNECTED}
    reached = client_count > 0 and len(connected) >= client_count
    last_end = max((e["t_end"] for e in events), default=None)

    from_start: Optional[float] = None
    from_stall_end: Optional[float] = None
    if reached and last_end is not None:
        from_start = max(0.0, last_end - storm_start)
        if stall_end is not None:
            from_stall_end = max(0.0, last_end - stall_end)
    return {"from_start": from_start, "from_stall_end": from_stall_end}


def timeline(
    events: List[dict],
    *,
    bucket_ms: int = DEFAULT_BUCKET_MS,
    storm_start: float = 0.0,
) -> List[dict]:
    """Bucketed counts over time: attempts started, connected, timed out.

    Each bucket is ``{t_ms, started, connected, timeouts}`` where ``t_ms`` is
    the bucket's left edge in milliseconds from ``storm_start``. ``started``
    counts attempts whose ``t_start`` falls in the bucket; ``connected`` and
    ``timeouts`` count attempts whose ``t_end`` falls in the bucket.
    """
    if bucket_ms <= 0:
        raise ValueError(f"bucket_ms must be > 0, got {bucket_ms}")
    if not events:
        return []
    bucket_s = bucket_ms / 1000.0
    buckets: Dict[int, Dict[str, int]] = {}

    def slot(t: float) -> int:
        return int((t - storm_start) // bucket_s)

    for event in events:
        start_idx = slot(event["t_start"])
        buckets.setdefault(start_idx, {"started": 0, "connected": 0, "timeouts": 0})["started"] += 1
        end_idx = slot(event["t_end"])
        entry = buckets.setdefault(end_idx, {"started": 0, "connected": 0, "timeouts": 0})
        if event.get("outcome") == OUTCOME_CONNECTED:
            entry["connected"] += 1
        elif event.get("outcome") in ("connect_timeout", "reply_timeout"):
            entry["timeouts"] += 1

    return [
        {
            "t_ms": idx * bucket_ms,
            "started": buckets[idx]["started"],
            "connected": buckets[idx]["connected"],
            "timeouts": buckets[idx]["timeouts"],
        }
        for idx in sorted(buckets)
    ]


def summarize(
    events: List[dict],
    client_count: int,
    *,
    storm_start: float = 0.0,
    stall_end: Optional[float] = None,
    bucket_ms: int = DEFAULT_BUCKET_MS,
) -> dict:
    """Assemble every metric into one document (no I/O)."""
    return {
        "totals": totals(events, client_count),
        "outcomes": outcome_counts(events),
        "attempt_latency": attempt_latency_percentiles(events),
        "time_to_quiescence": time_to_quiescence(events, client_count, storm_start=storm_start, stall_end=stall_end),
        "timeline": timeline(events, bucket_ms=bucket_ms, storm_start=storm_start),
    }
