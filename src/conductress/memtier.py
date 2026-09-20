"""Helpers for driving memtier_benchmark and reading its output.

memtier_benchmark remains the load generator for scenario tasks, which need
its finite request-count mode, its per-second JSON timeseries, and a key
prefix for a separate namespace.  Everything here is pure: command-line
fragments in, parsed values out.
"""

from math import gcd
from typing import Optional

from conductress.config import PERF_BENCH_KEYSPACE

# Default memtier concurrency: 8 threads x 50 clients = 400 connections, the
# same connection count the throughput sweeps use.
MEMTIER_THREADS = 8
MEMTIER_CLIENTS = 50
MEMTIER_KEYSPACE = PERF_BENCH_KEYSPACE


def set_ratio_to_memtier_ratio(set_pct: int) -> str:
    """Convert a SET percentage (0-100) to a memtier ``--ratio=SET:GET`` string.

    E.g. ``set_pct=20`` -> ``1:4`` (one SET per four GETs).  ``0`` is pure GET
    (``0:1``) and ``100`` is pure SET (``1:0``).
    """
    if set_pct == 0:
        return "0:1"
    if set_pct == 100:
        return "1:0"
    get_pct = 100 - set_pct
    g = gcd(set_pct, get_pct)
    return f"{set_pct // g}:{get_pct // g}"


def parse_memtier_total_rps(output: str) -> Optional[float]:
    """Return the ops/sec figure from the ``Totals`` row of memtier's summary table.

    The table looks like::

        Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency  ...
        ...
        Totals      1234567.89      ...
    """
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Totals":
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None
