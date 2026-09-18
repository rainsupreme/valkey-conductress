"""Kernel TCP listen-queue overflow counters from ``/proc/net/netstat``.

When the server's main thread stalls, the process stops calling ``accept()``.
Connections that completed the TCP handshake sit in the kernel's accept queue;
once that queue is full the kernel drops further completions. Linux counts
these under ``TcpExt``:

* ``ListenOverflows`` -- times a SYN was dropped because the accept queue of a
  listening socket was full
* ``ListenDrops``     -- listening-socket drops for any reason (a superset)

A rise in these during a run is direct evidence the accept backlog overflowed,
which is the kernel-side half of the connection-storm story. This module reads
the counters (no ``nstat`` dependency) and computes before/after deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

NETSTAT_PATH = "/proc/net/netstat"
_FIELDS = ("ListenOverflows", "ListenDrops")


@dataclass
class ListenCounters:
    """A snapshot of the listen-queue overflow counters."""

    listen_overflows: int
    listen_drops: int

    def as_dict(self) -> Dict[str, int]:
        return {"ListenOverflows": self.listen_overflows, "ListenDrops": self.listen_drops}


def parse_netstat(text: str) -> ListenCounters:
    """Parse ``/proc/net/netstat`` content, returning the TcpExt listen counters.

    The file pairs a header line and a values line per prefix, e.g.::

        TcpExt: SyncookiesSent ... ListenOverflows ListenDrops ...
        TcpExt: 0 ... 12 15 ...

    Missing fields default to 0 so a kernel that does not expose a counter
    reads as "no overflows" rather than raising.
    """
    header: Optional[list] = None
    values: Optional[list] = None
    for line in text.splitlines():
        if not line.startswith("TcpExt:"):
            continue
        tokens = line.split()[1:]
        if header is None:
            header = tokens
        else:
            values = tokens
            break
    counts: Dict[str, int] = {}
    if header and values and len(header) == len(values):
        index = {name: i for i, name in enumerate(header)}
        for field in _FIELDS:
            if field in index:
                try:
                    counts[field] = int(values[index[field]])
                except ValueError:
                    counts[field] = 0
    return ListenCounters(
        listen_overflows=counts.get("ListenOverflows", 0),
        listen_drops=counts.get("ListenDrops", 0),
    )


def read_counters(path: str = NETSTAT_PATH) -> Optional[ListenCounters]:
    """Read and parse the counters from ``path``; ``None`` if unreadable.

    Unreadable is not an error: the generator runs on any platform, and the
    counters are simply absent where ``/proc/net/netstat`` does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return parse_netstat(handle.read())
    except OSError:
        return None


def counter_delta(before: Optional[ListenCounters], after: Optional[ListenCounters]) -> Dict[str, Optional[int]]:
    """Return the field-wise ``after - before`` delta.

    Each field is ``None`` when either snapshot is missing, so a consumer can
    tell "no overflow" (0) apart from "counters unavailable" (None).
    """
    if before is None or after is None:
        return {"ListenOverflows": None, "ListenDrops": None}
    return {
        "ListenOverflows": after.listen_overflows - before.listen_overflows,
        "ListenDrops": after.listen_drops - before.listen_drops,
    }
