"""Fan a client population across worker processes and aggregate their events.

The runner owns the storm's shared clock origin and deadline, splits the
client population into ``workers`` contiguous slices, and runs each slice in
its own process (each process an asyncio loop). Workers stream their attempt
events back as JSON lines over a pipe; the parent collects them into one list.

``workers == 1`` runs the single slice in-process (no child process, no pipe),
which is what the unit tests use for a deterministic, few-second storm.

The stall injector runs in the PARENT, on its own connection, scheduled to
fire ``stall_after`` seconds into the run. Its wall-clock start/end are
recorded so recovery can be measured from the moment the stall cleared. The
clock the clients share is ``time.monotonic``; the stall record is wall-clock
(``time.time``) and converted to the shared monotonic origin for metrics.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import netstat
from .client import ClientConfig, run_clients
from .policy import ReconnectPolicy, parse_policy
from .stall import StallInjector, StallRecord, parse_stall


@dataclass
class StormConfig:
    """A full storm run's parameters (generator side)."""

    host: str
    port: int
    clients: int
    burst_ms: int
    duration_s: float
    connect_timeout_ms: float
    reply_timeout_ms: float
    policy_spec: str
    stall_spec: str = "none"
    handshake: List[str] = field(default_factory=lambda: ["HELLO 3"])
    first_command: str = "GET stormkey"
    workers: int = 1
    bind_addrs: Optional[List[str]] = None
    stall_after_s: float = 1.0
    bucket_ms: int = 100

    def client_config(self) -> ClientConfig:
        return ClientConfig(
            host=self.host,
            port=self.port,
            connect_timeout=self.connect_timeout_ms / 1000.0,
            reply_timeout=self.reply_timeout_ms / 1000.0,
            handshake=list(self.handshake),
            first_command=self.first_command,
            bind_addrs=list(self.bind_addrs) if self.bind_addrs else None,
        )

    def policy(self) -> ReconnectPolicy:
        return parse_policy(self.policy_spec)

    def stall(self) -> StallInjector:
        return parse_stall(self.stall_spec)


def _slice_bounds(total: int, workers: int) -> List[range]:
    """Split ``range(total)`` into ``workers`` contiguous, near-equal ranges."""
    workers = max(1, min(workers, total)) if total else 1
    base, extra = divmod(total, workers)
    bounds: List[range] = []
    start = 0
    for w in range(workers):
        size = base + (1 if w < extra else 0)
        bounds.append(range(start, start + size))
        start += size
    return bounds


def _start_offsets(client_ids: range, burst_ms: int, origin: float) -> List[float]:
    """Uniformly spread each client's first-attempt time over ``[origin, origin+burst]``."""
    total = len(client_ids)
    if total == 0:
        return []
    burst_s = burst_ms / 1000.0
    if total == 1 or burst_s <= 0:
        return [origin for _ in client_ids]
    step = burst_s / total
    return [origin + i * step for i in range(total)]


async def _run_slice_async(config: StormConfig, ids: range, origin: float, deadline: float) -> List[dict]:
    """Run one contiguous slice of clients in this process's event loop."""

    def clock() -> float:
        return time.monotonic() - origin

    offsets = [off - origin for off in _start_offsets(ids, config.burst_ms, origin)]
    return await run_clients(
        list(ids),
        config.client_config(),
        config.policy(),
        clock,
        offsets,
        deadline - origin,
    )


def _worker_main(config: StormConfig, start: int, stop: int, origin: float, deadline: float, conn) -> None:
    """Child-process entry point: run a slice and stream its events as JSON lines."""
    try:
        events = asyncio.run(_run_slice_async(config, range(start, stop), origin, deadline))
        for event in events:
            conn.send_bytes((json.dumps(event) + "\n").encode())
    finally:
        conn.close()


@dataclass
class StormResult:
    """Everything one run produced, before metric reduction."""

    events: List[dict]
    stall: StallRecord
    origin_wall: float  # time.time() at the shared monotonic origin
    duration_s: float
    listen_delta: dict
    config: StormConfig

    def stall_end_relative(self) -> Optional[float]:
        """Stall end as seconds from the storm start, or None if no stall fired."""
        if self.stall.ended is None:
            return None
        return max(0.0, self.stall.ended - self.origin_wall)


async def _drive_stall(config: StormConfig, origin: float) -> StallRecord:
    """Wait ``stall_after_s`` into the run, then inject the stall (parent side)."""
    injector = config.stall()
    if not injector.injects:
        return StallRecord(kind=injector.kind)
    wait = origin + config.stall_after_s - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    return await injector.run(config.host, config.port, config.connect_timeout_ms / 1000.0)


async def run_storm(config: StormConfig) -> StormResult:
    """Run a full storm and return the raw result (events + stall + netstat delta).

    In-process for ``workers == 1``; otherwise one child process per slice with
    events streamed back over a pipe. The stall injector and (for the
    single-process case) the client slice run concurrently on the parent loop.
    """
    origin = time.monotonic()
    origin_wall = time.time()
    deadline = origin + config.duration_s

    before = netstat.read_counters()
    stall_task = asyncio.ensure_future(_drive_stall(config, origin))

    if config.workers <= 1:
        events = await _run_slice_async(config, range(config.clients), origin, deadline)
    else:
        events = await _run_multiprocess(config, origin, deadline)

    stall_record = await stall_task
    after = netstat.read_counters()
    return StormResult(
        events=events,
        stall=stall_record,
        origin_wall=origin_wall,
        duration_s=config.duration_s,
        listen_delta=netstat.counter_delta(before, after),
        config=config,
    )


async def _run_multiprocess(config: StormConfig, origin: float, deadline: float) -> List[dict]:
    """Spawn one child per client slice; collect streamed JSON-line events."""
    ctx = mp.get_context("spawn")
    procs = []
    parent_conns = []
    for bounds in _slice_bounds(config.clients, config.workers):
        if not bounds:
            continue
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_worker_main,
            args=(config, bounds.start, bounds.stop, origin, deadline, child_conn),
        )
        proc.start()
        child_conn.close()  # parent keeps only its read end
        procs.append(proc)
        parent_conns.append(parent_conn)

    # Collect events off the pipes without blocking the event loop.
    events = await asyncio.get_event_loop().run_in_executor(None, _collect_events, parent_conns, procs)
    return events


def _collect_events(parent_conns, procs) -> List[dict]:
    """Read newline-framed JSON events from every child until all pipes close."""
    events: List[dict] = []
    for conn in parent_conns:
        buffer = b""
        try:
            while True:
                try:
                    chunk = conn.recv_bytes()
                except EOFError:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        events.append(json.loads(line))
        finally:
            conn.close()
    for proc in procs:
        proc.join()
    return events
