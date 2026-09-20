"""Fan a client population across worker processes and aggregate their events.

The runner owns the storm's shared clock origin and deadline, splits the
client population into ``workers`` contiguous slices, and runs each slice in
its own process (each process an asyncio loop). Workers stream their attempt
events back as JSON lines over a pipe; the parent collects them into one list.

``workers == 1`` runs the single slice in-process (no child process, no pipe),
which is what the unit tests use for a deterministic, few-second storm. The
default worker count is ``min(8, os.cpu_count())`` because a single Python
event loop opens only a few thousand connections per second, so a one-process
"200 ms burst" of a few thousand clients actually takes far longer; fanning
across processes lets the burst approach its nominal rate.

**Ordering (stall-first by default).** The realistic scenario is a stall
already in progress when reconnecting clients arrive, so by default the stall
command is issued first and the client burst starts ``burst_after_stall_ms``
after the stall was issued (anchored to the command-issue moment, not the
stall's reply, which only returns once the stall clears). The old behaviour --
burst first, stall injected ``stall_after_s`` into the run -- stays reachable
with ``burst_first=True``.

The stall injector runs in the PARENT on its own connection. Its wall-clock
start/end are recorded so recovery can be measured from the moment the stall
cleared; the clients share ``time.monotonic`` (system-wide on Linux, so a
monotonic value is comparable across the worker processes).

A **prewarm** phase can precede the measured storm: it opens
``prewarm_connections`` connections (with the same handshake), closes them,
waits briefly, and only then takes the netstat baseline -- so a cold server's
per-connection first-contact cost is paid before measurement rather than
contaminating it. Its own listen-overflow delta is recorded separately.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import ssl
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import netstat
from .client import ClientConfig, run_clients
from .policy import ReconnectPolicy, parse_policy
from .probe import ProbeConfig, measure_connect_capacity, run_probe
from .stall import StallInjector, StallRecord, parse_stall

DEFAULT_MAX_WORKERS = 8
# Pause after the prewarm connections close, before the baseline + storm, so
# the server settles and TIME_WAIT churn from the throwaway connections is not
# counted against the measured burst.
PREWARM_SETTLE_S = 0.5


def default_workers() -> int:
    """Default worker process count: ``min(8, os.cpu_count() or 1)``."""
    return min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)


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
    workers: int = 0  # 0 -> default_workers()
    bind_addrs: Optional[List[str]] = None
    # Ordering. Default is stall-first: the burst starts burst_after_stall_ms
    # after the stall command is issued. burst_first restores the old ordering.
    burst_first: bool = False
    burst_after_stall_ms: int = 200
    # Seconds after the origin at which the stall is injected, in BOTH orderings.
    # Stall-first: the stall waits this long, then the burst follows it by
    # burst_after_stall_ms -- so a probe (and, with burst_first, a pre-connected
    # herd) gets a baseline before the event. burst_first: the burst starts at the
    # origin and the stall lands this long after it.
    stall_after_s: float = 1.0
    # Prewarm: None -> default to `clients`; 0 -> disabled.
    prewarm_connections: Optional[int] = None
    bucket_ms: int = 100
    # TLS: wrap the herd's connections in TLS verified against this CA cert.
    # None keeps plain TCP. The plaintext port is still used by the probe (and
    # by Conductress's own management traffic) unless probe_tls is set.
    tls: bool = False
    tls_ca_cert: Optional[str] = None
    # The server's TLS port. When tls is set the herd connects here (the whole
    # herd runs over TLS); config.port stays the PLAINTEXT port used by the
    # stall injector (DEBUG SLEEP is plaintext management traffic), the connect
    # capacity baseline, and the probe's default target. None with tls set is a
    # configuration error caught in config_from_args.
    tls_port: Optional[int] = None
    # Active herd: a connected client sends first_command every N ms (0 = idle).
    herd_command_interval_ms: float = 0.0
    # Fixed-rate goodput probe (0 clients disables it).
    probe_clients: int = 0
    probe_rate: int = 0
    probe_tls: bool = False  # probe uses the plaintext port unless set
    # Probe target port; None -> the herd's own --port. When the herd is TLS on
    # a dedicated TLS port, the probe still measures the main thread's goodput
    # on the PLAINTEXT port, so the runner passes that port here.
    probe_port: Optional[int] = None

    def resolved_workers(self) -> int:
        return self.workers if self.workers and self.workers > 0 else default_workers()

    def resolved_prewarm(self) -> int:
        return self.clients if self.prewarm_connections is None else self.prewarm_connections

    def _tls_context(self) -> Optional["ssl.SSLContext"]:
        """Build the herd's SSLContext from the CA cert, or None for plain TCP.

        ``check_hostname`` is off because the herd targets loopback aliases
        (127.0.0.2, ...) that the server cert does not name; the CA is still
        verified, so a wrong cert is rejected.
        """
        if not self.tls:
            return None
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.tls_ca_cert)
        ctx.check_hostname = False
        return ctx

    def client_config(self) -> ClientConfig:
        # The herd connects to the TLS port when TLS is on; otherwise the
        # plaintext port. Everything else (stall, capacity, probe) uses port.
        herd_port = self.tls_port if (self.tls and self.tls_port is not None) else self.port
        return ClientConfig(
            host=self.host,
            port=herd_port,
            connect_timeout=self.connect_timeout_ms / 1000.0,
            reply_timeout=self.reply_timeout_ms / 1000.0,
            handshake=list(self.handshake),
            first_command=self.first_command,
            bind_addrs=list(self.bind_addrs) if self.bind_addrs else None,
            ssl_context=self._tls_context(),
            herd_command_interval_ms=self.herd_command_interval_ms,
        )

    def probe_config(self) -> "ProbeConfig":
        """The fixed-rate probe's config, on the plaintext port unless probe_tls."""
        ctx = self._tls_context() if self.probe_tls else None
        return ProbeConfig(
            host=self.host,
            port=self.probe_port if self.probe_port is not None else self.port,
            clients=self.probe_clients,
            rate=self.probe_rate,
            reply_timeout=self.reply_timeout_ms / 1000.0,
            connect_timeout=self.connect_timeout_ms / 1000.0,
            handshake=list(self.handshake),
            first_command=self.first_command,
            ssl_context=ctx,
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


def _start_offsets(client_ids: range, burst_ms: int, burst_origin: float) -> List[float]:
    """Uniformly spread each client's first-attempt time over ``[burst_origin, +burst]``."""
    total = len(client_ids)
    if total == 0:
        return []
    burst_s = burst_ms / 1000.0
    if total == 1 or burst_s <= 0:
        return [burst_origin for _ in client_ids]
    step = burst_s / total
    return [burst_origin + i * step for i in range(total)]


async def _run_slice_async(
    config: StormConfig, ids: range, origin: float, burst_origin: float, deadline: float
) -> "tuple[List[dict], int]":
    """Run one contiguous slice of clients in this process's event loop.

    ``origin`` is the shared monotonic clock origin (events are stamped
    relative to it); ``burst_origin`` is the monotonic time the slice's first
    attempts spread out from (>= origin, later than origin in stall-first mode).
    Returns ``(events, herd_commands_ok)``.
    """

    def clock() -> float:
        return time.monotonic() - origin

    offsets = [off - origin for off in _start_offsets(ids, config.burst_ms, burst_origin)]
    return await run_clients(
        list(ids),
        config.client_config(),
        config.policy(),
        clock,
        offsets,
        deadline - origin,
    )


def _worker_main(
    config: StormConfig, start: int, stop: int, origin: float, burst_origin: float, deadline: float, conn
) -> None:
    """Child-process entry point: run a slice and stream its events as JSON lines.

    A final ``{"__summary__": {...}}`` line carries the slice's aggregate
    counters (herd_commands_ok) so the parent can total them across workers.
    """
    try:
        events, herd_commands_ok = asyncio.run(
            _run_slice_async(config, range(start, stop), origin, burst_origin, deadline)
        )
        for event in events:
            conn.send_bytes((json.dumps(event) + "\n").encode())
        conn.send_bytes((json.dumps({"__summary__": {"herd_commands_ok": herd_commands_ok}}) + "\n").encode())
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
    prewarm_listen_delta: dict
    prewarm_connections: int
    config: StormConfig
    probe_timeline: List[dict] = field(default_factory=list)
    herd_commands_ok: int = 0
    herd_connects_per_s_capacity: Optional[float] = None
    openssl_version: Optional[str] = None

    def stall_end_relative(self) -> Optional[float]:
        """Stall end as seconds from the storm start, or None if no stall fired."""
        if self.stall.ended is None:
            return None
        return max(0.0, self.stall.ended - self.origin_wall)


async def _prewarm(config: StormConfig) -> dict:
    """Open, briefly hold, then close ``prewarm_connections`` connections.

    Pays a cold server's per-connection first-contact cost before measurement.
    Returns this phase's own listen-overflow delta so the cold-start effect is
    itself observable. Connections are opened in bounded-concurrency waves --
    "sequentially-ish" -- so the prewarm does not itself become a storm; its
    purpose is to touch every connection slot once, not to race.
    """
    count = config.resolved_prewarm()
    if count <= 0:
        return netstat.counter_delta(None, None)
    before = netstat.read_counters()
    cfg = config.client_config()
    handshake = list(cfg.handshake)

    async def _touch() -> None:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(cfg.host, cfg.port), timeout=cfg.connect_timeout
            )
        except (OSError, asyncio.TimeoutError):
            return
        try:
            from .resp import ReplyParser, encode_command

            for command in handshake:
                writer.write(encode_command(*command.split()))
            if handshake:
                await writer.drain()
                parser = ReplyParser()
                try:
                    while True:
                        chunk = await asyncio.wait_for(reader.read(4096), timeout=cfg.reply_timeout)
                        if not chunk:
                            break
                        parser.feed(chunk)
                        complete, _ = parser.try_parse()
                        if complete:
                            break
                except (OSError, asyncio.TimeoutError):
                    pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass

    wave = 64
    for base in range(0, count, wave):
        await asyncio.gather(*(_touch() for _ in range(base, min(base + wave, count))))
    after = netstat.read_counters()
    await asyncio.sleep(PREWARM_SETTLE_S)
    return netstat.counter_delta(before, after)


async def run_storm(config: StormConfig) -> StormResult:
    """Run a full storm and return the raw result (events + stall + netstat deltas).

    Order: prewarm (optional) -> netstat baseline -> stall/burst (stall-first by
    default) -> netstat delta. In-process for one worker; otherwise one child
    process per slice with events streamed over a pipe. When a probe is
    configured it runs on the shared clock for the whole run (it captures the
    pre-storm baseline). When TLS is on, a startup connect-capacity check is
    measured so a generator-bound herd is flagged.
    """
    openssl_version = ssl.OPENSSL_VERSION

    # Capacity check (before the measured run): sequential connect+close cycles
    # at the herd's own transport (the TLS port + TLS context when TLS is on),
    # so a herd whose target rate exceeds what the generator can drive is
    # flagged rather than misread as a server ceiling.
    herd_port = config.tls_port if (config.tls and config.tls_port is not None) else config.port
    capacity = await measure_connect_capacity(
        ProbeConfig(
            host=config.host,
            port=herd_port,
            clients=1,
            rate=1,
            reply_timeout=config.reply_timeout_ms / 1000.0,
            connect_timeout=config.connect_timeout_ms / 1000.0,
            handshake=list(config.handshake),
            first_command=config.first_command,
            ssl_context=config._tls_context(),
        )
    )

    prewarm_delta = await _prewarm(config)

    origin = time.monotonic()
    origin_wall = time.time()
    deadline = origin + config.duration_s
    before = netstat.read_counters()

    injects = config.stall().injects
    burst_origin = origin  # burst_first, or no stall: burst starts at origin
    if injects and not config.burst_first:
        # Stall-first: the burst is anchored to the stall command's issue time.
        # Start the stall, and only begin the burst once it has been issued.
        issued = asyncio.Event()

        async def _driver() -> StallRecord:
            return await _drive_stall(config, delay_from=origin, on_issued=issued.set)

        stall_task = asyncio.ensure_future(_driver())
        await issued.wait()
        burst_origin = time.monotonic() + config.burst_after_stall_ms / 1000.0
        deadline = burst_origin + config.duration_s
    else:
        # burst_first (or no stall): burst at origin, stall (if any) at stall_after_s.
        stall_task = asyncio.ensure_future(_drive_stall(config, delay_from=origin))

    # The probe runs on the shared clock for the whole run (origin..deadline),
    # so it carries the pre-storm baseline the recovery metric compares against.
    def clock() -> float:
        return time.monotonic() - origin

    probe_task: Optional[asyncio.Future] = None
    if config.probe_clients > 0 and config.probe_rate > 0:
        probe_task = asyncio.ensure_future(run_probe(config.probe_config(), clock, deadline - origin))

    events, herd_commands_ok = await _run_clients(config, origin, burst_origin, deadline)

    stall_record = await stall_task
    probe_timeline: List[dict] = []
    if probe_task is not None:
        probe_timeline = await probe_task
    after = netstat.read_counters()
    return StormResult(
        events=events,
        stall=stall_record,
        origin_wall=origin_wall,
        duration_s=config.duration_s,
        listen_delta=netstat.counter_delta(before, after),
        prewarm_listen_delta=prewarm_delta,
        prewarm_connections=config.resolved_prewarm(),
        config=config,
        probe_timeline=probe_timeline,
        herd_commands_ok=herd_commands_ok,
        herd_connects_per_s_capacity=capacity,
        openssl_version=openssl_version,
    )


async def _run_clients(
    config: StormConfig, origin: float, burst_origin: float, deadline: float
) -> "tuple[List[dict], int]":
    """Run the client population in-process or across worker processes."""
    if config.resolved_workers() <= 1:
        return await _run_slice_async(config, range(config.clients), origin, burst_origin, deadline)
    return await _run_multiprocess(config, origin, burst_origin, deadline)


async def _drive_stall(config: StormConfig, *, delay_from: Optional[float] = None, on_issued=None) -> StallRecord:
    """Inject the stall (parent side).

    ``delay_from`` waits ``stall_after_s`` past that origin (both orderings)
    before injecting. In stall-first mode ``delay_from`` is None (inject at
    once) and ``on_issued`` fires when the stall command has been sent.
    """
    injector = config.stall()
    if not injector.injects:
        return StallRecord(kind=injector.kind)
    # A time-bounded stall (a slow loop) must end inside the storm window;
    # clamp it against the storm duration and the burst offset. One-shot stalls
    # ignore this (their duration is sized by the caller).
    injector.clamp_to_storm(max(1.0, config.duration_s - config.stall_after_s), config.burst_after_stall_ms / 1000.0)
    if delay_from is not None:
        wait = delay_from + config.stall_after_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
    return await injector.run(config.host, config.port, config.connect_timeout_ms / 1000.0, on_issued=on_issued)


async def _run_multiprocess(
    config: StormConfig, origin: float, burst_origin: float, deadline: float
) -> "tuple[List[dict], int]":
    """Spawn one child per client slice; collect streamed JSON-line events."""
    ctx = mp.get_context("spawn")
    procs = []
    parent_conns = []
    for bounds in _slice_bounds(config.clients, config.resolved_workers()):
        if not bounds:
            continue
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_worker_main,
            args=(config, bounds.start, bounds.stop, origin, burst_origin, deadline, child_conn),
        )
        proc.start()
        child_conn.close()  # parent keeps only its read end
        procs.append(proc)
        parent_conns.append(parent_conn)

    # Collect events off the pipes without blocking the event loop.
    events, herd_commands_ok = await asyncio.get_event_loop().run_in_executor(
        None, _collect_events, parent_conns, procs
    )
    return events, herd_commands_ok


def _collect_events(parent_conns, procs) -> "tuple[List[dict], int]":
    """Read newline-framed JSON events from every child until all pipes close.

    A ``{"__summary__": {...}}`` line is the slice's aggregate counters, not an
    attempt event: it is summed into ``herd_commands_ok`` rather than appended.
    """
    events: List[dict] = []
    herd_commands_ok = 0
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
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    summary = record.get("__summary__") if isinstance(record, dict) else None
                    if summary is not None:
                        herd_commands_ok += int(summary.get("herd_commands_ok", 0))
                    else:
                        events.append(record)
        finally:
            conn.close()
    for proc in procs:
        proc.join()
    return events, herd_commands_ok
