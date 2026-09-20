"""A fixed-rate open-loop goodput probe.

A few steady clients send ``first_command`` at a fixed aggregate rate against
the server's PLAINTEXT port (by default) and measure the main thread's goodput:
served commands per second and per-request latency percentiles at a 1 s
resolution. It runs for the whole generator lifetime -- from before the storm's
start delay is over (so it captures the pre-storm baseline) to the end -- on the
storm's shared monotonic clock, so its timeline lines up with the storm and the
sampler.

Open-loop means it sends on schedule regardless of replies: at most one request
is outstanding per client, and if the previous one has not returned when the
next slot arrives, the slot is *skipped* (counted ``skipped``) rather than
queued. That is what makes it a goodput measure -- a stalled main thread shows
as skips and a served/s collapse, not as a growing backlog that hides the stall.

This lives in stormgen (not memtier) because it needs per-request latencies,
which memtier's per-second aggregate output does not give.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import metrics as metrics_mod
from .resp import ReplyParser, encode_command


@dataclass
class ProbeConfig:
    """Parameters for the fixed-rate probe."""

    host: str
    port: int
    clients: int
    rate: int  # aggregate commands/s across all probe clients
    reply_timeout: float
    connect_timeout: float
    handshake: List[str] = field(default_factory=lambda: ["HELLO 3"])
    first_command: str = "GET stormkey"
    ssl_context: Optional[ssl.SSLContext] = None
    bucket_ms: int = 1000

    def per_client_interval(self) -> float:
        """Seconds between one client's sends: clients / rate."""
        if self.rate <= 0 or self.clients <= 0:
            return 0.0
        return self.clients / float(self.rate)


@dataclass
class ProbeSample:
    """One completed (or skipped) probe request, on the shared clock."""

    t_send: float  # scheduled send time (clock seconds)
    latency_ms: Optional[float]  # None when skipped or failed
    served: bool
    skipped: bool


def _bucketize(samples: List[ProbeSample], bucket_ms: int) -> List[dict]:
    """Reduce raw probe samples to per-bucket sent/served/skipped + latency pcts.

    Each bucket is keyed by the send time so a request counts in the second it
    was scheduled. ``t_ms`` is the bucket's left edge in ms from the shared
    origin (t=0), matching the storm timeline's convention.
    """
    if bucket_ms <= 0:
        raise ValueError(f"bucket_ms must be > 0, got {bucket_ms}")
    bucket_s = bucket_ms / 1000.0
    buckets: dict = {}
    for s in samples:
        idx = int(s.t_send // bucket_s)
        entry = buckets.setdefault(idx, {"sent": 0, "served": 0, "skipped": 0, "lat": []})
        entry["sent"] += 1
        if s.skipped:
            entry["skipped"] += 1
        elif s.served:
            entry["served"] += 1
            if s.latency_ms is not None:
                entry["lat"].append(s.latency_ms)
    out: List[dict] = []
    for idx in sorted(buckets):
        e = buckets[idx]
        lat = sorted(e["lat"])
        out.append(
            {
                "t_ms": idx * bucket_ms,
                "sent": e["sent"],
                "served": e["served"],
                "skipped": e["skipped"],
                "p50_ms": metrics_mod._percentile(lat, 50),
                "p99_ms": metrics_mod._percentile(lat, 99),
                "max_ms": lat[-1] if lat else 0.0,
            }
        )
    return out


class _ProbeClient:
    """One open-loop probe connection: reconnects on failure, one outstanding req."""

    def __init__(self, config: ProbeConfig, clock: Callable[[], float], deadline: float):
        self.config = config
        self._clock = clock
        self._deadline = deadline
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._parser = ReplyParser()

    async def _connect(self) -> bool:
        try:
            if self.config.ssl_context is not None:
                coro = asyncio.open_connection(
                    self.config.host, self.config.port, ssl=self.config.ssl_context, server_hostname=self.config.host
                )
            else:
                coro = asyncio.open_connection(self.config.host, self.config.port)
            self._reader, self._writer = await asyncio.wait_for(coro, timeout=self.config.connect_timeout)
            self._parser = ReplyParser()
            for command in self.config.handshake:
                if not await self._roundtrip(command):
                    return False
            return True
        except (OSError, asyncio.TimeoutError, ConnectionError):
            return False

    async def _roundtrip(self, command: str) -> bool:
        assert self._reader is not None and self._writer is not None
        self._writer.write(encode_command(*command.split()))
        await self._writer.drain()
        while True:
            complete, _ = self._parser.try_parse()
            if complete:
                return True
            chunk = await asyncio.wait_for(self._reader.read(65536), timeout=self.config.reply_timeout)
            if not chunk:
                return False
            self._parser.feed(chunk)

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._reader = self._writer = None

    async def run(self, offset: float, samples: List[ProbeSample]) -> None:
        """Send at a fixed cadence from ``offset`` until the deadline (open-loop).

        The round trip runs as a background task so the pacing loop keeps
        issuing on schedule regardless of replies: at most one request is
        outstanding per client, and a slot that arrives while the previous
        request is still in flight is recorded as ``skipped`` rather than
        queued. That is what makes the probe a goodput measure -- a stalled
        main thread shows as skips and a served/s collapse, not a backlog.
        """
        interval = self.config.per_client_interval()
        if interval <= 0:
            return
        # Stagger the first send by the offset so the clients' sends interleave.
        wait = offset - self._clock()
        if wait > 0:
            await asyncio.sleep(wait)
        inflight: Optional[asyncio.Future] = None
        next_send = self._clock()
        while self._clock() < self._deadline:
            now = self._clock()
            if now < next_send:
                await asyncio.sleep(min(next_send - now, max(0.0, self._deadline - now)))
                continue
            t_send = next_send
            next_send += interval
            if inflight is not None and not inflight.done():
                # Previous request still in flight at this slot: skip it.
                samples.append(ProbeSample(t_send=t_send, latency_ms=None, served=False, skipped=True))
                continue
            inflight = asyncio.ensure_future(self._one_request(t_send, samples))
        # Let a final in-flight request settle so its sample is recorded.
        if inflight is not None and not inflight.done():
            try:
                await asyncio.wait_for(inflight, timeout=self.config.reply_timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._close()

    async def _one_request(self, t_send: float, samples: List[ProbeSample]) -> None:
        """One open-loop request: (re)connect if needed, send, record one sample."""
        if self._writer is None and not await self._connect():
            samples.append(ProbeSample(t_send=t_send, latency_ms=None, served=False, skipped=False))
            self._close()
            return
        start = self._clock()
        try:
            ok = await self._roundtrip(self.config.first_command)
            if ok:
                latency_ms = (self._clock() - start) * 1000.0
                samples.append(ProbeSample(t_send=t_send, latency_ms=latency_ms, served=True, skipped=False))
            else:
                samples.append(ProbeSample(t_send=t_send, latency_ms=None, served=False, skipped=False))
                self._close()
        except (OSError, asyncio.TimeoutError, ConnectionError):
            samples.append(ProbeSample(t_send=t_send, latency_ms=None, served=False, skipped=False))
            self._close()


async def run_probe(config: ProbeConfig, clock: Callable[[], float], deadline: float) -> List[dict]:
    """Run the probe's clients concurrently; return the per-bucket ``probe_timeline``.

    ``clock`` is the storm's shared monotonic clock (origin at t=0) so the
    returned buckets' ``t_ms`` align with the storm timeline. Each client is
    offset by a fraction of the per-client interval so their sends interleave
    rather than firing in lockstep.
    """
    if config.clients <= 0 or config.rate <= 0:
        return []
    samples: List[ProbeSample] = []
    interval = config.per_client_interval()
    tasks = []
    for i in range(config.clients):
        offset = (interval / config.clients) * i if config.clients else 0.0
        tasks.append(_ProbeClient(config, clock, deadline).run(offset, samples))
    await asyncio.gather(*tasks)
    return _bucketize(samples, config.bucket_ms)


async def measure_connect_capacity(config: ProbeConfig, connects: int = 200) -> float:
    """Sequential connects/s capacity: time ``connects`` connect+close cycles.

    Opens (and, for TLS, handshakes) then closes ``connects`` connections one
    after another and returns the achieved connects/s, so a herd whose target
    connect rate exceeds what the generator host can drive is flagged in the
    result rather than misread as a server ceiling. TLS handshakes cost real
    client CPU, so this is measured with the herd's own ssl_context.
    """
    if connects <= 0:
        return 0.0
    start = time.monotonic()
    done = 0
    for _ in range(connects):
        try:
            if config.ssl_context is not None:
                coro = asyncio.open_connection(
                    config.host, config.port, ssl=config.ssl_context, server_hostname=config.host
                )
            else:
                coro = asyncio.open_connection(config.host, config.port)
            reader, writer = await asyncio.wait_for(coro, timeout=config.connect_timeout)
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass
            done += 1
        except (OSError, asyncio.TimeoutError, ConnectionError):
            continue
    elapsed = time.monotonic() - start
    return (done / elapsed) if elapsed > 0 else 0.0
