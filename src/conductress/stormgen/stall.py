"""Server-side stall injectors.

A stall injector runs on its own dedicated connection and makes the server's
main thread unavailable for a bounded interval, so the storm clients see their
first command block. The mechanism is what a connection storm forms around:
while the main thread is busy, freshly accepted connections cannot be served
and pile up in the kernel accept backlog.

Two regimes of stall are represented. A **hard stall** stops the main thread
completely for a window; a **slow loop** keeps the main thread progressing but
with long event-loop iterations, so it stays responsive between blocks yet
never has a full idle cycle to drain the accept queue. The two regimes stress
different server limits, so both are worth injecting.

Injectors are parsed from a compact spec string:

* ``none``                  -- no stall (baseline / control run)
* ``debug-sleep:<seconds>`` -- ``DEBUG SLEEP <seconds>`` on a dedicated
                               connection; blocks the main thread for the whole
                               duration (a hard stall). Portable across Redis
                               and Valkey.
* ``slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s>`` -- a repeating
                               partial stall: every ``period_ms`` the main
                               thread is blocked for ``block_ms`` (a duty cycle
                               of ``block_ms / period_ms``) for ``duration_s``.
                               ``kind`` selects how each block is produced --
                               ``debug-sleep`` (``DEBUG SLEEP``, exact and
                               dataset-independent) or ``lua`` (a self-calibrated
                               busy ``EVAL`` loop).

The interface (:meth:`StallInjector.run`) is intentionally small so future
injectors -- a large ``KEYS`` scan, say -- are one class plus one line in
:func:`parse_stall`. Each run records wall-clock start and end so the recovery
window can be measured from the moment the stall cleared; a slow loop's window
spans the whole degraded period (first block issued to loop end).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

from .resp import ReplyParser, encode_command

logger = logging.getLogger(__name__)

# A single slow-loop ``lua`` block is kept far below the server's
# ``busy-reply-threshold`` (5 s by default) so a block never trips the
# "BUSY script" error; the loop's whole point is a main thread that keeps
# progressing, not one that wedges.
LUA_MAX_BLOCK_MS = 1000.0
# Iteration count used for the two calibration rounds at loop start, before
# any measured block has scaled it. Small enough to be quick, large enough to
# be measurable.
LUA_CALIBRATION_ITERS = 200_000
# Per-step re-scale of the lua iteration count is clamped to this multiplicative
# range, so one noisy round trip cannot swing the block length wildly.
LUA_RESCALE_CLAMP = (0.5, 2.0)


@dataclass
class StallRecord:
    """When a stall started and ended (wall-clock, ``time.time()``).

    A hard stall (``debug-sleep``) sets only ``started`` / ``ended`` around its
    single blocking command. A slow loop additionally records how the repeated
    blocking behaved so the achieved duty cycle is visible in the result:

    * ``blocks_issued`` -- how many blocking commands the loop ran.
    * ``achieved_block_ms_p50`` / ``achieved_block_ms_max`` -- the measured
      block durations (the round-trip of each blocking command), in ms.
    * ``effective_duty`` -- sum of measured block durations over the loop's wall
      time; the realized ``block / period`` fraction.

    For a slow loop ``started`` is the first block issued and ``ended`` is the
    loop end, so the plot's shaded band covers the whole degraded window.
    """

    kind: str
    started: Optional[float] = None
    ended: Optional[float] = None
    blocks_issued: int = 0
    achieved_block_ms_p50: Optional[float] = None
    achieved_block_ms_max: Optional[float] = None
    effective_duty: Optional[float] = None


class StallInjector(ABC):
    """Runs a server-side stall on a dedicated connection and records its span."""

    kind: str = "none"

    @abstractmethod
    async def run(
        self, host: str, port: int, connect_timeout: float, on_issued: Optional[Callable[[], None]] = None
    ) -> StallRecord:
        """Perform the stall (opening its own connection) and return its record.

        ``on_issued`` is called once, right after the stall command has been
        sent (``record.started`` set), so a caller can anchor other work (a
        client burst) to the moment the stall began rather than to when its
        reply returns.
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Return the canonical spec string."""
        raise NotImplementedError

    @property
    def injects(self) -> bool:
        """True when this injector actually stalls the server."""
        return True

    def clamp_to_storm(self, storm_duration_s: float, burst_after_stall_s: float) -> None:
        """Shrink a time-bounded stall so it ends within the storm window.

        A no-op for injectors with no internal duration (a one-shot
        ``DEBUG SLEEP`` is sized by the caller). :class:`SlowLoopStall`
        overrides this to clamp its loop.
        """


class NoStall(StallInjector):
    """Baseline: never stalls the server."""

    kind = "none"

    async def run(
        self, host: str, port: int, connect_timeout: float, on_issued: Optional[Callable[[], None]] = None
    ) -> StallRecord:
        return StallRecord(kind=self.kind)

    def describe(self) -> str:
        return "none"

    @property
    def injects(self) -> bool:
        return False


class DebugSleepStall(StallInjector):
    """Block the main thread with ``DEBUG SLEEP <seconds>``.

    Opens its own connection so the stall is independent of the storm clients,
    issues the command, and waits for the reply (which only returns once the
    sleep is over), recording the true start and end around the blocking call.
    """

    kind = "debug-sleep"

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError(f"debug-sleep seconds must be > 0, got {seconds}")
        self.seconds = seconds

    async def run(
        self, host: str, port: int, connect_timeout: float, on_issued: Optional[Callable[[], None]] = None
    ) -> StallRecord:
        record = StallRecord(kind=self.kind)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=connect_timeout)
        try:
            writer.write(encode_command("DEBUG", "SLEEP", _fmt_seconds(self.seconds)))
            await writer.drain()
            record.started = time.time()
            if on_issued is not None:
                on_issued()
            parser = ReplyParser()
            # The reply arrives only after the sleep completes; read until we
            # have one full reply. A generous ceiling guards against a hang.
            deadline = time.monotonic() + self.seconds + max(connect_timeout, 5.0)
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=self.seconds + connect_timeout + 5.0)
                if not chunk:
                    break
                parser.feed(chunk)
                complete, _ = parser.try_parse()
                if complete or time.monotonic() > deadline:
                    break
            record.ended = time.time()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass
        return record

    def describe(self) -> str:
        return f"debug-sleep:{_fmt_seconds(self.seconds)}"


async def _read_one_reply(reader: asyncio.StreamReader, timeout: float) -> object:
    """Read exactly one full RESP reply from ``reader``, bounded by ``timeout``.

    Returns the parsed reply value (or ``None`` on a closed connection before a
    complete reply). Shared by the slow-loop blocks, which need each command's
    reply before timing the next one.
    """
    parser = ReplyParser()
    while True:
        complete, value = parser.try_parse()
        if complete:
            return value
        chunk = await asyncio.wait_for(reader.read(65536), timeout=timeout)
        if not chunk:
            return None
        parser.feed(chunk)


class SlowLoopStall(StallInjector):
    """A repeating partial stall: block ``block_ms`` out of every ``period_ms``.

    Unlike :class:`DebugSleepStall`, the main thread keeps progressing between
    blocks -- it is degraded, not frozen. On its own connection the loop, until
    ``duration_s`` has elapsed since the first block, records the time ``t``,
    issues ONE blocking command sized to ``block_ms``, awaits its reply, then
    sleeps until ``period_ms`` has elapsed since ``t``. ``block / period`` is
    the duty cycle.

    ``kind`` selects the block:

    * ``debug-sleep`` -- ``DEBUG SLEEP <block_ms/1000>``. Exact and
      dataset-independent (both Redis and Valkey accept fractional seconds).
    * ``lua`` -- ``EVAL`` of a pure busy loop. The per-iteration cost is
      unknown up front, so the loop calibrates: it times two rounds at a fixed
      iteration count at start, scales the count to ``block_ms``, then re-scales
      gently from each measured round trip (proportional correction, clamped
      per step). A single block is kept below 1 s (well under the 5 s
      ``busy-reply-threshold``), so ``block_ms`` above 1000 is rejected for this
      kind.

    The ``on_issued`` callback fires ONCE, at the first block, so the client
    burst is anchored to the moment the degraded window begins; the burst then
    arrives while the loop is running, which is the point.
    """

    kind = "slow-loop"

    # Pure busy loop, no keys: sum 1..N so the work cannot be optimized away and
    # the cost scales linearly with the iteration count ARGV[1].
    _LUA_SCRIPT = "local x=0 for i=1,tonumber(ARGV[1]) do x=x+i end return x"

    def __init__(self, block_kind: str, block_ms: float, period_ms: float, duration_s: float) -> None:
        if block_kind not in ("debug-sleep", "lua"):
            raise ValueError(f"slow-loop kind must be 'debug-sleep' or 'lua', got {block_kind!r}")
        if not 0 < block_ms <= period_ms:
            raise ValueError(
                f"slow-loop needs 0 < block_ms <= period_ms, got block_ms={block_ms}, period_ms={period_ms}"
            )
        if duration_s <= 0:
            raise ValueError(f"slow-loop duration_s must be > 0, got {duration_s}")
        if block_kind == "lua" and block_ms > LUA_MAX_BLOCK_MS:
            raise ValueError(
                f"slow-loop lua block_ms must be <= {int(LUA_MAX_BLOCK_MS)} "
                f"(kept below the busy-reply-threshold), got {block_ms}"
            )
        self.block_kind = block_kind
        self.block_ms = float(block_ms)
        self.period_ms = float(period_ms)
        self.duration_s = float(duration_s)

    def clamp_to_storm(self, storm_duration_s: float, burst_after_stall_s: float) -> None:
        """Clamp the loop so it fits inside the storm window.

        The burst begins ``burst_after_stall_s`` after the first block, so a
        loop that runs for its full ``duration_s`` plus that offset can outlast
        the storm's own duration. When it would, shrink the loop to the storm
        duration minus 1 s (never below one period) and log a warning.
        """
        if self.duration_s + burst_after_stall_s <= storm_duration_s:
            return
        clamped = max(self.period_ms / 1000.0, storm_duration_s - 1.0)
        logger.warning(
            "slow-loop: duration_s %.1f + burst_after_stall %.2f exceeds the %.1fs storm window; "
            "clamping the loop to %.1fs",
            self.duration_s,
            burst_after_stall_s,
            storm_duration_s,
            clamped,
        )
        self.duration_s = clamped

    async def run(
        self, host: str, port: int, connect_timeout: float, on_issued: Optional[Callable[[], None]] = None
    ) -> StallRecord:
        record = StallRecord(kind=self.kind)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=connect_timeout)
        # A block reply must never be given up on mid-block; allow the block
        # itself plus generous slack.
        reply_timeout = self.block_ms / 1000.0 + max(connect_timeout, 5.0)
        block_durations: List[float] = []
        try:
            iters = 0
            if self.block_kind == "lua":
                iters = await self._calibrate_lua(reader, writer, reply_timeout)
            period_s = self.period_ms / 1000.0
            first = True
            loop_start = time.monotonic()
            while True:
                now = time.monotonic()
                if not first and (now - loop_start) >= self.duration_s:
                    break
                t = now
                measured, iters = await self._one_block(reader, writer, reply_timeout, iters)
                if first:
                    record.started = time.time()
                    loop_start = t
                    if on_issued is not None:
                        on_issued()
                    first = False
                block_durations.append(measured)
                record.blocks_issued += 1
                # Sleep until period_ms has elapsed since this block began.
                sleep_for = period_s - (time.monotonic() - t)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            record.ended = time.time()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass
        self._finalize(record, block_durations)
        return record

    def _finalize(self, record: StallRecord, block_durations: List[float]) -> None:
        """Fill the achieved-block percentiles and effective duty from measurements."""
        if not block_durations:
            return
        ordered = sorted(block_durations)
        rank = max(1, min(len(ordered), round(0.5 * len(ordered))))
        record.achieved_block_ms_p50 = ordered[rank - 1] * 1000.0
        record.achieved_block_ms_max = ordered[-1] * 1000.0
        if record.started is not None and record.ended is not None:
            wall = record.ended - record.started
            if wall > 0:
                record.effective_duty = sum(block_durations) / wall

    async def _calibrate_lua(self, reader, writer, reply_timeout: float) -> int:
        """Time two calibration rounds and return the iteration count for block_ms."""
        target_s = self.block_ms / 1000.0
        iters = LUA_CALIBRATION_ITERS
        measured = 0.0
        for _ in range(2):
            measured, _ = await self._one_block(reader, writer, reply_timeout, iters)
            if measured > 0:
                per_iter = measured / iters
                if per_iter > 0:
                    iters = max(1, int(target_s / per_iter))
        return iters

    async def _one_block(self, reader, writer, reply_timeout: float, iters: int):
        """Issue one block, await its reply, and return (measured_seconds, next_iters).

        For ``debug-sleep`` the next iteration count is meaningless (returned
        unchanged). For ``lua`` the count is re-scaled from the measured round
        trip with a clamped proportional correction, so drift in the block
        length is gently corrected without overreacting to one noisy round.
        """
        start = time.monotonic()
        if self.block_kind == "debug-sleep":
            writer.write(encode_command("DEBUG", "SLEEP", _fmt_seconds(self.block_ms / 1000.0)))
        else:
            writer.write(encode_command("EVAL", self._LUA_SCRIPT, "0", str(iters)))
        await writer.drain()
        await _read_one_reply(reader, reply_timeout)
        measured = time.monotonic() - start
        next_iters = iters
        if self.block_kind == "lua" and measured > 0:
            target_s = self.block_ms / 1000.0
            factor = target_s / measured
            factor = max(LUA_RESCALE_CLAMP[0], min(LUA_RESCALE_CLAMP[1], factor))
            next_iters = max(1, int(iters * factor))
        return measured, next_iters

    def describe(self) -> str:
        return (
            f"slow-loop:{self.block_kind}:{_fmt_seconds(self.block_ms)}:"
            f"{_fmt_seconds(self.period_ms)}:{_fmt_seconds(self.duration_s)}"
        )


def _fmt_seconds(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds)


def parse_stall(spec: str) -> StallInjector:
    """Parse a stall spec string into a :class:`StallInjector`.

    Raises :class:`ValueError` naming the accepted forms for an unknown spec.
    """
    text = (spec or "").strip()
    if not text or text.lower() == "none":
        return NoStall()
    head, _, tail = text.partition(":")
    head = head.lower()
    if head == "debug-sleep":
        if not tail:
            raise ValueError("'debug-sleep' requires seconds: debug-sleep:<seconds>")
        try:
            seconds = float(tail)
        except ValueError as exc:
            raise ValueError(f"debug-sleep seconds must be a number, got {tail!r}") from exc
        return DebugSleepStall(seconds)
    if head == "slow-loop":
        return _parse_slow_loop(tail)
    raise ValueError(
        f"unknown stall spec {text!r}; expected one of: none, debug-sleep:<seconds>, "
        "slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s>"
    )


def _parse_slow_loop(tail: str) -> SlowLoopStall:
    """Parse the ``<kind>:<block_ms>:<period_ms>:<duration_s>`` tail of a slow-loop spec."""
    fields = tail.split(":")
    if len(fields) != 4:
        raise ValueError(
            "'slow-loop' requires: slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s> "
            "(kind is 'debug-sleep' or 'lua')"
        )
    block_kind, block_raw, period_raw, duration_raw = fields
    try:
        block_ms = float(block_raw)
        period_ms = float(period_raw)
        duration_s = float(duration_raw)
    except ValueError as exc:
        raise ValueError(
            f"slow-loop block_ms/period_ms/duration_s must be numbers, got {block_raw!r}, "
            f"{period_raw!r}, {duration_raw!r}"
        ) from exc
    return SlowLoopStall(block_kind, block_ms, period_ms, duration_s)
