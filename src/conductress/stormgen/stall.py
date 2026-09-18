"""Server-side stall injectors.

A stall injector runs on its own dedicated connection and makes the server's
main thread unavailable for a bounded interval, so the storm clients see their
first command block. The mechanism is what a connection storm forms around:
while the main thread is busy, freshly accepted connections cannot be served
and pile up in the kernel accept backlog.

Injectors are parsed from a compact spec string:

* ``none``                  -- no stall (baseline / control run)
* ``debug-sleep:<seconds>`` -- ``DEBUG SLEEP <seconds>`` on a dedicated
                               connection; blocks the main thread for the
                               duration. Portable across Redis and Valkey.

The interface (:meth:`StallInjector.run`) is intentionally small so future
injectors -- a busy Lua script, a large ``KEYS`` scan -- are one class plus one
line in :func:`parse_stall`. Each run records wall-clock start and end so the
recovery window can be measured from the moment the stall cleared.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .resp import ReplyParser, encode_command


@dataclass
class StallRecord:
    """When a stall started and ended (wall-clock, ``time.time()``)."""

    kind: str
    started: Optional[float] = None
    ended: Optional[float] = None


class StallInjector(ABC):
    """Runs a server-side stall on a dedicated connection and records its span."""

    kind: str = "none"

    @abstractmethod
    async def run(self, host: str, port: int, connect_timeout: float) -> StallRecord:
        """Perform the stall (opening its own connection) and return its record."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Return the canonical spec string."""
        raise NotImplementedError

    @property
    def injects(self) -> bool:
        """True when this injector actually stalls the server."""
        return True


class NoStall(StallInjector):
    """Baseline: never stalls the server."""

    kind = "none"

    async def run(self, host: str, port: int, connect_timeout: float) -> StallRecord:
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

    async def run(self, host: str, port: int, connect_timeout: float) -> StallRecord:
        record = StallRecord(kind=self.kind)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=connect_timeout)
        try:
            writer.write(encode_command("DEBUG", "SLEEP", _fmt_seconds(self.seconds)))
            await writer.drain()
            record.started = time.time()
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
    raise ValueError(f"unknown stall spec {text!r}; expected one of: none, debug-sleep:<seconds>")
