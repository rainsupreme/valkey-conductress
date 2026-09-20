"""Shared helper for the in-test asyncio servers used by the stormgen tests.

Python 3.12 changed ``asyncio.Server.wait_closed()``: it now waits until every
client connection has finished, where 3.9-3.11 returned as soon as the listening
socket was closed. A fake server whose handler is deliberately parked (a "stall"
gate that is never released, a black hole that never answers) therefore hangs
``stop()`` forever on 3.12, which is how a 1-second test became an unbounded CI
job. ``close_server`` releases the gates first so parked handlers can finish, then
bounds the wait; a handler that still does not exit is abandoned rather than
allowed to wedge the test.
"""

import asyncio
from typing import Optional


async def close_server(server: Optional[asyncio.AbstractServer], *gates: asyncio.Event, timeout: float = 2.0) -> None:
    """Close a test server: release ``gates``, stop listening, wait (bounded) for handlers."""
    if server is None:
        return
    for gate in gates:
        gate.set()
    server.close()
    try:
        await asyncio.wait_for(server.wait_closed(), timeout=timeout)
    except asyncio.TimeoutError:
        pass  # a parked handler; the test's assertions are already made
    except OSError:
        pass
