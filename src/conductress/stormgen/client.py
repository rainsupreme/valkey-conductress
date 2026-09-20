"""One storm client's asyncio connect/command/retry state machine.

A client repeatedly tries to establish a working connection:

1. open a TCP connection (bounded by ``connect_timeout``);
2. send the handshake commands (default ``["HELLO 3"]``; may be empty) and
   read each reply;
3. send the first command (default ``GET <key>``) and await its reply
   (bounded by ``reply_timeout``);
4. on success: hold the connection open and idle until the run's deadline,
   then close -- this models a real client that connects once and stays;
5. on any failure (timeout, refused, reset, protocol/other error): close,
   wait ``policy.next_delay(attempt)``, and retry from step 1.

Every attempt appends one event ``{client_id, attempt, t_start, t_end,
outcome}`` with an outcome from :data:`conductress.stormgen.metrics.OUTCOMES`.
Times use a shared monotonic ``clock`` whose origin is the storm start, so all
clients' events share one timeline.

``bind_addrs`` lets a client bind its socket to one of several loopback source
addresses (e.g. 127.0.0.2..127.0.0.250). Spreading source IPs multiplies the
available ephemeral-port space: a single (src_ip, dst_ip:port) tuple space is
capped near 28k ports, which a large storm can exhaust before it reaches the
server -- turning a server test into a client-side port-exhaustion test.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .policy import ReconnectPolicy
from .resp import ReplyParser, encode_command


@dataclass
class ClientConfig:
    """Everything a client needs that is identical across the population."""

    host: str
    port: int
    connect_timeout: float
    reply_timeout: float
    handshake: List[str] = field(default_factory=lambda: ["HELLO 3"])
    first_command: str = "GET stormkey"
    bind_addrs: Optional[List[str]] = None
    # TLS: an SSLContext wraps the connection when set (verify against the test
    # CA, check_hostname off for loopback aliases). None keeps plain TCP.
    ssl_context: Optional[ssl.SSLContext] = None
    # Active herd: when > 0 a connected client sends first_command every
    # herd_command_interval_ms and applies reply_timeout; a timeout closes the
    # connection and re-enters the reconnect policy (outcome
    # reply_timeout_steady). 0 (the default) is today's idle hold.
    herd_command_interval_ms: float = 0.0


@dataclass
class AttemptResult:
    """The outcome of one connect+command attempt.

    ``resolved_at`` is the clock time the attempt resolved -- reply receipt for
    a success, or the failure moment otherwise -- and is what the event's
    ``t_end`` records. For a success, ``reader``/``writer`` carry the open
    connection so the caller can hold it after recording the event.
    """

    outcome: str
    resolved_at: float
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None


def _classify_error(exc: BaseException) -> str:
    """Map a connection exception to a storm outcome label."""
    if isinstance(exc, asyncio.TimeoutError):
        return "connect_timeout"
    if isinstance(exc, ConnectionRefusedError):
        return "refused"
    if isinstance(exc, ConnectionResetError):
        return "reset"
    if isinstance(exc, (ConnectionError, OSError)):
        # A RST during connect surfaces as a plain OSError on some platforms.
        return "reset"
    return "error"


class StormClient:
    """Drives one connection: connect, command, hold-open, or retry on failure."""

    def __init__(
        self,
        client_id: int,
        config: ClientConfig,
        policy: ReconnectPolicy,
        clock: Callable[[], float],
        deadline: float,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.policy = policy
        self._clock = clock
        self._deadline = deadline
        # Count of successful steady (active-herd) command replies on this
        # client; 0 for the idle hold. Aggregated by run_clients.
        self.herd_commands_ok = 0

    def _bind_local_addr(self):
        """Pick this client's source address (round-robin across ``bind_addrs``)."""
        addrs = self.config.bind_addrs
        if not addrs:
            return None
        return (addrs[self.client_id % len(addrs)], 0)

    async def _handshake_and_command(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Send handshake commands then the first command, awaiting each reply."""
        for command in self.config.handshake:
            await self._send_and_read(reader, writer, command)
        await self._send_and_read(reader, writer, self.config.first_command)

    async def _send_and_read(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, command: str) -> None:
        """Write one command and read exactly one full reply, bounded by reply_timeout."""
        writer.write(encode_command(*command.split()))
        await writer.drain()
        parser = ReplyParser()
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=self.config.reply_timeout)
            if not chunk:
                raise ConnectionResetError("connection closed before a reply arrived")
            parser.feed(chunk)
            complete, _ = parser.try_parse()
            if complete:
                return

    async def _one_attempt(self) -> "AttemptResult":
        """Run a single connect+command attempt.

        Returns an :class:`AttemptResult` whose ``resolved_at`` is the clock
        time the attempt *resolved* -- for a success, the moment the first
        reply arrived, NOT the end of the idle hold. The hold is a separate
        phase that keeps a connected client alive; it is not part of the
        attempt's latency, and folding it in would corrupt attempt latency,
        time-to-quiescence and recovery. On success the open reader/writer are
        returned so the caller can hold the connection after the event is
        recorded.
        """
        writer: Optional[asyncio.StreamWriter] = None
        try:
            local_addr = self._bind_local_addr()
            # A TLS handshake is part of open_connection, so the single
            # connect_timeout below covers TCP connect AND the TLS handshake
            # together -- which is what a client library's connect timeout
            # covers. check_hostname is off on the context (loopback aliases),
            # so a server_hostname is only needed to drive SNI; pass the host.
            if self.config.ssl_context is not None:
                connect = asyncio.open_connection(
                    self.config.host,
                    self.config.port,
                    local_addr=local_addr,
                    ssl=self.config.ssl_context,
                    server_hostname=self.config.host,
                )
            else:
                connect = asyncio.open_connection(self.config.host, self.config.port, local_addr=local_addr)
            reader, opened_writer = await asyncio.wait_for(connect, timeout=self.config.connect_timeout)
            writer = opened_writer
            try:
                await self._handshake_and_command(reader, opened_writer)
            except asyncio.TimeoutError:
                # Resolved (as a failure) now; close in finally.
                return AttemptResult("reply_timeout", self._clock(), None, None)
            # Resolved as a success the instant the first reply arrived.
            resolved_at = self._clock()
            held_reader, held_writer = reader, opened_writer
            writer = None  # ownership passes to the caller's hold phase
            return AttemptResult("connected", resolved_at, held_reader, held_writer)
        except asyncio.TimeoutError:
            return AttemptResult("connect_timeout", self._clock(), None, None)
        except (ConnectionError, OSError, socket.gaierror) as exc:
            return AttemptResult(_classify_error(exc), self._clock(), None, None)
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, asyncio.TimeoutError):
                    pass

    async def _hold_open(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Optional[float]:
        """Hold an established connection until the deadline.

        Idle by default: wait for the deadline (or a server-initiated close),
        then close. When ``herd_command_interval_ms > 0`` this instead sends
        ``first_command`` every interval and reads each reply under
        ``reply_timeout``: each reply increments ``self.herd_commands_ok``; a
        reply timeout (or a closed connection) ends the hold and returns the
        clock time it happened so the caller records a ``reply_timeout_steady``
        event and reconnects. Returns ``None`` when the hold ended normally
        (deadline reached or a clean server close), a float when a steady
        command timed out (the connection was kicked out).
        """
        interval = self.config.herd_command_interval_ms / 1000.0
        try:
            if interval <= 0:
                remaining = self._deadline - self._clock()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(reader.read(1), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    except (ConnectionError, OSError):
                        pass
                return None
            # Active herd: keep issuing the first command on a fixed cadence.
            while self._clock() < self._deadline:
                await asyncio.sleep(min(interval, max(0.0, self._deadline - self._clock())))
                if self._clock() >= self._deadline:
                    break
                try:
                    await self._send_and_read(reader, writer, self.config.first_command)
                    self.herd_commands_ok += 1
                except asyncio.TimeoutError:
                    return self._clock()  # kicked out: a steady command timed out
                except (ConnectionError, OSError):
                    return self._clock()
            return None
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass

    async def run(self, events: List[dict]) -> None:
        """Attempt until connected or the deadline passes; append one event per attempt.

        A connected attempt's event is recorded at reply receipt, then the
        connection is held open until the deadline as a separate phase. With an
        active herd the hold can be kicked out (a steady command times out): a
        ``reply_timeout_steady`` event is recorded and the client re-enters the
        reconnect policy, so a stall at t=T can make an already-connected fleet
        reconnect (post-trigger hysteresis).
        """
        attempt = 0
        while self._clock() < self._deadline:
            t_start = self._clock()
            result = await self._one_attempt()
            events.append(
                {
                    "client_id": self.client_id,
                    "attempt": attempt,
                    "t_start": t_start,
                    "t_end": result.resolved_at,
                    "outcome": result.outcome,
                }
            )
            attempt += 1
            if result.outcome == "connected":
                assert result.reader is not None and result.writer is not None
                hold_start = self._clock()
                kicked_at = await self._hold_open(result.reader, result.writer)
                if kicked_at is None:
                    return  # held to the deadline (or a clean close); done
                # Kicked out: record the steady timeout as its own attempt, then
                # back off and reconnect like any other failure.
                events.append(
                    {
                        "client_id": self.client_id,
                        "attempt": attempt,
                        "t_start": hold_start,
                        "t_end": kicked_at,
                        "outcome": "reply_timeout_steady",
                    }
                )
                attempt += 1
            delay = self.policy.next_delay(attempt - 1)
            # Do not sleep past the deadline; a partial sleep is fine.
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                return
            if delay > 0:
                await asyncio.sleep(min(delay, remaining))


async def run_clients(
    client_ids: List[int],
    config: ClientConfig,
    policy: ReconnectPolicy,
    clock: Callable[[], float],
    start_offsets: List[float],
    deadline: float,
) -> "tuple[List[dict], int]":
    """Run a slice of clients concurrently; return (events, herd_commands_ok).

    ``start_offsets[i]`` is the clock time at which ``client_ids[i]`` should
    begin its first attempt, so the caller can spread connects over a burst
    window instead of firing them all at once. ``herd_commands_ok`` sums each
    client's successful steady (active-herd) command replies; it is 0 for the
    idle hold.
    """
    events: List[dict] = []
    clients: List["StormClient"] = []

    async def _staggered(client_id: int, start_at: float) -> None:
        wait = start_at - clock()
        if wait > 0:
            await asyncio.sleep(wait)
        client = StormClient(client_id, config, policy, clock, deadline)
        clients.append(client)
        await client.run(events)

    await asyncio.gather(*(_staggered(cid, off) for cid, off in zip(client_ids, start_offsets)))
    herd_commands_ok = sum(c.herd_commands_ok for c in clients)
    return events, herd_commands_ok
