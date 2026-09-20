"""TLS herd and active-herd behaviour in ``conductress.stormgen``.

TLS herd: a client wrapped in an SSLContext handshakes with an in-test asyncio
TLS server (self-signed cert generated with openssl; skipped if openssl is
absent); a server that stalls the handshake yields ``connect_timeout`` (no
separate handshake-timeout outcome, matching a client library's connect
timeout).

Active herd: with ``herd_command_interval_ms > 0`` a connected client re-sends
its first command on a cadence and reconnects on a stalled reply with outcome
``reply_timeout_steady``; interval 0 keeps today's idle-hold events unchanged.
"""

import asyncio
import shutil
import ssl
import subprocess
import time

import pytest

from conductress.stormgen.client import ClientConfig, StormClient, run_clients
from conductress.stormgen.policy import parse_policy
from conductress.stormgen.resp import ReplyParser
from tests.unit.asyncio_server_support import close_server

_OPENSSL = shutil.which("openssl")
requires_openssl = pytest.mark.skipif(_OPENSSL is None, reason="openssl not installed")


def _gen_self_signed(tmp_path):
    """Generate a self-signed cert+key for 127.0.0.1 into tmp_path; return (cert, key)."""
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    subprocess.run(
        [
            _OPENSSL,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


class _TLSFakeServer:
    """An asyncio TLS RESP server that answers HELLO/GET, or stalls the handshake."""

    def __init__(self, cert, key, stall_handshake=False):
        self._server = None
        self.host = "127.0.0.1"
        self.port = 0
        self._ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl.load_cert_chain(str(cert), str(key))
        self._stall_handshake = stall_handshake

    async def start(self):
        # A server that stalls the handshake: accept the raw TCP connection but
        # never complete TLS. Simulated by a plaintext listener that reads and
        # never writes, so the client's handshake wait_for times out.
        if self._stall_handshake:
            self._server = await asyncio.start_server(self._black_hole, self.host, 0)
        else:
            self._server = await asyncio.start_server(self._handle, self.host, 0, ssl=self._ssl)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            try:
                await close_server(self._server)
            except (ssl.SSLError, OSError):
                pass

    async def _black_hole(self, reader, writer):
        try:
            await asyncio.sleep(30)  # never progresses the (TLS) handshake
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()

    async def _handle(self, reader, writer):
        parser = ReplyParser()
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                parser.feed(chunk)
                if b"HELLO" in chunk:
                    writer.write(b"%1\r\n$6\r\nserver\r\n$6\r\nvalkey\r\n")
                else:
                    writer.write(b"$3\r\nval\r\n")
                await writer.drain()
        except (ConnectionError, OSError, ssl.SSLError):
            pass
        finally:
            writer.close()


def _client_ssl_context(cert):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(cert))
    ctx.check_hostname = False
    return ctx


@requires_openssl
@pytest.mark.asyncio
async def test_tls_herd_handshake_succeeds(tmp_path):
    """A TLS client connects, handshakes, and gets a 'connected' event."""
    cert, key = _gen_self_signed(tmp_path)
    server = _TLSFakeServer(cert, key)
    await server.start()
    try:
        origin = time.monotonic()

        def clock():
            return time.monotonic() - origin

        config = ClientConfig(
            host=server.host,
            port=server.port,
            connect_timeout=2.0,
            reply_timeout=2.0,
            ssl_context=_client_ssl_context(cert),
        )
        events, _ = await run_clients([0], config, parse_policy("fixed:50"), clock, [0.0], deadline=1.0)
        assert any(e["outcome"] == "connected" for e in events)
    finally:
        await server.stop()


@requires_openssl
@pytest.mark.asyncio
async def test_tls_herd_stalled_handshake_is_connect_timeout(tmp_path):
    """A server that never finishes the TLS handshake yields connect_timeout."""
    cert, key = _gen_self_signed(tmp_path)
    server = _TLSFakeServer(cert, key, stall_handshake=True)
    await server.start()
    try:
        origin = time.monotonic()

        def clock():
            return time.monotonic() - origin

        config = ClientConfig(
            host=server.host,
            port=server.port,
            connect_timeout=0.3,  # short: the handshake never completes
            reply_timeout=0.3,
            ssl_context=_client_ssl_context(cert),
        )
        events, _ = await run_clients([0], config, parse_policy("fixed:50"), clock, [0.0], deadline=0.8)
        # The unfinished handshake is a connect_timeout, not a new outcome.
        assert any(e["outcome"] == "connect_timeout" for e in events)
        assert all(e["outcome"] != "tls_handshake_timeout" for e in events)
    finally:
        await server.stop()


class _ActiveHerdServer:
    """Plaintext RESP server: answers the first command, then can stall replies."""

    def __init__(self):
        self._server = None
        self.host = "127.0.0.1"
        self.port = 0
        self._gate = asyncio.Event()
        self._gate.set()
        self._replies = 0

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        await close_server(self._server, self._gate)

    def stall(self):
        self._gate.clear()

    async def _handle(self, reader, writer):
        parser = ReplyParser()
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                parser.feed(chunk)
                if b"HELLO" in chunk:
                    writer.write(b"%1\r\n$6\r\nserver\r\n$6\r\nvalkey\r\n")
                    await writer.drain()
                    continue
                await self._gate.wait()  # a stall holds the steady-command reply
                writer.write(b"$3\r\nval\r\n")
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()


@pytest.mark.asyncio
async def test_active_herd_reconnects_on_stalled_steady_command():
    """A stalled steady command yields reply_timeout_steady and a reconnect."""
    server = _ActiveHerdServer()
    await server.start()
    try:
        origin = time.monotonic()

        def clock():
            return time.monotonic() - origin

        async def stall_after_connect():
            await asyncio.sleep(0.25)  # let the first steady command through
            server.stall()

        config = ClientConfig(
            host=server.host,
            port=server.port,
            connect_timeout=1.0,
            reply_timeout=0.2,
            herd_command_interval_ms=50.0,  # active herd
        )
        stall_task = asyncio.ensure_future(stall_after_connect())
        events, herd_ok = await run_clients([0], config, parse_policy("fixed:50"), clock, [0.0], deadline=1.2)
        await stall_task
        assert any(e["outcome"] == "reply_timeout_steady" for e in events)
        assert herd_ok >= 1  # at least one steady reply landed before the stall
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_idle_hold_records_no_steady_events_and_zero_herd_commands():
    """interval 0 is the idle hold: one connected event, no steady commands."""
    server = _ActiveHerdServer()
    await server.start()
    try:
        origin = time.monotonic()

        def clock():
            return time.monotonic() - origin

        config = ClientConfig(
            host=server.host,
            port=server.port,
            connect_timeout=1.0,
            reply_timeout=1.0,
            herd_command_interval_ms=0.0,  # idle hold (today's behaviour)
        )
        events, herd_ok = await run_clients([0], config, parse_policy("fixed:50"), clock, [0.0], deadline=0.4)
        assert herd_ok == 0
        assert all(e["outcome"] != "reply_timeout_steady" for e in events)
        assert sum(1 for e in events if e["outcome"] == "connected") == 1
    finally:
        await server.stop()
