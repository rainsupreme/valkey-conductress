"""Storm runner behaviour against an in-process fake TCP server, and CLI wiring.

The fake server speaks just enough RESP to answer HELLO and GET, and can be
told to stop replying (main-thread stall) or stop accepting for a window. A
small storm runs in ``--workers 1`` mode so it is deterministic and finishes
in a few seconds; we assert amplification rises during the pause and the storm
reaches quiescence afterwards.
"""

import asyncio

import pytest

from conductress.stormgen import metrics
from conductress.stormgen.__main__ import build_document, build_parser, config_from_args
from conductress.stormgen.resp import ReplyParser, encode_command
from conductress.stormgen.runner import StormConfig, run_storm


class FakeServer:
    """A minimal RESP server that can pause replying for a window.

    ``pause_replies_for(seconds)`` makes every connection stop answering its
    first command for that long -- the client-visible effect of a stalled main
    thread -- so a storm sees reply timeouts and reconnects.
    """

    def __init__(self):
        self._server = None
        self.host = "127.0.0.1"
        self.port = 0
        self._reply_gate = asyncio.Event()
        self._reply_gate.set()

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def pause_replies(self):
        self._reply_gate.clear()

    def resume_replies(self):
        self._reply_gate.set()

    async def _handle(self, reader, writer):
        parser = ReplyParser()
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                # We do not need to fully parse client commands; each newline-
                # terminated RESP command array ends in \r\n. Reply once per
                # command array seen. A simple heuristic: reply to each inbound
                # burst after the reply gate is open.
                parser.feed(chunk)
                await self._reply_gate.wait()
                # Answer HELLO with a small map, everything else with a bulk.
                if b"HELLO" in chunk:
                    writer.write(b"%1\r\n$6\r\nserver\r\n$6\r\nvalkey\r\n")
                else:
                    writer.write(b"$3\r\nval\r\n")
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()


@pytest.mark.asyncio
async def test_storm_amplifies_during_reply_pause_then_quiesces():
    server = FakeServer()
    await server.start()
    try:

        async def pause_window():
            # Pause replies for the first ~0.6s so early clients time out and retry.
            server.pause_replies()
            await asyncio.sleep(0.6)
            server.resume_replies()

        config = StormConfig(
            host=server.host,
            port=server.port,
            clients=30,
            burst_ms=50,
            duration_s=3.0,
            connect_timeout_ms=200.0,
            reply_timeout_ms=150.0,
            policy_spec="fixed:20",
            stall_spec="none",  # the pause is driven by the test, not a DEBUG SLEEP
            workers=1,
        )
        pause_task = asyncio.ensure_future(pause_window())
        result = await run_storm(config)
        await pause_task

        doc = build_document_for(result)
        totals = doc["metrics"]["totals"]
        # Every client eventually connected and held open.
        assert totals["connected_clients"] == 30
        # The reply pause forced retries: more attempts than clients.
        assert totals["attempts"] > 30
        assert totals["amplification"] > 1.0
        # Quiescence reached (finite) since all clients connected.
        assert doc["metrics"]["time_to_quiescence"]["from_start"] is not None
        # At least one reply timeout was recorded during the pause.
        assert doc["metrics"]["outcomes"]["reply_timeout"] >= 1
    finally:
        await server.stop()


def build_document_for(result):
    """Reduce a StormResult with the same code path the CLI uses."""
    return build_document(result)


@pytest.mark.asyncio
async def test_storm_no_stall_baseline_has_amplification_near_one():
    server = FakeServer()
    await server.start()
    try:
        config = StormConfig(
            host=server.host,
            port=server.port,
            clients=20,
            burst_ms=50,
            duration_s=2.0,
            connect_timeout_ms=500.0,
            reply_timeout_ms=500.0,
            policy_spec="fixed:20",
            stall_spec="none",
            workers=1,
        )
        result = await run_storm(config)
        totals = metrics.totals(result.events, config.clients)
        assert totals["connected_clients"] == 20
        # No stall -> every client connects on the first attempt.
        assert totals["amplification"] == pytest.approx(1.0)
    finally:
        await server.stop()


# --------------------------------------------------------------------------- CLI


def test_cli_args_map_to_config():
    args = build_parser().parse_args(
        [
            "--host",
            "10.0.0.5",
            "--port",
            "7000",
            "--clients",
            "2000",
            "--burst-ms",
            "200",
            "--connect-timeout-ms",
            "300",
            "--reply-timeout-ms",
            "500",
            "--policy",
            "exp:100:800:jitter",
            "--handshake",
            "HELLO 3",
            "--handshake",
            "AUTH user pass",
            "--first-command",
            "GET foo",
            "--duration-s",
            "12",
            "--stall",
            "debug-sleep:2",
            "--workers",
            "4",
            "--bind-addr",
            "127.0.0.2",
            "--bind-addr",
            "127.0.0.3",
        ]
    )
    config = config_from_args(args)
    assert config.host == "10.0.0.5" and config.port == 7000
    assert config.clients == 2000 and config.burst_ms == 200
    assert config.connect_timeout_ms == 300 and config.reply_timeout_ms == 500
    assert config.policy().describe() == "exp:100:800:jitter"
    assert config.handshake == ["HELLO 3", "AUTH user pass"]
    assert config.first_command == "GET foo"
    assert config.duration_s == 12 and config.workers == 4
    assert config.stall().describe() == "debug-sleep:2"
    assert config.bind_addrs == ["127.0.0.2", "127.0.0.3"]


def test_cli_empty_handshake_disables_handshake():
    args = build_parser().parse_args(["--handshake", ""])
    config = config_from_args(args)
    assert config.handshake == []


def test_cli_default_handshake_is_hello3():
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert config.handshake == ["HELLO 3"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--clients", "0"],
        ["--duration-s", "0"],
        ["--workers", "0"],
        ["--policy", "bogus"],
        ["--stall", "bogus"],
    ],
)
def test_cli_rejects_invalid_config(argv):
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError):
        config_from_args(args)
