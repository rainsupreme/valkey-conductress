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
from tests.unit.asyncio_server_support import close_server


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
        await close_server(self._server, self._reply_gate)

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
            prewarm_connections=0,  # measuring the storm itself, not the cold-start prewarm
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
            prewarm_connections=0,
        )
        result = await run_storm(config)
        totals = metrics.totals(result.events, config.clients)
        assert totals["connected_clients"] == 20
        # No stall -> every client connects on the first attempt.
        assert totals["amplification"] == pytest.approx(1.0)
        # Defect 1 regression guard: a connected attempt resolves at reply
        # receipt, NOT at the end of the ~2s idle hold. The old bug stamped
        # t_end after the hold, making every connected duration ~= duration_s.
        connected = [e for e in result.events if e["outcome"] == "connected"]
        assert connected
        for e in connected:
            assert (e["t_end"] - e["t_start"]) < 0.5, "connected attempt latency includes the hold-open time"
        # Quiescence is therefore ~= the burst, not ~= duration_s.
        doc = build_document_for(result)
        assert doc["metrics"]["time_to_quiescence"]["from_start"] < 1.0
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
        ["--workers", "-1"],
        ["--prewarm-connections", "-1"],
        ["--burst-after-stall-ms", "-1"],
        ["--policy", "bogus"],
        ["--stall", "bogus"],
    ],
)
def test_cli_rejects_invalid_config(argv):
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError):
        config_from_args(args)


def test_cli_workers_zero_is_auto():
    args = build_parser().parse_args(["--workers", "0"])
    config = config_from_args(args)
    assert config.resolved_workers() >= 1  # 0 -> min(8, cpu_count)


def test_cli_stall_first_is_default():
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert config.burst_first is False
    assert config.burst_after_stall_ms == 200


def test_cli_prewarm_default_and_explicit():
    assert config_from_args(build_parser().parse_args([])).prewarm_connections is None
    assert config_from_args(build_parser().parse_args(["--prewarm-connections", "0"])).prewarm_connections == 0


# --------------------------------------------------------------------------- client t_end (defect 1)


@pytest.mark.asyncio
async def test_connected_attempt_t_end_is_reply_receipt_not_hold_end():
    """A connected attempt resolves at reply receipt; the idle hold is separate.

    Regression guard for the bug that stamped t_end after the hold-open ended,
    which made a connected attempt's duration equal the whole run. Here the
    deadline (hold) is ~1.0s while the reply arrives in milliseconds, so the
    recorded duration must be far shorter than the hold.
    """
    from conductress.stormgen.client import ClientConfig, StormClient
    from conductress.stormgen.policy import parse_policy

    server = FakeServer()
    await server.start()
    try:
        origin = asyncio.get_event_loop().time()

        def clock():
            return asyncio.get_event_loop().time() - origin

        hold_deadline = clock() + 1.0
        cfg = ClientConfig(
            host=server.host,
            port=server.port,
            connect_timeout=0.5,
            reply_timeout=0.5,
            handshake=["HELLO 3"],
            first_command="GET stormkey",
        )
        events: list = []
        client = StormClient(0, cfg, parse_policy("fixed:20"), clock, hold_deadline)
        await client.run(events)

        assert len(events) == 1
        e = events[0]
        assert e["outcome"] == "connected"
        duration = e["t_end"] - e["t_start"]
        assert duration < 0.3, f"connected duration {duration:.3f}s includes the ~1s hold"
        # And t_end lands well before the hold deadline it was NOT supposed to include.
        assert e["t_end"] < 0.9
    finally:
        await server.stop()


# --------------------------------------------------------------------------- ordering (defect 2)


@pytest.mark.asyncio
async def test_stall_first_ordering_starts_burst_after_stall_is_issued(monkeypatch):
    """Default (stall-first): the stall command is issued before the burst begins.

    A recording injector captures the monotonic time it issued the stall; every
    client's first attempt must start at or after that moment, proving the burst
    was not already finished when the stall fired (the old default-1.0s bug).
    """
    import time as _time

    from conductress.stormgen import runner as runner_mod
    from conductress.stormgen import stall as stall_mod

    issued_at = {}

    class RecordingStall(stall_mod.StallInjector):
        kind = "debug-sleep"

        async def run(self, host, port, connect_timeout, on_issued=None):
            rec = stall_mod.StallRecord(kind=self.kind)
            rec.started = _time.time()
            issued_at["mono"] = _time.monotonic()
            if on_issued is not None:
                on_issued()
            await asyncio.sleep(0.05)  # brief "stall"
            rec.ended = _time.time()
            return rec

        def describe(self):
            return "debug-sleep:1"

    monkeypatch.setattr(runner_mod, "parse_stall", lambda spec: RecordingStall())

    # The startup connect-capacity check (200 sequential connects, ~100 ms on
    # loopback) runs before the storm's origin is set, which would inflate the
    # test's external run_origin reference relative to the events' origin. It is
    # orthogonal to ordering, so stub it out to keep the timing invariant clean.
    async def _no_capacity(config, connects=200):
        return 0.0

    monkeypatch.setattr(runner_mod, "measure_connect_capacity", _no_capacity)

    server = FakeServer()
    await server.start()
    try:
        config = StormConfig(
            host=server.host,
            port=server.port,
            clients=10,
            burst_ms=20,
            duration_s=1.5,
            connect_timeout_ms=300.0,
            reply_timeout_ms=300.0,
            policy_spec="fixed:20",
            stall_spec="debug-sleep:1",  # RecordingStall ignores the seconds
            workers=1,
            prewarm_connections=0,
            burst_after_stall_ms=100,
        )
        run_origin = _time.monotonic()
        result = await run_storm(config)

        assert "mono" in issued_at, "stall was never issued"
        first_starts = [e["t_start"] for e in result.events if e["attempt"] == 0]
        assert first_starts
        # Events are stamped relative to the run origin; the stall issued at
        # (issued_at["mono"] - run_origin) relative seconds. Every first attempt
        # must begin at/after the stall issue + burst_after_stall_ms.
        stall_issue_rel = issued_at["mono"] - run_origin
        assert min(first_starts) >= stall_issue_rel + 0.1 - 0.03  # small scheduling slack
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_burst_first_ordering_starts_burst_at_origin(monkeypatch):
    """burst_first restores the legacy ordering: the burst starts at the origin."""
    import time as _time

    from conductress.stormgen import runner as runner_mod
    from conductress.stormgen import stall as stall_mod

    class LateStall(stall_mod.StallInjector):
        kind = "debug-sleep"

        async def run(self, host, port, connect_timeout, on_issued=None):
            rec = stall_mod.StallRecord(kind=self.kind, started=_time.time())
            if on_issued is not None:
                on_issued()
            rec.ended = _time.time()
            return rec

        def describe(self):
            return "debug-sleep:1"

    monkeypatch.setattr(runner_mod, "parse_stall", lambda spec: LateStall())

    server = FakeServer()
    await server.start()
    try:
        config = StormConfig(
            host=server.host,
            port=server.port,
            clients=10,
            burst_ms=20,
            duration_s=1.0,
            connect_timeout_ms=300.0,
            reply_timeout_ms=300.0,
            policy_spec="fixed:20",
            stall_spec="debug-sleep:1",
            workers=1,
            prewarm_connections=0,
            burst_first=True,
            stall_after_s=0.5,
        )
        result = await run_storm(config)
        first_starts = [e["t_start"] for e in result.events if e["attempt"] == 0]
        # burst_first: first attempts start near the origin (< the 0.5s stall_after).
        assert min(first_starts) < 0.3
    finally:
        await server.stop()
