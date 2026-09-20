"""Unit tests for the slow-loop stall injector.

Covers parse/validation of ``slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s>``,
the loop's timing and duty-cycle bookkeeping against an in-process asyncio fake
server, the lua block's self-calibration converging on the target block length,
and the achieved-block fields serializing into the stormgen document.

The fake server sleeps ``block_ms`` on ``DEBUG SLEEP`` and answers ``EVAL``
after a delay proportional to the iteration count -- no real Redis/Valkey.
"""

import asyncio

import pytest

from conductress.stormgen import SCHEMA_VERSION
from conductress.stormgen.resp import ReplyParser
from conductress.stormgen.stall import SlowLoopStall, StallRecord, parse_stall

# --------------------------------------------------------------------------- parse / validate


def test_parse_slow_loop_debug_sleep():
    injector = parse_stall("slow-loop:debug-sleep:50:100:6")
    assert isinstance(injector, SlowLoopStall)
    assert injector.block_kind == "debug-sleep"
    assert injector.block_ms == 50 and injector.period_ms == 100 and injector.duration_s == 6
    assert injector.describe() == "slow-loop:debug-sleep:50:100:6"
    assert injector.injects is True


def test_parse_slow_loop_lua():
    injector = parse_stall("slow-loop:lua:50:100:6")
    assert isinstance(injector, SlowLoopStall)
    assert injector.block_kind == "lua"
    assert injector.describe() == "slow-loop:lua:50:100:6"


def test_parse_slow_loop_fractional_block():
    injector = parse_stall("slow-loop:debug-sleep:12.5:25:3")
    assert injector.block_ms == pytest.approx(12.5)
    assert injector.describe() == "slow-loop:debug-sleep:12.5:25:3"


@pytest.mark.parametrize(
    "spec",
    [
        "slow-loop",  # no tail
        "slow-loop:debug-sleep:50:100",  # too few fields
        "slow-loop:debug-sleep:50:100:6:7",  # too many fields
        "slow-loop:bogus:50:100:6",  # bad kind
        "slow-loop:debug-sleep:150:100:6",  # block > period
        "slow-loop:debug-sleep:0:100:6",  # block == 0
        "slow-loop:debug-sleep:-5:100:6",  # negative block
        "slow-loop:debug-sleep:50:100:0",  # zero duration
        "slow-loop:lua:1500:2000:6",  # lua block_ms > 1000
        "slow-loop:debug-sleep:x:100:6",  # non-numeric
    ],
)
def test_parse_slow_loop_rejects_bad(spec):
    with pytest.raises(ValueError):
        parse_stall(spec)


def test_lua_allows_block_at_the_1000ms_ceiling():
    # 1000 is allowed; only strictly above is rejected.
    assert parse_stall("slow-loop:lua:1000:2000:6").block_ms == 1000


# --------------------------------------------------------------------------- fake server


class SlowLoopFakeServer:
    """A RESP server that models block cost for DEBUG SLEEP and EVAL.

    ``DEBUG SLEEP <s>`` replies ``+OK`` after ``s`` seconds. ``EVAL <script> 0
    <iters>`` replies ``:0`` after ``iters * per_iter_s`` seconds, so a caller
    can calibrate against a known per-iteration cost. HELLO is answered at once.
    """

    def __init__(self, per_iter_s: float = 1e-7):
        self._server = None
        self.host = "127.0.0.1"
        self.port = 0
        self.per_iter_s = per_iter_s

    async def start(self):
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader, writer):
        parser = ReplyParser()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                parser.feed(chunk)
                while True:
                    complete, value = parser.try_parse()
                    if not complete:
                        break
                    await self._reply(writer, value)
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def _reply(self, writer, command):
        if not isinstance(command, list) or not command:
            return
        verb = str(command[0]).upper()
        if verb == "DEBUG" and len(command) >= 3 and str(command[1]).upper() == "SLEEP":
            await asyncio.sleep(float(command[2]))
            writer.write(b"+OK\r\n")
        elif verb == "EVAL":
            iters = int(command[-1])
            await asyncio.sleep(iters * self.per_iter_s)
            writer.write(b":0\r\n")
        else:  # HELLO or anything else
            writer.write(b"$3\r\nval\r\n")
        await writer.drain()


# --------------------------------------------------------------------------- loop timing


@pytest.mark.asyncio
async def test_slow_loop_debug_sleep_timing_and_duty():
    server = SlowLoopFakeServer()
    await server.start()
    try:
        issued = {"count": 0}

        def on_issued():
            issued["count"] += 1

        injector = SlowLoopStall("debug-sleep", block_ms=50, period_ms=100, duration_s=1.0)
        record = await injector.run(server.host, server.port, connect_timeout=2.0, on_issued=on_issued)

        # duration/period = 1000/100 = ~10 blocks, allow +/-1.
        assert abs(record.blocks_issued - 10) <= 1
        # on_issued fired exactly once.
        assert issued["count"] == 1
        # effective_duty ~= block/period = 0.5, within 15%.
        assert record.effective_duty == pytest.approx(0.5, abs=0.075)
        # The degraded window spans ~= duration.
        assert record.started is not None and record.ended is not None
        assert (record.ended - record.started) == pytest.approx(1.0, abs=0.2)
        # Achieved block ~= 50 ms.
        assert record.achieved_block_ms_p50 == pytest.approx(50.0, abs=15.0)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_slow_loop_on_issued_fires_once_before_second_block(monkeypatch):
    """on_issued is called exactly once, at the first block, before the second."""
    server = SlowLoopFakeServer()
    await server.start()
    try:
        marks = []
        injector = SlowLoopStall("debug-sleep", block_ms=30, period_ms=100, duration_s=0.6)

        orig_one_block = injector._one_block
        block_count = {"n": 0}
        fired = {"yes": False}

        async def counting_block(*args, **kwargs):
            block_count["n"] += 1
            marks.append(("block", block_count["n"], fired["yes"]))
            return await orig_one_block(*args, **kwargs)

        def on_issued():
            fired["yes"] = True
            marks.append(("issued", block_count["n"]))

        monkeypatch.setattr(injector, "_one_block", counting_block)
        await injector.run(server.host, server.port, connect_timeout=2.0, on_issued=on_issued)

        # on_issued fired after block 1 was issued and before block 2.
        issued_marks = [m for m in marks if m[0] == "issued"]
        assert len(issued_marks) == 1
        assert issued_marks[0][1] == 1
        # At the second block, on_issued had already fired.
        second_block = [m for m in marks if m[0] == "block" and m[1] == 2]
        assert second_block and second_block[0][2] is True
    finally:
        await server.stop()


# --------------------------------------------------------------------------- lua calibration


@pytest.mark.asyncio
async def test_slow_loop_lua_calibration_converges():
    # per_iter_s known: 5e-7 s/iter. target block 50 ms => ~100k iters.
    server = SlowLoopFakeServer(per_iter_s=5e-7)
    await server.start()
    try:
        injector = SlowLoopStall("lua", block_ms=50, period_ms=100, duration_s=1.0)
        record = await injector.run(server.host, server.port, connect_timeout=2.0)
        # After the two calibration rounds, the measured block p50 lands within
        # 20% of the 50 ms target.
        assert record.achieved_block_ms_p50 == pytest.approx(50.0, rel=0.2)
        assert record.blocks_issued >= 1
    finally:
        await server.stop()


# --------------------------------------------------------------------------- clamping


def test_slow_loop_clamp_to_storm_shrinks_long_loop():
    injector = SlowLoopStall("debug-sleep", block_ms=50, period_ms=100, duration_s=30.0)
    injector.clamp_to_storm(storm_duration_s=10.0, burst_after_stall_s=0.2)
    # Clamped to storm_duration - 1.
    assert injector.duration_s == pytest.approx(9.0)


def test_slow_loop_clamp_to_storm_leaves_short_loop_alone():
    injector = SlowLoopStall("debug-sleep", block_ms=50, period_ms=100, duration_s=5.0)
    injector.clamp_to_storm(storm_duration_s=20.0, burst_after_stall_s=0.2)
    assert injector.duration_s == pytest.approx(5.0)


# --------------------------------------------------------------------------- serialization


@pytest.mark.asyncio
async def test_slow_loop_fields_serialize_in_document():
    from conductress.stormgen.__main__ import build_document
    from conductress.stormgen.runner import StormConfig, run_storm

    server = SlowLoopFakeServer()
    await server.start()
    try:
        config = StormConfig(
            host=server.host,
            port=server.port,
            clients=4,
            burst_ms=20,
            duration_s=2.0,
            connect_timeout_ms=500.0,
            reply_timeout_ms=500.0,
            policy_spec="fixed:20",
            stall_spec="slow-loop:debug-sleep:30:100:1",
            workers=1,
            prewarm_connections=0,
            burst_after_stall_ms=50,
        )
        result = await run_storm(config)
        doc = build_document(result)
        stall = doc["stall"]
        assert stall["kind"] == "slow-loop"
        assert stall["blocks_issued"] >= 1
        assert stall["achieved_block_ms_p50"] is not None
        assert stall["achieved_block_ms_max"] is not None
        assert stall["effective_duty"] is not None
        # The slow-loop stall fields are present from schema v3 onward; v4 adds
        # TLS/herd/probe fields but does not remove them, so assert >= 3 and
        # that the document reports the current schema version.
        assert doc["schema_version"] == SCHEMA_VERSION
        assert SCHEMA_VERSION >= 3
    finally:
        await server.stop()


def test_document_schema_version_includes_slow_loop():
    # The slow-loop achieved-block fields landed at schema v3; the schema only
    # grows (v4 added TLS/herd/probe), so the version is at least 3.
    assert SCHEMA_VERSION >= 3


def test_stall_record_defaults_are_hard_stall_shaped():
    # A hard stall / no-stall StallRecord keeps the slow-loop fields at defaults.
    rec = StallRecord(kind="debug-sleep")
    assert rec.blocks_issued == 0
    assert rec.achieved_block_ms_p50 is None
    assert rec.achieved_block_ms_max is None
    assert rec.effective_duty is None
