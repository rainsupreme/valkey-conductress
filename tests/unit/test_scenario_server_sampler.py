"""Tests for the connection-storm server-side sampler and wall-clock anchors.

Covers the INFO-blob field parser, ``sampler_gaps`` on a synthetic series, the
local-vs-remote netstat guard, the new ``server_sample_ms`` field
(round-trip with and without it, validation), the CLI default of 100 only for
connection-storm, ``storm.origin_wall`` folding + the bumped schema_version,
the shared-time-axis alignment math, and a faked-runner test asserting the
sampler starts before the overlay and stops after the background measurement,
with its rows landing in ``scenario_metrics``.
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from conductress import config
from conductress.file_protocol import BenchmarkResults, FileProtocol
from conductress.stormgen import SCHEMA_VERSION
from conductress.task_queue import BaseTaskData
from conductress.tasks import task_scenario as module
from conductress.tasks.task_scenario import (
    GAP_TICK_MULTIPLE,
    ScenarioTaskData,
    ServerSampler,
    parse_info_fields,
    sampler_gaps,
    storm_metrics_namespace,
)
from conductress.topology import TopologySpec

HOST = config.ServerInfo(ip="127.0.0.1", username="ec2-user")


def _scenario_task(**overrides) -> ScenarioTaskData:
    fields = dict(
        source=config.REPO_NAMES[0],
        specifier="unstable",
        make_args="",
        topology=TopologySpec.standalone(),
        note="",
        requirements={},
        scenario="connection-storm",
        val_size=512,
        io_threads=1,
        pipelining=1,
        duration=8,
        repetitions=1,
    )
    fields.update(overrides)
    return ScenarioTaskData(**fields)


# --------------------------------------------------------------------------- INFO parsing


def test_parse_info_fields_extracts_sampled_ints():
    blob = (
        "# Clients\r\nconnected_clients:42\r\nblocked_clients:0\r\n"
        "# Stats\r\ntotal_connections_received:100\r\nrejected_connections:3\r\n"
        "total_commands_processed:9999\r\ninstantaneous_ops_per_sec:1234\r\n"
    )
    fields = parse_info_fields(blob)
    assert fields == {
        "connected_clients": 42,
        "total_connections_received": 100,
        "rejected_connections": 3,
        "total_commands_processed": 9999,
    }


def test_parse_info_fields_ignores_non_integer_and_unknown():
    blob = "connected_clients:not_a_number\nmaxmemory_policy:noeviction\nuptime_in_seconds:5\n"
    assert parse_info_fields(blob) == {}


# --------------------------------------------------------------------------- sampler_gaps


def test_sampler_gaps_flags_a_two_second_gap():
    tick_ms = 100
    # Normal ticks (reply ~12ms after send) plus one 2s gap.
    rows = [
        {"t_sent": 0.00, "t_reply": 0.012},
        {"t_sent": 0.10, "t_reply": 0.112},
        {"t_sent": 0.20, "t_reply": 2.200},  # main thread stalled: 2s late
        {"t_sent": 2.30, "t_reply": 2.312},
    ]
    gaps = sampler_gaps(rows, tick_ms)
    assert len(gaps) == 1
    assert gaps[0] == {"t_sent": 0.20, "t_reply": 2.200}


def test_sampler_gaps_threshold_is_five_ticks():
    tick_ms = 100  # threshold = 5 * 0.1 = 0.5s
    just_under = {"t_sent": 0.0, "t_reply": 0.49}
    just_over = {"t_sent": 1.0, "t_reply": 1.51}
    assert sampler_gaps([just_under], tick_ms) == []
    assert len(sampler_gaps([just_over], tick_ms)) == 1
    assert GAP_TICK_MULTIPLE == 5


def test_sampler_gaps_skips_rows_missing_timestamps():
    assert sampler_gaps([{"connected_clients": 1}], 100) == []


# --------------------------------------------------------------------------- netstat guard


def test_sampler_local_reads_netstat_deltas(monkeypatch):
    sampler = ServerSampler("127.0.0.1", 6379, 100)
    assert sampler._local is True
    base = module.netstat.ListenCounters(listen_overflows=10, listen_drops=12)
    later = module.netstat.ListenCounters(listen_overflows=40, listen_drops=52)
    sampler._baseline = base
    monkeypatch.setattr(module.netstat, "read_counters", lambda *a, **k: later)
    assert sampler._netstat_deltas() == (30, 40)


def test_sampler_remote_records_null_netstat():
    sampler = ServerSampler("10.0.0.5", 6379, 100)
    assert sampler._local is False
    assert sampler._netstat_deltas() == (None, None)


# --------------------------------------------------------------------------- field round-trip / validation


def test_server_sample_ms_round_trip(tmp_path):
    task = _scenario_task(scenario="bgsave", server_sample_ms=250)
    path = tmp_path / f"task_{task.task_id}.json"
    task.save_to_file(path)
    loaded = BaseTaskData.from_file(path)
    assert isinstance(loaded, ScenarioTaskData)
    assert loaded.server_sample_ms == 250


def test_scenario_task_loads_without_server_sample_ms_field(tmp_path):
    """A serialized task predating server_sample_ms must still load (defaults to 0)."""
    task = _scenario_task(scenario="bgsave")
    path = tmp_path / f"task_{task.task_id}.json"
    task.save_to_file(path)
    doc = json.loads(path.read_text())
    doc.pop("server_sample_ms", None)
    path.write_text(json.dumps(doc))
    loaded = BaseTaskData.from_file(path)
    assert loaded.server_sample_ms == 0


@pytest.mark.parametrize("bad", [1, 10, 19, -5])
def test_server_sample_ms_rejects_below_20(bad):
    with pytest.raises(ValueError, match="server_sample_ms"):
        _scenario_task(scenario="bgsave", server_sample_ms=bad)


def test_server_sample_ms_zero_is_allowed():
    assert _scenario_task(scenario="bgsave", server_sample_ms=0).server_sample_ms == 0


# --------------------------------------------------------------------------- CLI default


def test_cli_connection_storm_defaults_sample_ms_to_100(monkeypatch):
    from conductress import cli

    captured = {}

    class FakeSubmitter:
        def __init__(self, args):
            pass

        def submit_task(self, task):
            captured["task"] = task

        def finish(self):
            return {}

    monkeypatch.setattr(cli, "_TaskSubmitter", FakeSubmitter)
    monkeypatch.setattr(cli, "_finish_submission", lambda submission, args: False)

    parser = cli.build_parser()
    args = parser.parse_args(
        ["queue", "add-scenario", "--scenario", "connection-storm", "--source", config.REPO_NAMES[0]]
    )
    assert cli.handle_queue_add_scenario(args) == 0
    assert captured["task"].server_sample_ms == 100


def test_cli_other_scenario_defaults_sample_ms_off(monkeypatch):
    from conductress import cli

    captured = {}

    class FakeSubmitter:
        def __init__(self, args):
            pass

        def submit_task(self, task):
            captured["task"] = task

        def finish(self):
            return {}

    monkeypatch.setattr(cli, "_TaskSubmitter", FakeSubmitter)
    monkeypatch.setattr(cli, "_finish_submission", lambda submission, args: False)

    parser = cli.build_parser()
    args = parser.parse_args(["queue", "add-scenario", "--scenario", "bgsave", "--source", config.REPO_NAMES[0]])
    assert cli.handle_queue_add_scenario(args) == 0
    assert captured["task"].server_sample_ms == 0


def test_cli_explicit_sample_ms_overrides_connection_storm_default(monkeypatch):
    from conductress import cli

    captured = {}

    class FakeSubmitter:
        def __init__(self, args):
            pass

        def submit_task(self, task):
            captured["task"] = task

        def finish(self):
            return {}

    monkeypatch.setattr(cli, "_TaskSubmitter", FakeSubmitter)
    monkeypatch.setattr(cli, "_finish_submission", lambda submission, args: False)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "queue",
            "add-scenario",
            "--scenario",
            "connection-storm",
            "--source",
            config.REPO_NAMES[0],
            "--server-sample-ms",
            "50",
        ]
    )
    assert cli.handle_queue_add_scenario(args) == 0
    assert captured["task"].server_sample_ms == 50


# --------------------------------------------------------------------------- origin_wall / schema_version


def test_storm_namespace_folds_origin_wall():
    doc = {
        "schema_version": SCHEMA_VERSION,
        "origin_wall": 1789766100.5,
        "config": {},
        "stall": {"kind": "none"},
        "listen_overflow_delta": {},
        "prewarm_listen_overflow_delta": {},
        "metrics": {"totals": {}, "time_to_quiescence": {}},
    }
    ns = storm_metrics_namespace(doc)["storm"]
    assert ns["origin_wall"] == 1789766100.5
    assert ns["schema_version"] == SCHEMA_VERSION


def test_schema_version_is_at_least_2():
    # v2 is where origin_wall entered the document.
    assert SCHEMA_VERSION >= 2


def test_stormgen_document_serializes_origin_wall(monkeypatch):
    from conductress.stormgen import __main__ as sg_main
    from conductress.stormgen.runner import StormConfig, StormResult
    from conductress.stormgen.stall import StallRecord

    config_obj = StormConfig(
        host="127.0.0.1",
        port=6379,
        clients=10,
        burst_ms=200,
        duration_s=5.0,
        connect_timeout_ms=1000.0,
        reply_timeout_ms=500.0,
        policy_spec="fixed:200",
        stall_spec="none",
    )
    result = StormResult(
        events=[],
        stall=StallRecord(kind="none"),
        origin_wall=1789766100.25,
        duration_s=5.0,
        listen_delta={"ListenOverflows": None, "ListenDrops": None},
        prewarm_listen_delta={"ListenOverflows": None, "ListenDrops": None},
        prewarm_connections=0,
        config=config_obj,
    )
    doc = sg_main.build_document(result)
    assert doc["origin_wall"] == 1789766100.25
    assert doc["schema_version"] == SCHEMA_VERSION


# --------------------------------------------------------------------------- faked-runner sampler ordering


class _FakeServer:
    def __init__(self):
        self.ip = "127.0.0.1"
        self.port = 6379
        self.server_cpus = [0, 1]
        self.valkey_commands = []

    def get_build_hash(self):
        return "deadbeef"

    async def run_valkey_command(self, command):
        self.valkey_commands.append(command)
        return "OK"

    async def run_host_command(self, command, check=True):  # pylint: disable=unused-argument
        # memtier measure command: return a parseable Totals line + a tiny JSON timeseries.
        if "memtier_benchmark" in command and "--test-time" in command:
            return "Type Ops/sec\nTotals 50000.0 ... \n", ""
        if command.startswith("cat "):
            return json.dumps({"ALL STATS": {"Totals": {"Time-Serie": {"0": {"Count": 50000}, "1": {"Count": 1}}}}}), ""
        return "", ""


class _FakeReplicationGroup:
    events = []
    server = None

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.primary = _FakeReplicationGroup.server

    async def kill_all_valkey_instances(self):
        _FakeReplicationGroup.events.append("kill")

    async def start(self):
        _FakeReplicationGroup.events.append("start")

    async def stop_all_servers(self):
        _FakeReplicationGroup.events.append("stop-all")


class _FakeSampler:
    """Records lifecycle order relative to the overlay via a shared log."""

    order: list = []

    def __init__(self, host, port, tick_ms, connect_timeout=5.0):  # pylint: disable=unused-argument
        self.tick_ms = tick_ms
        self.rows = [{"t_sent": 1.0, "t_reply": 1.01, "connected_clients": 33, "listen_overflows": 0}]
        self._stopped = False
        _FakeSampler.order.append("sampler-created")

    async def run(self):
        _FakeSampler.order.append("sampler-run")
        # Yield until stopped.
        while not self._stopped:
            await asyncio.sleep(0)

    def stop(self):
        self._stopped = True
        _FakeSampler.order.append("sampler-stop")


class _FakeStormOverlay:
    def __init__(self, spec, work_dir):  # pylint: disable=unused-argument
        pass

    def extra_server_args(self):
        return ""

    async def start(self, server, work_dir):  # pylint: disable=unused-argument
        _FakeSampler.order.append("overlay-start")
        return {"cmd": None}

    async def finish(self, server, handle):  # pylint: disable=unused-argument
        _FakeSampler.order.append("overlay-finish")
        return module.OverlayResult(rate=None, metrics=storm_metrics_namespace(_storm_doc()))

    async def abort(self, server, handle):  # pylint: disable=unused-argument
        pass


def _storm_doc():
    return {
        "schema_version": SCHEMA_VERSION,
        "origin_wall": 1000.0,
        "config": {},
        "stall": {"kind": "debug-sleep", "started": 1001.0, "ended": 1003.0},
        "listen_overflow_delta": {"ListenOverflows": 500, "ListenDrops": 500},
        "prewarm_listen_overflow_delta": {"ListenOverflows": 0, "ListenDrops": 0},
        "metrics": {
            "totals": {"amplification": 2.0, "attempts": 20, "connected_clients": 10},
            "time_to_quiescence": {"from_start": 4.0, "from_stall_end": 2.0},
            "timeline": [],
        },
    }


@pytest.mark.asyncio
async def test_runner_starts_sampler_before_overlay_and_records_rows(monkeypatch, tmp_path):
    _FakeReplicationGroup.events = []
    _FakeReplicationGroup.server = _FakeServer()
    _FakeSampler.order = []

    real_sleep = module.asyncio.sleep

    async def yielding_sleep(_s):
        await real_sleep(0)

    monkeypatch.setattr(module, "ReplicationGroup", _FakeReplicationGroup)
    monkeypatch.setattr(module, "ServerSampler", _FakeSampler)
    monkeypatch.setattr(module, "StormOverlay", lambda spec, wd: _FakeStormOverlay(spec, wd))
    monkeypatch.setattr(module.asyncio, "sleep", yielding_sleep)

    task = _scenario_task(scenario="connection-storm", server_sample_ms=100, repetitions=1)
    runner = task.prepare_task_runner([HOST])
    runner.file_protocol = FileProtocol(runner.task_name, role_id="client", base_dir=tmp_path)
    written = {}
    runner.file_protocol.write_results = MagicMock(side_effect=lambda r: written.setdefault("r", r))

    await runner.run()

    # Sampler launched before the overlay and stopped after the background measurement.
    assert _FakeSampler.order[0] == "sampler-created"
    assert _FakeSampler.order.index("sampler-created") < _FakeSampler.order.index("overlay-start")
    assert _FakeSampler.order.index("sampler-stop") < _FakeSampler.order.index("overlay-finish")

    results: BenchmarkResults = written["r"]
    per_rep = results.data["scenario_metrics"]["per_rep"][0]
    assert per_rep["server_timeline_tick_ms"] == 100
    assert per_rep["server_timeline"] and per_rep["server_timeline"][0]["connected_clients"] == 33
    # Wall anchors recorded.
    assert "measure_start_wall" in per_rep
    assert "background_start_wall" in per_rep
    assert "background_end_wall" in per_rep
    # storm.origin_wall folded through.
    assert per_rep["storm"]["origin_wall"] == 1000.0


# --------------------------------------------------------------------------- coalesced replies
# Regression for the first fleet cell (bench, 2026-09-18, task 2026.09.18_22.21.13.806551):
# both INFO replies of a tick arrived in ONE TCP read; a per-reply parser returned
# the first and discarded the second, so the next read waited forever and the
# runner hung at `await sampler_task` until the connection was killed by hand.


def _bulk(text: str) -> bytes:
    body = text.encode()
    return b"$" + str(len(body)).encode() + b"\r\n" + body + b"\r\n"


class _CoalescingReader:
    """A StreamReader stand-in that delivers everything in one read, then EOF."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.reads = 0

    async def read(self, _n: int) -> bytes:
        self.reads += 1
        data, self._payload = self._payload, b""
        return data


@pytest.mark.asyncio
async def test_read_reply_keeps_second_reply_from_a_coalesced_read():
    sampler = module.ServerSampler("127.0.0.1", 6379, tick_ms=100)
    first_text = "# Clients\r\nconnected_clients:2\r\n"
    second_text = "# Stats\r\ntotal_connections_received:4002\r\nrejected_connections:0\r\n"
    reader = _CoalescingReader(_bulk(first_text) + _bulk(second_text))

    first = await asyncio.wait_for(sampler._read_reply(reader), timeout=1.0)  # pylint: disable=protected-access
    second = await asyncio.wait_for(sampler._read_reply(reader), timeout=1.0)  # pylint: disable=protected-access

    assert first == first_text
    assert second == second_text
    # The second reply came from the parser's leftover bytes, not from a new read.
    assert reader.reads == 1


@pytest.mark.asyncio
async def test_one_tick_parses_both_replies_from_one_read():
    sampler = module.ServerSampler("10.0.0.5", 6379, tick_ms=100)  # remote: netstat recorded as null
    payload = _bulk("# Clients\r\nconnected_clients:7\r\n") + _bulk(
        "# Stats\r\ntotal_connections_received:11\r\nrejected_connections:1\r\ntotal_commands_processed:99\r\n"
    )
    reader = _CoalescingReader(payload)

    class _Writer:
        def write(self, _data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    await asyncio.wait_for(sampler._one_tick(reader, _Writer()), timeout=1.0)  # pylint: disable=protected-access
    assert len(sampler.rows) == 1
    row = sampler.rows[0]
    assert row["connected_clients"] == 7
    assert row["total_connections_received"] == 11
    assert row["rejected_connections"] == 1
    assert row["total_commands_processed"] == 99
