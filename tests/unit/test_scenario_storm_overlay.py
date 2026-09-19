"""Tests for the connection-storm scenario overlay and the overlay abstraction.

Covers the CommandOverlay adapter equivalence (existing scenarios produce the
IDENTICAL command string), overlay_spec parsing/validation, the CLI --storm-*
flags serializing into overlay_spec, StormOverlay result assembly against a
faked generator (no real server), ScenarioTaskData backward-compat loading a
serialized task that predates overlay_spec, and the server-args auto-append
rule (only connection-storm + debug-sleep).
"""

import json

import pytest

from conductress import config
from conductress.task_queue import BaseTaskData
from conductress.tasks import task_scenario as module
from conductress.tasks.task_scenario import (
    DEFAULT_MAXCLIENTS,
    SCENARIO_CHOICES,
    STORM_MAXCLIENTS_HEADROOM,
    CommandOverlay,
    OverlayResult,
    ScenarioTaskData,
    StormOverlay,
    build_overlay_command,
    parse_storm_spec,
    storm_metrics_namespace,
    storm_uses_debug_sleep,
)
from conductress.topology import TopologySpec

HOST = config.ServerInfo(ip="127.0.0.1", username="ec2-user")

# The command-string scenarios (everything except the storm) the adapter wraps.
COMMAND_SCENARIOS = tuple(s for s in SCENARIO_CHOICES if s != "connection-storm")


def _scenario_task(**overrides) -> ScenarioTaskData:
    fields = dict(
        source=config.REPO_NAMES[0],
        specifier="unstable",
        make_args="",
        topology=TopologySpec.standalone(),
        note="",
        requirements={},
        scenario="eval-storm",
        val_size=512,
        io_threads=9,
        pipelining=1,
        duration=30,
    )
    fields.update(overrides)
    return ScenarioTaskData(**fields)


class _FakeServer:
    def __init__(self, ip="127.0.0.1", port=6379):
        self.ip = ip
        self.port = port


# --------------------------------------------------------------------------- adapter equivalence


@pytest.mark.parametrize("scenario", COMMAND_SCENARIOS)
def test_command_overlay_yields_identical_command(scenario):
    """CommandOverlay.command() must equal build_overlay_command() for every scenario."""
    runner = _scenario_task(scenario=scenario).prepare_task_runner([HOST])
    server = _FakeServer()
    expected = build_overlay_command(
        scenario=scenario,
        server_ip=server.ip,
        port=server.port,
        duration=runner.duration,
        keyspace=module.MIXED_KEYSPACE,
        val_size=runner.val_size,
        overlay_value_size=runner.overlay_value_size,
    )
    overlay = runner._overlay
    assert isinstance(overlay, CommandOverlay)
    assert overlay.command(server) == expected


def test_connection_storm_uses_storm_overlay():
    runner = _scenario_task(scenario="connection-storm").prepare_task_runner([HOST])
    assert isinstance(runner._overlay, StormOverlay)


# --------------------------------------------------------------------------- overlay_spec parse/validate


def test_parse_storm_spec_empty_is_all_defaults():
    spec = parse_storm_spec("")
    assert spec["clients"] == 2000
    assert spec["policy"] == "fixed:200"
    assert spec["stall"] == "none"
    assert spec["handshake"] == ["HELLO 3"]
    assert spec["first_command"] == "GET storm:key"
    assert spec["prewarm_connections"] is None


def test_parse_storm_spec_overrides_and_keeps_defaults():
    spec = parse_storm_spec(json.dumps({"clients": 500, "stall": "debug-sleep:2"}))
    assert spec["clients"] == 500
    assert spec["stall"] == "debug-sleep:2"
    assert spec["burst_ms"] == 200  # default preserved


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps([1, 2]),  # not an object
        json.dumps({"unknown_key": 1}),
        json.dumps({"policy": "bogus"}),
        json.dumps({"stall": "bogus"}),
        json.dumps({"clients": 0}),
        json.dumps({"workers": -1}),
        json.dumps({"burst_after_stall_ms": -1}),
        json.dumps({"prewarm_connections": -5}),
    ],
)
def test_parse_storm_spec_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_storm_spec(bad)


def test_storm_uses_debug_sleep():
    assert storm_uses_debug_sleep(parse_storm_spec(json.dumps({"stall": "debug-sleep:2"}))) is True
    assert storm_uses_debug_sleep(parse_storm_spec(json.dumps({"stall": "none"}))) is False


def test_storm_uses_debug_sleep_for_slow_loop_kinds():
    # A slow-loop whose block is DEBUG SLEEP needs enable-debug-command; a lua
    # slow-loop does not.
    ds = parse_storm_spec(json.dumps({"stall": "slow-loop:debug-sleep:50:100:6"}))
    lua = parse_storm_spec(json.dumps({"stall": "slow-loop:lua:50:100:6"}))
    assert storm_uses_debug_sleep(ds) is True
    assert storm_uses_debug_sleep(lua) is False


# --------------------------------------------------------------------------- ScenarioTaskData gating


def test_overlay_spec_rejected_for_non_storm_scenario():
    with pytest.raises(ValueError, match="only valid for scenario"):
        _scenario_task(scenario="eval-storm", overlay_spec=json.dumps({"clients": 10}))


def test_connection_storm_rejects_unparsable_spec_at_construction():
    with pytest.raises(ValueError):
        _scenario_task(scenario="connection-storm", overlay_spec="{not json")


def test_connection_storm_accepts_empty_spec():
    task = _scenario_task(scenario="connection-storm", overlay_spec="")
    assert task.storm_spec()["clients"] == 2000


# --------------------------------------------------------------------------- backward-compat round-trip


def test_scenario_task_loads_without_overlay_spec_field(tmp_path):
    """A serialized ScenarioTaskData predating overlay_spec must still load."""
    task = _scenario_task(scenario="bgsave")
    path = tmp_path / f"task_{task.task_id}.json"
    task.save_to_file(path)
    # Simulate an older on-disk task: strip the new field entirely.
    doc = json.loads(path.read_text())
    doc.pop("overlay_spec", None)
    path.write_text(json.dumps(doc))

    loaded = BaseTaskData.from_file(path)
    assert isinstance(loaded, ScenarioTaskData)
    assert loaded.scenario == "bgsave"
    assert loaded.overlay_spec == ""  # default fills in


def test_scenario_task_round_trip_preserves_overlay_spec(tmp_path):
    spec = json.dumps({"clients": 800, "stall": "debug-sleep:1"})
    task = _scenario_task(scenario="connection-storm", overlay_spec=spec)
    path = tmp_path / f"task_{task.task_id}.json"
    task.save_to_file(path)
    loaded = BaseTaskData.from_file(path)
    assert loaded.overlay_spec == spec
    assert loaded.storm_spec()["clients"] == 800


# --------------------------------------------------------------------------- server-args auto-append


def test_debug_sleep_storm_appends_enable_debug_command():
    runner = _scenario_task(
        scenario="connection-storm",
        overlay_spec=json.dumps({"stall": "debug-sleep:2"}),
        server_args="--maxmemory 1gb",
    ).prepare_task_runner([HOST])
    assert runner.server_args_effective == "--maxmemory 1gb --enable-debug-command local"


def test_no_stall_storm_does_not_append_debug_command():
    runner = _scenario_task(
        scenario="connection-storm",
        overlay_spec=json.dumps({"stall": "none"}),
    ).prepare_task_runner([HOST])
    assert "--enable-debug-command" not in runner.server_args_effective


def test_non_storm_scenario_never_appends_debug_command():
    runner = _scenario_task(scenario="eval-storm", server_args="--maxmemory 1gb").prepare_task_runner([HOST])
    assert runner.server_args_effective == "--maxmemory 1gb"
    assert "--enable-debug-command" not in runner.server_args_effective


# --------------------------------------------------------------------------- StormOverlay result assembly


def _storm_document():
    return {
        "schema_version": 1,
        "config": {"clients": 30, "workers": 4, "prewarm_connections": 30},
        "stall": {"kind": "debug-sleep", "started": 100.0, "ended": 102.0, "stall_end_relative": 2.0},
        "listen_overflow_delta": {"ListenOverflows": 512, "ListenDrops": 512},
        "prewarm_listen_overflow_delta": {"ListenOverflows": 900, "ListenDrops": 900},
        "metrics": {
            "totals": {"clients": 30, "attempts": 69, "amplification": 2.3, "connected_clients": 30},
            "burst_actual_ms": 175.0,
            "outcomes": {"connected": 30, "connect_timeout": 39},
            "attempt_latency": {"p50_ms": 100, "p99_ms": 500, "max_ms": 900},
            "time_to_quiescence": {"from_start": 6.0, "from_stall_end": 1.5},
            "timeline": [{"t_ms": 0, "started": 30, "connected": 10, "timeouts": 0}],
        },
    }


def test_storm_metrics_namespace_folds_document():
    ns = storm_metrics_namespace(_storm_document())
    storm = ns["storm"]
    assert storm["quiescence_from_stall_end_s"] == 1.5
    assert storm["quiescence_from_start_s"] == 6.0
    assert storm["amplification"] == 2.3
    assert storm["listen_overflow_delta"] == {"ListenOverflows": 512, "ListenDrops": 512}
    assert storm["prewarm_listen_overflow_delta"] == {"ListenOverflows": 900, "ListenDrops": 900}
    assert storm["burst_actual_ms"] == 175.0
    assert storm["schema_version"] == 1


@pytest.mark.asyncio
async def test_storm_overlay_start_finish_assembles_result(monkeypatch, tmp_path):
    """StormOverlay.start/finish parse the generator's JSON with the process faked."""

    class FakeProcess:
        def __init__(self):
            self.returncode = None

    class FakeCommand:
        launched: list = []

        def __init__(self, command):
            self.command = command
            self.p = FakeProcess()
            self._polls = 0
            tokens = command.split()
            self._json_path = tokens[tokens.index("--json") + 1].strip("'")

        def start(self):
            FakeCommand.launched.append(self.command)
            with open(self._json_path, "w", encoding="utf-8") as handle:
                json.dump(_storm_document(), handle)

        def is_running(self):
            self._polls += 1
            if self._polls <= 1:
                return True
            self.p.returncode = 0
            return False

        def poll_output(self):
            return None, None

    real_sleep = module.asyncio.sleep

    async def yielding_sleep(_s):
        await real_sleep(0)

    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)
    monkeypatch.setattr(module.asyncio, "sleep", yielding_sleep)

    spec = parse_storm_spec(json.dumps({"clients": 30, "stall": "debug-sleep:2"}))
    spec["_duration_s"] = 5.0
    overlay = StormOverlay(spec, tmp_path)
    server = _FakeServer()

    handle = await overlay.start(server, tmp_path)
    result = await overlay.finish(server, handle)

    assert isinstance(result, OverlayResult)
    assert result.rate is None  # the storm has no single ops/s rate
    assert result.metrics["storm"]["amplification"] == 2.3
    assert result.metrics["storm"]["quiescence_from_stall_end_s"] == 1.5
    assert "conductress.stormgen" in FakeCommand.launched[0]
    assert "--stall debug-sleep:2" in FakeCommand.launched[0]


def test_storm_overlay_extra_server_args():
    spec = parse_storm_spec(json.dumps({"stall": "debug-sleep:2"}))
    spec["_duration_s"] = 5.0
    assert StormOverlay(spec, "/tmp").extra_server_args() == "--enable-debug-command local"
    spec2 = parse_storm_spec(json.dumps({"stall": "none"}))
    spec2["_duration_s"] = 5.0
    assert StormOverlay(spec2, "/tmp").extra_server_args() == ""


# --------------------------------------------------------------------------- storm size guard


def test_storm_below_maxclients_threshold_adds_no_maxclients():
    # clients + headroom <= default: no --maxclients.
    clients = DEFAULT_MAXCLIENTS - STORM_MAXCLIENTS_HEADROOM  # exactly at the boundary
    spec = parse_storm_spec(json.dumps({"clients": clients, "stall": "none"}))
    spec["_duration_s"] = 30.0
    assert "--maxclients" not in StormOverlay(spec, "/tmp").extra_server_args()


def test_storm_above_maxclients_threshold_appends_maxclients():
    clients = DEFAULT_MAXCLIENTS  # 10000 + 4096 > 10000
    spec = parse_storm_spec(json.dumps({"clients": clients, "stall": "none"}))
    spec["_duration_s"] = 30.0
    args = StormOverlay(spec, "/tmp").extra_server_args()
    assert f"--maxclients {clients + STORM_MAXCLIENTS_HEADROOM}" in args


def test_storm_size_guard_combines_with_debug_command():
    clients = 12000
    spec = parse_storm_spec(json.dumps({"clients": clients, "stall": "debug-sleep:2"}))
    spec["_duration_s"] = 30.0
    args = StormOverlay(spec, "/tmp").extra_server_args()
    assert "--enable-debug-command local" in args
    assert f"--maxclients {clients + STORM_MAXCLIENTS_HEADROOM}" in args


def test_storm_size_guard_flows_into_server_args_effective():
    runner = _scenario_task(
        scenario="connection-storm",
        overlay_spec=json.dumps({"clients": 12000, "stall": "none"}),
        server_args="--maxmemory 1gb",
    ).prepare_task_runner([HOST])
    assert f"--maxclients {12000 + STORM_MAXCLIENTS_HEADROOM}" in runner.server_args_effective
    assert "--maxmemory 1gb" in runner.server_args_effective


def test_non_storm_scenario_never_appends_maxclients():
    # The guard is StormOverlay-only; a command scenario has no extra args.
    runner = _scenario_task(scenario="eval-storm").prepare_task_runner([HOST])
    assert "--maxclients" not in runner.server_args_effective


def test_slow_loop_debug_sleep_storm_appends_enable_debug_command():
    runner = _scenario_task(
        scenario="connection-storm",
        overlay_spec=json.dumps({"stall": "slow-loop:debug-sleep:50:100:6"}),
    ).prepare_task_runner([HOST])
    assert "--enable-debug-command local" in runner.server_args_effective


def test_slow_loop_lua_storm_does_not_append_debug_command():
    runner = _scenario_task(
        scenario="connection-storm",
        overlay_spec=json.dumps({"stall": "slow-loop:lua:50:100:6"}),
    ).prepare_task_runner([HOST])
    assert "--enable-debug-command" not in runner.server_args_effective


# --------------------------------------------------------------------------- CLI flags -> overlay_spec


def test_cli_storm_flags_serialize_into_overlay_spec():
    from conductress import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "queue",
            "add-scenario",
            "--scenario",
            "connection-storm",
            "--source",
            config.REPO_NAMES[0],
            "--storm-clients",
            "1500",
            "--storm-stall",
            "debug-sleep:2",
            "--storm-policy",
            "exp:50:400",
            "--storm-burst-first",
            "--storm-handshake",
            "HELLO 3",
            "--storm-handshake",
            "AUTH u p",
            "--storm-bind-addrs",
            "127.0.0.2,127.0.0.3",
        ]
    )
    spec_str = cli.build_scenario_overlay_spec(args)
    spec = json.loads(spec_str)
    assert spec["clients"] == 1500
    assert spec["stall"] == "debug-sleep:2"
    assert spec["policy"] == "exp:50:400"
    assert spec["burst_first"] is True
    assert spec["handshake"] == ["HELLO 3", "AUTH u p"]
    assert spec["bind_addrs"] == ["127.0.0.2", "127.0.0.3"]
    # And it parses cleanly through the validator.
    assert parse_storm_spec(spec_str)["clients"] == 1500


def test_cli_storm_flags_rejected_without_connection_storm():
    from conductress import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "queue",
            "add-scenario",
            "--scenario",
            "bgsave",
            "--source",
            config.REPO_NAMES[0],
            "--storm-clients",
            "1500",
        ]
    )
    with pytest.raises(ValueError, match="connection-storm"):
        cli.build_scenario_overlay_spec(args)


def test_cli_no_storm_flags_yields_empty_spec():
    from conductress import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["queue", "add-scenario", "--scenario", "connection-storm", "--source", config.REPO_NAMES[0]]
    )
    assert cli.build_scenario_overlay_spec(args) == ""


# --------------------------------------------------------------------------- start delay
# The first fleet cell showed the stall landing at the very start of the
# background measurement (no undisturbed baseline in the memtier series). The
# storm now launches start_delay_s after the overlay starts.


def test_parse_storm_spec_start_delay_default_and_validation():
    assert parse_storm_spec("")["start_delay_s"] == 5.0
    assert parse_storm_spec(json.dumps({"start_delay_s": 8}))["start_delay_s"] == 8
    with pytest.raises(ValueError):
        parse_storm_spec(json.dumps({"start_delay_s": -1}))


def test_storm_duration_is_window_minus_delay_and_margin():
    spec = parse_storm_spec(json.dumps({"start_delay_s": 5}))
    spec["_duration_s"] = 30.0
    assert (
        StormOverlay(spec, "/tmp")._duration_s() == 30.0 - 5.0 - StormOverlay.STORM_END_MARGIN_S
    )  # pylint: disable=protected-access
    short = parse_storm_spec(json.dumps({"start_delay_s": 5}))
    short["_duration_s"] = 5.0
    assert (
        StormOverlay(short, "/tmp")._duration_s() == StormOverlay.STORM_MIN_DURATION_S
    )  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_storm_overlay_launches_after_delay_and_finish_waits(monkeypatch, tmp_path):
    launched = []

    class FakeCommand:
        def __init__(self, command):
            self.command = command
            self.p = type("P", (), {"returncode": None})()
            self._polls = 0

        def start(self):
            launched.append(self.command)
            with open(tmp_path / "storm_result.json", "w", encoding="utf-8") as handle:
                json.dump(_storm_document(), handle)

        def is_running(self):
            self._polls += 1
            if self._polls <= 1:
                return True
            self.p.returncode = 0
            return False

        def poll_output(self):
            return None, None

    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)

    spec = parse_storm_spec(json.dumps({"start_delay_s": 0.05}))
    spec["_duration_s"] = 30.0
    overlay = StormOverlay(spec, tmp_path)
    server = _FakeServer()

    handle = await overlay.start(server, tmp_path)
    # start() returns before the generator is launched: the background
    # measurement must not be delayed by the storm's own delay.
    assert launched == []
    result = await overlay.finish(server, handle)
    assert len(launched) == 1
    assert "--duration-s 27.95" in launched[0]  # 30 - 0.05 delay - 2.0 margin
    assert result.metrics["storm"]["launched_wall"] is not None


@pytest.mark.asyncio
async def test_storm_overlay_abort_cancels_pending_launch(monkeypatch, tmp_path):
    launched = []

    class FakeCommand:
        def __init__(self, command):
            self.command = command

        def start(self):
            launched.append(self.command)

    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)
    spec = parse_storm_spec(json.dumps({"start_delay_s": 30}))
    spec["_duration_s"] = 60.0
    overlay = StormOverlay(spec, tmp_path)
    handle = await overlay.start(_FakeServer(), tmp_path)
    await overlay.abort(_FakeServer(), handle)
    await module.asyncio.sleep(0)
    assert handle["launch"].cancelled() or handle["launch"].done()
    assert launched == []


def test_cli_storm_start_delay_flag_serializes():
    from conductress import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "queue",
            "add-scenario",
            "--scenario",
            "connection-storm",
            "--source",
            config.REPO_NAMES[0],
            "--storm-start-delay-s",
            "8",
        ]
    )
    spec = json.loads(cli.build_scenario_overlay_spec(args))
    assert spec["start_delay_s"] == 8.0
