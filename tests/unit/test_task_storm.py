"""Unit tests for the connection-storm Conductress task.

Covers the task dataclass (validation + a save->load->use round trip through
the queue's own serialization), the ``add-storm`` CLI mapping, and the runner
with the server, generator subprocess, and INFO sampler faked so phase order
and result assembly are asserted without a real server.
"""

import json
from unittest.mock import MagicMock

import pytest

from conductress import config
from conductress.file_protocol import BenchmarkResults, FileProtocol
from conductress.task_queue import BaseTaskData
from conductress.tasks import task_storm as module
from conductress.tasks.task_storm import METHOD, StormTaskData, StormTaskRunner, parse_handshake

HOST = config.ServerInfo(ip="127.0.0.1", username="ec2-user")


def _task(**overrides) -> StormTaskData:
    fields = dict(
        source=config.REPO_NAMES[0],
        specifier="unstable",
        make_args="",
        replicas=0,
        note="",
        requirements={},
        clients=30,
        duration_s=5.0,
        stall="debug-sleep:2",
        stall_after_s=1.0,
    )
    fields.update(overrides)
    return StormTaskData(**fields)


# --------------------------------------------------------------------------- dataclass


def test_defaults_and_short_description():
    task = _task(stall="none")
    assert "storm" in task.short_description()
    assert task.handshake_commands() == ["HELLO 3"]
    assert task.first_command == "GET stormkey"


def test_parse_handshake_variants():
    assert parse_handshake("HELLO 3") == ["HELLO 3"]
    assert parse_handshake("") == []
    assert parse_handshake("HELLO 3;AUTH u p") == ["HELLO 3", "AUTH u p"]


def test_empty_handshake_means_no_handshake():
    assert _task(handshake="").handshake_commands() == []


def test_bind_addr_list():
    assert _task(bind_addrs="127.0.0.2, 127.0.0.3").bind_addr_list() == ["127.0.0.2", "127.0.0.3"]
    assert _task(bind_addrs="").bind_addr_list() == []


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"clients": 0}, "clients"),
        ({"duration_s": 0}, "duration_s"),
        ({"workers": 0}, "workers"),
        ({"tick_ms": 0}, "tick_ms"),
        ({"policy": "bogus"}, "policy"),
        ({"stall": "bogus"}, "stall"),
        ({"stall_after_s": 5.0, "duration_s": 5.0}, "stall_after_s"),
        ({"background_load": "cachecannon"}, "background_load"),
    ],
)
def test_validation_rejects_bad_fields(overrides, match):
    with pytest.raises(ValueError, match=match):
        _task(**overrides)


def test_queue_round_trip_preserves_every_field(tmp_path):
    """save -> load -> use: a persisted field must survive asdict/from_file.

    A dataclass field without a default would be silently dropped from the
    serialized document; this asserts every storm field round-trips and the
    reconstructed task behaves identically.
    """
    task = _task(
        clients=1234,
        burst_ms=321,
        connect_timeout_ms=750.0,
        reply_timeout_ms=250.0,
        policy="exp:100:800:jitter",
        handshake="HELLO 3;AUTH u p",
        first_command="PING",
        duration_s=17.0,
        stall="debug-sleep:3",
        stall_after_s=4.0,
        workers=2,
        bind_addrs="127.0.0.2,127.0.0.3",
        tick_ms=50,
        io_threads=4,
        server_args="--maxmemory 1gb",
        note="round-trip",
    )
    path = tmp_path / f"task_{task.task_id}.json"
    task.save_to_file(path)

    loaded = BaseTaskData.from_file(path)
    assert isinstance(loaded, StormTaskData)
    # Every persisted field matches.
    for field in (
        "clients",
        "burst_ms",
        "connect_timeout_ms",
        "reply_timeout_ms",
        "policy",
        "handshake",
        "first_command",
        "duration_s",
        "stall",
        "stall_after_s",
        "workers",
        "bind_addrs",
        "tick_ms",
        "io_threads",
        "server_args",
        "note",
    ):
        assert getattr(loaded, field) == getattr(task, field), field
    # And the reconstructed task is usable.
    assert loaded.handshake_commands() == ["HELLO 3", "AUTH u p"]
    assert loaded.bind_addr_list() == ["127.0.0.2", "127.0.0.3"]
    assert loaded.task_type == "StormTaskData"


def test_prepare_task_runner_returns_storm_runner():
    runner = _task().prepare_task_runner([HOST])
    assert isinstance(runner, StormTaskRunner)


# --------------------------------------------------------------------------- CLI


def test_cli_add_storm_maps_to_task_data(monkeypatch):
    from conductress import cli

    captured = {}

    class FakeSubmitter:
        def __init__(self, args):
            captured["args"] = args

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
            "add-storm",
            "--source",
            config.REPO_NAMES[0],
            "--specifier",
            "unstable",
            "--clients",
            "1500",
            "--burst-ms",
            "150",
            "--policy",
            "exp:50:400",
            "--handshake",
            "HELLO 3",
            "--first-command",
            "GET stormkey",
            "--duration",
            "12",
            "--stall",
            "debug-sleep:2",
            "--stall-after",
            "3",
            "--workers",
            "2",
            "--io-threads",
            "1",
            "--tick-ms",
            "100",
        ]
    )
    rc = cli.handle_queue_add_storm(args)
    assert rc == 0
    task = captured["task"]
    assert isinstance(task, StormTaskData)
    assert task.clients == 1500 and task.burst_ms == 150
    assert task.policy == "exp:50:400"
    assert task.duration_s == 12.0 and task.stall == "debug-sleep:2"
    assert task.stall_after_s == 3.0 and task.workers == 2
    assert task.replicas == 0  # storm is single-instance


def test_cli_add_storm_rejects_bad_policy(monkeypatch, capsys):
    from conductress import cli

    parser = cli.build_parser()
    args = parser.parse_args(["queue", "add-storm", "--source", config.REPO_NAMES[0], "--policy", "nonsense"])
    rc = cli.handle_queue_add_storm(args)
    assert rc == 1
    assert "policy" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- runner


class FakeServer:
    """Stands in for the started primary Server the runner talks to."""

    def __init__(self):
        self.ip = "127.0.0.1"
        self.port = 6379
        self.server_cpus = [0, 1]
        self.valkey_commands = []
        self.info_calls = 0

    def get_build_hash(self):
        return "deadbeef"

    async def run_valkey_command(self, command):
        self.valkey_commands.append(command)
        return "OK"

    async def info(self, section):
        self.info_calls += 1
        if section == "clients":
            return {"connected_clients": "42"}
        return {"total_connections_received": "100", "rejected_connections": "0"}

    async def run_host_command(self, command, check=True):  # pylint: disable=unused-argument
        return f"ran: {command}", ""


class FakeReplicationGroup:
    """Records lifecycle calls; hands back one primary FakeServer."""

    events = []
    server = None

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        FakeReplicationGroup.events.append("construct")
        self.primary = FakeReplicationGroup.server

    async def kill_all_valkey_instances(self):
        FakeReplicationGroup.events.append("kill")

    async def start(self):
        FakeReplicationGroup.events.append("start")

    async def stop_all_servers(self):
        FakeReplicationGroup.events.append("stop-all")


class FakeProcess:
    def __init__(self):
        self.pid = 999999
        self.returncode = None


class FakeCommand:
    """The stormgen subprocess: writes the canned document to --json, exits 0."""

    document = {}
    launched = []

    def __init__(self, command: str):
        self.command = command
        self.p = FakeProcess()
        self._polls = 0
        # Extract the --json path so the runner reads a real file.
        tokens = command.split()
        self._json_path = tokens[tokens.index("--json") + 1].strip("'")

    def start(self):
        FakeCommand.launched.append(self.command)
        with open(self._json_path, "w", encoding="utf-8") as handle:
            json.dump(FakeCommand.document, handle)

    def is_running(self):
        self._polls += 1
        if self._polls <= 2:
            return True
        self.p.returncode = 0
        return False

    def poll_output(self):
        return None, None


def _document(recovery=1.5, amplification=2.3, from_start=6.0):
    return {
        "schema_version": 1,
        "config": {"clients": 30},
        "stall": {"kind": "debug-sleep", "started": 100.0, "ended": 102.0, "stall_end_relative": 2.0},
        "listen_overflow_delta": {"ListenOverflows": 512, "ListenDrops": 512},
        "metrics": {
            "totals": {"clients": 30, "attempts": 69, "amplification": amplification, "connected_clients": 30},
            "outcomes": {"connected": 30, "connect_timeout": 20, "reply_timeout": 19},
            "attempt_latency": {"p50_ms": 100, "p99_ms": 500, "max_ms": 900},
            "time_to_quiescence": {"from_start": from_start, "from_stall_end": recovery},
            "timeline": [{"t_ms": 0, "started": 30, "connected": 10, "timeouts": 0}],
        },
    }


@pytest.fixture
def faked(monkeypatch, tmp_path):
    FakeReplicationGroup.events = []
    FakeReplicationGroup.server = FakeServer()
    FakeCommand.launched = []
    FakeCommand.document = _document()

    real_sleep = module.asyncio.sleep

    async def yielding_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(module, "ReplicationGroup", FakeReplicationGroup)
    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)
    monkeypatch.setattr(module.asyncio, "sleep", yielding_sleep)

    def make_runner(**overrides) -> StormTaskRunner:
        runner = StormTaskRunner(_task(**overrides), [HOST])
        runner.file_protocol = FileProtocol(runner.task_name, role_id="client", base_dir=tmp_path)
        runner.file_protocol.write_results = MagicMock()
        return runner

    return make_runner


@pytest.mark.asyncio
async def test_run_executes_phases_and_records_recovery_as_score(faked):
    runner = faked()

    await runner.run()

    # Server brought up then torn down.
    assert FakeReplicationGroup.events[:3] == ["construct", "kill", "start"]
    assert FakeReplicationGroup.events[-1] == "stop-all"
    # The storm key was prefilled before the generator launched.
    assert any(c.startswith("SET stormkey") for c in FakeReplicationGroup.server.valkey_commands)
    assert len(FakeCommand.launched) == 1
    # Generator invoked as a python -m conductress.stormgen subprocess.
    assert "conductress.stormgen" in FakeCommand.launched[0]

    runner.file_protocol.write_results.assert_called_once()
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.method == METHOD
    assert results.commit_hash == "deadbeef"
    assert results.score == pytest.approx(1.5)  # recovery_seconds = from_stall_end
    data = results.data
    assert data["recovery_seconds"] == pytest.approx(1.5)
    assert data["recovery_lower_is_better"] is True
    assert data["amplification"] == pytest.approx(2.3)
    assert data["listen_overflow_delta"] == {"ListenOverflows": 512, "ListenDrops": 512}
    assert data["schema_version"] == 1
    # INFO sampler produced at least one sample from the fake server.
    assert data["info_timeline"] and data["info_timeline"][0]["connected_clients"] == 42
    assert runner.status.state == "completed"


@pytest.mark.asyncio
async def test_run_no_stall_falls_back_to_quiescence_from_start(faked):
    FakeCommand.document = _document(recovery=None, from_start=4.0)
    runner = faked(stall="none", stall_after_s=0.0)

    await runner.run()

    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    # No stall -> recovery_seconds is None, score falls back to from_start.
    assert results.data["recovery_seconds"] is None
    assert results.score == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_run_stops_server_when_generator_fails(faked, monkeypatch):
    class FailingCommand(FakeCommand):
        def is_running(self):
            self._polls += 1
            if self._polls <= 1:
                return True
            self.p.returncode = 2
            return False

    monkeypatch.setattr(module, "RealtimeCommand", FailingCommand)
    runner = faked()

    with pytest.raises(RuntimeError, match="exited with code 2"):
        await runner.run()

    assert FakeReplicationGroup.events[-1] == "stop-all"
    runner.file_protocol.write_results.assert_not_called()


@pytest.mark.asyncio
async def test_generator_command_includes_no_handshake_flag(faked):
    runner = faked(handshake="")
    command = runner._stormgen_command("127.0.0.1", 6379, "/tmp/out.json")
    # An empty handshake is passed explicitly so the generator sends none.
    assert "--handshake ''" in command or "--handshake ''" in command.replace('"', "'")
    assert "--stall debug-sleep:2" in command
