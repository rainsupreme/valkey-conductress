"""--topology / --primary-args / --replica-args on the queue add-* commands.

Every task that brings servers up through TopologyGroup takes the same layout
flags, parses them through one helper, and prints the same summary line.
add-memory builds its server directly and deliberately has no layout flag;
add-replica-read keeps its own --replicas/--io-threads levers and shares only
the per-role arguments.
"""

from unittest.mock import MagicMock, patch

import pytest

from conductress import cli, config
from conductress.topology import TopologySpec

TOPOLOGY_COMMANDS = ["add", "add-insertion", "add-mixed", "add-scenario", "add-latency", "add-cachecannon"]

# Minimal valid argv per command (after the command name).
REQUIRED = {
    "add": ["--source", "SRC", "--specifier", "abc", "--tests", "get"],
    "add-insertion": [
        "--source",
        "SRC",
        "--specifier",
        "abc",
        "--insertions",
        "1000",
        "--maxmemory",
        "1GB",
        "--max-rss",
        "2GB",
    ],
    "add-mixed": ["--source", "SRC", "--specifier", "abc", "--set-ratio", "20"],
    "add-scenario": ["--source", "SRC", "--specifier", "abc", "--scenario", "bgsave"],
    "add-latency": ["SRC", "abc", "100000"],
    "add-cachecannon": ["--source", "SRC", "--specifier", "abc"],
}


def _argv(command, *extra):
    return ["queue", command] + [a.replace("SRC", config.REPO_NAMES[0]) for a in REQUIRED[command]] + list(extra)


def _flags(command):
    parser = cli.build_parser()
    queue = next(a for a in parser._subparsers._group_actions if "queue" in a.choices).choices["queue"]
    sub = next(a for a in queue._subparsers._group_actions if command in a.choices).choices[command]
    return {opt for action in sub._actions for opt in action.option_strings}


@pytest.mark.parametrize("command", TOPOLOGY_COMMANDS)
def test_topology_commands_expose_the_layout_flags(command):
    flags = _flags(command)
    assert {"--topology", "--primary-args", "--replica-args"} <= flags


def test_memory_has_no_layout_flag_and_replica_read_shares_only_role_args():
    assert "--topology" not in _flags("add-memory")
    rr = _flags("add-replica-read")
    assert "--topology" not in rr and {"--primary-args", "--replica-args", "--replicas"} <= rr


@pytest.mark.parametrize("command", TOPOLOGY_COMMANDS)
@patch("conductress.cli.TaskQueue")
def test_default_topology_is_standalone(mock_queue_cls, command, capsys):
    mock_queue = MagicMock()
    mock_queue_cls.return_value = mock_queue
    assert cli.main(_argv(command)) == 0
    task = mock_queue.submit_task.call_args.args[0]
    assert task.topology == TopologySpec.standalone()
    assert "topology: standalone" in capsys.readouterr().out


@pytest.mark.parametrize("command", TOPOLOGY_COMMANDS)
@patch("conductress.cli.TaskQueue")
def test_replica_topology_is_built_with_role_args_and_flagged_not_comparable(mock_queue_cls, command, capsys):
    if command == "add-insertion":
        pytest.skip("bounded insertion rejects replicas by design; covered below")
    mock_queue = MagicMock()
    mock_queue_cls.return_value = mock_queue
    argv = _argv(
        command, "--topology", "replica:2", "--primary-args", "--repl-backlog-size 64mb", "--replica-args", "--x y"
    )
    assert cli.main(argv) == 0
    task = mock_queue.submit_task.call_args.args[0]
    spec = task.topology
    assert [i.role for i in spec.instances] == ["primary", "replica", "replica"]
    assert [i.port for i in spec.instances] == [6379, 6380, 6381]
    assert spec.primary.server_args == "--repl-backlog-size 64mb"
    assert all(r.server_args == "--x y" and r.own_dir for r in spec.replicas)
    assert all(i.io_threads is None for i in spec.instances)  # the task's --io-threads applies at bring-up
    assert spec.host_count() == 1
    assert "NOT sweep-comparable" in capsys.readouterr().out


@patch("conductress.cli.TaskQueue")
def test_insertion_rejects_replica_topology(mock_queue_cls, capsys):
    mock_queue_cls.return_value = MagicMock()
    assert cli.main(_argv("add-insertion", "--topology", "replica:1")) == 1
    assert "do not support replicas" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["cluster:3", "replica:", "replica:x", "replicas:2", "two"])
@patch("conductress.cli.TaskQueue")
def test_bad_topology_values_are_rejected_before_submission(mock_queue_cls, bad, capsys):
    mock_queue = MagicMock()
    mock_queue_cls.return_value = mock_queue
    assert cli.main(_argv("add", "--topology", bad)) == 1
    assert mock_queue.submit_task.call_count == 0
    assert "--topology" in capsys.readouterr().err


@patch("conductress.cli.TaskQueue")
def test_replica_args_without_replicas_is_rejected(mock_queue_cls, capsys):
    mock_queue_cls.return_value = MagicMock()
    assert cli.main(_argv("add", "--replica-args", "--x y")) == 1
    assert "no replicas" in capsys.readouterr().err


# --------------------------------------------------------------------------- factory


def test_on_host_replicas_layout():
    spec = TopologySpec.on_host_replicas(2, primary_args="--a 1", replica_args="--b 2", base_port=7000)
    assert [(i.role, i.port, i.own_dir, i.server_args) for i in spec.instances] == [
        ("primary", 7000, True, "--a 1"),
        ("replica", 7001, True, "--b 2"),
        ("replica", 7002, True, "--b 2"),
    ]
    assert not spec.is_standalone
    resolved = spec.resolved(io_threads=4, server_args="--save ''")
    assert resolved.primary.server_args == "--save '' --a 1"
    assert all(i.io_threads == 4 for i in resolved.instances)


def test_on_host_replicas_zero_is_standalone_with_primary_args():
    spec = TopologySpec.on_host_replicas(0, primary_args="--a 1")
    assert spec.is_standalone and spec.primary.server_args == "--a 1" and spec.primary.own_dir is False
    with pytest.raises(ValueError, match="no replicas"):
        TopologySpec.on_host_replicas(0, replica_args="--b 2")
    with pytest.raises(ValueError, match=">= 0"):
        TopologySpec.on_host_replicas(-1)
