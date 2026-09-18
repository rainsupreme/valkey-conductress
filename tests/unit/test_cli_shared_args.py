"""The queue add-* commands share one argument contract; these tests pin it.

Before the shared helpers each command declared --source/--specifier,
--warmup/--duration/--repetitions and --note/--make-args by hand, and they had
drifted (one --make-args had no help text, one --repetitions had none, "task"
vs "tasks"). The helpers fix names, types and defaults in one place; this file
makes any future drift a test failure rather than a surprise at the prompt.
"""

import argparse

import pytest

from conductress import cli, config
from conductress.cachecannon import DEFAULT_CACHECANNON_BINARY

# Every add-* command that takes the shared flags via --source/--specifier.
# add-latency is positional (source specifier target_rps) and is excluded on purpose.
SOURCE_COMMANDS = [
    "add",
    "add-insertion",
    "add-memory",
    "add-mixed",
    "add-scenario",
    "add-cachecannon",
    "add-replica-read",
]
RUN_LENGTH_COMMANDS = {
    # command -> flags it must expose from the run-length trio
    "add": {"--warmup", "--duration", "--repetitions"},
    "add-insertion": {"--repetitions"},
    "add-mixed": {"--warmup", "--duration", "--repetitions"},
    "add-scenario": {"--duration", "--repetitions"},
    "add-cachecannon": {"--warmup", "--duration", "--repetitions"},
    "add-replica-read": {"--warmup", "--duration", "--repetitions"},
}
REQUIRED = {
    "add": ["--tests", "get"],
    "add-insertion": ["--insertions", "1000", "--maxmemory", "1GB", "--max-rss", "2GB"],
    "add-mixed": ["--set-ratio", "20"],
    "add-scenario": ["--scenario", "bgsave"],
}


def _subparser(name: str) -> argparse.ArgumentParser:
    parser = cli.build_parser()
    queue = next(a for a in parser._subparsers._group_actions if "queue" in a.choices).choices["queue"]
    sub = next(a for a in queue._subparsers._group_actions if name in a.choices)
    return sub.choices[name]


def _flags(parser: argparse.ArgumentParser) -> dict:
    return {opt: action for action in parser._actions for opt in action.option_strings}


@pytest.mark.parametrize("command", SOURCE_COMMANDS)
def test_every_add_command_takes_source_and_specifier_with_the_same_defaults(command):
    flags = _flags(_subparser(command))
    assert flags["--source"].default == "valkey"
    assert flags["--specifier"].default == "unstable"
    assert flags["--source"].help == cli.SOURCE_HELP


@pytest.mark.parametrize("command", SOURCE_COMMANDS)
def test_every_add_command_has_the_note_and_make_args_footer(command):
    flags = _flags(_subparser(command))
    assert flags["--note"].default == ""
    assert flags["--note"].help.startswith("Optional note for the task")
    assert flags["--make-args"].default == config.DEFAULT_MAKE_ARGS
    assert flags["--make-args"].help, f"{command}: --make-args must carry help text"


@pytest.mark.parametrize("command,expected", RUN_LENGTH_COMMANDS.items())
def test_run_length_flags_share_defaults_and_types(command, expected):
    flags = _flags(_subparser(command))
    present = {f for f in ("--warmup", "--duration", "--repetitions") if f in flags}
    assert present == expected
    if "--warmup" in present:
        assert flags["--warmup"].default == f"{config.DEFAULT_WARMUP}s"
        assert f"Default: {config.DEFAULT_WARMUP}s" in flags["--warmup"].help
    if "--duration" in present:
        assert flags["--duration"].default == f"{config.DEFAULT_DURATION}s"
        assert f"Default: {config.DEFAULT_DURATION}s" in flags["--duration"].help
    assert flags["--repetitions"].type is int
    assert flags["--repetitions"].default == config.DEFAULT_REPETITIONS
    assert flags["--repetitions"].help, f"{command}: --repetitions must carry help text"


def test_per_command_help_wording_is_preserved():
    assert "per config" in _flags(_subparser("add"))["--repetitions"].help
    assert "memtier" in _flags(_subparser("add-mixed"))["--warmup"].help
    assert _flags(_subparser("add-replica-read"))["--warmup"].help.startswith("Reader warmup")
    assert "or path" in _flags(_subparser("add-memory"))["--specifier"].help


@pytest.mark.parametrize("command", ["add-cachecannon", "add-replica-read"])
def test_cachecannon_binary_default_comes_from_one_constant(command):
    action = _flags(_subparser(command))["--cachecannon-binary"]
    assert action.default == DEFAULT_CACHECANNON_BINARY
    assert DEFAULT_CACHECANNON_BINARY in action.help


@pytest.mark.parametrize("command", SOURCE_COMMANDS)
def test_shared_flags_parse_with_defaults(command):
    parser = cli.build_parser()
    args = parser.parse_args(["queue", command, *REQUIRED.get(command, [])])
    assert args.source == "valkey" and args.specifier == "unstable"
    assert args.note == "" and args.make_args == config.DEFAULT_MAKE_ARGS


# --- handler-side helpers ---


def test_parse_bytes_names_the_flag():
    assert cli._parse_bytes("1KB", "sizes") == 1024
    with pytest.raises(ValueError, match=r"^--sizes: "):
        cli._parse_bytes("lots", "sizes")


def test_check_cpulist_names_the_flag():
    cli._check_cpulist("", "client-cpus")
    cli._check_cpulist("0-3,8", "client-cpus")
    with pytest.raises(ValueError, match=r"^--client-cpus: "):
        cli._check_cpulist("a-b", "client-cpus")


def test_source_is_valid_prints_the_valid_list_once(capsys, monkeypatch):
    monkeypatch.setattr(config, "REPO_NAMES", ["valkey", "fork"])
    monkeypatch.setattr(config, "MANUALLY_UPLOADED", "manual")
    assert cli._source_is_valid("fork")
    assert capsys.readouterr().err == ""
    assert not cli._source_is_valid("nope")
    err = capsys.readouterr().err
    assert "Invalid source 'nope'" in err and "valkey, fork, manual" in err
