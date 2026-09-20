"""Tests for the runner-detection argv matcher in conductress.status."""

import pytest

from conductress.status import _cmdline_is_runner


@pytest.mark.parametrize(
    "argv",
    [
        # Console-script form, as installed in a venv.
        ["/home/x/.venv/bin/conductress", "run"],
        ["/home/x/.venv/bin/conductress", "run", "--sweep", "--publish", "u@h:/p"],
        # Module form, as bootstrap.py / deploy write it.
        ["python3", "-m", "conductress", "run", "--publish"],
        ["/usr/bin/python3", "-m", "conductress", "run", "--sweep", "--fleet-mode", "shadow"],
        # Legacy module name still accepted (README documents `python -m src`).
        ["python3", "-m", "src", "run"],
        # Options may precede the `run` positional.
        ["conductress", "--verbose", "run"],
    ],
)
def test_cmdline_is_runner_accepts_real_runner(argv):
    assert _cmdline_is_runner(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        # pytest / mypy under a venv that merely has "conductress" in its path.
        ["python3", "-m", "pytest", "tests/unit"],
        ["/home/x/conductress/.venv/bin/python", "-m", "pytest"],
        ["/home/x/conductress/.venv/bin/python", "-m", "mypy", "src/"],
        # Other conductress subcommands are not the runner.
        ["conductress", "status"],
        ["conductress", "queue", "add", "--tests", "get"],
        ["/home/x/.venv/bin/conductress", "sweep", "status"],
        # `run` appears only as a value, not the subcommand.
        ["conductress", "queue", "add", "--note", "run"],
        # Module form without `run`.
        ["python3", "-m", "conductress"],
        ["python3", "-m", "conductress", "status"],
        # `-m` present but pointing at an unrelated module.
        ["python3", "-m", "conductress.stormgen", "--json"],
        # Empty / malformed.
        [],
        ["python3"],
        ["conductress"],
    ],
)
def test_cmdline_is_runner_rejects_non_runner(argv):
    assert _cmdline_is_runner(argv) is False
