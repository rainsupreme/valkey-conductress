"""Tests for the run_rsync publish helper.

Regression guard for the silent-publish-failure incident: a full data-server
disk caused every rsync push (status + series data) to fail with rc=23 while the
code only logged at WARNING and swallowed the error, freezing the dashboard for
~12h with no signal. run_rsync must return success/failure and log failures
loudly at ERROR, calling out a full disk explicitly.
"""

import logging
import subprocess
from unittest.mock import patch

from conductress.utility import run_rsync


def _completed(rc: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["rsync"], returncode=rc, stdout="", stderr=stderr)


class TestRunRsync:
    """run_rsync must never fail silently."""

    def test_success_returns_true_without_error(self, caplog):
        with patch("conductress.utility.subprocess.run", return_value=_completed(0)):
            with caplog.at_level(logging.INFO, logger="conductress.utility"):
                assert run_rsync(["rsync", "src", "dst"], "dst") is True
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_disk_full_logs_error_and_returns_false(self, caplog):
        stderr = 'rsync: [receiver] mkstemp ".x.json.zK" failed: No space left on device (28)'
        with patch("conductress.utility.subprocess.run", return_value=_completed(23, stderr)):
            with caplog.at_level(logging.ERROR, logger="conductress.utility"):
                assert run_rsync(["rsync", "src", "dst"], "dst") is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "disk-full failure must be logged at ERROR"
        assert "FULL" in errors[0].getMessage()

    def test_generic_failure_logs_error_with_returncode(self, caplog):
        with patch("conductress.utility.subprocess.run", return_value=_completed(255, "ssh: connect timed out")):
            with caplog.at_level(logging.ERROR, logger="conductress.utility"):
                assert run_rsync(["rsync", "src", "dst"], "dst") is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors
        assert "rc=255" in errors[0].getMessage()

    def test_timeout_returns_false_and_logs_error(self, caplog):
        with patch(
            "conductress.utility.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="rsync", timeout=15),
        ):
            with caplog.at_level(logging.ERROR, logger="conductress.utility"):
                assert run_rsync(["rsync", "src", "dst"], "dst", timeout=15) is False
        assert [r for r in caplog.records if r.levelno >= logging.ERROR]
