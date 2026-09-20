"""Tests for the `conductress --version` banner."""

import io
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress import banner
from conductress.__main__ import main

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_SCRIPT = REPO_ROOT / "brand" / "terminal" / "conductress-logo-ascii.py"
_SGR = re.compile(r"\x1b\[[0-9;]*m")


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestLockupAsset:
    def test_loads_with_wordmark_and_colour(self) -> None:
        ansi = banner.load_lockup()
        plain = _SGR.sub("", ansi)
        assert "C O N D U C T R E S S" in plain
        assert "ONLY DATA IS REAL" in plain
        assert "\x1b[38;2;" in ansi
        assert not ansi.endswith("\n")

    def test_fits_a_default_terminal(self) -> None:
        lines = _SGR.sub("", banner.load_lockup()).split("\n")
        assert len(lines) <= 20
        assert max(len(line) for line in lines) <= 80

    @pytest.mark.skipif(not BRAND_SCRIPT.exists(), reason="brand kit not present in this checkout")
    def test_packaged_lockup_matches_brand_generator(self) -> None:
        """The asset is generated, not hand-edited: it must equal the brand script's output."""
        result = subprocess.run(
            [sys.executable, str(BRAND_SCRIPT), "--palette", "sunset", "--size", "24", "--no-labels"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.rstrip("\n") == banner.load_lockup()


class TestWantsLockup:
    def test_tty_with_clean_env(self) -> None:
        assert banner.wants_lockup(_Stream(tty=True), {}) is True

    def test_pipe_never(self) -> None:
        assert banner.wants_lockup(_Stream(tty=False), {}) is False

    def test_no_color_any_value(self) -> None:
        assert banner.wants_lockup(_Stream(tty=True), {"NO_COLOR": ""}) is False
        assert banner.wants_lockup(_Stream(tty=True), {"NO_COLOR": "1"}) is False

    def test_dumb_terminal(self) -> None:
        assert banner.wants_lockup(_Stream(tty=True), {"TERM": "dumb"}) is False

    def test_stream_without_isatty(self) -> None:
        assert banner.wants_lockup(object(), {}) is False  # type: ignore[arg-type]


class TestRenderVersion:
    def test_plain_line_is_machine_readable(self) -> None:
        with patch.object(banner, "package_version", return_value="1.2.3"):
            out = banner.render_version(_Stream(tty=False), {})
        assert out == "conductress 1.2.3\n"

    def test_terminal_gets_lockup_then_version(self) -> None:
        with patch.object(banner, "package_version", return_value="1.2.3"):
            out = banner.render_version(_Stream(tty=True), {})
        assert out.startswith(banner.load_lockup())
        assert out.endswith("\n\nconductress 1.2.3\n")

    def test_uninstalled_source_tree_reports_unknown(self) -> None:
        with patch.object(banner.metadata, "version", side_effect=banner.metadata.PackageNotFoundError):
            assert banner.version_line() == "conductress unknown"


class TestVersionFlag:
    @patch("sys.argv", ["conductress", "--version"])
    def test_prints_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(banner, "package_version", return_value="9.9.9"):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 0
        assert capsys.readouterr().out == "conductress 9.9.9\n"  # captured stdout is not a tty

    @patch("sys.argv", ["conductress", "--help"])
    def test_help_lists_the_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main()
        assert "--version" in capsys.readouterr().out
