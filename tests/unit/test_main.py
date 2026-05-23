"""Unit tests for src/__main__.py subcommand dispatch.

Validates: Requirements 9.6
"""

from unittest.mock import MagicMock, patch

import pytest

from src.__main__ import main


class TestTuiSubcommand:
    """Test that the 'tui' subcommand dispatches to src.tui.BenchmarkApp."""

    @patch("sys.argv", ["conductress", "tui"])
    @patch("src.__main__.logging")
    def test_tui_dispatches_to_benchmark_app(self, mock_logging):
        mock_app = MagicMock()
        with patch("src.tui.BenchmarkApp", return_value=mock_app) as mock_cls:
            main()
            mock_cls.assert_called_once()
            mock_app.run.assert_called_once()


class TestRunSubcommand:
    """Test that the 'run' subcommand dispatches to src.task_runner.TaskRunner."""

    @patch("sys.argv", ["conductress", "run"])
    @patch("src.__main__.logging")
    def test_run_dispatches_to_task_runner(self, mock_logging):
        mock_runner = MagicMock()
        with (
            patch("src.task_runner.TaskRunner", return_value=mock_runner) as mock_cls,
            patch("asyncio.run") as mock_asyncio_run,
        ):
            main()
            mock_cls.assert_called_once()
            mock_asyncio_run.assert_called_once_with(mock_runner.run())


class TestSetupSubcommand:
    """Test that the 'setup' subcommand dispatches to src.bootstrap functions."""

    @patch("sys.argv", ["conductress", "setup"])
    @patch("src.__main__.logging")
    def test_setup_dispatches_to_bootstrap(self, mock_logging):
        with (
            patch("src.bootstrap.ensure_ssh_key") as mock_ssh_key,
            patch("src.bootstrap.ensure_server_ssh_fingerprints") as mock_fingerprints,
            patch("src.bootstrap.update_host_list") as mock_update,
            patch("src.bootstrap.SERVERS", [MagicMock()]),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            main()
            mock_ssh_key.assert_called_once()
            # asyncio.run is called twice: once for fingerprints, once for update_host_list
            assert mock_asyncio_run.call_count == 2


class TestQueueSubcommand:
    """Test that the 'queue' subcommand dispatches to src.cli.main()."""

    @patch("sys.argv", ["conductress", "queue"])
    @patch("src.__main__.logging")
    def test_queue_dispatches_to_cli_main(self, mock_logging):
        with patch("src.cli.main", return_value=0) as mock_cli_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_cli_main.assert_called_once_with(["queue"])

    @patch("sys.argv", ["conductress", "queue"])
    @patch("src.__main__.logging")
    def test_queue_propagates_nonzero_exit_code(self, mock_logging):
        with patch("src.cli.main", return_value=1) as mock_cli_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestCompareSubcommand:
    """Test that the 'compare' subcommand dispatches to src.analysis.main()."""

    @patch("sys.argv", ["conductress", "compare", "branch-a", "branch-b"])
    @patch("src.__main__.logging")
    def test_compare_dispatches_to_analysis_main(self, mock_logging):
        with patch("src.analysis.main", return_value=0) as mock_analysis_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_analysis_main.assert_called_once_with(["branch-a", "branch-b"])

    @patch("sys.argv", ["conductress", "compare"])
    @patch("src.__main__.logging")
    def test_compare_dispatches_with_no_extra_args(self, mock_logging):
        with patch("src.analysis.main", return_value=0) as mock_analysis_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_analysis_main.assert_called_once_with([])

    @patch(
        "sys.argv",
        ["conductress", "compare", "branch-a", "branch-b", "--source", "valkey"],
    )
    @patch("src.__main__.logging")
    def test_compare_passes_remaining_args(self, mock_logging):
        with patch("src.analysis.main", return_value=0) as mock_analysis_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_analysis_main.assert_called_once_with(["branch-a", "branch-b", "--source", "valkey"])

    @patch("sys.argv", ["conductress", "compare", "branch-a", "branch-b"])
    @patch("src.__main__.logging")
    def test_compare_propagates_nonzero_exit_code(self, mock_logging):
        with patch("src.analysis.main", return_value=1) as mock_analysis_main:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestNoSubcommand:
    """Test that invoking without a subcommand prints usage information."""

    @patch("sys.argv", ["conductress"])
    @patch("src.__main__.logging")
    def test_no_subcommand_prints_usage(self, mock_logging, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "Usage:" in captured.out
