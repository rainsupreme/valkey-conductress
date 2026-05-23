"""Unit tests for ProfilingManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.profiling_manager import ProfilingManager


@pytest.fixture
def mock_host():
    host = MagicMock()
    host.ip = "127.0.0.1"
    host.run_host_command = AsyncMock(return_value=("", ""))
    host.get_remote_file = AsyncMock()
    return host


@pytest.fixture
def manager(mock_host):
    return ProfilingManager(mock_host)


class TestProfilingStart:
    def test_raises_if_already_profiling(self, manager):
        import threading
        import time

        # Keep thread alive so is_profiling() returns True
        event = threading.Event()
        manager._profiling_thread = threading.Thread(target=event.wait)
        manager._profiling_thread.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                manager.profiling_start(99)
        finally:
            event.set()
            manager._profiling_thread.join()

    def test_not_profiling_initially(self, manager):
        assert not manager.is_profiling()


class TestProfilingStop:
    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self, manager):
        # Should not raise
        await manager.profiling_stop()

    @pytest.mark.asyncio
    async def test_stop_removes_status_file(self, manager, mock_host):
        with patch.object(manager, "_profiling_run_sync"):
            manager.profiling_start(99)
        await manager.profiling_stop()
        mock_host.run_host_command.assert_called_with("rm -f /tmp/profiling_running")


class TestPerfStatStart:
    @pytest.mark.asyncio
    async def test_raises_if_already_running(self, manager, mock_host):
        import threading

        event = threading.Event()
        manager._perf_stat_thread = threading.Thread(target=event.wait)
        manager._perf_stat_thread.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await manager.perf_stat_start()
        finally:
            event.set()
            manager._perf_stat_thread.join()

    @pytest.mark.asyncio
    async def test_touches_status_file(self, manager, mock_host):
        with patch.object(manager, "_perf_stat_run_sync"):
            await manager.perf_stat_start()
        mock_host.run_host_command.assert_any_call("touch /tmp/perf_stat_running")


class TestPerfStatStop:
    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self, manager):
        await manager.perf_stat_stop()

    @pytest.mark.asyncio
    async def test_stop_removes_status_file(self, manager, mock_host):
        with patch.object(manager, "_perf_stat_run_sync"):
            await manager.perf_stat_start()
        await manager.perf_stat_stop()
        mock_host.run_host_command.assert_called_with("rm -f /tmp/perf_stat_running")


class TestPerfStatReport:
    @pytest.mark.asyncio
    async def test_copies_file_and_parses(self, manager, mock_host, tmp_path):
        result_dir = tmp_path / "results"
        result_dir.mkdir()

        # Create a fake perf stat file that get_remote_file will "produce"
        async def fake_get_remote(src, dest):
            dest.write_text("     1000      instructions:u\n     500      cycles:u\n")

        mock_host.get_remote_file = fake_get_remote

        result = await manager.perf_stat_report(result_dir)
        assert result["instructions"] == 1000
        assert result["cycles"] == 500

    @pytest.mark.asyncio
    async def test_raises_if_result_dir_missing(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            await manager.perf_stat_report(tmp_path / "nonexistent")


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_artifacts(self, manager, mock_host):
        await manager.cleanup()
        call_args = mock_host.run_host_command.call_args[0][0]
        assert "perf.data" in call_args
        assert "flamegraph.svg" in call_args
        assert "perf_stat_output" in call_args
