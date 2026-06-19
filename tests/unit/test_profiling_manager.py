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


class TestCpuProfileStart:
    def test_raises_if_already_running(self, manager):
        import threading

        event = threading.Event()
        manager._cpu_profile_thread = threading.Thread(target=event.wait)
        manager._cpu_profile_thread.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                manager.cpu_profile_start(30)
        finally:
            event.set()
            manager._cpu_profile_thread.join()

    def test_no_thread_initially(self, manager):
        assert manager._cpu_profile_thread is None


class TestCpuProfileCollect:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_main_thread(self, manager, mock_host):
        """If we can't identify the main thread, return empty."""
        mock_host.run_host_command = AsyncMock(return_value=("", ""))
        manager._target_pid = 12345
        main, io = await manager.cpu_profile_collect()
        assert main == []
        assert io == []

    @pytest.mark.asyncio
    async def test_parses_collapsed_stacks(self, manager, mock_host):
        """Test that collapsed stacks output is correctly parsed."""
        manager._target_pid = 12345

        # First call: discover TIDs
        tid_output = "12345 valkey-server\n12346 io_thd_1\n12347 io_thd_2\n"
        # Second call: main thread stacks
        main_output = "func_a;func_b;hashtableFind 500\nfunc_a;func_c;zmalloc 200\n"
        # Third call: IO thread stacks
        io_output = "io_thd_1;IOThreadMain;pthread_mutex_lock 1000\n"

        call_count = [0]

        async def mock_run(cmd):
            call_count[0] += 1
            if call_count[0] == 1:
                return (tid_output, "")
            elif call_count[0] == 2:  # write main stacks to file
                return ("", "")
            elif call_count[0] == 3:  # cat main stacks file
                return (main_output, "")
            elif call_count[0] == 4:  # write io stacks to file
                return ("", "")
            elif call_count[0] == 5:  # cat io stacks file
                return (io_output, "")
            return ("", "")

        mock_host.run_host_command = mock_run

        main, io = await manager.cpu_profile_collect()
        assert len(main) == 2
        assert main[0] == ["func_a;func_b;hashtableFind", 500]
        assert main[1] == ["func_a;func_c;zmalloc", 200]
        assert len(io) == 1
        assert io[0] == ["io_thd_1;IOThreadMain;pthread_mutex_lock", 1000]


class TestParseCollapsed:
    def test_basic_parsing(self):
        output = "a;b;c 100\nd;e;f 200\n"
        result = ProfilingManager._parse_collapsed(output)
        assert result == [["a;b;c", 100], ["d;e;f", 200]]

    def test_empty_input(self):
        assert ProfilingManager._parse_collapsed("") == []
        assert ProfilingManager._parse_collapsed("\n\n") == []

    def test_invalid_lines_skipped(self):
        output = "valid;stack 100\ninvalid line without count\nanother;valid 50\n"
        result = ProfilingManager._parse_collapsed(output)
        assert len(result) == 2


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
        assert "perf-cpu-profile.data" in call_args
        assert "perf_stat_output" in call_args
