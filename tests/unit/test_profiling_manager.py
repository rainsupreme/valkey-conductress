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
        manager._main_tid = "12345"
        manager._io_tids = ["12346", "12347"]

        main_output = "func_a;func_b;hashtableFind 500\nfunc_a;func_c;zmalloc 200\n"
        io_output = "io_thd_1;IOThreadMain;pthread_mutex_lock 1000\n"

        async def mock_run(cmd):
            if cmd.startswith("stat -c %s"):
                return ("4096000\n", "")
            if cmd.startswith("cat ") and cmd.endswith("collapsed-main.txt"):
                return (main_output, "")
            if cmd.startswith("cat ") and cmd.endswith("collapsed-io.txt"):
                return (io_output, "")
            return ("", "")

        mock_host.run_host_command = mock_run

        main, io = await manager.cpu_profile_collect()
        assert len(main) == 2
        assert main[0] == ["func_a;func_b;hashtableFind", 500]
        assert main[1] == ["func_a;func_c;zmalloc", 200]
        assert len(io) == 1
        assert io[0] == ["io_thd_1;IOThreadMain;pthread_mutex_lock", 1000]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stat_output", ["0\n", "missing\n", ""])
    async def test_empty_or_missing_recording_is_an_error_not_an_empty_profile(
        self, manager, mock_host, stat_output, caplog
    ):
        """A 0-byte or absent perf.data (e.g. full filesystem) must not collapse silently to []."""
        manager._target_pid = 12345
        manager._main_tid = "12345"
        manager._io_tids = ["12346"]
        manager._cpu_profile_returncode = 255
        commands: list[str] = []

        async def mock_run(cmd):
            commands.append(cmd)
            if cmd.startswith("stat -c %s"):
                return (stat_output, "")
            return ("", "")

        mock_host.run_host_command = mock_run

        with caplog.at_level("ERROR", logger="conductress.profiling_manager"):
            main, io = await manager.cpu_profile_collect()

        assert (main, io) == ([], [])
        assert not any("perf script" in c for c in commands), "must not script an empty recording"
        assert any(c.startswith("sudo rm -f") for c in commands), "artifacts are still cleaned up"
        assert any("CPU profile produced no data" in r.message and "255" in r.message for r in caplog.records)

    def test_recording_lives_under_home_not_tmp(self, manager):
        """perf.data and the collapsed stacks are written under ~, not the tmpfs-backed /tmp."""
        from conductress import profiling_manager as pm

        for path in (pm.CPU_PROFILE_DATA, pm.CPU_PROFILE_COLLAPSED_MAIN, pm.CPU_PROFILE_COLLAPSED_IO):
            assert path.startswith("~/"), path
            assert not path.startswith("/tmp"), path

    def test_record_command_creates_directory_and_keeps_exit_status(self, manager):
        """The recorder mkdir -p's its directory first and records perf's exit status."""
        manager._target_pid = 4242
        launched: dict = {}

        class FakeProc:
            returncode = 7

            def wait(self):
                return 7

            def terminate(self):
                pass

        def fake_popen(cmd, *args, **kwargs):
            launched["cmd"] = cmd
            return FakeProc()

        with patch("conductress.profiling_manager.subprocess.Popen", side_effect=fake_popen):
            manager._cpu_profile_run_sync(duration=10)

        from conductress import profiling_manager as pm

        assert launched["cmd"].startswith(f"mkdir -p {pm.CPU_PROFILE_DIR} && exec sudo perf record")
        assert f"-o {pm.CPU_PROFILE_DATA}" in launched["cmd"]
        assert manager._cpu_profile_returncode == 7


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

        # Per-thread output: main TID = 100, IO TIDs = 101, 102.
        manager._main_tid = "100"
        manager._io_tids = ["101", "102"]

        async def fake_get_remote(src, dest):
            dest.write_text(
                "   valkey-server-100     1000      instructions\n"
                "        io_thd_1-101      400      instructions\n"
                "        io_thd_2-102      600      instructions\n"
                "   valkey-server-100      500      cycles\n"
                "        io_thd_1-101      250      cycles\n"
                "        io_thd_2-102      250      cycles\n"
            )

        mock_host.get_remote_file = fake_get_remote

        result = await manager.perf_stat_report(result_dir)
        # Process-wide total (all monitored TIDs summed)
        assert result["all"]["instructions"] == 2000
        assert result["all"]["cycles"] == 1000
        # Main thread only
        assert result["main"]["instructions"] == 1000
        assert result["main"]["cycles"] == 500
        # IO threads summed
        assert result["io"]["instructions"] == 1000
        assert result["io"]["cycles"] == 500

    @pytest.mark.asyncio
    async def test_records_counting_scope(self, manager, mock_host, tmp_path):
        result_dir = tmp_path / "results"
        result_dir.mkdir()
        manager._main_tid = "100"
        manager._io_tids = []
        assert manager.perf_stat_scope is None

        async def fake_get_remote(src, dest):
            dest.write_text("   valkey-server-100     1000      instructions:u\n")

        mock_host.get_remote_file = fake_get_remote
        result = await manager.perf_stat_report(result_dir)
        # Event name is normalised, the scope it came from is kept alongside.
        assert result["main"] == {"instructions": 1000}
        assert manager.perf_stat_scope == "user"

        async def fake_get_remote_full(src, dest):
            dest.write_text("   valkey-server-100     1000      instructions\n")

        mock_host.get_remote_file = fake_get_remote_full
        await manager.perf_stat_report(result_dir)
        assert manager.perf_stat_scope == "user+kernel"

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
