"""Unit tests for StabilizationManager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from conductress.stabilization_manager import StabilizationManager


@pytest.fixture
def mock_host():
    host = MagicMock()
    host.ip = "127.0.0.1"
    host.run_host_command = AsyncMock(return_value=("", ""))
    host.check_file_exists = AsyncMock(return_value=False)
    return host


@pytest.fixture
def manager(mock_host):
    return StabilizationManager(mock_host)


class TestVerify:
    @pytest.mark.asyncio
    async def test_verify_passes_when_aslr_disabled(self, manager, mock_host):
        mock_host.run_host_command = AsyncMock(return_value=("0", ""))
        mock_host.check_file_exists = AsyncMock(return_value=False)

        result = await manager.verify()
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_fails_when_aslr_enabled(self, manager, mock_host):
        # First call returns wrong value, retry also returns wrong value
        mock_host.run_host_command = AsyncMock(return_value=("2", ""))
        mock_host.check_file_exists = AsyncMock(return_value=False)

        result = await manager.verify()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_checks_governor_when_cpufreq_exists(self, manager, mock_host):
        call_count = [0]

        async def mock_run(cmd, check=True):
            call_count[0] += 1
            if "randomize_va_space" in cmd:
                return ("0", "")
            if "scaling_governor" in cmd:
                return ("performance", "")
            return ("", "")

        async def mock_exists(path):
            return "scaling_available_governors" in str(path)

        mock_host.run_host_command = mock_run
        mock_host.check_file_exists = mock_exists

        result = await manager.verify()
        assert result is True


class TestEnable:
    @pytest.mark.asyncio
    async def test_enable_disables_aslr(self, manager, mock_host):
        await manager.enable()
        calls = [str(c) for c in mock_host.run_host_command.call_args_list]
        assert any("randomize_va_space" in c for c in calls)

    @pytest.mark.asyncio
    async def test_enable_sets_platform_info(self, manager, mock_host):
        # detect_platform needs lscpu output
        mock_host.run_host_command = AsyncMock(return_value=("Architecture: aarch64\n", ""))
        mock_host.check_file_exists = AsyncMock(return_value=False)

        await manager.enable()
        assert manager.platform_info is not None


class TestDisable:
    @pytest.mark.asyncio
    async def test_disable_restores_aslr(self, manager, mock_host):
        await manager.disable()
        calls = [str(c) for c in mock_host.run_host_command.call_args_list]
        assert any("randomize_va_space" in c and "2" in c for c in calls)
