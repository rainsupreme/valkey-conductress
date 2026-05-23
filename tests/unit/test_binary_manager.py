"""Tests for BinaryManager build caching and git operations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from conductress.binary_manager import BinaryManager


@pytest.fixture
def mock_host():
    host = MagicMock()
    host.ip = "127.0.0.1"
    host.run_host_command = AsyncMock(return_value=("", ""))
    host.check_file_exists = AsyncMock(return_value=False)
    host.put_remote_file = AsyncMock()
    return host


@pytest.fixture
def manager(mock_host):
    mgr = BinaryManager(mock_host)
    mgr.source = "valkey"
    mgr.specifier = "unstable"
    mgr.make_args = "USE_FAST_FLOAT=yes"
    return mgr


class TestCacheHitSkipsBuild:
    """When binary is already cached, no build should be triggered."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_path_without_building(self, manager, mock_host):
        mock_host.check_file_exists = AsyncMock(return_value=True)
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("refs/remotes/origin/unstable\n", ""),  # rev-parse
                ("", ""),  # git reset --hard
                ("abc123def456\n", ""),  # git rev-parse HEAD
            ]
        )

        result = await manager._ensure_build_cached()

        # Should NOT have called make
        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        assert not any("make" in cmd for cmd in commands)
        assert "abc123def456" in str(result)

    @pytest.mark.asyncio
    async def test_cache_miss_triggers_build(self, manager, mock_host):
        mock_host.check_file_exists = AsyncMock(return_value=False)
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("refs/remotes/origin/unstable\n", ""),  # rev-parse
                ("", ""),  # git reset --hard
                ("abc123\n", ""),  # git rev-parse HEAD
                ("", ""),  # make distclean && make -j
                ("", ""),  # mkdir -p
                ("", ""),  # cp binary
            ]
        )

        await manager._ensure_build_cached()

        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        assert any("make" in cmd for cmd in commands)
        assert any("mkdir" in cmd for cmd in commands)
        assert any("cp" in cmd for cmd in commands)


class TestBuildFailureHandling:
    @pytest.mark.asyncio
    async def test_build_failure_propagates_exception(self, manager, mock_host):
        mock_host.check_file_exists = AsyncMock(return_value=False)

        error = asyncssh.ProcessError(None, "make", None, 1, None, 1, "", "compilation error")
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("--\n", ""),  # rev-parse (commit hash)
                ("", ""),  # git reset --hard
                ("abc123\n", ""),  # git rev-parse HEAD
                error,  # make fails
            ]
        )

        with pytest.raises(asyncssh.ProcessError):
            await manager._ensure_build_cached()


class TestSpecifierNormalization:
    @pytest.mark.asyncio
    async def test_branch_name_prefixed_with_origin(self, manager, mock_host):
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("refs/remotes/origin/unstable\n", ""),  # rev-parse
            ]
        )

        result = await manager._normalize_specifier("unstable")
        assert result == "origin/unstable"

    @pytest.mark.asyncio
    async def test_commit_hash_used_as_is(self, manager, mock_host):
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("--\n", ""),  # rev-parse returns -- for raw hashes
            ]
        )

        result = await manager._normalize_specifier("abc123def")
        assert result == "abc123def"

    @pytest.mark.asyncio
    async def test_invalid_specifier_raises(self, manager, mock_host):
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("\n", ""),  # empty result
            ]
        )

        with pytest.raises(ValueError, match="invalid specifier"):
            await manager._normalize_specifier("nonexistent")


class TestEnsureBinaryCached:
    @pytest.mark.asyncio
    async def test_unknown_source_raises(self, manager, mock_host):
        manager.source = "unknown_repo"

        with pytest.raises(ValueError, match="Unknown source"):
            await manager.ensure_binary_cached()

    @pytest.mark.asyncio
    async def test_updates_state_from_args(self, mock_host, monkeypatch):
        import conductress.config as config

        monkeypatch.setattr(config, "REPO_NAMES", ["valkey", "rainsupreme"])

        mgr = BinaryManager(mock_host)

        mock_host.check_file_exists = AsyncMock(return_value=True)
        mock_host.run_host_command = AsyncMock(
            side_effect=[
                ("", ""),  # git fetch
                ("refs/remotes/origin/main\n", ""),  # rev-parse
                ("", ""),  # git reset
                ("deadbeef\n", ""),  # rev-parse HEAD
            ]
        )

        await mgr.ensure_binary_cached(source="valkey", specifier="main", make_args="")

        assert mgr.source == "valkey"
        assert mgr.specifier == "main"
        assert mgr.make_args == ""
        assert mgr.hash == "deadbeef"


class TestMakeArgsAffectCacheKey:
    def test_different_make_args_produce_different_paths(self, mock_host):
        mgr1 = BinaryManager(mock_host)
        mgr1.source = "valkey"
        mgr1.hash = "abc123"
        mgr1.make_args = "USE_FAST_FLOAT=yes"

        mgr2 = BinaryManager(mock_host)
        mgr2.source = "valkey"
        mgr2.hash = "abc123"
        mgr2.make_args = ""

        assert mgr1.get_cached_build_path() != mgr2.get_cached_build_path()

    def test_empty_make_args_is_valid_cache_key(self, mock_host):
        mgr = BinaryManager(mock_host)
        mgr.source = "valkey"
        mgr.hash = "abc123"
        mgr.make_args = ""

        # Should not raise
        path = mgr.get_cached_build_path()
        assert "abc123" in str(path)
