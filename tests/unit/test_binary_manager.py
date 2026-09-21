"""Tests for BinaryManager build caching and git operations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from conductress.binary_manager import BinaryManager, _rpath_names_lua_module


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
    mgr.make_args = "OPTIMIZATION=-O2"
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
                ("", ""),  # readelf -d (no RPATH: binary needs no module)
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
                ("", ""),  # readelf -d (no RPATH: binary needs no module)
            ]
        )

        await mgr.ensure_binary_cached(source="valkey", specifier="main", make_args="")

        assert mgr.source == "valkey"
        assert mgr.specifier == "main"
        assert mgr.make_args == ""
        assert mgr.hash == "deadbeef"


class TestLuaModuleCaching:
    """Regression tests for the missing-libvalkeylua.so build-cache bug.

    Module-era valkey commits (valkey-io#2858..#3392) dlopen libvalkeylua.so
    at startup and SIGABRT if it is missing. The cache used to store only the
    server binary, so every cached module-era commit boot-crashed in the
    perf-sweep (280 coredumps, Jul-Aug 2026). Worse, the binary's DT_RPATH
    points into the shared source tree, so a leftover tree .so from a
    different commit would load silently.

    Pre-existing half-cached entries (binary present, module absent) are
    detected at cache-hit time: the binary's DT_RPATH says whether it dlopens
    the module, and an entry missing a required module is discarded and
    rebuilt. Correctness for new entries relies on write ordering: runtime
    artifacts are cached before the binary, whose presence is the cache-hit
    marker.
    """

    @pytest.mark.asyncio
    async def test_module_artifact_cached_and_removed_from_tree(self, manager, mock_host):
        """After a build that produced libvalkeylua.so, the module must be copied
        into the cache dir and deleted from the shared source tree."""

        async def check(path):
            s = str(path)
            if s.endswith("modules/lua/libvalkeylua.so"):
                return True  # build produced the module in the tree
            if "build_cache" in s:
                return False  # nothing cached yet -> build
            return True  # tree build output exists

        mock_host.check_file_exists = AsyncMock(side_effect=check)
        mock_host.run_host_command = AsyncMock(return_value=("abc123\n", ""))

        await manager._ensure_build_cached()

        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        module_cmds = [cmd for cmd in commands if "libvalkeylua.so" in cmd and "cp " in cmd]
        assert module_cmds, f"module artifact was not cached; commands: {commands}"
        assert any("rm -f" in cmd for cmd in module_cmds), (
            "tree copy of libvalkeylua.so must be removed after caching "
            "(DT_RPATH would silently load a wrong-commit module otherwise)"
        )

    @pytest.mark.asyncio
    async def test_module_cached_before_binary(self, manager, mock_host):
        """The binary must be the LAST artifact written to the cache entry.

        Its presence is the cache-hit marker: writing it last guarantees an
        interrupted build reads as a cache miss, never as a binary without
        its module (which would boot-crash on every future cache hit)."""

        async def check(path):
            s = str(path)
            if s.endswith("modules/lua/libvalkeylua.so"):
                return True
            if "build_cache" in s:
                return False
            return True

        mock_host.check_file_exists = AsyncMock(side_effect=check)
        mock_host.run_host_command = AsyncMock(return_value=("abc123\n", ""))

        await manager._ensure_build_cached()

        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        module_idx = next(i for i, c in enumerate(commands) if "libvalkeylua.so" in c and "cp " in c)
        binary_idx = next(i for i, c in enumerate(commands) if c.startswith("cp ") and "libvalkeylua.so" not in c)
        assert module_idx < binary_idx, (
            f"module must be cached before the binary (module at {module_idx}, " f"binary at {binary_idx}): {commands}"
        )


class TestRpathNamesLuaModule:
    """The pure parser behind the cache-hit predicate, fed real readelf shapes."""

    MODULE_ERA = (
        "Dynamic section at offset 0x3f1c58 contains 30 entries:\n"
        "  Tag        Type                         Name/Value\n"
        " 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]\n"
        " 0x000000000000000f (RPATH)              Library rpath: "
        "[/usr/local/lib:/home/ec2-user/valkey/src/modules/lua]\n"
        " 0x000000000000000c (INIT)               0x5c000\n"
    )
    NO_RPATH = (
        "Dynamic section at offset 0x3e2c30 contains 28 entries:\n"
        "  Tag        Type                         Name/Value\n"
        " 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]\n"
        " 0x000000000000000c (INIT)               0x5b000\n"
    )

    def test_module_era_rpath_detected(self):
        assert _rpath_names_lua_module(self.MODULE_ERA) is True

    def test_runpath_variant_detected(self):
        assert _rpath_names_lua_module(self.MODULE_ERA.replace("(RPATH)", "(RUNPATH)")) is True

    def test_trailing_slash_in_rpath_entry_detected(self):
        assert _rpath_names_lua_module(self.MODULE_ERA.replace("modules/lua]", "modules/lua/]")) is True

    def test_no_rpath_is_false(self):
        assert _rpath_names_lua_module(self.NO_RPATH) is False

    def test_rpath_without_module_dir_is_false(self):
        out = self.MODULE_ERA.replace(":/home/ec2-user/valkey/src/modules/lua", "")
        assert _rpath_names_lua_module(out) is False

    def test_module_dir_outside_rpath_line_does_not_count(self):
        out = self.NO_RPATH + " 0x0000000000000001 (NEEDED)             Shared library: [modules/lua]\n"
        assert _rpath_names_lua_module(out) is False

    def test_empty_output_is_false(self):
        assert _rpath_names_lua_module("") is False


class TestCacheHitRequiresRuntimeArtifacts:
    """A cached binary counts as a hit only with every artifact it needs.

    Entries holding a module-era binary without libvalkeylua.so were written
    before the module was cached beside the binary; they boot-crash on every
    launch. The predicate must discard them and report a miss so the normal
    build path rebuilds the entry.
    """

    MODULE_ERA_READELF = TestRpathNamesLuaModule.MODULE_ERA
    NO_RPATH_READELF = TestRpathNamesLuaModule.NO_RPATH

    @staticmethod
    def _exists(binary: bool, module: bool):
        async def check(path):
            s = str(path)
            if s.endswith("libvalkeylua.so"):
                return module
            return binary

        return check

    @pytest.mark.asyncio
    async def test_module_era_binary_with_module_is_a_hit(self, manager, mock_host):
        manager.hash = "abc123"
        mock_host.check_file_exists = AsyncMock(side_effect=self._exists(binary=True, module=True))
        mock_host.run_host_command = AsyncMock(return_value=(self.MODULE_ERA_READELF, ""))

        assert await manager._is_binary_cached() is True
        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        assert not any("rm -rf" in cmd for cmd in commands)

    @pytest.mark.asyncio
    async def test_module_era_binary_without_module_is_discarded(self, manager, mock_host):
        manager.hash = "abc123"
        mock_host.check_file_exists = AsyncMock(side_effect=self._exists(binary=True, module=False))
        mock_host.run_host_command = AsyncMock(return_value=(self.MODULE_ERA_READELF, ""))

        assert await manager._is_binary_cached() is False
        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        removed = [cmd for cmd in commands if cmd.startswith("rm -rf ")]
        assert removed == [f"rm -rf {manager.get_cached_build_path()}"], commands

    @pytest.mark.asyncio
    async def test_binary_without_rpath_is_a_hit_regardless_of_module(self, manager, mock_host):
        manager.hash = "abc123"
        mock_host.check_file_exists = AsyncMock(side_effect=self._exists(binary=True, module=False))
        mock_host.run_host_command = AsyncMock(return_value=(self.NO_RPATH_READELF, ""))

        assert await manager._is_binary_cached() is True
        module_checks = [
            c for c in mock_host.check_file_exists.call_args_list if str(c[0][0]).endswith("libvalkeylua.so")
        ]
        assert not module_checks, "a binary that needs no module must not be judged on one"

    @pytest.mark.asyncio
    async def test_missing_binary_is_a_miss_without_reading_elf(self, manager, mock_host):
        manager.hash = "abc123"
        mock_host.check_file_exists = AsyncMock(return_value=False)
        mock_host.run_host_command = AsyncMock(return_value=("", ""))

        assert await manager._is_binary_cached() is False
        mock_host.run_host_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreadable_elf_is_treated_as_needing_nothing(self, manager, mock_host):
        manager.hash = "abc123"
        mock_host.check_file_exists = AsyncMock(side_effect=self._exists(binary=True, module=False))
        error = asyncssh.ProcessError(None, "readelf", None, 127, None, 127, "", "readelf: not found")
        mock_host.run_host_command = AsyncMock(side_effect=error)

        assert await manager._is_binary_cached() is True

    @pytest.mark.asyncio
    async def test_poisoned_entry_triggers_rebuild(self, manager, mock_host):
        """End to end through _ensure_build_cached: a poisoned entry is removed,
        then rebuilt with the module cached before the binary."""

        async def check(path):
            s = str(path)
            if s.endswith("modules/lua/libvalkeylua.so"):
                return True  # rebuilt tree produced the module
            if "build_cache" in s and s.endswith("libvalkeylua.so"):
                return False  # poisoned entry: no module
            if "build_cache" in s:
                return True  # poisoned entry: binary present
            return True

        async def run(command, check=True):
            if command.startswith("readelf"):
                return (self.MODULE_ERA_READELF, "")
            if "rev-parse --symbolic-full-name" in command:
                return ("refs/remotes/origin/unstable\n", "")
            return ("abc123\n", "")

        mock_host.check_file_exists = AsyncMock(side_effect=check)
        mock_host.run_host_command = AsyncMock(side_effect=run)

        await manager._ensure_build_cached()

        commands = [call[0][0] for call in mock_host.run_host_command.call_args_list]
        rm_idx = next(i for i, c in enumerate(commands) if c.startswith("rm -rf "))
        make_idx = next(i for i, c in enumerate(commands) if "make -j" in c)
        module_idx = next(i for i, c in enumerate(commands) if "libvalkeylua.so" in c and "cp " in c)
        binary_idx = next(i for i, c in enumerate(commands) if c.startswith("cp ") and "libvalkeylua.so" not in c)
        assert rm_idx < make_idx < module_idx < binary_idx, commands


class TestMakeArgsAffectCacheKey:
    def test_different_make_args_produce_different_paths(self, mock_host):
        mgr1 = BinaryManager(mock_host)
        mgr1.source = "valkey"
        mgr1.hash = "abc123"
        mgr1.make_args = "OPTIMIZATION=-O2"

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
