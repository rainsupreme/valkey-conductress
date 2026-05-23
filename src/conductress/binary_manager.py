"""Binary management: build caching, git operations, and binary provisioning.

Handles fetching source, building Valkey binaries, and caching them on
remote hosts. Separated from Server to isolate build logic from runtime
server management.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

import asyncssh

from . import config
from .ssh_host import SshHost
from .utility import async_run

VALKEY_BINARY = "valkey-server"

logger = logging.getLogger(__name__)


class BinaryManager:
    """Manages Valkey binary builds and caching on a remote host.

    Uses an SshHost for remote command execution and file transfer.
    Maintains state about the current source, specifier, and build hash.
    """

    # Remote paths for build cache
    path_root = Path("~")
    remote_build_cache = path_root / "build_cache"

    def __init__(self, host: SshHost) -> None:
        self._host = host
        self._logger = logging.getLogger(f"{self.__class__.__name__}.{host.ip}")

        self.source: Optional[str] = None
        self.specifier: Optional[str] = None
        self.hash: Optional[str] = None
        self.make_args: str = config.DEFAULT_MAKE_ARGS

    async def ensure_binary_cached(
        self,
        source: Optional[str] = None,
        specifier: Optional[str] = None,
        make_args: Optional[str] = None,
    ) -> Path:
        """Ensure a binary is built and cached. Returns path to cached binary.

        This needs to be done before running multiple instances on a single host.
        """
        if source:
            self.source = source
        if specifier:
            self.specifier = specifier
        if make_args is not None:
            self.make_args = make_args

        if self.source == config.MANUALLY_UPLOADED:
            return await self._ensure_binary_uploaded(self.specifier)
        else:
            if self.source not in config.REPO_NAMES:
                raise ValueError(f"Unknown source: {self.source}")
            return await self._ensure_build_cached()

    def get_build_hash(self) -> Optional[str]:
        """Get the commit hash of the current build."""
        return self.hash

    def get_cached_build_path(self) -> Path:
        """Get the path to the cached binary directory."""
        if self.source is None or self.hash is None:
            raise RuntimeError("source and hash must be set before accessing cached build path")
        make_args_hash = hashlib.md5(self.make_args.encode()).hexdigest()[:16]
        return self.remote_build_cache / self.source / self.hash / make_args_hash

    def get_source_binary_path(self) -> Path:
        """Get the path to the source repo's build output."""
        if self.source is None:
            raise RuntimeError("source must be set before accessing source binary path")
        return self.path_root / self.source / "src"

    async def _is_binary_cached(self) -> bool:
        return await self._host.check_file_exists(self.get_cached_build_path() / VALKEY_BINARY)

    async def _normalize_specifier(self, specifier: Optional[str]) -> str:
        """Resolve a specifier to a valid git ref. Fetches from origin first."""
        source_path = self.get_source_binary_path()
        await self._host.run_host_command(f"cd {source_path} && git fetch --quiet --prune")
        try:
            result, _ = await self._host.run_host_command(
                f"cd {source_path} && git rev-parse --symbolic-full-name origin/{specifier} --"
            )
        except asyncssh.ProcessError:
            self._logger.info("Failed to resolve %s, trying as-is", specifier)
            result, _ = await self._host.run_host_command(
                f"cd {source_path} && git rev-parse --symbolic-full-name {specifier} --"
            )
        result = result.strip()
        if result == "":
            raise ValueError(f"{specifier} is an invalid specifier in {self.source} (empty result)")
        if result == "--":
            return specifier  # type: ignore[return-value]
        if result.startswith("refs/remotes/origin/"):
            return f"origin/{specifier}"
        return specifier  # type: ignore[return-value]

    async def _ensure_build_cached(self) -> Path:
        """Build from source if not already cached."""
        source_path = self.get_source_binary_path()
        sync_target = await self._normalize_specifier(self.specifier)
        await self._host.run_host_command(f"cd {source_path} && git reset --hard {sync_target}")
        self.hash = await self._get_current_commit_hash()

        cached_build_path = self.get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not await self._is_binary_cached():
            self._logger.info("building %s:%s...", self.source, self.specifier)
            try:
                make_command = f"cd {source_path}; rm -f valkey-server redis-server; make distclean && make -j"
                if self.make_args:
                    make_command += f" {self.make_args}"
                await self._host.run_host_command(make_command)
            except asyncssh.ProcessError as e:
                self._logger.error("Build failed %d:\n%s", e.returncode, e.stderr)
                raise
            build_binary = source_path / VALKEY_BINARY
            if not await self._host.check_file_exists(build_binary):
                build_binary = source_path / "redis-server"  # pre-fork commits
            await self._host.run_host_command(f"mkdir -p {cached_build_path}")
            await self._host.run_host_command(f"cp {build_binary} {cached_binary_path}")

        return cached_binary_path

    async def _ensure_binary_uploaded(self, local_path: Optional[str]) -> Path:
        """Upload a local binary to the remote cache."""
        out, _ = await async_run(f"sha1sum {str(local_path)}")
        if not out:
            raise RuntimeError("Failed to run sha1sum on local binary")
        self.hash = out.strip().split()[0]

        cached_binary_path = self.get_cached_build_path() / VALKEY_BINARY

        if not await self._is_binary_cached():
            self._logger.info("copying %s to server... (not cached)", local_path)
            await self._host.run_host_command(f"mkdir -p {cached_binary_path.parent}")
            await self._host.put_remote_file(Path(str(local_path)), cached_binary_path)

        return cached_binary_path

    async def _get_current_commit_hash(self) -> str:
        """Get HEAD commit hash from the source repo."""
        source_path = self.get_source_binary_path()
        out, _ = await self._host.run_host_command(f"cd {source_path}; git rev-parse HEAD")
        return out.strip()

    @staticmethod
    async def delete_entire_build_cache(server_ips: list[str]) -> None:
        """Delete the entire build cache on all servers. Destructive operation."""

        async def delete_host_cache(ip: str) -> None:
            async with asyncssh.connect(ip, client_keys=[str(config.SSH_KEYFILE)]) as conn:
                await conn.run(f"rm -rf {BinaryManager.remote_build_cache}", check=False)

        await asyncio.gather(*(delete_host_cache(ip) for ip in server_ips))
