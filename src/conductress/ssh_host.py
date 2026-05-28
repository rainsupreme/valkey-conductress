"""SSH host connection and remote command execution.

Provides the base layer for running commands and transferring files
on remote (or local) hosts via SSH. Used as a base class by Server
to separate connection management from Valkey-specific logic.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import asyncssh

from . import config

logger = logging.getLogger(__name__)


class SshHost:
    """Manages SSH connections and remote command execution on a single host."""

    # Class-level connection pool: reuse one SSH connection per (ip, username) pair.
    # Eliminates connection leaks from throwaway SshHost/Server instances.
    _pool: dict[tuple[str, str], asyncssh.SSHClientConnection] = {}

    def __init__(self, ip: str, username: str = "") -> None:
        self.ip = ip
        self.username = username
        self.ssh: Optional[asyncssh.SSHClientConnection] = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}.{ip}")

    async def ensure_ssh_connection(self) -> None:
        """Establish or reuse a pooled SSH connection."""
        pool_key = (self.ip, self.username)

        # Reuse from pool if alive
        if not self.ssh or self.ssh.is_closed():
            pooled = self._pool.get(pool_key)
            if pooled and not pooled.is_closed():
                self.ssh = pooled
                return
            # Open new connection
            if self.ip in ["127.0.0.1", "localhost"]:
                self.ssh = await asyncssh.connect(self.ip, known_hosts=None, client_keys=[str(config.SSH_KEYFILE)])
            elif self.username:
                self.ssh = await asyncssh.connect(
                    self.ip,
                    username=self.username,
                    client_keys=[str(config.SSH_KEYFILE)],
                )
            else:
                self.ssh = await asyncssh.connect(self.ip, client_keys=[str(config.SSH_KEYFILE)])
            self._pool[pool_key] = self.ssh

    async def run_host_command(self, command: str, check: bool = True) -> tuple[str, str]:
        """Run a terminal command on the host and return (stdout, stderr)."""
        self._logger.info("Host command: %s", command)
        await self.ensure_ssh_connection()
        if not self.ssh:
            raise RuntimeError(f"SSH connection to {self.ip} not established")
        result: asyncssh.SSHCompletedProcess = await self.ssh.run(command, check=check)
        return self._ensure_str(result.stdout), self._ensure_str(result.stderr)

    async def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the host."""
        result, _ = await self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

    async def get_remote_file(self, server_src: Path, local_dest: Path) -> None:
        """Copy a file from the host to the local machine."""
        server_str = await self._normalize_remote_path(server_src)
        await asyncssh.scp((self.ssh, server_str), local_dest)

    async def put_remote_file(self, local_src: Path, server_dest: Path) -> None:
        """Copy a file from the local machine to the host."""
        server_str = await self._normalize_remote_path(server_dest)
        await asyncssh.scp(local_src, (self.ssh, server_str))

    async def _normalize_remote_path(self, server_path: Path) -> str:
        """Expand ~ and variables in a remote path."""
        out, _ = await self.run_host_command(f"echo {server_path}")
        return out.strip()

    @staticmethod
    def _ensure_str(output: Union[None, bytes, str]) -> str:
        """Normalize SSH command output to a Python str."""
        if not output:
            return ""
        elif isinstance(output, memoryview):
            return bytes(output).decode()
        elif isinstance(output, bytes) or isinstance(output, bytearray):
            return output.decode()
        else:
            return output
