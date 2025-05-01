"""Represents a server running a Valkey instance."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

import asyncssh

import config
from utility import hash_file, run_command

logger = logging.getLogger(__name__)

VALKEY_BINARY = "valkey-server"


class Server:
    """Represents a server running a Valkey instance."""

    build_cache_dir = Path("~") / "build_cache"

    @classmethod
    def with_build(cls, ip: str, binary_source: str, specifier: str, args: list[str]) -> "Server":
        """Create a server instance and ensure it is running with the specified build."""
        server = cls(ip)
        server.start(binary_source, specifier, args)
        return server

    def __init__(self, ip: str) -> None:
        self.ip = ip

        self.source: Optional[str] = None
        self.specifier: Optional[str] = None
        self.args: Optional[list[str]] = None
        self.hash: Optional[str] = None

        self.ssh_connection: Optional[asyncssh.SSHClientConnection] = None

    def start(self, binary_source: str, specifier: str, args: list[str]) -> None:
        """Ensure specified build is running on the server."""
        self.__ensure_stopped()
        self.source = binary_source
        self.specifier = specifier
        self.args = args
        if self.source == config.MANUALLY_UPLOADED:
            cached_binary_path = self.__ensure_binary_uploaded(self.specifier)
        else:
            assert self.source in config.REPO_NAMES, f"Unknown source: {self.source}"
            cached_binary_path = self.__ensure_build_cached()
        command = f"{cached_binary_path} --save --protected-mode no --daemonize yes " + " ".join(self.args)
        self.run_host_command(command)

    def info(self, section: str) -> dict:
        """Run the 'info' command on the server and return the specified section."""
        result = run_command(f"./valkey-cli -h {self.ip} info {section}")
        result = result.strip().split("\n")
        keypairs = {}
        for item in result:
            if ":" in item:
                (key, value) = item.strip().split(":")
                keypairs[key.strip()] = value.strip()
        return keypairs

    def used_memory(self) -> int:
        """Get the amount of memory used by the server."""
        info = self.info("memory")
        return int(info["used_memory"])

    def count_items_expires(self) -> tuple[int, int]:
        """Count total items and items with expiry in the keyspace."""
        info = self.info("keyspace")
        keys = 0
        expires = 0
        for line in info.values():
            # 'keys=98331,expires=0,avg_ttl=0'
            (key, expire, _) = [int(x.split("=")[1]) for x in line.split(",")]
            keys += key
            expires += expire
        return (keys, expires)

    def get_build_hash(self):
        """
        Get unique hash for current version of valkey running on the server.
        Typically this is the commit hash of the source code used to build the server.
        """
        return self.hash

    def __valkey_benchmark_on_keyspace(self, keyspace_size: int, operation: str) -> None:
        """Run the valkey benchmark on the keyspace."""
        run_command(
            (
                f"./valkey-benchmark -h {self.ip} -c 650 -P 4 --threads 50 -q "
                f"--sequential -r {keyspace_size} -n {keyspace_size} {operation}"
            )
        )

    def fill_keyspace(self, valsize: int, keyspace_size: int, test: str) -> None:
        """Load the keyspace with data for a specific test type."""
        load_type_for_test = {
            "set": "set",
            "get": "set",
            "sadd": "sadd",
            "hset": "hset",
            "zadd": "zadd",
            "zrange": "zadd",
        }
        test = load_type_for_test[test]
        self.__valkey_benchmark_on_keyspace(keyspace_size, f"-d {valsize} -t {test}")

    def expire_keyspace(self, keyspace_size: int) -> None:
        """
        Adds expiry data to all keys in keyspace with a long duration.
        No keys should actually expire in a test of any reasonable duration.
        """
        day = 60 * 60 * 24
        self.__valkey_benchmark_on_keyspace(keyspace_size, f"EXPIRE key:__rand_int__ {7 * day}")

    def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the server."""
        result = self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

    def run_host_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        return run_command(command, remote_ip=self.ip, remote_pseudo_terminal=False, check=check)

    async def __ensure_ssh_connection(self) -> None:
        """Ensure SSH connection to the server."""
        if self.ssh_connection is None:
            try:
                self.ssh_connection = await asyncssh.connect(
                    self.ip,
                    client_keys=[config.SSH_KEYFILE],
                )
            except asyncssh.Error as e:
                logger.error("SSH connection failed: %s", str(e))
                raise

    async def run_async_host_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        await self.__ensure_ssh_connection()
        assert self.ssh_connection is not None
        result = await self.ssh_connection.run(command)
        if check and result.exit_status is not None and result.exit_status != 0:
            raise subprocess.CalledProcessError(result.exit_status, command, result.stdout, result.stderr)
        assert result.stdout is not None
        return result.stdout.strip()

    def run_host_interactive_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        return run_command(command, remote_ip=self.ip, remote_pseudo_terminal=True, check=check)

    def run_valkey_command(self, command: str):
        """Run a valkey command on the server and return its output."""
        return run_command(f"./valkey-cli -h {self.ip} {command}")

    def __get_cached_build_path(self) -> Path:
        assert self.source is not None and self.hash is not None
        return Server.build_cache_dir / self.source / self.hash

    def __get_source_binary_path(self) -> Path:
        assert self.source is not None
        return Path("~") / self.source / "src"

    def __ensure_stopped(self):
        self.run_host_command(f"pkill -f {VALKEY_BINARY}", check=False)

    def __is_binary_cached(self) -> bool:
        return self.check_file_exists(self.__get_cached_build_path() / VALKEY_BINARY)

    def __should_pull_after_checkout(self, specifier) -> bool:
        source_path = self.__get_source_binary_path()
        command = f"cd {source_path}; git rev-parse --symbolic-full-name {specifier} --"
        result = self.run_host_command(command).strip()
        if result == "":
            raise ValueError(f"{specifier} is an invalid specifier in {self.source} (empty result)")
        if result == "--":
            return False  # a specific commit by hash, unstable~2, etc.
        if result.startswith("refs/heads/"):
            return True  # local branch
        if result.startswith("refs/remotes/"):
            return True  # remote reference
        if result.startswith("refs/tags/"):
            return False  # tag
        raise ValueError(
            f"{specifier} is an unhandled type of specifier in {self.source} (got {repr(result)})"
        )

    def __ensure_build_cached(self) -> Path:
        source_path = self.__get_source_binary_path()
        self.run_host_command(
            f"cd {source_path}; git reset --hard && git fetch && git checkout {self.specifier}"
        )
        if self.__should_pull_after_checkout(self.specifier):
            # ensure branch/etc is up to date with remote
            self.run_host_command(f"cd {source_path}; git pull")
        self.hash = self.__get_current_commit_hash()

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not self.__is_binary_cached():
            logger.info("building %s... (no cached build)", self.specifier)

            self.run_host_command(f"cd {source_path}; make distclean && make -j USE_FAST_FLOAT=yes")
            self.run_host_command(f"mkdir -p {cached_build_path}")
            build_binary = source_path / VALKEY_BINARY
            self.run_host_command(f"cp {build_binary} {cached_binary_path}")

        return cached_binary_path

    @staticmethod
    def delete_entire_build_cache(server_ips) -> None:
        """Delete the entire build cache on all servers. This is a destructive operation."""
        for server_ip in server_ips:
            run_command(
                f"rm -rf {Server.build_cache_dir}",
                remote_ip=server_ip,
                remote_pseudo_terminal=False,
                check=False,
            )

    def scp_file_from_server(self, server_src: Path, local_dest: Path) -> None:
        """Copy a file from the server to the local machine."""
        command = [
            "scp",
            "-i",
            config.SSH_KEYFILE,
            f"{self.ip}:{str(server_src)}",
            str(local_dest),
        ]
        subprocess.run(command, check=True, encoding="utf-8")

    def scp_file_to_server(self, local_src: Path, server_dest: Path) -> None:
        """Copy a file from the local machine to the server."""
        command = [
            "scp",
            "-i",
            config.SSH_KEYFILE,
            str(local_src),
            f"{self.ip}:{str(server_dest)}",
        ]
        subprocess.run(command, check=True, encoding="utf-8")

    def __ensure_binary_uploaded(self, local_path) -> Path:
        self.hash = hash_file(local_path)

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not self.__is_binary_cached():
            logger.info("copying %s to server... (not cached)", local_path)

            self.run_host_command(f"mkdir -p {cached_build_path}")
            self.scp_file_to_server(local_path, cached_binary_path)

        return cached_binary_path

    def __get_current_commit_hash(self) -> str:
        source_path = self.__get_source_binary_path()
        return self.run_host_command(f"cd {source_path}; git rev-parse HEAD").strip()
