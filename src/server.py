"""Represents a server running a Valkey instance."""

import logging
import time
from pathlib import Path
from threading import Thread
from typing import Optional

from fabric import Connection
from invoke import run
from invoke.exceptions import UnexpectedExit
from invoke.runners import Result as RunResult

from . import config

VALKEY_BINARY = "valkey-server"
PERF_STATUS_FILE = "/tmp/perf_running"


class Server:
    """Represents a server running a Valkey instance."""

    # These are remote paths - they exist on the valkey server
    path_root = Path("~")
    remote_build_cache = path_root / "build_cache"
    # profiling related paths
    flamegraph = path_root / "FlameGraph"
    perf_data_path = path_root / "perf.data"
    flamegraph_path = path_root / "flamegraph.svg"

    @classmethod
    def with_build(
        cls, ip: str, binary_source: str, specifier: str, io_threads: int, args: list[str]
    ) -> "Server":
        """Create a server instance and ensure it is running with the specified build."""
        server = cls(ip)
        server.threads = io_threads
        if io_threads > 1:
            args += ["--io-threads", str(io_threads)]
        server.start(binary_source, specifier, args)
        return server

    def __init__(self, ip: str) -> None:
        self.ip = ip

        self.logger = logging.getLogger(self.__class__.__name__ + "." + ip)

        self.ssh: Connection = Connection(self.ip, connect_kwargs={"key_filename": str(config.SSH_KEYFILE)})

        self.source: Optional[str] = None
        self.specifier: Optional[str] = None
        self.args: Optional[list[str]] = None
        self.hash: Optional[str] = None
        self.threads: Optional[int] = None

        self.profiling_thread: Optional[Thread] = None
        self.profiling_abort = False

    def start(self, binary_source: str, specifier: str, args: list[str]) -> None:
        """Ensure specified build is running on the server."""
        self.__ensure_stopped_and_clean()
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
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        """Wait until the server is ready to accept commands."""
        for _ in range(10):
            try:
                out = self.run_valkey_command("PING")
                if out == "PONG":
                    return
            except UnexpectedExit:
                pass
            time.sleep(1)

        raise RuntimeError("Server did not start successfully")

    def info(self, section: str) -> dict[str, str]:
        """Run the 'info' command on the server and return the specified section."""
        stdout = self.run_valkey_command(f"info {section}")
        if stdout is None:
            raise RuntimeError(f"Failed to run 'info {section}' on server {self.ip}")
        out = stdout.strip().split("\n")
        keypairs: dict[str, str] = {}
        for item in out:
            if ":" in item:
                (key, value) = item.split(":")
                keypairs[key.strip()] = value.strip()
        return keypairs

    def is_primary(self) -> bool:
        """Check if the server is a replica."""
        info = self.info("replication")
        return info.get("role") == "master"

    def is_synced_replica(self) -> bool:
        """Check if the server is a replica and in sync with the primary."""
        info = self.info("replication")
        return info.get("role") == "slave" and info.get("master_link_status") == "up"

    def get_replicas(self) -> list[str]:
        """Get a list of replicas for the server."""
        info = self.info("replication")
        count = int(info.get("connected_slaves", 0))
        replicas: list[str] = []
        for i in range(count):
            replica = info.get(f"slave{i}")
            assert replica is not None, f"Replica {i} not found in replication info"
            ip = replica.split(",")[0].split("=")[1]
            replicas.append(ip)
        return replicas

    def replicate(self, primary_ip: Optional[str], port="6379") -> None:
        """Set the server to replicate from the specified primary."""
        if primary_ip is None:
            primary_ip = "no"
            port = "one"
        response = self.run_valkey_command(f"replicaof {primary_ip} {port}")
        if response != "OK":
            raise RuntimeError(f"Failed REPLICAOF {repr(primary_ip)} {repr(port)}: {repr(response)}")

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

    def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the server."""
        result, _ = self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

    def run_host_command(self, command: str, check=True) -> tuple[str, str]:
        """Run a terminal command on the server and return its output."""
        self.logger.info("Host command: %s", command)
        result: RunResult = self.ssh.run(command, hide=True, warn=not check)
        return result.stdout, result.stderr

    def run_valkey_command(self, command: str) -> Optional[str]:
        """Run a valkey command on the server and return its output."""
        self.logger.info("Valkey cli command: %s", command)
        cli_command: str = f"{str(config.VALKEY_CLI)} -h {self.ip} " + command
        result = run(cli_command, hide=True)
        return result.stdout.strip() if result else None

    def run_valkey_command_over_keyspace(self, keyspace_size: int, command: str) -> None:
        """Run valkey-benchmark, sequentially covering the entire keyspace."""
        sequential_command: str = (
            f"{str(config.VALKEY_BENCHMARK)} -h {self.ip} -c 650 -P 4 "
            f"--threads 50 -q --sequential -r {keyspace_size} -n {keyspace_size} "
        )
        sequential_command += command
        self.logger.info("Benchmark Command: %s", sequential_command)
        run(sequential_command, hide=True)

    def __get_cached_build_path(self) -> Path:
        assert self.source is not None and self.hash is not None
        return Server.remote_build_cache / self.source / self.hash

    def __get_source_binary_path(self) -> Path:
        assert self.source is not None
        return Server.path_root / self.source / "src"

    def __ensure_stopped_and_clean(self):
        self.run_host_command(f"pkill -f {VALKEY_BINARY}", check=False)

        # clean up any rdb files from replication or snapshotting
        # valkey will automatically load "dump.rdb" if it is present
        self.run_host_command("rm -f *.rdb", check=False)
        # clean up any profiling files
        self.__profiling_cleanup()

    def __is_binary_cached(self) -> bool:
        return self.check_file_exists(self.__get_cached_build_path() / VALKEY_BINARY)

    def __normalize_specifier(self, specifier) -> str:
        """Checks if specifier is a valid branch on origin. Fetches first."""
        source_path = self.__get_source_binary_path()
        self.run_host_command(f"cd {source_path} && git fetch --quiet --prune")
        try:
            result, _ = self.run_host_command(
                f"cd {source_path} && git rev-parse --symbolic-full-name origin/{specifier} --"
            )
        except UnexpectedExit:
            print(f"Failed to resolve {specifier} in {self.source}, trying as-is")
            result, _ = self.run_host_command(
                f"cd {source_path} && git rev-parse --symbolic-full-name {specifier} --"
            )
        result = result.strip()
        if result == "":
            raise ValueError(f"{specifier} is an invalid specifier in {self.source} (empty result)")
        if result == "--":
            return specifier
        if result.startswith("refs/remotes/origin/"):
            return f"origin/{specifier}"
        return specifier

    def __ensure_build_cached(self) -> Path:
        source_path = self.__get_source_binary_path()
        sync_target = self.__normalize_specifier(self.specifier)
        self.run_host_command(f"cd {source_path} && git reset --hard {sync_target}")
        self.hash = self.__get_current_commit_hash()

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not self.__is_binary_cached():
            self.logger.info("building %s:%s...", self.source, self.specifier)

            self.run_host_command(
                f"cd {source_path}; "
                "make distclean && "
                'make -j USE_FAST_FLOAT=yes CFLAGS="-fno-omit-frame-pointer"'
            )
            self.run_host_command(f"mkdir -p {cached_build_path}")
            build_binary = source_path / VALKEY_BINARY
            self.run_host_command(f"cp {build_binary} {cached_binary_path}")

        return cached_binary_path

    @staticmethod
    def delete_entire_build_cache(server_ips) -> None:
        """Delete the entire build cache on all servers. This is a destructive operation."""
        for server_ip in server_ips:
            with Connection(server_ip, connect_kwargs={"key_filename": str(config.SSH_KEYFILE)}) as conn:
                conn.run(
                    f"rm -rf {Server.remote_build_cache}",
                    hide=True,
                    warn=True,
                )

    def get_file(self, server_src: Path, local_dest: Path) -> None:
        """Copy a file from the server to the local machine."""
        server_str = self.run_host_command(f"echo {server_src}")[0].strip()
        self.ssh.get(server_str, local=str(local_dest))

    def put_file(self, local_src: Path, server_dest: Path) -> None:
        """Copy a file from the local machine to the server."""
        server_str = self.run_host_command(f"echo {server_dest}")[0].strip()
        self.ssh.put(str(local_src), remote=server_str)

    def __ensure_binary_uploaded(self, local_path) -> Path:
        result = run(f"sha1sum {str(local_path)}")
        assert result is not None, "Failed to run sha1sum on local binary"
        self.hash = result.stdout.strip().split()[0]

        cached_binary_path = self.__get_cached_build_path() / VALKEY_BINARY

        if not self.__is_binary_cached():
            self.logger.info("copying %s to server... (not cached)", local_path)

            self.run_host_command(f"mkdir -p {cached_binary_path.parent}")
            self.put_file(local_path, cached_binary_path)

        return cached_binary_path

    def __get_current_commit_hash(self) -> str:
        source_path = self.__get_source_binary_path()
        out, _ = self.run_host_command(f"cd {source_path}; git rev-parse HEAD")
        return out.strip()

    def profiling_start(self, sample_rate: int) -> None:
        """Start profiling the server using perf."""
        if self.is_profiling():
            raise RuntimeError("Profiling already started")

        self.profiling_thread = Thread(target=self.__profiling_run, args=(sample_rate,))
        self.profiling_thread.start()

    def __profiling_run(self, sample_rate: int) -> None:
        """Profile performance using perf for specified duration. Leaves data file on server."""
        self.run_host_command(f"touch {PERF_STATUS_FILE}")
        command = (
            f"sudo perf record -F {sample_rate} -a -g -o {Server.perf_data_path} "
            f"-- sh -c 'while [ -f {PERF_STATUS_FILE} ]; do sleep 1; done'"
        )
        self.run_host_command(command)

    def is_profiling(self) -> bool:
        """Check if profiling is currently running."""
        if self.profiling_thread and self.profiling_thread.is_alive():
            return True
        return False

    def profiling_stop(self) -> None:
        """Signals profiling to stop. Use profiling_wait() to ensure that it actually finishes."""
        if self.profiling_thread is None:
            return
        self.run_host_command(f"rm -f {PERF_STATUS_FILE}")

    def profiling_wait(self) -> None:
        """Block until profiling finishes. Call profiling_stop() first or you'll wait forever"""
        if self.profiling_thread is None:
            return
        self.profiling_thread.join()
        self.profiling_thread = None

    def profiling_report(self, task_name: str, server_name: str) -> None:
        """Retrieve profile data from server and generate flamegraph report.
        Stops profiling first if needed"""

        out_perf_path = Server.path_root / "out.perf"
        out_folded_path = Server.path_root / "out.folded"

        if self.is_profiling():
            self.profiling_stop()
        self.profiling_wait()

        self.run_host_command(
            f"sudo chmod a+r {Server.perf_data_path}",
        )
        self.run_host_command(f"perf script -i {Server.perf_data_path} > {out_perf_path}")
        print("collapsing stacks")
        self.run_host_command(
            f"{Server.flamegraph/'stackcollapse-perf.pl'} " f"{out_perf_path} > {out_folded_path}"
        )
        self.run_host_command(
            f"{Server.flamegraph/'flamegraph.pl'} {out_folded_path} > {Server.flamegraph_path}"
        )
        self.run_host_command(f"rm -f {out_perf_path} {out_folded_path}")

        print("copying perf data from server")
        test_results = config.CONDUCTRESS_RESULTS / task_name
        test_results.mkdir(parents=True, exist_ok=True)

        for remote_file in [Server.perf_data_path, Server.flamegraph_path]:
            filename = f"{server_name}-{remote_file.name}"
            local_file = test_results / filename
            self.get_file(remote_file, local_file)

    def __profiling_cleanup(self):
        command = f"rm -f {Server.perf_data_path} {Server.flamegraph_path}"
        self.run_host_command(command)

    # async def get_valkey_server_threads(self) -> list[tuple[int, int]]:
    #     """Get the PID and TID of all valkey-server threads."""
    #     await self.__ensure_ssh_connection()
    #     command = "ps -T -C valkey-server -o pid,spid,comm"
    #     output, _ = self.run_host_command(command)
    #     threads: list[tuple[int, int]] = []
    #     for line in output.splitlines()[1:]:  # Skip header
    #         parts = line.split()
    #         if len(parts) >= 2:
    #             pid, tid = parts[:2]
    #             threads.append((int(pid), int(tid)))
    #     return threads

    # def get_network_irqs(self) -> list[tuple[int, str]]:
    #     """Get network IRQs and their names."""
    #     command = """
    #     grep -E 'eth|eno' /proc/interrupts |
    #     awk '{printf "%s %s\\n", $1, $NF}' |
    #     tr -d :
    #     """
    #     output, _ = self.run_host_command(command)
    #     irqs: list[tuple[int, str]] = []
    #     for line in output.splitlines():
    #         parts = line.split(maxsplit=1)
    #         if len(parts) == 2:
    #             irq, name = parts
    #             irqs.append((int(irq), name))
    #     return irqs

    # def get_thread_cpus(self, thread_list: list[tuple[int, int]]) -> list[int]:
    #     """Get the CPUs that last executed specific threads.

    #     Args:
    #         thread_list: List of (pid, tid) tuples

    #     Returns:
    #         list of CPUs to last execute threads (same order as input)
    #     """
    #     if not thread_list:
    #         return []

    #     commands = [f"cat /proc/{pid}/task/{tid}/stat" for pid, tid in thread_list]
    #     combined_command = " & ".join(commands)
    #     output, _ = self.run_host_command(combined_command)

    #     # The 39th field (0-based index 38) contains the last executed CPU number
    #     results = [int(x.split()[38]) for x in output.splitlines()]
    #     assert len(results) == len(thread_list), "Mismatch in number of threads and output lines"
    #     return results

    # def get_irq_cpus(self, irq_list: list[int]) -> list[int]:
    #     """Get the CPUs that last handled specific IRQs."""
    #     if not irq_list:
    #         return []

    #     commands = [f"cat /proc/irq/{irq}/smp_affinity_list" for irq in irq_list]
    #     combined_command = " & ".join(commands)
    #     output, _ = self.run_host_command(combined_command)

    #     results = [int(x.strip()) for x in output.splitlines()]
    #     assert len(results) == len(irq_list), "Mismatch in number of IRQs and output lines"
    #     return results

    # def monitor_cpu_execution(self):
    #     print("\nMonitoring current/last CPU execution:")
    #     threads = self.get_valkey_server_threads()
    #     irqs = self.get_network_irqs()

    #     print("\nValkey-server threads:")
    #     print(self.get_thread_cpus(threads))

    #     print("\nNetwork IRQs:")
    #     print(self.get_irq_cpus([irq for irq, _ in irqs]))
