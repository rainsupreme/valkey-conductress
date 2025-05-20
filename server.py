"""Represents a server running a Valkey instance."""

import logging
import subprocess
import time
from pathlib import Path
from threading import Thread
from typing import Optional, Sequence

import config
from utility import hash_file, run_command

logger = logging.getLogger(__name__)

VALKEY_BINARY = "valkey-server"
PERF_STATUS_FILE = "/tmp/perf_running"


class Server:
    """Represents a server running a Valkey instance."""

    build_cache_dir = Path("~") / "build_cache"
    remote_perf_data_path = Path("~")

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
                out = self.run_valkey_command(["PING"])
                if out.strip() == "PONG":
                    return
            except subprocess.CalledProcessError:
                pass
            time.sleep(1)

        raise RuntimeError("Server did not start successfully")

    def info(self, section: str) -> dict[str, str]:
        """Run the 'info' command on the server and return the specified section."""
        result, _ = run_command(f"./valkey-cli -h {self.ip} info {section}")
        result = result.strip().split("\n")
        keypairs: dict[str, str] = {}
        for item in result:
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
        command = ["replicaof", primary_ip, port]
        response = self.run_valkey_command(command)
        if response != "OK":
            raise RuntimeError(f"Failed REPLICAOF {repr(primary_ip)} {repr(port)}: {repr(response)}")

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

    def __valkey_benchmark_on_keyspace(self, keyspace_size: int, operation: list[str]) -> None:
        """Run valkey-benchmark, sequentially covering the entire keyspace."""
        run_command(
            [
                "./valkey-benchmark",
                "-h",
                self.ip,
                "-c",
                "650",
                "-P",
                "4",
                "--threads",
                "50",
                "-q",
                "--sequential",
                "-r",
                str(keyspace_size),
                "-n",
                str(keyspace_size),
            ]
            + operation
        )

    def fill_keyspace(self, valsize: int, keyspace_size: int, test: str) -> None:
        """Load the keyspace with data for a specific test type."""
        load_type_for_test = {
            "set": "set",
            "get": "set",
            "sadd": "sadd",
            "hset": "hset",
            "zadd": "zadd",
            "zrank": "zadd",
        }
        test = load_type_for_test[test]
        self.__valkey_benchmark_on_keyspace(keyspace_size, f"-d {valsize} -t {test}".split())

    def expire_keyspace(self, keyspace_size: int, test: str) -> None:
        """
        Adds expiry data to all keys in keyspace with a long duration.
        No keys should actually expire in a test of any reasonable duration.
        """
        day = 60 * 60 * 24
        if test == "set" or test == "get":
            self.__valkey_benchmark_on_keyspace(keyspace_size, f"EXPIRE key:__rand_int__ {7 * day}".split())
        else:
            # other test types only have one key, and expire is (currently) only per-key
            # Note: hash may get per-field expiry in the future
            logger.warning("Expire keyspace not supported for test type %s. Skipping.", test)

    def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the server."""
        result, _ = self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

    def run_host_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        return run_command(command, remote_ip=self.ip, remote_pseudo_terminal=False, check=check)

    def run_interactive_host_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        return run_command(command, remote_ip=self.ip, remote_pseudo_terminal=True, check=check)

    def run_valkey_command(self, command: Sequence[str]):
        """Run a valkey command on the server and return its output."""
        response, _ = run_command(["./valkey-cli", "-h", self.ip] + list(command))
        return response.strip()

    def __get_cached_build_path(self) -> Path:
        assert self.source is not None and self.hash is not None
        return Server.build_cache_dir / self.source / self.hash

    def __get_source_binary_path(self) -> Path:
        assert self.source is not None
        return Path("~") / self.source / "src"

    def __ensure_stopped_and_clean(self):
        self.run_host_command(f"pkill -f {VALKEY_BINARY}", check=False)

        # clean up any rdb files from replication or snapshotting
        # valkey will automatically load "dump.rdb" if it is present
        self.run_host_command("rm -f *.rdb", check=False)
        # clean up any profiling files
        self.__profiling_cleanup()

    def __is_binary_cached(self) -> bool:
        return self.check_file_exists(self.__get_cached_build_path() / VALKEY_BINARY)

    def __is_remote_branch(self, specifier) -> bool:
        """Checks if specifier is a valid branch on origin. Fetches first."""
        source_path = self.__get_source_binary_path()
        command = (
            f"cd {source_path} && git fetch --quiet --prune && "
            f"git rev-parse --symbolic-full-name origin/{specifier} --"
        )
        result, _ = self.run_host_command(command)
        result = result.strip()

        if result == "":
            raise ValueError(f"{specifier} is an invalid specifier in {self.source} (empty result)")
        if result == "--":
            return False  # a specific commit by hash, unstable~2, etc.

        return result.startswith("refs/remotes/origin/")

    def __ensure_build_cached(self) -> Path:
        source_path = self.__get_source_binary_path()
        is_branch = self.__is_remote_branch(self.specifier)
        sync_target = f"origin/{self.specifier}" if is_branch else self.specifier
        self.run_host_command(f"cd {source_path} && git reset --hard {sync_target}")
        self.hash = self.__get_current_commit_hash()

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not self.__is_binary_cached():
            logger.info("building %s... (no cached build)", self.specifier)

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
            f"sudo perf record -F {sample_rate} -a -g -o {Server.remote_perf_data_path/'perf.data'} "
            f"-- sh -c 'while [ -f {PERF_STATUS_FILE} ]; do sleep 1; done'"
        )
        self.run_host_command(command)

    def is_profiling(self) -> bool:
        """Check if profiling is currently running."""
        if self.profiling_thread and self.profiling_thread.is_alive():
            return True
        return False

    def profiling_end(self) -> None:
        """End profiling and generate a report."""
        if self.profiling_thread is None:
            raise RuntimeError("Profiling not started")
        self.run_host_command(f"rm {PERF_STATUS_FILE}")
        self.profiling_thread.join()
        self.profiling_thread = None

    def profiling_report(self, task_name: str) -> None:
        """Retrieve profile data from server and generate flamegraph report."""
        self.run_host_command(
            f"sudo chmod a+r {Server.remote_perf_data_path / 'perf.data'}",
        )
        self.run_host_command("perf script -i perf.data > out.perf")
        print("collapsing stacks")
        self.run_host_command(
            f"{Server.remote_perf_data_path/'FlameGraph/stackcollapse-perf.pl'} out.perf > out.folded"
        )
        self.run_host_command("FlameGraph/flamegraph.pl out.folded > flamegraph.svg")

        print("copying perf data from server")
        test_results = Path("results").resolve() / task_name
        test_results.mkdir(parents=True, exist_ok=True)

        for file in ["perf.data", "flamegraph.svg"]:
            remote_path = Server.remote_perf_data_path / file
            local_path = test_results / file
            self.scp_file_from_server(remote_path, local_path)

    def __profiling_cleanup(self):
        files = ["perf.data", "out.perf", "out.folded", "flamegraph.svg"]
        command = " && ".join([f"rm -f {file}" for file in files])
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
