"""Represents a server running a Valkey instance."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread
from typing import Optional, Union

import asyncssh

from src.utility import async_run

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
    async def with_build(
        cls, ip: str, port: int, username: str, binary_source: str, specifier: str, io_threads: int
    ) -> "Server":
        """Create a server instance and ensure it is running with the specified build."""
        server = cls(ip, port, username)

        cached_binary_path: Path = await server.ensure_binary_cached(binary_source, specifier)
        await server.start(cached_binary_path, io_threads)
        return server

    @classmethod
    async def with_path(cls, ip: str, port: int, binary_path: Path, io_threads: int):
        """Create a server instance running the specified binary"""
        server = cls(ip, port)
        await server.start(binary_path, io_threads)
        return server

    def __init__(self, ip: str, port: int = 6379, username="") -> None:
        self.ip = ip
        self.port = port
        self.username = username

        self.logger = logging.getLogger(self.__class__.__name__ + "." + ip)

        self.ssh: Optional[asyncssh.SSHClientConnection] = None

        self.source: Optional[str] = None
        self.specifier: Optional[str] = None
        self.args: Optional[list[str]] = None
        self.hash: Optional[str] = None
        self.threads: Optional[int] = None

        self.server_started: bool = False
        self.numa_node = -1  # numa node for valkey, net irqs, and net interface
        self.numa_cpus = []
        self.net_irq_cpus: list[int] = []

        self.profiling_thread: Optional[Thread] = None
        self.profiling_abort = False

    async def __ensure_net_irqs_pinned(self):
        # determine network interface for specified ip
        stdout, _ = await self.run_host_command(f"ip -4 -o addr show | grep {self.ip}")
        net_interface = stdout.strip().split()[1]
        if net_interface == "lo":
            logging.info("Connected via loopback - no network IRQs to pin")
            self.net_irq_cpus = []
            return

        # ensure irqbalance is disabled and not running
        await self.run_host_command("sudo systemctl stop irqbalance")
        await self.run_host_command("sudo systemctl disable irqbalance")
        await self.run_host_command("sudo pkill irqbalance", check=False)

        # get the numa node for interface
        stdout, _ = await self.run_host_command(f'cat "/sys/class/net/{net_interface}/device/numa_node"')
        self.numa_node = int(stdout.strip())

        # get the cores for that numa node
        stdout, _ = await self.run_host_command("lscpu -p=node,cpu | grep -v '^#'")
        num_cpus = len(stdout.strip().split("\n"))
        self.numa_cpus = [
            int(x.split(",")[1]) for x in stdout.strip().split("\n") if x.startswith(f"{self.numa_node},")
        ]
        self.numa_cpus.sort()

        # get IRQ interrupts for interface
        stdout, _ = await self.run_host_command(f"grep -i {net_interface} /proc/interrupts")
        net_irqs = [int(irq.split(":")[0].strip()) for irq in stdout.strip().split("\n")]

        # pin IRQs to cores
        self.net_irq_cpus = []
        for i, irq in enumerate(net_irqs):
            # Use the last CPUs in the NUMA node for network IRQs
            irq_cpu = self.numa_cpus[-(i + 1)]
            self.net_irq_cpus.append(irq_cpu)
            # Create affinity mask with just this CPU enabled
            digits = (num_cpus + 3) // 4
            mask = 1 << irq_cpu
            mask = f"{mask:X}".zfill(digits)
            if digits > 8:
                # Insert commas between every 8 hex digits
                mask = ",".join(mask[i : i + 8] for i in range(0, len(mask), 8))

            # Set the IRQ affinity using the mask
            stdout, _ = await self.run_host_command(
                f"sudo sh -c 'echo {mask} > /proc/irq/{irq}/smp_affinity'"
            )
        logging.info(
            "Pinned net IRQs %s for %s to CPUs %s (Numa node %d)",
            net_irqs,
            net_interface,
            self.numa_cpus[-len(net_irqs) :],
            self.numa_node,
        )

        # warn if server threads and IRQ cores will overlap
        if not self.threads or self.threads <= 0:
            self.threads = 1
        if len(net_irqs) + self.threads > len(self.numa_cpus):
            logging.warning(
                "%d network IRQs and %d Valkey threads but only %d cores available.",
                len(net_irqs),
                self.threads,
                len(self.numa_cpus),
            )

    async def __ensure_socket_limit_sufficient(self) -> None:
        """Ensure the file descriptor limit (also used for sockets) is not too low"""
        # Increase max open files to allow more benchmark connections
        desired_fileno_limit = 65536

        fileno_limit, _ = await async_run("ulimit -n")
        if int(fileno_limit) >= desired_fileno_limit:
            return  # already good

        stdout, _ = await async_run("cat /etc/security/limits.conf")
        lines = [line.split() for line in stdout.split("\n") if "nofile" in line]
        lines = [line for line in lines if line[0] == "*"]
        if lines:
            configured_limit = min([int(line[3]) for line in lines])
            if configured_limit < desired_fileno_limit:
                self.logger.error(
                    "Insufficient socket number limit already configured (need %d, limit is %d)",
                    desired_fileno_limit,
                    configured_limit,
                )
                assert configured_limit >= desired_fileno_limit, "Insufficient socket number limit configured"
            else:
                print("Max open files already increased (try new session - log out and log in?)")
                assert not lines, "Max open files already increased (try new session - log out and log in?)"

        await async_run(
            f"sudo sh -c \"echo '* soft nofile {desired_fileno_limit}' >> /etc/security/limits.conf\""
        )
        await async_run(
            f"sudo sh -c \"echo '* hard nofile {desired_fileno_limit}' >> /etc/security/limits.conf\""
        )

        fileno_limit, _ = await async_run("ulimit -n")
        assert (
            int(fileno_limit) == desired_fileno_limit
        ), "Increased files/sockets not available. (try new session - log out and log in?)"
        logging.info("Increased max open files/sockets to %s", fileno_limit)

    async def __pre_start(self) -> None:
        """Configuration and preparation before starting Valkey server"""
        await self.__ensure_net_irqs_pinned()
        await self.__ensure_socket_limit_sufficient()

        # Enable memory overcommit
        await self.run_host_command("sudo sh -c 'echo 1 > /proc/sys/vm/overcommit_memory'")

        # Increase max open files to allow more benchmark connections
        desired_fileno_limit = 65536
        fileno_limit, _ = await async_run("ulimit -n")
        if int(fileno_limit) < desired_fileno_limit:
            stdout, _ = await async_run("cat /etc/security/limits.conf")
            lines = [line for line in stdout.split("\n") if "nofile" in line and line.startswith("*")]
            assert not lines, "Max open files already increased"

            await async_run(
                f"sudo sh -c \"echo '* soft nofile {desired_fileno_limit}' >> /etc/security/limits.conf\""
            )
            await async_run(
                f"sudo sh -c \"echo '* hard nofile {desired_fileno_limit}' >> /etc/security/limits.conf\""
            )

            fileno_limit, _ = await async_run("ulimit -n")
            assert (
                int(fileno_limit) == desired_fileno_limit
            ), "Increased files/sockets not available. (try new session - log out and log in?)"
            logging.info("Increased max open files/sockets to %s", fileno_limit)

        await self.__ensure_stopped_and_clean()
        await asyncio.sleep(1)  # short delay to it doesn't get our new server (TODO verify this)

    async def start(self, cached_binary_path: Path, io_threads: int) -> None:
        """Ensure specified build is running on the server."""
        self.threads = io_threads
        await self.__pre_start()

        self.args = []
        if io_threads > 1:
            self.args += ["--io-threads", str(io_threads)]

        # Pin Valkey process to first CPUs (avoiding network IRQ CPUs)
        valkey_cpus = ",".join(str(cpu) for cpu in self.numa_cpus[:io_threads])
        command = (
            f"taskset -c {valkey_cpus} {cached_binary_path} --port {self.port} "
            f"--save --protected-mode no --daemonize yes " + " ".join(self.args)
        )
        out, err = await self.run_host_command(command)
        self.server_started = True
        print(out, err)

        await self.wait_until_ready()

    async def ensure_binary_cached(
        self, source: Optional[str] = None, specifier: Optional[str] = None
    ) -> Path:
        """Ensure that an arbitrary binary has been cached. This needs to be done before running multiple
        instances on a single host."""
        # don't break assumption that version running matches state
        # also don't want to be running builds while benchmarks are running
        assert not self.server_started

        if source:
            self.source = source
        if specifier:
            self.specifier = specifier

        if self.source == config.MANUALLY_UPLOADED:
            return await self.__ensure_binary_uploaded(self.specifier)
        else:
            assert self.source in config.REPO_NAMES, f"Unknown source: {self.source}"
            return await self.__ensure_build_cached()

    async def wait_until_ready(self) -> None:
        """Wait until the server is ready to accept commands."""
        for _ in range(10):
            try:
                out = await self.run_valkey_command("PING")
                print(out)
                if out == "PONG":
                    return
            except asyncssh.ProcessError:
                print("cli error")
            time.sleep(1)

        raise RuntimeError("Server did not start successfully")

    async def info(self, section: str) -> dict[str, str]:
        """Run the 'info' command on the server and return the specified section."""
        response = await self.run_valkey_command(f"info {section}")
        if not response:
            raise RuntimeError(f"Failed to run 'info {section}' on server {self.ip}")
        lines = response.strip().split("\n")
        keypairs: dict[str, str] = {}
        for item in lines:
            if ":" in item:
                (key, value) = item.split(":")
                keypairs[key.strip()] = value.strip()
        return keypairs

    async def is_primary(self) -> bool:
        """Check if the server is a replica."""
        info = await self.info("replication")
        return info.get("role") == "master"

    async def is_synced_replica(self) -> bool:
        """Check if the server is a replica and in sync with the primary."""
        info = await self.info("replication")
        return info.get("role") == "slave" and info.get("master_link_status") == "up"

    async def get_replicas(self) -> list[str]:
        """Get a list of replicas for the server."""
        info = await self.info("replication")
        count = int(info.get("connected_slaves", 0))
        replicas: list[str] = []
        for i in range(count):
            replica = info.get(f"slave{i}")
            assert replica is not None, f"Replica {i} not found in replication info"
            ip = replica.split(",")[0].split("=")[1]
            replicas.append(ip)
        return replicas

    async def replicate(self, primary_ip: Optional[str], port="6379") -> None:
        """Set the server to replicate from the specified primary."""
        if primary_ip is None:
            primary_ip = "no"
            port = "one"
        response = await self.run_valkey_command(f"replicaof {primary_ip} {port}")
        if response != "OK":
            raise RuntimeError(f"Failed REPLICAOF {repr(primary_ip)} {repr(port)}: {repr(response)}")

    async def count_items_expires(self) -> tuple[int, int]:
        """Count total items and items with expiry in the keyspace."""
        info = await self.info("keyspace")
        counts = defaultdict(int)
        for line in info.values():
            # 'keys=98331,expires=0,avg_ttl=0'
            for item in line.split(","):
                name, val = item.split("=")
                counts[name] += int(val)
        return (counts["keys"], counts["expires"])

    def get_build_hash(self):
        """
        Get unique hash for current version of valkey running on the server.
        Typically this is the commit hash of the source code used to build the server.
        """
        return self.hash

    async def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the server."""
        result, _ = await self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

    async def __ensure_ssh_connection(self) -> None:
        if not self.ssh:
            if self.ip in ["127.0.0.1", "localhost"]:
                self.ssh = await asyncssh.connect(
                    self.ip, client_keys=[str(config.SSH_KEYFILE)], known_hosts=None
                )
            elif self.username:
                self.ssh = await asyncssh.connect(
                    self.ip, username=self.username, client_keys=[str(config.SSH_KEYFILE)]
                )
            else:
                self.ssh = await asyncssh.connect(self.ip, client_keys=[str(config.SSH_KEYFILE)])

    @staticmethod
    def __ensure_str(output: Union[None, bytes, str]) -> str:
        if not output:
            return ""
        elif isinstance(output, memoryview):
            # Convert memoryview to bytes, then decode
            return bytes(output).decode()
        elif isinstance(output, bytes) or isinstance(output, bytearray):
            # If it's already bytes, just decode
            return output.decode()
        else:
            return output

    async def run_host_command(self, command: str, check=True):
        """Run a terminal command on the server and return its output."""
        self.logger.info("Host command: %s", command)
        await self.__ensure_ssh_connection()
        assert self.ssh
        result: asyncssh.SSHCompletedProcess = await self.ssh.run(command, check=check)
        return self.__ensure_str(result.stdout), self.__ensure_str(result.stderr)

    async def run_valkey_command(self, command: str) -> Optional[str]:
        """Run a valkey command on the server and return its output."""
        self.logger.info("Valkey cli command: %s", command)
        cli_command: str = (
            f"{str(config.PROJECT_ROOT / config.VALKEY_CLI)} -h {self.ip} -p {self.port} " + command
        )
        stdout, _ = await async_run(cli_command, check=False)
        return stdout.strip() if stdout else None

    async def run_valkey_command_over_keyspace(self, keyspace_size: int, command: str) -> None:
        """Run valkey-benchmark, sequentially covering the entire keyspace."""
        sequential_command: str = (
            f"{str(config.PROJECT_ROOT / config.VALKEY_BENCHMARK)} -h {self.ip} -p {self.port} -c 32 -P 20 "
            f"--threads 8 -q --sequential -r {keyspace_size} -n {keyspace_size} "
        )
        sequential_command += command
        self.logger.info("Benchmark Command: %s", sequential_command)
        await async_run(sequential_command)

    def __get_cached_build_path(self) -> Path:
        assert self.source is not None and self.hash is not None
        return Server.remote_build_cache / self.source / self.hash

    def __get_source_binary_path(self) -> Path:
        assert self.source is not None
        return Server.path_root / self.source / "src"

    async def kill_all_valkey_instances_on_host(self):
        """Stop all instances of valkey server, regardless of which port they're running on"""
        await self.run_host_command(f"pkill -f {VALKEY_BINARY}", check=False)

    async def __ensure_stopped_and_clean(self):
        self.server_started = False
        process_id, _ = await self.run_host_command(f"lsof -ti :{self.port}", check=False)
        process_id = process_id.strip()
        if not process_id:
            return

        # ensure process using the port is one of our expected valkey processes
        # (in case an unexpected process is already using the port for some reason)
        name, _ = await self.run_host_command(f"ps -p {process_id}")
        name = name.strip().split()[-1]
        assert name == VALKEY_BINARY

        await self.run_host_command(f"kill -9 {process_id}")

        # clean up any rdb files from replication or snapshotting
        # valkey will automatically load "dump.rdb" if it is present
        await self.run_host_command("rm -f *.rdb", check=False)
        # clean up any profiling files
        await self.__profiling_cleanup()

    async def __is_binary_cached(self) -> bool:
        return await self.check_file_exists(self.__get_cached_build_path() / VALKEY_BINARY)

    async def __normalize_specifier(self, specifier) -> str:
        """Checks if specifier is a valid branch on origin. Fetches first."""
        source_path = self.__get_source_binary_path()
        await self.run_host_command(f"cd {source_path} && git fetch --quiet --prune")
        try:
            result, _ = await self.run_host_command(
                f"cd {source_path} && git rev-parse --symbolic-full-name origin/{specifier} --"
            )
        except asyncssh.ProcessError:
            print(f"Failed to resolve {specifier} in {self.source}, trying as-is")
            result, _ = await self.run_host_command(
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

    async def __ensure_build_cached(self) -> Path:
        source_path: Path = self.__get_source_binary_path()
        sync_target: str = await self.__normalize_specifier(self.specifier)
        await self.run_host_command(f"cd {source_path} && git reset --hard {sync_target}")
        self.hash = await self.__get_current_commit_hash()

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = cached_build_path / VALKEY_BINARY

        if not await self.__is_binary_cached():
            self.logger.info("building %s:%s...", self.source, self.specifier)

            try:
                await self.run_host_command(
                    f"cd {source_path}; "
                    "make distclean && "
                    'make -j USE_FAST_FLOAT=yes CFLAGS="-fno-omit-frame-pointer"'
                )
            except asyncssh.ProcessError as e:
                self.logger.error("Build failed %d:\n%s", e.returncode, e.stderr)
                raise e
            build_binary = source_path / VALKEY_BINARY

            await self.run_host_command(f"mkdir -p {cached_build_path}")
            await self.run_host_command(f"cp {build_binary} {cached_binary_path}")

        return cached_binary_path

    @staticmethod
    async def delete_entire_build_cache(server_ips) -> None:
        """Delete the entire build cache on all servers. This is a destructive operation."""

        async def delete_host_cache(ip):
            async with asyncssh.connect(ip, client_keys=[str(config.SSH_KEYFILE)]) as conn:
                await conn.run(f"rm -rf {Server.remote_build_cache}", check=False)

        await asyncio.gather(*(delete_host_cache(ip) for ip in server_ips))

    async def __normalize_remote_path(self, server_path: Path):
        out, _ = await self.run_host_command(f"echo {server_path}")
        return out.strip()

    async def get_remote_file(self, server_src: Path, local_dest: Path) -> None:
        """Copy a file from the server to the local machine."""
        server_str = await self.__normalize_remote_path(server_src)
        await asyncssh.scp((self.ssh, server_str), local_dest)

    async def put_remote_file(self, local_src: Path, server_dest: Path) -> None:
        """Copy a file from the local machine to the server."""
        server_str = await self.__normalize_remote_path(server_dest)
        await asyncssh.scp(local_src, (self.ssh, server_str))

    async def __ensure_binary_uploaded(self, local_path) -> Path:
        out, _ = await async_run(f"sha1sum {str(local_path)}")
        assert out is not None, "Failed to run sha1sum on local binary"
        self.hash = out.strip().split()[0]

        cached_binary_path = self.__get_cached_build_path() / VALKEY_BINARY

        if not self.__is_binary_cached():
            self.logger.info("copying %s to server... (not cached)", local_path)

            await self.run_host_command(f"mkdir -p {cached_binary_path.parent}")
            await self.put_remote_file(local_path, cached_binary_path)

        return cached_binary_path

    async def __get_current_commit_hash(self) -> str:
        source_path = self.__get_source_binary_path()
        out, _ = await self.run_host_command(f"cd {source_path}; git rev-parse HEAD")
        return out.strip()

    def profiling_start(self, sample_rate: int) -> None:
        """Start profiling the server using perf."""
        if self.is_profiling():
            raise RuntimeError("Profiling already started")

        self.profiling_thread = Thread(
            target=self.__profiling_run, args=(sample_rate,)
        )  # TODO make this async instead
        self.profiling_thread.start()

    async def __profiling_run(self, sample_rate: int) -> None:
        """Profile performance using perf for specified duration. Leaves data file on server."""
        await self.run_host_command(f"touch {PERF_STATUS_FILE}")
        command = (
            f"sudo perf record -F {sample_rate} -a -g -o {Server.perf_data_path} "
            f"-- sh -c 'while [ -f {PERF_STATUS_FILE} ]; do sleep 1; done'"
        )
        await self.run_host_command(command)

    def is_profiling(self) -> bool:
        """Check if profiling is currently running."""
        if self.profiling_thread and self.profiling_thread.is_alive():
            return True
        return False

    async def profiling_stop(self) -> None:
        """Signals profiling to stop. Use profiling_wait() to ensure that it actually finishes."""
        if self.profiling_thread is None:
            return
        await self.run_host_command(f"rm -f {PERF_STATUS_FILE}")

    def profiling_wait(self) -> None:
        """Block until profiling finishes. Call profiling_stop() first or you'll wait forever"""
        if self.profiling_thread is None:
            return
        self.profiling_thread.join()
        self.profiling_thread = None

    async def profiling_report(self, task_name: str, server_name: str) -> None:
        """Retrieve profile data from server and generate flamegraph report.
        Stops profiling first if needed"""

        out_perf_path = Server.path_root / "out.perf"
        out_folded_path = Server.path_root / "out.folded"

        if self.is_profiling():
            await self.profiling_stop()
        self.profiling_wait()

        await self.run_host_command(
            f"sudo chmod a+r {Server.perf_data_path}",
        )
        await self.run_host_command(f"perf script -i {Server.perf_data_path} > {out_perf_path}")
        print("collapsing stacks")
        await self.run_host_command(
            f"{Server.flamegraph/'stackcollapse-perf.pl'} " f"{out_perf_path} > {out_folded_path}"
        )
        await self.run_host_command(
            f"{Server.flamegraph/'flamegraph.pl'} {out_folded_path} > {Server.flamegraph_path}"
        )
        await self.run_host_command(f"rm -f {out_perf_path} {out_folded_path}")

        print("copying perf data from server")
        test_results = config.CONDUCTRESS_RESULTS / task_name
        test_results.mkdir(parents=True, exist_ok=True)

        async def copy_remote_file(remote_file: Path):
            filename = f"{server_name}-{remote_file.name}"
            local_file = test_results / filename
            await self.get_remote_file(remote_file, local_file)

        await asyncio.gather(
            *(copy_remote_file(file) for file in [Server.perf_data_path, Server.flamegraph_path])
        )

    async def __profiling_cleanup(self):
        command = f"rm -f {Server.perf_data_path} {Server.flamegraph_path}"
        await self.run_host_command(command)
