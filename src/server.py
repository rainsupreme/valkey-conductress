"""Represents a server running a Valkey instance."""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread
from typing import Optional, Union
import subprocess

import asyncssh

from src.utility import async_run
from src.cpu_allocator import CpuAllocator, AllocationTag

from . import config

VALKEY_BINARY = "valkey-server"
PERF_STATUS_FILE = "/tmp/profiling_running"
PERF_STAT_STATUS_FILE = "/tmp/perf_stat_running"


class Server:
    """Represents a server running a Valkey instance."""

    # Class-level CPU allocator shared across all Server instances
    _cpu_allocator = CpuAllocator()

    # These are remote paths - they exist on the valkey server
    path_root = Path("~")
    remote_build_cache = path_root / "build_cache"
    # profiling related paths
    flamegraph = path_root / "FlameGraph"
    perf_data_path = path_root / "perf.data"
    flamegraph_path = path_root / "flamegraph.svg"
    perf_stats_path = path_root / "perf_stat_output"

    # =============================================================================
    # INITIALIZATION AND FACTORY METHODS
    # =============================================================================

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
        self.make_args: str = config.DEFAULT_MAKE_ARGS
        self.threads: int = 1

        self.valkey_pid: int = -1  # -1 indicates server is not running
        self.server_cpus: list[int] = []  # CPUs allocated to this server
        self._allocation_tag: Optional[AllocationTag] = None  # Tag for CPU allocation

        self.profiling_thread: Optional[Thread] = None
        self.perf_stat_thread: Optional[Thread] = None

    @classmethod
    async def with_build(
        cls,
        ip: str,
        port: int,
        username: str,
        binary_source: str,
        specifier: str,
        io_threads: int,
        make_args: str,
    ) -> "Server":
        """Create a server instance and ensure it is running with the specified build."""
        server = cls(ip, port, username)

        cached_binary_path: Path = await server.ensure_binary_cached(binary_source, specifier, make_args)
        await server.start(cached_binary_path, io_threads)
        return server

    @classmethod
    async def with_path(cls, ip: str, port: int, binary_path: Path, io_threads: int):
        """Create a server instance running the specified binary"""
        server = cls(ip, port)
        await server.start(binary_path, io_threads)
        return server

    # =============================================================================
    # CPU AND IRQ PINNING METHODS
    # =============================================================================

    async def ensure_host_cpu_allocation(self):
        """Main orchestrator method for network IRQ pinning"""
        net_interface = await self._detect_interface_for_ip()

        await self._register_host_cpus(net_interface)

        # Skip IRQ pinning for loopback, but do everything else
        if net_interface != "lo":
            await self._disable_irqbalance()
            await self._pin_network_irqs(net_interface)
        else:
            logging.info("Connected via loopback - no network IRQs to pin")

    async def _detect_interface_for_ip(self) -> str:
        """Find network interface for the server IP"""
        stdout, _ = await self.run_host_command(f"ip -4 -o addr show | grep {self.ip}")
        return stdout.strip().split()[1]

    async def _register_host_cpus(self, net_interface: str):
        """Get NUMA node for the interface and register host with allocator"""
        # Skip if already registered
        if self._cpu_allocator.is_host_registered(self.ip):
            return

        stdout, _ = await self.run_host_command("lscpu -p=node,cpu | grep -v '^#'")
        lines = stdout.strip().split("\n")

        # Parse all CPUs and build NUMA topology
        all_cpus = []
        numa_topology = defaultdict(list)
        for line in lines:
            node, cpu = line.split(",")
            cpu_id = int(cpu)
            numa_node = int(node)
            all_cpus.append(cpu_id)
            numa_topology[numa_node].append(cpu_id)

        # Detect L3 cache topology
        l3_cache_topology = await self._detect_l3_cache_topology(all_cpus)

        if net_interface == "lo":
            net_interface_numa = 0  # use NUMA node 0 for cache locality
        else:
            # Get the numa node matching interface
            numa_stdout, _ = await self.run_host_command(
                f'cat "/sys/class/net/{net_interface}/device/numa_node"'
            )
            net_interface_numa = int(numa_stdout.strip())

        # Register host with allocator
        self._cpu_allocator.register_host(
            self.ip,
            all_cpus=all_cpus,
            numa_topology=dict(numa_topology),
            l3_cache_topology=l3_cache_topology,
            net_interface_numa=net_interface_numa,
        )

    async def _detect_l3_cache_topology(self, all_cpus: list[int]) -> dict[int, list[int]]:
        """Detect L3 cache topology by parsing sysfs.
        
        Returns mapping of L3 cache ID -> list of CPUs sharing that cache.
        """
        l3_topology = defaultdict(list)
        
        for cpu in all_cpus:
            # Read L3 cache ID for this CPU
            cache_path = f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/id"
            result, _ = await self.run_host_command(f"cat {cache_path} 2>/dev/null || echo -1", check=False)
            
            try:
                l3_id = int(result.strip())
                if l3_id >= 0:
                    l3_topology[l3_id].append(cpu)
            except ValueError:
                # L3 cache info not available, skip
                pass
        
        return dict(l3_topology) if l3_topology else {}

    async def _disable_irqbalance(self):
        """Disable irqbalance to allow manual IRQ pinning"""
        await self.run_host_command("sudo systemctl stop irqbalance")
        await self.run_host_command("sudo systemctl disable irqbalance")
        await self.run_host_command("sudo pkill irqbalance", check=False)

    async def _pin_network_irqs(self, net_interface: str):
        """Find and pin network IRQs to specific CPUs"""
        # Skip for loopback interface - no IRQs to pin
        if net_interface == "lo":
            return

        # Skip if IRQs already configured on this host
        irq_tag = AllocationTag(task_id="system", purpose="irq")
        if self._cpu_allocator.get_allocation(self.ip, irq_tag) is not None:
            logging.info(
                "Network IRQs already configured on %s, using existing CPUs %s",
                self.ip,
                self._cpu_allocator.get_allocation(self.ip, irq_tag),
            )
            return

        # Get IRQ interrupts for interface
        stdout, _ = await self.run_host_command(f"grep -i {net_interface} /proc/interrupts")
        net_irqs = [int(irq.split(":")[0].strip()) for irq in stdout.strip().split("\n")]

        # Allocate CPUs for IRQs on the network interface NUMA node
        # Try to avoid server cache if server already allocated
        net_numa = self._cpu_allocator.get_net_interface_numa(self.ip)
        server_tags = [tag for tag in self._cpu_allocator.get_all_allocations(self.ip).keys() 
                       if tag.purpose == "server"]
        
        irq_cpus = self._cpu_allocator.allocate(
            self.ip, 
            irq_tag, 
            count=len(net_irqs), 
            require_numa=net_numa,
            avoid_tags=server_tags,
            prefer_different_cache=True,
        )

        # Get total CPU count for mask calculation
        stdout, _ = await self.run_host_command("lscpu -p=node,cpu | grep -v '^#'")
        total_cpus = len(stdout.strip().split("\n"))

        # Pin IRQs to allocated CPUs
        for i, irq in enumerate(net_irqs):
            irq_cpu = irq_cpus[-(i + 1)]  # Use last CPUs from allocation

            # Create affinity mask with just this CPU enabled
            digits = (total_cpus + 3) // 4
            mask = 1 << irq_cpu
            mask = f"{mask:X}".zfill(digits)
            if digits > 8:
                # Insert commas between every 8 hex digits from the right
                mask = mask[::-1]  # Reverse for right-to-left processing
                mask = ",".join(mask[i : i + 8] for i in range(0, len(mask), 8))
                mask = mask[::-1]  # Reverse back to normal order

            # Set the IRQ affinity using the mask
            await self.run_host_command(f"sudo sh -c 'echo {mask} > /proc/irq/{irq}/smp_affinity'")

        logging.info(
            "Pinned net IRQs %s for %s to CPUs %s (NUMA node %d)",
            net_irqs,
            net_interface,
            irq_cpus,
            net_numa,
        )

    async def _allocate_server_cpus(self, cpu_count: int) -> list[int]:
        """Allocate CPUs for this server on the network interface NUMA node"""
        # Validate before attempting allocation
        self._validate_sufficient_cpus()
        
        # Create allocation tag for this server
        self._allocation_tag = AllocationTag(task_id=f"server_{self.ip}_{self.port}", purpose="server")

        # Allocate on network interface NUMA node
        net_numa = self._cpu_allocator.get_net_interface_numa(self.ip)
        cpus = self._cpu_allocator.allocate(
            self.ip,
            self._allocation_tag,
            count=cpu_count,
            require_numa=net_numa,
        )

        logging.info(
            "Allocated CPUs %s for server on %s:%d (NUMA node %d)",
            cpus,
            self.ip,
            self.port,
            net_numa,
        )

        return cpus

    async def _pin_valkey_threads(self):
        """Pin main thread and I/O threads to individual CPUs after server starts"""
        pid_out, _ = await self.run_host_command(f"lsof -ti :{self.port}")
        main_pid = pid_out.strip()

        threads_out, _ = await self.run_host_command(f"ps -T -p {main_pid} -o tid,comm --no-headers")

        # Pin main thread to first CPU
        main_cpu = self.server_cpus[0]
        await self.run_host_command(f"taskset -cp {main_cpu} {main_pid}")
        logging.info("Pinned main thread %s to CPU %d", main_pid, main_cpu)

        # Pin I/O threads to individual CPUs
        io_thread_cpu_index = 1
        for line in threads_out.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                tid, comm = parts[0], parts[1]

                if tid == main_pid:
                    continue

                if comm.startswith("io_thd_"):
                    cpu = self.server_cpus[io_thread_cpu_index]
                    await self.run_host_command(f"taskset -cp {cpu} {tid}")
                    logging.info("Pinned I/O thread %s (%s) to CPU %d", tid, comm, cpu)
                    io_thread_cpu_index += 1

        # Brief delay to ensure scheduler applies affinity changes
        await asyncio.sleep(0.1)

    def _release_server_cpus(self):
        """Release CPUs allocated to this server"""
        if self._allocation_tag and self.server_cpus:
            self._cpu_allocator.release(self.ip, self._allocation_tag)
            logging.info(
                "Released CPUs %s for server on %s:%d",
                self.server_cpus,
                self.ip,
                self.port,
            )
            self.server_cpus = []
            self._allocation_tag = None

    async def get_available_cpu_count(self) -> int:
        """Get the number of CPUs available on this host."""
        # Ensure host is registered with allocator
        await self.ensure_host_cpu_allocation()

        net_numa = self._cpu_allocator.get_net_interface_numa(self.ip)
        return self._cpu_allocator.get_available_count(self.ip, prefer_numa=net_numa)

    def _validate_sufficient_cpus(self) -> None:
        """Validate sufficient CPUs available on network interface NUMA node.
        
        Raises:
            RuntimeError: If insufficient CPUs available for server needs
        """
        net_numa = self._cpu_allocator.get_net_interface_numa(self.ip)
        available = self._cpu_allocator.get_available_count(self.ip, prefer_numa=net_numa)
        needed = Server.getNumCPUs(self.threads)
        
        if available < needed:
            raise RuntimeError(
                f"Insufficient CPUs on {self.ip} NUMA node {net_numa}: "
                f"need {needed}, available {available}"
            )

    # =============================================================================
    # PROFILING METHODS
    # =============================================================================

    def profiling_start(self, sample_rate: int) -> None:
        """Start profiling the server using perf."""
        if self.is_profiling():
            raise RuntimeError("Profiling already started")

        self.profiling_thread = Thread(target=self.__profiling_run_sync, args=(sample_rate,))
        self.profiling_thread.start()

    def __profiling_run_sync(self, sample_rate: int) -> None:
        """Profile performance using perf for specified duration. Leaves data file on server."""
        command = f"touch {PERF_STATUS_FILE}"
        if self.ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), self.ip, command], check=True)
        
        perf_command = (
            f"sudo perf record -F {sample_rate} -a -g -o {Server.perf_data_path} "
            f"-- sh -c 'while [ -f {PERF_STATUS_FILE} ]; do sleep 1; done'"
        )
        if self.ip in ["127.0.0.1", "localhost"]:
            subprocess.run(perf_command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), self.ip, perf_command], check=True)

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

    async def profiling_report(self, result_dir: Path) -> None:
        """Retrieve profile data from server and generate flamegraph report.
        Stops profiling first if needed"""

        assert result_dir.exists(), f"Result directory {result_dir} must exist"

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
        await self.get_remote_file(Server.perf_data_path, result_dir / "perf.data")
        await self.get_remote_file(Server.flamegraph_path, result_dir / "flamegraph.svg")

    async def __data_collection_cleanup(self):
        command = f"rm -f {Server.perf_data_path} {Server.flamegraph_path} {Server.perf_stats_path}"
        await self.run_host_command(command)

    # =============================================================================
    # PERF STAT METHODS
    # =============================================================================

    async def perf_stat_start(self) -> None:
        """Start perf stat collection for specified duration."""
        if self.perf_stat_thread and self.perf_stat_thread.is_alive():
            raise RuntimeError("Perf stat already running")

        await self.run_host_command(f"touch {PERF_STAT_STATUS_FILE}")
        self.perf_stat_thread = Thread(target=self.__perf_stat_run_sync)
        self.perf_stat_thread.start()

    def __perf_stat_run_sync(self) -> None:
        """Run perf stat synchronously in thread."""
        command = (
            f"sudo perf stat -d -p {self.valkey_pid} -o {Server.perf_stats_path} "
            f"-- sh -c 'while [ -f {PERF_STAT_STATUS_FILE} ]; do sleep 1; done'"
        )
        if self.ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), self.ip, command], check=True)

    async def perf_stat_stop(self) -> None:
        """Signals perf stat to stop. Use perf_stat_wait() to ensure that it actually finishes."""
        if self.perf_stat_thread is None:
            return
        await self.run_host_command(f"rm -f {PERF_STAT_STATUS_FILE}")

    def perf_stat_wait(self) -> None:
        """Wait for perf stat to complete."""
        if self.perf_stat_thread:
            self.perf_stat_thread.join()
            self.perf_stat_thread = None

    async def perf_stat_report(self, result_dir: Path) -> None:
        """Copy perf stat results to result directory."""
        assert result_dir.exists(), f"Result directory {result_dir} must exist"
        perf_stat_path = Path(Server.perf_stats_path)
        local_path = result_dir / "perf_stat.txt"
        await self.get_remote_file(perf_stat_path, local_path)

    # =============================================================================
    # CPU CONSISTENCY TUNING METHODS
    # =============================================================================

    async def enable_cpu_consistency_mode(self) -> None:
        """Configure CPU settings for consistent benchmarks across ARM/x86 platforms."""
        # Check if CPU frequency scaling is supported
        cpufreq_exists = await self.check_file_exists(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors")
        )

        if cpufreq_exists:
            # Try performance governor first, fallback to available governors
            governors_out, _ = await self.run_host_command(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors", check=False
            )
            available_governors = governors_out.strip().split() if governors_out else []

            if "performance" in available_governors:
                await self.run_host_command("sudo cpupower frequency-set -g performance")
                logging.info("Set performance governor")
            elif "userspace" in available_governors:
                # Fallback for some ARM systems
                await self.run_host_command("sudo cpupower frequency-set -g userspace")
                logging.info("Set userspace governor (performance fallback)")
            else:
                logging.warning("No suitable performance governor available: %s", available_governors)

            # Disable turbo/boost on x86, check ARM boost paths
            boost_paths = [
                "/sys/devices/system/cpu/cpufreq/boost",  # x86 turbo boost
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_boost_frequencies",  # Some ARM
            ]
            for boost_path in boost_paths:
                if await self.check_file_exists(Path(boost_path)):
                    await self.run_host_command(f"echo 0 | sudo tee {boost_path}", check=False)
                    logging.info("Disabled boost/turbo at %s", boost_path)
                    break
        else:
            logging.info(
                "No CPU frequency scaling - likely fixed-frequency processor (Graviton/server-class)"
            )

        # Handle idle states across platforms
        idle_result, _ = await self.run_host_command("cpupower idle-info", check=False)
        if "No idle states" in idle_result or "CPUidle driver: none" in idle_result:
            logging.info("No CPU idle states - processor maintains consistent performance")
        else:
            # Disable common idle states (C1, C2) that can cause latency spikes
            for state in [1, 2, 3]:  # C1, C2, C3
                await self.run_host_command(f"sudo cpupower idle-set -d {state}", check=False)
            logging.info("Disabled CPU idle states for latency consistency")

        # Cross-platform scheduler optimizations
        scheduler_settings = [
            ("/proc/sys/kernel/sched_energy_aware", "0", "energy-aware scheduling"),
            ("/proc/sys/kernel/sched_autogroup_enabled", "0", "automatic process grouping"),
        ]

        for path, value, description in scheduler_settings:
            if await self.check_file_exists(Path(path)):
                await self.run_host_command(f"echo {value} | sudo tee {path}", check=False)
                logging.info("Disabled %s for consistent performance", description)

    async def disable_cpu_consistency_mode(self) -> None:
        """Restore default CPU settings across ARM/x86 platforms."""
        # Check if CPU frequency scaling is supported
        cpufreq_exists = await self.check_file_exists(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors")
        )

        if cpufreq_exists:
            # Restore appropriate default governor based on platform
            governors_out, _ = await self.run_host_command(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors", check=False
            )
            available_governors = governors_out.strip().split() if governors_out else []

            # Prefer schedutil (modern), fallback to ondemand (older systems)
            if "schedutil" in available_governors:
                await self.run_host_command("sudo cpupower frequency-set -g schedutil")
                logging.info("Restored schedutil governor")
            elif "ondemand" in available_governors:
                await self.run_host_command("sudo cpupower frequency-set -g ondemand")
                logging.info("Restored ondemand governor")
            else:
                logging.warning("No suitable default governor found: %s", available_governors)

            # Re-enable turbo/boost
            boost_paths = [
                "/sys/devices/system/cpu/cpufreq/boost",
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_boost_frequencies",
            ]
            for boost_path in boost_paths:
                if await self.check_file_exists(Path(boost_path)):
                    await self.run_host_command(f"echo 1 | sudo tee {boost_path}", check=False)
                    logging.info("Re-enabled boost/turbo at %s", boost_path)
                    break
        else:
            logging.info("Fixed-frequency processor - no frequency settings to restore")

        # Re-enable idle states if they were disabled
        idle_result, _ = await self.run_host_command("cpupower idle-info", check=False)
        if "No idle states" not in idle_result and "CPUidle driver: none" not in idle_result:
            for state in [1, 2, 3]:  # C1, C2, C3
                await self.run_host_command(f"sudo cpupower idle-set -e {state}", check=False)
            logging.info("Re-enabled CPU idle states")

        # Restore default scheduler settings
        scheduler_settings = [
            ("/proc/sys/kernel/sched_energy_aware", "1", "energy-aware scheduling"),
            ("/proc/sys/kernel/sched_autogroup_enabled", "1", "automatic process grouping"),
        ]

        for path, value, description in scheduler_settings:
            if await self.check_file_exists(Path(path)):
                await self.run_host_command(f"echo {value} | sudo tee {path}", check=False)
                logging.info("Re-enabled %s", description)

    # =============================================================================
    # SERVER LIFECYCLE METHODS
    # =============================================================================

    async def __pre_start(self) -> None:
        """Configuration and preparation before starting Valkey server"""
        await self.ensure_host_cpu_allocation()

        if config.check_feature(config.Features.ENABLE_CPU_CONSISTENCY_MODE):
            await self.enable_cpu_consistency_mode()
        else:
            await self.disable_cpu_consistency_mode()

        # Enable memory overcommit
        await self.run_host_command("sudo sh -c 'echo 1 > /proc/sys/vm/overcommit_memory'")

        await self.stop()
        await asyncio.sleep(1)  # short delay to it doesn't get our new server (TODO verify this)

    @staticmethod
    def getNumCPUs(io_threads: int) -> int:
        """Get number of CPUs allocated for server with specified io-threads parameter"""
        return io_threads + 2  # (io-threads + extra for bio threads, aof rewrite, and bgsave)

    async def start(self, cached_binary_path: Path, io_threads: int) -> None:
        """Ensure specified build is running on the server."""
        if io_threads < 1:
            io_threads = 1
        self.threads = io_threads
        await self.__pre_start()

        # Allocate CPUs for this server
        needed_cpus = Server.getNumCPUs(self.threads)
        self.server_cpus = await self._allocate_server_cpus(needed_cpus)

        self.args = []
        if self.threads > 1:
            self.args += ["--io-threads", str(self.threads)]

        # Configure CPU pinning using Valkey's built-in options
        # Main thread + I/O threads: CPUs 0 through N
        main_io_cpu_list = ",".join(map(str, self.server_cpus[: self.threads + 1]))
        self.args += ["--server-cpulist", main_io_cpu_list]

        # Background threads: remaining CPUs (bio, AOF rewrite, bgsave)
        bg_thread_start_cpu = self.threads
        assert bg_thread_start_cpu < len(self.server_cpus), "Not enough CPUs allocated for background threads"
        bg_cpu_list = ",".join(map(str, self.server_cpus[bg_thread_start_cpu:]))
        self.args += ["--bio-cpulist", bg_cpu_list]
        self.args += ["--aof-rewrite-cpulist", bg_cpu_list]
        self.args += ["--bgsave-cpulist", bg_cpu_list]

        command = (
            f"{cached_binary_path} --port {self.port} "
            f"--save --protected-mode no --daemonize yes " + " ".join(self.args)
        )

        # Optionally bind memory to NUMA node for consistent performance
        if config.check_feature(config.Features.BIND_NUMA_MEMORY):
            numa_node = self._cpu_allocator.get_net_interface_numa(self.ip)
            command = f"numactl --membind={numa_node} {command}"
            logging.info("Binding memory to NUMA node %d", numa_node)

        out, err = await self.run_host_command(command)

        pid_out, _ = await self.run_host_command(f"lsof -ti :{self.port}")
        self.valkey_pid = int(pid_out.strip())

        await self.wait_until_ready()

        # Optionally pin individual threads to specific CPUs for precise control
        if config.check_feature(config.Features.PIN_VALKEY_THREADS):
            await self._pin_valkey_threads()

    async def wait_until_ready(self) -> None:
        """Wait until the server is ready to accept commands."""
        for _ in range(10):
            try:
                out = await self.run_valkey_command("PING")
                if out == "PONG":
                    return
                print(out)
            except asyncssh.ProcessError:
                print("cli error")
            time.sleep(1)

        raise RuntimeError("Server did not start successfully")

    async def kill_all_valkey_instances_on_host(self):
        """Stop all instances of valkey server, regardless of which port they're running on"""
        await self.run_host_command(f"pkill -f {VALKEY_BINARY}", check=False)

        # Clear all non-IRQ allocations for this host
        all_allocations = self._cpu_allocator.get_all_allocations(self.ip)
        for tag in list(all_allocations.keys()):
            if tag.purpose != "irq":  # Keep IRQ allocations
                self._cpu_allocator.release(self.ip, tag)

    async def stop(self):
        # Release allocated CPUs
        if self.server_cpus:
            self._release_server_cpus()

        if self.valkey_pid == -1:
            return

        # ensure process using the port is one of our expected valkey processes
        # (in case an unexpected process is already using the port for some reason)
        name, _ = await self.run_host_command(f"ps -p {self.valkey_pid}")
        name = name.strip().split()[-1]
        assert name == VALKEY_BINARY

        await self.run_host_command(f"kill -9 {self.valkey_pid}")
        self.valkey_pid = -1

        # clean up any rdb files from replication or snapshotting
        # valkey will automatically load "dump.rdb" if it is present
        await self.run_host_command(f"cd {Server.path_root} && rm -f *.rdb", check=False)
        # clean up any files created by profiling or other metric collection
        await self.__data_collection_cleanup()

    # =============================================================================
    # VALKEY COMMAND EXECUTION METHODS
    # =============================================================================

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
            f"--threads 16 -q --sequential -r {keyspace_size} -n {keyspace_size} "
        )
        sequential_command += command
        self.logger.info("Benchmark Command: %s", sequential_command)
        await async_run(sequential_command)

    async def info(self, section: str) -> dict[str, str]:
        """Run the 'info' command on the server and return the specified section."""
        response = await self.run_valkey_command(f"info {section}")
        if not response:
            raise RuntimeError(f"Failed to run 'info {section}' on server {self.ip}")
        lines = response.strip().split("\n")
        keypairs: dict[str, str] = {}
        for item in lines:
            if ":" in item:
                (key, value) = item.split(":", 1)
                keypairs[key.strip()] = value.strip()
        return keypairs

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

    # =============================================================================
    # REPLICATION METHODS
    # =============================================================================

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

    # =============================================================================
    # BINARY MANAGEMENT METHODS
    # =============================================================================

    async def ensure_binary_cached(
        self, source: Optional[str] = None, specifier: Optional[str] = None, make_args: Optional[str] = None
    ) -> Path:
        """Ensure that an arbitrary binary has been cached. This needs to be done before running multiple
        instances on a single host."""
        # don't break assumption that version running matches state
        # also don't want to be running builds while benchmarks are running
        assert self.valkey_pid == -1, "Cannot change binary while server is running"

        if source:
            self.source = source
        if specifier:
            self.specifier = specifier
        if make_args is not None:
            self.make_args = make_args

        if self.source == config.MANUALLY_UPLOADED:
            return await self.__ensure_binary_uploaded(self.specifier)
        else:
            assert self.source in config.REPO_NAMES, f"Unknown source: {self.source}"
            return await self.__ensure_build_cached()

    def get_build_hash(self):
        """
        Get unique hash for current version of valkey running on the server.
        Typically this is the commit hash of the source code used to build the server.
        """
        return self.hash

    def __get_cached_build_path(self) -> Path:
        assert self.source is not None and self.hash is not None
        # Use actual make_args value - empty string is a valid override
        make_args_hash = hashlib.md5(self.make_args.encode()).hexdigest()[:16]
        return Server.remote_build_cache / self.source / self.hash / make_args_hash

    def __get_source_binary_path(self) -> Path:
        assert self.source is not None
        return Server.path_root / self.source / "src"

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
                # Use actual make_args value - empty string is a valid override
                make_command = f"cd {source_path}; make distclean && make -j"
                if self.make_args:
                    make_command += f" {self.make_args}"
                # Note: empty make_args means no additional flags (bare make -j)
                await self.run_host_command(make_command)
            except asyncssh.ProcessError as e:
                self.logger.error("Build failed %d:\n%s", e.returncode, e.stderr)
                raise e
            build_binary = source_path / VALKEY_BINARY

            await self.run_host_command(f"mkdir -p {cached_build_path}")
            await self.run_host_command(f"cp {build_binary} {cached_binary_path}")

        return cached_binary_path

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

    @staticmethod
    async def delete_entire_build_cache(server_ips) -> None:
        """Delete the entire build cache on all servers. This is a destructive operation."""

        async def delete_host_cache(ip):
            async with asyncssh.connect(ip, client_keys=[str(config.SSH_KEYFILE)]) as conn:
                await conn.run(f"rm -rf {Server.remote_build_cache}", check=False)

        await asyncio.gather(*(delete_host_cache(ip) for ip in server_ips))

    # =============================================================================
    # SSH AND FILE OPERATIONS
    # =============================================================================

    async def __ensure_ssh_connection(self) -> None:
        if not self.ssh:
            if self.ip in ["127.0.0.1", "localhost"]:
                self.ssh = await asyncssh.connect(
                    self.ip, client_keys=[str(config.SSH_KEYFILE)], known_hosts=None
                )
            elif self.username:
                self.ssh = await asyncssh.connect(
                    self.ip,
                    username=self.username,
                    client_keys=[str(config.SSH_KEYFILE)],
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

    async def check_file_exists(self, path: Path) -> bool:
        """Check if a file exists on the server."""
        result, _ = await self.run_host_command(f"[[ -f {path} ]] && echo 1 || echo 0;")
        return result.strip() == "1"

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
