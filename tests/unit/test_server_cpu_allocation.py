"""Unit tests for Server CPU allocation integration with CpuAllocator."""

import pytest
from unittest.mock import AsyncMock, patch
from src.server import Server
from src.cpu_allocator import CpuAllocator, AllocationTag


class TestServerCpuAllocation:
    """Test Server's integration with CpuAllocator."""

    def setup_method(self):
        """Reset CPU allocator before each test."""
        Server._cpu_allocator = CpuAllocator()

    @pytest.mark.asyncio
    async def test_allocate_uses_network_interface_numa_node(self):
        """Test that Server allocates CPUs on the network interface NUMA node."""
        server = Server("192.168.1.1", 9000)

        # Register with network interface on NUMA 1
        Server._cpu_allocator.register_host(
            "192.168.1.1",
            all_cpus=list(range(8)),
            numa_topology={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
            net_interface_numa=1,
        )

        cpus = await server._allocate_server_cpus(2)

        # Should allocate from NUMA node 1 (network interface node)
        assert all(cpu in [4, 5, 6, 7] for cpu in cpus)

    @pytest.mark.asyncio
    async def test_allocate_creates_proper_tag(self):
        """Test that Server creates allocation tag with correct format."""
        server = Server("192.168.1.1", 9000)

        Server._cpu_allocator.register_host(
            "192.168.1.1", all_cpus=list(range(4)), numa_topology={0: list(range(4))}, net_interface_numa=0
        )

        await server._allocate_server_cpus(2)

        # Check tag was created correctly
        assert server._allocation_tag is not None
        assert server._allocation_tag.task_id == "server_192.168.1.1_9000"
        assert server._allocation_tag.purpose == "server"

    @pytest.mark.asyncio
    async def test_release_clears_allocation_tag(self):
        """Test that releasing CPUs clears the allocation tag."""
        server = Server("192.168.1.1", 9000)

        Server._cpu_allocator.register_host(
            "192.168.1.1", all_cpus=list(range(4)), numa_topology={0: list(range(4))}, net_interface_numa=0
        )

        cpus = await server._allocate_server_cpus(2)
        server.server_cpus = cpus

        assert server._allocation_tag is not None

        server._release_server_cpus()

        assert server._allocation_tag is None
        assert server.server_cpus == []

    @pytest.mark.asyncio
    async def test_stop_releases_allocation(self):
        """Test that stop() releases CPU allocation."""
        server = Server("192.168.1.1", 9000)
        server.valkey_pid = 12345

        Server._cpu_allocator.register_host(
            "192.168.1.1", all_cpus=list(range(8)), numa_topology={0: list(range(8))}, net_interface_numa=0
        )

        cpus = await server._allocate_server_cpus(3)
        server.server_cpus = cpus
        tag = server._allocation_tag

        server.run_host_command = AsyncMock(
            side_effect=[
                ("valkey-server", ""),
                ("", ""),
                ("", ""),
                ("", ""),
            ]
        )

        await server.stop()

        # Verify allocation was released
        assert Server._cpu_allocator.get_allocation("192.168.1.1", tag) is None

    @pytest.mark.asyncio
    async def test_kill_all_preserves_irq_allocations(self):
        """Test that kill_all clears server allocations but preserves IRQ allocations."""
        server = Server("192.168.1.1", 9000)
        server.run_host_command = AsyncMock(return_value=("", ""))

        Server._cpu_allocator.register_host(
            "192.168.1.1", all_cpus=list(range(8)), numa_topology={0: list(range(8))}, net_interface_numa=0
        )

        # Allocate IRQ
        irq_tag = AllocationTag("system", "irq")
        Server._cpu_allocator.allocate("192.168.1.1", irq_tag, count=2, require_numa=0)

        # Allocate server
        cpus = await server._allocate_server_cpus(3)
        server.server_cpus = cpus
        server_tag = server._allocation_tag

        await server.kill_all_valkey_instances_on_host()

        # IRQ should remain, server should be cleared
        assert Server._cpu_allocator.get_allocation("192.168.1.1", irq_tag) is not None
        assert Server._cpu_allocator.get_allocation("192.168.1.1", server_tag) is None

    @pytest.mark.asyncio
    async def test_get_available_cpu_count_uses_network_numa(self):
        """Test that get_available_cpu_count queries the network interface NUMA node."""
        server = Server("192.168.1.1", 9000)

        async def mock_setup():
            Server._cpu_allocator.register_host(
                "192.168.1.1",
                all_cpus=list(range(8)),
                numa_topology={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
                net_interface_numa=1,
            )
            # Allocate some CPUs on NUMA 1
            tag = AllocationTag("other", "server")
            Server._cpu_allocator.allocate("192.168.1.1", tag, count=2, require_numa=1)

        server.ensure_host_cpu_allocation = mock_setup

        available = await server.get_available_cpu_count()

        # Should count only NUMA 1 CPUs: 4 total - 2 allocated = 2 available
        assert available == 2

    @pytest.mark.asyncio
    async def test_validate_cpu_allocation_checks_network_numa(self):
        """Test that validation fails when insufficient CPUs on network interface NUMA node."""
        server = Server("192.168.1.1", 9000)
        server.threads = 3

        Server._cpu_allocator.register_host(
            "192.168.1.1",
            all_cpus=list(range(8)),
            numa_topology={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
            net_interface_numa=1,
        )

        # Allocate IRQs on NUMA 1
        irq_tag = AllocationTag("system", "irq")
        Server._cpu_allocator.allocate("192.168.1.1", irq_tag, count=2, require_numa=1)

        # threads=3 needs 5 CPUs, but only 2 available on NUMA 1
        with pytest.raises(RuntimeError, match="Insufficient CPUs.*NUMA node 1"):
            server._validate_sufficient_cpus()

    def test_get_num_cpus_calculation(self):
        """Test CPU count calculation for io-threads."""
        # io-threads + 2 for background threads
        assert Server.getNumCPUs(1) == 3
        assert Server.getNumCPUs(4) == 6
        assert Server.getNumCPUs(8) == 10

    @pytest.mark.asyncio
    async def test_multiple_servers_different_ports_same_host(self):
        """Test that multiple servers on same host get non-overlapping CPUs."""
        server1 = Server("192.168.1.1", 9000)
        server2 = Server("192.168.1.1", 9001)

        Server._cpu_allocator.register_host(
            "192.168.1.1", all_cpus=list(range(8)), numa_topology={0: list(range(8))}, net_interface_numa=0
        )

        cpus1 = await server1._allocate_server_cpus(3)
        cpus2 = await server2._allocate_server_cpus(2)

        # Should not overlap
        assert set(cpus1).isdisjoint(set(cpus2))

        # Both should have proper tags
        assert server1._allocation_tag.task_id == "server_192.168.1.1_9000"
        assert server2._allocation_tag.task_id == "server_192.168.1.1_9001"

    @pytest.mark.asyncio
    async def test_allocation_fails_when_numa_node_full(self):
        """Test that allocation fails when network interface NUMA node is full."""
        server = Server("192.168.1.1", 9000)

        Server._cpu_allocator.register_host(
            "192.168.1.1",
            all_cpus=list(range(8)),
            numa_topology={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
            net_interface_numa=0,
        )

        # Fill up NUMA node 0
        tag1 = AllocationTag("other1", "server")
        tag2 = AllocationTag("other2", "server")
        Server._cpu_allocator.allocate("192.168.1.1", tag1, count=2, require_numa=0)
        Server._cpu_allocator.allocate("192.168.1.1", tag2, count=2, require_numa=0)

        # Only 0 CPUs left in NUMA 0, requesting 2 should fail
        with pytest.raises(RuntimeError, match="Insufficient CPUs"):
            await server._allocate_server_cpus(2)

    @pytest.mark.asyncio
    async def test_validate_sufficient_cpus_passes_when_enough_available(self):
        """Test validation passes when sufficient CPUs are available."""
        server = Server("192.168.1.1", 9000)
        server.threads = 2  # Needs 4 CPUs (2 + 2)

        Server._cpu_allocator.register_host(
            "192.168.1.1",
            all_cpus=list(range(8)),
            numa_topology={0: list(range(8))},
            net_interface_numa=0,
        )

        # Should not raise
        server._validate_sufficient_cpus()

    @pytest.mark.asyncio
    async def test_detect_l3_cache_topology_parses_sysfs(self):
        """Test L3 cache detection with mocked sysfs responses."""
        server = Server("192.168.1.1", 9000)

        # Mock responses: CPUs 0-3 share L3 cache 0, CPUs 4-7 share L3 cache 1
        async def mock_run_host_command(cmd, check=True):
            if "cpu0/cache/index3/id" in cmd:
                return ("0", "")
            elif "cpu1/cache/index3/id" in cmd:
                return ("0", "")
            elif "cpu2/cache/index3/id" in cmd:
                return ("0", "")
            elif "cpu3/cache/index3/id" in cmd:
                return ("0", "")
            elif "cpu4/cache/index3/id" in cmd:
                return ("1", "")
            elif "cpu5/cache/index3/id" in cmd:
                return ("1", "")
            elif "cpu6/cache/index3/id" in cmd:
                return ("1", "")
            elif "cpu7/cache/index3/id" in cmd:
                return ("1", "")
            return ("-1", "")

        server.run_host_command = mock_run_host_command

        result = await server._detect_l3_cache_topology(list(range(8)))

        assert result == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}

    @pytest.mark.asyncio
    async def test_detect_l3_cache_topology_handles_missing_sysfs(self):
        """Test graceful fallback when L3 cache sysfs doesn't exist."""
        server = Server("192.168.1.1", 9000)

        async def mock_run_host_command(cmd, check=True):
            return ("-1", "")  # Simulates missing sysfs

        server.run_host_command = mock_run_host_command

        result = await server._detect_l3_cache_topology(list(range(4)))

        assert result == {}
