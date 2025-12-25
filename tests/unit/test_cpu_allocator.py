"""Unit tests for CPU allocator."""

import pytest

from src.cpu_allocator import AllocationTag, CpuAllocator


class TestAllocationTag:
    def test_tag_creation(self):
        tag = AllocationTag(task_id="task_123", purpose="server")
        assert tag.task_id == "task_123"
        assert tag.purpose == "server"

    def test_tag_string_representation(self):
        tag = AllocationTag(task_id="task_456", purpose="benchmark")
        assert str(tag) == "task_456:benchmark"

    def test_tag_immutable(self):
        tag = AllocationTag(task_id="task_789", purpose="irq")
        with pytest.raises(Exception):  # dataclass frozen
            tag.task_id = "new_id"

    def test_tag_equality(self):
        tag1 = AllocationTag(task_id="task_1", purpose="server")
        tag2 = AllocationTag(task_id="task_1", purpose="server")
        tag3 = AllocationTag(task_id="task_1", purpose="benchmark")
        assert tag1 == tag2
        assert tag1 != tag3


class TestCpuAllocator:
    def test_register_host_simple(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        assert allocator.get_available_count("192.168.1.1") == 4

    def test_register_host_with_numa(self):
        allocator = CpuAllocator()
        numa_topology = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        allocator.register_host("192.168.1.1", all_cpus=list(range(8)), numa_topology=numa_topology)

        assert allocator.get_available_count("192.168.1.1") == 8
        assert allocator.get_available_count("192.168.1.1", prefer_numa=0) == 4
        assert allocator.get_available_count("192.168.1.1", prefer_numa=1) == 4

    def test_register_host_with_net_interface_numa(self):
        allocator = CpuAllocator()
        numa_topology = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        allocator.register_host(
            "192.168.1.1",
            all_cpus=list(range(8)),
            numa_topology=numa_topology,
            net_interface_numa=1,
        )

        assert allocator.get_net_interface_numa("192.168.1.1") == 1

    def test_allocate_basic(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")
        cpus = allocator.allocate("192.168.1.1", tag, count=2)

        assert len(cpus) == 2
        assert cpus == [0, 1]
        assert allocator.get_available_count("192.168.1.1") == 2

    def test_allocate_multiple_tags(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=list(range(8)))

        tag1 = AllocationTag(task_id="task_1", purpose="server")
        tag2 = AllocationTag(task_id="task_1", purpose="benchmark")
        tag3 = AllocationTag(task_id="task_2", purpose="server")

        cpus1 = allocator.allocate("192.168.1.1", tag1, count=2)
        cpus2 = allocator.allocate("192.168.1.1", tag2, count=3)
        cpus3 = allocator.allocate("192.168.1.1", tag3, count=1)

        assert cpus1 == [0, 1]
        assert cpus2 == [2, 3, 4]
        assert cpus3 == [5]
        assert allocator.get_available_count("192.168.1.1") == 2

    def test_allocate_with_numa_requirement(self):
        allocator = CpuAllocator()
        numa_topology = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        allocator.register_host("192.168.1.1", all_cpus=list(range(8)), numa_topology=numa_topology)

        tag = AllocationTag(task_id="task_1", purpose="server")
        cpus = allocator.allocate("192.168.1.1", tag, count=2, require_numa=1)

        assert cpus == [4, 5]  # From NUMA node 1

    def test_allocate_numa_requirement_insufficient_cpus(self):
        allocator = CpuAllocator()
        numa_topology = {0: [0, 1], 1: [2, 3, 4, 5]}
        allocator.register_host("192.168.1.1", all_cpus=list(range(6)), numa_topology=numa_topology)

        tag = AllocationTag(task_id="task_1", purpose="server")
        # NUMA node 0 only has 2 CPUs, requesting 4 should fail
        with pytest.raises(RuntimeError, match="Insufficient CPUs.*NUMA node 0"):
            allocator.allocate("192.168.1.1", tag, count=4, require_numa=0)

    def test_allocate_numa_requirement_invalid_node(self):
        allocator = CpuAllocator()
        numa_topology = {0: [0, 1, 2, 3]}
        allocator.register_host("192.168.1.1", all_cpus=list(range(4)), numa_topology=numa_topology)

        tag = AllocationTag(task_id="task_1", purpose="server")
        with pytest.raises(ValueError, match="NUMA node 1 not found"):
            allocator.allocate("192.168.1.1", tag, count=2, require_numa=1)

    def test_allocate_unregistered_host(self):
        allocator = CpuAllocator()
        tag = AllocationTag(task_id="task_1", purpose="server")

        with pytest.raises(ValueError, match="Host .* not registered"):
            allocator.allocate("192.168.1.1", tag, count=2)

    def test_allocate_duplicate_tag(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")
        allocator.allocate("192.168.1.1", tag, count=2)

        with pytest.raises(ValueError, match="already exists"):
            allocator.allocate("192.168.1.1", tag, count=1)

    def test_allocate_insufficient_cpus(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")

        with pytest.raises(RuntimeError, match="Insufficient CPUs"):
            allocator.allocate("192.168.1.1", tag, count=5)

    def test_release_basic(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")
        allocated = allocator.allocate("192.168.1.1", tag, count=2)
        assert allocator.get_available_count("192.168.1.1") == 2

        released = allocator.release("192.168.1.1", tag)
        assert released == allocated
        assert allocator.get_available_count("192.168.1.1") == 4

    def test_release_makes_cpus_available(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag1 = AllocationTag(task_id="task_1", purpose="server")
        tag2 = AllocationTag(task_id="task_2", purpose="server")

        allocator.allocate("192.168.1.1", tag1, count=4)
        assert allocator.get_available_count("192.168.1.1") == 0

        allocator.release("192.168.1.1", tag1)
        cpus = allocator.allocate("192.168.1.1", tag2, count=4)
        assert cpus == [0, 1, 2, 3]

    def test_release_unregistered_host(self):
        allocator = CpuAllocator()
        tag = AllocationTag(task_id="task_1", purpose="server")

        with pytest.raises(ValueError, match="Host .* not registered"):
            allocator.release("192.168.1.1", tag)

    def test_release_nonexistent_tag(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")

        with pytest.raises(ValueError, match="not found"):
            allocator.release("192.168.1.1", tag)

    def test_get_allocation(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")
        allocated = allocator.allocate("192.168.1.1", tag, count=2)

        result = allocator.get_allocation("192.168.1.1", tag)
        assert result == allocated

    def test_get_allocation_nonexistent(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        tag = AllocationTag(task_id="task_1", purpose="server")
        result = allocator.get_allocation("192.168.1.1", tag)
        assert result is None

    def test_get_allocation_unregistered_host(self):
        allocator = CpuAllocator()
        tag = AllocationTag(task_id="task_1", purpose="server")
        result = allocator.get_allocation("192.168.1.1", tag)
        assert result is None

    def test_get_available_count_unregistered_host(self):
        allocator = CpuAllocator()
        assert allocator.get_available_count("192.168.1.1") == 0

    def test_get_all_allocations(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=list(range(8)))

        tag1 = AllocationTag(task_id="task_1", purpose="server")
        tag2 = AllocationTag(task_id="task_1", purpose="benchmark")

        cpus1 = allocator.allocate("192.168.1.1", tag1, count=2)
        cpus2 = allocator.allocate("192.168.1.1", tag2, count=3)

        allocations = allocator.get_all_allocations("192.168.1.1")
        assert len(allocations) == 2
        assert allocations[tag1] == cpus1
        assert allocations[tag2] == cpus2

    def test_get_all_allocations_empty(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])

        allocations = allocator.get_all_allocations("192.168.1.1")
        assert allocations == {}

    def test_get_all_allocations_unregistered_host(self):
        allocator = CpuAllocator()
        allocations = allocator.get_all_allocations("192.168.1.1")
        assert allocations == {}

    def test_get_net_interface_numa(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3], net_interface_numa=0)
        assert allocator.get_net_interface_numa("192.168.1.1") == 0

    def test_get_net_interface_numa_not_set(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])
        assert allocator.get_net_interface_numa("192.168.1.1") is None

    def test_get_net_interface_numa_unregistered_host(self):
        allocator = CpuAllocator()
        assert allocator.get_net_interface_numa("192.168.1.1") is None

    def test_is_host_registered(self):
        allocator = CpuAllocator()
        assert not allocator.is_host_registered("192.168.1.1")
        
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])
        assert allocator.is_host_registered("192.168.1.1")
        assert not allocator.is_host_registered("192.168.1.2")

    def test_multiple_hosts(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])
        allocator.register_host("192.168.1.2", all_cpus=[0, 1, 2, 3, 4, 5])

        tag1 = AllocationTag(task_id="task_1", purpose="server")
        tag2 = AllocationTag(task_id="task_2", purpose="server")

        cpus1 = allocator.allocate("192.168.1.1", tag1, count=2)
        cpus2 = allocator.allocate("192.168.1.2", tag2, count=3)

        assert cpus1 == [0, 1]
        assert cpus2 == [0, 1, 2]
        assert allocator.get_available_count("192.168.1.1") == 2
        assert allocator.get_available_count("192.168.1.2") == 3

    def test_same_tag_different_hosts(self):
        allocator = CpuAllocator()
        allocator.register_host("192.168.1.1", all_cpus=[0, 1, 2, 3])
        allocator.register_host("192.168.1.2", all_cpus=[0, 1, 2, 3])

        # Same tag can be used on different hosts
        tag = AllocationTag(task_id="task_1", purpose="server")
        cpus1 = allocator.allocate("192.168.1.1", tag, count=2)
        cpus2 = allocator.allocate("192.168.1.2", tag, count=2)

        assert cpus1 == [0, 1]
        assert cpus2 == [0, 1]

    def test_cpu_sorting(self):
        allocator = CpuAllocator()
        # Register with unsorted CPUs
        allocator.register_host("192.168.1.1", all_cpus=[3, 1, 0, 2])

        tag = AllocationTag(task_id="task_1", purpose="server")
        cpus = allocator.allocate("192.168.1.1", tag, count=3)

        # Should return sorted CPUs
        assert cpus == [0, 1, 2]

    def test_complex_scenario(self):
        """Test a realistic scenario with IRQ, server, and benchmark allocations."""
        allocator = CpuAllocator()
        numa_topology = {0: list(range(0, 32)), 1: list(range(32, 64))}
        allocator.register_host("192.168.1.1", all_cpus=list(range(64)), numa_topology=numa_topology)

        # Allocate IRQs
        irq_tag = AllocationTag(task_id="system", purpose="irq")
        irq_cpus = allocator.allocate("192.168.1.1", irq_tag, count=4, require_numa=1)
        assert irq_cpus == [63, 62, 61, 60]  # Last CPUs from NUMA 1

        # Allocate server
        server_tag = AllocationTag(task_id="task_123", purpose="server")
        server_cpus = allocator.allocate("192.168.1.1", server_tag, count=8, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5, 6, 7]

        # Allocate benchmark
        bench_tag = AllocationTag(task_id="task_123", purpose="benchmark")
        bench_cpus = allocator.allocate("192.168.1.1", bench_tag, count=16, require_numa=0)
        assert bench_cpus == [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        # Check remaining
        assert allocator.get_available_count("192.168.1.1") == 36

        # Release server
        allocator.release("192.168.1.1", server_tag)
        assert allocator.get_available_count("192.168.1.1") == 44

        # Verify allocations
        allocations = allocator.get_all_allocations("192.168.1.1")
        assert len(allocations) == 2
        assert irq_tag in allocations
        assert bench_tag in allocations
        assert server_tag not in allocations
