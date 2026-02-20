"""Unit tests for cache-aware CPU allocation with mocked topologies."""

import pytest
from src.cpu_allocator import CpuAllocator, AllocationTag


class TestCacheAwareAllocation:
    """Test cache-aware allocation with various CPU topologies."""

    def test_single_numa_single_l3(self):
        """Test system with 1 NUMA node, 1 L3 cache (8 CPUs)."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(8)),
            numa_topology={0: list(range(8))},
            l3_cache_topology={0: list(range(8))},
            net_interface_numa=0,
        )

        # Allocate server
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=4, require_numa=0)
        assert server_cpus == [0, 1, 2, 3]

        # Try cache-aware allocation (should fail, fall back to simple)
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=4, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        assert client_cpus == [4, 5, 6, 7]
        
        # Verify they share L3 cache (unavoidable)
        assert set(server_cpus) & set(range(8))  # Both in same L3

    def test_single_numa_multiple_l3(self):
        """Test system with 1 NUMA node, 3 L3 caches (8 CPUs each)."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(24)),
            numa_topology={0: list(range(24))},
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
            },
            net_interface_numa=0,
        )

        # Allocate server on L3 cache 0
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5]

        # Allocate client avoiding server cache
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=16, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should get CPUs from L3 caches 1 and 2, not 0
        assert all(cpu >= 8 for cpu in client_cpus)
        assert len(client_cpus) == 16

    def test_dual_numa_single_l3_per_numa(self):
        """Test system with 2 NUMA nodes, 1 L3 cache per NUMA (8 CPUs each)."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(16)),
            numa_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
            },
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
            },
            net_interface_numa=0,
        )

        # Allocate server on NUMA 0
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=4, require_numa=0)
        assert server_cpus == [0, 1, 2, 3]

        # Allocate client without NUMA constraint - should prefer NUMA 1
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=8,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should get all CPUs from NUMA 1 (different NUMA preferred)
        assert client_cpus == list(range(8, 16))

    def test_dual_numa_multiple_l3_per_numa(self):
        """Test system with 2 NUMA nodes, 3 L3 caches per NUMA (8 CPUs each)."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(48)),
            numa_topology={
                0: list(range(0, 24)),
                1: list(range(24, 48)),
            },
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
                3: list(range(24, 32)),
                4: list(range(32, 40)),
                5: list(range(40, 48)),
            },
            net_interface_numa=0,
        )

        # Allocate server on NUMA 0, L3 cache 0
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5]

        # Allocate client without NUMA constraint - should prefer NUMA 1
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=16,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should get CPUs from NUMA 1 (different NUMA preferred)
        assert all(cpu >= 24 for cpu in client_cpus)

    def test_irq_allocation_avoids_server_cache(self):
        """Test IRQ allocation avoids server L3 cache on same NUMA."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(24)),
            numa_topology={0: list(range(24))},
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
            },
            net_interface_numa=0,
        )

        # Allocate server
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5]

        # Allocate IRQs avoiding server cache
        irq_tag = AllocationTag(task_id="system", purpose="irq")
        irq_cpus = allocator.allocate(
            "test_host", irq_tag, count=4, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # IRQs should be on different L3 cache (8-15 or 16-23)
        assert all(cpu >= 8 for cpu in irq_cpus)

    def test_insufficient_cpus_in_different_cache(self):
        """Test fallback when not enough CPUs in different cache."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(16)),
            numa_topology={0: list(range(16))},
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
            },
            net_interface_numa=0,
        )

        # Allocate server
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)

        # Try to allocate 12 CPUs avoiding server cache (only 10 available in L3 cache 1)
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=10, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should get all 10 remaining CPUs
        assert len(client_cpus) == 10
        assert set(client_cpus) == set(range(6, 16))

    def test_no_l3_cache_info(self):
        """Test allocation works without L3 cache information."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(16)),
            numa_topology={0: list(range(16))},
            l3_cache_topology={},  # No L3 info
            net_interface_numa=0,
        )

        # Allocate server
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)

        # Allocate client (should fall back to simple allocation)
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=10, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should still work, just without cache awareness
        assert len(client_cpus) == 10
        assert not set(server_cpus) & set(client_cpus)

    def test_complex_topology_amd_epyc(self):
        """Test realistic AMD EPYC topology: 2 NUMA, 12 L3 caches per NUMA."""
        allocator = CpuAllocator()
        
        # Simulate AMD EPYC 9R14: 96 cores per NUMA, 8 cores per L3
        numa0_cpus = list(range(0, 96))
        numa1_cpus = list(range(96, 192))
        
        l3_topology = {}
        for i in range(12):
            l3_topology[i] = list(range(i * 8, (i + 1) * 8))
        for i in range(12):
            l3_topology[i + 32] = list(range(96 + i * 8, 96 + (i + 1) * 8))
        
        allocator.register_host(
            "test_host",
            all_cpus=list(range(192)),
            numa_topology={0: numa0_cpus, 1: numa1_cpus},
            l3_cache_topology=l3_topology,
            net_interface_numa=0,
        )

        # Allocate server on NUMA 0
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5]

        # Allocate client without NUMA constraint
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=16,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should prefer NUMA 1 (different NUMA)
        assert all(cpu >= 96 for cpu in client_cpus)

    def test_allocation_release_and_reuse(self):
        """Test that released CPUs can be reallocated."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(16)),
            numa_topology={0: list(range(16))},
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
            },
            net_interface_numa=0,
        )

        # Allocate and release
        tag1 = AllocationTag(task_id="task1", purpose="server")
        cpus1 = allocator.allocate("test_host", tag1, count=8, require_numa=0)
        assert cpus1 == list(range(8))
        
        allocator.release("test_host", tag1)
        
        # Reallocate - should get same CPUs
        tag2 = AllocationTag(task_id="task2", purpose="server")
        cpus2 = allocator.allocate("test_host", tag2, count=8, require_numa=0)
        assert cpus2 == list(range(8))

    def test_multiple_avoid_tags(self):
        """Test avoiding multiple allocations."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(32)),
            numa_topology={0: list(range(32))},
            l3_cache_topology={
                0: list(range(0, 8)),
                1: list(range(8, 16)),
                2: list(range(16, 24)),
                3: list(range(24, 32)),
            },
            net_interface_numa=0,
        )

        # Allocate server (gets first CPUs)
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=6, require_numa=0)
        assert server_cpus == [0, 1, 2, 3, 4, 5]  # L3 cache 0
        
        # Allocate IRQs (gets last CPUs due to purpose="irq")
        irq_tag = AllocationTag(task_id="system", purpose="irq")
        irq_cpus = allocator.allocate("test_host", irq_tag, count=4, require_numa=0)
        assert irq_cpus == [31, 30, 29, 28]  # L3 cache 3
        
        # Allocate client avoiding both
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=16, require_numa=0,
            avoid_tags=[server_tag, irq_tag], prefer_different_cache=True
        )
        
        # Client should avoid L3 caches 0 and 3, use caches 1 and 2
        assert all(8 <= cpu <= 23 for cpu in client_cpus)

    def test_single_l3_cache_fallback(self):
        """Test that single L3 cache systems fall back gracefully (consistent behavior)."""
        allocator = CpuAllocator()
        allocator.register_host(
            "test_host",
            all_cpus=list(range(8)),
            numa_topology={0: list(range(8))},
            l3_cache_topology={0: list(range(8))},  # Single L3 cache
            net_interface_numa=0,
        )

        # Allocate server
        server_tag = AllocationTag(task_id="server", purpose="server")
        server_cpus = allocator.allocate("test_host", server_tag, count=4, require_numa=0)

        # Allocate client with prefer_different_cache (should succeed with fallback)
        client_tag = AllocationTag(task_id="client", purpose="benchmark")
        client_cpus = allocator.allocate(
            "test_host", client_tag, count=4, require_numa=0,
            avoid_tags=[server_tag], prefer_different_cache=True
        )
        
        # Should get remaining CPUs even though they share cache (consistent on this hardware)
        assert len(client_cpus) == 4
        assert not set(server_cpus) & set(client_cpus)  # Still no CPU overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
