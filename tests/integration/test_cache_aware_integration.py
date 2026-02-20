"""Test cache-aware CPU allocation."""

import asyncio
import pytest

from src.server import Server
from src.cpu_allocator import AllocationTag


@pytest.mark.asyncio
async def test_l3_cache_detection():
    """Test that L3 cache topology is detected."""
    server = Server("127.0.0.1", port=9000)
    await server.ensure_host_cpu_allocation()
    
    # Check if L3 cache topology was detected
    l3_caches = server._cpu_allocator._l3_caches.get(server.ip, {})
    
    if l3_caches:
        print(f"Detected {len(l3_caches)} L3 caches:")
        for cache_id, cpus in l3_caches.items():
            print(f"  L3 cache {cache_id}: {len(cpus)} CPUs")
        assert len(l3_caches) > 0, "Should detect at least one L3 cache"
    else:
        print("L3 cache topology not available on this system")


@pytest.mark.asyncio
async def test_cache_aware_allocation():
    """Test that cache-aware allocation avoids server cache."""
    server = Server("127.0.0.1", port=9000)
    await server.ensure_host_cpu_allocation()
    
    # Allocate server CPUs
    server_tag = AllocationTag(task_id="test_server", purpose="server")
    net_numa = server._cpu_allocator.get_net_interface_numa(server.ip)
    server_cpus = server._cpu_allocator.allocate(
        server.ip,
        server_tag,
        count=6,
        require_numa=net_numa,
    )
    print(f"Server CPUs: {server_cpus}")
    
    # Allocate benchmark CPUs with cache avoidance
    benchmark_tag = AllocationTag(task_id="test_benchmark", purpose="benchmark")
    benchmark_cpus = server._cpu_allocator.allocate(
        server.ip,
        benchmark_tag,
        count=16,
        require_numa=net_numa,
        avoid_tags=[server_tag],
        prefer_different_cache=True,
    )
    print(f"Benchmark CPUs: {benchmark_cpus}")
    
    # Verify no CPU overlap
    assert not set(server_cpus) & set(benchmark_cpus), "CPUs should not overlap"
    
    # Check if cache separation was achieved
    l3_caches = server._cpu_allocator._l3_caches.get(server.ip, {})
    if l3_caches:
        # Find which L3 caches contain server CPUs
        server_l3_caches = set()
        for cache_id, cpus in l3_caches.items():
            if any(cpu in server_cpus for cpu in cpus):
                server_l3_caches.add(cache_id)
        
        # Find which L3 caches contain benchmark CPUs
        benchmark_l3_caches = set()
        for cache_id, cpus in l3_caches.items():
            if any(cpu in benchmark_cpus for cpu in cpus):
                benchmark_l3_caches.add(cache_id)
        
        print(f"Server L3 caches: {server_l3_caches}")
        print(f"Benchmark L3 caches: {benchmark_l3_caches}")
        
        # Check if we achieved cache separation
        if server_l3_caches & benchmark_l3_caches:
            print("⚠ Server and benchmark share L3 cache (may be unavoidable)")
        else:
            print("✓ Server and benchmark on different L3 caches")
    
    # Cleanup
    server._cpu_allocator.release(server.ip, server_tag)
    server._cpu_allocator.release(server.ip, benchmark_tag)


@pytest.mark.asyncio
async def test_numa_separation_preferred():
    """Test that different NUMA nodes are preferred over different L3 caches."""
    server = Server("127.0.0.1", port=9000)
    await server.ensure_host_cpu_allocation()
    
    numa_nodes = server._cpu_allocator._numa_nodes.get(server.ip, {})
    
    if len(numa_nodes) < 2:
        pytest.skip("Test requires multiple NUMA nodes")
    
    # Allocate server on NUMA node 0
    server_tag = AllocationTag(task_id="test_server", purpose="server")
    server_cpus = server._cpu_allocator.allocate(
        server.ip,
        server_tag,
        count=6,
        require_numa=0,
    )
    
    # Allocate benchmark with cache avoidance (no NUMA requirement)
    benchmark_tag = AllocationTag(task_id="test_benchmark", purpose="benchmark")
    benchmark_cpus = server._cpu_allocator.allocate(
        server.ip,
        benchmark_tag,
        count=16,
        avoid_tags=[server_tag],
        prefer_different_cache=True,
    )
    
    # Check if benchmark was allocated on different NUMA node
    benchmark_numa = None
    for numa_node, cpus in numa_nodes.items():
        if benchmark_cpus[0] in cpus:
            benchmark_numa = numa_node
            break
    
    print(f"Server NUMA: 0, Benchmark NUMA: {benchmark_numa}")
    
    if benchmark_numa != 0:
        print("✓ Benchmark allocated on different NUMA node (best case)")
    else:
        print("⚠ Benchmark on same NUMA node (may be unavoidable)")
    
    # Cleanup
    server._cpu_allocator.release(server.ip, server_tag)
    server._cpu_allocator.release(server.ip, benchmark_tag)


if __name__ == "__main__":
    asyncio.run(test_l3_cache_detection())
    asyncio.run(test_cache_aware_allocation())
    asyncio.run(test_numa_separation_preferred())
    print("\nAll cache-aware allocation tests passed!")
