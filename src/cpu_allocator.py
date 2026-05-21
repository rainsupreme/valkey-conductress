"""CPU allocation manager for pinning processes to specific cores across hosts."""

import logging
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AllocationTag:
    """Identifies a CPU allocation by task and purpose."""

    task_id: str
    purpose: str  # e.g., "server", "benchmark", "irq", "monitoring"

    def __str__(self) -> str:
        return f"{self.task_id}:{self.purpose}"


class CpuAllocator:
    """Manages CPU core allocation across multiple hosts with NUMA and cache awareness."""

    def __init__(self):
        # Per-host state: host_ip -> data
        self._all_cpus: dict[str, list[int]] = {}  # All available CPUs
        self._numa_nodes: dict[str, dict[int, list[int]]] = {}  # numa_node -> [cpus]
        self._l3_caches: dict[str, dict[int, list[int]]] = {}  # l3_cache_id -> [cpus]
        self._allocations: dict[str, dict[AllocationTag, list[int]]] = {}  # tag -> [cpus]
        self._net_interface_numa: dict[str, int] = {}  # Network interface NUMA node

    def register_host(
        self,
        host_ip: str,
        all_cpus: list[int],
        numa_topology: Optional[dict[int, list[int]]] = None,
        l3_cache_topology: Optional[dict[int, list[int]]] = None,
        net_interface_numa: Optional[int] = None,
    ) -> None:
        """Register a host with its CPU topology.

        Args:
            host_ip: Host identifier
            all_cpus: List of all CPU IDs on the host
            numa_topology: Optional mapping of NUMA node -> CPU list
            l3_cache_topology: Optional mapping of L3 cache ID -> CPU list
            net_interface_numa: Optional NUMA node for network interface
        """
        self._all_cpus[host_ip] = sorted(all_cpus)
        self._numa_nodes[host_ip] = numa_topology or {0: sorted(all_cpus)}
        self._l3_caches[host_ip] = l3_cache_topology or {}
        self._allocations[host_ip] = {}
        if net_interface_numa is not None:
            self._net_interface_numa[host_ip] = net_interface_numa

    def allocate(
        self,
        host_ip: str,
        tag: AllocationTag,
        count: int,
        require_numa: Optional[int] = None,
        avoid_tags: Optional[list[AllocationTag]] = None,
        prefer_different_cache: bool = False,
        minimize_cache_groups: bool = False,
    ) -> list[int]:
        """Allocate CPU cores with cache awareness.

        Allocation priority (applied in order):
        1. require_numa — filter to a specific NUMA node
        2. minimize_cache_groups — pack into fewest L3 cache groups (CCDs)
        3. prefer_different_cache — pick a different L3 group than avoid_tags
        4. Fallback — first available CPUs

        Args:
            host_ip: Host to allocate from
            tag: Allocation identifier
            count: Number of CPUs to allocate
            require_numa: Required NUMA node (None = any node)
            avoid_tags: List of existing allocations whose CPUs we want to avoid.
                       Used to reduce cache contention (e.g., client avoiding server's L3).
            prefer_different_cache: Try to allocate from a different L3 cache group
                                   than avoid_tags. Soft preference — falls back if impossible.
                                   Purpose: separate client from server's L3 for isolation.
            minimize_cache_groups: Pack all allocated CPUs into the fewest L3 cache
                                  groups possible (ideally one). Prevents non-deterministic
                                  thread placement across chiplets on AMD EPYC.
                                  Purpose: benchmark client threads stay on one CCD.

        Returns:
            List of allocated CPU IDs

        Raises:
            ValueError: If host not registered or tag already exists
            RuntimeError: If insufficient CPUs available
        """
        if host_ip not in self._all_cpus:
            raise ValueError(f"Host {host_ip} not registered")

        if tag in self._allocations[host_ip]:
            raise ValueError(f"Allocation {tag} already exists on {host_ip}")

        available = self._get_available_cpus(host_ip)

        # Get CPUs to avoid if cache separation requested
        avoid_cpus = set()
        if prefer_different_cache and avoid_tags:
            for avoid_tag in avoid_tags:
                if avoid_tag in self._allocations[host_ip]:
                    avoid_cpus.update(self._allocations[host_ip][avoid_tag])

        # Restrict to required NUMA node if specified
        if require_numa is not None:
            if require_numa not in self._numa_nodes[host_ip]:
                raise ValueError(f"NUMA node {require_numa} not found on {host_ip}")
            available = available & set(self._numa_nodes[host_ip][require_numa])

        # Try cache-aware allocation if requested
        allocated = None
        if minimize_cache_groups and self._l3_caches.get(host_ip):
            allocated = self._allocate_single_cache(host_ip, available, avoid_cpus, count)

        if allocated is None and prefer_different_cache and avoid_cpus:
            allocated = self._allocate_cache_aware(host_ip, available, avoid_cpus, count, require_numa)

        # Fallback to simple allocation
        if allocated is None:
            if len(available) < count:
                numa_msg = f" on NUMA node {require_numa}" if require_numa is not None else ""
                raise RuntimeError(
                    f"Insufficient CPUs on {host_ip}{numa_msg}: need {count}, available {len(available)}"
                )
            # IRQs get last CPUs, others get first CPUs
            if tag.purpose == "irq":
                allocated = sorted(available, reverse=True)[:count]
            else:
                allocated = sorted(available)[:count]

        self._allocations[host_ip][tag] = allocated
        return allocated

    def _allocate_single_cache(
        self,
        host_ip: str,
        available: set[int],
        avoid_cpus: set[int],
        count: int,
    ) -> Optional[list[int]]:
        """Allocate CPUs from the minimum number of L3 cache groups (CCDs).

        On chiplet architectures (AMD EPYC), cross-CCD thread placement causes
        non-deterministic performance. This method minimizes the number of CCDs used:
        - If count fits in one CCD: use one CCD (ideal)
        - If not: use the fewest CCDs needed to cover the request

        Prefers cache groups that don't overlap with avoid_cpus.
        Returns None only if topology data is unavailable.
        """
        l3_caches = self._l3_caches.get(host_ip)
        if not l3_caches:
            return None

        # Try single CCD first (ideal case)
        single_candidates = []
        for l3_id, cpus in l3_caches.items():
            group_available = sorted(available & set(cpus) - avoid_cpus)
            if len(group_available) >= count:
                overlap = len(set(cpus) & avoid_cpus)
                single_candidates.append((overlap, l3_id, group_available))

        if single_candidates:
            single_candidates.sort()
            return single_candidates[0][2][:count]

        # No single CCD is large enough — use minimum number of CCDs
        # Sort groups by size (largest first) to minimize CCD count
        groups = []
        for l3_id, cpus in l3_caches.items():
            group_available = sorted(available & set(cpus) - avoid_cpus)
            if group_available:
                groups.append((len(group_available), l3_id, group_available))
        groups.sort(reverse=True)

        allocated: list[int] = []
        ccds_used = 0
        for _, l3_id, group_available in groups:
            needed = count - len(allocated)
            if needed <= 0:
                break
            allocated.extend(group_available[:needed])
            ccds_used += 1

        if len(allocated) >= count:
            logging.info(
                "minimize_cache_groups: needed %d CPUs across %d CCDs (max %d per CCD)",
                count, ccds_used, groups[0][0] if groups else 0,
            )
            return allocated[:count]

        logging.warning(
            "minimize_cache_groups: only %d CPUs available across all CCDs, need %d",
            len(allocated), count,
        )
        return None

    def _allocate_cache_aware(
        self,
        host_ip: str,
        available: set[int],
        avoid_cpus: set[int],
        count: int,
        require_numa: Optional[int],
    ) -> Optional[list[int]]:
        """Try to allocate CPUs avoiding cache overlap.

        Priority: different NUMA node > different L3 cache > any available
        Returns None if cache-aware allocation not possible.
        """
        # Get NUMA nodes of CPUs to avoid
        avoid_numa_nodes = set()
        for numa_node, cpus in self._numa_nodes[host_ip].items():
            if any(cpu in avoid_cpus for cpu in cpus):
                avoid_numa_nodes.add(numa_node)

        # Try different NUMA node first (best cache isolation)
        if require_numa is None:  # Only if NUMA not required
            for numa_node, cpus in self._numa_nodes[host_ip].items():
                if numa_node not in avoid_numa_nodes:
                    numa_available = available & set(cpus)
                    if len(numa_available) >= count:
                        return sorted(numa_available)[:count]

        # Try different L3 cache on same NUMA node
        if self._l3_caches[host_ip]:
            avoid_l3_caches = set()
            for l3_id, cpus in self._l3_caches[host_ip].items():
                if any(cpu in avoid_cpus for cpu in cpus):
                    avoid_l3_caches.add(l3_id)

            # Collect CPUs from non-overlapping L3 caches
            cache_aware_cpus = []
            for l3_id, cpus in sorted(self._l3_caches[host_ip].items()):
                if l3_id not in avoid_l3_caches:
                    l3_available = sorted(available & set(cpus))
                    cache_aware_cpus.extend(l3_available)
                    if len(cache_aware_cpus) >= count:
                        return cache_aware_cpus[:count]

        return None  # Cache-aware allocation not possible

    def release(self, host_ip: str, tag: AllocationTag) -> list[int]:
        """Release CPUs allocated to a tag.

        Args:
            host_ip: Host to release from
            tag: Allocation identifier

        Returns:
            List of released CPU IDs

        Raises:
            ValueError: If host not registered or tag doesn't exist
        """
        if host_ip not in self._all_cpus:
            raise ValueError(f"Host {host_ip} not registered")

        if tag not in self._allocations[host_ip]:
            raise ValueError(f"Allocation {tag} not found on {host_ip}")

        cpus = self._allocations[host_ip].pop(tag)
        return cpus

    def get_allocation(self, host_ip: str, tag: AllocationTag) -> Optional[list[int]]:
        """Get CPUs allocated to a tag, or None if not allocated."""
        if host_ip not in self._allocations:
            return None
        return self._allocations[host_ip].get(tag)

    def get_available_count(self, host_ip: str, prefer_numa: Optional[int] = None) -> int:
        """Get number of available CPUs on a host.

        Args:
            host_ip: Host to query
            prefer_numa: If specified, count only CPUs in this NUMA node

        Returns:
            Number of available CPUs
        """
        if host_ip not in self._all_cpus:
            return 0

        available = self._get_available_cpus(host_ip)

        if prefer_numa is not None and prefer_numa in self._numa_nodes[host_ip]:
            numa_cpus = set(self._numa_nodes[host_ip][prefer_numa])
            available = available & numa_cpus

        return len(available)

    def get_all_allocations(self, host_ip: str) -> dict[AllocationTag, list[int]]:
        """Get all allocations for a host."""
        if host_ip not in self._allocations:
            return {}
        return dict(self._allocations[host_ip])

    def get_net_interface_numa(self, host_ip: str) -> Optional[int]:
        """Get the NUMA node for the network interface, or None if not set."""
        return self._net_interface_numa.get(host_ip)

    def is_host_registered(self, host_ip: str) -> bool:
        """Check if a host has been registered."""
        return host_ip in self._all_cpus

    def _get_available_cpus(self, host_ip: str) -> set[int]:
        """Get set of unallocated CPUs on a host."""
        all_cpus = set(self._all_cpus[host_ip])
        allocated = set()
        for cpus in self._allocations[host_ip].values():
            allocated.update(cpus)
        return all_cpus - allocated
