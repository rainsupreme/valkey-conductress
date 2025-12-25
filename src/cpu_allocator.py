"""CPU allocation manager for pinning processes to specific cores across hosts."""

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
    """Manages CPU core allocation across multiple hosts with NUMA awareness."""

    def __init__(self):
        # Per-host state: host_ip -> data
        self._all_cpus: dict[str, list[int]] = {}  # All available CPUs
        self._numa_nodes: dict[str, dict[int, list[int]]] = {}  # numa_node -> [cpus]
        self._allocations: dict[str, dict[AllocationTag, list[int]]] = {}  # tag -> [cpus]
        self._net_interface_numa: dict[str, int] = {}  # Network interface NUMA node

    def register_host(
        self,
        host_ip: str,
        all_cpus: list[int],
        numa_topology: Optional[dict[int, list[int]]] = None,
        net_interface_numa: Optional[int] = None,
    ) -> None:
        """Register a host with its CPU topology.

        Args:
            host_ip: Host identifier
            all_cpus: List of all CPU IDs on the host
            numa_topology: Optional mapping of NUMA node -> CPU list
            net_interface_numa: Optional NUMA node for network interface
        """
        self._all_cpus[host_ip] = sorted(all_cpus)
        self._numa_nodes[host_ip] = numa_topology or {0: sorted(all_cpus)}
        self._allocations[host_ip] = {}
        if net_interface_numa is not None:
            self._net_interface_numa[host_ip] = net_interface_numa

    def allocate(
        self,
        host_ip: str,
        tag: AllocationTag,
        count: int,
        require_numa: Optional[int] = None,
    ) -> list[int]:
        """Allocate CPU cores for a specific purpose.

        Args:
            host_ip: Host to allocate from
            tag: Allocation identifier
            count: Number of CPUs to allocate
            require_numa: Required NUMA node (None = any node)

        Returns:
            List of allocated CPU IDs

        Raises:
            ValueError: If host not registered or tag already exists
            RuntimeError: If insufficient CPUs available on required NUMA node
        """
        if host_ip not in self._all_cpus:
            raise ValueError(f"Host {host_ip} not registered")

        if tag in self._allocations[host_ip]:
            raise ValueError(f"Allocation {tag} already exists on {host_ip}")

        available = self._get_available_cpus(host_ip)

        # If NUMA node required, restrict to that node
        if require_numa is not None:
            if require_numa not in self._numa_nodes[host_ip]:
                raise ValueError(f"NUMA node {require_numa} not found on {host_ip}")
            
            numa_cpus = set(self._numa_nodes[host_ip][require_numa]) & available
            if len(numa_cpus) < count:
                raise RuntimeError(
                    f"Insufficient CPUs on {host_ip} NUMA node {require_numa}: "
                    f"need {count}, available {len(numa_cpus)}"
                )
            
            # IRQs get last CPUs, others get first CPUs
            if tag.purpose == "irq":
                allocated = sorted(numa_cpus, reverse=True)[:count]
            else:
                allocated = sorted(numa_cpus)[:count]
        else:
            if len(available) < count:
                raise RuntimeError(
                    f"Insufficient CPUs on {host_ip}: need {count}, available {len(available)}"
                )
            allocated = sorted(available)[:count]

        self._allocations[host_ip][tag] = allocated
        return allocated

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
