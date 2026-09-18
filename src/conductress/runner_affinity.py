"""Confine the runner process to management cores while a task runs.

The runner is itself a CPU consumer on the benchmark host: its log and status
writes and any per-second sampling measured about 0.35 of a core on a 192-CPU
runner. Unpinned, it lands on whatever core the scheduler picks, including
cores that share a cache with the server under test, and the one task that
checks for foreign CPU activity correctly reported it as interference.

For each task the runner reserves ``config.RUNNER_MANAGEMENT_CPUS`` through
the CPU allocator (highest-numbered free CPUs, on a NUMA node other than the
network interface's when the host has one, so no server or generator
placement moves), pins every thread of itself there, and publishes the mask
it had before pinning as ``RealtimeCommand.launch_cpus``. Local generator
launches start under that original mask: a child inherits its parent's mask
at fork, and the generators must not be dragged onto the management cores.
Host commands (``Server.run_host_command``) run under sshd and are unaffected;
work that must stay off measured cores names ``pin_cpus`` explicitly.

Everything here is best effort. A host too small to spare a core, or a
``taskset`` that fails, leaves the runner unpinned with a warning; it never
fails the task.
"""

import logging
import os
import subprocess
from typing import Optional

from conductress import config
from conductress.cpu_allocator import AllocationTag
from conductress.server import Server
from conductress.utility import RealtimeCommand

logger = logging.getLogger(__name__)

MANAGEMENT_PURPOSE = "monitoring"


def format_cpulist(cpus: list) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    ranges: list = []
    for cpu in cpus:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def current_affinity_cpulist() -> str:
    """This process's CPU mask as a compact cpulist (e.g. ``0-191``)."""
    return format_cpulist(sorted(os.sched_getaffinity(0)))


def pin_current_process(cpulist: str) -> None:
    """Confine every thread of this process to ``cpulist`` (``taskset -a``)."""
    subprocess.run(["taskset", "-acp", cpulist, str(os.getpid())], check=True, capture_output=True)


def is_local_host(ip: str) -> bool:
    """True when the benchmark host is the machine this process runs on."""
    return ip in ("127.0.0.1", "localhost", "::1")


class ManagementCores:
    """Reserve management cores for one task, pin the runner to them, undo it afterwards.

    ``cpulist`` is the reserved cores ("" when nothing was reserved); tasks
    that check for foreign CPU activity treat them as claimed.
    """

    def __init__(self, server: Server, task_id: str, count: int = config.RUNNER_MANAGEMENT_CPUS) -> None:
        self._server = server
        self._tag = AllocationTag(task_id=f"runner_{task_id}", purpose=MANAGEMENT_PURPOSE)
        self._count = count
        self._original_mask: Optional[str] = None
        self.cpulist = ""

    async def acquire(self) -> str:
        """Reserve and pin. Returns the management cpulist, or "" if the runner stays unpinned."""
        if self._count < 1:
            return ""
        try:
            await self._server.ensure_host_cpu_allocation()
            cpus = self._server.allocate_management_cpus(self._tag, self._count)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("No management CPUs for the runner (%s); running unpinned", exc)
            return ""
        cpulist = format_cpulist(sorted(cpus))
        original = current_affinity_cpulist()
        try:
            pin_current_process(cpulist)
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("Could not pin the runner to %s (%s); running unpinned", cpulist, exc)
            self._server.release_cpus(self._tag)
            return ""
        self._original_mask = original
        self.cpulist = cpulist
        RealtimeCommand.launch_cpus = original
        logger.info("Runner pinned to management CPUs %s (launches keep %s)", cpulist, original)
        return cpulist

    def release(self) -> None:
        """Restore the runner's mask, stop prefixing launches, free the cores. Safe to call twice."""
        if self._original_mask is not None:
            try:
                pin_current_process(self._original_mask)
            except (OSError, subprocess.CalledProcessError) as exc:
                logger.warning("Could not restore the runner's CPU mask %s: %s", self._original_mask, exc)
            self._original_mask = None
        RealtimeCommand.launch_cpus = ""
        if self.cpulist:
            self._server.release_cpus(self._tag)
            self.cpulist = ""
