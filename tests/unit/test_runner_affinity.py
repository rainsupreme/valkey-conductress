"""The runner confines itself to management cores while a task runs.

Pinned here: which cores the allocator hands out for management (tail of a
non-network NUMA node, so no server or generator placement moves), that
acquire pins and publishes the original mask for local launches while release
undoes all of it, that both failure modes leave the runner unpinned without
failing the task, that RealtimeCommand prefixes local launches (and only
local ones) with the published mask, and that Server.run_host_command wraps a
pinned host command in taskset with the inner command intact.
"""

import shlex
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress import runner_affinity
from conductress.cpu_allocator import AllocationTag, CpuAllocator
from conductress.runner_affinity import ManagementCores, format_cpulist, is_local_host
from conductress.server import Server
from conductress.utility import RealtimeCommand


@pytest.fixture(autouse=True)
def _clean_launch_cpus():
    RealtimeCommand.launch_cpus = ""
    yield
    RealtimeCommand.launch_cpus = ""


# --------------------------------------------------------------------------- helpers


def test_format_cpulist_collapses_runs():
    assert format_cpulist([]) == ""
    assert format_cpulist([3]) == "3"
    assert format_cpulist([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_is_local_host():
    assert is_local_host("127.0.0.1") and is_local_host("localhost") and is_local_host("::1")
    assert not is_local_host("10.0.0.7")


# --------------------------------------------------------------------------- allocator placement


def _allocator_two_nodes() -> CpuAllocator:
    allocator = CpuAllocator()
    allocator.register_host(
        "10.0.0.1",
        all_cpus=list(range(16)),
        numa_topology={0: list(range(8)), 1: list(range(8, 16))},
        net_interface_numa=0,
    )
    return allocator


def test_from_end_takes_the_highest_free_cpus():
    allocator = _allocator_two_nodes()
    tag = AllocationTag(task_id="mgmt", purpose="monitoring")
    assert allocator.allocate("10.0.0.1", tag, count=2, from_end=True) == [15, 14]
    # The next ordinary allocation still starts at CPU 0: nothing moved.
    server = AllocationTag(task_id="server", purpose="server")
    assert allocator.allocate("10.0.0.1", server, count=3, require_numa=0) == [0, 1, 2]


def test_get_numa_nodes_lists_registered_nodes():
    assert _allocator_two_nodes().get_numa_nodes("10.0.0.1") == [0, 1]
    assert CpuAllocator().get_numa_nodes("nope") == [0]


def test_server_allocate_management_cpus_prefers_the_non_network_node_tail():
    allocator = _allocator_two_nodes()
    with patch.object(Server, "_cpu_allocator", allocator):
        server = Server("10.0.0.1")
        tag = AllocationTag(task_id="runner_t", purpose="monitoring")
        assert server.allocate_management_cpus(tag, 1) == [15]  # node 1 (not the NIC's node 0), from the end


def test_server_allocate_management_cpus_on_a_single_node_host_still_takes_the_tail():
    allocator = CpuAllocator()
    allocator.register_host(
        "10.0.0.2", all_cpus=list(range(8)), numa_topology={0: list(range(8))}, net_interface_numa=0
    )
    with patch.object(Server, "_cpu_allocator", allocator):
        tag = AllocationTag(task_id="runner_t", purpose="monitoring")
        assert Server("10.0.0.2").allocate_management_cpus(tag, 1) == [7]


# --------------------------------------------------------------------------- ManagementCores


class FakeServer:
    def __init__(self, cpus=(15,), fail_allocate=False):
        self.cpus = list(cpus)
        self.fail_allocate = fail_allocate
        self.released = []
        self.ensure_host_cpu_allocation = AsyncMock()

    def allocate_management_cpus(self, tag, count):  # pylint: disable=unused-argument
        if self.fail_allocate:
            raise RuntimeError("Insufficient CPUs")
        return self.cpus

    def release_cpus(self, tag):
        self.released.append(tag.task_id)


@pytest.fixture
def pins(monkeypatch):
    calls: list = []
    monkeypatch.setattr(runner_affinity, "current_affinity_cpulist", lambda: "0-15")
    monkeypatch.setattr(runner_affinity, "pin_current_process", calls.append)
    return calls


@pytest.mark.asyncio
async def test_acquire_pins_and_publishes_the_original_mask_and_release_undoes_it(pins):
    server = FakeServer(cpus=[14, 15])
    mgmt = ManagementCores(server, "task1", count=2)
    assert await mgmt.acquire() == "14-15"
    assert mgmt.cpulist == "14-15"
    assert pins == ["14-15"]
    assert RealtimeCommand.launch_cpus == "0-15"

    mgmt.release()
    assert pins == ["14-15", "0-15"]
    assert RealtimeCommand.launch_cpus == ""
    assert server.released == ["runner_task1"]
    assert mgmt.cpulist == ""
    mgmt.release()  # idempotent
    assert pins == ["14-15", "0-15"] and server.released == ["runner_task1"]


@pytest.mark.asyncio
async def test_acquire_without_spare_cpus_leaves_the_runner_unpinned(pins, caplog):
    mgmt = ManagementCores(FakeServer(fail_allocate=True), "task1", count=1)
    assert await mgmt.acquire() == ""
    assert pins == [] and RealtimeCommand.launch_cpus == ""
    assert "running unpinned" in caplog.text
    mgmt.release()  # nothing to undo


@pytest.mark.asyncio
async def test_acquire_when_taskset_fails_releases_the_cores(monkeypatch, caplog):
    monkeypatch.setattr(runner_affinity, "current_affinity_cpulist", lambda: "0-15")

    def boom(_cpulist):
        raise subprocess.CalledProcessError(1, "taskset")

    monkeypatch.setattr(runner_affinity, "pin_current_process", boom)
    server = FakeServer()
    mgmt = ManagementCores(server, "task1", count=1)
    assert await mgmt.acquire() == ""
    assert server.released == ["runner_task1"]
    assert RealtimeCommand.launch_cpus == ""
    assert "Could not pin" in caplog.text


@pytest.mark.asyncio
async def test_zero_count_disables_pinning(pins):
    server = FakeServer()
    assert await ManagementCores(server, "task1", count=0).acquire() == ""
    assert not server.ensure_host_cpu_allocation.called and pins == []


# --------------------------------------------------------------------------- RealtimeCommand


def _popen_argv(command: RealtimeCommand) -> list:
    with patch("conductress.utility.subprocess.Popen") as popen:
        popen.return_value = MagicMock()
        command.start(ignore_output=True)
    return popen.call_args.args[0]


def test_realtime_command_launches_plain_when_no_mask_is_published():
    assert _popen_argv(RealtimeCommand("cachecannon run.toml")) == ["cachecannon", "run.toml"]


def test_realtime_command_prefixes_local_launches_with_the_published_mask():
    RealtimeCommand.launch_cpus = "0-15"
    assert _popen_argv(RealtimeCommand("cachecannon run.toml")) == ["taskset", "-c", "0-15", "cachecannon", "run.toml"]


def test_realtime_command_leaves_remote_launches_alone():
    RealtimeCommand.launch_cpus = "0-15"
    argv = _popen_argv(RealtimeCommand("memtier --x", remote="10.0.0.9"))
    assert argv[0] == "ssh" and "taskset" not in argv


# --------------------------------------------------------------------------- Server.run_host_command pin


@pytest.mark.asyncio
async def test_run_host_command_wraps_pinned_commands_in_taskset():
    server = Server("10.0.0.1")
    server._host = MagicMock()
    server._host.run_host_command = AsyncMock(return_value=("", ""))
    inner = "echo '=== x'; valkey-cli -p 6379 info"

    await server.run_host_command(inner, check=False)
    assert server._host.run_host_command.call_args.args[0] == inner

    await server.run_host_command(inner, check=False, pin_cpus="15")
    sent = server._host.run_host_command.call_args.args[0]
    assert sent.startswith("taskset -c 15 sh -c ")
    assert shlex.split(sent)[-1] == inner  # inner command survives quoting intact
