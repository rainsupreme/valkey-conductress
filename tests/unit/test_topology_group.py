"""TopologySpec/TopologyGroup as the successor of ReplicationGroup.

Two things are pinned here. The spec additions (per-instance host, legacy
layout, the ``standalone`` factory). And the acceptance test for retiring
``ReplicationGroup``: driven through the same fake ``Server``, the two classes
must issue the same calls for the one-instance layout every sweep cell uses,
and the same set of calls for a two-host replication group.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.config import ServerInfo
from conductress.replication_group import ReplicationGroup
from conductress.topology import InstanceSpec, TopologyGroup, TopologySpec

HOST = ServerInfo(ip="10.0.0.1", username="ec2-user")
OTHER = ServerInfo(ip="10.0.0.2", username="ec2-user")


# --------------------------------------------------------------------------- spec


def test_instance_host_defaults_to_the_group_host():
    inst = InstanceSpec(role="primary", port=6379)
    assert inst.host is None
    assert inst.resolved_host(HOST) is HOST
    pinned = InstanceSpec(role="replica", port=6379, host=OTHER)
    assert pinned.resolved_host(HOST) is OTHER


def test_same_port_allowed_across_hosts_but_not_within_one():
    TopologySpec(
        instances=[
            InstanceSpec(role="primary", port=6379),
            InstanceSpec(role="replica", port=6379, host=OTHER),
        ]
    )
    with pytest.raises(ValueError, match="unique per host"):
        TopologySpec(instances=[InstanceSpec(role="primary", port=6379), InstanceSpec(role="replica", port=6379)])


def test_hosts_are_distinct_in_first_appearance_order():
    spec = TopologySpec(
        instances=[
            InstanceSpec(role="primary", port=6379),
            InstanceSpec(role="replica", port=6380),
            InstanceSpec(role="replica", port=6379, host=OTHER),
        ]
    )
    assert [h.ip for h in spec.hosts(HOST)] == ["10.0.0.1", "10.0.0.2"]


def test_standalone_is_one_legacy_primary():
    spec = TopologySpec.standalone(7, server_args="--maxmemory 1gb", cpu_override="0-3")
    assert len(spec.instances) == 1
    inst = spec.primary
    assert (inst.port, inst.io_threads, inst.server_args, inst.cpu_override) == (6379, 7, "--maxmemory 1gb", "0-3")
    assert inst.own_dir is False
    assert inst.instance_dir is None
    assert spec.replicas == []
    assert spec.cluster_enabled is False


def test_replica_read_instances_keep_their_own_dir():
    spec = TopologySpec.replica_read(replicas=1, replica_io_threads=4)
    assert all(i.own_dir for i in spec.instances)
    assert spec.primary.instance_dir == Path("~") / "conductress-instances" / "6379"


def test_to_dict_carries_host_and_layout():
    d = TopologySpec.standalone(1, host=OTHER).to_dict()
    assert d["instances"][0]["host"]["ip"] == "10.0.0.2"
    assert d["instances"][0]["host"]["username"] == "ec2-user"
    assert d["instances"][0]["own_dir"] is False


# --------------------------------------------------------------------------- equivalence


class FakeServer:
    """Records every call ReplicationGroup or TopologyGroup makes, per (ip, port)."""

    log: list = []
    replicas_by_ip: dict = {}

    def __init__(self, ip, port=6379, username="", instance_dir=None):
        self.ip, self.port, self.username, self.instance_dir = ip, port, username, instance_dir
        self.valkey_pid = 4242

    @classmethod
    async def with_build(cls, ip, port, username, binary_source, specifier, io_threads, make_args, **kw):
        # Mirrors Server.with_build on main exactly.
        server = cls(ip, port, username)
        path = await server.ensure_binary_cached(binary_source, specifier, make_args)
        await server.start(
            path,
            io_threads,
            server_cpu_override=kw.get("server_cpu_override", ""),
            server_args=kw.get("server_args", ""),
        )
        return server

    async def ensure_binary_cached(self, source, specifier, make_args):
        FakeServer.log.append(
            ("build", self.ip, self.port, self.username, self.instance_dir, source, specifier, make_args)
        )
        return Path("/cache/valkey-server")

    async def start(self, path, io_threads, env_prefix="", server_cpu_override="", server_args=""):
        FakeServer.log.append(("start", self.ip, self.port, str(path), io_threads, server_cpu_override, server_args))

    async def replicate(self, primary_ip, port="6379"):
        FakeServer.log.append(("replicate", self.ip, self.port, primary_ip, str(port)))

    async def wait_until_ready(self):
        FakeServer.log.append(("ready", self.ip, self.port))

    async def get_replicas(self):
        FakeServer.log.append(("get_replicas", self.ip, self.port))
        return FakeServer.replicas_by_ip.get(self.ip, [])

    async def info(self, section):
        FakeServer.log.append(("info", self.ip, self.port, section))
        return {"master_link_status": "up", "master_sync_in_progress": "0", "master_repl_offset": "0"}

    async def stop(self):
        FakeServer.log.append(("stop", self.ip, self.port))

    async def kill_all_valkey_instances_on_host(self):
        FakeServer.log.append(("kill", self.ip))


@pytest.fixture
def fake_server():
    FakeServer.log = []
    FakeServer.replicas_by_ip = {}
    with patch("conductress.replication_group.Server", FakeServer), patch("conductress.topology.Server", FakeServer):
        yield FakeServer


def _without_kill(log):
    return sorted((e for e in log if e[0] != "kill"), key=repr)


async def _drive(group):
    """Two reps the way the tasks do it: kill (a no-op for ReplicationGroup before its first start), start, wire, stop."""
    for _ in range(2):
        await group.kill_all_valkey_instances()
        await group.start()
        await group.begin_replication()
        await group.wait_for_repl_sync()
        await group.end_replication()
        await group.stop_all_servers()


@pytest.mark.asyncio
async def test_standalone_issues_the_same_calls_as_a_one_server_replication_group(fake_server):
    """The sweep path: one instance, default port, legacy layout. Call-for-call identical."""
    args = dict(binary_source="valkey", specifier="unstable", make_args="OPTIMIZATION=-O2")
    legacy = ReplicationGroup([HOST], threads=7, server_cpu_override="0-7", server_args="--save ''", **args)
    with patch("conductress.topology.asyncio.sleep"), patch("conductress.replication_group.asyncio.sleep"):
        await _drive(legacy)
    legacy_log = list(fake_server.log)

    fake_server.log = []
    spec = TopologySpec.standalone(7, server_args="--save ''", cpu_override="0-7")
    new = TopologyGroup(HOST, spec, **args)
    with patch("conductress.topology.asyncio.sleep"):
        await _drive(new)

    # ReplicationGroup's pre-start kill is a no-op until it has servers; TopologyGroup
    # always kills. Everything else must match call for call.
    assert [e for e in fake_server.log if e[0] != "kill"] == [e for e in legacy_log if e[0] != "kill"]
    assert fake_server.log.count(("kill", "10.0.0.1")) == 2 and legacy_log.count(("kill", "10.0.0.1")) == 1
    assert ("build", "10.0.0.1", 6379, "ec2-user", None, "valkey", "unstable", "OPTIMIZATION=-O2") in legacy_log
    assert new.primary.port == 6379 and new.replicas == []


@pytest.mark.asyncio
async def test_two_host_group_issues_the_same_set_of_calls(fake_server):
    """ReplicationGroup starts hosts in parallel and TopologyGroup sequentially, so compare as multisets."""
    args = dict(binary_source="valkey", specifier="unstable", make_args="")
    legacy = ReplicationGroup([HOST, OTHER], threads=1, **args)
    with patch("conductress.replication_group.asyncio.sleep"):
        await _drive(legacy)
    legacy_log = sorted(fake_server.log, key=repr)

    fake_server.log = []
    spec = TopologySpec(
        instances=[
            InstanceSpec(role="primary", port=6379, own_dir=False),
            InstanceSpec(role="replica", port=6379, host=OTHER, own_dir=False),
        ]
    )
    new = TopologyGroup(HOST, spec, **args)
    with patch("conductress.topology.asyncio.sleep"):
        await _drive(new)

    assert _without_kill(fake_server.log) == _without_kill(legacy_log)
    assert ("replicate", "10.0.0.2", 6379, "10.0.0.1", "6379") in fake_server.log
    assert fake_server.log.count(("kill", "10.0.0.2")) == 2


@pytest.mark.asyncio
async def test_unknown_replica_is_detached_then_rechecked(fake_server):
    fake_server.replicas_by_ip = {"10.0.0.1": ["10.0.0.9"]}
    group = TopologyGroup(HOST, TopologySpec.standalone(1), "valkey", "unstable")

    async def detach_clears(*_a, **_k):
        fake_server.replicas_by_ip = {}

    with patch("conductress.topology.asyncio.sleep", detach_clears):
        await group.start()
    assert ("replicate", "10.0.0.9", 6379, None, "6379") in fake_server.log
    assert [e for e in fake_server.log if e[0] == "get_replicas"] == [("get_replicas", "10.0.0.1", 6379)] * 2


@pytest.mark.asyncio
async def test_unknown_replica_that_stays_attached_is_an_error(fake_server):
    fake_server.replicas_by_ip = {"10.0.0.1": ["10.0.0.9"]}
    group = TopologyGroup(HOST, TopologySpec.standalone(1), "valkey", "unstable")
    with patch("conductress.topology.asyncio.sleep"), pytest.raises(RuntimeError, match="remain after cleanup"):
        await group.start()


@pytest.mark.asyncio
async def test_sample_refuses_multi_host_layouts(fake_server):
    spec = TopologySpec(
        instances=[InstanceSpec(role="primary", port=6379), InstanceSpec(role="replica", port=6379, host=OTHER)]
    )
    group = TopologyGroup(HOST, spec, "valkey", "unstable")
    with patch("conductress.topology.asyncio.sleep"):
        await group.start()
    with pytest.raises(NotImplementedError, match="multi-host"):
        await group.sample()
