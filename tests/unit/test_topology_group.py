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
            InstanceSpec(role="primary", port=6379, io_threads=1, own_dir=False),
            InstanceSpec(role="replica", port=6379, io_threads=1, host=OTHER, own_dir=False),
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
    ).resolved(io_threads=1)
    group = TopologyGroup(HOST, spec, "valkey", "unstable")
    with patch("conductress.topology.asyncio.sleep"):
        await group.start()
    with pytest.raises(NotImplementedError, match="multi-host"):
        await group.sample()


# --------------------------------------------------------------------------- task-field plumbing


def test_standalone_default_defers_every_setting_to_the_task():
    spec = TopologySpec.standalone()
    assert spec.is_standalone
    assert spec.primary.io_threads is None and spec.primary.server_args == "" and spec.primary.cpu_override == ""
    assert spec.host_count() == 1


def test_resolved_fills_unset_settings_and_orders_role_args_after_task_args():
    spec = TopologySpec.replica_read(replicas=1, replica_io_threads=4, replica_args="--io-threads-ownership yes")
    out = spec.resolved(io_threads=7, server_args="--save ''", cpu_override="0-3")
    assert out.primary.io_threads == 1  # explicit values win
    assert out.replicas[0].io_threads == 4
    assert out.replicas[0].server_args == "--save '' --io-threads-ownership yes"
    assert out.primary.cpu_override == "0-3"
    plain = TopologySpec.standalone().resolved(io_threads=7)
    assert plain.primary.io_threads == 7 and plain.primary.own_dir is False


@pytest.mark.asyncio
async def test_group_refuses_a_spec_with_unresolved_io_threads(fake_server):
    group = TopologyGroup(HOST, TopologySpec.standalone(), "valkey", "unstable")
    with pytest.raises(RuntimeError, match="no io_threads"):
        await group.start()


def test_replication_hosts_maps_the_legacy_replicas_count():
    assert TopologySpec.replication_hosts(0) == TopologySpec.standalone()
    assert TopologySpec.replication_hosts(-1) == TopologySpec.standalone()
    spec = TopologySpec.replication_hosts(2)
    assert [(i.role, i.port, i.host_slot, i.own_dir) for i in spec.instances] == [
        ("primary", 6379, 0, False),
        ("replica", 6379, 1, False),
        ("replica", 6379, 2, False),
    ]
    assert spec.host_count() == 3
    assert not spec.is_standalone


def test_host_count_counts_slots_and_pinned_addresses():
    spec = TopologySpec(
        instances=[
            InstanceSpec(role="primary", port=6379),
            InstanceSpec(role="replica", port=6380),  # same host, other port
            InstanceSpec(role="replica", port=6379, host_slot=1),
            InstanceSpec(role="replica", port=6379, host=OTHER),
        ]
    )
    assert spec.host_count() == 3


def test_bind_hosts_pins_slots_and_rejects_missing_servers():
    spec = TopologySpec.replication_hosts(1)
    with pytest.raises(RuntimeError, match="host slot 1"):
        spec.bind_hosts([HOST])
    bound = spec.bind_hosts([HOST, OTHER])
    assert bound.replicas[0].host == OTHER and bound.replicas[0].host_slot == 0
    assert bound.primary.host is None  # slot 0 stays the default host
    assert [h.ip for h in bound.hosts(HOST)] == ["10.0.0.1", "10.0.0.2"]
    with pytest.raises(RuntimeError, match="not bound"):
        spec.hosts(HOST)


def test_spec_round_trips_through_dict_with_hosts_and_slots():
    spec = TopologySpec(
        instances=[
            InstanceSpec(role="primary", port=6379, io_threads=4, own_dir=False),
            InstanceSpec(role="replica", port=6379, host=OTHER, own_dir=False),
            InstanceSpec(role="replica", port=6380, host_slot=2),
        ]
    )
    assert TopologySpec.from_dict(spec.to_dict()) == spec
    assert TopologySpec.from_dict(spec) is spec
    with pytest.raises(ValueError, match="instances"):
        TopologySpec.from_dict({"nope": 1})
