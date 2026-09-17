"""Unit tests for the replica-read task and the topology module behind it."""

import json

import pytest

from conductress import config
from conductress.cachecannon import generate_toml_config
from conductress.cli import build_parser
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_replica_read import METHOD, ReplicaReadTaskData, ReplicaReadTaskRunner
from conductress.topology import InstanceSpec, TopologySpec, parse_multi_info, replication_lag_stats


def _valid_source():
    return config.REPO_NAMES[0]


def _task(**overrides) -> ReplicaReadTaskData:
    fields = dict(
        source=_valid_source(),
        specifier="unstable",
        make_args="",
        replicas=1,
        note="",
        requirements={},
    )
    fields.update(overrides)
    return ReplicaReadTaskData(**fields)


# --- topology spec ---


def test_replica_read_spec_layout():
    spec = TopologySpec.replica_read(replicas=2, replica_io_threads=8, primary_io_threads=1, base_port=7000)
    assert [i.role for i in spec.instances] == ["primary", "replica", "replica"]
    assert [i.port for i in spec.instances] == [7000, 7001, 7002]
    assert spec.primary.io_threads == 1
    assert all(r.io_threads == 8 for r in spec.replicas)
    assert not spec.cluster_enabled


def test_replica_read_spec_role_args_follow_shared_args():
    spec = TopologySpec.replica_read(
        replicas=1,
        replica_io_threads=4,
        server_args="--maxmemory 1gb",
        primary_args="--repl-backlog-size 64mb",
        replica_args="--io-threads-ownership yes",
    )
    assert spec.primary.server_args == "--maxmemory 1gb --repl-backlog-size 64mb"
    assert spec.replicas[0].server_args == "--maxmemory 1gb --io-threads-ownership yes"


def test_replica_read_spec_requires_a_replica():
    with pytest.raises(ValueError, match="replicas must be >= 1"):
        TopologySpec.replica_read(replicas=0, replica_io_threads=4)


def test_topology_spec_rejects_two_primaries_and_duplicate_ports():
    with pytest.raises(ValueError, match="exactly one primary"):
        TopologySpec(instances=[InstanceSpec("primary", 6379), InstanceSpec("primary", 6380)])
    with pytest.raises(ValueError, match="unique"):
        TopologySpec(instances=[InstanceSpec("primary", 6379), InstanceSpec("replica", 6379)])


def test_instance_spec_port_leaves_room_for_cluster_bus():
    with pytest.raises(ValueError, match="cluster bus"):
        InstanceSpec("primary", 60000)
    with pytest.raises(ValueError, match="role"):
        InstanceSpec("leader", 6379)


def test_instance_dir_is_per_port():
    a, b = InstanceSpec("primary", 6379), InstanceSpec("replica", 6380)
    assert a.instance_dir != b.instance_dir
    assert str(a.instance_dir).endswith("6379")


# --- lag stats ---


def test_replication_lag_stats_from_samples():
    samples = [
        {"t": 0.0, "instances": {6379: {"master_repl_offset": 1000}, 6380: {"master_repl_offset": 900}}},
        {"t": 1.0, "instances": {6379: {"master_repl_offset": 2000}, 6380: {"master_repl_offset": 2000}}},
        {"t": 2.0, "instances": {6379: {"master_repl_offset": 3000}, 6380: {"master_repl_offset": 2700}}},
        {"t": 3.0, "instances": {6379: {"master_repl_offset": None}, 6380: {"master_repl_offset": 2700}}},
    ]
    stats = replication_lag_stats(samples, 6379, 6380)
    assert stats["samples"] == 3
    assert stats["max_bytes"] == 300
    assert stats["mean_bytes"] == pytest.approx((100 + 0 + 300) / 3)
    assert replication_lag_stats([], 6379, 6380) == {"samples": 0}


def test_parse_multi_info_splits_instances_and_extra_fields():
    output = """=== conductress-instance 6379
# Replication
role:master
master_repl_offset:5000
# Stats
total_commands_processed:10
instantaneous_ops_per_sec:3
dplus_speculated:0
=== conductress-instance 6380
role:slave
master_link_status:up
master_last_io_seconds_ago:0
master_repl_offset:4900
instantaneous_ops_per_sec:250000
dplus_speculated:123456
dplus_pressure_ratio:0.25
"""
    rows = parse_multi_info(output, ["dplus_speculated", "dplus_pressure_ratio", "missing_field"])
    assert set(rows) == {6379, 6380}
    assert rows[6379]["role"] == "master" and rows[6379]["master_repl_offset"] == 5000
    assert rows[6380]["master_link_status"] == "up"
    assert rows[6380]["dplus_speculated"] == 123456
    assert rows[6380]["dplus_pressure_ratio"] == 0.25
    assert "missing_field" not in rows[6380]
    assert rows[6379]["total_reads_processed"] is None  # absent field, not a crash


# --- TOML knobs shared with the cachecannon task ---


def _toml(**kw):
    base = dict(
        duration=30,
        warmup=5,
        threads=4,
        cpu_list="1,2,3,4",
        endpoint="127.0.0.1:6379",
        connections=16,
        pipeline_depth=1,
        keyspace_count=1000,
        val_size=16,
    )
    base.update(kw)
    return generate_toml_config(**base)


def test_toml_rate_limit_and_prefill_off():
    toml = _toml(test="set", rate_limit=50000, prefill=False)
    assert "prefill = false" in toml
    assert "rate_limit = 50000" in toml
    assert "set = 100" in toml and "get = 0" in toml


def test_toml_defaults_unchanged_for_existing_callers():
    toml = _toml()
    assert "prefill = true" in toml
    assert "rate_limit" not in toml


def test_toml_negative_rate_limit_rejected():
    with pytest.raises(ValueError, match="rate_limit"):
        _toml(rate_limit=-1)


# --- task data ---


def test_task_defaults_and_description():
    task = _task()
    assert task.task_type == "ReplicaReadTaskData"
    assert task.io_threads == 8 and task.primary_io_threads == 1
    assert task.pipelining == 1  # replica reads are the non-pipelined real-client case
    desc = task.short_description()
    assert "replica-read" in desc and "50000/s" in desc


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"replicas": 0}, "replicas"),
        ({"write_rate": 0}, "write_rate"),
        ({"duration": 0}, "duration"),
        ({"sample_interval": 0}, "sample_interval"),
        ({"write_connections": 0}, "write_connections"),
        ({"base_port": 60000}, "cluster bus"),
    ],
)
def test_task_rejects_invalid_levers(overrides, match):
    with pytest.raises(ValueError, match=match):
        _task(**overrides)


def test_task_round_trips_through_queue_document(tmp_path):
    task = _task(replicas=2, write_rate=200_000, replica_args="--io-threads-ownership yes", info_fields="a, b")
    path = tmp_path / "task.json"
    task.save_to_file(path)
    loaded = BaseTaskData.from_file(path)
    assert isinstance(loaded, ReplicaReadTaskData)
    assert loaded.replicas == 2
    assert loaded.write_rate == 200_000
    assert loaded.replica_args == "--io-threads-ownership yes"
    assert loaded.extra_info_fields() == ["a", "b"]
    assert json.loads(path.read_text())["task_type"] == "ReplicaReadTaskData"


def test_runner_title_and_status():
    task = _task(replicas=1, io_threads=16, write_rate=10_000)
    runner = ReplicaReadTaskRunner(task, [config.ServerInfo(ip="127.0.0.1", username="ec2-user")])
    assert runner.status.task_type == METHOD
    assert runner.status.steps_total > 0
    assert "io-threads=16" in runner.title
    assert runner.spec.replicas[0].port == task.base_port + 1


# --- CLI ---


def test_cli_add_replica_read_parses_role_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "queue",
            "add-replica-read",
            "--source",
            _valid_source(),
            "--specifier",
            "abc123",
            "--replicas",
            "2",
            "--io-threads",
            "16",
            "--write-rate",
            "200000",
            "--replica-args",
            "--io-threads-ownership yes",
            "--info-fields",
            "dplus_speculated,dplus_punted",
        ]
    )
    assert args.queue_command == "add-replica-read"
    assert args.replicas == 2
    assert args.io_threads == 16
    assert args.write_rate == 200000
    assert args.replica_args == "--io-threads-ownership yes"
    assert args.info_fields == "dplus_speculated,dplus_punted"
    assert args.primary_io_threads == 1
