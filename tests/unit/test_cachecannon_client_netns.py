"""Tests for --client-netns on the cachecannon task (dual-ENI real-NIC hairpin,
docs/real-nic-hairpin.md). Mirrors tests/unit/test_client_netns.py, which covers
the valkey-benchmark task's lever.
"""

import json
from unittest.mock import patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.task_queue import TaskQueue
from conductress.tasks.task_cachecannon import CachecannonTaskData, CachecannonTaskRunner
from conductress.topology import TopologySpec


def _valid_source() -> str:
    return config.REPO_NAMES[0] if config.REPO_NAMES else "valkey"


def _data(**overrides) -> CachecannonTaskData:
    kwargs = dict(
        source=_valid_source(),
        specifier="unstable",
        make_args="",
        topology=TopologySpec.standalone(),
        note="test",
        requirements={},
        test="get",
        val_size=512,
        pipelining=1,
        connections=64,
        threads=8,
        warmup=5,
        duration=30,
        repetitions=3,
    )
    kwargs.update(overrides)
    return CachecannonTaskData(**kwargs)


def _runner(**overrides) -> CachecannonTaskRunner:
    kwargs = dict(
        task_id="t1",
        server_infos=[],
        source=_valid_source(),
        specifier="unstable",
        make_args="",
        io_threads=9,
        test="get",
        val_size=512,
        pipelining=1,
        connections=64,
        threads=8,
        warmup=5,
        duration=30,
        repetitions=3,
        max_reps=0,
        target_cv=0.0,
        keyspace_count=4096,
        cachecannon_binary="/opt/cachecannon",
        server_args="",
        server_cpu_override="",
        benchmark_cpu_override="",
        set_ratio=0,
        distribution="uniform",
        note="",
    )
    kwargs.update(overrides)
    return CachecannonTaskRunner(**kwargs)


def test_default_is_loopback_and_omitted_from_description():
    task = _data()
    assert task.client_netns == ""
    assert "netns" not in task.short_description()


def test_netns_shows_in_description_and_reaches_the_runner():
    task = _data(client_netns="loadgen")
    assert "netns loadgen" in task.short_description()
    runner = task.prepare_task_runner([])
    assert runner.client_netns == "loadgen"


def test_netns_and_tls_are_refused():
    """The bootstrap certificate only covers loopback; a hairpin endpoint is the
    primary ENI's address, so the handshake would fail. Refuse up front."""
    with pytest.raises(ValueError, match="client_netns and tls"):
        _data(client_netns="loadgen", tls=True)


def test_build_command_default_has_no_netns_prefix():
    cmd = _runner()._build_command("/tmp/c.toml", "")
    assert cmd == "/opt/cachecannon /tmp/c.toml"
    assert "ip netns exec" not in _runner()._build_command("/tmp/c.toml", "1,2,3")


def test_build_command_netns_prefix_with_and_without_cpu_list():
    runner = _runner(client_netns="loadgen")
    assert runner._build_command("/tmp/c.toml", "") == "sudo ip netns exec loadgen /opt/cachecannon /tmp/c.toml"
    assert runner._build_command("/tmp/c.toml", "1,2,3").startswith("sudo ip netns exec loadgen numactl --membind=")


def test_netns_preflight_and_target_rewrite_are_present():
    """The namespace vanishes on reboot, so the runner must preflight it and
    point at the fix; a loopback endpoint must be rewritten to the primary ENI."""
    import inspect

    src = inspect.getsource(CachecannonTaskRunner.run)
    assert "ip netns list" in src
    assert "setup-loadgen-netns.sh" in src
    assert "real-nic-hairpin.md" in src
    assert "get_primary_interface_ip()" in src


def test_task_round_trips_through_queue_json(tmp_path):
    """client_netns is a real dataclass field, so it survives asdict() serialization
    (a dynamic attribute would deserialize to '' on the runner and silently run loopback)."""
    task = _data(client_netns="loadgen")
    queue = TaskQueue(queue_dir=tmp_path)
    queue.submit_task(task)
    (path,) = tmp_path.glob("task_*.json")
    assert json.loads(path.read_text())["client_netns"] == "loadgen"


@pytest.fixture(autouse=True)
def _ensure_valid_source(monkeypatch):
    if "valkey" not in config.REPO_NAMES:
        monkeypatch.setattr(config, "REPO_NAMES", config.REPO_NAMES + ["valkey"])


@pytest.fixture
def queue_path(tmp_path):
    path = tmp_path / "queue"
    path.mkdir()
    _Orig = TaskQueue

    class _Isolated(_Orig):
        def __init__(self, queue_dir_override=None):
            super().__init__(queue_dir=path)

    with patch("conductress.cli.TaskQueue", _Isolated):
        yield path


def test_cli_add_cachecannon_client_netns(queue_path):
    base = ["queue", "add-cachecannon", "--source", "valkey", "--specifier", "unstable", "--keyspace", "4096"]
    assert main(base) == 0
    assert main(base + ["--client-netns", "loadgen"]) == 0
    tasks = sorted((json.loads(p.read_text()) for p in queue_path.glob("task_*.json")), key=lambda d: d["client_netns"])
    assert [d["client_netns"] for d in tasks] == ["", "loadgen"]
