from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.tasks.task_perf_benchmark import BoundedInsertionTaskData, checked_memory_snapshot


@pytest.fixture(autouse=True)
def patch_sources(monkeypatch):
    monkeypatch.setattr(config, "MANUALLY_UPLOADED", "manual")
    monkeypatch.setattr(config, "REPO_NAMES", ["valkey", "repo1"])
    monkeypatch.setattr("conductress.task_queue.config.MANUALLY_UPLOADED", "manual")
    monkeypatch.setattr("conductress.task_queue.config.REPO_NAMES", ["valkey", "repo1"])


def make_task(**overrides):
    values = dict(
        source="repo1",
        specifier="abc123",
        replicas=0,
        note="bounded",
        requirements={},
        make_args="",
        val_size=16,
        key_size=16,
        io_threads=9,
        pipelining=10,
        insertions=1_000_000,
        repetitions=3,
        maxmemory_bytes=8_000_000_000,
        max_rss_bytes=12_000_000_000,
        perf_stat_enabled=True,
    )
    values.update(overrides)
    return BoundedInsertionTaskData(**values)


def test_task_builds_exact_sequential_command_and_appends_memory_policy():
    runner = make_task().prepare_task_runner([])
    client = MagicMock()
    client.ip = "127.0.0.1"
    client._cpu_allocator.get_net_interface_numa.return_value = 0

    command = runner._build_benchmark_command(client, "127.0.0.1", None)

    assert "--sequential" in command
    assert "-r 1000000" in command
    assert "-n 1000000" in command
    assert " -l " not in command
    assert "--maxmemory 8000000000 --maxmemory-policy noeviction" in runner.server_args


def test_task_rejects_unsafe_or_unbounded_shapes():
    with pytest.raises(ValueError, match="insertions must be at least 1"):
        make_task(insertions=0)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        make_task(maxmemory_bytes=13_000_000_000, max_rss_bytes=12_000_000_000)
    with pytest.raises(ValueError, match="key_size must be at least 16"):
        make_task(key_size=15)
    with pytest.raises(ValueError, match="insertions must not exceed"):
        make_task(insertions=2_000_000_001)
    with pytest.raises(ValueError, match="must not be negative"):
        make_task(bench_threads=-1)
    with pytest.raises(ValueError, match="do not support replicas"):
        make_task(replicas=1)


def test_checked_memory_snapshot_accepts_below_limits_and_rejects_crossings():
    assert checked_memory_snapshot({"used_memory": "100", "used_memory_rss": "150"}, 200, 300) == (100, 150)
    with pytest.raises(RuntimeError, match="used_memory safety limit exceeded"):
        checked_memory_snapshot({"used_memory": "201", "used_memory_rss": "150"}, 200, 300)
    with pytest.raises(RuntimeError, match="RSS safety limit exceeded"):
        checked_memory_snapshot({"used_memory": "100", "used_memory_rss": "301"}, 200, 300)


@patch("conductress.cli.TaskQueue")
def test_cli_add_insertion_builds_bounded_task(mock_queue_cls):
    queue = MagicMock()
    mock_queue_cls.return_value = queue

    exit_code = main(
        [
            "queue",
            "add-insertion",
            "--source",
            "repo1",
            "--specifier",
            "abc123",
            "--insertions",
            "20M",
            "--size",
            "16",
            "--key-size",
            "16",
            "--maxmemory",
            "8GB",
            "--max-rss",
            "12GB",
            "--perf-stat",
        ]
    )

    assert exit_code == 0
    task = queue.submit_task.call_args.args[0]
    assert isinstance(task, BoundedInsertionTaskData)
    assert task.insertions == 20_000_000
    assert task.maxmemory_bytes == 8 * 1024**3
    assert task.max_rss_bytes == 12 * 1024**3
    assert task.perf_stat_enabled is True


@patch("conductress.cli.TaskQueue")
def test_cli_add_insertion_rejects_rss_below_maxmemory(mock_queue_cls):
    exit_code = main(
        [
            "queue",
            "add-insertion",
            "--source",
            "repo1",
            "--insertions",
            "1M",
            "--maxmemory",
            "8GB",
            "--max-rss",
            "4GB",
        ]
    )

    assert exit_code == 1
    mock_queue_cls.return_value.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_finite_fill_profiles_full_interval_and_keeps_live_client_cpu(monkeypatch):
    from conductress.tasks import task_perf_benchmark as module

    events = []

    class FakeProcess:
        pid = 123
        returncode = None

    class FakeCommand:
        def __init__(self, command):
            self.command = command
            self.p = FakeProcess()
            self.polls = 0

        def start(self):
            events.append("command-start")

        def is_running(self):
            self.polls += 1
            if self.polls == 1:
                return True
            self.p.returncode = 0
            return False

        def poll_output(self):
            return "", ""

        def kill(self):
            self.p.returncode = -9

    async def no_sleep(_seconds):
        return None

    cpu_samples = iter([10.0, 11.0])
    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(module, "sample_process_tree_cpu", lambda _pid: next(cpu_samples))

    runner = make_task(insertions=1, repetitions=1).prepare_task_runner([])
    server = MagicMock()
    server.info = AsyncMock(return_value={"used_memory": "100", "used_memory_rss": "150"})
    server.count_items_expires = AsyncMock(return_value=(1, 0))
    server.perf_stat_start = AsyncMock(side_effect=lambda: events.append("perf-start"))
    server.perf_stat_stop = AsyncMock(side_effect=lambda: events.append("perf-stop"))

    sample = await runner._run_finite_fill("generator", server, 0)

    assert events[0:2] == ["perf-start", "command-start"]
    assert events[-1] == "perf-stop"
    assert sample["inserted_keys"] == 1
    assert runner._client_cores_busy_per_rep
