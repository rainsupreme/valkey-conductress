"""Runner tests for the replica-read task, with the topology, servers and generators faked.

What is faked and why:

* ``TopologyGroup`` -- would build and start valkey on a host. The fake records
  the lifecycle calls and serves canned INFO/CPU samples with a controlled clock.
* ``Server`` (module-level) -- the runner constructs one for the local generator
  host (CPU allocation) and one for the drop-caches call. The fake carries a
  recording allocator.
* ``RealtimeCommand`` -- cachecannon. The fake answers with a canned NDJSON
  result whose numbers depend on which TOML (preload/writer/reader) it was
  launched with, so the guards see the values the test chose.
* ``asyncio.sleep`` -- replaced by a zero-length sleep that still yields, so the
  sampler task interleaves with the reader wait instead of the test taking real
  seconds.

Nothing here touches ``cpu_sampling``'s rules (those have their own tests); the
point is the wiring: right phase order, right identities into the verdict,
release in ``finally``, and the shape of the row that lands in ``output.jsonl``.
"""

import asyncio
import json
import logging
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from conductress import config
from conductress import cpu_sampling as cs
from conductress.file_protocol import BenchmarkResults, FileProtocol
from conductress.tasks import task_replica_read as module
from conductress.tasks.task_replica_read import (
    METHOD,
    ReplicaReadTaskData,
    ReplicaReadTaskRunner,
    TopologySpec,
    _Measurement,
    _Placement,
)
from conductress.topology import TopologySpec

# A pid no live process can have (pid_max on 64-bit Linux is 4194304), so
# read_local_thread_stats finds nothing in /proc for it.
IMPOSSIBLE_PID = 4_194_304 + 7
PRIMARY_PID = 3001
REPLICA_PID = 3002
HOST = config.ServerInfo(ip="127.0.0.1", username="ec2-user")


_LEVERS = (
    "replica_count",
    "io_threads",
    "primary_io_threads",
    "server_args",
    "primary_args",
    "replica_args",
    "base_port",
)


def _spec(
    replica_count=1,
    io_threads=8,
    primary_io_threads=1,
    server_args="",
    primary_args="",
    replica_args="",
    base_port=6379,
):
    return TopologySpec.replica_read(
        replicas=replica_count,
        replica_io_threads=io_threads,
        primary_io_threads=primary_io_threads,
        server_args=server_args,
        primary_args=primary_args,
        replica_args=replica_args,
        base_port=base_port,
    )


def _task(**overrides) -> ReplicaReadTaskData:
    """Build a task; topology levers (replica_count, io_threads, ...) are folded into the spec."""
    levers = {k: overrides.pop(k) for k in list(overrides) if k in _LEVERS}
    fields = dict(
        source=config.REPO_NAMES[0],
        specifier="unstable",
        make_args="",
        note="",
        requirements={},
        keyspace_count=1000,
        warmup=0,
        duration=5,
        repetitions=1,
        write_rate=20_000,
        sample_interval=0.01,
        topology=overrides.pop("topology", None) or _spec(**levers),
    )
    fields.update(overrides)
    return ReplicaReadTaskData(**fields)


def _cc_result(throughput: float, hit_pct: float = 100.0, err_pct: float = 0.0, command: str = "get") -> str:
    latency = {"p50_us": 100, "p90_us": 200, "p99_us": 400, "p999_us": 800, "p9999_us": 1600, "max_us": 3000}
    return json.dumps(
        {
            "type": "result",
            "throughput": throughput,
            "err_pct": err_pct,
            "hits": 100,
            "misses": 0,
            "hit_pct": hit_pct,
            command: dict(latency, count=1000),
        }
    )


# --------------------------------------------------------------------------- fakes


class FakeInstance:
    """Stands in for a started ``Server`` inside the topology."""

    profile_calls: list = []  # ("start", port, duration, delay) / ("collect", port) / ("cancel", port)
    profile_stacks = ([["main;a;b", 10]], [["io;c", 4]])  # what cpu_profile_collect returns
    profile_error: Optional[Exception] = None  # raise from cpu_profile_collect instead

    def __init__(self, port: int, pid: int, cpus: list, keys: int):
        self.ip = "127.0.0.1"
        self.port = port
        self.valkey_pid = pid
        self.server_cpus = cpus
        self.keys = keys

    async def count_items_expires(self):
        return self.keys, 0

    def get_build_hash(self):
        return "deadbeef"

    async def run_host_command(self, command, check=True):  # pylint: disable=unused-argument
        return f"ran: {command}", ""

    def cpu_profile_start(self, duration, delay_seconds=0):
        FakeInstance.profile_calls.append(("start", self.port, duration, delay_seconds))

    async def cpu_profile_collect(self):
        FakeInstance.profile_calls.append(("collect", self.port))
        if FakeInstance.profile_error is not None:
            raise FakeInstance.profile_error
        return FakeInstance.profile_stacks

    def cpu_profile_cancel(self):
        FakeInstance.profile_calls.append(("cancel", self.port))


class FakeGroup:
    """Records lifecycle calls; serves samples with a controlled 1 s clock."""

    lag_bytes = 50  # replica offset behind the primary in every sample; the stream advances 1000 B/s

    instances = {}  # port -> FakeInstance, set per test
    events: list = []

    def __init__(self, host, spec, source, specifier, make_args):  # pylint: disable=too-many-arguments
        self.spec = spec
        self.args = (host, spec, source, specifier, make_args)
        self.primary = FakeGroup.instances[spec.primary.port]
        self.replicas = [FakeGroup.instances[r.port] for r in spec.replicas]
        self.servers = [self.primary, *self.replicas]
        self._samples = 0
        self._t0 = None

    async def kill_all_valkey_instances(self):
        FakeGroup.events.append("kill")

    async def start(self):
        FakeGroup.events.append("start")

    async def begin_replication(self):
        FakeGroup.events.append("replicaof")

    async def wait_for_repl_sync(self, timeout_s=600.0):  # pylint: disable=unused-argument
        FakeGroup.events.append("wait-sync")

    async def wait_for_offsets_caught_up(self, timeout_s=300.0):  # pylint: disable=unused-argument
        FakeGroup.events.append("wait-offsets")

    async def stop_all_servers(self):
        FakeGroup.events.append("stop-all")

    pins: list = []  # pin_cpus seen by sample(), per call
    host_scans: list = []  # host_threads flag seen by sample(), per call

    async def sample(
        self, extra_fields=None, cpu=True, pin_cpus="", host_threads=False
    ):  # pylint: disable=unused-argument
        FakeGroup.pins.append(pin_cpus)
        FakeGroup.host_scans.append(host_threads)
        """Cumulative counters advancing one second per call; replica main spins, cores idle."""
        if self._t0 is None:
            self._t0 = time.monotonic()
        n = self._samples
        self._samples += 1
        ticks = n * cs.USER_HZ
        cores = {c: {"busy": 0, "idle": ticks, "softirq": 0, "total": ticks} for c in range(8)}
        for c in self.replicas[0].server_cpus:
            cores[c] = {"busy": ticks, "idle": 0, "softirq": 0, "total": ticks}
        return {
            "t": self._t0 + n,
            "instances": {
                self.primary.port: {"master_repl_offset": 1000 * n, "eventloop_duration_sum": 100_000 * n},
                self.replicas[0].port: {
                    "master_repl_offset": 1000 * n - FakeGroup.lag_bytes,
                    "eventloop_duration_sum": 800_000 * n,
                },
            },
            "cores": cores,
            "threads": {
                self.primary.port: {PRIMARY_PID: {"comm": "valkey-server", "ticks": ticks // 10}},
                self.replicas[0].port: {REPLICA_PID: {"comm": "valkey-server", "ticks": ticks}},
            },
        }


class FakeAllocator:
    """Records what the runner asked for through Server's client-CPU API."""

    def __init__(self):
        self.allocated = {}
        self.released = []

    def allocate(self, tag, count):
        cpus = list(range(20 + 10 * len(self.allocated), 20 + 10 * len(self.allocated) + count))
        self.allocated[tag.task_id] = cpus
        return cpus


class FakeLocalServer:
    """The module-level ``Server``: generator host (client CPUs) and drop-caches host."""

    allocator = FakeAllocator()
    commands: list = []

    def __init__(self, ip, port=None, username=None, instance_dir=None):  # pylint: disable=unused-argument
        self.ip = ip

    async def ensure_host_cpu_allocation(self):
        return None

    def allocate_client_cpus(self, tag, count, *, avoid_tags, minimize_cache_groups=False):
        assert avoid_tags, "generators must be placed away from the server instances"
        assert not minimize_cache_groups
        return FakeLocalServer.allocator.allocate(tag, count)

    def allocated_cpu_list(self, tag):
        cpus = FakeLocalServer.allocator.allocated.get(tag.task_id)
        return ",".join(map(str, cpus)) if cpus else ""

    def release_cpus(self, tag):
        FakeLocalServer.allocator.released.append(tag.task_id)

    async def run_host_command(self, command, check=True):  # pylint: disable=unused-argument
        FakeLocalServer.commands.append(command)
        return "", ""


class FakeProcess:
    def __init__(self):
        self.pid = IMPOSSIBLE_PID
        self.returncode = None


class FakeCommand:
    """cachecannon: which TOML it got decides the canned result it prints."""

    results = {}  # role -> NDJSON line, set per test
    running_polls = {"preload": 1, "writer": 1, "reader": 8}
    launched: list = []
    commands: list = []  # full command strings, in launch order
    launch_cpus: list = []  # (role, launch_cpus kwarg) in launch order

    def __init__(self, command: str, remote=None, launch_cpus=None):
        self.command = command
        self.remote = remote
        self.role = next(r for r in ("preload", "writer", "reader") if f"/{r}_rep" in command)
        self.p = FakeProcess()
        self._polls = 0
        self._emitted = False
        self._launch_cpus = launch_cpus

    def start(self):
        FakeCommand.launched.append(self.role)
        FakeCommand.commands.append(self.command)
        FakeCommand.launch_cpus.append((self.role, self._launch_cpus))

    def is_running(self):
        self._polls += 1
        if self._polls <= FakeCommand.running_polls[self.role]:
            return True
        self.p.returncode = 0
        return False

    def poll_output(self):
        if self._emitted:
            return None, None
        self._emitted = True
        return FakeCommand.results[self.role], ""


@pytest.fixture
def faked(monkeypatch, tmp_path):
    """Install every fake, reset their recorders, isolate the file protocol under tmp_path."""
    FakeGroup.events = []
    FakeGroup.instances = {
        6379: FakeInstance(6379, PRIMARY_PID, [0], keys=1000),
        6380: FakeInstance(6380, REPLICA_PID, [1, 2], keys=1000),
    }
    FakeLocalServer.allocator = FakeAllocator()
    FakeLocalServer.commands = []
    FakeCommand.launched = []
    FakeCommand.commands = []
    FakeCommand.launch_cpus = []
    FakeGroup.pins = []
    FakeGroup.host_scans = []
    FakeGroup.lag_bytes = 50
    FakeInstance.profile_calls = []
    FakeInstance.profile_error = None
    FakeCommand.results = {
        "preload": _cc_result(50_000),
        "writer": _cc_result(19_800, hit_pct=0.0, command="set"),
        "reader": _cc_result(265_000),
    }
    real_sleep = asyncio.sleep

    async def yielding_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(module, "TopologyGroup", FakeGroup)
    monkeypatch.setattr(module, "Server", FakeLocalServer)
    monkeypatch.setattr(module, "RealtimeCommand", FakeCommand)
    monkeypatch.setattr(module.asyncio, "sleep", yielding_sleep)

    def make_runner(**overrides) -> ReplicaReadTaskRunner:
        runner = ReplicaReadTaskRunner(_task(**overrides), [HOST])
        runner.file_protocol = FileProtocol(runner.task_name, role_id="client", base_dir=tmp_path)
        runner.file_protocol.write_results = MagicMock()
        runner.management_cpus = "190,191"  # what the task runner loop hands every task
        return runner

    return make_runner


# --------------------------------------------------------------------------- run()


@pytest.mark.asyncio
async def test_run_executes_phases_in_order_and_records_once(faked):
    runner = faked(repetitions=2)

    await runner.run()

    # Rep 1 has no teardown before it; rep 2 stops servers and drops caches first.
    per_rep = ["kill", "start", "replicaof", "wait-sync", "wait-offsets"]
    assert FakeGroup.events == per_rep + ["stop-all"] + per_rep + ["stop-all"]
    assert any("drop_caches" in c for c in FakeLocalServer.commands)
    # Generators launch preload -> writer -> reader each rep.
    assert FakeCommand.launched == ["preload", "writer", "reader"] * 2
    # One TOML per generator per rep, numbered from 1.
    names = sorted(p.name for p in runner.file_protocol.work_dir.glob("*.toml"))
    assert names == sorted(f"{r}_rep{n}.toml" for r in ("preload", "writer", "reader") for n in (1, 2))
    # Generator CPUs allocated once, both released at the end.
    assert sorted(FakeLocalServer.allocator.released) == sorted(FakeLocalServer.allocator.allocated)
    assert len(FakeLocalServer.allocator.released) == 2
    # The sampler shell is pinned to the management cores the task runner loop handed us.
    assert set(FakeGroup.pins) == {"190,191"}
    # Every sample carries the host-wide thread scan the foreign-core attribution needs.
    assert FakeGroup.host_scans and all(FakeGroup.host_scans)
    # Each generator is launched under its own allocation (not the runner's whole
    # mask): cachecannon pins only its workers, so its main/admin threads must
    # inherit claimed cores or the verdict sees them as foreign work.
    allocated = FakeLocalServer.allocator.allocated
    reader_cpus = ",".join(map(str, allocated[f"{runner.task_name}_reader"]))
    writer_cpus = ",".join(map(str, allocated[f"{runner.task_name}_writer"]))
    assert reader_cpus and writer_cpus and reader_cpus != writer_cpus
    assert FakeCommand.launch_cpus == [("preload", writer_cpus), ("writer", writer_cpus), ("reader", reader_cpus)] * 2
    # The command string itself stays plain; the prefix is RealtimeCommand's job.
    assert all(not cmd.startswith("taskset") for cmd in FakeCommand.commands)
    # Recorded exactly once, with both reps and the reader's exact throughput as the score.
    runner.file_protocol.write_results.assert_called_once()
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.method == METHOD and results.reps == 2 and results.score == 265_000.0
    assert results.commit_hash == "deadbeef"
    assert results.data["write_rate_achieved_mean"] == 19_800.0
    assert results.data["bottleneck"]["verdict"] == cs.VERDICT_SERVER
    assert results.data["bottleneck"]["per_rep"] == [cs.VERDICT_SERVER] * 2
    assert runner.status.state == "completed"


@pytest.mark.asyncio
async def test_run_stops_servers_and_releases_generators_when_a_rep_fails(faked):
    FakeGroup.instances[6380].keys = 999  # replica short one key -> preload mismatch
    runner = faked()

    with pytest.raises(RuntimeError, match="preload mismatch"):
        await runner.run()

    assert FakeGroup.events[-1] == "stop-all"
    assert len(FakeLocalServer.allocator.released) == 2
    assert FakeCommand.launched == ["preload"]  # never reached the measure phase
    runner.file_protocol.write_results.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_client_cpu_override_skips_the_allocator(faked):
    runner = faked(benchmark_cpu_override="40,41")

    await runner.run()

    # Generators take the override; management cores are the task runner loop's business, not this task's.
    assert FakeLocalServer.allocator.allocated == {}
    assert FakeLocalServer.allocator.released == []
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert 'cpu_list = "40,41"' in results.data["reader_toml"]
    # The override is the allocation: every generator launches confined to it.
    assert {cpus for _, cpus in FakeCommand.launch_cpus} == {"40,41"}
    # An override is still a pinning: the CPU map is complete, so the foreign check runs.
    rep = results.data["per_rep_results"][0]["bottleneck"]
    assert rep["cores"]["foreign_checked"] is True


# --------------------------------------------------------------------------- cpu profile


@pytest.mark.asyncio
async def test_run_profiles_the_replica_on_the_last_rep_only(faked):
    runner = faked(repetitions=3, warmup=5, duration=30, cpu_profile=True)

    await runner.run()

    # One perf record, on the measured replica, started after the reader launch
    # and delayed by the warmup so only the scored window is sampled.
    assert FakeInstance.profile_calls == [("start", 6380, 30, 5), ("collect", 6380)]
    reader_launches = [i for i, role in enumerate(FakeCommand.launched) if role == "reader"]
    assert len(reader_launches) == 3
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.data["cpu_profile"] is True
    assert results.data["cpu_profile_rep"] == 3
    assert results.data["cpu_stacks_main"] == [["main;a;b", 10]]
    assert results.data["cpu_stacks_io"] == [["io;c", 4]]
    assert results.reps == 3 and results.score == 265_000.0


@pytest.mark.asyncio
async def test_run_without_cpu_profile_never_touches_perf(faked):
    runner = faked(repetitions=2)

    await runner.run()

    assert FakeInstance.profile_calls == []
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.data["cpu_profile"] is False
    assert "cpu_stacks_main" not in results.data and "cpu_profile_rep" not in results.data


@pytest.mark.asyncio
async def test_failed_profile_collection_leaves_the_row_without_stacks(faked):
    FakeInstance.profile_error = RuntimeError("perf script exploded")
    runner = faked(repetitions=1, cpu_profile=True)

    await runner.run()  # a diagnostic must never fail the cell

    assert FakeInstance.profile_calls == [("start", 6380, runner.task.duration, runner.task.warmup), ("collect", 6380)]
    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.data["cpu_profile"] is True
    assert "cpu_stacks_main" not in results.data
    assert runner.status.state == "completed"


@pytest.mark.asyncio
async def test_reader_failure_cancels_the_profile_instead_of_waiting_for_it(faked):
    # A reader that reports errors fails inside the measure phase, while perf
    # may still be recording: cancel it rather than wait out the window.
    FakeCommand.results["reader"] = _cc_result(1000, err_pct=2.0)
    runner = faked(repetitions=1, cpu_profile=True)

    with pytest.raises(RuntimeError, match="reader cachecannon reported"):
        await runner.run()

    assert FakeInstance.profile_calls == [("start", 6380, runner.task.duration, runner.task.warmup), ("cancel", 6380)]


@pytest.mark.asyncio
async def test_guard_failure_after_the_window_still_collects_the_profile(faked):
    # The hit-rate guard runs in _judge, after the reader finished cleanly, so
    # the completed profile is collected before the cell is failed.
    FakeCommand.results["reader"] = _cc_result(1000, hit_pct=50.0)
    runner = faked(repetitions=1, cpu_profile=True)

    with pytest.raises(RuntimeError, match="hit rate"):
        await runner.run()

    assert FakeInstance.profile_calls == [("start", 6380, runner.task.duration, runner.task.warmup), ("collect", 6380)]


def test_cpu_profile_appears_in_descriptions():
    on, off = _task(cpu_profile=True), _task()
    assert on.short_description().endswith(", cpu-profile")
    assert "cpu-profile" not in off.short_description()
    assert ReplicaReadTaskRunner(on, [HOST]).title.endswith(", cpu-profile")


# --------------------------------------------------------------------------- guards


NO_LAG = {"samples": 5, "mean_bytes": 0, "max_bytes": 0, "stream_bytes_per_second": 1000.0, "mean_seconds": 0.0}


def _lag(mean_seconds, *, rate=100_000.0, samples=5):
    mean_bytes = mean_seconds * rate
    return {
        "samples": samples,
        "mean_bytes": mean_bytes,
        "max_bytes": int(mean_bytes * 2),
        "stream_bytes_per_second": rate,
        "mean_seconds": mean_seconds,
    }


def test_check_guards_rejects_low_hit_rate_naming_the_counts(faked):
    runner = faked()
    reader = {"hit_rate": {"percent": 98.9, "hits": 989.0, "misses": 11.0}, "throughput_rps": 1.0}
    writer = {"throughput_rps": 20_000.0}
    with pytest.raises(RuntimeError, match=r"hit rate 98.9% < 99.0% \(989.0 hits, 11.0 misses\)"):
        runner._check_guards(reader, writer, NO_LAG)


def test_check_guards_rejects_underdelivering_writer(faked):
    runner = faked(write_rate=20_000)
    reader = {"hit_rate": {"percent": 100.0}, "throughput_rps": 1.0}
    with pytest.raises(RuntimeError, match="writer achieved 17999/s"):
        runner._check_guards(reader, {"throughput_rps": 17_999.0}, NO_LAG)
    runner._check_guards(reader, {"throughput_rps": 18_000.0}, NO_LAG)  # exactly 90% passes


def test_check_guards_treats_missing_hit_rate_as_zero(faked):
    runner = faked()
    with pytest.raises(RuntimeError, match="hit rate 0.0%"):
        runner._check_guards({"hit_rate": None}, {"throughput_rps": 20_000.0}, NO_LAG)


def test_lag_guard_fails_a_replica_behind_its_primary(faked):
    runner = faked()  # default max_lag_seconds 1.0
    with pytest.raises(RuntimeError, match=r"replica lag 2.50 s mean over the scored window > max_lag_seconds 1.0"):
        runner._check_lag_guard(_lag(2.5))


def test_lag_guard_threshold_is_in_seconds_of_stream_not_bytes(faked):
    # The same 200 KB mean lag is 2 s behind a 100 KB/s stream and 0.02 s behind a 10 MB/s one.
    runner = faked()
    with pytest.raises(RuntimeError, match="replica lag 2.00 s"):
        runner._check_lag_guard(_lag(2.0, rate=100_000.0))
    runner._check_lag_guard(_lag(0.02, rate=10_000_000.0))


def test_lag_guard_passes_at_the_threshold_and_honours_the_task_value(faked):
    runner = faked(max_lag_seconds=3.0)
    runner._check_lag_guard(_lag(3.0))
    with pytest.raises(RuntimeError, match="> max_lag_seconds 3.0"):
        runner._check_lag_guard(_lag(3.01))


def test_lag_guard_names_the_evidence(faked):
    runner = faked()
    with pytest.raises(RuntimeError) as exc:
        runner._check_lag_guard(_lag(2.0, rate=100_000.0, samples=30))
    assert "mean 200,000 bytes, max 400,000 bytes over 30 samples, stream 100,000 bytes/s" in str(exc.value)


def test_lag_guard_fails_when_lag_exists_but_the_stream_did_not_advance(faked):
    runner = faked()
    stalled = {
        "samples": 5,
        "mean_bytes": 700.0,
        "max_bytes": 700,
        "stream_bytes_per_second": 0.0,
        "mean_seconds": None,
    }
    with pytest.raises(RuntimeError, match="stream that did not advance"):
        runner._check_lag_guard(stalled)


def test_lag_guard_skips_with_a_warning_when_the_window_has_too_few_samples(faked, caplog):
    runner = faked()
    with caplog.at_level(logging.WARNING):
        runner._check_lag_guard({"samples": 1, "mean_bytes": 10**9, "max_bytes": 10**9, "mean_seconds": None})
        runner._check_lag_guard({"samples": 0})
    assert caplog.text.count("lag guard skipped") == 2


def test_max_lag_seconds_must_be_positive():
    with pytest.raises(ValueError, match="max_lag_seconds must be > 0"):
        _task(max_lag_seconds=0)
    with pytest.raises(ValueError, match="max_lag_seconds must be > 0"):
        _task(max_lag_seconds=-1)
    assert _task(max_lag_seconds="2.5").max_lag_seconds == 2.5  # queue files hand strings back


@pytest.mark.asyncio
async def test_run_fails_the_rep_when_the_replica_lags(faked):
    # The fake stream advances 1000 B/s; a 5000 B lag is five seconds behind.
    FakeGroup.lag_bytes = 5000
    runner = faked()
    with pytest.raises(RuntimeError, match="replica lag 5.00 s mean over the scored window"):
        await runner.run()
    assert "stop-all" in FakeGroup.events  # the topology is torn down after a guard failure


# --------------------------------------------------------------------------- judge


@pytest.mark.asyncio
async def test_judge_passes_replica_identity_and_placement_into_the_verdict(faked, monkeypatch):
    runner = faked(warmup=7)
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    captured = {}

    def fake_verdict(samples, parties, **kwargs):
        captured["samples"] = samples
        captured["parties"] = parties
        captured.update(kwargs)
        return {"verdict": cs.VERDICT_SERVER, "valid": True, "reason": "test"}

    monkeypatch.setattr(module.cpu_sampling, "bottleneck_verdict", fake_verdict)
    samples = [
        {"t": 100.0, "instances": {6379: {"master_repl_offset": 500}, 6380: {"master_repl_offset": 300}}, "cores": {}},
        {
            "t": 101.0,
            "instances": {6379: {"master_repl_offset": 900}, 6380: {"master_repl_offset": 850}},
            "threads": {},
        },
    ]
    m = _Measurement(
        reader={"throughput_rps": 1.0, "latency": {"p99_ms": 1}, "hit_rate": {"percent": 100.0}},
        writer={"throughput_rps": 20_000.0, "latency": {"p99_ms": 2}},
        samples=samples,
        reader_started=1000.0,
        reader_toml="r",
        writer_toml="w",
    )

    row = runner._judge(group, 1, _Placement(reader_cpus="20,21", writer_cpus="30", runner_cpus="190,191"), m)

    parties = captured["parties"]
    assert parties.replica == cs.ServerIdentity(6380, REPLICA_PID)
    assert parties.primary == cs.ServerIdentity(6379, PRIMARY_PID)
    assert parties.allocated_cpus == {
        "primary": [0],
        "replica": [1, 2],
        "reader": [20, 21],
        "writer": [30],
        "runner": [190, 191],
    }
    assert captured["window_start"] == 1007.0  # reader start + warmup
    assert row["lag"]["max_bytes"] == 200 and row["bottleneck"]["verdict"] == cs.VERDICT_SERVER
    # Stored samples keep the INFO series but drop the raw cumulative counters.
    assert all("cores" not in s and "threads" not in s for s in row["samples"])
    assert row["samples"][0]["instances"][6379]["master_repl_offset"] == 500


def test_judge_applies_guards_before_computing_anything(faked):
    runner = faked()
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    m = _Measurement(
        reader={"throughput_rps": 1.0, "latency": None, "hit_rate": {"percent": 10.0}},
        writer={"throughput_rps": 20_000.0, "latency": None},
        samples=[],
        reader_started=0.0,
        reader_toml="",
        writer_toml="",
    )
    with pytest.raises(RuntimeError, match="hit rate"):
        runner._judge(group, 1, _Placement("", ""), m)


def test_allocated_cpus_includes_extra_replicas(faked):
    FakeGroup.instances[6381] = FakeInstance(6381, 3003, [5, 6], keys=1000)
    runner = faked(replica_count=2)
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    allocated = runner._allocated_cpus(group, _Placement(reader_cpus="", writer_cpus="9"))
    assert allocated["replica:6381"] == [5, 6]
    assert allocated["reader"] == [] and allocated["writer"] == [9]


# --------------------------------------------------------------------------- sample loop


@pytest.mark.asyncio
async def test_sample_loop_merges_remote_sample_with_local_generator_reads(faked, monkeypatch):
    runner = faked()
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    seen_pids = []

    def fake_local(pid):
        seen_pids.append(pid)
        return {pid: {"comm": f"gen-{pid}", "ticks": 1}} if pid else {}

    monkeypatch.setattr(module.cpu_sampling, "read_local_thread_stats", fake_local)
    reader = FakeCommand("/x/reader_rep1.toml")
    reader.p.pid = 111
    writer = FakeCommand("/x/writer_rep1.toml")
    writer.p.pid = 222
    samples: list = []

    await runner._sample_loop(group, samples, reader, writer)

    assert samples, "sampler produced nothing while the reader was running"
    first = samples[0]
    assert set(first) >= {"t", "t_local", "instances", "cores", "threads", "generators"}
    assert first["generators"]["reader"] == {111: {"comm": "gen-111", "ticks": 1}}
    assert first["generators"]["writer"] == {222: {"comm": "gen-222", "ticks": 1}}
    assert seen_pids[:2] == [111, 222]


@pytest.mark.asyncio
async def test_sample_loop_survives_a_failed_sample(faked, monkeypatch):
    runner = faked()
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    calls = {"n": 0}
    good_sample = group.sample

    async def flaky(extra_fields=None, cpu=True, pin_cpus="", host_threads=False):  # pylint: disable=unused-argument
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ssh hiccup")
        return await good_sample(extra_fields, cpu, pin_cpus, host_threads)

    monkeypatch.setattr(group, "sample", flaky)
    reader = FakeCommand("/x/reader_rep1.toml")
    samples: list = []

    await runner._sample_loop(group, samples, reader, None)

    assert calls["n"] >= 2 and len(samples) == calls["n"] - 1
    assert samples[0]["generators"]["writer"] == {}  # no writer command -> empty, not an error


# --------------------------------------------------------------------------- result row


def _rep(reader_rps: float, verdict: str, loop_duty: float = 0.8, reader_util: float = 0.5) -> dict:
    return {
        "reader_rps": reader_rps,
        "reader_latency": {"p99_ms": 1.0},
        "reader_hit_rate": {"percent": 100.0},
        "writer_rps": 19_900.0,
        "writer_latency": {"p99_ms": 2.0},
        "lag": {"samples": 5, "mean_bytes": 10.0, "max_bytes": 40, "p99_bytes": 40},
        "bottleneck": {
            "verdict": verdict,
            "valid": verdict == cs.VERDICT_SERVER,
            "reason": f"because {verdict}",
            "replica": {"loop_duty": loop_duty},
            "reader": {"max_thread_util": reader_util},
        },
        "samples": [],
    }


def test_aggregate_bottleneck_is_server_only_when_every_rep_is(faked):
    runner = faked()
    agg = runner._aggregate_bottleneck([_rep(1.0, cs.VERDICT_SERVER, 0.7), _rep(1.0, cs.VERDICT_SERVER, 0.9)])
    assert agg["verdict"] == cs.VERDICT_SERVER and agg["valid"]
    assert agg["replica_loop_duty_mean"] == pytest.approx(0.8)

    agg = runner._aggregate_bottleneck(
        [_rep(1.0, cs.VERDICT_SERVER), _rep(1.0, cs.VERDICT_GENERATOR, reader_util=0.99), _rep(1.0, cs.VERDICT_SERVER)]
    )
    assert agg["verdict"] == cs.VERDICT_GENERATOR and not agg["valid"]
    assert agg["per_rep"] == [cs.VERDICT_SERVER, cs.VERDICT_GENERATOR, cs.VERDICT_SERVER]
    assert agg["reason"] == "because generator"
    assert agg["reader_max_thread_util"] == 0.99


@pytest.mark.asyncio
async def test_record_result_row_shape(faked):
    runner = faked(repetitions=3, io_threads=8, write_rate=20_000)
    runner.commit_hash = "cafebabe"
    group = FakeGroup(HOST, runner.spec, "valkey", "unstable", "")
    reps = [_rep(100.0, cs.VERDICT_SERVER), _rep(110.0, cs.VERDICT_SERVER), _rep(90.0, cs.VERDICT_HOST)]

    await runner._record_result(group, reps, "reader-toml", "writer-toml")

    results: BenchmarkResults = runner.file_protocol.write_results.call_args.args[0]
    assert results.method == METHOD and results.commit_hash == "cafebabe"
    assert results.score == pytest.approx(100.0) and results.reps == 3
    assert results.cv == pytest.approx(10.0)  # stdev 10 / mean 100
    data = results.data
    assert data["per_run_rps"] == [100.0, 110.0, 90.0]
    assert data["topology"]["instances"][1]["role"] == "replica"
    assert data["management_cpus"] == "190,191"  # what the runner loop handed this task, recorded on the row
    assert data["server_cpus"] == {6379: [0], 6380: [1, 2]}
    assert data["io-threads"] == 8 and data["write_rate_target"] == 20_000
    assert data["write_rate_achieved_mean"] == 19_900.0
    assert data["replication_lag"] == {
        "max_bytes": 40,
        "mean_bytes": 10.0,
        "max_scored_mean_seconds": 0.0,
        "max_lag_seconds": 1.0,
    }
    assert data["bottleneck"]["verdict"] == cs.VERDICT_HOST and data["bottleneck"]["per_rep"][2] == cs.VERDICT_HOST
    assert data["reader_toml"] == "reader-toml" and data["writer_toml"] == "writer-toml"
    assert data["lscpu"].startswith("ran: lscpu")
    assert len(data["per_rep_results"]) == 3
