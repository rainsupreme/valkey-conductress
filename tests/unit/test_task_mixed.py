"""Unit tests for the mixed GET/SET throughput task.

Round 2: adds empirical client CPU measurement tests (GNU time parsing),
upper-bound validation, allocated_cores semantics, and result schema checks.

Round 3: adds taskset pinning enforcement tests, command construction
assertions (default vs pinned), mixed-task-specific capacity model tests,
and end-to-end runner-path mocks proving GNU-time → utilization → saturated.

Round 4: delayed-start perf/profile measurement boundary tests.

Round 5: cancellation, lifecycle correctness, cleanup.

Consolidated: shared fixtures for mock servers, replication groups, and
runner construction to reduce boilerplate across end-to-end tests.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.task_queue import TaskQueue
from conductress.tasks.task_mixed import (
    MAX_MEMTIER_CLIENTS,
    MAX_MEMTIER_THREADS,
    MIXED_CLIENTS,
    MIXED_THREADS,
    MixedTaskData,
    MixedTaskRunner,
    _effective_memtier_clients,
    _effective_memtier_threads,
    _validate_memtier_bounds,
    parse_gnu_time_stderr,
    parse_memtier_total_rps,
    set_ratio_to_memtier_ratio,
)

# =============================================================================
# Shared test constants and helpers
# =============================================================================

MEMTIER_STDOUT_FULL = (
    "Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency"
    "     p50 Latency     p99 Latency   p99.9 Latency       KB/sec\n"
    "--------------------------------------------------------------"
    "------------------------------------------------------------------\n"
    "Sets        250000.00          ---          ---"
    "         0.100         0.100         0.500         1.000      1234.56\n"
    "Gets       1000000.00    1000000.00         0.00"
    "         0.050         0.080         0.400         0.900      5678.90\n"
    "Totals     1250000.00    1000000.00         0.00"
    "         0.060         0.085         0.420         0.920      6913.46\n"
)

MEMTIER_STDOUT_SHORT = (
    "Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency\n"
    "Totals     1250000.00    1000000.00         0.00         0.060\n"
)

GNU_TIME_STDERR_8CORE = (
    '\tCommand being timed: "memtier_benchmark ..."\n'
    "\tUser time (seconds): 210.50\n"
    "\tSystem time (seconds): 30.20\n"
    "\tPercent of CPU this job got: 802%\n"
    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
    "\tMaximum resident set size (kbytes): 524288\n"
)

_RUNNER_DEFAULTS = dict(
    task_name="test",
    server_infos=[],
    source="valkey",
    specifier="unstable",
    make_args="",
    io_threads=9,
    val_size=512,
    pipelining=10,
    set_ratio=20,
    warmup=5,
    duration=30,
    repetitions=3,
)


def make_runner(**overrides) -> MixedTaskRunner:
    """Build a MixedTaskRunner with sensible defaults."""
    kw = {**_RUNNER_DEFAULTS, **overrides}
    return MixedTaskRunner(**kw)


def make_mock_server(ip: str = "10.0.0.1", port: int = 6379) -> MagicMock:
    """Build a mock Server with stub methods for e2e tests."""
    server = MagicMock()
    server.ip = ip
    server.port = port
    server.get_build_hash.return_value = "abc123"
    return server


def make_repl_group(server: MagicMock) -> MagicMock:
    """Build a mock ReplicationGroup wired to *server* as primary."""
    rg = MagicMock()
    rg.primary = server
    rg.start = AsyncMock()
    rg.stop_all_servers = AsyncMock()
    rg.kill_all_valkey_instances = AsyncMock()
    return rg


def wire_file_protocol(runner: MixedTaskRunner) -> list:
    """Attach a mock file_protocol to *runner*; returns the results list."""
    results: list = []
    runner.file_protocol = MagicMock()
    runner.file_protocol.get_result_dir.return_value = "/tmp/results"
    runner.file_protocol.write_results = MagicMock(side_effect=lambda r: results.append(r))
    return results


def gnu_time_cmd_router(
    *,
    gnu_time_available: bool = True,
    memtier_stdout: str = MEMTIER_STDOUT_FULL,
    gnu_time_stderr: str = GNU_TIME_STDERR_8CORE,
    fail_on_measure: bool = False,
    garbage_output: bool = False,
):
    """Return an async side_effect for server.run_host_command covering
    the standard GNU-time probe → prefill → measure → result pipeline."""

    async def capture_cmd(cmd, check=True):
        if "time --version" in cmd:
            return ("GNU time 1.9", "") if gnu_time_available else ("", "")
        if "key-pattern P:P" in cmd:
            return ("OK", "")
        if fail_on_measure and ("--test-time" in cmd or "/usr/bin/time -v" in cmd):
            raise RuntimeError("Simulated memtier crash")
        if garbage_output and ("--test-time" in cmd or "/usr/bin/time -v" in cmd):
            return ("GARBAGE OUTPUT", "")
        if "/usr/bin/time -v" in cmd:
            return (memtier_stdout, gnu_time_stderr)
        return (memtier_stdout, "")

    return capture_cmd


@pytest.fixture(autouse=True)
def _patch_sources():
    """Patch config repo names for all tests in this module."""
    with (
        patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
        patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
        patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
        patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
    ):
        yield


def _make_task(**overrides) -> MixedTaskData:
    """Helper to construct a MixedTaskData with reasonable defaults."""
    defaults = dict(
        source="valkey",
        specifier="unstable",
        make_args="",
        replicas=0,
        note="",
        requirements={},
        set_ratio=20,
        val_size=512,
        io_threads=9,
        pipelining=10,
        duration=30,
    )
    defaults.update(overrides)
    return MixedTaskData(**defaults)


# =============================================================================
# Pure-function unit tests (no mocking needed)
# =============================================================================


class TestSetRatioConversion:
    """Tests for set_ratio_to_memtier_ratio conversion."""

    @pytest.mark.parametrize(
        ("pct", "expected"),
        [
            (0, "0:1"),
            (10, "1:9"),
            (20, "1:4"),
            (33, "33:67"),
            (50, "1:1"),
            (75, "3:1"),
            (100, "1:0"),
        ],
    )
    def test_conversion(self, pct, expected):
        assert set_ratio_to_memtier_ratio(pct) == expected


class TestMemtierOutputParsing:
    """Tests for memtier stdout parsing."""

    def test_parse_totals_line(self):
        result = parse_memtier_total_rps(MEMTIER_STDOUT_FULL)
        assert result == 1250000.00

    def test_parse_no_totals(self):
        assert parse_memtier_total_rps("some random output\nno totals here") is None

    def test_parse_empty_output(self):
        assert parse_memtier_total_rps("") is None


class TestGnuTimeParsing:
    """Tests for parse_gnu_time_stderr — core of empirical client CPU measurement."""

    SAMPLE = (
        '\tCommand being timed: "memtier_benchmark ..."\n'
        "\tUser time (seconds): 18.42\n"
        "\tSystem time (seconds): 6.31\n"
        "\tPercent of CPU this job got: 82%\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.15\n"
        "\tMaximum resident set size (kbytes): 524288\n"
    )

    def test_parse_valid_output(self):
        r = parse_gnu_time_stderr(self.SAMPLE)
        assert r is not None
        assert r["user_seconds"] == 18.42
        assert r["system_seconds"] == 6.31
        assert r["wall_seconds"] == pytest.approx(30.15, abs=0.01)
        assert r["cpu_seconds"] == pytest.approx(24.73, abs=0.01)
        assert r["cores_busy"] == pytest.approx(0.820, abs=0.01)

    def test_parse_with_hours(self):
        stderr = (
            "\tUser time (seconds): 3600.50\n"
            "\tSystem time (seconds): 100.25\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:30.75\n"
        )
        r = parse_gnu_time_stderr(stderr)
        assert r is not None
        assert r["wall_seconds"] == pytest.approx(3750.75, abs=0.01)

    @pytest.mark.parametrize(
        "stderr",
        [
            "",
            "WARNING: Connection timeout\nSome memtier warning\n",
            "\tSystem time (seconds): 6.31\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.15\n",
            "\tUser time (seconds): 18.42\n\tSystem time (seconds): 6.31\n",
        ],
        ids=["empty", "non_gnu", "missing_user", "missing_wall"],
    )
    def test_parse_returns_none(self, stderr):
        assert parse_gnu_time_stderr(stderr) is None

    def test_parse_zero_wall_time(self):
        stderr = (
            "\tUser time (seconds): 0.01\n"
            "\tSystem time (seconds): 0.00\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00\n"
        )
        r = parse_gnu_time_stderr(stderr)
        assert r is not None
        assert r["cores_busy"] == 0.0

    def test_parse_high_cpu_multithread(self):
        stderr = (
            "\tUser time (seconds): 210.50\n"
            "\tSystem time (seconds): 30.20\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
        )
        assert parse_gnu_time_stderr(stderr)["cores_busy"] > 8.0

    def test_result_fields_are_rounded(self):
        stderr = (
            "\tUser time (seconds): 1.23456789\n"
            "\tSystem time (seconds): 0.98765432\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:05.00\n"
        )
        r = parse_gnu_time_stderr(stderr)
        assert r["user_seconds"] == 1.235
        assert r["system_seconds"] == 0.988
        assert r["cpu_seconds"] == 2.222


class TestEffectiveMemtierDefaults:
    def test_zero_threads_uses_default(self):
        assert _effective_memtier_threads(0) == MIXED_THREADS == 8

    def test_zero_clients_uses_default(self):
        assert _effective_memtier_clients(0) == MIXED_CLIENTS == 50

    def test_positive_threads_overrides(self):
        assert _effective_memtier_threads(24) == 24

    def test_positive_clients_overrides(self):
        assert _effective_memtier_clients(100) == 100


class TestValidateMemtierBounds:
    def test_defaults_pass(self):
        _validate_memtier_bounds(0, 0)

    def test_reasonable_overrides_pass(self):
        _validate_memtier_bounds(24, 100)

    def test_max_bounds_pass(self):
        _validate_memtier_bounds(MAX_MEMTIER_THREADS, MAX_MEMTIER_CLIENTS)

    @pytest.mark.parametrize(
        ("threads", "clients", "match"),
        [
            (-1, 0, "memtier_threads must be >= 0"),
            (0, -1, "memtier_clients must be >= 0"),
            (MAX_MEMTIER_THREADS + 1, 0, f"memtier_threads must be <= {MAX_MEMTIER_THREADS}"),
            (0, MAX_MEMTIER_CLIENTS + 1, f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}"),
        ],
    )
    def test_invalid_rejected(self, threads, clients, match):
        with pytest.raises(ValueError, match=match):
            _validate_memtier_bounds(threads, clients)

    def test_zero_effective_connections_unreachable(self):
        assert _effective_memtier_threads(0) * _effective_memtier_clients(0) == 400

    def test_connections_exceed_keyspace_rejected(self):
        from conductress.tasks import task_mixed

        saved = task_mixed.MAX_MEMTIER_THREADS, task_mixed.MAX_MEMTIER_CLIENTS
        try:
            task_mixed.MAX_MEMTIER_THREADS = 10000
            task_mixed.MAX_MEMTIER_CLIENTS = 10000
            with pytest.raises(ValueError, match="prefill requests per connection would be zero"):
                _validate_memtier_bounds(5000, 5000)
        finally:
            task_mixed.MAX_MEMTIER_THREADS, task_mixed.MAX_MEMTIER_CLIENTS = saved


# =============================================================================
# MixedTaskData validation and serialization
# =============================================================================


class TestMixedTaskDataValidation:

    def test_valid_ratio_boundaries(self):
        assert _make_task(set_ratio=0).set_ratio == 0
        assert _make_task(set_ratio=100).set_ratio == 100

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("set_ratio", -1, "set_ratio must be 0-100"),
            ("set_ratio", 101, "set_ratio must be 0-100"),
            ("warmup", -1, "warmup must be >= 0"),
            ("key_size", 32, "key_size is not supported"),
            ("memtier_threads", -1, "memtier_threads must be >= 0"),
            ("memtier_clients", -1, "memtier_clients must be >= 0"),
            ("memtier_threads", MAX_MEMTIER_THREADS + 1, f"memtier_threads must be <= {MAX_MEMTIER_THREADS}"),
            ("memtier_clients", MAX_MEMTIER_CLIENTS + 1, f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}"),
        ],
    )
    def test_invalid_field_rejected(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            _make_task(**{field: value})

    def test_defaults_zero_means_default(self):
        t = _make_task()
        assert t.memtier_threads == 0
        assert t.memtier_clients == 0
        assert t.effective_threads == 8
        assert t.effective_clients == 50
        assert t.total_connections == 400

    def test_override_threads_24(self):
        t = _make_task(memtier_threads=24)
        assert t.effective_threads == 24
        assert t.total_connections == 1200

    def test_override_both_for_2400c(self):
        assert _make_task(memtier_threads=24, memtier_clients=100).total_connections == 2400

    def test_short_description_default_omits_concurrency(self):
        desc = _make_task().short_description()
        assert "20%SET" in desc
        assert "1200c" not in desc

    def test_short_description_override_shows_concurrency(self):
        assert "1200c(24t×50c)" in _make_task(memtier_threads=24, memtier_clients=50).short_description()

    def test_serialization_round_trip(self, tmp_path):
        task = _make_task(repetitions=5, perf_stat_enabled=True, note="test note")
        filepath = tmp_path / "task.json"
        task.save_to_file(filepath)
        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(filepath)
        assert isinstance(loaded, MixedTaskData)
        for attr in ("set_ratio", "val_size", "io_threads", "pipelining", "duration"):
            assert getattr(loaded, attr) == getattr(task, attr)
        assert loaded.perf_stat_enabled is True

    def test_serialization_round_trip_with_concurrency(self, tmp_path):
        task = _make_task(memtier_threads=24, memtier_clients=100)
        fp = tmp_path / "task.json"
        task.save_to_file(fp)
        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(fp)
        assert loaded.total_connections == 2400

    def test_backward_compat_load_without_new_fields(self, tmp_path):
        task = _make_task()
        fp = tmp_path / "task.json"
        task.save_to_file(fp)
        data = json.loads(fp.read_text())
        data.pop("memtier_threads", None)
        data.pop("memtier_clients", None)
        fp.write_text(json.dumps(data, indent=2))
        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(fp)
        assert loaded.total_connections == 400


# =============================================================================
# MixedTaskRunner construction
# =============================================================================


class TestMixedTaskRunnerInit:

    def test_default_concurrency(self):
        r = make_runner()
        assert (r.memtier_threads, r.memtier_clients, r.total_connections) == (8, 50, 400)

    def test_overridden_concurrency(self):
        r = make_runner(memtier_threads=24, memtier_clients=50)
        assert r.total_connections == 1200

    def test_client_telemetry_init(self):
        r = make_runner()
        assert r._client_cores_busy_per_rep == []
        assert r._gnu_time_available is None

    def test_no_taskset_without_cpu_override(self):
        assert make_runner()._build_taskset_prefix() == ""

    def test_with_benchmark_cpu_override(self):
        r = make_runner(benchmark_cpu_override="0-7,16-23")
        assert r._build_taskset_prefix() == "taskset -c 0-7,16-23 "


class TestClientCpuResultSchema:

    def test_empirical_cpu_schema(self):
        from conductress.utility import summarize_client_cpu

        r = summarize_client_cpu([4.5, 5.0, 4.8], allocated_cores=8)
        assert r["utilization"] == pytest.approx(0.625, abs=0.01)
        assert r["saturated"] is False

    def test_saturated_flag_at_threshold(self):
        from conductress.utility import summarize_client_cpu

        assert summarize_client_cpu([7.5], allocated_cores=8)["saturated"] is True

    def test_no_allocated_cores_no_utilization(self):
        from conductress.utility import summarize_client_cpu

        r = summarize_client_cpu([4.5, 5.0], allocated_cores=None)
        assert "utilization" not in r and "saturated" not in r


# =============================================================================
# Taskset prefix and command construction
# =============================================================================


class TestTasksetAndCommands:

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ("", ""),
            ("0-7,16-23", "taskset -c 0-7,16-23 "),
            ("5", "taskset -c 5 "),
        ],
    )
    def test_taskset_prefix(self, override, expected):
        r = make_runner(benchmark_cpu_override=override) if override else make_runner()
        assert r._build_taskset_prefix() == expected

    def test_taskset_prefix_has_trailing_space(self):
        assert make_runner(benchmark_cpu_override="0-3")._build_taskset_prefix().endswith(" ")

    def test_default_prefill_no_taskset(self):
        r = make_runner()
        pfx = r._build_taskset_prefix()
        cmd = f"{pfx}~/conductress/memtier_benchmark --server 10.0.0.1 --port 6379"
        assert cmd.startswith("~/conductress/memtier_benchmark")
        assert "taskset" not in cmd

    def test_default_measure_legacy_threads_clients(self):
        r = make_runner()
        cmd = f"{r._build_taskset_prefix()}~/conductress/memtier_benchmark --threads {r.memtier_threads} --clients {r.memtier_clients}"
        assert "--threads 8" in cmd and "--clients 50" in cmd and "taskset" not in cmd

    def test_pinned_prefill_has_taskset(self):
        r = make_runner(benchmark_cpu_override="0-7,16-23")
        cmd = f"{r._build_taskset_prefix()}~/conductress/memtier_benchmark --server 10.0.0.1"
        assert cmd.startswith("taskset -c 0-7,16-23 ")

    def test_timed_command_wraps_taskset(self):
        r = make_runner(benchmark_cpu_override="0-3")
        cmd = r._build_timed_command(f"{r._build_taskset_prefix()}~/conductress/memtier_benchmark --test-time 30")
        assert cmd.startswith("/usr/bin/time -v taskset -c 0-3 ")

    def test_timed_command_no_override(self):
        r = make_runner()
        cmd = r._build_timed_command("~/conductress/memtier_benchmark --test-time 30")
        assert cmd == "/usr/bin/time -v ~/conductress/memtier_benchmark --test-time 30"

    def test_positive_warmup_builds_memtier_option(self):
        assert make_runner(warmup=7)._build_warmup_arg() == "--warmup-period 7 "

    def test_zero_warmup_omits_memtier_option(self):
        assert make_runner(warmup=0)._build_warmup_arg() == ""


# =============================================================================
# Capacity model
# =============================================================================


class TestCapacityModel:
    PINNED_BASIS = "min(memtier_thread_count,taskset_cpulist)"

    def test_unpinned_default(self):
        m = make_runner()._compute_client_cpu_meta()
        assert m["capacity_cores"] == 8
        assert m["capacity_basis"] == "memtier_thread_count"

    def test_unpinned_24_threads(self):
        assert make_runner(memtier_threads=24)._compute_client_cpu_meta()["capacity_cores"] == 24

    def test_pinned_capacity_uses_lower_thread_limit(self):
        m = make_runner(benchmark_cpu_override="0-7,16-23")._compute_client_cpu_meta()
        assert m["capacity_cores"] == 8  # min(8 threads, 16 pinned)
        assert m["capacity_basis"] == self.PINNED_BASIS

    def test_pinned_single_cpu(self):
        assert make_runner(benchmark_cpu_override="5")._compute_client_cpu_meta()["capacity_cores"] == 1

    def test_unpinned_no_benchmark_field(self):
        assert "benchmark_cpu_override" not in make_runner()._compute_client_cpu_meta()

    def test_unavailable_no_utilization(self):
        m = make_runner()._compute_client_cpu_meta()
        assert m["measurement_method"] == "unavailable"
        assert "utilization" not in m

    def test_empirical_data_produces_utilization(self):
        r = make_runner()
        r._client_cores_busy_per_rep = [4.5, 5.0, 4.8]
        m = r._compute_client_cpu_meta()
        assert m["measurement_method"] == "gnu_time"
        assert m["utilization"] == pytest.approx(0.625, abs=0.01)
        assert m["saturated"] is False

    def test_empirical_saturated_flag(self):
        r = make_runner()
        r._client_cores_busy_per_rep = [7.5]
        assert r._compute_client_cpu_meta()["saturated"] is True

    def test_pinned_with_empirical(self):
        r = make_runner(memtier_threads=24, benchmark_cpu_override="0-15")
        r._client_cores_busy_per_rep = [12.0, 13.0, 12.5]
        m = r._compute_client_cpu_meta()
        assert m["capacity_cores"] == 16
        assert m["utilization"] == pytest.approx(0.8125, abs=0.01)

    def test_meta_has_connection_info(self):
        m = make_runner(memtier_threads=24, memtier_clients=100)._compute_client_cpu_meta()
        assert m["total_connections"] == 2400


class TestGnuTimeProbe:
    @pytest.mark.asyncio
    async def test_busybox_time_not_accepted(self):
        r = make_runner(repetitions=1)
        server = MagicMock()
        server.run_host_command = AsyncMock(return_value=("BusyBox time: unrecognized option --version", ""))
        assert await r._probe_gnu_time(server) is False


# =============================================================================
# End-to-end runner path tests (mock server + repl group)
# =============================================================================


class TestEndToEndRunnerMock:
    """E2E tests: commands, GNU time parsing, result schema, capacity model."""

    @pytest.mark.asyncio
    async def test_default_commands_no_taskset(self):
        runner = make_runner(repetitions=1)
        server = make_mock_server()
        commands: list[str] = []

        async def capture(cmd, check=True):
            commands.append(cmd)
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                return (MEMTIER_STDOUT_FULL, GNU_TIME_STDERR_8CORE)
            return (MEMTIER_STDOUT_FULL, "")

        server.run_host_command = AsyncMock(side_effect=capture)
        rg = make_repl_group(server)
        wire_file_protocol(runner)

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()

        for cmd in commands:
            assert "taskset" not in cmd
        prefills = [c for c in commands if "--key-pattern P:P" in c]
        assert len(prefills) == 1 and "--threads 8" in prefills[0] and "--clients 50" in prefills[0]
        measures = [c for c in commands if "/usr/bin/time -v" in c]
        assert len(measures) == 1 and "--warmup-period 5" in measures[0]

    @pytest.mark.asyncio
    async def test_pinned_commands_have_taskset(self):
        runner = make_runner(repetitions=1, benchmark_cpu_override="0-7,16-23")
        server = make_mock_server()
        commands: list[str] = []

        async def capture(cmd, check=True):
            commands.append(cmd)
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                return (MEMTIER_STDOUT_FULL, GNU_TIME_STDERR_8CORE)
            return (MEMTIER_STDOUT_FULL, "")

        server.run_host_command = AsyncMock(side_effect=capture)
        rg = make_repl_group(server)
        wire_file_protocol(runner)

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()

        prefills = [c for c in commands if "--key-pattern P:P" in c]
        assert prefills[0].startswith("taskset -c 0-7,16-23 ")
        timed = [c for c in commands if "/usr/bin/time -v" in c]
        assert "taskset -c 0-7,16-23" in timed[0]

    @pytest.mark.asyncio
    async def test_gnu_time_parsed_into_result(self):
        runner = make_runner(repetitions=1, server_cpu_override="0-8", server_args="--io-threads-ownership yes")
        server = make_mock_server()
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router())
        rg = make_repl_group(server)
        results = wire_file_protocol(runner)

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()

        assert len(results) == 1
        d = results[0].data
        assert d["server_cpu_override"] == "0-8"
        assert d["warmup_applied"] is True
        cc = d["client_cpu"]
        assert cc["measurement_method"] == "gnu_time"
        assert cc["cores_busy_per_rep"][0] == pytest.approx(8.023, abs=0.01)
        assert cc["saturated"] is True

    @pytest.mark.asyncio
    async def test_pinned_capacity_in_result(self):
        runner = make_runner(repetitions=1, benchmark_cpu_override="0-15", memtier_threads=24)
        server = make_mock_server()

        async def capture(cmd, check=True):
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                stderr = (
                    "\tUser time (seconds): 300.00\n"
                    "\tSystem time (seconds): 60.00\n"
                    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
                )
                return (MEMTIER_STDOUT_FULL, stderr)
            return (MEMTIER_STDOUT_FULL, "")

        server.run_host_command = AsyncMock(side_effect=capture)
        rg = make_repl_group(server)
        results = wire_file_protocol(runner)

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()

        cc = results[0].data["client_cpu"]
        assert cc["capacity_cores"] == 16  # min(24, cpulist 0-15)
        assert cc["utilization"] == pytest.approx(0.75, abs=0.01)
        assert cc["saturated"] is False

    @pytest.mark.asyncio
    async def test_no_gnu_time_graceful(self):
        runner = make_runner(repetitions=1)
        server = make_mock_server()
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        results = wire_file_protocol(runner)

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()

        cc = results[0].data["client_cpu"]
        assert cc["measurement_method"] == "unavailable"
        assert "utilization" not in cc


# =============================================================================
# Delayed perf stat / CPU profile measurement boundary tests
# =============================================================================


class TestDelayedPerfStatCommand:

    def _run_perf_stat(self, delay: float) -> list[str]:
        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 12345
        pm._main_tid = "12345"
        pm._io_tids = ["12346"] if delay == 0 else []
        calls = []
        mock_result = MagicMock(stdout="")

        def capture(*args, **kwargs):
            calls.append(args[0] if args else kwargs.get("command"))
            return mock_result

        with patch("conductress.profiling_manager.subprocess.run", side_effect=capture):
            pm._perf_stat_run_sync(delay_seconds=delay)
        return [c for c in calls if isinstance(c, str) and "perf stat" in c]

    def test_no_delay_no_sleep_prefix(self):
        cmds = self._run_perf_stat(0)
        assert len(cmds) == 1 and not cmds[0].startswith("sleep")

    def test_with_delay_has_sleep_prefix(self):
        cmds = self._run_perf_stat(5)
        assert len(cmds) == 1 and cmds[0].startswith("sleep 5 && perf stat")


class TestDelayedCpuProfileCommand:

    def test_no_delay_immediate(self):
        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 12345
        waits = []
        pm._cpu_profile_cancel_event.wait = lambda timeout=None: waits.append(timeout) or False
        mock_proc = MagicMock(wait=MagicMock(return_value=0))
        with patch("conductress.profiling_manager.subprocess.Popen", return_value=mock_proc):
            pm._cpu_profile_run_sync(duration=30, delay_seconds=0)
        assert len(waits) == 0

    def test_with_delay_waits_on_event(self):
        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 12345
        timeouts = []
        pm._cpu_profile_cancel_event.wait = lambda timeout=None: timeouts.append(timeout) or True
        with patch("conductress.profiling_manager.subprocess.Popen"):
            pm._cpu_profile_run_sync(duration=30, delay_seconds=5)
        assert timeouts == [5]


class TestMixedMeasurementBoundary:
    """perf_stat_start gets delay=warmup; result records duration-only."""

    @pytest.mark.asyncio
    async def _run_with_warmup(self, warmup: int) -> tuple[list[float], dict]:
        """Shared helper: run with given warmup, return (delays, result.data)."""
        runner = make_runner(repetitions=1, warmup=warmup, perf_stat_enabled=True)
        server = make_mock_server()
        delays: list[float] = []
        server.perf_stat_start = AsyncMock(side_effect=lambda delay_seconds=0: delays.append(delay_seconds))
        server.perf_stat_stop = AsyncMock()
        server.perf_stat_wait = MagicMock()
        server.perf_stat_report = AsyncMock(return_value={"all": {"instructions": 100}, "main": {}, "io": {}})
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        results = wire_file_protocol(runner)
        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()
        return delays, results[0].data

    @pytest.mark.asyncio
    async def test_perf_delay_equals_warmup(self):
        delays, data = await self._run_with_warmup(7)
        assert delays == [7.0]
        assert data["perf_duration_seconds"] == 30.0
        assert data["perf_warmup_included"] is False

    @pytest.mark.asyncio
    async def test_warmup_zero_no_delay(self):
        delays, data = await self._run_with_warmup(0)
        assert delays == [0.0]
        assert data["perf_duration_seconds"] == 30.0


class TestFinalRepCpuProfile:

    @pytest.mark.asyncio
    async def test_profile_only_on_final_rep(self):
        runner = make_runner(repetitions=3, perf_stat_enabled=True)
        runner._profile_internals = True
        server = make_mock_server()
        cpu_calls: list[dict] = []
        server.cpu_profile_start = MagicMock(
            side_effect=lambda d, delay_seconds=0: cpu_calls.append({"duration": d, "delay": delay_seconds})
        )
        server.cpu_profile_collect = AsyncMock(return_value=([["func1;func2", 100]], [["io_func1", 50]]))
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock()
        server.perf_stat_wait = MagicMock()
        server.perf_stat_report = AsyncMock(return_value={"all": {"instructions": 100}, "main": {}, "io": {}})
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        results = wire_file_protocol(runner)
        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()
        assert len(cpu_calls) == 1 and cpu_calls[0]["duration"] == 30 and cpu_calls[0]["delay"] == 5.0
        assert results[0].data["cpu_stacks_main"] == [["func1;func2", 100]]

    @pytest.mark.asyncio
    async def test_no_profile_without_perf_stat(self):
        runner = make_runner(repetitions=1, perf_stat_enabled=False)
        server = make_mock_server()
        server.cpu_profile_start = MagicMock()
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()
        server.cpu_profile_start.assert_not_called()


# =============================================================================
# Failure cleanup
# =============================================================================


class TestFailureCleanup:

    @pytest.mark.asyncio
    async def test_perf_stopped_on_memtier_failure(self):
        runner = make_runner(repetitions=1, perf_stat_enabled=True)
        server = make_mock_server()
        stop_called, wait_called = [], []
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock(side_effect=lambda: stop_called.append(True))
        server.perf_stat_wait = MagicMock(side_effect=lambda: wait_called.append(True))
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(fail_on_measure=True))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with pytest.raises(RuntimeError, match="Simulated memtier crash"):
            with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
                await runner.run()
        assert stop_called and wait_called

    @pytest.mark.asyncio
    async def test_cpu_profile_cancelled_on_failure(self):
        runner = make_runner(repetitions=1, perf_stat_enabled=True)
        runner._profile_internals = True
        server = make_mock_server()
        cancel_called = []
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock()
        server.perf_stat_wait = MagicMock()
        server.cpu_profile_start = MagicMock()
        server.cpu_profile_cancel = MagicMock(side_effect=lambda: cancel_called.append(True))
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(fail_on_measure=True))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with pytest.raises(RuntimeError, match="Simulated memtier crash"):
            with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
                await runner.run()
        assert cancel_called

    @pytest.mark.asyncio
    async def test_collectors_cleaned_on_rps_parse_failure(self):
        runner = make_runner(repetitions=1, perf_stat_enabled=True)
        runner._profile_internals = True
        server = make_mock_server()
        stop_called, wait_called, cancel_called = [], [], []
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock(side_effect=lambda: stop_called.append(True))
        server.perf_stat_wait = MagicMock(side_effect=lambda: wait_called.append(True))
        server.cpu_profile_start = MagicMock()
        server.cpu_profile_cancel = MagicMock(side_effect=lambda: cancel_called.append(True))
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(garbage_output=True))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with pytest.raises(RuntimeError, match="Failed to parse memtier output"):
            with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
                await runner.run()
        assert stop_called and wait_called and cancel_called

    @pytest.mark.asyncio
    async def test_no_double_stop_on_success(self):
        runner = make_runner(repetitions=1, perf_stat_enabled=True)
        server = make_mock_server()
        stop_count = []
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock(side_effect=lambda: stop_count.append(True))
        server.perf_stat_wait = MagicMock()
        server.perf_stat_report = AsyncMock(return_value={"all": {}, "main": {}, "io": {}})
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()
        assert len(stop_count) == 1


class TestWarmupZeroBehavior:

    @pytest.mark.asyncio
    async def test_warmup_zero_perf_starts_immediately(self):
        runner = make_runner(repetitions=1, warmup=0, perf_stat_enabled=True)
        server = make_mock_server()
        delays = []
        server.perf_stat_start = AsyncMock(side_effect=lambda delay_seconds=0: delays.append(delay_seconds))
        server.perf_stat_stop = AsyncMock()
        server.perf_stat_wait = MagicMock()
        server.perf_stat_report = AsyncMock(return_value={"all": {}, "main": {}, "io": {}})
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(gnu_time_available=False))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
            await runner.run()
        assert delays == [0.0]

    def test_warmup_zero_no_warmup_period_in_command(self):
        assert make_runner(warmup=0)._build_warmup_arg() == ""


# =============================================================================
# Cancellation and lifecycle correctness
# =============================================================================


class TestCpuProfileCancelApi:

    def test_cancel_noop_when_not_started(self):
        from conductress.profiling_manager import ProfilingManager

        ProfilingManager(MagicMock(ip="127.0.0.1")).cpu_profile_cancel()

    def test_cancel_during_delay_returns_promptly(self):
        import time as _time

        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 99999
        pm._main_tid = "99999"
        subprocess_calls = []
        with patch("conductress.profiling_manager.subprocess.Popen") as mock_popen:
            mock_popen.side_effect = lambda *a, **kw: subprocess_calls.append(a)
            pm.cpu_profile_start(duration=30, delay_seconds=60)
            _time.sleep(0.1)
            start = _time.monotonic()
            pm.cpu_profile_cancel()
            elapsed = _time.monotonic() - start
        assert elapsed < 2.0
        assert len(subprocess_calls) == 0

    def test_cancel_after_perf_started_terminates_process(self):
        import time as _time
        from threading import Event

        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 99999
        pm._main_tid = "99999"
        proc_wait_event = Event()
        mock_proc = MagicMock()
        mock_proc.wait = MagicMock(side_effect=lambda *a, **kw: proc_wait_event.wait(timeout=10))
        mock_proc.terminate = MagicMock(side_effect=lambda: proc_wait_event.set())
        with patch("conductress.profiling_manager.subprocess.Popen", return_value=mock_proc):
            pm.cpu_profile_start(duration=300, delay_seconds=0)
            _time.sleep(0.2)
            pm.cpu_profile_cancel()
        mock_proc.terminate.assert_called_once()

    def test_cancel_race_during_process_launch(self):
        from conductress.profiling_manager import ProfilingManager

        pm = ProfilingManager(MagicMock(ip="127.0.0.1"))
        pm._target_pid = 99999
        pm._main_tid = "99999"
        mock_proc = MagicMock(wait=MagicMock(return_value=0))

        def launch_and_cancel(*args, **kwargs):
            pm._cpu_profile_cancel_event.set()
            return mock_proc

        with patch("conductress.profiling_manager.subprocess.Popen", side_effect=launch_and_cancel):
            pm._cpu_profile_run_sync(duration=30, delay_seconds=0)
        mock_proc.terminate.assert_called_once()


class TestServerDelegation:

    def test_cpu_profile_cancel_delegates(self):
        from conductress.profiling_manager import ProfilingManager
        from conductress.server import Server

        pm = MagicMock(spec=ProfilingManager)
        s = Server.__new__(Server)
        s._profiling = pm
        s.cpu_profile_cancel()
        pm.cpu_profile_cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_perf_stat_start_passes_delay(self):
        from conductress.profiling_manager import ProfilingManager
        from conductress.server import Server

        pm = MagicMock(spec=ProfilingManager)
        pm.perf_stat_start = AsyncMock()
        s = Server.__new__(Server)
        s._profiling = pm
        s.valkey_pid = 12345
        await s.perf_stat_start(delay_seconds=7.5)
        pm.perf_stat_start.assert_awaited_once_with(delay_seconds=7.5)
        assert pm.target_pid == 12345

    def test_cpu_profile_start_passes_delay(self):
        from conductress.profiling_manager import ProfilingManager
        from conductress.server import Server

        pm = MagicMock(spec=ProfilingManager)
        s = Server.__new__(Server)
        s._profiling = pm
        s.valkey_pid = 12345
        s.cpu_profile_start(30, delay_seconds=5.0)
        pm.cpu_profile_start.assert_called_once_with(30, delay_seconds=5.0)
        assert pm.target_pid == 12345


class TestFailureDuringDelayedWarmup:

    @pytest.mark.asyncio
    async def test_cancel_prompt_on_delayed_warmup_failure(self):
        runner = make_runner(repetitions=1, warmup=60, perf_stat_enabled=True)
        runner._profile_internals = True
        server = make_mock_server()
        server.perf_stat_start = AsyncMock()
        server.perf_stat_stop = AsyncMock()
        server.perf_stat_wait = MagicMock()
        server.cpu_profile_start = MagicMock()
        server.cpu_profile_cancel = MagicMock()
        server.run_host_command = AsyncMock(side_effect=gnu_time_cmd_router(fail_on_measure=True))
        rg = make_repl_group(server)
        wire_file_protocol(runner)
        with pytest.raises(RuntimeError, match="Simulated memtier crash"):
            with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=rg):
                await runner.run()
        server.cpu_profile_cancel.assert_called_once()


# =============================================================================
# CLI add-mixed tests
# =============================================================================


class TestCliAddMixed:

    @pytest.fixture(autouse=True)
    def isolate_queue(self, tmp_path):
        queue_path = tmp_path / "queue"
        queue_path.mkdir()
        _Orig = TaskQueue

        class _Isolated(_Orig):
            def __init__(self, queue_dir_override=None):
                super().__init__(queue_dir=queue_path)

        with patch("conductress.cli.TaskQueue", _Isolated):
            self.queue_path = queue_path
            yield

    def _tasks_json(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in self.queue_path.glob("task_*.json")]

    def _run(self, *extra_args) -> int:
        base = ["queue", "add-mixed", "--source", "valkey", "--specifier", "unstable", "--set-ratio", "20"]
        return main(list(base) + list(extra_args))

    def test_basic_add_mixed(self):
        assert self._run("--sizes", "512", "--io-threads", "9", "--pipelining", "10") == 0
        tasks = self._tasks_json()
        assert len(tasks) == 1
        d = tasks[0]
        assert d["task_type"] == "MixedTaskData"
        assert d["set_ratio"] == 20 and d["memtier_threads"] == 0 and d["memtier_clients"] == 0

    @pytest.mark.parametrize(("value", "expected"), [("12s", 12), ("0s", 0)])
    def test_warmup_serialized(self, value, expected):
        assert self._run("--warmup", value) == 0
        assert self._tasks_json()[0]["warmup"] == expected

    def test_nonzero_key_size_rejected(self, capsys):
        assert self._run("--key-sizes", "32") == 1
        assert "--key-sizes is not supported" in capsys.readouterr().err

    def test_invalid_ratio_rejected(self, capsys):
        assert main(["queue", "add-mixed", "--source", "valkey", "--specifier", "unstable", "--set-ratio", "150"]) == 1
        assert "set-ratio must be 0-100" in capsys.readouterr().err

    def test_cartesian_product(self):
        assert (
            main(
                [
                    "queue",
                    "add-mixed",
                    "--source",
                    "valkey",
                    "--specifier",
                    "unstable",
                    "--set-ratio",
                    "50",
                    "--sizes",
                    "16,512",
                    "--io-threads",
                    "7,9",
                    "--pipelining",
                    "10",
                ]
            )
            == 0
        )
        assert len(self._tasks_json()) == 4

    def test_invalid_source_rejected(self, capsys):
        assert (
            main(["queue", "add-mixed", "--source", "nosuchrepo", "--specifier", "unstable", "--set-ratio", "20"]) == 1
        )
        assert "Invalid source" in capsys.readouterr().err

    def test_perf_stat_flag(self):
        assert self._run("--perf-stat") == 0
        assert self._tasks_json()[0]["perf_stat_enabled"] is True

    def test_note_stored(self):
        assert self._run("--note", "regression check") == 0
        assert self._tasks_json()[0]["note"] == "regression check"

    def test_memtier_threads_override(self):
        assert self._run("--memtier-threads", "24") == 0
        d = self._tasks_json()[0]
        assert d["memtier_threads"] == 24 and d["memtier_clients"] == 0

    def test_memtier_clients_override(self):
        assert self._run("--memtier-clients", "100") == 0
        d = self._tasks_json()[0]
        assert d["memtier_clients"] == 100 and d["memtier_threads"] == 0

    def test_1200c_configuration(self):
        assert self._run("--memtier-threads", "24", "--memtier-clients", "50") == 0
        d = self._tasks_json()[0]
        assert d["memtier_threads"] == 24 and d["memtier_clients"] == 50

    def test_2400c_configuration(self):
        assert self._run("--memtier-threads", "24", "--memtier-clients", "100") == 0
        d = self._tasks_json()[0]
        assert d["memtier_threads"] == 24 and d["memtier_clients"] == 100

    @pytest.mark.parametrize(
        ("flag", "val", "match"),
        [
            ("--memtier-threads", "-1", "memtier_threads must be >= 0"),
            ("--memtier-clients", "-1", "memtier_clients must be >= 0"),
            ("--memtier-threads", str(MAX_MEMTIER_THREADS + 1), f"memtier_threads must be <= {MAX_MEMTIER_THREADS}"),
            ("--memtier-clients", str(MAX_MEMTIER_CLIENTS + 1), f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}"),
        ],
    )
    def test_bounds_rejected(self, flag, val, match, capsys):
        assert self._run(flag, val) == 1
        assert match in capsys.readouterr().err
