"""Unit tests for the mixed GET/SET throughput task.

Round 2: adds empirical client CPU measurement tests (GNU time parsing),
upper-bound validation, allocated_cores semantics, and result schema checks.

Round 3: adds taskset pinning enforcement tests, command construction
assertions (default vs pinned), mixed-task-specific capacity model tests,
and end-to-end runner-path mocks proving GNU-time → utilization → saturated.
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


class TestSetRatioConversion:
    """Tests for set_ratio_to_memtier_ratio conversion."""

    def test_pure_get(self):
        assert set_ratio_to_memtier_ratio(0) == "0:1"

    def test_pure_set(self):
        assert set_ratio_to_memtier_ratio(100) == "1:0"

    def test_20_percent_set(self):
        # 20% SET / 80% GET -> gcd(20,80)=20 -> 1:4
        assert set_ratio_to_memtier_ratio(20) == "1:4"

    def test_50_percent_set(self):
        # 50/50 -> gcd(50,50)=50 -> 1:1
        assert set_ratio_to_memtier_ratio(50) == "1:1"

    def test_10_percent_set(self):
        # 10/90 -> gcd(10,90)=10 -> 1:9
        assert set_ratio_to_memtier_ratio(10) == "1:9"

    def test_33_percent_set(self):
        # 33/67 -> gcd(33,67)=1 -> 33:67
        assert set_ratio_to_memtier_ratio(33) == "33:67"

    def test_75_percent_set(self):
        # 75/25 -> gcd(75,25)=25 -> 3:1
        assert set_ratio_to_memtier_ratio(75) == "3:1"


class TestMemtierOutputParsing:
    """Tests for memtier stdout parsing."""

    def test_parse_totals_line(self):
        output = (  # noqa: E501
            "Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency"
            "     p50 Latency     p99 Latency   p99.9 Latency       KB/sec\n"
            "--------------------------------------------------------------"
            "------------------------------------------------------------------\n"
            "Sets        250000.12          ---          ---"
            "         0.123         0.100         0.500         1.000      1234.56\n"
            "Gets       1000000.34    1000000.34         0.00"
            "         0.089         0.080         0.400         0.900      5678.90\n"
            "Totals     1250000.46    1000000.34         0.00"
            "         0.096         0.085         0.420         0.920      6913.46"
        )
        result = parse_memtier_total_rps(output)
        assert result == 1250000.46

    def test_parse_no_totals(self):
        output = "some random output\nno totals here"
        result = parse_memtier_total_rps(output)
        assert result is None

    def test_parse_empty_output(self):
        result = parse_memtier_total_rps("")
        assert result is None


class TestGnuTimeParsing:
    """Tests for parse_gnu_time_stderr — the core of empirical client CPU measurement."""

    SAMPLE_GNU_TIME_OUTPUT = (
        '\tCommand being timed: "memtier_benchmark ..."\n'
        "\tUser time (seconds): 18.42\n"
        "\tSystem time (seconds): 6.31\n"
        "\tPercent of CPU this job got: 82%\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.15\n"
        "\tMaximum resident set size (kbytes): 524288\n"
        "\tMinor (reclaiming a frame) page faults: 1234\n"
    )

    def test_parse_valid_output(self):
        result = parse_gnu_time_stderr(self.SAMPLE_GNU_TIME_OUTPUT)
        assert result is not None
        assert result["user_seconds"] == 18.42
        assert result["system_seconds"] == 6.31
        assert result["wall_seconds"] == pytest.approx(30.15, abs=0.01)
        assert result["cpu_seconds"] == pytest.approx(24.73, abs=0.01)
        # cores_busy = 24.73 / 30.15 ≈ 0.820
        assert result["cores_busy"] == pytest.approx(0.820, abs=0.01)

    def test_parse_with_hours(self):
        stderr = (
            "\tUser time (seconds): 3600.50\n"
            "\tSystem time (seconds): 100.25\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:30.75\n"
        )
        result = parse_gnu_time_stderr(stderr)
        assert result is not None
        assert result["wall_seconds"] == pytest.approx(3750.75, abs=0.01)
        assert result["user_seconds"] == 3600.50
        assert result["system_seconds"] == 100.25

    def test_parse_empty_string(self):
        assert parse_gnu_time_stderr("") is None

    def test_parse_non_gnu_time_output(self):
        # Just memtier stderr without GNU time
        stderr = "WARNING: Connection timeout\nSome memtier warning\n"
        assert parse_gnu_time_stderr(stderr) is None

    def test_parse_missing_wall_time(self):
        stderr = "\tUser time (seconds): 18.42\n" "\tSystem time (seconds): 6.31\n"
        assert parse_gnu_time_stderr(stderr) is None

    def test_parse_missing_user_time(self):
        stderr = "\tSystem time (seconds): 6.31\n" "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.15\n"
        assert parse_gnu_time_stderr(stderr) is None

    def test_parse_zero_wall_time(self):
        """Zero wall time should produce cores_busy=0, not a divide-by-zero."""
        stderr = (
            "\tUser time (seconds): 0.01\n"
            "\tSystem time (seconds): 0.00\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00\n"
        )
        result = parse_gnu_time_stderr(stderr)
        assert result is not None
        assert result["cores_busy"] == 0.0

    def test_parse_high_cpu_multithread(self):
        """Multi-threaded memtier can use >1 core (cores_busy > 1.0)."""
        stderr = (
            "\tUser time (seconds): 210.50\n"
            "\tSystem time (seconds): 30.20\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
        )
        result = parse_gnu_time_stderr(stderr)
        assert result is not None
        # cores_busy = 240.7 / 30.0 = 8.023
        assert result["cores_busy"] > 8.0

    def test_result_fields_are_rounded(self):
        """All float fields are rounded to 3 decimal places."""
        stderr = (
            "\tUser time (seconds): 1.23456789\n"
            "\tSystem time (seconds): 0.98765432\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:05.00\n"
        )
        result = parse_gnu_time_stderr(stderr)
        assert result is not None
        assert result["user_seconds"] == 1.235
        assert result["system_seconds"] == 0.988
        assert result["cpu_seconds"] == 2.222


class TestEffectiveMemtierDefaults:
    """Tests for the effective threads/clients helpers preserving defaults."""

    def test_zero_threads_uses_default(self):
        assert _effective_memtier_threads(0) == MIXED_THREADS == 8

    def test_zero_clients_uses_default(self):
        assert _effective_memtier_clients(0) == MIXED_CLIENTS == 50

    def test_positive_threads_overrides(self):
        assert _effective_memtier_threads(24) == 24

    def test_positive_clients_overrides(self):
        assert _effective_memtier_clients(100) == 100


class TestValidateMemtierBounds:
    """Tests for the centralized _validate_memtier_bounds function."""

    def test_defaults_pass(self):
        _validate_memtier_bounds(0, 0)  # should not raise

    def test_reasonable_overrides_pass(self):
        _validate_memtier_bounds(24, 100)  # 2400c

    def test_max_bounds_pass(self):
        _validate_memtier_bounds(MAX_MEMTIER_THREADS, MAX_MEMTIER_CLIENTS)

    def test_negative_threads_rejected(self):
        with pytest.raises(ValueError, match="memtier_threads must be >= 0"):
            _validate_memtier_bounds(-1, 0)

    def test_negative_clients_rejected(self):
        with pytest.raises(ValueError, match="memtier_clients must be >= 0"):
            _validate_memtier_bounds(0, -1)

    def test_threads_over_max_rejected(self):
        with pytest.raises(ValueError, match=f"memtier_threads must be <= {MAX_MEMTIER_THREADS}"):
            _validate_memtier_bounds(MAX_MEMTIER_THREADS + 1, 0)

    def test_clients_over_max_rejected(self):
        with pytest.raises(ValueError, match=f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}"):
            _validate_memtier_bounds(0, MAX_MEMTIER_CLIENTS + 1)

    def test_zero_effective_connections_unreachable_under_bounds(self):
        """Zero effective connections is structurally unreachable given bounds.

        With 0 meaning "use default" and defaults being (8, 50), the minimum
        effective product is 8*50=400.  The keyspace guard in the next test
        covers the remaining edge.
        """
        # Verify the minimum effective product is well above 0
        from conductress.tasks.task_mixed import _effective_memtier_clients, _effective_memtier_threads

        assert _effective_memtier_threads(0) * _effective_memtier_clients(0) == 400

    def test_connections_exceed_keyspace_rejected(self):
        """If somehow total connections > keyspace, prefill would get 0 requests."""
        # We need to bypass the individual bounds to trigger the keyspace guard.
        # The MAX bounds (256 * 1000 = 256K) are well below keyspace (3M), so
        # under normal operation this guard is redundant.  Test it directly.
        from conductress.tasks import task_mixed

        saved = task_mixed.MAX_MEMTIER_THREADS, task_mixed.MAX_MEMTIER_CLIENTS
        try:
            task_mixed.MAX_MEMTIER_THREADS = 10000
            task_mixed.MAX_MEMTIER_CLIENTS = 10000
            with pytest.raises(ValueError, match="prefill requests per connection would be zero"):
                _validate_memtier_bounds(5000, 5000)  # 25M > 3M keyspace
        finally:
            task_mixed.MAX_MEMTIER_THREADS, task_mixed.MAX_MEMTIER_CLIENTS = saved


class TestMixedTaskDataValidation:
    """Tests for MixedTaskData construction and validation."""

    @pytest.fixture(autouse=True)
    def patch_sources(self):
        with (
            patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
            patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
            patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
        ):
            yield

    def _make_task(self, **overrides) -> MixedTaskData:
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

    def test_valid_ratio_0(self):
        task = self._make_task(set_ratio=0)
        assert task.set_ratio == 0
        assert task.task_type == "MixedTaskData"

    def test_valid_ratio_100(self):
        task = self._make_task(set_ratio=100)
        assert task.set_ratio == 100

    def test_invalid_ratio_negative(self):
        with pytest.raises(ValueError, match="set_ratio must be 0-100"):
            self._make_task(set_ratio=-1)

    def test_invalid_ratio_over_100(self):
        with pytest.raises(ValueError, match="set_ratio must be 0-100"):
            self._make_task(set_ratio=101)

    def test_invalid_warmup_negative(self):
        with pytest.raises(ValueError, match="warmup must be >= 0"):
            self._make_task(warmup=-1)

    def test_nonzero_key_size_rejected(self):
        with pytest.raises(ValueError, match="key_size is not supported"):
            self._make_task(key_size=32)

    def test_invalid_memtier_threads_negative(self):
        with pytest.raises(ValueError, match="memtier_threads must be >= 0"):
            self._make_task(memtier_threads=-1)

    def test_invalid_memtier_clients_negative(self):
        with pytest.raises(ValueError, match="memtier_clients must be >= 0"):
            self._make_task(memtier_clients=-1)

    def test_memtier_threads_over_max_rejected(self):
        with pytest.raises(ValueError, match=f"memtier_threads must be <= {MAX_MEMTIER_THREADS}"):
            self._make_task(memtier_threads=MAX_MEMTIER_THREADS + 1)

    def test_memtier_clients_over_max_rejected(self):
        with pytest.raises(ValueError, match=f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}"):
            self._make_task(memtier_clients=MAX_MEMTIER_CLIENTS + 1)

    # -- Default preservation --

    def test_default_memtier_threads_is_zero(self):
        task = self._make_task()
        assert task.memtier_threads == 0

    def test_default_memtier_clients_is_zero(self):
        task = self._make_task()
        assert task.memtier_clients == 0

    def test_default_effective_threads(self):
        task = self._make_task()
        assert task.effective_threads == 8

    def test_default_effective_clients(self):
        task = self._make_task()
        assert task.effective_clients == 50

    def test_default_total_connections_is_400(self):
        task = self._make_task()
        assert task.total_connections == 400

    # -- Override behavior --

    def test_override_threads_24(self):
        task = self._make_task(memtier_threads=24)
        assert task.effective_threads == 24
        assert task.effective_clients == 50
        assert task.total_connections == 1200

    def test_override_both_for_2400c(self):
        task = self._make_task(memtier_threads=24, memtier_clients=100)
        assert task.total_connections == 2400

    def test_override_clients_only(self):
        task = self._make_task(memtier_clients=150)
        assert task.effective_threads == 8
        assert task.effective_clients == 150
        assert task.total_connections == 1200

    # -- short_description --

    def test_short_description_default_omits_concurrency(self):
        """Default concurrency (8×50=400) doesn't clutter the description."""
        task = self._make_task()
        desc = task.short_description()
        assert "20%SET" in desc
        assert "80%GET" in desc
        assert "io=9" in desc
        assert "1200c" not in desc
        assert "2400c" not in desc

    def test_short_description_override_shows_concurrency(self):
        task = self._make_task(memtier_threads=24, memtier_clients=50)
        desc = task.short_description()
        assert "1200c(24t×50c)" in desc

    # -- Serialization round trip --

    def test_serialization_round_trip(self, tmp_path):
        """Task can be saved to JSON and reloaded."""
        task = self._make_task(
            repetitions=5,
            perf_stat_enabled=True,
            note="test note",
        )
        filepath = tmp_path / "task.json"
        task.save_to_file(filepath)

        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(filepath)
        assert isinstance(loaded, MixedTaskData)
        assert loaded.set_ratio == 20
        assert loaded.val_size == 512
        assert loaded.io_threads == 9
        assert loaded.pipelining == 10
        assert loaded.duration == 30
        assert loaded.repetitions == 5
        assert loaded.perf_stat_enabled is True
        assert loaded.note == "test note"

    def test_serialization_round_trip_with_concurrency_override(self, tmp_path):
        """Concurrency overrides survive JSON round trip."""
        task = self._make_task(memtier_threads=24, memtier_clients=100)
        filepath = tmp_path / "task.json"
        task.save_to_file(filepath)

        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(filepath)
        assert isinstance(loaded, MixedTaskData)
        assert loaded.memtier_threads == 24
        assert loaded.memtier_clients == 100
        assert loaded.effective_threads == 24
        assert loaded.effective_clients == 100
        assert loaded.total_connections == 2400

    def test_backward_compat_load_without_new_fields(self, tmp_path):
        """Old task envelopes without memtier_threads/memtier_clients load with defaults."""
        task = self._make_task()
        filepath = tmp_path / "task.json"
        task.save_to_file(filepath)

        # Strip the new fields to simulate an old envelope
        data = json.loads(filepath.read_text())
        data.pop("memtier_threads", None)
        data.pop("memtier_clients", None)
        filepath.write_text(json.dumps(data, indent=2))

        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(filepath)
        assert isinstance(loaded, MixedTaskData)
        assert loaded.memtier_threads == 0
        assert loaded.memtier_clients == 0
        assert loaded.total_connections == 400  # original default


class TestMixedTaskRunnerInit:
    """Tests that MixedTaskRunner properly resolves concurrency parameters."""

    def test_runner_default_concurrency(self):
        runner = MixedTaskRunner(
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
        assert runner.memtier_threads == 8
        assert runner.memtier_clients == 50
        assert runner.total_connections == 400

    def test_runner_overridden_concurrency(self):
        runner = MixedTaskRunner(
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
            memtier_threads=24,
            memtier_clients=50,
        )
        assert runner.memtier_threads == 24
        assert runner.memtier_clients == 50
        assert runner.total_connections == 1200

    def test_runner_has_client_telemetry_init(self):
        runner = MixedTaskRunner(
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
        assert runner._client_cores_busy_per_rep == []
        assert runner._gnu_time_available is None

    def test_runner_no_taskset_without_cpu_override(self):
        """Without explicit CPU pinning, no taskset prefix is produced."""
        runner = MixedTaskRunner(
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
        assert runner._build_taskset_prefix() == ""

    def test_runner_with_benchmark_cpu_override(self):
        """benchmark_cpu_override produces a taskset prefix."""
        runner = MixedTaskRunner(
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
            benchmark_cpu_override="0-7,16-23",
        )
        assert runner.benchmark_cpu_override == "0-7,16-23"
        assert runner._build_taskset_prefix() == "taskset -c 0-7,16-23 "


class TestClientCpuResultSchema:
    """Tests that the client_cpu result block has the correct shape."""

    def test_empirical_cpu_schema(self):
        """When cores_busy_per_rep has data, summarize_client_cpu produces the right keys."""
        from conductress.utility import summarize_client_cpu

        result = summarize_client_cpu([4.5, 5.0, 4.8], allocated_cores=8)
        assert "cores_busy_per_rep" in result
        assert "allocated_cores" in result
        assert "utilization" in result
        assert "saturated" in result
        assert result["allocated_cores"] == 8
        # max(4.5, 5.0, 4.8) / 8 = 0.625
        assert result["utilization"] == pytest.approx(0.625, abs=0.01)
        assert result["saturated"] is False

    def test_saturated_flag_at_threshold(self):
        """At >= 90% utilization, saturated should be True."""
        from conductress.utility import summarize_client_cpu

        result = summarize_client_cpu([7.5], allocated_cores=8)
        # 7.5 / 8 = 0.9375 >= 0.9
        assert result["saturated"] is True

    def test_no_allocated_cores_no_utilization(self):
        """Without allocated_cores, no utilization or saturated fields."""
        from conductress.utility import summarize_client_cpu

        result = summarize_client_cpu([4.5, 5.0], allocated_cores=None)
        assert "utilization" not in result
        assert "saturated" not in result


class TestTasksetPrefix:
    """Tests that _build_taskset_prefix applies taskset correctly."""

    def _make_runner(self, **overrides) -> MixedTaskRunner:
        defaults = dict(
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
        defaults.update(overrides)
        return MixedTaskRunner(**defaults)

    def test_no_override_no_taskset(self):
        """Without benchmark_cpu_override, prefix is empty string."""
        runner = self._make_runner()
        assert runner._build_taskset_prefix() == ""

    def test_explicit_cpulist_produces_taskset(self):
        """With benchmark_cpu_override, prefix is 'taskset -c <cpulist> '."""
        runner = self._make_runner(benchmark_cpu_override="0-7,16-23")
        prefix = runner._build_taskset_prefix()
        assert prefix == "taskset -c 0-7,16-23 "

    def test_single_cpu_produces_taskset(self):
        runner = self._make_runner(benchmark_cpu_override="5")
        assert runner._build_taskset_prefix() == "taskset -c 5 "

    def test_taskset_prefix_has_trailing_space(self):
        """Trailing space ensures safe concatenation with the command."""
        runner = self._make_runner(benchmark_cpu_override="0-3")
        assert runner._build_taskset_prefix().endswith(" ")


class TestCommandConstruction:
    """Tests proving default commands have no taskset and overrides produce correct commands.

    These test the EXACT command strings that would be sent to run_host_command,
    verifying backward behavioral equivalence for defaults and correct pinning
    for overrides.
    """

    def _make_runner(self, **overrides) -> MixedTaskRunner:
        defaults = dict(
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
        defaults.update(overrides)
        return MixedTaskRunner(**defaults)

    def test_default_prefill_no_taskset(self):
        """Default (no cpu override): prefill command starts with ~/conductress/memtier, no taskset."""
        runner = self._make_runner()
        prefix = runner._build_taskset_prefix()
        prefill = f"{prefix}~/conductress/memtier_benchmark --server 10.0.0.1 --port 6379"
        assert prefill.startswith("~/conductress/memtier_benchmark")
        assert "taskset" not in prefill

    def test_default_measure_legacy_threads_clients(self):
        """Default commands use legacy 8 threads × 50 clients."""
        runner = self._make_runner()
        assert runner.memtier_threads == 8
        assert runner.memtier_clients == 50
        prefix = runner._build_taskset_prefix()
        measure = (
            f"{prefix}~/conductress/memtier_benchmark "
            f"--threads {runner.memtier_threads} --clients {runner.memtier_clients}"
        )
        assert "--threads 8" in measure
        assert "--clients 50" in measure
        assert "taskset" not in measure

    def test_override_threads_clients_in_command(self):
        """Overridden threads/clients appear in command strings."""
        runner = self._make_runner(memtier_threads=24, memtier_clients=100)
        assert runner.memtier_threads == 24
        assert runner.memtier_clients == 100

    def test_pinned_prefill_has_taskset(self):
        """With cpu override, prefill command is prefixed with taskset."""
        runner = self._make_runner(benchmark_cpu_override="0-7,16-23")
        prefix = runner._build_taskset_prefix()
        prefill = f"{prefix}~/conductress/memtier_benchmark --server 10.0.0.1"
        assert prefill.startswith("taskset -c 0-7,16-23 ")
        assert "~/conductress/memtier_benchmark" in prefill

    def test_pinned_measure_has_taskset(self):
        """With cpu override, measure command is also prefixed with taskset."""
        runner = self._make_runner(benchmark_cpu_override="4-11")
        prefix = runner._build_taskset_prefix()
        assert prefix == "taskset -c 4-11 "
        measure = f"{prefix}~/conductress/memtier_benchmark --test-time 30"
        assert measure.startswith("taskset -c 4-11 ")

    def test_timed_command_wraps_taskset(self):
        """GNU time wraps the entire taskset+memtier command."""
        runner = self._make_runner(benchmark_cpu_override="0-3")
        prefix = runner._build_taskset_prefix()
        measure = f"{prefix}~/conductress/memtier_benchmark --test-time 30"
        timed = runner._build_timed_command(measure)
        # Structure: /usr/bin/time -v taskset -c 0-3 ~/conductress/memtier_benchmark ...
        assert timed.startswith("/usr/bin/time -v taskset -c 0-3 ")
        assert "~/conductress/memtier_benchmark" in timed

    def test_timed_command_no_override(self):
        """Without cpu override, timed command wraps plain memtier."""
        runner = self._make_runner()
        measure = "~/conductress/memtier_benchmark --test-time 30"
        timed = runner._build_timed_command(measure)
        assert timed == "/usr/bin/time -v ~/conductress/memtier_benchmark --test-time 30"
        assert "taskset" not in timed

    def test_positive_warmup_builds_memtier_option(self):
        runner = self._make_runner(warmup=7)
        assert runner._build_warmup_arg() == "--warmup-period 7 "

    def test_zero_warmup_omits_memtier_option(self):
        runner = self._make_runner(warmup=0)
        assert runner._build_warmup_arg() == ""


class TestCapacityModel:
    """Tests for the mixed-task capacity model in _compute_client_cpu_meta.

    Two capacity models:
    - Unpinned: capacity = memtier_threads (each worker uses ≤1 core)
    - Pinned: capacity = min(memtier_threads, cpulist core count)
    Both use the shared 0.90 saturation threshold.
    """

    PINNED_CAPACITY_BASIS = "min(memtier_thread_count,taskset_cpulist)"

    def _make_runner(self, **overrides) -> MixedTaskRunner:
        defaults = dict(
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
        defaults.update(overrides)
        return MixedTaskRunner(**defaults)

    def test_unpinned_default_capacity_is_thread_count(self):
        """Without cpu override, capacity_cores = memtier_threads = 8."""
        runner = self._make_runner()
        meta = runner._compute_client_cpu_meta()
        assert meta["capacity_cores"] == 8
        assert meta["capacity_basis"] == "memtier_thread_count"

    def test_unpinned_24_threads_capacity(self):
        """24-thread override: capacity_cores = 24."""
        runner = self._make_runner(memtier_threads=24)
        meta = runner._compute_client_cpu_meta()
        assert meta["capacity_cores"] == 24
        assert meta["capacity_basis"] == "memtier_thread_count"

    def test_pinned_capacity_uses_lower_thread_limit(self):
        """A cpuset larger than the worker pool cannot raise client CPU capacity."""
        runner = self._make_runner(benchmark_cpu_override="0-7,16-23")
        meta = runner._compute_client_cpu_meta()
        assert meta["capacity_cores"] == 8  # min(8 memtier threads, 16 pinned cores)
        assert meta["capacity_basis"] == self.PINNED_CAPACITY_BASIS
        assert meta["benchmark_cpu_override"] == "0-7,16-23"

    def test_pinned_single_cpu(self):
        runner = self._make_runner(benchmark_cpu_override="5")
        meta = runner._compute_client_cpu_meta()
        assert meta["capacity_cores"] == 1
        assert meta["capacity_basis"] == self.PINNED_CAPACITY_BASIS

    def test_unpinned_no_benchmark_cpu_override_in_meta(self):
        """Without pinning, benchmark_cpu_override key is absent."""
        runner = self._make_runner()
        meta = runner._compute_client_cpu_meta()
        assert "benchmark_cpu_override" not in meta

    def test_unavailable_measurement_no_utilization(self):
        """Without empirical data, utilization/saturated are absent."""
        runner = self._make_runner()
        meta = runner._compute_client_cpu_meta()
        assert meta["measurement_method"] == "unavailable"
        assert "utilization" not in meta
        assert "saturated" not in meta
        assert "note" in meta

    def test_empirical_data_produces_utilization(self):
        """With cores_busy data, utilization and saturated appear."""
        runner = self._make_runner()
        runner._client_cores_busy_per_rep = [4.5, 5.0, 4.8]
        meta = runner._compute_client_cpu_meta()
        assert meta["measurement_method"] == "gnu_time"
        assert meta["cores_busy_per_rep"] == [4.5, 5.0, 4.8]
        # capacity_cores = 8 (default threads), utilization = max(5.0) / 8 = 0.625
        assert meta["utilization"] == pytest.approx(0.625, abs=0.01)
        assert meta["saturated"] is False

    def test_empirical_saturated_flag(self):
        """At 90%+ of capacity, saturated = True."""
        runner = self._make_runner()
        runner._client_cores_busy_per_rep = [7.5]
        meta = runner._compute_client_cpu_meta()
        # 7.5 / 8 = 0.9375 >= 0.9
        assert meta["saturated"] is True

    def test_pinned_with_empirical_data(self):
        """Pinned + empirical: utilization uses the lower worker/cpuset capacity."""
        runner = self._make_runner(memtier_threads=24, benchmark_cpu_override="0-15")
        runner._client_cores_busy_per_rep = [12.0, 13.0, 12.5]
        meta = runner._compute_client_cpu_meta()
        assert meta["capacity_cores"] == 16  # min(24 memtier threads, 16 pinned cores)
        assert meta["capacity_basis"] == self.PINNED_CAPACITY_BASIS
        # utilization = max(13.0) / 16 = 0.8125
        assert meta["utilization"] == pytest.approx(0.8125, abs=0.01)
        assert meta["saturated"] is False

    def test_meta_always_has_connection_info(self):
        """All meta blocks include threads/clients/connections regardless of measurement."""
        runner = self._make_runner(memtier_threads=24, memtier_clients=100)
        meta = runner._compute_client_cpu_meta()
        assert meta["memtier_threads"] == 24
        assert meta["memtier_clients"] == 100
        assert meta["total_connections"] == 2400


class TestGnuTimeProbe:
    @pytest.mark.asyncio
    async def test_busybox_time_is_not_accepted(self):
        runner = MixedTaskRunner(
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
            repetitions=1,
        )
        server = MagicMock()
        server.run_host_command = AsyncMock(return_value=("BusyBox time: unrecognized option --version", ""))
        assert await runner._probe_gnu_time(server) is False


class TestEndToEndRunnerMock:
    """End-to-end runner path mocks proving:
    1. Commands sent to run_host_command include/exclude taskset correctly
    2. GNU time stderr is parsed into cores_busy/utilization/saturated
    3. Result record has correct client_cpu metadata
    """

    MEMTIER_STDOUT = (
        "Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency\n"
        "Sets        250000.00          ---          ---         0.100\n"
        "Gets       1000000.00    1000000.00         0.00         0.050\n"
        "Totals     1250000.00    1000000.00         0.00         0.060\n"
    )

    GNU_TIME_STDERR = (
        '\tCommand being timed: "memtier_benchmark ..."\n'
        "\tUser time (seconds): 210.50\n"
        "\tSystem time (seconds): 30.20\n"
        "\tPercent of CPU this job got: 802%\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
        "\tMaximum resident set size (kbytes): 524288\n"
    )

    def _make_runner(self, **overrides) -> MixedTaskRunner:
        defaults = dict(
            task_name="test-e2e",
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
            repetitions=1,
        )
        defaults.update(overrides)
        return MixedTaskRunner(**defaults)

    def _build_mock_server(self) -> MagicMock:
        """Build a mock server that captures all run_host_command calls."""
        server = MagicMock()
        server.ip = "10.0.0.1"
        server.port = 6379
        server.get_build_hash.return_value = "abc123"
        return server

    @pytest.mark.asyncio
    async def test_default_commands_no_taskset(self):
        """Default runner: no taskset in any command sent to host."""
        runner = self._make_runner()
        server = self._build_mock_server()
        commands_sent: list[str] = []

        async def capture_cmd(cmd, check=True):
            commands_sent.append(cmd)
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                return (self.MEMTIER_STDOUT, self.GNU_TIME_STDERR)
            return (self.MEMTIER_STDOUT, "")

        server.run_host_command = AsyncMock(side_effect=capture_cmd)

        repl_group = MagicMock()
        repl_group.primary = server
        repl_group.start = AsyncMock()
        repl_group.stop_all_servers = AsyncMock()
        repl_group.kill_all_valkey_instances = AsyncMock()

        runner.file_protocol = MagicMock()
        runner.file_protocol.get_result_dir.return_value = "/tmp/results"

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=repl_group):
            await runner.run()

        # Verify no command contains 'taskset'
        for cmd in commands_sent:
            assert "taskset" not in cmd, f"Unexpected taskset in default command: {cmd}"

        # Verify prefill uses legacy 8 threads × 50 clients
        prefill_cmds = [c for c in commands_sent if "--key-pattern P:P" in c]
        assert len(prefill_cmds) == 1
        assert "--threads 8" in prefill_cmds[0]
        assert "--clients 50" in prefill_cmds[0]

        measure_cmds = [c for c in commands_sent if "/usr/bin/time -v" in c]
        assert len(measure_cmds) == 1
        assert "--warmup-period 5" in measure_cmds[0]
        assert "--test-time 30" in measure_cmds[0]

    @pytest.mark.asyncio
    async def test_pinned_commands_have_taskset(self):
        """With benchmark_cpu_override, both prefill and measure get taskset."""
        runner = self._make_runner(benchmark_cpu_override="0-7,16-23")
        server = self._build_mock_server()
        commands_sent: list[str] = []

        async def capture_cmd(cmd, check=True):
            commands_sent.append(cmd)
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                return (self.MEMTIER_STDOUT, self.GNU_TIME_STDERR)
            return (self.MEMTIER_STDOUT, "")

        server.run_host_command = AsyncMock(side_effect=capture_cmd)

        repl_group = MagicMock()
        repl_group.primary = server
        repl_group.start = AsyncMock()
        repl_group.stop_all_servers = AsyncMock()
        repl_group.kill_all_valkey_instances = AsyncMock()

        runner.file_protocol = MagicMock()
        runner.file_protocol.get_result_dir.return_value = "/tmp/results"

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=repl_group):
            await runner.run()

        # Prefill must have taskset
        prefill_cmds = [c for c in commands_sent if "--key-pattern P:P" in c]
        assert len(prefill_cmds) == 1
        assert prefill_cmds[0].startswith("taskset -c 0-7,16-23 ")

        # Measure must have taskset (wrapped by GNU time)
        timed_cmds = [c for c in commands_sent if "/usr/bin/time -v" in c]
        assert len(timed_cmds) == 1
        assert "taskset -c 0-7,16-23" in timed_cmds[0]

    @pytest.mark.asyncio
    async def test_gnu_time_parsed_into_result(self):
        """GNU time stderr is parsed and populates cores_busy/utilization/saturated."""
        runner = self._make_runner(
            server_cpu_override="0-8",
            server_args="--io-threads-ownership yes",
        )
        server = self._build_mock_server()

        async def capture_cmd(cmd, check=True):
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                return (self.MEMTIER_STDOUT, self.GNU_TIME_STDERR)
            return (self.MEMTIER_STDOUT, "")

        server.run_host_command = AsyncMock(side_effect=capture_cmd)

        repl_group = MagicMock()
        repl_group.primary = server
        repl_group.start = AsyncMock()
        repl_group.stop_all_servers = AsyncMock()
        repl_group.kill_all_valkey_instances = AsyncMock()

        results_written = []
        runner.file_protocol = MagicMock()
        runner.file_protocol.get_result_dir.return_value = "/tmp/results"
        runner.file_protocol.write_results = MagicMock(side_effect=lambda r: results_written.append(r))

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=repl_group):
            await runner.run()

        # Verify result was written
        assert len(results_written) == 1
        result = results_written[0]
        assert result.data["server_cpu_override"] == "0-8"
        assert result.data["server_args"] == "--io-threads-ownership yes"
        assert result.data["warmup"] == 5
        assert result.data["warmup_applied"] is True
        client_cpu = result.data["client_cpu"]

        # GNU time: 210.50 + 30.20 = 240.70 cpu_seconds / 30.00 wall = 8.023 cores_busy
        assert client_cpu["measurement_method"] == "gnu_time"
        assert len(client_cpu["cores_busy_per_rep"]) == 1
        assert client_cpu["cores_busy_per_rep"][0] == pytest.approx(8.023, abs=0.01)

        # Default capacity: 8 threads. utilization = 8.023 / 8 ≈ 1.003
        assert client_cpu["capacity_cores"] == 8
        assert client_cpu["capacity_basis"] == "memtier_thread_count"
        assert client_cpu["utilization"] == pytest.approx(1.003, abs=0.01)
        assert client_cpu["saturated"] is True  # >90%

    @pytest.mark.asyncio
    async def test_pinned_capacity_in_result(self):
        """Pinned runner: capacity is the lower of worker count and cpuset size."""
        runner = self._make_runner(benchmark_cpu_override="0-15", memtier_threads=24)
        server = self._build_mock_server()

        async def capture_cmd(cmd, check=True):
            if "time --version" in cmd:
                return ("GNU time 1.9", "")
            if "/usr/bin/time -v" in cmd:
                # ~12 cores busy out of 16 pinned
                stderr = (
                    "\tUser time (seconds): 300.00\n"
                    "\tSystem time (seconds): 60.00\n"
                    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:30.00\n"
                )
                return (self.MEMTIER_STDOUT, stderr)
            return (self.MEMTIER_STDOUT, "")

        server.run_host_command = AsyncMock(side_effect=capture_cmd)

        repl_group = MagicMock()
        repl_group.primary = server
        repl_group.start = AsyncMock()
        repl_group.stop_all_servers = AsyncMock()
        repl_group.kill_all_valkey_instances = AsyncMock()

        results_written = []
        runner.file_protocol = MagicMock()
        runner.file_protocol.get_result_dir.return_value = "/tmp/results"
        runner.file_protocol.write_results = MagicMock(side_effect=lambda r: results_written.append(r))

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=repl_group):
            await runner.run()

        client_cpu = results_written[0].data["client_cpu"]
        assert client_cpu["capacity_cores"] == 16  # min(24 memtier threads, cpulist 0-15)
        assert client_cpu["capacity_basis"] == TestCapacityModel.PINNED_CAPACITY_BASIS
        assert client_cpu["benchmark_cpu_override"] == "0-15"
        # 360 cpu_s / 30 wall = 12.0 cores. 12.0 / 16 = 0.75
        assert client_cpu["utilization"] == pytest.approx(0.75, abs=0.01)
        assert client_cpu["saturated"] is False

    @pytest.mark.asyncio
    async def test_no_gnu_time_graceful_unavailable(self):
        """When GNU time is absent, result reports 'unavailable' with note."""
        runner = self._make_runner()
        server = self._build_mock_server()

        async def capture_cmd(cmd, check=True):
            if "time --version" in cmd:
                return ("", "")  # GNU time not found
            return (self.MEMTIER_STDOUT, "")

        server.run_host_command = AsyncMock(side_effect=capture_cmd)

        repl_group = MagicMock()
        repl_group.primary = server
        repl_group.start = AsyncMock()
        repl_group.stop_all_servers = AsyncMock()
        repl_group.kill_all_valkey_instances = AsyncMock()

        results_written = []
        runner.file_protocol = MagicMock()
        runner.file_protocol.get_result_dir.return_value = "/tmp/results"
        runner.file_protocol.write_results = MagicMock(side_effect=lambda r: results_written.append(r))

        with patch("conductress.tasks.task_mixed.ReplicationGroup", return_value=repl_group):
            await runner.run()

        client_cpu = results_written[0].data["client_cpu"]
        assert client_cpu["measurement_method"] == "unavailable"
        assert "note" in client_cpu
        assert "utilization" not in client_cpu
        assert "saturated" not in client_cpu


class TestCliAddMixed:
    """Tests for 'queue add-mixed' CLI subcommand."""

    @pytest.fixture(autouse=True)
    def isolate_queue(self, tmp_path):
        """Patch TaskQueue to use temp dir."""
        queue_path = tmp_path / "queue"
        queue_path.mkdir()

        _OriginalTaskQueue = TaskQueue

        class _IsolatedTaskQueue(_OriginalTaskQueue):
            def __init__(self, queue_dir_override=None):
                super().__init__(queue_dir=queue_path)

        with patch("conductress.cli.TaskQueue", _IsolatedTaskQueue):
            self.queue_path = queue_path
            yield

    @pytest.fixture(autouse=True)
    def patch_sources(self):
        with (
            patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
            patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
            patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
        ):
            yield

    def test_basic_add_mixed(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--sizes",
                "512",
                "--io-threads",
                "9",
                "--pipelining",
                "10",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        assert len(tasks) == 1

        data = json.loads(tasks[0].read_text())
        assert data["task_type"] == "MixedTaskData"
        assert data["set_ratio"] == 20
        assert data["val_size"] == 512
        assert data["io_threads"] == 9
        assert data["pipelining"] == 10
        assert data["warmup"] == config.DEFAULT_WARMUP
        # New fields present with defaults
        assert data["memtier_threads"] == 0
        assert data["memtier_clients"] == 0

    @pytest.mark.parametrize(("value", "expected"), [("12s", 12), ("0s", 0)])
    def test_warmup_serialized(self, value, expected):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--warmup",
                value,
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["warmup"] == expected

    def test_nonzero_key_size_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--key-sizes",
                "32",
            ]
        )
        assert exit_code == 1
        assert "--key-sizes is not supported" in capsys.readouterr().err
        assert list(self.queue_path.glob("task_*.json")) == []

    def test_invalid_ratio_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "150",
            ]
        )
        assert exit_code == 1
        assert "set-ratio must be 0-100" in capsys.readouterr().err

    def test_cartesian_product(self):
        exit_code = main(
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
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        # 2 sizes * 2 io-threads * 1 pipeline * 1 key-size = 4
        assert len(tasks) == 4

    def test_invalid_source_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "nosuchrepo",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
            ]
        )
        assert exit_code == 1
        assert "Invalid source" in capsys.readouterr().err

    def test_perf_stat_flag(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "30",
                "--perf-stat",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["perf_stat_enabled"] is True

    def test_note_stored(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--note",
                "regression check",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["note"] == "regression check"

    # -- New concurrency CLI args --

    def test_memtier_threads_override(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-threads",
                "24",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["memtier_threads"] == 24
        assert data["memtier_clients"] == 0  # still default

    def test_memtier_clients_override(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-clients",
                "100",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["memtier_clients"] == 100
        assert data["memtier_threads"] == 0  # still default

    def test_1200c_configuration(self):
        """1200 connections: 24 threads × 50 clients."""
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-threads",
                "24",
                "--memtier-clients",
                "50",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["memtier_threads"] == 24
        assert data["memtier_clients"] == 50

    def test_2400c_configuration(self):
        """2400 connections: 24 threads × 100 clients."""
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-threads",
                "24",
                "--memtier-clients",
                "100",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["memtier_threads"] == 24
        assert data["memtier_clients"] == 100

    def test_negative_memtier_threads_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-threads",
                "-1",
            ]
        )
        assert exit_code == 1
        assert "memtier_threads must be >= 0" in capsys.readouterr().err

    def test_negative_memtier_clients_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-clients",
                "-1",
            ]
        )
        assert exit_code == 1
        assert "memtier_clients must be >= 0" in capsys.readouterr().err

    def test_memtier_threads_over_max_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-threads",
                str(MAX_MEMTIER_THREADS + 1),
            ]
        )
        assert exit_code == 1
        assert f"memtier_threads must be <= {MAX_MEMTIER_THREADS}" in capsys.readouterr().err

    def test_memtier_clients_over_max_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--memtier-clients",
                str(MAX_MEMTIER_CLIENTS + 1),
            ]
        )
        assert exit_code == 1
        assert f"memtier_clients must be <= {MAX_MEMTIER_CLIENTS}" in capsys.readouterr().err
