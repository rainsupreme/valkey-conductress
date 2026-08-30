"""Unit tests for the cachecannon task type (second-opinion generator instrument)."""

from conductress import config
from conductress.tasks.task_cachecannon import (
    CachecannonTaskData,
    CachecannonTaskRunner,
    generate_toml_config,
    parse_error_rate,
    parse_hit_rate,
    parse_latency_row,
    parse_results_block,
    parse_throughput,
)


def _valid_source():
    """Return whatever source is valid in the current test context.

    test_cli.py mutates config.REPO_NAMES at module-level, so we must read it
    at call time to stay consistent with the runtime state.
    """
    return config.REPO_NAMES[0]


# --- TOML generation tests ---


def test_toml_generation_basic():
    """TOML config includes all required sections and key fields."""
    toml = generate_toml_config(
        duration=30,
        warmup=5,
        threads=16,
        cpu_list="8,9,10,11,12,13,14,15",
        endpoint="127.0.0.1:6379",
        connections=1200,
        pipeline_depth=10,
        keyspace_count=3000000,
        val_size=512,
        test="get",
    )

    # Check sections exist
    assert "[general]" in toml
    assert "[target]" in toml
    assert "[connection]" in toml
    assert "[workload]" in toml
    assert "[workload.keyspace]" in toml
    assert "[workload.commands]" in toml
    assert "[workload.values]" in toml
    assert "[timestamps]" in toml

    # Check key values
    assert 'duration = "30s"' in toml
    assert 'warmup = "5s"' in toml
    assert "threads = 16" in toml
    assert 'cpu_list = "8,9,10,11,12,13,14,15"' in toml
    assert 'io_engine = "uring"' in toml
    assert '"127.0.0.1:6379"' in toml
    assert 'protocol = "resp"' in toml
    assert "connections = 1200" in toml
    assert "pipeline_depth = 10" in toml
    assert "prefill = true" in toml
    assert "count = 3000000" in toml
    assert "length = 16" in toml
    assert 'distribution = "uniform"' in toml
    assert "get = 100" in toml
    assert "length = 512" in toml
    assert "userspace = true" not in toml
    assert "enabled = true" in toml
    assert 'mode = "userspace"' in toml


def test_toml_humantime_and_schema_shape():
    """Regression for the Aug 5 production failure: cachecannon parses
    [general] duration/warmup with humantime_serde (REQUIRES strings like
    "300s"; bare integers are rejected at startup with 'invalid type:
    integer, expected a string'), and [timestamps] has keys enabled/mode
    (there is no 'userspace' key). Parse the generated TOML and verify the
    actual types, not just substrings."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        import tomli as tomllib  # Python <3.11 (pytest dependency chain)

    toml = generate_toml_config(
        duration=300,
        warmup=30,
        threads=16,
        cpu_list="8,9",
        endpoint="127.0.0.1:6379",
        connections=1200,
        pipeline_depth=10,
        keyspace_count=3000000,
        val_size=16,
        test="get",
    )
    parsed = tomllib.loads(toml)
    assert isinstance(parsed["general"]["duration"], str)
    assert parsed["general"]["duration"] == "300s"
    assert isinstance(parsed["general"]["warmup"], str)
    assert parsed["general"]["warmup"] == "30s"
    assert parsed["timestamps"] == {"enabled": True, "mode": "userspace"}


def test_toml_generation_set_test():
    """TOML config correctly sets 'set = 100' for SET test."""
    toml = generate_toml_config(
        duration=60,
        warmup=10,
        threads=8,
        cpu_list="0,1,2,3",
        endpoint="10.0.0.1:6379",
        connections=600,
        pipeline_depth=1,
        keyspace_count=1000000,
        val_size=64,
        test="set",
    )
    assert "set = 100" in toml
    assert "get = 100" not in toml
    # All three weights must be explicit: cachecannon serde-defaults omitted
    # weights (get=80, set=20), so 'set = 100' alone would run 80% GET.
    assert "get = 0" in toml
    assert "delete = 0" in toml


def test_toml_generation_mixed_ratio():
    """set_ratio > 0 produces a mixed GET/SET workload overriding 'test'."""
    toml = generate_toml_config(
        duration=60,
        warmup=10,
        threads=8,
        cpu_list="0,1,2,3",
        endpoint="10.0.0.1:6379",
        connections=600,
        pipeline_depth=10,
        keyspace_count=1000000,
        val_size=64,
        test="get",
        set_ratio=30,
    )
    assert "get = 70" in toml
    assert "set = 30" in toml
    assert "delete = 0" in toml


def test_toml_generation_zipf_distribution():
    """distribution='zipf' is templated into the keyspace section."""
    toml = generate_toml_config(
        duration=60,
        warmup=10,
        threads=8,
        cpu_list="0,1,2,3",
        endpoint="10.0.0.1:6379",
        connections=600,
        pipeline_depth=10,
        keyspace_count=1000000,
        val_size=64,
        test="get",
        distribution="zipf",
    )
    assert 'distribution = "zipf"' in toml
    assert 'distribution = "uniform"' not in toml


def test_toml_generation_invalid_set_ratio_raises():
    """set_ratio outside 0-100 raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="set_ratio"):
        generate_toml_config(
            duration=30,
            warmup=5,
            threads=4,
            cpu_list="",
            endpoint="127.0.0.1:6379",
            connections=400,
            pipeline_depth=10,
            keyspace_count=1000000,
            val_size=128,
            test="get",
            set_ratio=101,
        )


def test_toml_generation_invalid_distribution_raises():
    """Unknown distribution raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="distribution"):
        generate_toml_config(
            duration=30,
            warmup=5,
            threads=4,
            cpu_list="",
            endpoint="127.0.0.1:6379",
            connections=400,
            pipeline_depth=10,
            keyspace_count=1000000,
            val_size=128,
            test="get",
            distribution="gaussian",
        )


def test_toml_generation_empty_cpu_list():
    """TOML config handles empty cpu_list (OS scheduling)."""
    toml = generate_toml_config(
        duration=30,
        warmup=5,
        threads=4,
        cpu_list="",
        endpoint="127.0.0.1:6379",
        connections=400,
        pipeline_depth=10,
        keyspace_count=1000000,
        val_size=128,
        test="get",
    )
    assert 'cpu_list = ""' in toml


# --- Throughput parsing tests ---


def test_parse_throughput_millions():
    assert parse_throughput("throughput   1.7M req/s, 0.00% errors") == 1_700_000.0


def test_parse_throughput_thousands():
    assert parse_throughput("throughput   850K req/s, 0.00% errors") == 850_000.0


def test_parse_throughput_plain():
    assert parse_throughput("throughput   1234567 req/s, 0.00% errors") == 1_234_567.0


def test_parse_throughput_decimal_millions():
    assert parse_throughput("throughput   2.53M req/s, 0.01% errors") == 2_530_000.0


# --- Error rate parsing tests ---


def test_parse_error_rate_zero():
    assert parse_error_rate("throughput   1.7M req/s, 0.00% errors") == 0.0


def test_parse_error_rate_nonzero():
    assert parse_error_rate("throughput   1.7M req/s, 1.50% errors") == 1.5


# --- Hit rate parsing tests ---


def test_parse_hit_rate_full():
    result = parse_hit_rate("hit rate     100% (50.4M hit, 0 miss)")
    assert result["percent"] == 100.0
    assert result["hits"] == 50_400_000.0
    assert result["misses"] == 0.0


def test_parse_hit_rate_partial():
    result = parse_hit_rate("hit rate     99.5% (1.2M hit, 6K miss)")
    assert result["percent"] == 99.5
    assert result["hits"] == 1_200_000.0
    assert result["misses"] == 6_000.0


def test_parse_hit_rate_billions():
    # Regression: exact line from g4bench i6-cc failure 2026-08-29 --
    # 5-minute runs at multi-M req/s cross 1e9 hits and cachecannon
    # format_count() switches to the 'B' suffix.
    result = parse_hit_rate("hit rate     100% (2.0B hit, 0 miss)")
    assert result["percent"] == 100.0
    assert result["hits"] == 2_000_000_000.0
    assert result["misses"] == 0.0


def test_parse_throughput_billions():
    assert parse_throughput("throughput   1.2B req/s, 0.00% errors") == 1_200_000_000.0


# --- Latency row parsing tests ---


def test_parse_latency_row_basic():
    row = "GET          7.24 ms   7.34 ms   8.12 ms   9.45 ms   12.3 ms   15.6 ms"
    result = parse_latency_row(row)
    assert result["command"] == "GET"
    assert result["p50_ms"] == 7.24
    assert result["p90_ms"] == 7.34
    assert result["p99_ms"] == 8.12
    assert result["p999_ms"] == 9.45
    assert result["p9999_ms"] == 12.3
    assert result["max_ms"] == 15.6


def test_parse_latency_row_set():
    row = "SET          3.10 ms   3.45 ms   4.20 ms   5.00 ms   6.50 ms   8.00 ms"
    result = parse_latency_row(row)
    assert result["command"] == "SET"
    assert result["p50_ms"] == 3.10
    assert result["max_ms"] == 8.00


def test_parse_latency_row_mixed_units():
    # Regression: exact shape from g4bench i6-cc cells (2026-08-30) --
    # cachecannon formats each value independently, so sub-ms percentiles
    # get a µs suffix while tails get ms. Old parser dropped the units and
    # recorded p50_ms=938.0 (> p999_ms=1.88).
    import pytest

    row = "GET           938 \u00b5s   958 \u00b5s   975 \u00b5s   1.88 ms   1.91 ms   2.21 ms"
    result = parse_latency_row(row)
    assert result["command"] == "GET"
    assert result["p50_ms"] == pytest.approx(0.938)
    assert result["p90_ms"] == pytest.approx(0.958)
    assert result["p99_ms"] == pytest.approx(0.975)
    assert result["p999_ms"] == pytest.approx(1.88)
    assert result["p9999_ms"] == pytest.approx(1.91)
    assert result["max_ms"] == pytest.approx(2.21)
    assert result["p50_ms"] < result["p999_ms"]  # monotonic sanity


def test_parse_latency_row_multiword_command():
    import pytest

    row = "GET TTFB      512 \u00b5s   600 \u00b5s   700 \u00b5s   1.10 ms   1.20 ms   1.50 ms"
    result = parse_latency_row(row)
    assert result["command"] == "GET TTFB"
    assert result["p50_ms"] == pytest.approx(0.512)


def test_parse_latency_row_seconds_unit():
    row = "SET          800 ms   900 ms   950 ms   1.20 s   1.50 s   2.00 s"
    result = parse_latency_row(row)
    assert result["p999_ms"] == 1200.0
    assert result["max_ms"] == 2000.0


def test_parse_latency_row_unknown_unit_rejected():
    import pytest

    row = "GET          7.24 xx   7.34 ms   8.12 ms   9.45 ms   12.3 ms   15.6 ms"
    with pytest.raises(ValueError):
        parse_latency_row(row)


# --- Full RESULTS block parsing ---

SAMPLE_OUTPUT = """
[ringline diag] worker 0: 142315 ops/sec
[ringline diag] worker 1: 141982 ops/sec
[ringline diag] worker 2: 143211 ops/sec
RESULTS (30s)
throughput   1.7M req/s, 0.00% errors
hit rate     100% (50.4M hit, 0 miss)
GET          7.24 ms   7.34 ms   8.12 ms   9.45 ms   12.3 ms   15.6 ms
"""


def test_parse_results_block_full():
    result = parse_results_block(SAMPLE_OUTPUT)
    assert result["throughput_rps"] == 1_700_000.0
    assert result["error_pct"] == 0.0
    assert result["hit_rate"]["percent"] == 100.0
    assert result["hit_rate"]["hits"] == 50_400_000.0
    assert result["latency"]["command"] == "GET"
    assert result["latency"]["p50_ms"] == 7.24
    assert result["latency"]["max_ms"] == 15.6


def test_parse_results_block_with_errors_raises():
    """Non-zero errors should be parseable (task runner decides to fail)."""
    output = """
RESULTS (30s)
throughput   500K req/s, 2.50% errors
hit rate     98% (14.7M hit, 300K miss)
GET          10.0 ms   12.0 ms   15.0 ms   20.0 ms   25.0 ms   30.0 ms
"""
    result = parse_results_block(output)
    assert result["error_pct"] == 2.5
    assert result["throughput_rps"] == 500_000.0


def test_parse_results_block_missing_raises():
    """Missing RESULTS block should raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="No RESULTS block"):
        parse_results_block("some random output with no results")


# --- Task data tests ---


def test_task_data_short_description():
    task = CachecannonTaskData(
        source=_valid_source(),
        specifier="unstable",
        make_args="",
        replicas=0,
        note="test",
        requirements={},
        test="get",
        val_size=512,
        pipelining=10,
        connections=1200,
        threads=16,
        warmup=5,
        duration=30,
        repetitions=3,
    )
    desc = task.short_description()
    assert "cachecannon" in desc
    assert "get" in desc
    assert "x3" in desc


def test_task_data_short_description_mixed_zipf():
    """Mixed/zipf workloads are reflected in the description."""
    task = CachecannonTaskData(
        source=_valid_source(),
        specifier="unstable",
        make_args="",
        replicas=0,
        note="test",
        requirements={},
        test="get",
        val_size=16,
        pipelining=10,
        connections=1200,
        threads=16,
        warmup=5,
        duration=30,
        repetitions=3,
        set_ratio=30,
        distribution="zipf",
    )
    desc = task.short_description()
    assert "mixed s30" in desc
    assert "zipf" in desc


def test_task_data_task_type_registration():
    """CachecannonTaskData should be registered in the task registry."""
    from conductress.task_queue import BaseTaskData

    # Registration happens at import time via __init_subclass__
    assert "CachecannonTaskData" in BaseTaskData._BaseTaskData__task_registry


def test_task_data_serialization_roundtrip():
    """Task data should serialize to dict and back."""
    import json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from conductress.task_queue import BaseTaskData

    task = CachecannonTaskData(
        source=_valid_source(),
        specifier="abc123",
        make_args="",
        replicas=0,
        note="roundtrip test",
        requirements={},
        test="get",
        val_size=256,
        pipelining=5,
        connections=600,
        threads=8,
        warmup=3,
        duration=15,
        repetitions=2,
        keyspace_count=1000000,
        cachecannon_binary="/usr/local/bin/cachecannon",
        server_args="--io-threads-ownership yes",
        set_ratio=50,
        distribution="zipf",
    )

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        task.save_to_file(Path(f.name))
        loaded = BaseTaskData.from_file(Path(f.name))

    assert isinstance(loaded, CachecannonTaskData)
    assert loaded.test == "get"
    assert loaded.val_size == 256
    assert loaded.pipelining == 5
    assert loaded.connections == 600
    assert loaded.threads == 8
    assert loaded.cachecannon_binary == "/usr/local/bin/cachecannon"
    assert loaded.server_args == "--io-threads-ownership yes"
    assert loaded.set_ratio == 50
    assert loaded.distribution == "zipf"
    assert loaded.note == "roundtrip test"

    import os

    os.unlink(f.name)


# --- Runner construction test ---


def test_runner_construction():
    """CachecannonTaskRunner should initialize from task data."""
    from conductress.config import ServerInfo

    task = CachecannonTaskData(
        source=_valid_source(),
        specifier="abc123",
        make_args="",
        replicas=0,
        note="",
        requirements={},
    )
    server_infos = [ServerInfo(ip="127.0.0.1")]
    runner = task.prepare_task_runner(server_infos)
    assert isinstance(runner, CachecannonTaskRunner)
    assert runner.test == "get"
    assert runner.val_size == 512
    assert runner.pipelining == 10
