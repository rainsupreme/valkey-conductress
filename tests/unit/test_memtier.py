"""Tests for the memtier_benchmark helpers used by scenario tasks."""

import pytest

from conductress.config import PERF_BENCH_KEYSPACE
from conductress.memtier import (
    MEMTIER_CLIENTS,
    MEMTIER_KEYSPACE,
    MEMTIER_THREADS,
    parse_memtier_total_rps,
    set_ratio_to_memtier_ratio,
)

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


class TestDefaults:
    def test_connection_count_matches_throughput_sweeps(self):
        assert MEMTIER_THREADS * MEMTIER_CLIENTS == 400

    def test_keyspace_matches_perf_task(self):
        assert MEMTIER_KEYSPACE == PERF_BENCH_KEYSPACE


class TestSetRatioConversion:
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


class TestTotalsParsing:
    def test_parse_totals_line(self):
        assert parse_memtier_total_rps(MEMTIER_STDOUT_FULL) == 1250000.00

    def test_parse_no_totals(self):
        assert parse_memtier_total_rps("some random output\nno totals here") is None

    def test_parse_empty_output(self):
        assert parse_memtier_total_rps("") is None

    def test_parse_skips_unparseable_totals_row(self):
        assert parse_memtier_total_rps("Totals     n/a\nTotals     42.5\n") == 42.5
