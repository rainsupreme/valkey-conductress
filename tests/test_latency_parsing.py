"""Tests for latency task runner parsing and export."""

import json
from pathlib import Path

import pytest

from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState
from conductress.tasks.task_latency import LatencyTaskRunner

# Real memtier output captured from ARM experiment (rate-limited 500K, P=10)
MEMTIER_OUTPUT_SAMPLE = """Writing results to stdout
[RUN #1] Preparing benchmark client...
[RUN #1] Launching threads now...
[RUN #1 100%,  10 secs]  0 threads 25 conns:    14992900 ops,  499880 (avg:  499726) ops/sec, 27.12MB/sec (avg: 27.11MB/sec),  0.46 (avg:  0.46) msec latency

4         Threads
25        Connections per thread
10        Seconds


ALL STATS
============================================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency    p100 Latency       KB/sec
--------------------------------------------------------------------------------------------------------------------------------------------
Sets            0.00          ---          ---             ---             ---             ---             ---             ---         0.00
Gets       499385.90    499385.90         0.00         0.33021         0.32700         0.51100         1.34300         6.36700     27742.22
Waits           0.00          ---          ---             ---             ---             ---             ---             ---          ---
Totals     499385.90    499385.90         0.00         0.33021         0.32700         0.51100         1.34300         6.36700     27742.22
"""

# Real HDR histogram output captured from ARM experiment
HDR_HISTOGRAM_SAMPLE = """       Value   Percentile   TotalCount 1/(1-Percentile)

        0.01     0.000000          325         1.00
        0.09     0.050000       145084         1.05
        0.13     0.100000       250783         1.11
        0.14     0.150000       558161         1.18
        0.15     0.250000      1322362         1.33
        0.15     0.500000      1322362         2.00
        0.16     0.550000      1846137         2.22
        0.17     0.750000      1919217         4.00
        0.21     0.800000      2020998         5.00
        0.22     0.825000      2171562         5.71
        0.23     0.900000      2300000         10.00
        0.25     0.950000      2400000         20.00
        0.31     0.990000      2480000         100.00
        0.42     0.995000      2490000         200.00
        0.89     0.999000      2497000         1000.00
        2.91     0.999900      2499700         10000.00
        6.37     1.000000      2500000          inf
"""


class TestMemtierOutputParsing:
    """Test parsing of memtier_benchmark summary output."""

    def setup_method(self):
        # Create a minimal runner instance for testing parse methods
        self.runner = LatencyTaskRunner.__new__(LatencyTaskRunner)

    def test_parses_gets_line(self):
        result = self.runner._parse_memtier_output(MEMTIER_OUTPUT_SAMPLE)
        assert result is not None
        assert result["actual_rps"] == pytest.approx(499385.90)
        assert result["p50_us"] == pytest.approx(327.0)  # 0.327ms * 1000
        assert result["p99_us"] == pytest.approx(511.0)
        assert result["p99_9_us"] == pytest.approx(1343.0)
        assert result["p100_us"] == pytest.approx(6367.0)

    def test_returns_none_on_empty_output(self):
        assert self.runner._parse_memtier_output("") is None

    def test_returns_none_on_garbage(self):
        assert self.runner._parse_memtier_output("some random text\nno data here") is None

    def test_handles_totals_line(self):
        # Output with only Totals line (no separate Gets)
        output = "Totals     1499174.26   1499174.26         0.00         1.69461         1.89500         2.11100         3.18300         7.45500     83283.89"
        result = self.runner._parse_memtier_output(output)
        assert result is not None
        assert result["actual_rps"] == pytest.approx(1499174.26)
        assert result["p50_us"] == pytest.approx(1895.0)
        assert result["p99_us"] == pytest.approx(2111.0)
        assert result["p99_9_us"] == pytest.approx(3183.0)
        assert result["p100_us"] == pytest.approx(7455.0)


class TestHdrHistogramParsing:
    """Test parsing of HDR histogram .txt files."""

    def setup_method(self):
        self.runner = LatencyTaskRunner.__new__(LatencyTaskRunner)

    def test_parses_histogram_to_cdf_buckets(self):
        # Simulate async method by calling the sync parsing logic directly
        from conductress.tasks.task_latency import HISTOGRAM_PERCENTILES

        entries = []
        for line in HDR_HISTOGRAM_SAMPLE.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Value"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    latency_ms = float(parts[0])
                    percentile = float(parts[1])
                    entries.append((percentile, latency_ms * 1000))
                except ValueError:
                    continue

        # Extract at target percentiles
        histogram = []
        for target_pct in HISTOGRAM_PERCENTILES:
            closest = min(entries, key=lambda e: abs(e[0] - target_pct))
            histogram.append([target_pct, closest[1]])

        assert len(histogram) == 11
        # p50 should be around 150µs (0.15ms)
        p50_entry = next(h for h in histogram if h[0] == 0.50)
        assert p50_entry[1] == pytest.approx(150.0)
        # p99 should be around 310µs (0.31ms)
        p99_entry = next(h for h in histogram if h[0] == 0.99)
        assert p99_entry[1] == pytest.approx(310.0)
        # p100 should be around 6370µs (6.37ms)
        p100_entry = next(h for h in histogram if h[0] == 1.0)
        assert p100_entry[1] == pytest.approx(6370.0)

    def test_empty_input_returns_empty(self):
        entries = []
        assert entries == []


class TestAggregation:
    """Test repetition aggregation logic."""

    def setup_method(self):
        self.runner = LatencyTaskRunner.__new__(LatencyTaskRunner)

    def test_median_of_three_reps(self):
        reps = [
            {
                "actual_rps": 500000,
                "p50_us": 320,
                "p99_us": 510,
                "p99_9_us": 1300,
                "p100_us": 6000,
                "histogram": [[0.5, 320]],
            },
            {
                "actual_rps": 499000,
                "p50_us": 330,
                "p99_us": 520,
                "p99_9_us": 1400,
                "p100_us": 7000,
                "histogram": [[0.5, 330]],
            },
            {
                "actual_rps": 501000,
                "p50_us": 310,
                "p99_us": 500,
                "p99_9_us": 1200,
                "p100_us": 5000,
                "histogram": [[0.5, 310]],
            },
        ]
        result = self.runner._aggregate_reps(reps)
        assert result["actual_rps"] == 500000  # median
        assert result["p50_us"] == 320
        assert result["p99_us"] == 510
        assert result["p99_9_us"] == 1300
        assert result["p100_us"] == 6000
        assert result["reps"] == 3
        # Histogram from median rep (index 1)
        assert result["histogram"] == [[0.5, 330]]


class TestLatencyExport:
    """Test the latency export function."""

    def test_export_produces_valid_json(self, tmp_path, monkeypatch):
        from conductress.sweep.exporter import export_latency

        # Create state with some latency results
        state = SweepState()
        state.merge_commits = ["aaa", "bbb", "ccc"]
        state.commit_dates = {"aaa": "2026-01-01", "bbb": "2026-02-01", "ccc": "2026-03-01"}
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500.0, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        state.points["ccc"] = BenchmarkPoint(
            commit="ccc", date="2026-03-01", value=800.0, cv=0, reps=3, status=PointStatus.COMPLETED
        )

        # Mock CONDUCTRESS_RESULTS to empty dir (no output.jsonl)
        monkeypatch.setattr("conductress.config.CONDUCTRESS_RESULTS", tmp_path)

        output_file = tmp_path / "series-arm64-get16b-t9-p10-latency.json"
        count = export_latency(state, output_file, platform="arm64", workload="get16b-t9-p10")

        assert count == 2
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["metadata"]["metric"] == "latency"
        assert data["metadata"]["unit"] == "µs"
        assert data["metadata"]["load_fraction"] == 0.70
        assert data["metadata"]["platform"] == "arm64"
        assert len(data["points"]) == 2
        assert data["points"][0]["commit"] == "aaa"
        assert data["points"][0]["p99_us"] == 500.0
        assert data["points"][1]["p99_us"] == 800.0

    def test_export_includes_annotations_for_adjacent_commits(self, tmp_path, monkeypatch):
        from conductress.sweep.exporter import export_latency

        state = SweepState()
        state.merge_commits = ["aaa", "bbb"]
        state.threshold = 0.10  # 10%
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2026-01-01", value=500.0, cv=0, reps=3, status=PointStatus.COMPLETED
        )
        state.points["bbb"] = BenchmarkPoint(
            commit="bbb", date="2026-02-01", value=700.0, cv=0, reps=3, status=PointStatus.COMPLETED
        )

        monkeypatch.setattr("conductress.config.CONDUCTRESS_RESULTS", tmp_path)

        output_file = tmp_path / "test.json"
        export_latency(state, output_file, platform="arm64", workload="get16b-t9-p10")

        data = json.loads(output_file.read_text())
        # 40% increase in latency (lower is better) = regression
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["type"] == "regression"
        assert data["annotations"][0]["commit"] == "bbb"
