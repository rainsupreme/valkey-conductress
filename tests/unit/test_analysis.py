"""Tests for the comparison and statistical analysis module.

Covers:
- Task 10.3: Property test for Welch's t-test correctness (Property 7)
- Task 10.4: Unit tests for analysis module (Requirements 9.5)
"""

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import ttest_ind

from src.analysis import AnalysisModule, ComparisonRow, _format_size


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_result(
    specifier: str = "branch-a",
    method: str = "perf-get",
    score: float = 100000.0,
    source: str = "valkey",
    val_size: int = 512,
    key_size: int = 0,
    io_threads: int = 1,
    pipeline: int = 1,
    make_args: str = "",
    repetitions: int = 1,
    per_run_rps: Optional[List[float]] = None,
) -> dict:
    """Helper to build a single result record for output.jsonl."""
    data: dict = {
        "size": val_size,
        "key_size": key_size,
        "io-threads": io_threads,
        "pipeline": pipeline,
        "repetitions": repetitions,
    }
    if per_run_rps is not None:
        data["per_run_rps"] = per_run_rps

    return {
        "specifier": specifier,
        "method": method,
        "score": score,
        "source": source,
        "make_args": make_args,
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 10.3 — Property test for Welch's t-test (Property 7)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWelchTTestProperty:
    """Property-based test for Welch's t-test correctness."""

    @settings(max_examples=100)
    @given(
        samples_a=st.lists(
            st.floats(
                min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=20,
        ),
        samples_b=st.lists(
            st.floats(
                min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=20,
        ),
    )
    def test_welch_ttest_correctness(self, samples_a, samples_b):
        """**Validates: Requirements 5.4**

        Feature: benchmark-tooling-enhancements, Property 7: Welch's t-test correctness

        For any two lists of samples each with length >= 2, the AnalysisModule's
        t-test computation shall produce a p-value equal to
        scipy.stats.ttest_ind(a, b, equal_var=False).pvalue within floating-point tolerance.
        """
        tmp = tempfile.mkdtemp()
        try:
            # Build JSONL with one record per sample for each specifier
            records = []
            for val in samples_a:
                records.append(
                    _make_result(specifier="spec-a", method="perf-get", score=val)
                )
            for val in samples_b:
                records.append(
                    _make_result(specifier="spec-b", method="perf-get", score=val)
                )

            output_path = Path(tmp) / "output.jsonl"
            _write_jsonl(output_path, records)

            module = AnalysisModule(results_path=output_path)
            rows = module.compare("spec-a", "spec-b")

            assert len(rows) == 1, f"Expected 1 comparison row, got {len(rows)}"
            row = rows[0]

            # Verify p-value matches scipy's ttest_ind
            expected = ttest_ind(samples_a, samples_b, equal_var=False)
            assert (
                row.p_value is not None
            ), "p_value should not be None when both sides have >= 2 samples"

            # When both samples have zero variance, scipy returns NaN — our module
            # should match that behavior.
            if math.isnan(expected.pvalue):
                assert math.isnan(
                    row.p_value
                ), f"Expected NaN p_value (zero-variance inputs), got {row.p_value}"
            else:
                assert math.isclose(
                    row.p_value, expected.pvalue, rel_tol=1e-9
                ), f"p_value={row.p_value} != expected={expected.pvalue}"
        finally:
            shutil.rmtree(tmp)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 10.4 — Unit tests for analysis module (Requirements 9.5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadResults:
    """Test result loading from output.jsonl."""

    def test_load_results_basic(self, temp_dir):
        """Load all results from a simple JSONL file."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="b", score=200),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        assert len(results) == 2

    def test_load_results_missing_file(self, temp_dir):
        """Return empty list when the results file does not exist."""
        path = temp_dir / "nonexistent.jsonl"
        module = AnalysisModule(results_path=path)
        results = module.load_results()
        assert results == []

    def test_load_results_empty_file(self, temp_dir):
        """Return empty list when the results file is empty."""
        path = temp_dir / "output.jsonl"
        path.write_text("")
        module = AnalysisModule(results_path=path)
        results = module.load_results()
        assert results == []

    def test_load_results_skips_malformed_json(self, temp_dir):
        """Skip malformed JSON lines and continue processing valid ones."""
        path = temp_dir / "output.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_make_result(specifier="a")) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps(_make_result(specifier="b")) + "\n")

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        assert len(results) == 2

    def test_load_results_source_filter(self, temp_dir):
        """Filter results by source."""
        records = [
            _make_result(source="valkey", score=100),
            _make_result(source="rainsupreme", score=200),
            _make_result(source="valkey", score=300),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results(source_filter="valkey")
        assert len(results) == 2
        assert all(r["source"] == "valkey" for r in results)

    def test_load_results_method_filter(self, temp_dir):
        """Filter results by method."""
        records = [
            _make_result(method="perf-get", score=100),
            _make_result(method="perf-set", score=200),
            _make_result(method="perf-get", score=300),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results(method_filter="perf-set")
        assert len(results) == 1
        assert results[0]["method"] == "perf-set"

    def test_load_results_combined_filters(self, temp_dir):
        """Filter results by both source and method simultaneously."""
        records = [
            _make_result(source="valkey", method="perf-get", score=100),
            _make_result(source="valkey", method="perf-set", score=200),
            _make_result(source="rainsupreme", method="perf-get", score=300),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results(source_filter="valkey", method_filter="perf-get")
        assert len(results) == 1
        assert results[0]["source"] == "valkey"
        assert results[0]["method"] == "perf-get"


class TestGroupResults:
    """Test result grouping logic."""

    def test_group_results_basic(self, temp_dir):
        """Group results by specifier into a and b sides."""
        records = [
            _make_result(specifier="branch-a", method="perf-get", score=100),
            _make_result(specifier="branch-b", method="perf-get", score=200),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        groups = module.group_results(results, "branch-a", "branch-b")

        assert len(groups) == 1
        key = list(groups.keys())[0]
        assert groups[key]["a"] == [100.0]
        assert groups[key]["b"] == [200.0]

    def test_group_results_multiple_samples(self, temp_dir):
        """Multiple results for the same specifier and group key are collected."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=110),
            _make_result(specifier="b", score=200),
            _make_result(specifier="b", score=210),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        groups = module.group_results(results, "a", "b")

        key = list(groups.keys())[0]
        assert len(groups[key]["a"]) == 2
        assert len(groups[key]["b"]) == 2

    def test_group_results_ignores_other_specifiers(self, temp_dir):
        """Results with specifiers other than a or b are ignored."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="b", score=200),
            _make_result(specifier="c", score=300),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        groups = module.group_results(results, "a", "b")

        key = list(groups.keys())[0]
        assert groups[key]["a"] == [100.0]
        assert groups[key]["b"] == [200.0]

    def test_group_results_multiple_groups(self, temp_dir):
        """Results with different parameters form separate groups."""
        records = [
            _make_result(specifier="a", method="perf-get", val_size=512, score=100),
            _make_result(specifier="b", method="perf-get", val_size=512, score=200),
            _make_result(specifier="a", method="perf-set", val_size=512, score=300),
            _make_result(specifier="b", method="perf-set", val_size=512, score=400),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        results = module.load_results()
        groups = module.group_results(results, "a", "b")

        assert len(groups) == 2


class TestCompare:
    """Test the compare() method with known values."""

    def test_compare_basic_two_specifiers(self, temp_dir):
        """Compare two specifiers with multiple samples each."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=110),
            _make_result(specifier="a", score=105),
            _make_result(specifier="b", score=200),
            _make_result(specifier="b", score=210),
            _make_result(specifier="b", score=205),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.n_a == 3
        assert row.n_b == 3
        assert row.mean_a == pytest.approx(105.0)
        assert row.mean_b == pytest.approx(205.0)
        assert row.p_value is not None

        # Verify p-value matches scipy directly
        expected_p = ttest_ind([100, 110, 105], [200, 210, 205], equal_var=False).pvalue
        assert row.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_compare_known_p_value(self, temp_dir):
        """Verify Welch's t-test computation against known values."""
        samples_a = [100.0, 200.0, 300.0]
        samples_b = [400.0, 500.0, 600.0]

        records = []
        for v in samples_a:
            records.append(_make_result(specifier="a", score=v))
        for v in samples_b:
            records.append(_make_result(specifier="b", score=v))

        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]

        expected = ttest_ind(samples_a, samples_b, equal_var=False)
        assert row.p_value == pytest.approx(expected.pvalue, rel=1e-9)
        assert row.mean_a == pytest.approx(200.0)
        assert row.mean_b == pytest.approx(500.0)

    def test_compare_delta_pct(self, temp_dir):
        """Verify delta percentage calculation."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=100),
            _make_result(specifier="b", score=110),
            _make_result(specifier="b", score=110),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.delta_pct == pytest.approx(10.0)

    def test_compare_no_results(self, temp_dir):
        """Return empty list when no results file exists."""
        path = temp_dir / "nonexistent.jsonl"
        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")
        assert rows == []

    def test_compare_no_matching_specifiers(self, temp_dir):
        """Return empty list when specifiers don't match any results."""
        records = [
            _make_result(specifier="x", score=100),
            _make_result(specifier="y", score=200),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")
        assert rows == []

    def test_compare_with_source_filter(self, temp_dir):
        """Verify source filter is applied during comparison."""
        records = [
            _make_result(specifier="a", source="valkey", score=100),
            _make_result(specifier="a", source="valkey", score=110),
            _make_result(specifier="b", source="valkey", score=200),
            _make_result(specifier="b", source="valkey", score=210),
            _make_result(specifier="a", source="other", score=999),
            _make_result(specifier="b", source="other", score=999),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b", source_filter="valkey")

        assert len(rows) == 1
        row = rows[0]
        assert row.mean_a == pytest.approx(105.0)
        assert row.mean_b == pytest.approx(205.0)

    def test_compare_with_method_filter(self, temp_dir):
        """Verify method filter is applied during comparison."""
        records = [
            _make_result(specifier="a", method="perf-get", score=100),
            _make_result(specifier="a", method="perf-get", score=110),
            _make_result(specifier="b", method="perf-get", score=200),
            _make_result(specifier="b", method="perf-get", score=210),
            _make_result(specifier="a", method="perf-set", score=999),
            _make_result(specifier="b", method="perf-set", score=999),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b", method_filter="perf-get")

        assert len(rows) == 1
        row = rows[0]
        assert row.method == "perf-get"


class TestAggregatedVsSingleRun:
    """Test handling of aggregated (repetitions > 1) vs single-run results."""

    def test_single_run_uses_score_as_sample(self, temp_dir):
        """Single-run results (repetitions=1) use score as a single sample."""
        records = [
            _make_result(specifier="a", score=100, repetitions=1),
            _make_result(specifier="a", score=110, repetitions=1),
            _make_result(specifier="b", score=200, repetitions=1),
            _make_result(specifier="b", score=210, repetitions=1),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.n_a == 2
        assert row.n_b == 2
        assert row.mean_a == pytest.approx(105.0)
        assert row.mean_b == pytest.approx(205.0)

    def test_aggregated_uses_per_run_rps(self, temp_dir):
        """Aggregated results (repetitions > 1) use per_run_rps as samples."""
        records = [
            _make_result(
                specifier="a",
                score=105,
                repetitions=3,
                per_run_rps=[100.0, 105.0, 110.0],
            ),
            _make_result(
                specifier="b",
                score=205,
                repetitions=3,
                per_run_rps=[200.0, 205.0, 210.0],
            ),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        # Aggregated results expand per_run_rps into individual samples
        assert row.n_a == 3
        assert row.n_b == 3
        assert row.mean_a == pytest.approx(105.0)
        assert row.mean_b == pytest.approx(205.0)
        assert row.p_value is not None

        # Verify p-value matches scipy
        expected_p = ttest_ind(
            [100.0, 105.0, 110.0], [200.0, 205.0, 210.0], equal_var=False
        ).pvalue
        assert row.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_mixed_aggregated_and_single_run(self, temp_dir):
        """Mix of aggregated and single-run results for the same specifier."""
        records = [
            # Specifier a: one aggregated result (3 samples) + one single-run
            _make_result(
                specifier="a",
                score=105,
                repetitions=3,
                per_run_rps=[100.0, 105.0, 110.0],
            ),
            _make_result(specifier="a", score=120, repetitions=1),
            # Specifier b: two single-run results
            _make_result(specifier="b", score=200, repetitions=1),
            _make_result(specifier="b", score=210, repetitions=1),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        # a: 3 from per_run_rps + 1 from single-run = 4 samples
        assert row.n_a == 4
        assert row.n_b == 2


class TestInsufficientData:
    """Test handling of insufficient data (< 2 samples per specifier)."""

    def test_single_sample_each_no_p_value(self, temp_dir):
        """When both specifiers have only 1 sample, p_value is None."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="b", score=200),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.p_value is None
        assert row.n_a == 1
        assert row.n_b == 1

    def test_one_side_insufficient(self, temp_dir):
        """When one specifier has < 2 samples, p_value is None."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=110),
            _make_result(specifier="b", score=200),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.p_value is None
        assert row.n_a == 2
        assert row.n_b == 1

    def test_both_sides_sufficient(self, temp_dir):
        """When both specifiers have >= 2 samples, p_value is computed."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=110),
            _make_result(specifier="b", score=200),
            _make_result(specifier="b", score=210),
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.p_value is not None

    def test_one_side_empty_no_row(self, temp_dir):
        """When one specifier has 0 samples, no row is produced for that group."""
        records = [
            _make_result(specifier="a", score=100),
            _make_result(specifier="a", score=110),
            # No results for specifier "b"
        ]
        path = temp_dir / "output.jsonl"
        _write_jsonl(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        # No rows because b has 0 samples (skipped entirely)
        assert rows == []


class TestFormatTable:
    """Test formatted table output."""

    def test_format_table_empty(self):
        """Empty rows produce a 'no data' message."""
        module = AnalysisModule(results_path=Path("/dev/null"))
        result = module.format_table([])
        assert result == "No comparison data available."

    def test_format_table_single_row(self):
        """Single row produces a header, separator, one data line, and a summary footer."""
        row = ComparisonRow(
            method="perf-get",
            val_size=512,
            key_size=0,
            io_threads=1,
            pipelining=1,
            make_args="",
            mean_a=100000.0,
            mean_b=110000.0,
            delta_pct=10.0,
            p_value=0.0234,
            n_a=3,
            n_b=3,
        )
        module = AnalysisModule(results_path=Path("/dev/null"))
        table = module.format_table([row])

        lines = table.split("\n")
        # Table has: header, separator, data row, blank, summary lines
        assert len(lines) >= 3  # at minimum header, separator, data row

        # Verify header contains expected column names
        assert "Test" in lines[0]
        assert "Mean A" in lines[0]
        assert "Mean B" in lines[0]
        assert "Delta" in lines[0]
        assert "p-value" in lines[0]

        # Verify data row contains expected values
        data_line = lines[2]
        assert "perf-get" in data_line
        assert "100,000 rps" in data_line
        assert "110,000 rps" in data_line
        assert "+10.00%" in data_line
        assert "0.0234" in data_line

        # Verify summary footer
        assert "Comparisons: 1" in table
        assert "Significant (p < 0.05): 1/1" in table

    def test_format_table_na_p_value(self):
        """Row with p_value=None shows 'N/A'."""
        row = ComparisonRow(
            method="perf-set",
            val_size=1024,
            key_size=0,
            io_threads=1,
            pipelining=1,
            make_args="",
            mean_a=50000.0,
            mean_b=55000.0,
            delta_pct=10.0,
            p_value=None,
            n_a=1,
            n_b=1,
        )
        module = AnalysisModule(results_path=Path("/dev/null"))
        table = module.format_table([row])

        assert "N/A" in table

    def test_format_table_multiple_rows(self):
        """Multiple rows produce header + separator + N data lines."""
        rows = [
            ComparisonRow(
                method="perf-get",
                val_size=512,
                key_size=0,
                io_threads=1,
                pipelining=1,
                make_args="",
                mean_a=100000.0,
                mean_b=110000.0,
                delta_pct=10.0,
                p_value=0.01,
                n_a=3,
                n_b=3,
            ),
            ComparisonRow(
                method="perf-set",
                val_size=512,
                key_size=0,
                io_threads=1,
                pipelining=1,
                make_args="",
                mean_a=90000.0,
                mean_b=95000.0,
                delta_pct=5.56,
                p_value=0.05,
                n_a=3,
                n_b=3,
            ),
        ]
        module = AnalysisModule(results_path=Path("/dev/null"))
        table = module.format_table(rows)

        lines = table.split("\n")
        # header + separator + 2 data rows + blank + summary lines
        assert len(lines) >= 4  # at minimum header, separator, 2 data rows
        # Verify both data rows are present
        assert "perf-get" in lines[2]
        assert "perf-set" in lines[3]
        assert "Comparisons: 2" in table

    def test_format_table_negative_delta(self):
        """Negative delta percentage is formatted with minus sign."""
        row = ComparisonRow(
            method="perf-get",
            val_size=512,
            key_size=0,
            io_threads=1,
            pipelining=1,
            make_args="",
            mean_a=110000.0,
            mean_b=100000.0,
            delta_pct=-9.09,
            p_value=0.03,
            n_a=3,
            n_b=3,
        )
        module = AnalysisModule(results_path=Path("/dev/null"))
        table = module.format_table([row])

        assert "-9.09%" in table


class TestFormatSize:
    """Test the _format_size helper."""

    def test_format_size_zero(self):
        assert _format_size(0) == "0"

    def test_format_size_bytes(self):
        assert _format_size(512) == "512B"

    def test_format_size_kilobytes(self):
        assert _format_size(1024) == "1KB"

    def test_format_size_kilobytes_fractional(self):
        assert _format_size(1536) == "1.5KB"

    def test_format_size_megabytes(self):
        assert _format_size(1048576) == "1MB"
