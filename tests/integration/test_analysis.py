"""Integration tests for the analysis module.

Tests verify the analysis module against a fixture output.jsonl file
containing known benchmark results, confirming correct statistical output
and table formatting.

Requirements: 9.9
"""

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest
from scipy.stats import ttest_ind

from src.analysis import AnalysisModule, main


def _make_record(
    method: str = "perf-get",
    source: str = "valkey",
    specifier: str = "branch-a",
    commit_hash: str = "abc123",
    score: float = 141000.5,
    val_size: int = 512,
    key_size: int = 0,
    io_threads: int = 1,
    pipeline: int = 1,
    make_args: str = "",
    warmup: int = 60,
    duration: int = 900,
    has_expire: bool = False,
    preload_keys: bool = True,
    repetitions: int = 1,
    per_run_rps: Optional[List[float]] = None,
    mean_rps: Optional[float] = None,
    ci_95: Optional[float] = None,
    note: str = "",
) -> dict:
    """Build a realistic output.jsonl record matching the production format."""
    data: dict = {
        "warmup": warmup,
        "duration": duration,
        "io-threads": io_threads,
        "pipeline": pipeline,
        "has_expire": has_expire,
        "size": val_size,
        "key_size": key_size,
        "preload_keys": preload_keys,
        "profiling_enabled": False,
        "perf_stat_enabled": False,
        "avg_rps": score,
        "lscpu": "",
        "server_cpus": [],
    }
    if repetitions > 1:
        data["repetitions"] = repetitions
    if per_run_rps is not None:
        data["per_run_rps"] = per_run_rps
    if mean_rps is not None:
        data["mean_rps"] = mean_rps
    if ci_95 is not None:
        data["ci_95"] = ci_95

    return {
        "method": method,
        "source": source,
        "specifier": specifier,
        "commit_hash": commit_hash,
        "score": score,
        "end_time": "2025.01.15_12.00.00.000000",
        "data": data,
        "make_args": make_args,
        "features": {},
        "note": note,
    }


def _write_fixture(path: Path, records: list[dict]) -> None:
    """Write records as JSONL to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def fixture_file(tmp_path) -> Path:
    """Create a fixture output.jsonl with known benchmark results for two specifiers.

    Contains results for:
    - Two specifiers: branch-a and branch-b
    - Two methods: perf-get and perf-set
    - Multiple samples per group to enable t-test
    """
    records = [
        # perf-get, branch-a: 3 single-run samples
        _make_record(specifier="branch-a", method="perf-get", score=140000.0),
        _make_record(specifier="branch-a", method="perf-get", score=141000.0),
        _make_record(specifier="branch-a", method="perf-get", score=142000.0),
        # perf-get, branch-b: 3 single-run samples
        _make_record(specifier="branch-b", method="perf-get", score=145000.0),
        _make_record(specifier="branch-b", method="perf-get", score=146000.0),
        _make_record(specifier="branch-b", method="perf-get", score=147000.0),
        # perf-set, branch-a: 3 single-run samples
        _make_record(specifier="branch-a", method="perf-set", score=130000.0),
        _make_record(specifier="branch-a", method="perf-set", score=131000.0),
        _make_record(specifier="branch-a", method="perf-set", score=132000.0),
        # perf-set, branch-b: 3 single-run samples
        _make_record(specifier="branch-b", method="perf-set", score=135000.0),
        _make_record(specifier="branch-b", method="perf-set", score=136000.0),
        _make_record(specifier="branch-b", method="perf-set", score=137000.0),
    ]
    path = tmp_path / "output.jsonl"
    _write_fixture(path, records)
    return path


class TestAnalysisCompare:
    """Test AnalysisModule.compare() produces correct comparison rows."""

    def test_compare_produces_correct_rows(self, fixture_file):
        """compare() returns one row per parameter group with correct statistics."""
        module = AnalysisModule(results_path=fixture_file)
        rows = module.compare("branch-a", "branch-b")

        assert len(rows) == 2

        # Rows are sorted by group key; perf-get comes before perf-set
        get_row = next(r for r in rows if r.method == "perf-get")
        set_row = next(r for r in rows if r.method == "perf-set")

        # perf-get: mean_a = (140000+141000+142000)/3 = 141000
        assert get_row.mean_a == pytest.approx(141000.0)
        # perf-get: mean_b = (145000+146000+147000)/3 = 146000
        assert get_row.mean_b == pytest.approx(146000.0)
        assert get_row.n_a == 3
        assert get_row.n_b == 3

        # delta = (146000 - 141000) / 141000 * 100 = ~3.546%
        expected_delta = ((146000.0 - 141000.0) / 141000.0) * 100.0
        assert get_row.delta_pct == pytest.approx(expected_delta)

        # p-value should match scipy
        expected_p = ttest_ind(
            [140000.0, 141000.0, 142000.0],
            [145000.0, 146000.0, 147000.0],
            equal_var=False,
        ).pvalue
        assert get_row.p_value == pytest.approx(expected_p, rel=1e-9)

        # perf-set: mean_a = 131000, mean_b = 136000
        assert set_row.mean_a == pytest.approx(131000.0)
        assert set_row.mean_b == pytest.approx(136000.0)
        assert set_row.n_a == 3
        assert set_row.n_b == 3
        assert set_row.p_value is not None

    def test_compare_with_different_parameters(self, tmp_path):
        """Results with different io-threads form separate comparison groups."""
        records = [
            _make_record(
                specifier="a", method="perf-get", io_threads=1, score=100000.0
            ),
            _make_record(
                specifier="a", method="perf-get", io_threads=1, score=101000.0
            ),
            _make_record(
                specifier="b", method="perf-get", io_threads=1, score=110000.0
            ),
            _make_record(
                specifier="b", method="perf-get", io_threads=1, score=111000.0
            ),
            _make_record(
                specifier="a", method="perf-get", io_threads=9, score=800000.0
            ),
            _make_record(
                specifier="a", method="perf-get", io_threads=9, score=810000.0
            ),
            _make_record(
                specifier="b", method="perf-get", io_threads=9, score=900000.0
            ),
            _make_record(
                specifier="b", method="perf-get", io_threads=9, score=910000.0
            ),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 2
        io1_row = next(r for r in rows if r.io_threads == 1)
        io9_row = next(r for r in rows if r.io_threads == 9)

        assert io1_row.mean_a == pytest.approx(100500.0)
        assert io9_row.mean_a == pytest.approx(805000.0)


class TestAnalysisFormatTable:
    """Test AnalysisModule.format_table() produces correctly formatted output."""

    def test_format_table_contains_all_rows(self, fixture_file):
        """Formatted table contains header, separator, data rows, and summary."""
        module = AnalysisModule(results_path=fixture_file)
        rows = module.compare("branch-a", "branch-b")
        table = module.format_table(rows)

        lines = table.split("\n")
        # header + separator + 2 data rows + blank + summary lines
        assert len(lines) >= 4  # at minimum header, separator, 2 data rows
        # Verify both comparison rows are present
        assert "perf-get" in table
        assert "perf-set" in table
        assert "Comparisons: 2" in table

    def test_format_table_contains_expected_values(self, fixture_file):
        """Formatted table contains the method names and rps values."""
        module = AnalysisModule(results_path=fixture_file)
        rows = module.compare("branch-a", "branch-b")
        table = module.format_table(rows)

        assert "perf-get" in table
        assert "perf-set" in table
        assert "rps" in table
        assert "141,000 rps" in table
        assert "146,000 rps" in table

    def test_format_table_header_columns(self, fixture_file):
        """Table header contains all expected column names."""
        module = AnalysisModule(results_path=fixture_file)
        rows = module.compare("branch-a", "branch-b")
        table = module.format_table(rows)

        header = table.split("\n")[0]
        for col in [
            "Test",
            "Size",
            "Key",
            "IO",
            "Pipe",
            "Mean A",
            "Mean B",
            "Delta",
            "p-value",
        ]:
            assert col in header


class TestAnalysisCLIEntryPoint:
    """Test the CLI entry point src.analysis.main() with the fixture file."""

    def test_main_with_fixture_file(self, fixture_file, capsys):
        """main() prints a formatted comparison table to stdout."""
        with patch("src.analysis.AnalysisModule") as MockModule:
            # Use a real AnalysisModule pointed at the fixture
            real_module = AnalysisModule(results_path=fixture_file)
            MockModule.return_value = real_module
            # Patch results_path.exists() to return True on the default path
            real_module.results_path = fixture_file

            exit_code = main(["branch-a", "branch-b"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "perf-get" in captured.out
        assert "perf-set" in captured.out
        assert "rps" in captured.out

    def test_main_missing_results_file(self, tmp_path, capsys):
        """main() returns 1 when the results file does not exist."""
        nonexistent = tmp_path / "nonexistent.jsonl"
        with patch("src.analysis.AnalysisModule") as MockModule:
            real_module = AnalysisModule(results_path=nonexistent)
            MockModule.return_value = real_module

            exit_code = main(["branch-a", "branch-b"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No results file" in captured.err

    def test_main_no_matching_results(self, fixture_file, capsys):
        """main() returns 0 with a message when no results match the specifiers."""
        with patch("src.analysis.AnalysisModule") as MockModule:
            real_module = AnalysisModule(results_path=fixture_file)
            MockModule.return_value = real_module
            real_module.results_path = fixture_file

            exit_code = main(["nonexistent-a", "nonexistent-b"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No matching results" in captured.err

    def test_main_with_source_filter(self, fixture_file, capsys):
        """main() passes --source filter through to the analysis module."""
        with patch("src.analysis.AnalysisModule") as MockModule:
            real_module = AnalysisModule(results_path=fixture_file)
            MockModule.return_value = real_module
            real_module.results_path = fixture_file

            exit_code = main(["branch-a", "branch-b", "--source", "valkey"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "perf-get" in captured.out

    def test_main_with_method_filter(self, fixture_file, capsys):
        """main() passes --method filter through to the analysis module."""
        with patch("src.analysis.AnalysisModule") as MockModule:
            real_module = AnalysisModule(results_path=fixture_file)
            MockModule.return_value = real_module
            real_module.results_path = fixture_file

            exit_code = main(["branch-a", "branch-b", "--method", "perf-get"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "perf-get" in captured.out
        # perf-set should be filtered out
        assert "perf-set" not in captured.out


class TestAnalysisFiltering:
    """Test filtering by source and method in the analysis module."""

    def test_source_filter_excludes_other_sources(self, tmp_path):
        """Only results matching the source filter are included in comparison."""
        records = [
            _make_record(
                specifier="a", source="valkey", method="perf-get", score=100000.0
            ),
            _make_record(
                specifier="a", source="valkey", method="perf-get", score=101000.0
            ),
            _make_record(
                specifier="b", source="valkey", method="perf-get", score=110000.0
            ),
            _make_record(
                specifier="b", source="valkey", method="perf-get", score=111000.0
            ),
            _make_record(
                specifier="a", source="other-repo", method="perf-get", score=999999.0
            ),
            _make_record(
                specifier="a", source="other-repo", method="perf-get", score=999999.0
            ),
            _make_record(
                specifier="b", source="other-repo", method="perf-get", score=999999.0
            ),
            _make_record(
                specifier="b", source="other-repo", method="perf-get", score=999999.0
            ),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b", source_filter="valkey")

        assert len(rows) == 1
        row = rows[0]
        assert row.mean_a == pytest.approx(100500.0)
        assert row.mean_b == pytest.approx(110500.0)

    def test_method_filter_excludes_other_methods(self, tmp_path):
        """Only results matching the method filter are included in comparison."""
        records = [
            _make_record(specifier="a", method="perf-get", score=100000.0),
            _make_record(specifier="a", method="perf-get", score=101000.0),
            _make_record(specifier="b", method="perf-get", score=110000.0),
            _make_record(specifier="b", method="perf-get", score=111000.0),
            _make_record(specifier="a", method="perf-set", score=50000.0),
            _make_record(specifier="a", method="perf-set", score=51000.0),
            _make_record(specifier="b", method="perf-set", score=60000.0),
            _make_record(specifier="b", method="perf-set", score=61000.0),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b", method_filter="perf-get")

        assert len(rows) == 1
        assert rows[0].method == "perf-get"
        assert rows[0].mean_a == pytest.approx(100500.0)

    def test_combined_source_and_method_filter(self, tmp_path):
        """Both source and method filters are applied simultaneously."""
        records = [
            _make_record(
                specifier="a", source="valkey", method="perf-get", score=100000.0
            ),
            _make_record(
                specifier="a", source="valkey", method="perf-get", score=101000.0
            ),
            _make_record(
                specifier="b", source="valkey", method="perf-get", score=110000.0
            ),
            _make_record(
                specifier="b", source="valkey", method="perf-get", score=111000.0
            ),
            _make_record(
                specifier="a", source="valkey", method="perf-set", score=50000.0
            ),
            _make_record(
                specifier="b", source="valkey", method="perf-set", score=60000.0
            ),
            _make_record(
                specifier="a", source="other", method="perf-get", score=999999.0
            ),
            _make_record(
                specifier="b", source="other", method="perf-get", score=999999.0
            ),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare(
            "a", "b", source_filter="valkey", method_filter="perf-get"
        )

        assert len(rows) == 1
        assert rows[0].method == "perf-get"
        assert rows[0].mean_a == pytest.approx(100500.0)


class TestAnalysisAggregatedResults:
    """Test with aggregated results (repetitions > 1)."""

    def test_aggregated_results_use_per_run_rps(self, tmp_path):
        """Aggregated results expand per_run_rps into individual samples for comparison."""
        records = [
            _make_record(
                specifier="branch-a",
                method="perf-get",
                score=141000.0,
                repetitions=3,
                per_run_rps=[140000.0, 141000.0, 142000.0],
                mean_rps=141000.0,
                ci_95=500.0,
            ),
            _make_record(
                specifier="branch-b",
                method="perf-get",
                score=146000.0,
                repetitions=3,
                per_run_rps=[145000.0, 146000.0, 147000.0],
                mean_rps=146000.0,
                ci_95=500.0,
            ),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("branch-a", "branch-b")

        assert len(rows) == 1
        row = rows[0]
        assert row.n_a == 3
        assert row.n_b == 3
        assert row.mean_a == pytest.approx(141000.0)
        assert row.mean_b == pytest.approx(146000.0)
        assert row.p_value is not None

        # Verify p-value matches scipy using the per_run_rps samples
        expected_p = ttest_ind(
            [140000.0, 141000.0, 142000.0],
            [145000.0, 146000.0, 147000.0],
            equal_var=False,
        ).pvalue
        assert row.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_mixed_aggregated_and_single_run(self, tmp_path):
        """Mix of aggregated and single-run results for the same specifier."""
        records = [
            # branch-a: one aggregated result (3 samples) + one single-run
            _make_record(
                specifier="branch-a",
                method="perf-get",
                score=141000.0,
                repetitions=3,
                per_run_rps=[140000.0, 141000.0, 142000.0],
            ),
            _make_record(specifier="branch-a", method="perf-get", score=143000.0),
            # branch-b: two single-run results
            _make_record(specifier="branch-b", method="perf-get", score=150000.0),
            _make_record(specifier="branch-b", method="perf-get", score=151000.0),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("branch-a", "branch-b")

        assert len(rows) == 1
        row = rows[0]
        # branch-a: 3 from per_run_rps + 1 from single-run = 4 samples
        assert row.n_a == 4
        assert row.n_b == 2
        # mean_a = (140000 + 141000 + 142000 + 143000) / 4 = 141500
        assert row.mean_a == pytest.approx(141500.0)
        assert row.p_value is not None

    def test_single_aggregated_result_per_specifier_insufficient_for_ttest(
        self, tmp_path
    ):
        """A single aggregated result with repetitions > 1 provides enough samples for t-test."""
        records = [
            _make_record(
                specifier="a",
                method="perf-get",
                score=100000.0,
                repetitions=5,
                per_run_rps=[99000.0, 100000.0, 101000.0, 100500.0, 99500.0],
            ),
            _make_record(
                specifier="b",
                method="perf-get",
                score=110000.0,
                repetitions=5,
                per_run_rps=[109000.0, 110000.0, 111000.0, 110500.0, 109500.0],
            ),
        ]
        path = tmp_path / "output.jsonl"
        _write_fixture(path, records)

        module = AnalysisModule(results_path=path)
        rows = module.compare("a", "b")

        assert len(rows) == 1
        row = rows[0]
        assert row.n_a == 5
        assert row.n_b == 5
        assert row.p_value is not None
