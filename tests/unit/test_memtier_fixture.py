"""Tier 1: Test memtier JSON fixture parsing.

Validates that the real-format memtier --json-out-file fixture is parseable by
parse_memtier_json_intervals and contains the structural keys the scenario task
relies on.

Fixture provenance: real output captured from pinned memtier_benchmark on g4bench
(c8g.metal-48xl, Graviton 4) with --json-out-file --test-time 5 --hide-histogram.
Structure: ALL STATS -> Totals -> Time-Serie -> {"0": {"Count": N}, ...}
"""

import json
from pathlib import Path

import pytest

from conductress.tasks.task_scenario import parse_memtier_json_intervals, parse_memtier_stdout_intervals

FIXTURE_JSON_PATH = Path(__file__).parent.parent / "fixtures" / "memtier_real_output.json"
FIXTURE_STDOUT_PATH = Path(__file__).parent.parent / "fixtures" / "memtier_real_stdout.txt"


class TestMemtierFixtureParsing:
    """Validate the memtier JSON fixture is structurally correct for our parsers."""

    @pytest.fixture
    def fixture_content(self) -> str:
        return FIXTURE_JSON_PATH.read_text()

    @pytest.fixture
    def fixture_data(self, fixture_content: str) -> dict:
        return json.loads(fixture_content)

    def test_fixture_file_exists(self):
        """Fixture file must exist for CI reproducibility."""
        assert FIXTURE_JSON_PATH.exists(), f"Fixture not found at {FIXTURE_JSON_PATH}"

    def test_fixture_is_valid_json(self, fixture_content: str):
        """Fixture must be valid JSON."""
        data = json.loads(fixture_content)
        assert isinstance(data, dict)

    def test_all_stats_key_exists(self, fixture_data: dict):
        """parse_memtier_json_intervals depends on top-level 'ALL STATS' key."""
        assert "ALL STATS" in fixture_data, "Missing 'ALL STATS' key -- parser will return empty list"

    def test_totals_has_time_serie(self, fixture_data: dict):
        """Totals section must contain Time-Serie with per-second interval data."""
        all_stats = fixture_data["ALL STATS"]
        assert "Totals" in all_stats, "Missing 'Totals' entry"
        totals = all_stats["Totals"]
        assert (
            "Time-Serie" in totals
        ), f"Missing 'Time-Serie' in Totals -- parser depends on this. Keys: {list(totals.keys())}"
        time_serie = totals["Time-Serie"]
        assert len(time_serie) > 0, "Time-Serie is empty"

    def test_time_serie_entries_have_count(self, fixture_data: dict):
        """Each Time-Serie entry must have 'Count' -- parser uses this as ops-per-second."""
        time_serie = fixture_data["ALL STATS"]["Totals"]["Time-Serie"]
        for key, entry in time_serie.items():
            assert key.isdigit(), f"Time-Serie key '{key}' is not a numeric second index"
            assert "Count" in entry, f"Missing 'Count' in Time-Serie[{key}]"
            assert isinstance(entry["Count"], (int, float)), f"Count in [{key}] is not numeric"

    def test_totals_has_ops_sec(self, fixture_data: dict):
        """Totals Ops/sec used by parse_memtier_total_rps (task_mixed.py) for aggregate RPS."""
        totals = fixture_data["ALL STATS"]["Totals"]
        assert "Ops/sec" in totals, "Missing 'Ops/sec' in Totals"
        assert totals["Ops/sec"] > 0, "Totals Ops/sec must be positive"

    def test_parse_returns_nonempty_list(self, fixture_content: str):
        """parse_memtier_json_intervals must return non-empty list of floats."""
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        assert len(result) > 0, "Parser returned empty list from valid fixture"

    def test_parse_returns_plausible_rps_values(self, fixture_content: str):
        """All parsed interval values must be positive (plausible ops-per-second counts)."""
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        for val in result:
            assert isinstance(val, float), f"Expected float, got {type(val)}"
            assert val > 0, f"Ops count {val} is not positive"

    def test_parse_count_matches_run_seconds(self, fixture_content: str):
        """Number of parsed intervals should approximate run duration (±2 seconds).

        Fixture is a 5-second run, so we expect 3-7 intervals (last partial excluded).
        """
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        expected_seconds = 5
        assert (
            abs(len(result) - expected_seconds) <= 2
        ), f"Expected ~{expected_seconds} intervals (±2), got {len(result)}"

    def test_time_serie_keys_sort_numerically(self, fixture_data: dict):
        """Parser relies on sorted(..., key=int) producing chronological order."""
        time_serie = fixture_data["ALL STATS"]["Totals"]["Time-Serie"]
        keys = list(time_serie.keys())
        sorted_keys = sorted(keys, key=lambda k: int(k))
        nums = [int(k) for k in sorted_keys]
        assert nums == sorted(nums), "Time-Serie keys don't sort numerically correctly"


class TestMemtierStdoutFixture:
    """Validate the memtier stdout fixture is parseable for progress-line fallback."""

    def test_stdout_fixture_exists(self):
        """Stdout fixture must exist."""
        assert FIXTURE_STDOUT_PATH.exists(), f"Stdout fixture not found at {FIXTURE_STDOUT_PATH}"

    def test_stdout_fixture_parseable(self):
        """parse_memtier_stdout_intervals must extract values from real stdout."""
        content = FIXTURE_STDOUT_PATH.read_text()
        result = parse_memtier_stdout_intervals(content)
        assert len(result) >= 3, f"Expected at least 3 progress intervals, got {len(result)}"
        for val in result:
            assert val > 0, f"Parsed ops/sec not positive: {val}"
