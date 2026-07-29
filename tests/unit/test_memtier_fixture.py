"""Tier 1: Test memtier JSON fixture parsing.

Validates that the real-format memtier --json-out-file fixture is parseable by
parse_memtier_json_intervals and contains the structural keys the scenario task
relies on.

Fixture provenance: synthetic data matching memtier_benchmark d52544b1 output
format (--json-out-file with --test-time 5). Structure verified against the
parser in task_scenario.py:parse_memtier_json_intervals.
"""

import json
from pathlib import Path

import pytest

from conductress.tasks.task_scenario import parse_memtier_json_intervals

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "memtier_scenario_output.json"


class TestMemtierFixtureParsing:
    """Validate the memtier JSON fixture is structurally correct for our parsers."""

    @pytest.fixture
    def fixture_content(self) -> str:
        return FIXTURE_PATH.read_text()

    @pytest.fixture
    def fixture_data(self, fixture_content: str) -> dict:
        return json.loads(fixture_content)

    def test_fixture_file_exists(self):
        """Fixture file must exist for CI reproducibility."""
        assert FIXTURE_PATH.exists(), f"Fixture not found at {FIXTURE_PATH}"

    def test_fixture_is_valid_json(self, fixture_content: str):
        """Fixture must be valid JSON."""
        data = json.loads(fixture_content)
        assert isinstance(data, dict)

    def test_all_stats_key_exists(self, fixture_data: dict):
        """parse_memtier_json_intervals depends on top-level 'ALL STATS' key."""
        assert "ALL STATS" in fixture_data, "Missing 'ALL STATS' key -- parser will return empty list"

    def test_interval_entries_have_ops_sec(self, fixture_data: dict):
        """Each 'Second N' interval must have 'Ops/sec' -- parser extracts this field."""
        all_stats = fixture_data["ALL STATS"]
        second_keys = [k for k in all_stats if k.startswith("Second ")]
        assert len(second_keys) > 0, "No 'Second N' interval entries found"
        for key in second_keys:
            assert "Ops/sec" in all_stats[key], f"Missing 'Ops/sec' in {key}"
            assert isinstance(all_stats[key]["Ops/sec"], (int, float)), f"'Ops/sec' in {key} is not numeric"

    def test_totals_entry_exists(self, fixture_data: dict):
        """Totals entry used by parse_memtier_total_rps (task_mixed.py)."""
        all_stats = fixture_data["ALL STATS"]
        assert "Totals" in all_stats, "Missing 'Totals' entry"
        assert "Ops/sec" in all_stats["Totals"], "Missing 'Ops/sec' in Totals"

    def test_parse_returns_nonempty_list(self, fixture_content: str):
        """parse_memtier_json_intervals must return non-empty list of floats."""
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        assert len(result) > 0, "Parser returned empty list from valid fixture"

    def test_parse_returns_plausible_rps_values(self, fixture_content: str):
        """All parsed interval values must be positive (plausible RPS)."""
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        for val in result:
            assert isinstance(val, float), f"Expected float, got {type(val)}"
            assert val > 0, f"RPS value {val} is not positive"

    def test_parse_count_matches_run_seconds(self, fixture_content: str):
        """Number of parsed intervals should approximate run duration (±2 seconds).

        Fixture is a 5-second run, so we expect 3-7 intervals.
        """
        result = parse_memtier_json_intervals("/tmp/fixture.json", fixture_content)
        expected_seconds = 5
        assert (
            abs(len(result) - expected_seconds) <= 2
        ), f"Expected ~{expected_seconds} intervals (±2), got {len(result)}"

    def test_interval_keys_are_sorted_numerically(self, fixture_data: dict):
        """Parser relies on sorted() of 'Second N' keys producing chronological order."""
        all_stats = fixture_data["ALL STATS"]
        second_keys = sorted(k for k in all_stats if k.startswith("Second "))
        # Verify sorted order matches numeric order
        nums = [int(k.split()[1]) for k in second_keys]
        assert nums == sorted(nums), "Second keys don't sort numerically correctly"
