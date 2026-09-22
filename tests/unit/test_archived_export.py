"""Unit tests for the one-time archived-epoch exporter.

The runner no longer builds or loads v1 coordinators (v1 is archived), so a
runner whose publish export dir was wiped has no v1 files and nothing to
regenerate them. ``conductress sweep export-archived --epoch v1`` is the
one-time repair: it loads the v1 state files once, writes the v1 dashboard
files into the export dir, and exits. These tests pin that behaviour and its
guards.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.sweep.archived_export import export_archived_epoch
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_v1_primary_state(state_dir: Path) -> None:
    """Write a minimal completed-point state file for the v1 primary GET series."""
    state_dir.mkdir(parents=True, exist_ok=True)
    state = SweepState(merge_commits=["a"], commit_dates={"a": "2024-01-01"})
    state.points["a"] = BenchmarkPoint(
        commit="a", date="2024-01-01", value=2_000_000, cv=0.2, reps=5, status=PointStatus.COMPLETED
    )
    # The primary GET coordinator's label is get-k16-v16-t7-p10.
    state.save(state_dir / "state_get-k16-v16-t7-p10.json")


class TestExportArchivedEpoch:
    def test_rejects_non_archived_epoch(self, tmp_path):
        with patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ("v1",)):
            with pytest.raises(ValueError, match="not archived"):
                export_archived_epoch("v3", repo_path=tmp_path, output_dir=tmp_path)

    def test_rejects_unsupported_archived_epoch(self, tmp_path):
        with patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ("v2",)):
            with pytest.raises(ValueError, match="only 'v1'"):
                export_archived_epoch("v2", repo_path=tmp_path, output_dir=tmp_path)

    def test_returns_zero_and_writes_nothing_when_no_state(self, tmp_path):
        """Tolerates a wiped/empty state dir: logs and exports nothing."""
        state_dir = tmp_path / "sweep_data"
        out_dir = tmp_path / "export"
        with (
            patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ("v1",)),
            patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", state_dir),
            patch("conductress.sweep.memory_coordinator.MEMORY_STATE_DIR", state_dir),
            patch("conductress.sweep.latency_coordinator.LATENCY_STATE_FILE", state_dir / "latency_state.json"),
            patch("conductress.config.SWEEP_ENGINES", []),
            patch("conductress.publisher.detect_platform", return_value=("graviton4", "Graviton 4")),
        ):
            written = export_archived_epoch("v1", repo_path=tmp_path / "valkey", output_dir=out_dir)
        assert written == 0
        assert not list(out_dir.glob("*.json"))

    def test_exports_present_v1_series_and_manifest(self, tmp_path):
        state_dir = tmp_path / "sweep_data"
        out_dir = tmp_path / "export"
        _write_v1_primary_state(state_dir)
        with (
            patch("conductress.config.SWEEP_ARCHIVED_EPOCHS", ("v1",)),
            patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", state_dir),
            patch("conductress.sweep.memory_coordinator.MEMORY_STATE_DIR", state_dir),
            patch("conductress.sweep.latency_coordinator.LATENCY_STATE_FILE", state_dir / "latency_state.json"),
            patch("conductress.config.SWEEP_ENGINES", []),
            patch("conductress.publisher.detect_platform", return_value=("graviton4", "Graviton 4")),
        ):
            written = export_archived_epoch("v1", repo_path=tmp_path / "valkey", output_dir=out_dir)

        assert written >= 1
        # The primary GET series is written under its legacy unqualified name.
        series = out_dir / "series-graviton4-get-k16-v16-t7-p10-throughput.json"
        assert series.exists()
        # The v1 manifest keeps its legacy name and advertises the same epoch
        # list a live publish writes: live epochs first, then archived, each
        # flagged, so the dashboard defaults to a live epoch from this file too.
        manifest = out_dir / "manifest-graviton4.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["epoch"] == "v1"
        assert [e["id"] for e in data["epochs"]] == ["v3", "v1"]
        assert {e["id"]: e["archived"] for e in data["epochs"]} == {"v3": False, "v1": True}
