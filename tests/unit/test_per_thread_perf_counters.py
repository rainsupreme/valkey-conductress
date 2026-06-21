"""Unit tests for per-thread (main vs IO) perf-stat hardware counters.

Covers the four layers added by the feature:
  1. Collection — per-TID parsing + main/io/all bucketing (profiling_manager)
  2. Target selection — `-t TID,...` (race-free) vs `-p PID` fallback
  3. Persistence — BenchmarkPoint per-thread fields survive save -> load
  4. Export — per-thread series variants + per-thread manifest groups
"""

import json
import tempfile
from pathlib import Path

import pytest

from conductress.profiling_manager import ProfilingManager
from conductress.sweep.exporter import PER_THREAD_PERF_GROUPS, export_manifest, export_perf_metrics
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# Realistic `perf stat --per-thread` output. Note `valkey-server-100` has a
# hyphen in the comm name (TID must be recovered by right-split), and there are
# `<not counted>` / `<not supported>` rows plus a background bio_ thread that is
# neither main nor IO (must land in `all` only).
PER_THREAD_OUTPUT = """\
 Performance counter stats for process id '100':

   valkey-server-100     14,000,000,000      instructions      #    2.80  insn per cycle    (58.55%)
  bio_close_file-105        <not counted>      instructions
        io_thd_1-101     20,000,000,000      instructions      #    4.00  insn per cycle    (58.55%)
        io_thd_2-102     26,000,000,000      instructions      #    4.33  insn per cycle    (58.55%)
   valkey-server-100      5,000,000,000      cycles                                          (58.05%)
        io_thd_1-101      5,000,000,000      cycles                                          (58.05%)
        io_thd_2-102     10,000,000,000      cycles                                          (58.05%)
   valkey-server-100         <not supported>  LLC-load-misses

 2.001 seconds time elapsed
"""


class TestParsePerThread:
    def _write(self, tmp_path, text):
        p = tmp_path / "perf_stat.txt"
        p.write_text(text)
        return p

    def test_buckets_main_io_all(self, tmp_path):
        path = self._write(tmp_path, PER_THREAD_OUTPUT)
        res = ProfilingManager.parse_perf_stat_per_thread(path, "100", ["101", "102"])

        # main = the valkey-server-100 rows only
        assert res["main"]["instructions"] == 14_000_000_000
        assert res["main"]["cycles"] == 5_000_000_000
        # io = sum of io_thd_1 + io_thd_2
        assert res["io"]["instructions"] == 46_000_000_000
        assert res["io"]["cycles"] == 15_000_000_000
        # all = every monitored TID summed (main + io; bio_ row is <not counted>)
        assert res["all"]["instructions"] == 60_000_000_000
        assert res["all"]["cycles"] == 20_000_000_000

    def test_not_counted_and_not_supported_skipped(self, tmp_path):
        path = self._write(tmp_path, PER_THREAD_OUTPUT)
        res = ProfilingManager.parse_perf_stat_per_thread(path, "100", ["101", "102"])
        # LLC-load-misses was <not supported> on the only row that had it
        assert "LLC-load-misses" not in res["all"]
        assert "LLC-load-misses" not in res["main"]

    def test_multi_hyphen_comm_tid_recovery(self, tmp_path):
        # comm "valkey-server" contains a hyphen; TID must be the trailing digits
        path = self._write(
            tmp_path,
            "   valkey-server-100     1,000      instructions\n",
        )
        res = ProfilingManager.parse_perf_stat_per_thread(path, "100", [])
        assert res["main"]["instructions"] == 1000
        assert res["all"]["instructions"] == 1000

    def test_missing_file_returns_empty_buckets(self, tmp_path):
        res = ProfilingManager.parse_perf_stat_per_thread(tmp_path / "nope.txt", "100", ["101"])
        assert res == {"all": {}, "main": {}, "io": {}}

    def test_no_io_tids_leaves_io_empty(self, tmp_path):
        path = self._write(tmp_path, PER_THREAD_OUTPUT)
        res = ProfilingManager.parse_perf_stat_per_thread(path, "100", [])
        assert res["io"] == {}
        assert res["main"]["instructions"] == 14_000_000_000
        # io thread rows still counted toward the process-wide total
        assert res["all"]["instructions"] == 60_000_000_000


class TestPerfStatTarget:
    def _mgr(self):
        return ProfilingManager(host=object())  # host unused for target building

    def test_explicit_tid_list_when_io_present(self):
        m = self._mgr()
        m.target_pid = 100
        m._main_tid = "100"
        m._io_tids = ["101", "102"]
        assert m._build_perf_stat_target() == "-t 100,101,102"

    def test_falls_back_to_pid_when_no_tids(self):
        m = self._mgr()
        m.target_pid = 100
        m._main_tid = None
        m._io_tids = []
        assert m._build_perf_stat_target() == "-p 100"

    def test_main_only_still_uses_tid_form(self):
        m = self._mgr()
        m.target_pid = 100
        m._main_tid = "100"
        m._io_tids = []
        assert m._build_perf_stat_target() == "-t 100"


class TestPersistenceRoundTrip:
    def test_per_thread_counters_survive_save_load(self, tmp_path):
        path = tmp_path / "state.json"
        state = SweepState()
        state.merge_commits = ["abc123"]
        p = BenchmarkPoint(commit="abc123", date="2026-06-20", value=2_000_000.0, status=PointStatus.COMPLETED)
        p.perf_counters = {"instructions": 60_000_000_000, "cycles": 20_000_000_000}
        p.perf_counters_main = {"instructions": 14_000_000_000, "cycles": 5_000_000_000}
        p.perf_counters_io = {"instructions": 46_000_000_000, "cycles": 15_000_000_000}
        state.points["abc123"] = p
        state.save(path)

        # confirm it's actually serialized to disk (not just in-memory)
        raw = json.loads(path.read_text())
        assert raw["points"]["abc123"]["perf_counters_main"]["instructions"] == 14_000_000_000
        assert raw["points"]["abc123"]["perf_counters_io"]["cycles"] == 15_000_000_000

        loaded = SweepState.load(path)
        lp = loaded.points["abc123"]
        assert lp.perf_counters_main == {"instructions": 14_000_000_000, "cycles": 5_000_000_000}
        assert lp.perf_counters_io == {"instructions": 46_000_000_000, "cycles": 15_000_000_000}

    def test_legacy_state_without_per_thread_fields_loads(self, tmp_path):
        # Old state files won't have the new keys — load must default them to None
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "points": {
                        "old1": {
                            "commit": "old1",
                            "date": "2026-01-01",
                            "value": 1000.0,
                            "perf_counters": {"instructions": 5},
                            "status": "COMPLETED",
                        }
                    }
                }
            )
        )
        loaded = SweepState.load(path)
        op = loaded.points["old1"]
        assert op.perf_counters == {"instructions": 5}
        assert op.perf_counters_main is None
        assert op.perf_counters_io is None


class TestExportPerThreadSeries:
    def _state_with_point(self):
        state = SweepState()
        state.merge_commits = ["abc123"]
        p = BenchmarkPoint(commit="abc123", date="2026-06-20", value=2_000_000.0, status=PointStatus.COMPLETED)
        p.perf_rps = 2_000_000.0
        p.perf_duration_seconds = 30.0
        p.perf_counters = {"instructions": 60_000_000_000, "cycles": 20_000_000_000}
        p.perf_counters_main = {"instructions": 14_000_000_000, "cycles": 5_000_000_000}
        p.perf_counters_io = {"instructions": 46_000_000_000, "cycles": 15_000_000_000}
        state.points["abc123"] = p
        return state

    def test_emits_process_wide_and_per_thread_variants(self, tmp_path):
        state = self._state_with_point()
        exported = export_perf_metrics(state, tmp_path, platform="arm64", workload="get-k16-v16-t7-p10")
        for key in ("ipc", "ipc-main", "ipc-io"):
            assert key in exported, f"missing {key}"
            assert (tmp_path / f"series-arm64-get-k16-v16-t7-p10-{key}.json").exists()

    def test_per_thread_ipc_is_counts_then_ratio(self, tmp_path):
        state = self._state_with_point()
        export_perf_metrics(state, tmp_path, platform="arm64", workload="wl")
        main = json.loads((tmp_path / "series-arm64-wl-ipc-main.json").read_text())["points"][0]["value"]
        io = json.loads((tmp_path / "series-arm64-wl-ipc-io.json").read_text())["points"][0]["value"]
        # 14e9/5e9 = 2.8, 46e9/15e9 = 3.0667 — derived from summed counts, not avg of ratios
        assert main == pytest.approx(2.8)
        assert io == pytest.approx(46 / 15)

    def test_point_without_per_thread_data_skips_variants(self, tmp_path):
        state = SweepState()
        state.merge_commits = ["c"]
        p = BenchmarkPoint(commit="c", date="2026-06-20", value=1.0, status=PointStatus.COMPLETED)
        p.perf_rps = 1.0
        p.perf_duration_seconds = 30.0
        p.perf_counters = {"instructions": 10, "cycles": 5}  # process-wide only
        state.points["c"] = p
        exported = export_perf_metrics(state, tmp_path, platform="arm64", workload="wl")
        assert "ipc" in exported
        assert "ipc-main" not in exported  # no per-thread data -> no variant file
        assert not (tmp_path / "series-arm64-wl-ipc-main.json").exists()


class TestPerThreadManifestGroups:
    def test_groups_generated_for_each_base(self):
        gids = {g["id"] for g in PER_THREAD_PERF_GROUPS}
        for base in ("efficiency", "cache", "pipeline", "tma", "branching"):
            assert f"{base}-main" in gids
            assert f"{base}-io" in gids

    def test_no_per_thread_variant_for_cpu_groups(self):
        gids = {g["id"] for g in PER_THREAD_PERF_GROUPS}
        # cpu-main/cpu-io are already thread-specific flamegraphs — not re-split
        assert "cpu-main-main" not in gids
        assert "cpu-io-io" not in gids

    def test_group_series_reference_suffixed_metrics(self):
        eff_main = next(g for g in PER_THREAD_PERF_GROUPS if g["id"] == "efficiency-main")
        assert eff_main["series"] == ["ipc-main", "instructions-per-req-main"]
        # y-axis series are suffixed too
        left = next(a for a in eff_main["y_axes"] if a["id"] == "left")
        assert left["series"] == ["ipc-main"]
        assert eff_main["per_thread"] == "main"

    def test_manifest_includes_per_thread_groups(self, tmp_path, monkeypatch):
        # export_manifest detects platform via publisher.detect_platform
        import conductress.sweep.exporter as exporter_mod

        monkeypatch.setattr(exporter_mod, "detect_platform", lambda: ("arm64", None), raising=False)
        # detect_platform is imported inside the function from conductress.publisher
        import conductress.publisher as publisher_mod

        monkeypatch.setattr(publisher_mod, "detect_platform", lambda: ("arm64", None))
        export_manifest(tmp_path, platforms=["arm64"], workloads=[("get-k16-v16-t7-p10", "throughput")])
        manifest = json.loads((tmp_path / "manifest-arm64.json").read_text())
        gids = {g["id"] for g in manifest["groups"]}
        assert "efficiency-main" in gids
        assert "tma-io" in gids
        # process-wide groups still present
        assert "efficiency" in gids and "tma" in gids and "cpu-main" in gids
