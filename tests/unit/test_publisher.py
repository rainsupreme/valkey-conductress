"""Tests for the dashboard publisher."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress.publisher import DashboardPublisher, detect_platform


class TestDetectPlatform:
    @pytest.mark.parametrize(
        ("platform_info", "expected_id", "label_fragment"),
        [
            (("arm64", "arm64/c7g.metal/graviton3", ["graviton3", "arm64"]), "arm64", "graviton3"),
            (("graviton4", "arm64/c8g.metal/graviton4", ["graviton4"]), "graviton4", "graviton4"),
            (("amd64", "amd64/epyc-9r14/zen4", ["amd64", "amd"]), "amd64", "zen4"),
            (("intel", "intel/xeon-8488c/sapphire-rapids", ["intel"]), "intel", "sapphire"),
        ],
    )
    def test_shared_platform_detection(self, platform_info, expected_id, label_fragment):
        with patch("conductress.platform.get_local_platform_info", return_value=platform_info):
            platform_id, label = detect_platform()

        assert platform_id == expected_id
        assert label_fragment in label


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Keep every publisher under test off the real export dir and temp dirs."""
    from conductress import config as _config

    monkeypatch.setattr(_config, "PUBLISH_EXPORT_DIR", tmp_path / "publish-export")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "var-tmp").mkdir()
    monkeypatch.setattr(
        DashboardPublisher, "_legacy_temp_dirs", staticmethod(lambda: [tmp_path / "tmp", tmp_path / "var-tmp"])
    )


class TestDashboardPublisher:
    def test_init(self, tmp_path):
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        pub = DashboardPublisher("user@host:/path", [coord])
        assert pub.target == "user@host:/path"
        assert pub.coordinators == [coord]
        assert pub._export_dir == tmp_path / "publish-export"
        assert pub._export_dir.is_dir()

    def test_export_dir_is_fixed_and_kept_across_restarts(self, tmp_path):
        """A second publisher reuses the same path and keeps what the first left.

        The per-commit cpu-stacks files are exported once and skipped by mtime
        afterwards; wiping the directory on start-up would regenerate all of
        them with fresh mtimes and make the next rsync move the whole tree.
        Only staging directories from an interrupted publish are removed.
        """
        export_dir = tmp_path / "publish-export"
        export_dir.mkdir()
        stacks = export_dir / "series-arm64-get-k16-v16-t7-p10-cpu-stacks-aaa.epoch-v3.json"
        stacks.write_text("{}")
        stale_stage = export_dir / ".stage-v3-get-k16-v16-t7-p10"
        stale_stage.mkdir()
        (stale_stage / "series-x.json").write_text("{}")
        first = DashboardPublisher("user@host:/path", [])
        assert first._export_dir == export_dir
        assert stacks.exists()
        assert not stale_stage.exists()
        second = DashboardPublisher("user@host:/path", [])
        assert second._export_dir == export_dir
        assert stacks.exists()
        assert not str(export_dir).startswith(str(tmp_path / "tmp")), "export dir must not live under tempdir"

    def test_legacy_mkdtemp_export_dirs_are_swept_on_init(self, tmp_path):
        """Directories left by the retired mkdtemp publisher are removed from every temp dir; nothing else is touched."""
        tmp = tmp_path / "tmp"
        var_tmp = tmp_path / "var-tmp"
        for name in ("conductress-publish-abc123", "conductress-publish-def456"):
            (tmp / name).mkdir()
            (tmp / name / "series-x.json").write_text("{}")
        (var_tmp / "conductress-publish-fallback").mkdir()
        (tmp / "conductress-publish-not-a-dir").write_text("")
        (tmp / "unrelated-dir").mkdir()
        (tmp / "unrelated-dir" / "keep.txt").write_text("keep")

        DashboardPublisher("user@host:/path", [])

        assert not (tmp / "conductress-publish-abc123").exists()
        assert not (tmp / "conductress-publish-def456").exists()
        assert not (var_tmp / "conductress-publish-fallback").exists()
        assert (tmp / "conductress-publish-not-a-dir").exists()
        assert (tmp / "unrelated-dir" / "keep.txt").read_text() == "keep"

    def test_legacy_temp_dirs_cover_tmp_and_var_tmp(self, monkeypatch):
        """gettempdir() moves to /var/tmp once /tmp is full, so both are always candidates, deduplicated."""
        monkeypatch.undo()  # drop the autouse patch of _legacy_temp_dirs for this test
        monkeypatch.setattr("conductress.publisher.tempfile.gettempdir", lambda: "/var/tmp")
        dirs = DashboardPublisher._legacy_temp_dirs()
        assert dirs == [Path("/var/tmp"), Path("/tmp")]

    def test_on_task_failed_is_noop(self):
        pub = DashboardPublisher("user@host:/path", [])
        pub.on_task_failed(MagicMock())  # should not raise

    def test_on_queue_empty_is_noop(self):
        pub = DashboardPublisher("user@host:/path", [])
        pub.on_queue_empty()  # should not raise

    @patch("conductress.utility.subprocess.run")
    def test_publish_calls_rsync_in_two_passes_dashboard_files_first(self, mock_run):
        """Dashboard files sync first, then the per-commit stacks, each pass bounded.

        rsync sends in name order, so a single pass with ~10 GB of stacks files
        can hit its timeout before reaching series files that sort after them.
        """
        from conductress import config as _config

        mock_run.return_value = MagicMock(returncode=0)
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        coord.state = MagicMock()
        coord.export.return_value = 5

        pub = DashboardPublisher("user@host:/path", [coord])

        with patch("conductress.sweep.exporter.export_perf_metrics", return_value={}):
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())

        assert mock_run.call_count == 2
        first, second = (c[0][0] for c in mock_run.call_args_list)
        assert first[0] == "rsync" and second[0] == "rsync"
        assert first[-1] == "user@host:/path" and second[-1] == "user@host:/path"
        assert "--exclude=series-*-cpu-stacks-[0-9a-f]*.json" in first
        assert "--include=series-*-cpu-stacks-[0-9a-f]*.json" in second
        assert second.index("--include=series-*-cpu-stacks-[0-9a-f]*.json") < second.index("--exclude=*")
        for call in mock_run.call_args_list:
            assert call[1]["timeout"] == _config.PUBLISH_RSYNC_TIMEOUT_SECONDS

    def test_stacks_glob_matches_per_commit_files_but_not_the_index(self):
        """The filter must leave cpu-stacks-index.json, which the dashboard reads, in pass one."""
        import fnmatch

        from conductress.publisher import STACKS_FILE_GLOB

        commit = "15aa872383dc34a20cc99a9fa4aca0525a747a03"
        assert fnmatch.fnmatchcase(f"series-arm64-get-k16-v16-t7-p10-cpu-stacks-{commit}.json", STACKS_FILE_GLOB)
        assert fnmatch.fnmatchcase(
            f"series-arm64-get-k16-v16-t7-p10-cpu-stacks-{commit}.epoch-v3.json", STACKS_FILE_GLOB
        )
        assert not fnmatch.fnmatchcase("series-arm64-get-k16-v16-t7-p10-cpu-stacks-index.json", STACKS_FILE_GLOB)
        assert not fnmatch.fnmatchcase(
            "series-arm64-get-k16-v16-t7-p10-cpu-stacks-index.epoch-v3.json", STACKS_FILE_GLOB
        )
        assert not fnmatch.fnmatchcase("series-arm64-get-k16-v16-t7-p10-cpu-main.epoch-v3.json", STACKS_FILE_GLOB)
        assert not fnmatch.fnmatchcase("series-arm64-get-k16-v16-t7-p10-throughput.epoch-v3.json", STACKS_FILE_GLOB)
        assert not fnmatch.fnmatchcase("manifest-arm64.epoch-v3.json", STACKS_FILE_GLOB)

    @patch("conductress.utility.subprocess.run")
    def test_publish_failure_does_not_raise(self, mock_run):
        """Publish failures are non-fatal."""
        mock_run.side_effect = Exception("network error")
        coord = MagicMock()
        coord.workload_id = "get16b-t7-p10"
        coord.metric_id = "throughput"
        coord.state = MagicMock()
        coord.export.return_value = 5

        pub = DashboardPublisher("user@host:/path", [coord])

        with patch("conductress.sweep.exporter.export_perf_metrics", return_value={}):
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())  # should not raise

    @patch("conductress.utility.subprocess.run")
    def test_perf_metrics_exported_for_all_throughput_coordinators(self, mock_run):
        """Regression: perf metrics must export for every throughput coordinator, not just the first."""
        mock_run.return_value = MagicMock(returncode=0)
        coord_16b = MagicMock()
        coord_16b.workload_id = "get-k16-v16-t7-p10"
        coord_16b.metric_id = "throughput"
        coord_16b.state = MagicMock()
        coord_16b.export.return_value = 5

        coord_64b = MagicMock()
        coord_64b.workload_id = "get-k16-v64-t7-p10"
        coord_64b.metric_id = "throughput"
        coord_64b.state = MagicMock()
        coord_64b.export.return_value = 2

        pub = DashboardPublisher("user@host:/path", [coord_16b, coord_64b])

        with patch("conductress.sweep.exporter.export_perf_metrics") as mock_perf:
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())

        # Both coordinators must have perf metrics exported
        assert mock_perf.call_count == 2
        workload_ids = [call.args[3] for call in mock_perf.call_args_list]
        assert "get-k16-v16-t7-p10" in workload_ids
        assert "get-k16-v64-t7-p10" in workload_ids

    @patch("conductress.utility.subprocess.run")
    def test_perf_metrics_exported_for_latency_but_not_memory(self, mock_run):
        """Rate-limited latency cells collect the same counters as throughput cells
        and must publish the same per-request series; memory cells collect none."""
        mock_run.return_value = MagicMock(returncode=0)

        def make_coord(workload_id, metric_id):
            coord = MagicMock()
            coord.workload_id = workload_id
            coord.metric_id = metric_id
            coord.state = MagicMock()
            coord.export.return_value = 1
            return coord

        pub = DashboardPublisher(
            "user@host:/path",
            [
                make_coord("get-k16-v16-t7-p10", "throughput"),
                make_coord("get-k16-v16-t7-p1-r100k", "latency"),
                make_coord("memory-set-k16-v64", "memory"),
            ],
        )

        with patch("conductress.sweep.exporter.export_perf_metrics") as mock_perf:
            with patch("conductress.sweep.exporter.export_manifest"):
                pub.on_task_completed(MagicMock())

        workload_ids = [call.args[3] for call in mock_perf.call_args_list]
        assert workload_ids == ["get-k16-v16-t7-p10", "get-k16-v16-t7-p1-r100k"]

    @patch("conductress.utility.subprocess.run")
    def test_notable_export_includes_valkey_throughput_and_memory_only(self, mock_run):
        """Notable feed aggregates Valkey throughput+memory series; Redis and latency are excluded."""
        mock_run.return_value = MagicMock(returncode=0)

        def make_coord(workload_id, metric_id, engine):
            coord = MagicMock()
            coord.workload_id = workload_id
            coord.metric_id = metric_id
            coord.engine = engine
            coord.lower_is_better = metric_id != "throughput"
            coord.state = MagicMock()
            coord.export.return_value = 1
            return coord

        valkey_engine = MagicMock()
        valkey_engine.source = "valkey"
        redis_engine = MagicMock()
        redis_engine.source = "redis"

        coords = [
            make_coord("get-k16-v16-t7-p10", "throughput", valkey_engine),
            make_coord("set-m20", "memory", None),  # legacy state, no engine -> Valkey
            make_coord("redis-get-k16-v16-t7-p10", "throughput", redis_engine),
            make_coord("get-lat", "latency", valkey_engine),
        ]
        pub = DashboardPublisher("user@host:/path", coords)

        with patch("conductress.sweep.exporter.export_perf_metrics"):
            with patch("conductress.sweep.exporter.export_manifest"):
                with patch("conductress.sweep.exporter.export_notable") as mock_notable:
                    pub.on_task_completed(MagicMock())

        mock_notable.assert_called_once()
        sources = mock_notable.call_args.args[0]
        included = {(s.workload, s.metric) for s in sources}
        assert included == {("get-k16-v16-t7-p10", "throughput"), ("set-m20", "memory")}
        # Output filename is platform-scoped
        output_path = mock_notable.call_args.args[1]
        assert output_path.name == f"notable-{pub._platform_id}.json"


class TestEpochPublishing:
    def test_v1_path_is_unchanged(self):
        path = Path("series-arm64-get-k16-v16-t7-p10-throughput.json")
        assert DashboardPublisher._epoch_path(path, "v1") == path

    def test_v2_path_is_epoch_qualified(self):
        path = Path("series-arm64-get-k16-v16-t7-p10-throughput.json")
        assert DashboardPublisher._epoch_path(path, "v2").name == (
            "series-arm64-get-k16-v16-t7-p10-throughput.epoch-v2.json"
        )

    def test_legacy_magic_mock_coordinator_defaults_to_v1(self):
        assert DashboardPublisher._coord_epoch(MagicMock()) == "v1"


def test_publish_writes_isolated_v1_v2_series_and_manifests(tmp_path):
    def make_coord(epoch_id):
        coord = MagicMock()
        coord.epoch_id = epoch_id
        coord.workload_id = "get-k16-v16-t7-p10"
        coord.metric_id = "memory"  # avoid perf side exports in this contract test
        coord.engine = None
        coord.lower_is_better = False
        coord.state = MagicMock()

        def export(path, platform):
            path.write_text('{"metadata": {}, "points": []}')
            return 0

        coord.export.side_effect = export
        return coord

    publisher = DashboardPublisher("user@host:/path", [make_coord("v1"), make_coord("v2")])
    publisher._export_dir = tmp_path

    def export_notable(_sources, path, _platform):
        path.write_text('{"metadata": {}, "annotations": []}')

    with (
        patch("conductress.sweep.exporter.export_notable", side_effect=export_notable),
        patch("conductress.publisher.run_rsync"),
    ):
        publisher.on_task_completed(MagicMock())

    platform = publisher._platform_id
    legacy = tmp_path / f"series-{platform}-get-k16-v16-t7-p10-memory.json"
    v2 = tmp_path / f"series-{platform}-get-k16-v16-t7-p10-memory.epoch-v2.json"
    assert legacy.exists()
    assert v2.exists()
    assert json.loads(legacy.read_text())["metadata"]["epoch"] == "v1"
    assert json.loads(v2.read_text())["metadata"]["epoch"] == "v2"

    base_manifest = json.loads((tmp_path / f"manifest-{platform}.json").read_text())
    v2_manifest = json.loads((tmp_path / f"manifest-{platform}.epoch-v2.json").read_text())
    assert [epoch["id"] for epoch in base_manifest["epochs"]] == ["v1", "v2"]
    assert v2_manifest["epoch"] == "v2"


def _v3_coord_with_stacks():
    """A v3 throughput coordinator over a real SweepState with stacks on one point."""
    from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState

    state = SweepState(merge_commits=["aaa", "bbb"], commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01"})
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa",
        date="2024-01-01",
        value=2_000_000.0,
        status=PointStatus.COMPLETED,
        cpu_stacks_main=[["valkey-server;main;aeMain", 2000]],
        cpu_stacks_io=[["io_thd_1;IOThreadMain", 6000]],
    )
    state.points["bbb"] = BenchmarkPoint(
        commit="bbb", date="2024-02-01", value=2_100_000.0, status=PointStatus.COMPLETED
    )

    coord = MagicMock()
    coord.epoch_id = "v3"
    coord.workload_id = "get-k16-v16-t7-p10"
    coord.metric_id = "throughput"
    coord.lower_is_better = False
    coord.state = state
    coord._sweep_ref = "origin/unstable"
    coord.engine = MagicMock(source="valkey", profile_internals=True)

    def export(path, platform):
        path.write_text('{"metadata": {}, "points": []}')
        return 2

    coord.export.side_effect = export
    return coord


def test_second_publish_does_not_rewrite_promoted_stacks_files(tmp_path):
    """Regression: every publish regenerated every per-commit stacks file.

    The v3 export goes through a staging directory that is emptied each
    publish, so the exporter's own exists() check never saw the files it had
    already promoted. It rewrote all of them (multi-MB each, thousands per
    runner), promotion gave them fresh mtimes, and the publish rsync moved the
    whole tree again every boundary, hitting its timeout before reaching the
    series files that sort after them. The promoted file must keep its mtime
    across a second publish, and the series file must still be refreshed.
    """
    import os
    import time

    publisher = DashboardPublisher("user@host:/path", [_v3_coord_with_stacks()])
    publisher._export_dir = tmp_path
    platform = publisher._platform_id
    stacks = tmp_path / f"series-{platform}-get-k16-v16-t7-p10-cpu-stacks-aaa.epoch-v3.json"
    series = tmp_path / f"series-{platform}-get-k16-v16-t7-p10-throughput.epoch-v3.json"

    with patch("conductress.publisher.run_rsync"), patch("conductress.sweep.exporter.export_perf_metrics"):
        publisher.on_task_completed(MagicMock())
        assert stacks.exists() and series.exists()
        old_time = 1_600_000_000
        os.utime(stacks, (old_time, old_time))
        os.utime(series, (old_time, old_time))
        time.sleep(0.01)
        publisher.on_task_completed(MagicMock())

    assert int(stacks.stat().st_mtime) == old_time, "promoted stacks file was regenerated"
    assert int(series.stat().st_mtime) != old_time, "series file must be refreshed every publish"
    assert not list(tmp_path.glob(".stage-*")), "stage must be cleaned up after promotion"


def test_new_stacks_point_is_exported_when_older_ones_are_already_published(tmp_path):
    """The skip is per file: a point that gains stacks later is still exported."""
    from conductress.sweep.planner import BenchmarkPoint, PointStatus

    coord = _v3_coord_with_stacks()
    publisher = DashboardPublisher("user@host:/path", [coord])
    publisher._export_dir = tmp_path
    platform = publisher._platform_id

    with patch("conductress.publisher.run_rsync"), patch("conductress.sweep.exporter.export_perf_metrics"):
        publisher.on_task_completed(MagicMock())
        coord.state.points["bbb"] = BenchmarkPoint(
            commit="bbb",
            date="2024-02-01",
            value=2_100_000.0,
            status=PointStatus.COMPLETED,
            cpu_stacks_main=[["valkey-server;main;aeMain", 10]],
            cpu_stacks_io=[],
        )
        publisher.on_task_completed(MagicMock())

    assert (tmp_path / f"series-{platform}-get-k16-v16-t7-p10-cpu-stacks-aaa.epoch-v3.json").exists()
    assert (tmp_path / f"series-{platform}-get-k16-v16-t7-p10-cpu-stacks-bbb.epoch-v3.json").exists()
    index = json.loads((tmp_path / f"series-{platform}-get-k16-v16-t7-p10-cpu-stacks-index.epoch-v3.json").read_text())
    assert [entry["commit"] for entry in index["commits"]] == ["aaa", "bbb"]
