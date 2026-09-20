"""Tests for the connection-storm plot builder and CLI.

matplotlib is an optional extra; these tests importorskip it. The fixture at
tests/fixtures/storm_plot/output.jsonl is derived from the calibration
mock-up data (arms A-cycle10 and C-backlog4096), two task ids with two reps
each, subsampled to keep it small (provenance in the PR body).
"""

import io
import math
from pathlib import Path

import pytest

from conductress.cli import build_parser, handle_plot, parse_xrange

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "storm_plot" / "output.jsonl"
TASK_A = "20260101T000001"
TASK_C = "20260101T000002"


# --------------------------------------------------------------------------- xrange parse (pure)


def test_parse_xrange_negative_lower_bound():
    assert parse_xrange("-3:8") == (-3.0, 8.0)


def test_parse_xrange_none_and_empty():
    assert parse_xrange(None) is None
    assert parse_xrange("") is None


@pytest.mark.parametrize("bad", ["8", "3:2", "a:b", "1:"])
def test_parse_xrange_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_xrange(bad)


# --------------------------------------------------------------------------- alignment (pure)


def test_alignment_maps_series_to_shared_zero():
    """t=0 is the stall start; every series' wall clock maps onto it exactly."""
    from conductress.plots.connection_storm import series_t0, wall_to_axis

    stall_started = 1000.0
    scenario_metrics = {
        "measure_start_wall": 995.0,
        "background_start_wall": 996.0,
        "storm": {"origin_wall": 995.2, "stall": {"started": stall_started, "ended": stall_started + 2.0}},
    }
    t0 = series_t0(scenario_metrics)
    assert t0 == pytest.approx(stall_started)
    # A background bucket at second 5 (wall 996+5) lands at 996+5-1000 = 1.0s.
    assert wall_to_axis(996.0 + 5.0, t0) == pytest.approx(1.0)
    # A sampler row sent at wall 1002.5 lands at 2.5s.
    assert wall_to_axis(1002.5, t0) == pytest.approx(2.5)


def test_alignment_uses_storm_origin_without_stall():
    """Without a stall, t=0 is the storm event itself: the generator's origin."""
    from conductress.plots.connection_storm import measurement_start_axis, series_t0

    scenario_metrics = {
        "measure_start_wall": 500.0,
        "background_start_wall": 501.0,
        "storm": {"origin_wall": 506.4, "stall": {"kind": "none", "started": None, "ended": None}},
    }
    t0 = series_t0(scenario_metrics)
    assert t0 == pytest.approx(506.4)
    # The measurement started 5.4 s before the storm: that is the window's left edge.
    assert measurement_start_axis(scenario_metrics, t0) == pytest.approx(-5.4)


def test_alignment_falls_back_to_measure_start_without_storm_origin():
    from conductress.plots.connection_storm import series_t0

    scenario_metrics = {"measure_start_wall": 500.0, "background_start_wall": 501.0, "storm": {}}
    assert series_t0(scenario_metrics) == pytest.approx(500.0)


# --------------------------------------------------------------------------- loader


def test_load_run_views_reads_fixture():
    from conductress.plots.connection_storm import load_run_views

    views = load_run_views([TASK_A, TASK_C], results_path=str(FIXTURE))
    assert [v.task_id for v in views] == [TASK_A, TASK_C]
    assert all(len(v.reps) == 2 for v in views)
    assert views[0].source == "valkey"
    assert views[0].stall_seconds == pytest.approx(2.0, abs=0.05)


def test_load_run_views_missing_task_id_raises():
    from conductress.plots.connection_storm import load_run_views

    with pytest.raises(ValueError, match="no scenario result"):
        load_run_views(["nope"], results_path=str(FIXTURE))


# --------------------------------------------------------------------------- figure builder


@pytest.fixture
def views():
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import load_run_views

    return load_run_views([TASK_A, TASK_C], results_path=str(FIXTURE))


def test_build_figure_has_four_rows_two_columns(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views, xrange=(-3, 8))
    axes = fig.get_axes()
    # 4 rows x 2 columns = 8 axes (no twin axes -- overflows have their own row).
    assert len(axes) == 8


def test_build_figure_row_labels_present(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import ROW_LABELS, build_storm_figure

    fig = build_storm_figure(views)
    ylabels = {ax.get_ylabel() for ax in fig.get_axes() if ax.get_ylabel()}
    for label in ROW_LABELS:
        assert label in ylabels


def test_build_figure_one_stall_band_per_axis(views):
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba

    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views, rep=0)  # one rep so exactly one band per axis
    band_rgba = to_rgba("#7f8c8d", 0.18)
    for ax in fig.get_axes():
        bands = [p for p in ax.patches if _rgba_close(p.get_facecolor(), band_rgba)]
        assert len(bands) == 1, f"expected exactly one stall band, found {len(bands)}"


def _rgba_close(a, b, tol=0.02):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _probe_view():
    """A one-column view whose single rep carries a probe_timeline."""
    from conductress.plots.connection_storm import ScenarioRunView

    rep = {
        "background_start_wall": 1000.0,
        "storm": {
            "origin_wall": 1000.0,
            "timeline": [{"t_ms": 0, "started": 4, "connected": 4, "timeouts": 0}],
            "probe_timeline": [
                {"t_ms": 0, "sent": 1000, "served": 1000, "skipped": 0, "p50_ms": 0.3, "p99_ms": 0.5, "max_ms": 1.0},
                {"t_ms": 1000, "sent": 1000, "served": 950, "skipped": 50, "p50_ms": 0.4, "p99_ms": 2.0, "max_ms": 5.0},
            ],
            "stall": {"kind": "none"},
        },
    }
    return ScenarioRunView(
        task_id="probe-task",
        short_description="scenario:connection-storm",
        source="valkey",
        commit_hash="deadbeef0000",
        runner_id="testrunner",
        reps=[rep],
    )


def test_build_figure_probe_row_adds_a_twin_axis():
    """A probe series makes row 1 draw a p99 twin axis (extra axis beyond the 4 rows)."""
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure([_probe_view()], rep=0)
    # 4 rows x 1 column = 4, plus one twin axis for the probe p99.
    assert len(fig.get_axes()) == 5
    twin_labels = {ax.get_ylabel() for ax in fig.get_axes()}
    assert "probe p99 (ms, dotted)" in twin_labels


def test_probe_row_has_one_twin_axis_for_all_reps():
    """Three reps share ONE right-hand p99 axis per column, not one per rep."""
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    view = _probe_view()
    view.reps = [dict(view.reps[0]) for _ in range(3)]
    fig = build_storm_figure([view])  # all reps drawn
    assert len(fig.get_axes()) == 5  # 4 rows + exactly one twin
    twin = [ax for ax in fig.get_axes() if ax.get_ylabel() == "probe p99 (ms, dotted)"]
    assert len(twin) == 1 and len(twin[0].get_lines()) == 3  # one dotted line per rep on the shared twin


def test_figure_has_a_legend_naming_the_line_styles(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views)
    assert fig.legends, "figure-level legend missing"
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("median" in lbl for lbl in labels)
    assert any("stall" in lbl for lbl in labels)
    assert any("connected" in lbl for lbl in labels)


def test_gap_breaks_the_connected_line_with_nan(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views, rep=0)
    # Row 2 (connected_clients) is axes index 2 and 3 (row-major: row*ncols+col).
    # Column for TASK_C (index 1) is where the gap lives.
    found_nan = False
    for line in fig.get_axes()[3].get_lines():  # row 1 (0-based), col 1 -> index 1*2+1=3
        ydata = line.get_ydata()
        if any(isinstance(y, float) and math.isnan(y) for y in ydata):
            found_nan = True
    assert found_nan, "expected a NaN break in the connected_clients line where the sampler had a gap"


def test_build_figure_renders_to_png_under_cap(views, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.image as mpimg

    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views, xrange=(-3, 8))
    out = tmp_path / "fig.png"
    fig.savefig(out)
    img = mpimg.imread(out)
    height, width = img.shape[0], img.shape[1]
    assert width <= 2000 and height <= 2000
    # Default 15x11 in @ 100 dpi = 1500x1100.
    assert width == 1500 and height == 1100


def test_build_figure_caps_at_three_columns():
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import ScenarioRunView, build_storm_figure

    def _view(tid):
        return ScenarioRunView(
            task_id=tid,
            short_description=f"scenario:connection-storm {tid}",
            source="valkey",
            commit_hash="abc123",
            runner_id="bench",
            reps=[{"interval_rps": [1.0, 2.0], "background_start_wall": 100.0, "storm": {"stall": {}}}],
        )

    fig = build_storm_figure([_view("a"), _view("b"), _view("c"), _view("d")])
    assert len(fig.get_axes()) == 4 * 3  # 4th column dropped


# --------------------------------------------------------------------------- CLI


def test_cli_plot_args_map_and_render(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")

    out = tmp_path / "cli.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "connection-storm", TASK_A, TASK_C, "--out", str(out), "--xrange=-3:8"])
    # Point the loader at the fixture.
    import conductress.plots.connection_storm as mod

    real_loader = mod.load_run_views
    monkeypatch.setattr(mod, "load_run_views", lambda ids, **kw: real_loader(ids, results_path=str(FIXTURE)))

    rc = handle_plot(args, parser)
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


def test_cli_plot_rejects_more_than_three_task_ids():
    parser = build_parser()
    args = parser.parse_args(["plot", "connection-storm", "a", "b", "c", "d", "--out", "/tmp/x.png"])
    assert handle_plot(args, parser) == 1


def test_cli_plot_bad_xrange():
    parser = build_parser()
    args = parser.parse_args(["plot", "connection-storm", TASK_A, "--out", "/tmp/x.png", "--xrange", "8"])
    assert handle_plot(args, parser) == 1


def test_cli_plot_matplotlib_missing_message(monkeypatch, capsys):
    """The install hint is printed and exit code 2 when matplotlib is absent."""
    import conductress.plots.connection_storm as mod
    from conductress.plots.connection_storm import INSTALL_HINT

    parser = build_parser()
    args = parser.parse_args(["plot", "connection-storm", TASK_A, "--out", "/tmp/x.png"])

    # Loader succeeds against the fixture; the builder fails as if matplotlib is absent.
    real_loader = mod.load_run_views
    monkeypatch.setattr(mod, "load_run_views", lambda ids, **kw: real_loader(ids, results_path=str(FIXTURE)))

    def _raise(*a, **k):
        raise ImportError(INSTALL_HINT)

    monkeypatch.setattr(mod, "build_storm_figure", _raise)
    rc = handle_plot(args, parser)
    assert rc == 2
    assert "matplotlib" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- comparability across columns


def test_rows_share_y_axis_across_columns(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views)
    axes = fig.axes
    ncols = len(views)
    for row in range(4):
        left, right = axes[row * ncols], axes[row * ncols + 1]
        assert left.get_shared_y_axes().joined(left, right), f"row {row} columns do not share y"
        assert left.get_ylim() == right.get_ylim()


def test_column_titles_tell_arms_apart(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == len(views)
    for view, title in zip(views, titles):
        assert view.task_id in title
        assert "stall=" in title
        assert ("server defaults" in title) or (view.server_args and view.server_args in title)
    assert len(set(titles)) == len(titles)  # no two columns carry the same title


def test_column_title_names_server_args_and_stall():
    from conductress.plots.connection_storm import ScenarioRunView

    view = ScenarioRunView(
        task_id="t1",
        short_description="scenario:connection-storm v=512B io=9 30s",
        source="valkey",
        commit_hash="abc",
        runner_id="r",
        server_args="--tcp-backlog 4096",
        stall_spec="none",
    )
    title = view.column_title()
    assert "--tcp-backlog 4096" in title and "stall=none" in title and "t1" in title
    bare = ScenarioRunView(task_id="t2", short_description="d", source="v", commit_hash="c", runner_id="r")
    assert "server defaults" in bare.column_title()


def test_x_label_names_the_time_origin(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import build_storm_figure

    fig = build_storm_figure(views)
    labels = [ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()]
    assert labels, "bottom row carries x labels"
    for view, label in zip(views, labels):
        expected = "stall start" if view.stall_seconds is not None else "storm start"
        assert expected in label


def test_default_window_covers_the_storm(views):
    pytest.importorskip("matplotlib")
    from conductress.plots.connection_storm import _default_xrange, _storm_bucket_xy, build_storm_figure, series_t0

    lo, hi = _default_xrange(views)
    # Left edge is the measurement start (before the storm), never later than -3 s.
    assert lo <= -3.0
    last_bucket = max(
        max(_storm_bucket_xy(rep, series_t0(rep), "timeouts")[0] or [0.0]) for view in views for rep in view.reps
    )
    assert hi >= last_bucket + 3.0
    fig = build_storm_figure(views)  # no xrange given -> the default applies
    assert fig.axes[0].get_xlim()[1] >= last_bucket + 3.0
