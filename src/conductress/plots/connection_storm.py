"""Figure builder and results loader for the ``connection-storm`` scenario.

Renders one column per task id (at most three), each column a stack of four
rows that share the x axis (seconds from the stall start, or from the
measurement start when there is no stall):

    row 1  background throughput   memtier interval_rps, step plot at 1 s
    row 2  connected_clients       server sampler, line broken across gaps
    row 3  listen-queue overflows  cumulative ListenOverflows, own scale
    row 4  storm clients           timeouts per bucket as bars, with cumulative
                                    connected as a light filled area behind

Design choices (Rain's review of the earlier overlaid mock-up): before/after
are SIDE-BY-SIDE columns, not overlays; one hue per column (series within a
column are told apart by row, not colour); timeouts get their own row. Every
row carries the stall band, a quiescence marker, and grey gap markers.

matplotlib is an optional dependency (``conductress[plots]``); it is imported
inside the functions that need it so importing this module is always cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from conductress.analysis import AnalysisModule

# One hue per column. High-contrast trio (matches the calibration mock-up).
COLUMN_HUES = ("#c0392b", "#e67e22", "#1e8449")
MAX_COLUMNS = 3
ROW_LABELS = (
    "goodput/s\n(probe served, else GET)",
    "server: connected_clients",
    "listen-queue overflows\n(cumulative)",
    "storm clients per bucket",
)

INSTALL_HINT = "matplotlib is required for plots: pip install 'conductress[plots]'"


def _require_matplotlib():
    """Import matplotlib (Agg backend) or raise a one-line install hint."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401  (imported so the caller can use it)

        return matplotlib
    except ImportError as exc:  # pragma: no cover - exercised via the CLI test path
        raise ImportError(INSTALL_HINT) from exc


# --------------------------------------------------------------------------- alignment (pure)


def series_t0(scenario_metrics: Dict[str, Any]) -> float:
    """The wall-clock time that maps to axis ``t = 0`` for one repetition.

    Convention: ``t = 0`` is the storm event. That is the stall start when a
    stall exists, otherwise the storm generator's origin (``storm.origin_wall``,
    the instant its burst clock starts, after prewarm). Runs with neither fall
    back to ``measure_start_wall``.
    """
    storm = scenario_metrics.get("storm") or {}
    stall = storm.get("stall") or {}
    started = stall.get("started")
    if started is not None:
        return float(started)
    origin = storm.get("origin_wall")
    if origin is not None:
        return float(origin)
    measure_start = scenario_metrics.get("measure_start_wall")
    if measure_start is not None:
        return float(measure_start)
    return float(scenario_metrics.get("background_start_wall", 0.0))


def measurement_start_axis(scenario_metrics: Dict[str, Any], t0: float) -> Optional[float]:
    """Axis position of the background measurement's start, or None if unknown."""
    start = scenario_metrics.get("background_start_wall")
    if start is None:
        start = scenario_metrics.get("measure_start_wall")
    return None if start is None else wall_to_axis(float(start), t0)


def wall_to_axis(wall: float, t0: float) -> float:
    """Map a wall-clock instant to axis seconds (relative to ``t0``)."""
    return float(wall) - float(t0)


# --------------------------------------------------------------------------- run view + loader


@dataclass
class ScenarioRunView:
    """One task's connection-storm result, distilled for plotting.

    Holds the metadata for titles plus the per-repetition scenario metrics
    (each rep's ``interval_rps``, ``server_timeline``, ``storm`` namespace and
    wall anchors). Built from a ``BenchmarkResults`` record; no I/O.
    """

    task_id: str
    short_description: str
    source: str
    commit_hash: str
    runner_id: str
    reps: List[Dict[str, Any]] = field(default_factory=list)
    server_args: str = ""
    stall_spec: str = "none"

    def column_title(self) -> str:
        """Title that tells this column apart from its neighbours.

        The scenario's short description is identical for arms that differ only
        in server arguments or stall, so the title also names the stall spec, the
        effective server arguments (or "server defaults") and the task id.
        """
        args = self.server_args.strip() or "server defaults"
        return f"{self.short_description}\nstall={self.stall_spec}  {args}\n{self.task_id}"

    @property
    def stall_seconds(self) -> Optional[float]:
        """Stall duration in seconds from the first rep's storm record, or None."""
        for rep in self.reps:
            stall = (rep.get("storm") or {}).get("stall") or {}
            started, ended = stall.get("started"), stall.get("ended")
            if started is not None and ended is not None:
                return float(ended) - float(started)
        return None

    def median_rep_index(self) -> int:
        """Index of the rep whose background mean RPS is the median (bold line)."""
        if not self.reps:
            return 0
        means: List[Tuple[float, int]] = []
        for i, rep in enumerate(self.reps):
            rps = rep.get("interval_rps") or []
            means.append((sum(rps) / len(rps) if rps else 0.0, i))
        means.sort()
        return means[len(means) // 2][1]


def _server_args_for(record: Dict[str, Any], agg: Dict[str, Any]) -> str:
    """Effective server arguments, from the aggregate metrics or the task data."""
    for candidate in (agg.get("server_args_effective"), record.get("data", {}).get("server_args")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _stall_spec_for(reps: Sequence[Dict[str, Any]]) -> str:
    """The storm's stall spec (e.g. ``debug-sleep:2``) from the first rep that recorded one."""
    for rep in reps:
        storm = rep.get("storm") or {}
        config = storm.get("generator_config") or {}
        spec = config.get("stall")
        if isinstance(spec, str) and spec:
            return spec
        kind = (storm.get("stall") or {}).get("kind")
        if isinstance(kind, str) and kind:
            return kind
    return "none"


def _default_xrange(views: Sequence["ScenarioRunView"]) -> Tuple[float, float]:
    """A window from the measurement start to the storm's tail.

    The left edge is the earliest background-measurement start across the runs
    (negative, since ``t = 0`` is the storm event); the right edge covers the
    last storm bucket and the stall end plus 3 s.
    """
    lo = -3.0
    hi = 6.0
    for view in views:
        for data in view.reps:
            t0 = series_t0(data)
            start = measurement_start_axis(data, t0)
            if start is not None:
                lo = min(lo, start)
            tx, _ = _storm_bucket_xy(data, t0, "timeouts")
            if tx:
                hi = max(hi, max(tx) + 3.0)
            stall = (data.get("storm") or {}).get("stall") or {}
            ended = stall.get("ended")
            if ended is not None:
                hi = max(hi, float(ended) - t0 + 3.0)
    return (lo, hi)


def _short_description_for(record: Dict[str, Any]) -> str:
    """Rebuild the task's short description from its stored data, best effort."""
    data = record.get("data", {})
    scenario = data.get("scenario", record.get("method", "scenario"))
    size = data.get("size", 0)
    io_threads = data.get("io_threads", 0)
    duration = data.get("duration", 0)
    return f"scenario:{scenario} v={size}B io={io_threads} {duration}s"


def load_run_views(
    task_ids: Sequence[str],
    *,
    results_path: Optional[Any] = None,
) -> List[ScenarioRunView]:
    """Load a ``ScenarioRunView`` for each task id, reusing the analysis reader.

    Reuses :class:`AnalysisModule` to read ``output.jsonl`` (the one results
    reader); does not add a second one. Raises ``ValueError`` naming any task
    id with no scenario result.
    """
    module = AnalysisModule(Path(results_path)) if results_path is not None else AnalysisModule()
    records = module.load_results()
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        tid = record.get("task_id")
        if tid and str(record.get("method", "")).startswith("scenario-"):
            by_id[tid] = record  # last write wins; results are append-only per task

    views: List[ScenarioRunView] = []
    missing: List[str] = []
    for task_id in task_ids:
        found = by_id.get(task_id)
        if found is None:
            missing.append(task_id)
            continue
        data: Dict[str, Any] = dict(found.get("data") or {})
        agg: Dict[str, Any] = dict(data.get("scenario_metrics") or {})
        reps: List[Any] = list(agg.get("per_rep") or [])
        views.append(
            ScenarioRunView(
                task_id=task_id,
                short_description=_short_description_for(found),
                source=found.get("source", "?"),
                commit_hash=found.get("commit_hash", "?"),
                runner_id=found.get("runner_id", "?"),
                reps=list(reps),
                server_args=_server_args_for(found, agg),
                stall_spec=_stall_spec_for(reps),
            )
        )
    if missing:
        raise ValueError(f"no scenario result found for task id(s): {', '.join(missing)}")
    return views


# --------------------------------------------------------------------------- figure builder


def _remote_note(axis) -> None:
    axis.text(0.5, 0.5, "server was remote:\nno listen-overflow counters", ha="center", va="center", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def _bg_series(rep: Dict[str, Any], t0: float) -> Tuple[List[float], List[float]]:
    """Background throughput: interval_rps at 1 s, anchored at background_start_wall."""
    rps = rep.get("interval_rps") or []
    start = rep.get("background_start_wall")
    if start is None:
        return [], []
    xs = [wall_to_axis(float(start) + i, t0) for i in range(len(rps))]
    return xs, [float(v) for v in rps]


def _sampler_xy(rep: Dict[str, Any], t0: float, field_name: str) -> Tuple[List[float], List[float]]:
    """A sampler field over time, with NaN inserted across gaps to break the line."""
    from conductress.tasks.task_scenario import sampler_gaps

    rows = rep.get("server_timeline") or []
    tick_ms = rep.get("server_timeline_tick_ms", 100)
    gap_reply_times = {g["t_reply"] for g in sampler_gaps(rows, tick_ms)}
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        t_reply = row.get("t_reply")
        value = row.get(field_name)
        if t_reply is None:
            continue
        if t_reply in gap_reply_times:
            xs.append(wall_to_axis(t_reply, t0))
            ys.append(float("nan"))  # break the line where the sampler was blocked
        xs.append(wall_to_axis(t_reply, t0))
        ys.append(float("nan") if value is None else float(value))
    return xs, ys


def _probe_xy(rep: Dict[str, Any], t0: float, key: str) -> Tuple[List[float], List[float]]:
    """A ``storm.probe_timeline`` field per 1 s bucket, on the shared axis.

    Probe buckets carry ``t_ms`` relative to the storm origin (same convention
    as the storm timeline), so they map onto the axis by adding ``origin_wall``
    and subtracting ``t0``. Empty lists when there is no probe series.
    """
    storm = rep.get("storm") or {}
    origin_wall = storm.get("origin_wall")
    timeline = storm.get("probe_timeline") or []
    if origin_wall is None or not timeline:
        return [], []
    xs = [wall_to_axis(float(origin_wall) + b.get("t_ms", 0) / 1000.0, t0) for b in timeline]
    ys = [float(b.get(key, 0)) for b in timeline]
    return xs, ys


def _storm_bucket_xy(rep: Dict[str, Any], t0: float, key: str) -> Tuple[List[float], List[float]]:
    """A storm timeline field per bucket, anchored at the storm origin_wall."""
    storm = rep.get("storm") or {}
    origin_wall = storm.get("origin_wall")
    timeline = storm.get("timeline") or []
    if origin_wall is None:
        return [], []
    xs = [wall_to_axis(float(origin_wall) + b.get("t_ms", 0) / 1000.0, t0) for b in timeline]
    ys = [float(b.get(key, 0)) for b in timeline]
    return xs, ys


def _draw_common(axis, rep: Dict[str, Any], t0: float, hue: str) -> None:
    """Stall band, quiescence marker and gap markers -- drawn on every row."""
    from conductress.tasks.task_scenario import sampler_gaps

    storm = rep.get("storm") or {}
    stall = storm.get("stall") or {}
    started, ended = stall.get("started"), stall.get("ended")
    if started is not None and ended is not None:
        axis.axvspan(wall_to_axis(started, t0), wall_to_axis(ended, t0), color="#7f8c8d", alpha=0.18, lw=0)
        quiescence = storm.get("quiescence_from_stall_end_s")
        if quiescence is not None:
            axis.axvline(wall_to_axis(ended, t0) + float(quiescence), color=hue, lw=1.0, ls="-.", alpha=0.8)
    rows = rep.get("server_timeline") or []
    tick_ms = rep.get("server_timeline_tick_ms", 100)
    for gap in sampler_gaps(rows, tick_ms):
        axis.axvline(wall_to_axis(gap["t_sent"], t0), color="#b0b0b0", lw=0.8, alpha=0.7)
    axis.grid(alpha=0.25)


def _has_local_netstat(rep: Dict[str, Any]) -> bool:
    """True if any sampler row carried a non-null listen_overflows (server was local)."""
    for row in rep.get("server_timeline") or []:
        if row.get("listen_overflows") is not None:
            return True
    return False


def _cumsum(values: Sequence[float]) -> List[float]:
    total = 0.0
    out: List[float] = []
    for value in values:
        total += float(value)
        out.append(total)
    return out


def _bucket_width(xs: Sequence[float]) -> float:
    """Bar width from the median bucket spacing (fallback 0.1s)."""
    if len(xs) < 2:
        return 0.1
    diffs = sorted(xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] > xs[i])
    return diffs[len(diffs) // 2] if diffs else 0.1


def build_storm_figure(
    results: Sequence[ScenarioRunView],
    *,
    rep: Optional[int] = None,
    xrange: Optional[Tuple[float, float]] = None,
):
    """Build the connection-storm figure. Returns a matplotlib Figure.

    One column per task id (max three), four rows sharing the x axis. Reps draw
    as thin lines with the median rep bold; ``rep`` selects a single rep. Colour
    is per column; rows tell series apart.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    views = list(results)[:MAX_COLUMNS]
    if not views:
        raise ValueError("build_storm_figure needs at least one run view")
    ncols = len(views)

    # Rows share their y axis across columns so arms compare at a glance.
    fig, axes = plt.subplots(4, ncols, figsize=(15, 11), dpi=100, sharex=True, sharey="row", squeeze=False)
    if xrange is None:
        xrange = _default_xrange(views)

    for col, view in enumerate(views):
        hue = COLUMN_HUES[col % len(COLUMN_HUES)]
        reps = view.reps
        rep_indices = [rep] if (rep is not None and 0 <= rep < len(reps)) else list(range(len(reps)))
        median_idx = view.median_rep_index()

        for ax_row in range(4):
            axis = axes[ax_row][col]
            for ridx in rep_indices:
                data = reps[ridx]
                bold = (rep is None) and (ridx == median_idx)
                lw = 1.8 if bold else 0.9
                alpha = 1.0 if bold else 0.5
                t0 = series_t0(data)
                if ax_row == 0:
                    px, pserved = _probe_xy(data, t0, "served")
                    if px:
                        # A probe series exists: it is the goodput measure. Draw
                        # probe served/s as a step in the column hue, its p99 on
                        # a twin axis dotted in the same hue, and (if memtier
                        # also ran) interval_rps as a faint line behind.
                        bx, by = _bg_series(data, t0)
                        if bx:
                            axis.plot(bx, by, color=hue, lw=0.8, alpha=0.25 * alpha)
                        axis.step(px, pserved, where="post", color=hue, lw=lw, alpha=alpha)
                        _px99, p99 = _probe_xy(data, t0, "p99_ms")
                        if _px99:
                            twin = axis.twinx()
                            twin.plot(_px99, p99, color=hue, lw=lw, ls=":", alpha=alpha)
                            twin.set_ylabel("probe p99 (ms)", fontsize=8)
                    else:
                        xs, ys = _bg_series(data, t0)
                        axis.step(xs, ys, where="post", color=hue, lw=lw, alpha=alpha)
                elif ax_row == 1:
                    xs, ys = _sampler_xy(data, t0, "connected_clients")
                    axis.plot(xs, ys, color=hue, lw=lw, alpha=alpha)
                elif ax_row == 2:
                    if not _has_local_netstat(data):
                        _remote_note(axis)
                        continue
                    xs, ys = _sampler_xy(data, t0, "listen_overflows")
                    axis.plot(xs, ys, color=hue, lw=lw, alpha=alpha)
                else:
                    tx, timeouts = _storm_bucket_xy(data, t0, "timeouts")
                    cx, connected = _storm_bucket_xy(data, t0, "connected")
                    if cx:
                        axis.fill_between(cx, _cumsum(connected), color=hue, alpha=0.12, step="post")
                    if tx:
                        axis.bar(tx, timeouts, width=_bucket_width(tx), color=hue, alpha=alpha, align="edge")
                _draw_common(axis, data, t0, hue)

            if col == 0:
                axis.set_ylabel(ROW_LABELS[ax_row])
        axes[0][col].set_title(view.column_title(), fontsize=9, color=hue)

    for col, view in enumerate(views):
        origin = "stall start" if view.stall_seconds is not None else "storm start"
        axes[3][col].set_xlabel(f"seconds from {origin}")
        if xrange is not None:
            axes[0][col].set_xlim(*xrange)

    meta = views[0]
    fig.suptitle(
        f"connection-storm: {meta.source} @ {meta.commit_hash[:12]} on {meta.runner_id}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
