# Canary Drift Analysis (PR2)

## Overview

Ingests completed canary benchmark outcomes into immutable daily observations
and computes rolling median/MAD statistics to detect environmental drift over
time. Designed to be consumed by the Phase 5 PR3 CLI and dashboard.

## Data Model

### canary_observations (schema v3)

Immutable daily observations keyed by `(runner_id, profile_id, profile_version,
utc_date, task_id)`. Each row stores:

- **score**: benchmark throughput (RPS). NULL for rejected observations.
- **phase**: `observation` (days 1-14), `calibrating` (15-27), `ready` (28+).
- **accepted**: 1 for valid scores, 0 for malformed/non-finite.
- **ref_median / ref_mad / delta_pct**: rolling statistics from *prior* window.
- **sample_count**: count of prior accepted observations (never includes self).
- **window_start / window_end**: date range of the reference window.
- **environment_json**: optional provenance fingerprint (kernel, instance, etc.).

### canary_calibration_reports

One report per `(runner_id, profile_id, profile_version)`. Generated
automatically at the 28th accepted observation. Status is always
`ready-for-review` — never automatically actionable.

## Algorithm

### Rolling Window

1. Collect all accepted observations for the same `(runner, profile, version)`
   with `utc_date < current_date` (strict less-than: a sample cannot move its
   own baseline).
2. Take the most recent 28 observations from that set.
3. Compute median and MAD (median absolute deviation) from the window.
4. Delta percentage: `(score - median) / |median| * 100`.

### Phase Progression

| Ordinal (nth accepted) | Phase | Behavior |
|---|---|---|
| 1–14 | `observation` | Accumulate data, no statistical assessment |
| 15–27 | `calibrating` | Statistics computed, calibration progress exposed |
| 28 | `ready` | Calibration report generated, system ready for review |
| 29+ | `ready` | Ongoing drift monitoring |

### Edge Cases

- **MAD = 0**: all prior samples identical. Delta is computed normally.
- **Median = 0**: delta clamped to ±999999.0 when score ≠ 0; 0.0 when score = 0.
- **Non-finite score**: recorded as rejected (accepted=0), does not count toward
  any phase/window.
- **Duplicate/replay**: idempotent by primary key; returns existing observation.
- **Missed days**: gaps in the date sequence; no interpolation or backfill.
- **Out-of-order**: accepted and slotted by UTC date, not arrival order.
- **Profile version change**: starts a completely independent series.

### Calibration Report

At the 28th sample, a report is stored containing:
- Observed median, MAD, coefficient of variation
- Score range (min/max/spread)
- Recommended threshold: `3 × MAD / |median| × 100%` (conservative)
- Status: `ready-for-review` (never auto-actionable)

## Integration

The `ControlService.record_outcome()` method automatically calls
`DriftAnalyzer.ingest_outcome()` after a canary task completes successfully.
Ingestion failures are logged but do not block the task completion path.

## Query Helpers (for PR3 CLI)

```python
analyzer = DriftAnalyzer(database)
analyzer.get_observation(runner_id, profile_id, version, date)
analyzer.get_observations(runner_id, profile_id, version, limit=60)
analyzer.get_all_observations(runner_id, profile_id, version)  # incl. rejected
analyzer.get_calibration_report(runner_id, profile_id, version)
analyzer.get_series_summary(runner_id, profile_id, version)
```

## DB Migration

Schema v3 adds `canary_observations` and `canary_calibration_reports` tables.
Migration is forward-compatible and idempotent (safe to run multiple times).
