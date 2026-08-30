# Canary Drift Analysis (PR2)

## Overview

Ingests completed canary benchmark outcomes into immutable daily observations
and computes rolling median/MAD statistics to detect environmental drift over
time. Designed to be consumed by the Phase 5 PR3 CLI and dashboard.

## Data Model

### canary_observations (schema v3)

Immutable daily observations keyed by `(runner_id, profile_id, profile_version,
utc_date, task_id)`. A partial unique index enforces at most one accepted
observation per `(runner_id, profile_id, profile_version, utc_date)`.

Each row stores:

- **score**: benchmark throughput (RPS). Must be positive and finite for
  acceptance; NULL for rejected observations.
- **phase**: `observation` (days 1-14), `calibrating` (15-27), `ready` (28+).
  Phase ordinal is computed from the total accepted count in the series,
  independent of the current UTC date.
- **accepted**: 1 for valid scores, 0 for malformed/non-finite/non-positive.
- **ref_median / ref_mad / delta_pct**: rolling statistics from *prior* window
  (strictly earlier UTC dates only).
- **series_ordinal**: the nth accepted observation in this series (1-based).
- **ref_sample_count**: count of strictly-earlier accepted observations used
  for reference statistics (never includes self).
- **candidate_warning_pct / candidate_alarm_pct**: per-platform thresholds
  resolved from the canary profile.
- **candidate_signal**: classification of this observation against candidate
  thresholds (`insufficient-data`, `within`, `warning`, `alarm`).
  Informational only in all phases.
- **actionable**: always 0 (never actionable during calibration or observation).
- **window_start / window_end**: date range of the reference window.
- **environment_json**: provenance fingerprint including runner_id, platform,
  provenance_schema_version, and optional host environment fields (kernel,
  instance_id, etc.).
- **env_change_annotation**: deterministic diff versus the latest earlier
  accepted observation's environment (NULL if unchanged or first observation).
- **provenance_schema_version**: schema version from the outcome record.

### canary_calibration_reports

One report per `(runner_id, profile_id, profile_version)`. Generated
automatically at the 28th accepted observation using exactly the first 28
accepted observations sorted deterministically by `(utc_date, task_id)`.

Report fields:

- **median_score / mad**: observed central tendency and dispersion.
- **robust_sigma**: `1.4826 * MAD` — consistent estimator of σ for normal data.
- **variability_floor_pct**: `3 * robust_sigma / |median| * 100` —
  the measurement-noise floor below which drift cannot be distinguished
  from random variation (≈99.7% of normal distribution).
- **candidate_warning_pct / candidate_alarm_pct**: per-platform thresholds
  from the canary profile (may be NULL if no profile match).
- **recommended_warning_pct**: `max(candidate_warning, variability_floor)`.
- **recommended_alarm_pct**: `max(candidate_alarm, 2 * variability_floor,
  recommended_warning)`.
- **status**: always `ready-for-review` — never automatically actionable.
- **zero_median_warning**: true when median is zero, requiring manual review.

## Algorithm

### Median

For even-length lists, the median is the **average of the two middle values**.
For odd-length lists, it is the single middle value.

### Rolling Window

1. Collect all accepted observations for the same `(runner, profile, version)`
   with `utc_date < current_date` (strict less-than: a sample cannot move its
   own baseline), ordered deterministically by `(utc_date, task_id)`.
2. Take the most recent 28 observations from that set.
3. Compute median and MAD (median absolute deviation) from the window.
4. Delta percentage: `(score - median) / |median| * 100`.

### Phase Progression

Phase ordinal uses the **total accepted count** in the series before insert
(independent of the current observation's UTC date), ensuring out-of-order
arrivals do not regress phase.

| Ordinal (nth accepted) | Phase | Behavior |
|---|---|---|
| 1–14 | `observation` | Accumulate data, no statistical assessment |
| 15–27 | `calibrating` | Statistics computed, calibration progress exposed |
| 28 | `ready` | Calibration report generated, system ready for review |
| 29+ | `ready` | Ongoing drift monitoring |

The report is generated when the 28th accepted observation arrives, regardless
of whether it is chronologically the latest.

### Candidate Signals

Each observation carries a `candidate_signal` derived from the profile's
per-platform thresholds:

- **insufficient-data**: no reference statistics or no candidate thresholds.
- **within**: `|delta_pct| < candidate_warning_pct`.
- **warning**: `candidate_warning_pct <= |delta_pct| < candidate_alarm_pct`.
- **alarm**: `|delta_pct| >= candidate_alarm_pct`.

Signals use absolute delta against candidate thresholds and are **informational
only** in every phase. `actionable` is always 0. No notifications are sent.

### Calibration Report Thresholds

At the 28-sample mark:

1. `robust_sigma = 1.4826 * MAD`
2. `variability_floor = 3 * robust_sigma / |median| * 100`
3. `recommended_warning = max(candidate_warning, variability_floor)`
4. `recommended_alarm = max(candidate_alarm, 2 * variability_floor, recommended_warning)`

### Edge Cases

- **MAD = 0**: all prior samples identical. Delta is computed normally.
- **Median = 0**: delta clamped to ±999999.0 when score ≠ 0; 0.0 when score = 0.
  Calibration variability floor is 0%; report includes `zero_median_warning: true`
  and status remains `ready-for-review` with an explicit warning message.
- **Non-finite score**: recorded as rejected (accepted=0), does not count toward
  any phase/window.
- **Non-positive score**: recorded as rejected (accepted=0). Canary benchmarks
  should always produce positive throughput.
- **Duplicate/replay**: idempotent by primary key; returns existing observation.
- **Same date, different task**: the partial unique index ensures at most one
  accepted observation per date. A different task_id for an already-accepted date
  returns the existing accepted observation without alteration.
- **Missed days**: gaps in the date sequence; no interpolation or backfill.
- **Out-of-order**: accepted and slotted by UTC date, not arrival order. Phase
  ordinal uses total series count, not calendar position.
- **Profile version change**: starts a completely independent series.
- **Malformed canary_id**: non-empty profile_id and utc_date are validated
  after the colon split; empty components are rejected with a warning.
- **Environment change**: a deterministic diff annotation is computed versus
  the latest earlier accepted observation's environment.

### Provenance

Each observation retains available provenance fields from the outcome record:

- `provenance_schema_version`: from the outcome's `provenance_schema_version`
  or `schema_version` field.
- `runner_id` and `platform`: resolved from the FleetRegistry.
- `environment`: any host-level environment fields (kernel, instance_id, etc.)
  from the result.

## Integration

The `ControlService.record_outcome()` method automatically calls
`DriftAnalyzer.ingest_outcome()` after a canary task completes successfully.
Ingestion failures are logged but do not block the task completion path.

Profile version is resolved **structurally** from the `CanaryProfileRegistry`
(matching profile_id). A regex fallback on the task note field is used only
for tests and legacy compatibility.

`CanaryProfileRegistry` and `FleetRegistry` are wired into `DriftAnalyzer`
from `create_app` via `ControlService`.

## Query Helpers (for PR3 CLI)

```python
analyzer = DriftAnalyzer(database)
analyzer.get_observation(runner_id, profile_id, version, date)  # accepted-only, deterministic
analyzer.get_observations(runner_id, profile_id, version, limit=60)
analyzer.get_all_observations(runner_id, profile_id, version)  # incl. rejected
analyzer.get_calibration_report(runner_id, profile_id, version)
analyzer.get_series_summary(runner_id, profile_id, version)
```

`get_observation` returns only accepted observations (deterministic, no
rejected results).

## DB Migration

Schema v3 adds `canary_observations` and `canary_calibration_reports` tables.
The `canary_observations` table includes a partial unique index enforcing at
most one accepted observation per `(runner, profile, version, date)`.
Migration is forward-compatible and idempotent (safe to run multiple times).
