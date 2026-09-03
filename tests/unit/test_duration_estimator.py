import json

from conductress.duration_estimator import estimate_task_duration_seconds, load_duration_calibration


def perf_task(**overrides):
    task = {
        "task_type": "PerfTaskData",
        "warmup": 5,
        "duration": 30,
        "repetitions": 10,
        "max_reps": 10,
        "target_cv": 0.5,
        "perf_stat_enabled": False,
    }
    task.update(overrides)
    return task


def test_task_shape_estimates_scale_with_repetitions_and_phases():
    one_rep = estimate_task_duration_seconds(perf_task(repetitions=1, max_reps=1))
    ten_reps = estimate_task_duration_seconds(perf_task())
    assert one_rep == 113
    assert ten_reps == 590
    assert ten_reps > one_rep

    mixed = estimate_task_duration_seconds(
        {"task_type": "MixedTaskData", "warmup": 5, "duration": 30, "repetitions": 3}
    )
    scenario = estimate_task_duration_seconds(
        {"task_type": "ScenarioTaskData", "warmup": 5, "duration": 30, "repetitions": 3}
    )
    assert mixed == 315
    assert scenario == 450


def test_adaptive_perf_estimate_allows_two_extra_repetitions():
    fixed = estimate_task_duration_seconds(perf_task(repetitions=3, max_reps=10, target_cv=0))
    adaptive = estimate_task_duration_seconds(perf_task(repetitions=3, max_reps=10, target_cv=0.5))
    assert fixed == 219
    assert adaptive == 325


def test_calibration_uses_recent_unique_tasks_and_requires_three_samples(tmp_path):
    output = tmp_path / "output.jsonl"
    records = [
        {
            "task_id": f"task-{index}",
            "duration_family": "perf",
            "expected_duration_sec": 100,
            "observed_duration_sec": observed,
        }
        for index, observed in enumerate((120, 130, 140), 1)
    ]
    records.append({**records[-1], "observed_duration_sec": 999})
    output.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    calibration = load_duration_calibration(output)
    assert calibration == {"perf": 1.3}
    assert estimate_task_duration_seconds(perf_task(repetitions=1, max_reps=1), calibration) == 147
