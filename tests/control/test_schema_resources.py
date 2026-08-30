from importlib.resources import files
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_NAMES = [
    "fleet-manifest.schema.json",
    "runner-config.schema.json",
    "runner-status.schema.json",
    "task-envelope.schema.json",
    "task-outcome.schema.json",
]


@pytest.mark.parametrize("schema_name", SCHEMA_NAMES)
def test_packaged_schema_matches_repository_contract(schema_name):
    packaged = files("conductress.schemas").joinpath(schema_name).read_text(encoding="utf-8")
    repository = (ROOT / "schemas" / schema_name).read_text(encoding="utf-8")
    assert packaged == repository


@pytest.mark.parametrize(
    "task_id",
    [
        "2026.08.30_22.35.54.757739",
        "canary:armbench:throughput-get-v1:2026-08-30",
        "manual-task_1.2",
    ],
)
def test_task_envelope_schema_accepts_safe_task_ids(task_id):
    from jsonschema import Draft202012Validator

    from conductress.control.schema import load_schema

    Draft202012Validator(load_schema("task-envelope.schema.json")).validate(
        {
            "schema_version": 1,
            "task_id": task_id,
            "runner_id": "armbench",
            "task_class": "manual",
            "priority": 100,
            "submitted_at": "2026-08-30T00:00:00Z",
            "submitted_by": "test",
            "task": {
                "task_type": "PerfTaskData",
                "source": "valkey",
                "specifier": "abc123",
                "timestamp": "2026-08-30T00:00:00.000000",
            },
        }
    )


@pytest.mark.parametrize("task_id", ["../escape", "contains/slash", "contains space", "x" * 201])
def test_task_envelope_schema_rejects_unsafe_task_ids(task_id):
    from jsonschema import Draft202012Validator, ValidationError

    from conductress.control.schema import load_schema

    document = {
        "schema_version": 1,
        "task_id": task_id,
        "runner_id": "armbench",
        "task_class": "manual",
        "priority": 100,
        "submitted_at": "2026-08-30T00:00:00Z",
        "submitted_by": "test",
        "task": {
            "task_type": "PerfTaskData",
            "source": "valkey",
            "specifier": "abc123",
            "timestamp": "2026-08-30T00:00:00.000000",
        },
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(load_schema("task-envelope.schema.json")).validate(document)
