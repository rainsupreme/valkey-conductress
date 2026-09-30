import json
from pathlib import Path

import pytest

from conductress import task_queue
from conductress.task_envelope import build_task_envelope, serialize_task
from conductress.task_queue import BaseTaskData

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "golden_tasks"


@pytest.mark.parametrize("fixture", sorted(GOLDEN_DIR.glob("*.json")), ids=lambda path: path.stem)
def test_all_golden_tasks_build_versioned_remote_envelopes(fixture, monkeypatch):
    monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])
    task = BaseTaskData.from_file(fixture)
    envelope = build_task_envelope(
        task,
        runner_id="armbench",
        priority=125,
        submitted_by="rain",
    )

    assert envelope["schema_version"] == 1
    assert envelope["task_id"] == task.task_id
    assert envelope["runner_id"] == "armbench"
    assert envelope["task_class"] == "manual"
    assert envelope["priority"] == 125
    assert envelope["submitted_by"] == "rain"
    assert envelope["task"] == json.loads(fixture.read_text(encoding="utf-8"))
    assert serialize_task(task) == envelope["task"]


def test_submitter_falls_back_when_user_lookup_fails(monkeypatch):
    from conductress.task_envelope import _default_submitter

    def fail_user_lookup():
        raise KeyError("no user")

    monkeypatch.setattr("conductress.task_envelope.getpass.getuser", fail_user_lookup)
    assert _default_submitter() == "unknown"


def test_envelope_carries_submitter_provenance_and_batch(monkeypatch):
    from conductress.task_envelope import build_provenance, build_submitter

    monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])
    task = BaseTaskData.from_file(next(GOLDEN_DIR.glob("*.json")))
    submitter = build_submitter("rimuru", "agent", sponsor="rainsupreme")
    provenance = build_provenance(
        "valkey-io/valkey",
        "deadbeef",
        recipe="pr-standard",
        pr={"repo": "valkey-io/valkey", "number": 42, "head_sha": "deadbeef"},
    )
    envelope = build_task_envelope(
        task,
        runner_id="armbench",
        submitter=submitter,
        provenance=provenance,
        batch_id="batch-1",
    )
    assert envelope["submitter"] == {"login": "rimuru", "kind": "agent", "sponsor": "rainsupreme"}
    assert envelope["provenance"]["repo"] == "valkey-io/valkey"
    assert envelope["provenance"]["pr"]["number"] == 42
    assert envelope["batch_id"] == "batch-1"


def test_envelope_defaults_new_fields_to_none(monkeypatch):
    monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])
    task = BaseTaskData.from_file(next(GOLDEN_DIR.glob("*.json")))
    envelope = build_task_envelope(task, runner_id="armbench", submitted_by="rain")
    assert envelope["submitter"] is None
    assert envelope["provenance"] is None
    assert envelope["batch_id"] is None


def test_build_submitter_rejects_bad_kind():
    from conductress.task_envelope import build_submitter

    with pytest.raises(ValueError):
        build_submitter("x", "robot")
