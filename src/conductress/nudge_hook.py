"""HTTP webhook subscriber that sends push notifications to an external dashboard."""

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

from .config import CONDUCTRESS_OUTPUT
from .task_queue import BaseTaskData

logger = logging.getLogger(__name__)

# Attributes available on PerfTaskData that are useful to include in nudges
_TASK_ATTRS = (
    "test",
    "val_size",
    "io_threads",
    "pipelining",
    "warmup",
    "duration",
    "replicas",
    "note",
)


class NudgeHook:
    """HTTP webhook subscriber that sends push notifications to an external dashboard.

    When a benchmark task completes or fails, this hook reads the latest result from
    the output log and POSTs a JSON payload to the configured endpoint URL. This is
    useful for integrating with AI dashboards (e.g. OpenMesh) that need to be
    notified of new data without polling.

    Args:
        endpoint_url: HTTP(S) endpoint to POST nudge payloads to.
        events: Set of event type strings that trigger nudges.
            Supported values: "completed", "failed", "empty".
            Defaults to all three.
    """

    def __init__(self, endpoint_url: str, events: Optional[set] = None) -> None:
        self._endpoint_url = endpoint_url
        self._events: set = events if events is not None else {"completed", "failed", "empty"}
        logger.info(
            "NudgeHook initialized: endpoint=%s, events=%s",
            endpoint_url,
            self._events,
        )

    def on_task_completed(self, task: BaseTaskData) -> None:
        """Send an HTTP POST with results when a task completes successfully."""
        if "completed" in self._events:
            payload = self._build_task_payload("completed", task)
            self._send(payload)

    def on_task_failed(self, task: BaseTaskData) -> None:
        """Send an HTTP POST with task info when a task fails."""
        if "failed" in self._events:
            payload = self._build_task_payload("failed", task)
            self._send(payload)

    def on_queue_empty(self) -> None:
        """Send a lightweight HTTP POST when the task queue is empty."""
        if "empty" in self._events:
            payload = {
                "event": "empty",
                "timestamp": datetime.now().isoformat(),
            }
            self._send(payload)

    def _build_task_payload(self, event_type: str, task: BaseTaskData) -> dict:
        """Construct the JSON payload for a task-completion nudge."""
        payload: dict = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "source": task.source,
            "specifier": task.specifier,
            "note": task.note,
            "task_type": task.task_type,
        }
        # Include PerfTaskData-specific attributes if present
        for attr in _TASK_ATTRS:
            if hasattr(task, attr):
                payload[attr] = getattr(task, attr)
        # Read latest result from the output log
        result = self._read_latest_result(task.task_id)
        if result:
            payload["score"] = result.get("score")
            payload["commit_hash"] = result.get("commit_hash")
            if result.get("data"):
                payload["data"] = result["data"]
        return payload

    @staticmethod
    def _read_latest_result(task_id: str) -> Optional[dict]:
        """Read the latest result record matching *task_id* from the output log."""
        try:
            latest_match: Optional[dict] = None
            with open(CONDUCTRESS_OUTPUT, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("task_id") == task_id:
                            latest_match = record
                    except (json.JSONDecodeError, KeyError):
                        continue
            return latest_match
        except (FileNotFoundError, PermissionError):
            return None

    def _send(self, payload: dict) -> None:
        """POST *payload* as JSON to the configured endpoint URL."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                logger.debug(
                    "Nudge sent to %s: HTTP %s",
                    self._endpoint_url,
                    response.status,
                )
        except urllib.error.URLError as e:
            logger.warning("Nudge HTTP error to %s: %s", self._endpoint_url, e)
        except Exception as e:
            logger.warning("Failed to send nudge to %s: %s", self._endpoint_url, e)
