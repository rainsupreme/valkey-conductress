"""File-based communication protocol for benchmark processes."""

import glob
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from .config import CONDUCTRESS_OUTPUT, CONDUCTRESS_RESULTS


@dataclass
class BenchmarkStatus:
    """Status information for a running benchmark."""

    steps_total: int
    state: str = "starting"
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    heartbeat: Optional[float] = None
    steps_completed: int = 0

    def __post_init__(self):
        if self.pid is None:
            self.pid = os.getpid()
        if self.start_time is None:
            self.start_time = time.time()


@dataclass
class MetricData:
    """Single metric data point."""

    metrics: dict[str, float]  # name-value pairs for flexible metrics
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkResults:
    """Final benchmark results."""

    method: str  # task type (perf-get, mem, sync, etc)
    source: str  # repo name
    specifier: str  # branch name, tag, or hash
    commit_hash: str  # actual commit used
    score: float  # primary result metric
    end_time: str  # completion timestamp
    data: dict[str, Any]  # detailed result info


class FileProtocol:
    """Handles file-based communication for benchmark processes."""

    def __init__(self, task_id: str, base_dir: Path = Path("/tmp")):
        self.task_id = task_id
        self.work_dir = base_dir / f"benchmark_{task_id}"
        self.work_dir.mkdir(exist_ok=True)

        self.status_file = self.work_dir / "status.json"
        self.metrics_file = self.work_dir / "metrics.jsonl"
        self.results_file = self.work_dir / "results.json"
        
        self._metrics_cache: list[MetricData] = []
        self._last_read_position = 0

    def write_status(self, status: BenchmarkStatus) -> None:
        """Atomically write status to file."""
        # Always update heartbeat timestamp
        status.heartbeat = time.time()
        self._write_json_atomic(self.status_file, asdict(status))

    def read_status(self) -> Optional[BenchmarkStatus]:
        """Read current status from file."""
        data = self._read_json_safe(self.status_file)
        return BenchmarkStatus(**data) if data else None

    def append_metric(self, metric: MetricData) -> None:
        """Append metric data to metrics file."""
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metric)) + "\n")
            f.flush()

    def read_metrics(self) -> list[MetricData]:
        """Monitor metrics file and return cached metrics.
        
        Uses tail approach to efficiently read only new lines. Results are cached.
        """
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    f.seek(self._last_read_position)
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                self._metrics_cache.append(MetricData(**data))
                            except (json.JSONDecodeError, TypeError):
                                continue
                    self._last_read_position = f.tell()
            except FileNotFoundError:
                pass
        
        return self._metrics_cache

    def write_results(self, results: BenchmarkResults) -> None:
        """Write final results to file."""
        self._write_json_atomic(self.results_file, asdict(results))

        # Also write to legacy output format
        self._write_legacy_output(results)

        # Copy metrics file to results folder
        self._copy_metrics_to_results(results)

    def _write_legacy_output(self, results: BenchmarkResults) -> None:
        """Write results to legacy output.txt format."""

        legacy_result = {
            "method": results.method,
            "source": results.source,
            "specifier": results.specifier,
            "commit_hash": results.commit_hash,
            "score": results.score,
            "end_time": results.end_time,
            "data": results.data,
        }

        os.makedirs(os.path.dirname(CONDUCTRESS_OUTPUT), exist_ok=True)
        with open(CONDUCTRESS_OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(legacy_result))
            f.write("\n")

    def _copy_metrics_to_results(self, results: BenchmarkResults) -> None:
        """Copy metrics file to results folder with useful filename."""
        if not self.metrics_file.exists():
            return

        # Create filename: method_source_specifier_timestamp.jsonl
        timestamp = results.end_time.replace(":", "-").replace(" ", "_")
        filename = f"{results.method}_{results.source}_{results.specifier}_{timestamp}.jsonl"

        dest_path = CONDUCTRESS_RESULTS / filename
        os.makedirs(CONDUCTRESS_RESULTS, exist_ok=True)
        shutil.copy2(self.metrics_file, dest_path)

    def read_results(self) -> Optional[BenchmarkResults]:
        """Read final results from file."""
        data = self._read_json_safe(self.results_file)
        return BenchmarkResults(**data) if data else None

    def cleanup(self) -> None:
        """Remove all files for this task."""
        for file_path in [self.status_file, self.metrics_file, self.results_file]:
            file_path.unlink(missing_ok=True)
        try:
            self.work_dir.rmdir()
        except OSError:
            # Directory might not be empty or might not exist, ignore
            pass

    def mark_completed_and_cleanup(self) -> None:
        """Mark task as completed and immediately clean up files.
        
        This should be called when a task completes successfully.
        """
        # Update status to completed before cleanup
        if self.status_file.exists():
            status = self.read_status()
            if status:
                status.state = "completed"
                status.end_time = time.time()
                self.write_status(status)
        
        # Clean up immediately since task is done
        self.cleanup()

    def _write_json_atomic(self, file_path: Path, data: dict[str, Any]) -> None:
        """Atomically write JSON data to file."""
        with tempfile.NamedTemporaryFile(mode="w", dir=file_path.parent, delete=False) as tmp_file:
            json.dump(data, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.rename(tmp_file.name, file_path)

    def _read_json_safe(self, file_path: Path) -> Optional[dict[str, Any]]:
        """Safely read JSON data from file."""
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _is_stale_task(status_file: Path, max_age_hours: int = 1) -> bool:
        """Check if a task is stale based on heartbeat and start time."""
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)

            current_time = time.time()
            start_time = status.get("start_time", 0)
            heartbeat = status.get("heartbeat", 0)
            state = status.get("state", "")

            # Task is stale if:
            # 1. It's been in 'starting' state for more than max_age_hours
            # 2. No heartbeat for more than max_age_hours
            max_age_seconds = max_age_hours * 3600

            if state == "starting" and (current_time - start_time) > max_age_seconds:
                return True

            if (current_time - heartbeat) > max_age_seconds:
                return True

            return False

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # If we can't read the status file, consider it stale
            return True

    @staticmethod
    def cleanup_orphaned_tasks(base_dir: Path = Path("/tmp")) -> int:
        """Clean up orphaned benchmark directories.

        Returns the number of directories cleaned up.
        """
        benchmark_dirs = glob.glob(str(base_dir / "benchmark_*"))
        cleaned_count = 0

        for dir_path in benchmark_dirs:
            dir_name = Path(dir_path).name
            status_file = Path(dir_path) / "status.json"

            should_cleanup = False
            reason = ""

            if not status_file.exists():
                # No status file means it's likely a leftover empty directory
                should_cleanup = True
                reason = "No status file found"
            else:
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        status = json.load(f)

                    pid = status.get("pid")
                    if pid and not FileProtocol._is_process_running(pid):
                        should_cleanup = True
                        reason = f"Process {pid} no longer running"
                    elif FileProtocol._is_stale_task(status_file):
                        should_cleanup = True
                        reason = "Task is stale (old heartbeat or stuck in starting state)"

                except (json.JSONDecodeError, FileNotFoundError):
                    should_cleanup = True
                    reason = "Corrupted or missing status file"

            if should_cleanup:
                try:
                    shutil.rmtree(dir_path)
                    print(f"Cleaned up orphaned benchmark directory: {dir_name} - {reason}")
                    cleaned_count += 1
                except OSError as e:
                    print(f"Failed to remove {dir_path}: {e}")

        return cleaned_count
