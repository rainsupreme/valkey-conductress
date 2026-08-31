"""Dashboard publisher: exports and rsyncs data to the dashboard server after task completions."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from conductress.config import should_profile_internals
from conductress.utility import run_rsync

if TYPE_CHECKING:
    from conductress.sweep.coordinator import BaseSweepCoordinator
    from conductress.task_queue import BaseTaskData

logger = logging.getLogger(__name__)


def detect_platform() -> tuple[str, str]:
    """Detect platform ID and label. Kept as a compatibility wrapper."""
    from conductress.platform import get_local_platform_info

    platform_id, label, _aliases = get_local_platform_info()
    return platform_id, label


class DashboardPublisher:
    """Subscriber that exports sweep data and rsyncs to a remote server after task completions."""

    def __init__(self, target: str, coordinators: "list[BaseSweepCoordinator]") -> None:
        """
        Args:
            target: rsync destination, e.g. "ec2-user@host:/var/www/data"
            coordinators: list of sweep coordinators whose data to export
        """
        self.target = target
        self.coordinators = coordinators
        # Key may be at different paths depending on host
        candidates = [Path.home() / "conductress" / "server-keyfile.pem", Path.home() / ".ssh" / "openssh-ec2-pair.pem"]
        self._ssh_key = next((k for k in candidates if k.exists()), candidates[0])
        self._platform_id, self._platform_label = detect_platform()
        self._export_dir = Path(tempfile.mkdtemp(prefix="conductress-publish-"))
        logger.info("Publisher initialized: target=%s, platform=%s", target, self._platform_id)

    def on_task_completed(self, task: "BaseTaskData") -> None:
        """Export and publish after each completed task."""
        self._publish()

    def on_task_failed(self, task: "BaseTaskData") -> None:
        """No-op on failure."""

    def on_queue_empty(self) -> None:
        """No-op."""

    @staticmethod
    def _coord_epoch(coord: object) -> str:
        epoch_id = getattr(coord, "epoch_id", "v1")
        return epoch_id if isinstance(epoch_id, str) and epoch_id else "v1"

    @staticmethod
    def _epoch_path(path: Path, epoch_id: str) -> Path:
        """Return the legacy path for v1 or add `.epoch-<id>` for v2+."""
        if not epoch_id or epoch_id == "v1":
            return path
        return path.with_name(f"{path.stem}.epoch-{epoch_id}{path.suffix}")

    @staticmethod
    def _stamp_epoch(path: Path, epoch_id: str) -> None:
        """Stamp exported JSON with its epoch before publication."""
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        metadata = payload.setdefault("metadata", {})
        metadata["epoch"] = epoch_id
        path.write_text(json.dumps(payload, indent=2))

    def _promote_epoch_stage(self, stage: Path, epoch_id: str) -> None:
        """Move staged exporter files into the epoch-qualified namespace."""
        for path in stage.glob("*.json"):
            target = self._epoch_path(self._export_dir / path.name, epoch_id)
            self._stamp_epoch(path, epoch_id)
            path.replace(target)
        shutil.rmtree(stage, ignore_errors=True)

    def _publish(self) -> None:
        """Export each measurement epoch independently, then rsync."""
        from conductress.sweep.exporter import (
            NotableSource,
            export_cpu_profile,
            export_cpu_stacks_raw,
            export_manifest,
            export_notable,
            export_perf_metrics,
        )

        try:
            epoch_ids = list(dict.fromkeys(self._coord_epoch(c) for c in self.coordinators))
            epoch_defs = [
                {
                    "id": epoch_id,
                    "label": "Legacy v1 (stock generator)" if epoch_id == "v1" else "Scalable v2 (patched generator)",
                    "generator": "stock" if epoch_id == "v1" else "patched",
                }
                for epoch_id in epoch_ids
            ]

            for coord in self.coordinators:
                epoch_id = self._coord_epoch(coord)
                base = self._export_dir / f"series-{self._platform_id}-{coord.workload_id}-{coord.metric_id}.json"
                output = self._epoch_path(base, epoch_id)
                coord.export(output, platform=self._platform_label)
                self._stamp_epoch(output, epoch_id)

                if coord.metric_id != "throughput":
                    continue

                repo = "redis/redis" if coord.engine and coord.engine.source == "redis" else "valkey-io/valkey"
                branch = coord._sweep_ref.replace("origin/", "") if coord.engine else "unstable"
                export_dir = self._export_dir
                stage = None
                if epoch_id != "v1":
                    stage = self._export_dir / f".stage-{epoch_id}-{coord.workload_id}"
                    shutil.rmtree(stage, ignore_errors=True)
                    stage.mkdir(parents=True)
                    export_dir = stage

                export_perf_metrics(
                    coord.state, export_dir, self._platform_id, coord.workload_id, repo=repo, branch=branch
                )
                if should_profile_internals(coord.engine):
                    export_cpu_profile(
                        coord.state, export_dir, self._platform_id, coord.workload_id, repo=repo, branch=branch
                    )
                    export_cpu_stacks_raw(
                        coord.state, export_dir, self._platform_id, coord.workload_id, repo=repo, branch=branch
                    )
                if stage is not None:
                    self._promote_epoch_stage(stage, epoch_id)

            # Notable feeds and manifests are isolated by epoch as well.  The
            # legacy manifest advertises every available epoch so old URLs stay
            # valid while new dashboards can discover v2.
            for epoch_id in epoch_ids:
                epoch_coords = [c for c in self.coordinators if self._coord_epoch(c) == epoch_id]
                notable_sources = [
                    NotableSource(
                        state=coord.state,
                        workload=coord.workload_id,
                        metric=coord.metric_id,
                        lower_is_better=coord.lower_is_better,
                    )
                    for coord in epoch_coords
                    if coord.metric_id in ("throughput", "memory")
                    and (not coord.engine or coord.engine.source == "valkey")
                ]
                notable = self._epoch_path(self._export_dir / f"notable-{self._platform_id}.json", epoch_id)
                export_notable(notable_sources, notable, self._platform_label)
                self._stamp_epoch(notable, epoch_id)

                workloads = list(dict.fromkeys((c.workload_id, c.metric_id) for c in epoch_coords))
                export_manifest(
                    self._export_dir,
                    platforms=["amd64", "arm64", "graviton4", "intel"],
                    workloads=workloads,
                    epoch_id=epoch_id,
                    epochs=epoch_defs,
                )

            self._rsync()
        except Exception:
            logger.error("Publish failed (non-fatal) — dashboard data may be stale", exc_info=True)

    def _rsync(self) -> None:
        """Rsync export directory to remote target."""
        ssh_cmd = f"ssh -i {self._ssh_key} -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=10"
        run_rsync(
            ["rsync", "-az", "--chmod=D755,F644", "-e", ssh_cmd, f"{self._export_dir}/", self.target],
            self.target,
        )
