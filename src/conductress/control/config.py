"""Configuration for the data-host fleet control service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ControlConfig:
    database_path: Path
    fleet_manifest_path: Path
    tokens_path: Path
    audit_jsonl_path: Path
    users_path: Optional[Path] = None
    notification_url: Optional[str] = None
    published_tasks_dir: Optional[Path] = None
    canary_profiles_dir: Path = Path(__file__).resolve().parent.parent / "canary_profiles"
    claim_lease_seconds: int = 300
    max_body_bytes: int = 64 * 1024

    @classmethod
    def from_env(cls) -> "ControlConfig":
        default_profiles = Path(__file__).resolve().parent.parent / "canary_profiles"
        users_value = os.environ.get("USERS_PATH", "/etc/conductress-control/users.toml")
        published_value = os.environ.get("PUBLISHED_TASKS_DIR")
        return cls(
            database_path=Path(os.environ.get("CONTROL_DB_PATH", "/var/lib/conductress-control/control.db")),
            fleet_manifest_path=Path(os.environ.get("FLEET_MANIFEST_PATH", "/etc/conductress-control/fleet.json")),
            tokens_path=Path(os.environ.get("TOKENS_PATH", "/etc/conductress-control/tokens.json")),
            audit_jsonl_path=Path(os.environ.get("AUDIT_JSONL_PATH", "/var/log/conductress-control/audit.jsonl")),
            users_path=Path(users_value) if users_value else None,
            notification_url=os.environ.get("NOTIFICATION_URL") or None,
            published_tasks_dir=Path(published_value) if published_value else None,
            canary_profiles_dir=Path(os.environ.get("CANARY_PROFILES_DIR", str(default_profiles))),
            claim_lease_seconds=int(os.environ.get("CLAIM_LEASE_SECONDS", "300")),
            max_body_bytes=int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024))),
        )

    def validate(self) -> None:
        if self.claim_lease_seconds < 1:
            raise ValueError("claim_lease_seconds must be positive")
        if self.max_body_bytes < 1024:
            raise ValueError("max_body_bytes must be at least 1024")
