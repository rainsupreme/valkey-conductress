"""Load, validate, and cache versioned canary profile definitions.

Profiles are immutable JSON documents checked into ``canary_profiles/``.
This module enforces the schema (including the 40-char pinned commit
requirement), rejects unknown fields, and provides a typed lookup for
the scheduler and CLI.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .errors import ControlError
from .schema import load_schema

logger = logging.getLogger(__name__)

_MUTABLE_REF = re.compile(
    r"^("
    r"(refs/)?(heads|tags|remotes)/"  # explicit refspec
    r"|HEAD"
    r"|origin/"
    r"|upstream/"
    r"|main$|master$|unstable$"  # common branch names
    r")",
    re.IGNORECASE,
)

_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def _looks_like_mutable_ref(value: str) -> bool:
    """Heuristic guard: reject anything that is obviously not a full SHA-1."""
    if _FULL_SHA1.match(value):
        return False
    return bool(_MUTABLE_REF.match(value)) or len(value) < 40


class CanaryProfile:
    """An immutable, validated canary profile."""

    __slots__ = ("data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    @property
    def profile_id(self) -> str:
        return self.data["profile_id"]

    @property
    def profile_version(self) -> int:
        return self.data["profile_version"]

    @property
    def source(self) -> str:
        return self.data["source"]

    @property
    def pinned_commit(self) -> str:
        return self.data["pinned_commit"]

    @property
    def build(self) -> dict[str, Any]:
        return self.data["build"]

    @property
    def workload(self) -> dict[str, Any]:
        return self.data["workload"]

    @property
    def schedule(self) -> dict[str, Any]:
        return self.data["schedule"]

    @property
    def thresholds(self) -> dict[str, Any]:
        return self.data["thresholds"]

    @property
    def utc_hour(self) -> int:
        return self.schedule["utc_hour"]

    @property
    def freshness_hours(self) -> int:
        return self.schedule["freshness_hours"]


class CanaryProfileRegistry:
    """Registry of validated canary profiles loaded from a directory."""

    def __init__(self, profiles: Dict[str, CanaryProfile]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def from_directory(cls, directory: Path) -> "CanaryProfileRegistry":
        """Load every ``*.json`` file in *directory*, validate, and index by profile_id."""
        schema = load_schema("canary-profile.schema.json")
        validator = Draft202012Validator(schema)
        profiles: Dict[str, CanaryProfile] = {}

        if not directory.is_dir():
            logger.warning("canary profiles directory does not exist: %s", directory)
            return cls({})

        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ControlError(
                    "PROFILE_LOAD_ERROR",
                    f"cannot read canary profile {path.name}: {exc}",
                ) from exc

            try:
                validator.validate(raw)
            except ValidationError as exc:
                loc = ".".join(str(p) for p in exc.absolute_path)
                raise ControlError(
                    "PROFILE_SCHEMA_INVALID",
                    f"invalid canary profile {path.name} at {loc}: {exc.message}",
                ) from exc

            # Extra safety: reject anything that smells like a mutable ref
            commit = raw["pinned_commit"]
            if _looks_like_mutable_ref(commit):
                raise ControlError(
                    "PROFILE_MUTABLE_REF",
                    f"canary profile {path.name} pinned_commit looks like a mutable ref: {commit!r}",
                )

            pid = raw["profile_id"]
            if pid in profiles:
                raise ControlError(
                    "PROFILE_DUPLICATE",
                    f"duplicate canary profile_id {pid!r} in {path.name}",
                )

            # Filename must match profile_id for discoverability
            expected_filename = f"{pid}.json"
            if path.name != expected_filename:
                raise ControlError(
                    "PROFILE_FILENAME_MISMATCH",
                    f"profile {pid!r} must be in {expected_filename}, not {path.name}",
                )

            profiles[pid] = CanaryProfile(raw)
            logger.info("loaded canary profile %s v%d from %s", pid, raw["profile_version"], path.name)

        return cls(profiles)

    def get(self, profile_id: str) -> Optional[CanaryProfile]:
        return self._profiles.get(profile_id)

    def require(self, profile_id: str) -> CanaryProfile:
        profile = self.get(profile_id)
        if profile is None:
            available = ", ".join(sorted(self._profiles)) or "(none)"
            raise ControlError(
                "PROFILE_NOT_FOUND",
                f"unknown canary profile {profile_id!r}; available: {available}",
            )
        return profile

    def list_profiles(self) -> list[CanaryProfile]:
        return list(self._profiles.values())

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, profile_id: str) -> bool:
        return profile_id in self._profiles
