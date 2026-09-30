"""User and agent identity directory read from a control-plane-local TOML file.

The directory maps a login to a person or agent, the GitHub account it acts
under, its role, its daily runner-minute quota, and the build sources it may
name. Bearer tokens (see :mod:`conductress.control.auth`) reference a login;
this module turns that login into the caller's full identity so the submit
path can apply the provenance gate, quota, and approval rules.

Quota is keyed on the GitHub account, not the login: an agent draws on its
sponsor's account, so an agent and the human it acts for share one budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as _toml  # type: ignore[no-redef]


VALID_KINDS = {"human", "agent"}
VALID_ROLES = {"owner", "collaborator", "approver"}


@dataclass(frozen=True)
class User:
    """One entry in the user directory."""

    login: str
    github: str
    kind: str
    role: str
    quota_runner_minutes_per_day: int
    sources: tuple[str, ...] = ()
    sponsor: Optional[str] = None

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    @property
    def is_approver(self) -> bool:
        # Owners are implicitly approvers; the approval flow treats them alike.
        return self.role in {"approver", "owner"}


@dataclass(frozen=True)
class UserDirectory:
    """All users keyed by login, with quota resolved per GitHub account."""

    users: dict[str, User] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "UserDirectory":
        with path.open("rb") as stream:
            data = _toml.load(stream)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "UserDirectory":
        raw_users = data.get("users")
        if not isinstance(raw_users, list):
            raise ValueError("users file must contain a [[users]] array")
        users: dict[str, User] = {}
        for record in raw_users:
            user = _parse_user(record)
            if user.login in users:
                raise ValueError(f"duplicate user login: {user.login}")
            users[user.login] = user
        _validate_sponsors(users)
        return cls(users=users)

    def get(self, login: str) -> Optional[User]:
        return self.users.get(login)

    def require(self, login: str) -> User:
        user = self.users.get(login)
        if user is None:
            raise KeyError(f"unknown user login: {login}")
        return user

    def quota_account(self, login: str) -> str:
        """Return the GitHub account a login's runner-minutes are charged to.

        An agent's usage is charged to its sponsor's GitHub account; a human's
        usage is charged to its own. The sponsor chain is one level deep (an
        agent's sponsor must be a human), enforced at load time.
        """
        user = self.require(login)
        if user.kind == "agent" and user.sponsor is not None:
            return self.require(user.sponsor).github
        return user.github

    def quota_minutes(self, login: str) -> int:
        """Return the daily runner-minute quota for a login's charging account."""
        user = self.require(login)
        if user.kind == "agent" and user.sponsor is not None:
            return self.require(user.sponsor).quota_runner_minutes_per_day
        return user.quota_runner_minutes_per_day


def _parse_user(record: object) -> User:
    if not isinstance(record, dict):
        raise ValueError("each user entry must be a table")
    allowed = {
        "login",
        "github",
        "kind",
        "role",
        "quota_runner_minutes_per_day",
        "sources",
        "sponsor",
    }
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f"unknown user fields: {', '.join(sorted(unknown))}")
    login = record.get("login")
    github = record.get("github")
    kind = record.get("kind")
    role = record.get("role")
    quota = record.get("quota_runner_minutes_per_day")
    sources = record.get("sources", [])
    sponsor = record.get("sponsor")
    if not isinstance(login, str) or not login:
        raise ValueError("user login must be a non-empty string")
    if not isinstance(github, str) or not github:
        raise ValueError(f"user {login}: github must be a non-empty string")
    if kind not in VALID_KINDS:
        raise ValueError(f"user {login}: kind must be one of {sorted(VALID_KINDS)}")
    if role not in VALID_ROLES:
        raise ValueError(f"user {login}: role must be one of {sorted(VALID_ROLES)}")
    if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
        raise ValueError(f"user {login}: quota_runner_minutes_per_day must be a non-negative integer")
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        raise ValueError(f"user {login}: sources must be a list of strings")
    if kind == "agent" and not sponsor:
        raise ValueError(f"user {login}: an agent must name a sponsor")
    if kind == "human" and sponsor is not None:
        raise ValueError(f"user {login}: a human must not name a sponsor")
    if sponsor is not None and not isinstance(sponsor, str):
        raise ValueError(f"user {login}: sponsor must be a string")
    return User(
        login=login,
        github=github,
        kind=kind,
        role=role,
        quota_runner_minutes_per_day=quota,
        sources=tuple(sources),
        sponsor=sponsor,
    )


def _validate_sponsors(users: dict[str, User]) -> None:
    for user in users.values():
        if user.sponsor is None:
            continue
        sponsor = users.get(user.sponsor)
        if sponsor is None:
            raise ValueError(f"user {user.login}: sponsor {user.sponsor} is not a known user")
        if sponsor.kind != "human":
            raise ValueError(f"user {user.login}: sponsor {user.sponsor} must be a human account")
