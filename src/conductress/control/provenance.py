"""Provenance gate: accept a benchmark SHA only from a trusted origin.

A submission names a ``provenance`` block (repo, sha, optional pull request).
The gate accepts the SHA only when it is reachable from an allowlisted
repository, or when it is the current head of an open pull request against the
upstream project. Anything else is rejected with a reason so the caller can see
why. An owner may bypass the gate with an explicit flag.

GitHub reachability is reached through an injected lookup object so the gate is
unit-testable offline and the network client can be swapped without touching
the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

UPSTREAM_REPO = "valkey-io/valkey"
# Repositories always trusted regardless of the user directory. Per-user forks
# are added from each user's declared sources at gate construction.
DEFAULT_ALLOWLIST = ("valkey-io/valkey", "valkey-rainfall/valkey")


class GitHubLookup(Protocol):
    """Reachability queries the gate needs, injected for offline testing."""

    def commit_in_repo(self, repo: str, sha: str) -> bool:
        """Return True if ``sha`` is reachable in ``repo``."""

    def open_pull_request_for_head(self, sha: str) -> Optional[dict[str, Any]]:
        """Return an open PR against the upstream project whose head is ``sha``.

        The mapping carries ``repo``, ``number``, and ``head_sha``; None means
        no matching open pull request.
        """


class RestGitHubLookup:
    """GitHub reachability over the public REST API using only the standard library.

    Kept dependency-free so the control service does not pull in an HTTP client.
    A token, when supplied, raises the anonymous rate limit; the queries are
    read-only either way.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        upstream_repo: str = "valkey-io/valkey",
        api_base: str = "https://api.github.com",
        timeout_seconds: float = 10.0,
        opener: Any = None,
    ):
        self._token = token
        self._upstream_repo = upstream_repo
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout_seconds
        # urlopen by default; injectable for tests.
        from urllib.request import urlopen  # local import keeps module import light

        self._opener = opener or urlopen

    def _get(self, path: str) -> Any:
        import json as _json
        from urllib.error import HTTPError
        from urllib.request import Request

        headers = {"Accept": "application/vnd.github+json", "User-Agent": "conductress-control"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(f"{self._api_base}/{path.lstrip('/')}", headers=headers, method="GET")
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return _json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def commit_in_repo(self, repo: str, sha: str) -> bool:
        # A 200 on the commit endpoint means the object exists in that repo.
        return self._get(f"repos/{repo}/commits/{sha}") is not None

    def open_pull_request_for_head(self, sha: str) -> Optional[dict[str, Any]]:
        results = self._get(f"repos/{self._upstream_repo}/commits/{sha}/pulls")
        if not results:
            return None
        for pull in results:
            if pull.get("state") == "open" and (pull.get("head") or {}).get("sha") == sha:
                return {
                    "repo": self._upstream_repo,
                    "number": pull.get("number"),
                    "head_sha": sha,
                }
        return None


@dataclass(frozen=True)
class ProvenanceVerdict:
    accepted: bool
    reason: str
    # When acceptance came from an open pull request, the resolved PR block that
    # should be stored on the envelope: {repo, number, head_sha}.
    pull_request: Optional[dict[str, Any]] = None
    # True when acceptance was granted only by an owner's explicit bypass.
    bypassed: bool = False


class ProvenanceGate:
    def __init__(
        self,
        github_lookup: Optional[GitHubLookup] = None,
        *,
        allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST,
        upstream_repo: str = UPSTREAM_REPO,
    ):
        self._github = github_lookup
        self._allowlist = tuple(dict.fromkeys(allowlist))
        self._upstream_repo = upstream_repo

    @property
    def allowlist(self) -> tuple[str, ...]:
        return self._allowlist

    def with_extra_repos(self, repos: tuple[str, ...]) -> "ProvenanceGate":
        """Return a gate whose allowlist also trusts ``repos`` (e.g. a user's forks)."""
        combined = self._allowlist + tuple(repos)
        return ProvenanceGate(self._github, allowlist=combined, upstream_repo=self._upstream_repo)

    def evaluate(self, provenance: Optional[dict[str, Any]], *, owner_bypass: bool = False) -> ProvenanceVerdict:
        """Decide whether a submission's provenance is acceptable.

        Order: an owner bypass is honoured first (and recorded). Otherwise the
        SHA must be reachable from an allowlisted repository, or be the head of
        an open upstream pull request. Missing provenance is rejected unless an
        owner bypasses.
        """
        if owner_bypass:
            return ProvenanceVerdict(True, "owner bypass", bypassed=True)

        if not provenance:
            return ProvenanceVerdict(False, "provenance is required")

        repo = provenance.get("repo")
        sha = provenance.get("sha")
        if not isinstance(repo, str) or not repo or not isinstance(sha, str) or not sha:
            return ProvenanceVerdict(False, "provenance must name repo and sha")

        if self._github is None:
            return ProvenanceVerdict(False, "provenance verification is unavailable")

        if repo in self._allowlist:
            if self._github.commit_in_repo(repo, sha):
                return ProvenanceVerdict(True, f"reachable from allowlisted repo {repo}")
            # Fall through to the pull-request check: an allowlisted repo that
            # does not (yet) contain the SHA may still match an open PR head.

        pull_request = self._github.open_pull_request_for_head(sha)
        if pull_request is not None:
            resolved = {
                "repo": pull_request.get("repo", self._upstream_repo),
                "number": pull_request.get("number"),
                "head_sha": pull_request.get("head_sha", sha),
            }
            return ProvenanceVerdict(True, "head of an open upstream pull request", pull_request=resolved)

        if repo not in self._allowlist:
            return ProvenanceVerdict(
                False,
                f"repo {repo} is not allowlisted and sha is not an open upstream pull request head",
            )
        return ProvenanceVerdict(
            False,
            f"sha {sha} is not reachable from {repo} and is not an open upstream pull request head",
        )
