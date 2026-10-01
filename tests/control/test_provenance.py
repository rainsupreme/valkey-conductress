"""Tests for the submit-time provenance gate."""

import pytest

from conductress.control.provenance import DEFAULT_ALLOWLIST, ProvenanceGate


class FakeGitHub:
    """Offline stand-in for GitHub reachability queries."""

    def __init__(self, *, commits=None, open_pr_heads=None):
        # commits: set of (repo, sha) pairs that exist
        self.commits = set(commits or set())
        # open_pr_heads: {sha: {"repo", "number", "head_sha"}}
        self.open_pr_heads = dict(open_pr_heads or {})
        self.commit_queries = []
        self.pr_queries = []

    def commit_in_repo(self, repo, sha):
        self.commit_queries.append((repo, sha))
        return (repo, sha) in self.commits

    def open_pull_request_for_head(self, sha):
        self.pr_queries.append(sha)
        return self.open_pr_heads.get(sha)


def test_accepts_sha_reachable_from_allowlisted_repo():
    github = FakeGitHub(commits={("valkey-io/valkey", "abc")})
    gate = ProvenanceGate(github)
    verdict = gate.evaluate({"repo": "valkey-io/valkey", "sha": "abc"})
    assert verdict.accepted is True
    assert "allowlisted" in verdict.reason


def test_accepts_open_pull_request_head_and_resolves_pr():
    github = FakeGitHub(open_pr_heads={"headsha": {"repo": "valkey-io/valkey", "number": 7, "head_sha": "headsha"}})
    gate = ProvenanceGate(github)
    # A repo that is not allowlisted, but the sha is an open upstream PR head.
    verdict = gate.evaluate({"repo": "someone/fork", "sha": "headsha"})
    assert verdict.accepted is True
    assert verdict.pull_request == {"repo": "valkey-io/valkey", "number": 7, "head_sha": "headsha"}


def test_rejects_unreachable_sha_with_reason():
    github = FakeGitHub()
    gate = ProvenanceGate(github)
    verdict = gate.evaluate({"repo": "valkey-io/valkey", "sha": "ghost"})
    assert verdict.accepted is False
    assert "not reachable" in verdict.reason


def test_rejects_non_allowlisted_repo_without_pr():
    github = FakeGitHub()
    gate = ProvenanceGate(github)
    verdict = gate.evaluate({"repo": "random/repo", "sha": "x"})
    assert verdict.accepted is False
    assert "not allowlisted" in verdict.reason


def test_rejects_missing_provenance():
    gate = ProvenanceGate(FakeGitHub())
    assert gate.evaluate(None).accepted is False
    assert gate.evaluate({"repo": "valkey-io/valkey"}).accepted is False


def test_owner_bypass_accepts_without_lookup():
    github = FakeGitHub()
    gate = ProvenanceGate(github)
    verdict = gate.evaluate({"repo": "random/repo", "sha": "x"}, owner_bypass=True)
    assert verdict.accepted is True
    assert verdict.bypassed is True
    assert github.commit_queries == []
    assert github.pr_queries == []


def test_with_extra_repos_trusts_a_user_fork():
    github = FakeGitHub(commits={("rain/valkey", "forksha")})
    gate = ProvenanceGate(github).with_extra_repos(("rain/valkey",))
    verdict = gate.evaluate({"repo": "rain/valkey", "sha": "forksha"})
    assert verdict.accepted is True
    assert "rain/valkey" in gate.allowlist


def test_default_allowlist_contains_project_repos():
    assert "valkey-io/valkey" in DEFAULT_ALLOWLIST
    assert "valkey-rainfall/valkey" in DEFAULT_ALLOWLIST
