"""Unit tests for sweep git operations."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from src.sweep.git_ops import (
    MergeCommit,
    _parse_merge_subject,
    get_head,
    get_merge_commits,
    get_release_branch_points,
)


@pytest.fixture
def git_repo():
    """Create a temporary git repo with merge commits for testing."""
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        run = lambda cmd: subprocess.run(cmd, cwd=repo, capture_output=True, check=True)
        run(["git", "init", "-b", "main"])
        run(["git", "config", "user.email", "test@test.com"])
        run(["git", "config", "user.name", "Test"])

        # Initial commit on main
        (repo / "file.txt").write_text("v1")
        run(["git", "add", "."])
        run(["git", "commit", "-m", "Initial commit"])

        # Create a branch and merge it (simulates a PR merge)
        run(["git", "checkout", "-b", "feature-1"])
        (repo / "feature1.txt").write_text("feature 1")
        run(["git", "add", "."])
        run(["git", "commit", "-m", "Add feature 1"])
        run(["git", "checkout", "main"])
        run(
            [
                "git",
                "merge",
                "--no-ff",
                "-m",
                "Merge pull request #101 from user/feature-1",
                "feature-1",
            ]
        )

        # Another merge
        run(["git", "checkout", "-b", "feature-2"])
        (repo / "feature2.txt").write_text("feature 2")
        run(["git", "add", "."])
        run(["git", "commit", "-m", "Add feature 2"])
        run(["git", "checkout", "main"])
        run(
            [
                "git",
                "merge",
                "--no-ff",
                "-m",
                "Merge pull request #202 from user/feature-2",
                "feature-2",
            ]
        )

        yield repo


class TestGetMergeCommits:
    """Tests for merge commit enumeration."""

    def test_finds_merge_commits(self, git_repo):
        commits = get_merge_commits(git_repo)
        # All first-parent commits returned (initial + 2 merges)
        assert len(commits) >= 2
        prs = [c.pr for c in commits if c.pr is not None]
        assert 101 in prs
        assert 202 in prs

    def test_oldest_first(self, git_repo):
        commits = get_merge_commits(git_repo)
        prs = [c.pr for c in commits if c.pr is not None]
        assert prs == [101, 202]

    def test_has_dates(self, git_repo):
        commits = get_merge_commits(git_repo)
        for c in commits:
            assert len(c.date) == 10  # YYYY-MM-DD
            assert c.date[4] == "-"


class TestGetHead:
    """Tests for HEAD resolution."""

    def test_returns_full_hash(self, git_repo):
        head = get_head(git_repo)
        assert len(head) == 40
        assert all(c in "0123456789abcdef" for c in head)


class TestParseMergeSubject:
    """Tests for merge commit subject parsing."""

    def test_standard_merge(self):
        pr, title = _parse_merge_subject(
            "Merge pull request #1847 from user/branch-name"
        )
        assert pr == 1847
        assert title == "Merge pull request #1847 from user/branch-name"

    def test_squash_merge(self):
        pr, title = _parse_merge_subject("Optimize dict rehashing (#2103)")
        assert pr == 2103
        assert title == "Optimize dict rehashing"

    def test_no_pr(self):
        pr, title = _parse_merge_subject("Regular commit message")
        assert pr is None
        assert title == "Regular commit message"

    def test_squash_with_parens_in_title(self):
        pr, title = _parse_merge_subject("Fix bug (important) in module (#999)")
        assert pr == 999
        assert title == "Fix bug (important) in module"


class TestGetReleaseBranchPoints:
    """Tests for release branch point discovery."""

    @pytest.fixture
    def repo_with_release_branch(self):
        """Create a repo with a simulated release branch (like origin/8.0)."""
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            run = lambda cmd: subprocess.run(
                cmd, cwd=repo, capture_output=True, check=True
            )
            run(["git", "init", "-b", "unstable"])
            run(["git", "config", "user.email", "test@test.com"])
            run(["git", "config", "user.name", "Test"])

            # Initial commits on unstable
            (repo / "file.txt").write_text("v1")
            run(["git", "add", "."])
            run(["git", "commit", "-m", "Initial"])

            (repo / "file.txt").write_text("v2")
            run(["git", "add", "."])
            run(["git", "commit", "-m", "Second commit (#1)"])

            # Create release branch at this point
            run(["git", "branch", "8.0"])

            # More commits on unstable after branch point
            (repo / "file.txt").write_text("v3")
            run(["git", "add", "."])
            run(["git", "commit", "-m", "Third commit (#2)"])

            # Set up origin remote (pointing to self for testing)
            run(["git", "remote", "add", "origin", repo.as_posix()])
            run(["git", "fetch", "origin"])

            yield repo

    def test_finds_branch_point(self, repo_with_release_branch):
        points = get_release_branch_points(repo_with_release_branch)
        assert len(points) == 1
        commit_hash, date, label = points[0]
        assert label == "8.0"
        assert len(commit_hash) == 40
        assert len(date) == 10  # YYYY-MM-DD

    def test_branch_point_is_on_unstable(self, repo_with_release_branch):
        points = get_release_branch_points(repo_with_release_branch)
        # The branch point commit should be reachable from unstable
        commit_hash = points[0][0]
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_with_release_branch),
                "merge-base",
                "--is-ancestor",
                commit_hash,
                "unstable",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
