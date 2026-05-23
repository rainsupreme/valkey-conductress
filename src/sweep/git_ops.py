"""Local git operations for sweep history analysis.

These run locally via subprocess to enumerate and understand commit history
for the sweep planner. They are distinct from the remote git operations in
server.py (fetch, checkout, build) which run via asyncssh on benchmark hosts.

Functions here answer "what is the history structure?" while server.py answers
"prepare a specific commit for benchmarking on a remote host."
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class MergeCommit:
    """A merge commit with its hash, date, and optional PR info."""

    hash: str
    date: str  # YYYY-MM-DD
    pr: Optional[int] = None
    pr_title: Optional[str] = None


def get_merge_commits(
    repo_path: Path, since_commit: Optional[str] = None, ref: str = "HEAD"
) -> List[MergeCommit]:
    """Enumerate first-parent commits (PRs) on the given ref.

    Valkey uses squash merges, so PRs appear as single-parent commits with
    (#NNNN) in the subject. We enumerate all first-parent commits and parse
    PR info from the subject line.

    Args:
        repo_path: Path to the git repository.
        since_commit: If provided, only return commits after this one (exclusive).
        ref: Git ref to enumerate (e.g. "origin/unstable", "HEAD", "main").

    Returns:
        List of MergeCommit, oldest first.
    """
    cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--first-parent",
        "--format=%H %aI %s",
        "--reverse",
    ]
    if since_commit:
        cmd.append(f"{since_commit}..{ref}")
    else:
        cmd.append(ref)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        hash_val, date_str, subject = parts
        date = date_str[:10]  # YYYY-MM-DD from ISO format
        pr, pr_title = _parse_merge_subject(subject)
        commits.append(MergeCommit(hash=hash_val, date=date, pr=pr, pr_title=pr_title))
    return commits


def get_head(repo_path: Path, ref: str = "HEAD") -> str:
    """Get the latest commit hash for the given ref."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_release_branch_points(repo_path: Path) -> List[Tuple[str, str, str]]:
    """Find where release branches diverged from unstable.

    Returns:
        List of (commit_hash, date, branch_label) tuples for commits on
        unstable where each release branch was cut.
    """
    # Discover release branches (e.g. origin/7.2, origin/8.0, origin/8.1)
    result = subprocess.run(
        ["git", "-C", str(repo_path), "branch", "-r", "--list", "origin/[0-9]*"],
        capture_output=True,
        text=True,
        check=True,
    )
    branches = []
    for line in result.stdout.strip().splitlines():
        branch = line.strip()
        if branch and re.match(r"origin/\d+\.\d+$", branch):
            branches.append(branch)

    points = []
    for branch in sorted(branches):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "merge-base", "origin/unstable", branch],
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            # Get date
            date_result = subprocess.run(
                ["git", "-C", str(repo_path), "log", "-1", "--format=%aI", commit],
                capture_output=True,
                text=True,
                check=True,
            )
            date = date_result.stdout.strip()[:10]
            label = branch.replace("origin/", "")
            points.append((commit, date, label))
        except subprocess.CalledProcessError:
            continue
    return points


def find_fork_point(
    repo_path: Path, branch: str = "unstable", upstream: str = "origin"
) -> Optional[str]:
    """Find the fork point where the branch diverged from a known base.

    For Valkey, this is approximately where it forked from Redis.
    Falls back to the oldest merge commit if merge-base fails.
    """
    # Try to find merge-base with a known old tag
    for tag in ["7.2.4", "7.2.3", "7.2.0"]:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "merge-base",
                    f"{upstream}/{branch}",
                    tag,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            continue
    return None


def _resolve_to_full_hash(repo_path: Path, short_hash: str) -> Optional[str]:
    """Resolve a short hash or ref to a full commit hash."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", short_hash],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _parse_merge_subject(subject: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse PR number and title from a merge commit subject.

    Valkey merge commits look like:
        "Merge pull request #1847 from user/branch"
    or GitHub squash merges:
        "Title of PR (#1847)"
    """
    # Pattern: "Merge pull request #NNNN from ..."
    match = re.match(r"Merge pull request #(\d+) from", subject)
    if match:
        return int(match.group(1)), subject

    # Pattern: "Title (#NNNN)"
    match = re.search(r"\(#(\d+)\)$", subject)
    if match:
        title = subject[: match.start()].strip()
        return int(match.group(1)), title

    return None, subject
