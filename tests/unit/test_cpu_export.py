"""Unit tests for CPU profile export (export_cpu_profile) and idle categorization."""

import json

import pytest

from conductress.cpu_profiler import (
    CPU_CATEGORIES_IO,
    CPU_CATEGORIES_MAIN,
    CPU_CATEGORY_NAMES_IO,
    CPU_CATEGORY_NAMES_MAIN,
    categorize_cpu_stacks,
)
from conductress.sweep.exporter import export_cpu_profile
from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepState

# Realistic collapsed-stack fragments captured from a live armbench sweep.
MAIN_STACKS = [
    ["valkey-server;main;aeMain;beforeSleep;processIOThreadsReadDone;addCommandToBatchAndProcess", 2000],
    ["valkey-server;main;aeMain;processCommand;getCommand;lookupKey;hashtableIncrementalFindStep", 1500],
    ["valkey-server;main;aeMain;processCommand;call;addReplyBulk;_addReplyToBuffer", 1000],
    ["valkey-server;main;aeMain;beforeSleep;processIOThreadsWriteDone", 500],
]
IO_STACKS = [
    ["io_thd_1;thread_start;start_thread;IOThreadMain;IOJobQueue_availableJobs", 6000],
    ["io_thd_2;thread_start;start_thread;IOThreadMain;IOThreadPoll;aeApiPoll.lto_priv.0;epoll_pwait", 3000],
    [
        "io_thd_3;thread_start;start_thread;IOThreadMain;handleClientsWithPendingWrites;connSocketWrite;__libc_write",
        1000,
    ],
]


class TestCategorizeIdle:
    def test_io_idle_dominates(self):
        """IO-thread spin/poll stacks land in the 'idle' band, not networking_io."""
        breakdown = categorize_cpu_stacks(IO_STACKS, CPU_CATEGORIES_IO)
        # 9000 of 10000 samples are idle (job-queue spin + poll wait)
        assert breakdown["idle"] == pytest.approx(90.0, abs=0.01)
        # Real write work categorized as networking_io
        assert breakdown["networking_io"] == pytest.approx(10.0, abs=0.01)

    def test_idle_matched_against_full_stack(self):
        """aeApiPoll appears mid-stack; idle detection must scan the whole stack."""
        stacks = [["valkey-server;main;aeMain;IOThreadPoll;aeApiPoll;epoll_pwait;deep_kernel_frame", 100]]
        breakdown = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)
        assert breakdown["idle"] == pytest.approx(100.0, abs=0.01)

    def test_breakdown_sums_to_100(self):
        for stacks, cats in [(MAIN_STACKS, CPU_CATEGORIES_MAIN), (IO_STACKS, CPU_CATEGORIES_IO)]:
            breakdown = categorize_cpu_stacks(stacks, cats)
            assert sum(breakdown.values()) == pytest.approx(100.0, abs=0.01)

    def test_empty_stacks(self):
        assert categorize_cpu_stacks([], CPU_CATEGORIES_MAIN) == {"other": 100.0}

    def test_category_name_lists_include_idle_and_other(self):
        assert CPU_CATEGORY_NAMES_MAIN[-2:] == ["idle", "other"]
        assert CPU_CATEGORY_NAMES_IO[-2:] == ["idle", "other"]
        assert "hash_lookup" in CPU_CATEGORY_NAMES_MAIN


@pytest.fixture
def state_with_cpu():
    state = SweepState(
        merge_commits=["aaa", "bbb", "ccc"],
        commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        commit_prs={"bbb": 42},
        commit_titles={"bbb": "Optimize pipelining"},
        landmarks=[Landmark(commit="aaa", date="2024-01-01", label="9.0")],
    )
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa",
        date="2024-01-01",
        value=2_000_000.0,
        status=PointStatus.COMPLETED,
        cpu_stacks_main=MAIN_STACKS,
        cpu_stacks_io=IO_STACKS,
    )
    state.points["bbb"] = BenchmarkPoint(
        commit="bbb",
        date="2024-02-01",
        value=2_100_000.0,
        status=PointStatus.COMPLETED,
        cpu_stacks_main=MAIN_STACKS,
        cpu_stacks_io=IO_STACKS,
    )
    # Completed point with NO cpu stacks — should be skipped (sparse/forward-only).
    state.points["ccc"] = BenchmarkPoint(
        commit="ccc", date="2024-03-01", value=2_050_000.0, status=PointStatus.COMPLETED
    )
    return state


class TestExportCpuProfile:
    def test_produces_two_files(self, state_with_cpu, tmp_path):
        exported = export_cpu_profile(state_with_cpu, tmp_path, platform="arm64", workload="get-k16-v16-t7-p10")
        assert exported == {"cpu-main": 2, "cpu-io": 2}
        assert (tmp_path / "series-arm64-get-k16-v16-t7-p10-cpu-main.json").exists()
        assert (tmp_path / "series-arm64-get-k16-v16-t7-p10-cpu-io.json").exists()

    def test_main_series_structure(self, state_with_cpu, tmp_path):
        export_cpu_profile(state_with_cpu, tmp_path, platform="arm64", workload="get-k16-v16-t7-p10")
        data = json.loads((tmp_path / "series-arm64-get-k16-v16-t7-p10-cpu-main.json").read_text())
        assert data["metadata"]["metric"] == "cpu-main"
        assert data["metadata"]["categories"] == CPU_CATEGORY_NAMES_MAIN
        assert data["metadata"]["thread"] == "Main thread"
        # Sparse: only the two points that have stacks
        assert len(data["points"]) == 2
        for p in data["points"]:
            assert sum(p["breakdown"].values()) == pytest.approx(100.0, abs=0.5)
        # PR metadata threaded through
        bbb = next(p for p in data["points"] if p["commit"] == "bbb")
        assert bbb["pr"] == 42
        assert bbb["pr_title"] == "Optimize pipelining"

    def test_io_series_has_idle_band(self, state_with_cpu, tmp_path):
        export_cpu_profile(state_with_cpu, tmp_path, platform="arm64", workload="get-k16-v16-t7-p10")
        data = json.loads((tmp_path / "series-arm64-get-k16-v16-t7-p10-cpu-io.json").read_text())
        for p in data["points"]:
            assert p["breakdown"].get("idle", 0) > 50.0

    def test_no_data_produces_no_files(self, tmp_path):
        empty = SweepState(merge_commits=[], commit_dates={}, commit_prs={}, commit_titles={}, landmarks=[])
        exported = export_cpu_profile(empty, tmp_path, platform="arm64", workload="get-k16-v16-t7-p10")
        assert exported == {}
        assert not list(tmp_path.glob("*.json"))
