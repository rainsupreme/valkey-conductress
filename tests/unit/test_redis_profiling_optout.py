"""Tests for the Redis profiling opt-out.

CPU flamegraphs and jemalloc allocation breakdowns expose the server binary's
symbols, so engines can opt out via SweepEngine.profile_internals (Redis does).
Redis keeps aggregate performance data (throughput/latency/total memory) only.
"""

from unittest.mock import MagicMock, patch

from conductress.config import get_sweep_engine, should_profile_internals
from conductress.publisher import DashboardPublisher


class TestShouldProfileInternals:
    def test_redis_opts_out(self):
        redis = get_sweep_engine("redis")
        assert redis.profile_internals is False
        assert should_profile_internals(redis) is False

    def test_valkey_profiles(self):
        assert should_profile_internals(get_sweep_engine("valkey")) is True

    def test_unknown_engine_defaults_on(self):
        # A fork (get_sweep_engine returns None) or legacy state must keep profiling.
        assert should_profile_internals(get_sweep_engine("valkey-rainfall")) is True
        assert should_profile_internals(None) is True


def _throughput_coord(source, profile_internals):
    coord = MagicMock()
    coord.workload_id = "get-k16-v16-t7-p10"
    coord.metric_id = "throughput"
    coord.state = MagicMock()
    coord.export.return_value = 1
    coord._sweep_ref = "origin/unstable"
    coord.engine = MagicMock(source=source, profile_internals=profile_internals)
    return coord


class TestPublisherCpuStackGating:
    @patch("conductress.utility.subprocess.run")
    def test_cpu_stacks_skipped_for_optout_engine(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        pub = DashboardPublisher("user@host:/path", [_throughput_coord("redis", False)])
        with (
            patch("conductress.sweep.exporter.export_perf_metrics"),
            patch("conductress.sweep.exporter.export_manifest"),
            patch("conductress.sweep.exporter.export_cpu_profile") as mprof,
            patch("conductress.sweep.exporter.export_cpu_stacks_raw") as mstacks,
        ):
            pub.on_task_completed(MagicMock())
        mprof.assert_not_called()
        mstacks.assert_not_called()

    @patch("conductress.utility.subprocess.run")
    def test_cpu_stacks_exported_for_valkey(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        pub = DashboardPublisher("user@host:/path", [_throughput_coord("valkey", True)])
        with (
            patch("conductress.sweep.exporter.export_perf_metrics"),
            patch("conductress.sweep.exporter.export_manifest"),
            patch("conductress.sweep.exporter.export_cpu_profile") as mprof,
            patch("conductress.sweep.exporter.export_cpu_stacks_raw") as mstacks,
        ):
            pub.on_task_completed(MagicMock())
        mprof.assert_called_once()
        mstacks.assert_called_once()
