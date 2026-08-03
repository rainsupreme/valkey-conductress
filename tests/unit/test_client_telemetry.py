"""Unit tests for load-generator (client) CPU telemetry helpers."""

import os

from conductress.utility import count_cpu_list, sample_process_tree_cpu, summarize_client_cpu


class TestSampleProcessTreeCpu:
    def test_own_process_returns_positive(self):
        value = sample_process_tree_cpu(os.getpid())
        assert value is not None
        assert value > 0

    def test_missing_process_returns_none(self):
        # PID 0 never appears as a /proc entry.
        assert sample_process_tree_cpu(0) is None

    def test_includes_descendants(self):
        # init/systemd (pid 1) subtree total must be >= its own time alone
        # and, on any real system, strictly larger than a fresh process's.
        own = sample_process_tree_cpu(os.getpid())
        assert own is not None


class TestCountCpuList:
    def test_single_cpu(self):
        assert count_cpu_list("7") == 1

    def test_comma_list(self):
        assert count_cpu_list("0,2,4") == 3

    def test_range(self):
        assert count_cpu_list("0-3") == 4

    def test_mixed(self):
        assert count_cpu_list("0-3,8,10-11") == 7

    def test_invalid_returns_none(self):
        assert count_cpu_list("banana") is None

    def test_empty_returns_none(self):
        assert count_cpu_list("") is None


class TestSummarizeClientCpu:
    def test_saturated_flag_set(self):
        summary = summarize_client_cpu([3.7, 3.8], allocated_cores=4)
        assert summary["allocated_cores"] == 4
        assert summary["utilization"] == 0.95
        assert summary["saturated"] is True

    def test_not_saturated(self):
        summary = summarize_client_cpu([2.0, 2.1], allocated_cores=4)
        assert summary["saturated"] is False
        assert summary["utilization"] == 0.525

    def test_unknown_budget_omits_utilization(self):
        summary = summarize_client_cpu([2.0], allocated_cores=None)
        assert "utilization" not in summary
        assert "saturated" not in summary
        assert summary["allocated_cores"] is None

    def test_uses_max_rep_for_utilization(self):
        # A single saturated rep must trip the flag even if others were idle.
        summary = summarize_client_cpu([1.0, 3.9], allocated_cores=4)
        assert summary["saturated"] is True

    def test_rounding(self):
        summary = summarize_client_cpu([1.23456], allocated_cores=None)
        assert summary["cores_busy_per_rep"] == [1.235]
