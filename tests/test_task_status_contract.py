"""Verify all task runners report progress via FileProtocol.

Every BaseTaskRunner subclass must call self.file_protocol.write_status()
in its run() method. This prevents silent "idle" status on the dashboard
when a task is actually running.
"""

import inspect

import pytest

from conductress.task_queue import BaseTaskRunner
from conductress.tasks.task_latency import LatencyTaskRunner  # noqa: F401
from conductress.tasks.task_mem_efficiency import MemTaskRunner  # noqa: F401
from conductress.tasks.task_perf_benchmark import PerfTaskRunner  # noqa: F401


def _all_concrete_runner_subclasses(base):
    discovered = set()
    pending = list(base.__subclasses__())
    while pending:
        runner = pending.pop()
        pending.extend(runner.__subclasses__())
        if not inspect.isabstract(runner) and runner.__module__.startswith("conductress."):
            discovered.add(runner)
    return sorted(discovered, key=lambda runner: runner.__name__)


ALL_RUNNERS = _all_concrete_runner_subclasses(BaseTaskRunner)


@pytest.mark.parametrize("runner_cls", ALL_RUNNERS, ids=lambda c: c.__name__)
def test_runner_writes_status_in_run(runner_cls):
    """Each task runner must call self.file_protocol.write_status() in run()."""
    source = inspect.getsource(runner_cls.run)
    assert "file_protocol.write_status" in source, (
        f"{runner_cls.__name__}.run() does not call self.file_protocol.write_status(). "
        f"All task runners must report progress for the status dashboard."
    )


@pytest.mark.parametrize("runner_cls", ALL_RUNNERS, ids=lambda c: c.__name__)
def test_runner_sets_status_with_steps(runner_cls):
    """Each task runner must initialize BenchmarkStatus with steps_total > 0."""
    source = inspect.getsource(runner_cls.run)
    assert "BenchmarkStatus(" in source or "self.status" in source, (
        f"{runner_cls.__name__}.run() does not create a BenchmarkStatus. "
        f"Status must be initialized with steps_total for progress tracking."
    )
