"""Unit tests for cpu_sampling: /proc parsing, utilisation math, bottleneck verdicts."""

from conductress import cpu_sampling as cs
from conductress.topology import _SAMPLE_SEPARATOR


def _stat_line(cpu: int, user: int, system: int, idle: int, softirq: int = 0) -> str:
    # user nice system idle iowait irq softirq steal guest guest_nice
    return f"cpu{cpu} {user} 0 {system} {idle} 0 0 {softirq} 0 0 0"


def _thread_line(tid: int, comm: str, utime: int, stime: int) -> str:
    # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime ...
    return f"{tid} ({comm}) R 1 1 1 0 -1 4194560 0 0 0 0 {utime} {stime} 0 0 20 0 5 0 100 0 0"


# --- parsing ---


def test_parse_proc_stat_reads_per_core_lines_and_skips_aggregate():
    lines = ["cpu  10 0 10 80 0 0 0 0 0 0", _stat_line(0, 50, 10, 40, softirq=5), _stat_line(7, 0, 0, 100)]
    cores = cs.parse_proc_stat(lines)
    assert set(cores) == {0, 7}
    assert cores[0] == {"busy": 65, "idle": 40, "softirq": 5, "total": 105}
    assert cores[7]["busy"] == 0 and cores[7]["idle"] == 100


def test_parse_thread_stats_handles_spaces_and_parens_in_comm():
    lines = [
        _thread_line(100, "valkey-server", 30, 10),
        _thread_line(101, "io_thd_1", 90, 5),
        "102 (weird (name) here) S 1 1 1 0 -1 0 0 0 0 0 7 3 0 0 20 0 1 0 1 0 0",
        "garbage line",
    ]
    threads = cs.parse_thread_stats(lines)
    assert threads[100] == {"comm": "valkey-server", "ticks": 40}
    assert threads[101] == {"comm": "io_thd_1", "ticks": 95}
    assert threads[102] == {"comm": "weird (name) here", "ticks": 10}
    assert 103 not in threads


def test_cpu_sample_command_skips_unknown_pids():
    cmd = cs.cpu_sample_command({6379: 4242, 6380: -1})
    assert "grep '^cpu[0-9]' /proc/stat" in cmd
    assert "/proc/4242/task/*/stat" in cmd
    assert f"{cs.THREADS_HEADER}6379" in cmd
    assert "6380" not in cmd


def test_split_sections_separates_info_procstat_and_threads():
    out = "\n".join(
        [
            f"{_SAMPLE_SEPARATOR}6379",
            "role:master",
            "master_repl_offset:10",
            f"{_SAMPLE_SEPARATOR}6380",
            "role:slave",
            cs.PROCSTAT_HEADER,
            _stat_line(0, 1, 1, 1),
            f"{cs.THREADS_HEADER}6379",
            _thread_line(1, "valkey-server", 1, 1),
            f"{cs.THREADS_HEADER}6380",
            _thread_line(2, "valkey-server", 2, 2),
            _thread_line(3, "io_thd_1", 3, 3),
        ]
    )
    info, procstat, threads = cs.split_sections(out, _SAMPLE_SEPARATOR)
    assert info[0].startswith(_SAMPLE_SEPARATOR) and "role:slave" in info
    assert procstat == [_stat_line(0, 1, 1, 1)]
    assert set(threads) == {6379, 6380}
    assert len(threads[6380]) == 2


# --- utilisation ---


def _sample(t, threads=None, cores=None, gens=None, t_local=None):
    s = {"t": t, "t_local": t if t_local is None else t_local, "instances": {}}
    if threads is not None:
        s["threads"] = threads
    if cores is not None:
        s["cores"] = cores
    if gens is not None:
        s["generators"] = gens
    return s


def test_thread_utilization_is_ticks_over_user_hz_per_second():
    samples = [
        _sample(0.0, threads={6380: {1: {"comm": "valkey-server", "ticks": 0}}}),
        _sample(2.0, threads={6380: {1: {"comm": "valkey-server", "ticks": 100}}}),  # 1.0s cpu in 2.0s -> 0.5
        _sample(4.0, threads={6380: {1: {"comm": "valkey-server", "ticks": 300}}}),  # 2.0s cpu in 2.0s -> 1.0
    ]
    util = cs.thread_utilization(samples, lambda s: s["threads"][6380])
    assert util[1]["mean"] == 0.75
    assert util[1]["max"] == 1.0
    assert util[1]["intervals"] == 2


def test_thread_utilization_ignores_threads_missing_from_either_sample():
    samples = [
        _sample(0.0, threads={6380: {1: {"comm": "a", "ticks": 0}}}),
        _sample(1.0, threads={6380: {1: {"comm": "a", "ticks": 50}, 2: {"comm": "late", "ticks": 10}}}),
    ]
    util = cs.thread_utilization(samples, lambda s: s["threads"][6380])
    assert set(util) == {1}


def test_core_utilization_busy_and_softirq_fractions():
    samples = [
        _sample(0.0, cores={0: {"busy": 0, "idle": 0, "softirq": 0, "total": 0}}),
        _sample(1.0, cores={0: {"busy": 80, "idle": 20, "softirq": 30, "total": 100}}),
    ]
    cores = cs.core_utilization(samples)
    assert cores[0]["busy_mean"] == 0.8
    assert cores[0]["softirq_mean"] == 0.3


def test_eventloop_duty_from_info_counters():
    samples = [
        {
            "t": 0.0,
            "instances": {6380: {"eventloop_duration_sum": 0, "eventloop_duration_cmd_sum": 0, "eventloop_cycles": 0}},
        },
        {
            "t": 2.0,
            "instances": {
                6380: {
                    "eventloop_duration_sum": 1_600_000,
                    "eventloop_duration_cmd_sum": 40_000,
                    "eventloop_cycles": 500,
                }
            },
        },
    ]
    duty = cs.eventloop_duty(samples, 6380)
    assert duty["loop_duty"] == 0.8
    assert duty["cmd_duty"] == 0.02
    assert duty["cycles_per_sec"] == 250.0
    assert cs.eventloop_duty(samples, 6379) == {"loop_duty": None, "cmd_duty": None, "cycles_per_sec": None}


# --- verdicts ---

REPLICA_PID = 500
PRIMARY_PID = 400
ALLOC = {"primary": [0], "replica": [1, 2], "reader": [3, 4], "writer": [5]}


def _series(reader_util, *, replica_cpu=1.0, loop_duty=0.8, foreign_busy=0.0, n=4, dt=1.0):
    """Build n samples where per-interval utilisation is constant at the given levels.

    The replica's CPU defaults to 1.0 on purpose: that is what a spinning
    io-threads build reads regardless of load, so no test may key off it.
    """
    samples = []
    for i in range(n):
        t = i * dt
        ticks = int(t * cs.USER_HZ)
        threads = {
            6380: {
                REPLICA_PID: {"comm": "valkey-server", "ticks": int(ticks * replica_cpu)},
                REPLICA_PID + 1: {"comm": "io_thd_1", "ticks": int(ticks * replica_cpu)},
                REPLICA_PID + 2: {"comm": "bio_lazy_free", "ticks": 0},
            },
            6379: {PRIMARY_PID: {"comm": "valkey-server", "ticks": int(ticks * 0.3)}},
        }
        cores = {}
        for cpu in range(8):
            if cpu in (1, 2):
                busy = int(ticks * replica_cpu)
            elif cpu in (3, 4):
                busy = int(ticks * reader_util)
            elif cpu >= 6:
                busy = int(ticks * foreign_busy)
            else:
                busy = int(ticks * 0.3)
            cores[cpu] = {"busy": busy, "idle": ticks - busy, "softirq": busy // 10, "total": ticks}
        gens = {
            "reader": {
                900: {"comm": "cachecannon", "ticks": 0},
                901: {"comm": "worker", "ticks": int(ticks * reader_util)},
            },
            "writer": {950: {"comm": "cachecannon", "ticks": int(ticks * 0.2)}},
        }
        instances = {
            6380: {
                "eventloop_duration_sum": int(t * 1e6 * loop_duty),
                "eventloop_duration_cmd_sum": 0,
                "eventloop_cycles": i,
            },
            6379: {
                "eventloop_duration_sum": int(t * 1e6 * 0.1),
                "eventloop_duration_cmd_sum": 0,
                "eventloop_cycles": i,
            },
        }
        s = _sample(t, threads=threads, cores=cores, gens=gens)
        s["instances"] = instances
        samples.append(s)
    return samples


def _verdict(samples, **overrides):
    kwargs = {
        "replica_port": 6380,
        "replica_pid": REPLICA_PID,
        "primary_port": 6379,
        "primary_pid": PRIMARY_PID,
        "allocated_cpus": ALLOC,
    }
    kwargs.update(overrides)
    return cs.bottleneck_verdict(samples, **kwargs)


def test_verdict_server_when_reader_has_headroom_and_host_is_clean():
    v = _verdict(_series(reader_util=0.5))
    assert v["verdict"] == cs.VERDICT_SERVER and v["valid"]
    assert v["reader"]["max_thread_util"] < cs.GENERATOR_HOT_UTIL
    assert v["cores"]["foreign_checked"] and v["cores"]["foreign_busy_max"] == 0.0
    assert "load step" in v["reason"]


def test_verdict_reports_server_evidence_without_thresholding_cpu():
    # Replica CPU reads 1.0 (spin) but that must not appear in the verdict logic; loop duty is attached.
    v = _verdict(_series(reader_util=0.5, replica_cpu=1.0, loop_duty=0.25))
    assert v["verdict"] == cs.VERDICT_SERVER
    assert v["replica"]["main_cpu"] > 0.99 and v["replica"]["io_threads_cpu_max"] > 0.99
    assert v["replica"]["io_threads"] == 1
    assert abs(v["replica"]["loop_duty"] - 0.25) < 1e-6
    assert v["replica"]["hottest_thread"]["comm"] in ("valkey-server", "io_thd_1")
    assert "0.25" in v["reason"]


def test_verdict_generator_when_reader_thread_pinned():
    v = _verdict(_series(reader_util=0.99))
    assert v["verdict"] == cs.VERDICT_GENERATOR and not v["valid"]
    assert "reader thread" in v["reason"]


def test_verdict_ambiguous_when_reader_hot_but_not_pinned():
    v = _verdict(_series(reader_util=0.90))
    assert v["verdict"] == cs.VERDICT_AMBIGUOUS and not v["valid"]


def test_verdict_host_when_unallocated_core_is_busy():
    v = _verdict(_series(reader_util=0.5, foreign_busy=0.5))
    assert v["verdict"] == cs.VERDICT_HOST and not v["valid"]
    assert set(v["cores"]["foreign_busy_cores"]) == {"6", "7"}


def test_verdict_host_outranks_generator():
    v = _verdict(_series(reader_util=0.99, foreign_busy=0.5))
    assert v["verdict"] == cs.VERDICT_HOST


def test_verdict_unknown_with_too_few_samples_in_window():
    samples = _series(reader_util=0.5, n=4)
    v = _verdict(samples, window_start=2.5)  # only the last sample (t=3) survives the warmup cut
    assert v["verdict"] == cs.VERDICT_UNKNOWN and not v["valid"]
    assert v["samples"] == 1


def test_verdict_skips_foreign_check_when_generators_not_pinned():
    alloc = dict(ALLOC, reader=[], writer=[])
    v = _verdict(_series(reader_util=0.5, foreign_busy=0.9), allocated_cpus=alloc)
    assert v["verdict"] == cs.VERDICT_SERVER
    assert not v["cores"]["foreign_checked"]
    assert v["cores"]["foreign_busy_max"] is None


def test_verdict_window_start_excludes_warmup_samples():
    # Reader pinned (0.99) for the first two intervals, then 0.5 for the rest.
    samples = _series(reader_util=0.5, n=6)
    ticks = [0, 99, 198, 248, 298, 348]
    for s, tk in zip(samples, ticks):
        s["generators"]["reader"][901]["ticks"] = tk
    whole = _verdict(samples)
    measured = _verdict(samples, window_start=2.0)
    assert whole["samples"] == 6 and measured["samples"] == 4
    assert abs(whole["reader"]["max_thread_util"] - (0.99 * 2 + 0.5 * 3) / 5) < 1e-9
    assert abs(measured["reader"]["max_thread_util"] - 0.5) < 1e-9
    assert measured["verdict"] == cs.VERDICT_SERVER
