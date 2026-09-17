"""Per-thread and per-core CPU accounting for single-host topology cells.

A topology cell (primary + replicas + generators on one host) scores a
closed-loop reader. If the reader generator, the kernel's loopback path, or a
foreign process on the host is the limit, the score comes out lower and looks
exactly like a server ceiling. Sampling cumulative CPU time for every server
thread, every generator thread and every core lets the task attach a
*bottleneck verdict* to the result instead of an unqualified number.

What CPU time can and cannot say here (measured on a 16-core x86 development host, a
replica with ``--io-threads 2`` under a rate-limited cachecannon reader):

* **Valkey server threads spin.** The I/O thread read 1.00 of a core at 20k
  ops/s (one fifth of what it later sustained) and main read 0.81 at half load,
  because I/O threads busy-wait for jobs and main polls with ``dontwait`` while
  jobs are in flight. Server CPU time is therefore NOT a saturation signal and
  is never thresholded. The main thread's ``eventloop_duration_sum`` (time
  inside loop iterations, excluding the poll) does scale with load
  (0.09/0.23/0.58/0.81 at 5%/20%/50%/100% of the ceiling) and is reported as
  ``loop_duty`` for a human to read; it approaches ~0.8 rather than 1.0 at
  saturation, so it is evidence, not a rule.
* **cachecannon worker threads are honest.** They block in the poller, so
  their CPU time rises with load (0.24/0.40/0.57 across the same steps) and a
  worker at ~1.0 really has no headroom. Generator saturation IS thresholded.
* **Foreign cores are unambiguous.** A core nobody allocated has no business
  being busy.

So the ``server`` verdict means "nothing else is implicated", not "the server
was proven saturated". Proving the ceiling needs a load step (same cell at a
higher connection count showing no gain), which is a separate control cell.

Everything here is a pure function over text or over the list of samples the
topology sampler collects. Counters are cumulative jiffies (``USER_HZ`` is a
fixed 100 on Linux regardless of the kernel's own ``HZ``); utilisation is the
difference between consecutive samples divided by the wall time between them.
"""

import glob
from typing import Optional

USER_HZ = 100

# A generator thread within this much of one full core has no headroom.
GENERATOR_SATURATED_UTIL = 0.95
# A generator above this is close enough that the server verdict is not trusted.
GENERATOR_HOT_UTIL = 0.85
# Mean busy fraction on a core nobody allocated above which the host is not ours.
FOREIGN_BUSY_MAX = 0.05

VERDICT_SERVER = "server"
VERDICT_GENERATOR = "generator"
VERDICT_HOST = "host"
VERDICT_AMBIGUOUS = "ambiguous"
VERDICT_UNKNOWN = "unknown"

PROCSTAT_HEADER = "=== conductress-procstat"
THREADS_HEADER = "=== conductress-threads "

# /proc/stat per-cpu fields after the "cpuN" label.
_STAT_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


# --------------------------------------------------------------------------- shell


def cpu_sample_command(pids_by_port: dict) -> str:
    """Shell snippet emitting per-core /proc/stat lines and per-thread stats.

    ``pids_by_port`` maps a server port to its main pid; servers whose pid is
    unknown (``<= 0``) are skipped. Output sections are introduced by
    ``PROCSTAT_HEADER`` and ``THREADS_HEADER<port>`` so they can share one
    remote shell with the INFO sections.
    """
    parts = [f"echo '{PROCSTAT_HEADER}'; grep '^cpu[0-9]' /proc/stat"]
    for port, pid in sorted(pids_by_port.items()):
        if pid is None or pid <= 0:
            continue
        parts.append(f"echo '{THREADS_HEADER}{port}'; cat /proc/{pid}/task/*/stat 2>/dev/null")
    return "; ".join(parts)


# --------------------------------------------------------------------------- parsing


def parse_proc_stat(lines: list) -> dict:
    """Parse ``cpuN ...`` lines into ``{cpu: {"busy", "idle", "softirq", "total"}}`` (jiffies)."""
    cores: dict = {}
    for raw in lines:
        parts = raw.split()
        if len(parts) < 5 or not parts[0].startswith("cpu") or not parts[0][3:].isdigit():
            continue
        values = [int(v) for v in parts[1 : 1 + len(_STAT_FIELDS)]]
        values += [0] * (len(_STAT_FIELDS) - len(values))
        fields = dict(zip(_STAT_FIELDS, values))
        total = sum(values)
        idle = fields["idle"] + fields["iowait"]
        cores[int(parts[0][3:])] = {
            "busy": total - idle,
            "idle": idle,
            "softirq": fields["softirq"],
            "total": total,
        }
    return cores


def parse_thread_stats(lines: list) -> dict:
    """Parse ``/proc/<pid>/task/<tid>/stat`` lines into ``{tid: {"comm", "ticks"}}``.

    ``comm`` may contain spaces or parentheses, so the line is split at the
    LAST ``)``; after it, ``utime`` and ``stime`` are the 12th and 13th
    whitespace fields (state is the first).
    """
    threads: dict = {}
    for raw in lines:
        head, sep, tail = raw.rpartition(")")
        if not sep or "(" not in head:
            continue
        tid_str, _, comm = head.partition("(")
        fields = tail.split()
        if len(fields) < 13 or not tid_str.strip().isdigit():
            continue
        try:
            ticks = int(fields[11]) + int(fields[12])
        except ValueError:
            continue
        threads[int(tid_str)] = {"comm": comm, "ticks": ticks}
    return threads


def split_sections(output: str, instance_header: str) -> tuple:
    """Split the combined remote output into INFO, procstat and thread sections.

    Returns ``(info_lines, procstat_lines, {port: thread_lines})`` where
    ``info_lines`` keeps the instance headers so the INFO parser is unchanged.
    """
    info_lines: list = []
    procstat: list = []
    threads: dict = {}
    target: Optional[list] = info_lines
    for raw in output.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped == PROCSTAT_HEADER:
            target = procstat
            continue
        if stripped.startswith(THREADS_HEADER):
            port_str = stripped[len(THREADS_HEADER) :].strip()
            target = threads.setdefault(int(port_str), []) if port_str.isdigit() else None
            continue
        if stripped.startswith(instance_header):
            target = info_lines
        if target is not None:
            target.append(line)
    return info_lines, procstat, threads


def read_local_thread_stats(pid: Optional[int]) -> dict:
    """Read ``/proc/<pid>/task/*/stat`` for a process on THIS host (a generator)."""
    if pid is None or pid <= 0:
        return {}
    lines = []
    for path in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines.append(f.read().strip())
        except OSError:
            continue  # thread exited between glob and read
    return parse_thread_stats(lines)


# --------------------------------------------------------------------------- utilisation


def _window(samples: list, window_start: float) -> list:
    return [s for s in samples if s.get("t", 0.0) >= window_start]


def thread_utilization(samples: list, getter, time_key: str = "t") -> dict:
    """Mean/max per-thread utilisation (fraction of one core) over consecutive sample pairs.

    ``getter(sample)`` returns that sample's ``{tid: {"comm", "ticks"}}`` (or
    ``None``). A thread only counts on intervals where it appears in both
    samples, so generator worker threads that spawn late are handled.
    """
    per_thread: dict = {}
    for prev, cur in zip(samples, samples[1:]):
        a, b = getter(prev), getter(cur)
        if not a or not b:
            continue
        dt = cur.get(time_key, 0.0) - prev.get(time_key, 0.0)
        if dt <= 0:
            continue
        for tid, info in b.items():
            if tid not in a:
                continue
            util = (info["ticks"] - a[tid]["ticks"]) / USER_HZ / dt
            entry = per_thread.setdefault(tid, {"comm": info["comm"], "utils": []})
            entry["utils"].append(max(0.0, util))
    return {
        tid: {
            "comm": e["comm"],
            "mean": sum(e["utils"]) / len(e["utils"]),
            "max": max(e["utils"]),
            "intervals": len(e["utils"]),
        }
        for tid, e in per_thread.items()
        if e["utils"]
    }


def core_utilization(samples: list) -> dict:
    """Mean busy and softirq fractions per core over consecutive sample pairs."""
    acc: dict = {}
    for prev, cur in zip(samples, samples[1:]):
        a, b = prev.get("cores"), cur.get("cores")
        if not a or not b:
            continue
        for cpu, now in b.items():
            then = a.get(cpu)
            if then is None:
                continue
            total = now["total"] - then["total"]
            if total <= 0:
                continue
            entry = acc.setdefault(cpu, {"busy": [], "softirq": []})
            entry["busy"].append((now["busy"] - then["busy"]) / total)
            entry["softirq"].append((now["softirq"] - then["softirq"]) / total)
    return {
        cpu: {
            "busy_mean": sum(e["busy"]) / len(e["busy"]),
            "busy_max": max(e["busy"]),
            "softirq_mean": sum(e["softirq"]) / len(e["softirq"]),
        }
        for cpu, e in acc.items()
        if e["busy"]
    }


def eventloop_duty(samples: list, port: int) -> dict:
    """Main-thread duty from INFO eventloop counters over consecutive sample pairs.

    ``loop_duty`` is seconds inside event-loop iterations per wall second
    (excludes the poll, so not spin-inflated); ``cmd_duty`` is the command
    execution share of that; ``cycles_per_sec`` is the iteration rate. A high
    cycle rate with low duty is main polling with ``dontwait`` for in-flight
    I/O jobs, which is exactly the state where CPU time reads 100%.
    """
    loop: list = []
    cmd: list = []
    cycles: list = []
    for prev, cur in zip(samples, samples[1:]):
        a = (prev.get("instances") or {}).get(port) or {}
        b = (cur.get("instances") or {}).get(port) or {}
        dt = cur.get("t", 0.0) - prev.get("t", 0.0)
        if dt <= 0 or a.get("eventloop_duration_sum") is None or b.get("eventloop_duration_sum") is None:
            continue
        loop.append(max(0, b["eventloop_duration_sum"] - a["eventloop_duration_sum"]) / 1e6 / dt)
        if a.get("eventloop_duration_cmd_sum") is not None and b.get("eventloop_duration_cmd_sum") is not None:
            cmd.append(max(0, b["eventloop_duration_cmd_sum"] - a["eventloop_duration_cmd_sum"]) / 1e6 / dt)
        if a.get("eventloop_cycles") is not None and b.get("eventloop_cycles") is not None:
            cycles.append(max(0, b["eventloop_cycles"] - a["eventloop_cycles"]) / dt)
    return {
        "loop_duty": sum(loop) / len(loop) if loop else None,
        "cmd_duty": sum(cmd) / len(cmd) if cmd else None,
        "cycles_per_sec": sum(cycles) / len(cycles) if cycles else None,
    }


def _server_summary(threads: dict, pid: Optional[int], loop: Optional[dict] = None) -> dict:
    main = threads.get(pid) if pid else None
    io_threads = {tid: t for tid, t in threads.items() if t["comm"].startswith("io_thd")}
    other = {tid: t for tid, t in threads.items() if tid != pid and tid not in io_threads}
    io_max = max((t["mean"] for t in io_threads.values()), default=None)
    summary = {
        # CPU time of main/io threads is spin-inflated (see module docstring): informational only.
        "main_cpu": main["mean"] if main else None,
        "io_threads": len(io_threads),
        "io_threads_cpu_max": io_max,
        "io_threads_cpu_mean": (sum(t["mean"] for t in io_threads.values()) / len(io_threads) if io_threads else None),
        "other_threads_cpu_max": max((t["mean"] for t in other.values()), default=None),
        "hottest_thread": _hottest(threads),
    }
    summary.update(loop or {"loop_duty": None, "cmd_duty": None, "cycles_per_sec": None})
    return summary


def _hottest(threads: dict) -> Optional[dict]:
    if not threads:
        return None
    tid, t = max(threads.items(), key=lambda kv: kv[1]["mean"])
    return {"tid": tid, "comm": t["comm"], "mean": t["mean"], "max": t["max"]}


def _generator_summary(threads: dict) -> dict:
    return {
        "threads": len(threads),
        "max_thread_util": max((t["mean"] for t in threads.values()), default=None),
        "mean_thread_util": (sum(t["mean"] for t in threads.values()) / len(threads) if threads else None),
        "hottest_thread": _hottest(threads),
    }


def _mean_over(cores: dict, cpus: list, key: str) -> Optional[float]:
    vals = [cores[c][key] for c in cpus if c in cores]
    return sum(vals) / len(vals) if vals else None


def bottleneck_verdict(
    samples: list,
    *,
    replica_port: int,
    replica_pid: Optional[int],
    primary_port: int,
    primary_pid: Optional[int],
    allocated_cpus: dict,
    window_start: float = 0.0,
) -> dict:
    """Decide what bounded the score and return the evidence alongside.

    ``allocated_cpus`` maps a label (``"primary"``, ``"replica"``, ``"reader"``,
    ``"writer"``, further replicas as ``"replica:<port>"``) to its CPU list.
    Cores in none of the lists are *foreign*: anything busy there is not ours.
    An empty ``"reader"`` list means the generators were not pinned by the
    allocator, so foreign-core detection is skipped rather than guessed.

    Verdicts, in precedence order:

    * ``host``: a core nobody allocated is busy. Interference; invalid.
    * ``generator``: a reader thread is saturated. Invalid as a server score.
    * ``ambiguous``: the reader is hot but not pinned; the server may or may not
      have given first.
    * ``server``: nothing else is implicated, so the score is the server's
      number. NOT a proof of server saturation (server CPU time spins, see the
      module docstring); ``replica.loop_duty`` is attached as the human-readable
      evidence, and a load-step control cell is what confirms a ceiling.
    * ``unknown``: too few samples in the measurement window to say anything.
    """
    window = _window(samples, window_start)
    thresholds = {
        "generator_saturated_util": GENERATOR_SATURATED_UTIL,
        "generator_hot_util": GENERATOR_HOT_UTIL,
        "foreign_busy_max": FOREIGN_BUSY_MAX,
    }
    if len(window) < 2:
        return {
            "verdict": VERDICT_UNKNOWN,
            "valid": False,
            "reason": f"{len(window)} samples in the measurement window; need at least 2",
            "samples": len(window),
            "thresholds": thresholds,
        }

    replica = _server_summary(
        thread_utilization(window, lambda s: (s.get("threads") or {}).get(replica_port)),
        replica_pid,
        eventloop_duty(window, replica_port),
    )
    primary = _server_summary(
        thread_utilization(window, lambda s: (s.get("threads") or {}).get(primary_port)),
        primary_pid,
        eventloop_duty(window, primary_port),
    )
    reader = _generator_summary(
        thread_utilization(window, lambda s: (s.get("generators") or {}).get("reader"), time_key="t_local")
    )
    writer = _generator_summary(
        thread_utilization(window, lambda s: (s.get("generators") or {}).get("writer"), time_key="t_local")
    )
    cores = core_utilization(window)

    claimed = {cpu for cpus in allocated_cpus.values() for cpu in cpus}
    reader_cpus = list(allocated_cpus.get("reader") or [])
    foreign_cpus = sorted(c for c in cores if c not in claimed) if reader_cpus else []
    foreign_busy = {c: cores[c]["busy_mean"] for c in foreign_cpus if cores[c]["busy_mean"] > FOREIGN_BUSY_MAX}
    core_summary = {
        "sampled": len(cores),
        "foreign_checked": bool(reader_cpus),
        "foreign_cores": len(foreign_cpus),
        "foreign_busy_max": max(foreign_busy.values(), default=(0.0 if foreign_cpus else None)),
        "foreign_busy_cores": {str(c): round(v, 4) for c, v in sorted(foreign_busy.items())},
        "replica_softirq_share": _mean_over(cores, list(allocated_cpus.get("replica") or []), "softirq_mean"),
        "reader_softirq_share": _mean_over(cores, reader_cpus, "softirq_mean"),
        "replica_busy_mean": _mean_over(cores, list(allocated_cpus.get("replica") or []), "busy_mean"),
    }

    reader_peak = reader["max_thread_util"] or 0.0
    duty = replica["loop_duty"]
    duty_text = f"{duty:.2f}" if duty is not None else "n/a"

    if foreign_busy:
        verdict, reason = (
            VERDICT_HOST,
            f"{len(foreign_busy)} unallocated core(s) busy: {core_summary['foreign_busy_cores']}",
        )
    elif reader_peak >= GENERATOR_SATURATED_UTIL:
        verdict, reason = VERDICT_GENERATOR, f"reader thread at {reader_peak:.2f} of a core; add reader threads"
    elif reader_peak >= GENERATOR_HOT_UTIL:
        verdict, reason = VERDICT_AMBIGUOUS, f"reader thread hot ({reader_peak:.2f}); replica loop duty {duty_text}"
    else:
        verdict, reason = (
            VERDICT_SERVER,
            f"reader peak {reader_peak:.2f}, no foreign activity; replica main loop duty {duty_text} "
            "(server CPU time spins; ceiling needs a load step to confirm)",
        )

    return {
        "verdict": verdict,
        "valid": verdict == VERDICT_SERVER,
        "reason": reason,
        "samples": len(window),
        "replica": replica,
        "primary": primary,
        "reader": reader,
        "writer": writer,
        "cores": core_summary,
        "thresholds": thresholds,
    }
