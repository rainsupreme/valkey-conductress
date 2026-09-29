"""Peak-memory capture for throughput and scenario cells (Stage 1).

Three layers, from cheapest to most complete, recorded side by side so a
consumer can pick the number that matches the question:

* ``INFO memory`` -- the engine's own view. ``used_memory_peak`` is zmalloc
  bytes sampled once per ``serverCron`` tick (``cronUpdateMemoryStats``), so it
  misses sub-tick spikes, excludes fragmentation and retained pages, and never
  sees the fork child (bgsave / full sync). Treat it as the engine's estimate,
  not the operator number.
* ``/proc/<pid>/status`` ``VmHWM`` -- the kernel's resident-set high-water mark
  for the server process: exact and unsampled, but process-scoped (no fork
  children).
* cgroup ``memory.peak`` (v2) -- the only number that includes fork children.
  In Stage 1 the server usually shares the runner's cgroup, so this reads the
  runner-wide peak; ``cgroup_peak_shared`` flags that. Stage 3 gives each cell
  its own transient scope and this becomes per-server.

Every reader returns ``None`` rather than raising when a field or file is
absent (older Valkey, a remote host, a kernel without ``memory.peak``): a
missing number must never fail a benchmark cell.

Window
------
All three peaks are monotonic since server start, and after a multi-million
key prefill the largest allocation the server ever sees is usually the prefill
itself (hashtable rehash holds two tables at once), so an un-reset peak says
nothing about the load. :func:`reset_memory_peaks` runs at the start of the
scored window (after the generator's warmup) so the recorded numbers describe
the measured load only:

* kernel: writing ``5`` to ``/proc/<pid>/clear_refs`` resets ``VmHWM`` to the
  current ``VmRSS`` (Linux >= 4.0, same uid, no root). Exact and unsampled;
  this is the primary windowed number. ``vm_rss_at_reset`` is kept so
  ``rss_hwm - vm_rss_at_reset`` is the transient.
* engine: ``used_memory_peak`` CANNOT be reset over the wire (``CONFIG
  RESETSTAT`` does not touch ``stat_peak_memory``; only ``initServer`` zeroes
  it). Instead ``used_memory_peak_at_reset`` is recorded and the window's
  engine peak is derived: if the end-of-run peak exceeds the floor, the window
  set a new peak and ``used_memory_peak`` is it (10 Hz cron-sampled); otherwise
  the window never rose above prefill and the best estimate is the 1 Hz
  sampler's max ``used_memory`` over the window. ``used_memory_window_peak``
  plus ``used_memory_window_peak_source`` (``engine`` / ``sampler``) carry that.
* cgroup: ``memory.peak`` is writable only on Linux >= 6.12 and only when the
  cgroup is writable by this user; in Stage 1 that is often false, so the
  field stays best-effort and ``reset.cgroup_peak`` records whether it took.

Each server's record carries a ``reset`` sub-dict saying which resets
succeeded and when, or ``None`` when the task did not reset (whole-run peak).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .stormgen.resp import ReplyParser, encode_command

logger = logging.getLogger(__name__)

# The current shape of the ``data.memory`` sub-dict. Bump when the layout
# changes so a downstream consumer can branch on it.
#   1: per-server peaks + 1 Hz samples, whole-run window
#   2: + per-server ``reset`` record (peaks re-armed at scored-window start)
MEMORY_SCHEMA_VERSION = 2

# INFO memory fields captured into the per-server record. Absent keys (older
# Valkey, or a build without the allocator section) are recorded as None.
_INFO_INT_FIELDS = (
    "used_memory",
    "used_memory_peak",
    "used_memory_rss",
    "mem_clients_normal",
    "mem_replication_backlog",
    "allocator_resident",
    "allocator_active",
)
# frag_ratio is a float, handled separately.
_INFO_FLOAT_FIELDS = ("allocator_frag_ratio",)

# INFO memory fields sampled at 1 Hz for the attribution time series.
_SAMPLE_INT_FIELDS = (
    "used_memory",
    "used_memory_rss",
    "mem_clients_normal",
    "mem_replication_backlog",
)

# Peak fields that are reduced by max() when folding multiple reps together.
PEAK_FIELDS = ("used_memory_peak", "used_memory_window_peak", "rss_hwm", "vm_rss", "used_memory_rss", "cgroup_peak")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_info_memory(fields: Dict[str, str]) -> Dict[str, Optional[float]]:
    """Pull the memory fields we record from an already-parsed INFO dict.

    ``fields`` is what ``Server.info("memory")`` returns (key -> string). A key
    the server did not report becomes ``None`` so a patched or older build
    simply shows the field as null rather than failing.
    """
    record: Dict[str, Optional[float]] = {}
    for name in _INFO_INT_FIELDS:
        record[name] = _to_int(fields.get(name))
    for name in _INFO_FLOAT_FIELDS:
        record[name] = _to_float(fields.get(name))
    return record


def read_info_memory(client: Any) -> Dict[str, Optional[float]]:
    """Read INFO memory from a client exposing ``info_memory()`` -> raw blob.

    Used by the sampler's own socket connection. ``client.info_memory()``
    returns the raw ``INFO memory`` text; we parse it to a field dict and pull
    the recorded keys. A caller that already holds a parsed dict should use
    :func:`parse_info_memory` directly.
    """
    blob = client.info_memory()
    return parse_info_memory(parse_info_blob(blob))


def parse_info_blob(blob: Optional[str]) -> Dict[str, str]:
    """Parse a raw ``INFO`` reply (``key:value`` lines) into a dict."""
    fields: Dict[str, str] = {}
    if not blob:
        return fields
    for line in blob.replace("\r\n", "\n").split("\n"):
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def read_vm_hwm(pid: int) -> Dict[str, Optional[int]]:
    """Return ``{"vm_hwm": bytes, "vm_rss": bytes}`` from /proc/<pid>/status.

    ``VmHWM`` is the process's resident high-water mark, ``VmRSS`` its current
    resident set. Both are reported in kB by the kernel and converted to bytes.
    Either is ``None`` if /proc is unreadable (remote host, dead pid) or the
    line is absent.
    """
    result: Dict[str, Optional[int]] = {"vm_hwm": None, "vm_rss": None}
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return result
    for line in text.split("\n"):
        if line.startswith("VmHWM:"):
            result["vm_hwm"] = _kb_line_to_bytes(line)
        elif line.startswith("VmRSS:"):
            result["vm_rss"] = _kb_line_to_bytes(line)
    return result


def _kb_line_to_bytes(line: str) -> Optional[int]:
    """Convert a ``VmHWM:   12345 kB`` status line to bytes."""
    parts = line.split()
    # parts: ["VmHWM:", "12345", "kB"]
    if len(parts) >= 2:
        kb = _to_int(parts[1])
        if kb is not None:
            return kb * 1024
    return None


def read_cgroup_memory_peak(pid: int, proc_self_cgroup: str = "/proc/self/cgroup") -> Dict[str, Any]:
    """Read cgroup v2 ``memory.peak`` for the cgroup containing ``pid``.

    Resolves the pid's cgroup path from ``/proc/<pid>/cgroup`` (the v2 unified
    line, ``0::<path>``), reads ``<mount>/<path>/memory.peak``, and reports
    whether that cgroup also contains this process (``cgroup_peak_shared``):
    when the server shares the runner's cgroup the peak is runner-wide, not
    per-server, and Stage 3's per-cell scope is what makes it exact.

    Returns ``{"cgroup_peak": bytes|None, "cgroup_peak_shared": bool|None}``.
    ``cgroup_peak`` is ``None`` on cgroup v1, a kernel without ``memory.peak``,
    or an unreadable /proc.
    """
    result: Dict[str, Any] = {"cgroup_peak": None, "cgroup_peak_shared": None}
    target_path = _resolve_cgroup_v2_path(f"/proc/{pid}/cgroup")
    if target_path is None:
        return result
    mount = _cgroup2_mount()
    if mount is None:
        return result
    peak_file = f"{mount.rstrip('/')}/{target_path.lstrip('/')}/memory.peak"
    peak = _read_int_file(peak_file)
    if peak is None:
        return result
    result["cgroup_peak"] = peak
    # Shared iff our own cgroup resolves to the same path as the server's.
    own_path = _resolve_cgroup_v2_path(proc_self_cgroup)
    result["cgroup_peak_shared"] = own_path is not None and own_path == target_path
    return result


def _resolve_cgroup_v2_path(cgroup_file: str) -> Optional[str]:
    """Return the cgroup v2 unified path (``0::<path>``) from a /proc cgroup file."""
    try:
        with open(cgroup_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                # v2 unified hierarchy: "0::/some/path"
                if line.startswith("0::"):
                    return line[3:]
    except OSError:
        return None
    return None


def _cgroup2_mount() -> Optional[str]:
    """Find the cgroup2 mount point from /proc/mounts (usually /sys/fs/cgroup)."""
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "cgroup2":
                    return parts[1]
    except OSError:
        return None
    return None


def _read_int_file(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _to_int(handle.read().strip())
    except OSError:
        return None


def downsample(samples: List[Dict[str, Any]], max_points: int = 60) -> List[Dict[str, Any]]:
    """Reduce a sample list to at most ``max_points`` evenly spaced entries.

    Keeps the first and last points and picks the rest at even index strides,
    so a 90-point run becomes 60 points spanning the same window. A list already
    at or under the cap is returned unchanged.
    """
    n = len(samples)
    if max_points <= 0:
        return []
    if n <= max_points:
        return list(samples)
    # Evenly spaced indices across [0, n-1], inclusive of both ends.
    step = (n - 1) / (max_points - 1)
    indices = sorted({int(round(i * step)) for i in range(max_points)})
    # Rounding collisions can drop a couple; backfill from the tail to keep the
    # last point and a full count where possible.
    return [samples[i] for i in indices]


class _SocketInfoClient:
    """A blocking-socket ``INFO memory`` client on its own connection.

    Kept tiny on purpose: the sampler must not share the task's CLI/connection,
    and must never let a read error escape into the cell.
    """

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._parser = ReplyParser()

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self._sock.settimeout(self.connect_timeout)
        self._parser = ReplyParser()

    def info_memory(self) -> Optional[str]:
        assert self._sock is not None
        self._sock.sendall(encode_command("INFO", "memory"))
        while True:
            ok, value = self._parser.try_parse()
            if ok:
                return value if isinstance(value, str) else None
            chunk = self._sock.recv(65536)
            if not chunk:
                return None
            self._parser.feed(chunk)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class MemorySampler:
    """Polls ``INFO memory`` at ~1 Hz on its OWN connection, in a thread.

    ``start()`` opens the connection and spawns the thread; ``stop()`` joins it.
    Each tick records ``{"t": wall, used_memory, used_memory_rss,
    mem_clients_normal, mem_replication_backlog}``. A failed tick increments
    ``sample_errors`` and is skipped -- the sampler never raises into the task.
    Call :meth:`samples` for the downsampled (<=60 point) series after stop.

    ``client_factory`` is injected in tests to supply a fake INFO client; in
    production it defaults to a real socket client against ``host:port``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        interval_s: float = 1.0,
        max_points: int = 60,
        client_factory: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.host = host
        self.port = port
        self.interval_s = interval_s
        self.max_points = max_points
        self._clock = clock
        self._client_factory = client_factory or (lambda: _SocketInfoClient(host, port))
        self._rows: List[Dict[str, Any]] = []
        self.sample_errors = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[Any] = None

    def start(self) -> None:
        try:
            self._client = self._client_factory()
            if hasattr(self._client, "connect"):
                self._client.connect()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MemorySampler could not connect to %s:%s: %s", self.host, self.port, exc)
            self.sample_errors += 1
            self._client = None
            return
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._one_tick()
            self._stop.wait(self.interval_s)

    def _one_tick(self) -> None:
        if self._client is None:
            self.sample_errors += 1
            return
        try:
            fields = parse_info_blob(self._client.info_memory())
            if not fields:
                self.sample_errors += 1
                return
            row: Dict[str, Any] = {"t": self._clock()}
            for name in _SAMPLE_INT_FIELDS:
                row[name] = _to_int(fields.get(name))
        except Exception as exc:  # never let a tick escape into the task thread
            self.sample_errors += 1
            logger.debug("MemorySampler tick failed: %s", exc)
            return
        self._rows.append(row)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 2.0)
            self._thread = None
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    def raw_samples(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def samples(self) -> List[Dict[str, Any]]:
        """Downsampled (<=max_points) series, evenly spaced."""
        return downsample(self._rows, self.max_points)


def build_server_memory_record(
    role: str,
    pid: Optional[int],
    info_fields: Dict[str, str],
    reset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one server's memory record from an INFO dict + /proc + cgroup.

    ``info_fields`` is the parsed ``INFO memory`` dict read after the scored
    window. ``pid`` is the server's main pid (the /proc and cgroup reads are
    skipped when it is ``None``, e.g. a remote host). ``reset`` is the record
    :func:`reset_memory_peaks` produced for this server at window start, or
    ``None`` when the peaks were not re-armed (whole-run window).
    """
    info = parse_info_memory(info_fields)
    record: Dict[str, Any] = {
        "pid": pid,
        "used_memory_peak": info["used_memory_peak"],
        "used_memory": info["used_memory"],
        "used_memory_rss": info["used_memory_rss"],
        "rss_hwm": None,
        "vm_rss": None,
        "cgroup_peak": None,
        "cgroup_peak_shared": None,
        "mem_clients_normal": info["mem_clients_normal"],
        "mem_replication_backlog": info["mem_replication_backlog"],
        "allocator_resident": info["allocator_resident"],
        "allocator_active": info["allocator_active"],
        "allocator_frag_ratio": info["allocator_frag_ratio"],
        "reset": reset,
    }
    if pid is not None and pid > 0:
        hwm = read_vm_hwm(pid)
        record["rss_hwm"] = hwm["vm_hwm"]
        record["vm_rss"] = hwm["vm_rss"]
        cgroup = read_cgroup_memory_peak(pid)
        record["cgroup_peak"] = cgroup["cgroup_peak"]
        record["cgroup_peak_shared"] = cgroup["cgroup_peak_shared"]
    return record


def window_engine_peak(
    used_memory_peak: Optional[float],
    reset: Optional[Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> tuple:
    """Best estimate of the engine's ``used_memory`` peak inside the scored window.

    ``used_memory_peak`` cannot be reset over the wire, so: if the end-of-run
    peak exceeds the peak recorded at window start, the window set it and the
    engine's own (cron-sampled) number is returned with source ``engine``. If
    not, the window never rose above the prefill's high-water mark, and the max
    of the 1 Hz sampler's ``used_memory`` at or after the reset time is returned
    with source ``sampler`` (``None`` when there are no such samples). With no
    reset record the whole-run peak is returned as-is (source ``engine``).
    """
    if reset is None:
        return used_memory_peak, "engine" if used_memory_peak is not None else None
    floor = reset.get("used_memory_peak_at_reset")
    if used_memory_peak is not None and (floor is None or used_memory_peak > floor):
        return used_memory_peak, "engine"
    t0 = reset.get("t") or 0.0
    values = [
        row["used_memory"] for row in samples if row.get("used_memory") is not None and (row.get("t") or 0.0) >= t0
    ]
    if values:
        return max(values), "sampler"
    return None, None


def collect_memory_record(
    servers: Dict[str, Any],
    sampler: Optional[MemorySampler],
    info_by_role: Dict[str, Dict[str, str]],
    resets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the ``data.memory`` sub-dict.

    ``servers`` maps ``role`` -> Server object (must expose ``valkey_pid``);
    ``info_by_role`` maps the same role -> the parsed ``INFO memory`` dict the
    task read after the scored window (one INFO round-trip per server, reused
    here so this function issues no I/O to the engine). ``sampler`` supplies the
    1 Hz series and its error count; ``None`` records an empty series.
    ``resets`` maps role -> the :func:`reset_memory_peaks` record for that
    server (absent role or ``None`` -> whole-run window for that server).
    """
    server_records: Dict[str, Any] = {}
    for role, server in servers.items():
        pid = getattr(server, "valkey_pid", None)
        if pid is not None and pid < 0:
            pid = None
        info_fields = info_by_role.get(role, {})
        reset = (resets or {}).get(role)
        record = build_server_memory_record(role, pid, info_fields, reset)
        record["used_memory_window_peak"], record["used_memory_window_peak_source"] = window_engine_peak(
            record["used_memory_peak"], reset, sampler.raw_samples() if sampler is not None else []
        )
        server_records[role] = record
    return {
        "schema": MEMORY_SCHEMA_VERSION,
        "servers": server_records,
        "samples": sampler.samples() if sampler is not None else [],
        "sample_errors": sampler.sample_errors if sampler is not None else 0,
    }


def fold_memory_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold per-rep ``data.memory`` records into one (peak fields -> max).

    Follows the ``per_run_rps`` convention: per-rep records are kept by the
    caller where per-rep fields live; the single folded ``data.memory`` takes
    the max of each peak field per server across reps, and the max of
    ``sample_errors``. The last rep's samples series is kept (representative;
    keeping every rep's series would bloat the row). Non-peak scalars come from
    the last rep.
    """
    if not records:
        return {"schema": MEMORY_SCHEMA_VERSION, "servers": {}, "samples": [], "sample_errors": 0}
    folded = {
        "schema": MEMORY_SCHEMA_VERSION,
        "servers": {},
        "samples": records[-1].get("samples", []),
        "sample_errors": max(r.get("sample_errors", 0) for r in records),
    }
    roles = set()
    for rec in records:
        roles.update(rec.get("servers", {}).keys())
    for role in roles:
        # Start from the last rep's record for this role (non-peak scalars).
        base: Dict[str, Any] = {}
        for rec in records:
            if role in rec.get("servers", {}):
                base = dict(rec["servers"][role])
        for field in PEAK_FIELDS:
            values = [
                rec["servers"][role].get(field)
                for rec in records
                if role in rec.get("servers", {}) and rec["servers"][role].get(field) is not None
            ]
            if values:
                base[field] = max(values)
        folded["servers"][role] = base
    return folded


# --- task-side helpers ----------------------------------------------------


def servers_by_role(primary: Any, replicas: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Map a cell's servers to role labels: ``primary``, ``replica0``, ....

    Accepts the primary Server and its replica Servers (a ``TopologyGroup``
    exposes ``.primary`` and ``.replicas``). A ``None`` primary yields an empty
    map so a caller need not special-case a failed start.
    """
    servers: Dict[str, Any] = {}
    if primary is not None:
        servers["primary"] = primary
    for i, replica in enumerate(replicas or []):
        servers[f"replica{i}"] = replica
    return servers


async def collect_memory_after(
    servers: Dict[str, Any],
    sampler: Optional["MemorySampler"],
    resets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read ``INFO memory`` once per server and assemble ``data.memory``.

    Call this after the scored window and before stopping the servers. Each
    server must expose an async ``info("memory")`` and a ``valkey_pid``. A
    server whose INFO read fails contributes an empty field set (all null)
    rather than failing the cell. ``resets`` is what :func:`reset_memory_peaks`
    returned at window start (or ``None`` if the task did not reset).
    """
    info_by_role: Dict[str, Dict[str, str]] = {}
    for role, server in servers.items():
        try:
            info_by_role[role] = await server.info("memory")
        except Exception as exc:  # a failed INFO must not fail the cell
            logger.warning("INFO memory failed for %s: %s", role, exc)
            info_by_role[role] = {}
    return collect_memory_record(servers, sampler, info_by_role, resets)


# Shell snippet run on the server's host to re-arm the kernel high-water mark
# and report the RSS it was reset to. ``clear_refs`` = 5 resets VmHWM to the
# current VmRSS (Linux >= 4.0); the write needs the same uid as the process,
# which the runner has. Prints ``CLEAR_REFS_OK`` only when the write took.
_CLEAR_REFS_CMD = (
    "sh -c 'echo 5 > /proc/{pid}/clear_refs && echo CLEAR_REFS_OK; grep -E \"^Vm(RSS|HWM):\" /proc/{pid}/status'"
)

# Best-effort cgroup v2 ``memory.peak`` reset (Linux >= 6.12, cgroup must be
# writable by this user). Resolves the pid's unified cgroup path on the host.
_CGROUP_RESET_CMD = (
    'sh -c \'p=$(sed -n "s/^0:://p" /proc/{pid}/cgroup); '
    'f="/sys/fs/cgroup${{p}}/memory.peak"; '
    '[ -n "$p" ] && [ -w "$f" ] && echo reset > "$f" && echo CGROUP_OK\''
)


def _parse_clear_refs_output(stdout: str) -> Dict[str, Any]:
    """Pull ``clear_refs`` success and the post-reset VmRSS/VmHWM from the snippet output."""
    result: Dict[str, Any] = {"clear_refs": False, "vm_rss_at_reset": None, "vm_hwm_at_reset": None}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line == "CLEAR_REFS_OK":
            result["clear_refs"] = True
        elif line.startswith("VmRSS:"):
            result["vm_rss_at_reset"] = _kb_line_to_bytes(line)
        elif line.startswith("VmHWM:"):
            result["vm_hwm_at_reset"] = _kb_line_to_bytes(line)
    return result


async def reset_server_memory_peaks(server: Any, clock: Callable[[], float] = time.time) -> Dict[str, Any]:
    """Re-arm the resettable peaks on one server and record the floors of the rest.

    Writes ``/proc/<pid>/clear_refs`` on the server's host (kernel ``VmHWM``),
    tries the cgroup ``memory.peak`` reset, and reads ``INFO memory`` for the
    engine floors (``used_memory_peak`` cannot be reset over the wire). Each
    step is best-effort and independently recorded; nothing here raises into
    the cell. The returned record is stored under the server's ``reset`` key::

        {"t": wall, "clear_refs": bool, "cgroup_peak": bool,
         "used_memory_at_reset": int|None, "used_memory_peak_at_reset": int|None,
         "vm_rss_at_reset": int|None, "vm_hwm_at_reset": int|None}
    """
    record: Dict[str, Any] = {
        "t": clock(),
        "clear_refs": False,
        "cgroup_peak": False,
        "used_memory_at_reset": None,
        "used_memory_peak_at_reset": None,
        "vm_rss_at_reset": None,
        "vm_hwm_at_reset": None,
    }
    # Engine floors: current usage and the (unresettable) peak so far.
    try:
        fields = await server.info("memory")
        record["used_memory_at_reset"] = _to_int(fields.get("used_memory"))
        record["used_memory_peak_at_reset"] = _to_int(fields.get("used_memory_peak"))
    except Exception as exc:  # never fail the cell
        logger.warning("INFO memory at reset failed on %s: %s", getattr(server, "port", "?"), exc)
    # Kernel and cgroup peaks need the pid on the server's host.
    pid = getattr(server, "valkey_pid", None)
    if pid is None or pid <= 0:
        return record
    try:
        stdout, _ = await server.run_host_command(_CLEAR_REFS_CMD.format(pid=pid), check=False)
        record.update(_parse_clear_refs_output(stdout))
    except Exception as exc:
        logger.warning("clear_refs failed for pid %s: %s", pid, exc)
    try:
        stdout, _ = await server.run_host_command(_CGROUP_RESET_CMD.format(pid=pid), check=False)
        record["cgroup_peak"] = "CGROUP_OK" in (stdout or "")
    except Exception as exc:
        logger.debug("cgroup memory.peak reset failed for pid %s: %s", pid, exc)
    return record


async def reset_memory_peaks(servers: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Re-arm the peaks on every server in ``servers`` (role -> Server).

    Call at the start of the scored window (after the generator's warmup) and
    hand the result to :func:`collect_memory_after` as ``resets``. Never raises.
    """
    resets: Dict[str, Dict[str, Any]] = {}
    for role, server in servers.items():
        try:
            resets[role] = await reset_server_memory_peaks(server)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("memory peak reset failed for %s: %s", role, exc)
    return resets
