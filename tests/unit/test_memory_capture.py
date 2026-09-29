"""Tests for conductress.memory_capture (Stage 1 peak-memory capture)."""

from __future__ import annotations

import os

import pytest

from conductress.memory_capture import (
    MEMORY_SCHEMA_VERSION,
    MemorySampler,
    build_server_memory_record,
    collect_memory_record,
    downsample,
    fold_memory_records,
    parse_info_blob,
    parse_info_memory,
    read_cgroup_memory_peak,
    read_info_memory,
    read_vm_hwm,
)

# --- INFO parsing ---------------------------------------------------------


FULL_INFO = (
    "# Memory\r\n"
    "used_memory:1048576\r\n"
    "used_memory_peak:2097152\r\n"
    "used_memory_rss:3145728\r\n"
    "mem_clients_normal:20480\r\n"
    "mem_replication_backlog:1048576\r\n"
    "allocator_resident:4194304\r\n"
    "allocator_active:4000000\r\n"
    "allocator_frag_ratio:1.05\r\n"
)


def test_parse_info_memory_full():
    rec = parse_info_memory(parse_info_blob(FULL_INFO))
    assert rec["used_memory"] == 1048576
    assert rec["used_memory_peak"] == 2097152
    assert rec["used_memory_rss"] == 3145728
    assert rec["mem_clients_normal"] == 20480
    assert rec["mem_replication_backlog"] == 1048576
    assert rec["allocator_resident"] == 4194304
    assert rec["allocator_active"] == 4000000
    assert rec["allocator_frag_ratio"] == pytest.approx(1.05)


def test_parse_info_memory_missing_keys_are_none():
    # An older Valkey (7.2..) may lack the allocator_* section entirely.
    older = "used_memory:1000\r\nused_memory_peak:2000\r\nused_memory_rss:3000\r\n"
    rec = parse_info_memory(parse_info_blob(older))
    assert rec["used_memory"] == 1000
    assert rec["mem_clients_normal"] is None
    assert rec["mem_replication_backlog"] is None
    assert rec["allocator_resident"] is None
    assert rec["allocator_active"] is None
    assert rec["allocator_frag_ratio"] is None


def test_parse_info_blob_ignores_comment_and_blank_lines():
    fields = parse_info_blob("# Memory\r\n\r\nused_memory:42\r\n")
    assert fields == {"used_memory": "42"}


def test_read_info_memory_via_client():
    class FakeClient:
        def info_memory(self):
            return FULL_INFO

    rec = read_info_memory(FakeClient())
    assert rec["used_memory_peak"] == 2097152


# --- /proc/<pid>/status ---------------------------------------------------


def _write_status(tmp_path, pid, vm_hwm_kb=None, vm_rss_kb=None):
    proc_dir = tmp_path / str(pid)
    proc_dir.mkdir(parents=True)
    lines = ["Name:\tvalkey-server", "VmPeak:\t 999999 kB"]
    if vm_hwm_kb is not None:
        lines.append(f"VmHWM:\t {vm_hwm_kb} kB")
    if vm_rss_kb is not None:
        lines.append(f"VmRSS:\t {vm_rss_kb} kB")
    (proc_dir / "status").write_text("\n".join(lines) + "\n")
    return proc_dir


def test_read_vm_hwm_parses_kb_to_bytes(tmp_path, monkeypatch):
    _write_status(tmp_path, 1234, vm_hwm_kb=5000, vm_rss_kb=4000)

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/1234/status":
            return real_open(str(tmp_path / "1234" / "status"), *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    hwm = read_vm_hwm(1234)
    assert hwm["vm_hwm"] == 5000 * 1024
    assert hwm["vm_rss"] == 4000 * 1024


def test_read_vm_hwm_unreadable_returns_none():
    hwm = read_vm_hwm(999999999)
    assert hwm == {"vm_hwm": None, "vm_rss": None}


# --- cgroup memory.peak ---------------------------------------------------


def test_read_cgroup_memory_peak_shared_tree(tmp_path, monkeypatch):
    # Fake a cgroup2 layout: server pid and self both in /runner.slice.
    cg_root = tmp_path / "cgroup"
    slice_dir = cg_root / "runner.slice"
    slice_dir.mkdir(parents=True)
    (slice_dir / "memory.peak").write_text("734003200\n")  # 700 MiB

    proc = tmp_path / "proc"
    (proc / "4321").mkdir(parents=True)
    (proc / "4321" / "cgroup").write_text("0::/runner.slice\n")
    self_cgroup = proc / "self_cgroup"
    self_cgroup.write_text("0::/runner.slice\n")
    mounts = proc / "mounts"
    mounts.write_text(f"cgroup2 {cg_root} cgroup2 rw 0 0\n")

    real_open = open

    def fake_open(path, *args, **kwargs):
        mapping = {
            "/proc/4321/cgroup": str(proc / "4321" / "cgroup"),
            "/proc/mounts": str(mounts),
        }
        return real_open(mapping.get(path, path), *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    out = read_cgroup_memory_peak(4321, proc_self_cgroup=str(self_cgroup))
    assert out["cgroup_peak"] == 734003200
    assert out["cgroup_peak_shared"] is True


def test_read_cgroup_memory_peak_absent_returns_none(tmp_path, monkeypatch):
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path.startswith("/proc/") or path == "/proc/mounts":
            raise OSError("no such file")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    out = read_cgroup_memory_peak(4321, proc_self_cgroup=str(tmp_path / "nope"))
    assert out == {"cgroup_peak": None, "cgroup_peak_shared": None}


# --- downsample -----------------------------------------------------------


def test_downsample_90_to_60_keeps_ends_and_count():
    samples = [{"t": float(i), "used_memory": i} for i in range(90)]
    out = downsample(samples, max_points=60)
    assert len(out) <= 60
    assert len(out) >= 58  # rounding may collide on a couple of indices
    assert out[0]["t"] == 0.0
    assert out[-1]["t"] == 89.0
    # strictly increasing in time
    ts = [row["t"] for row in out]
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)


def test_downsample_under_cap_is_identity():
    samples = [{"t": float(i)} for i in range(10)]
    assert downsample(samples, max_points=60) == samples


# --- record assembly ------------------------------------------------------


class FakeServer:
    def __init__(self, pid):
        self.valkey_pid = pid


def test_build_server_memory_record_remote_pid_none_skips_proc():
    rec = build_server_memory_record("primary", None, parse_info_blob(FULL_INFO))
    assert rec["pid"] is None
    assert rec["rss_hwm"] is None
    assert rec["cgroup_peak"] is None
    assert rec["used_memory_peak"] == 2097152


def test_collect_memory_record_schema_and_servers():
    servers = {"primary": FakeServer(-1), "replica0": FakeServer(-1)}
    info_by_role = {
        "primary": parse_info_blob(FULL_INFO),
        "replica0": parse_info_blob(FULL_INFO),
    }
    out = collect_memory_record(servers, sampler=None, info_by_role=info_by_role)
    assert out["schema"] == MEMORY_SCHEMA_VERSION
    assert set(out["servers"].keys()) == {"primary", "replica0"}
    # pid -1 normalizes to None (server not running / unknown)
    assert out["servers"]["primary"]["pid"] is None
    assert out["samples"] == []
    assert out["sample_errors"] == 0


def test_fold_memory_records_takes_max_of_peaks():
    rec_a = {
        "schema": 1,
        "servers": {"primary": {"pid": 1, "used_memory_peak": 100, "rss_hwm": 200, "used_memory": 50}},
        "samples": [{"t": 1.0}],
        "sample_errors": 0,
    }
    rec_b = {
        "schema": 1,
        "servers": {"primary": {"pid": 1, "used_memory_peak": 300, "rss_hwm": 150, "used_memory": 60}},
        "samples": [{"t": 2.0}],
        "sample_errors": 2,
    }
    folded = fold_memory_records([rec_a, rec_b])
    assert folded["servers"]["primary"]["used_memory_peak"] == 300  # max
    assert folded["servers"]["primary"]["rss_hwm"] == 200  # max
    assert folded["servers"]["primary"]["used_memory"] == 60  # last rep scalar
    assert folded["sample_errors"] == 2
    assert folded["samples"] == [{"t": 2.0}]  # last rep's series


# --- MemorySampler --------------------------------------------------------


class FakeInfoClient:
    """A fake INFO client the sampler can drive without a socket."""

    def __init__(self, blobs, fail_after=None):
        self._blobs = list(blobs)
        self._i = 0
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def info_memory(self):
        if self._i >= len(self._blobs):
            return self._blobs[-1]
        blob = self._blobs[self._i]
        self._i += 1
        if blob is None:
            raise OSError("simulated read failure")
        return blob

    def close(self):
        self.closed = True


def test_memory_sampler_collects_and_downsamples():
    blobs = [FULL_INFO] * 5
    import itertools

    clock = itertools.count()
    sampler = MemorySampler(
        "127.0.0.1",
        7500,
        interval_s=0.0,
        max_points=60,
        client_factory=lambda: FakeInfoClient(blobs),
        clock=lambda: float(next(clock)),
    )
    sampler.start()
    # Let the thread take a few ticks then stop.
    import time as _t

    _t.sleep(0.05)
    sampler.stop()
    rows = sampler.raw_samples()
    assert len(rows) >= 1
    assert rows[0]["used_memory"] == 1048576
    assert rows[0]["used_memory_rss"] == 3145728
    assert "mem_clients_normal" in rows[0]
    # sample() returns a <=60 downsample
    assert len(sampler.samples()) <= 60


def test_memory_sampler_counts_errors_and_never_raises():
    # A blob of None triggers an OSError inside the tick; must be counted, not raised.
    sampler = MemorySampler(
        "127.0.0.1",
        7500,
        interval_s=0.0,
        client_factory=lambda: FakeInfoClient([None, None, FULL_INFO]),
        clock=lambda: 1.0,
    )
    sampler.start()
    import time as _t

    _t.sleep(0.05)
    sampler.stop()
    assert sampler.sample_errors >= 1
