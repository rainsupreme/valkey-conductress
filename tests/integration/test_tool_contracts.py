"""Tier 3: Tool-contract acceptance tests.

Validates that the external tool outputs (memtier_benchmark, valkey-benchmark)
match the structural contracts that Conductress's parsers depend on.

Requires: real memtier_benchmark + valkey-benchmark binaries + a running server.
Marked requires_server because it needs the full benchmark host environment.
"""

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_server


def _find_binary(name: str) -> str:
    """Locate a benchmark binary (~/conductress/<name>, ~/valkey/src/<name>, or PATH)."""
    conductress_bin = Path.home() / "conductress" / name
    if conductress_bin.exists():
        return str(conductress_bin)
    # Local dev build
    valkey_src_bin = Path.home() / "valkey" / "src" / name
    if valkey_src_bin.exists():
        return str(valkey_src_bin)
    # Fallback to PATH
    result = subprocess.run(["which", name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    pytest.skip(f"{name} not found in ~/conductress/, ~/valkey/src/, or PATH")
    return ""  # unreachable


def _find_server_binary() -> str:
    """Locate valkey-server binary."""
    candidates = [
        Path.home() / "conductress" / "valkey-server",
        Path.home() / "valkey" / "src" / "valkey-server",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    pytest.skip("valkey-server not found")
    return ""


@pytest.fixture(scope="module")
def local_server():
    """Start a local valkey-server on port 7499 for tool contract tests."""
    server_bin = _find_server_binary()
    proc = subprocess.Popen(
        [server_bin, "--port", "7499", "--save", "", "--daemonize", "no", "--loglevel", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)  # Let server start
    if proc.poll() is not None:
        pytest.skip("valkey-server failed to start")
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


class TestMemtierJsonContract:
    """Contract: memtier --json-out-file output structure.

    Parser: task_scenario.py:parse_memtier_json_intervals
    Depends on: top-level 'ALL STATS' key with 'Second N' sub-keys each having 'Ops/sec'.
    """

    def test_json_out_file_has_all_stats(self, local_server):
        """memtier --json-out-file must produce top-level 'ALL STATS' key.

        Consumer: parse_memtier_json_intervals() -> data.get("ALL STATS", {})
        """
        memtier = _find_binary("memtier_benchmark")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            result = subprocess.run(
                [
                    memtier,
                    "--server",
                    "127.0.0.1",
                    "--port",
                    "7499",
                    "--protocol",
                    "redis",
                    "--threads",
                    "1",
                    "--clients",
                    "5",
                    "--ratio",
                    "1:1",
                    "--test-time",
                    "3",
                    "--json-out-file",
                    json_path,
                    "--hide-histogram",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"memtier failed: {result.stderr}"

            data = json.loads(Path(json_path).read_text())
            assert "ALL STATS" in data, "Missing 'ALL STATS' -- parse_memtier_json_intervals will fail"
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_json_intervals_have_ops_sec(self, local_server):
        """JSON Time-Serie entries must have 'Count' field for per-second ops.

        Consumer: parse_memtier_json_intervals() -> Time-Serie -> entry["Count"]
        Real memtier structure: ALL STATS -> Totals -> Time-Serie -> {"0": {"Count": N}, ...}
        """
        memtier = _find_binary("memtier_benchmark")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            result = subprocess.run(
                [
                    memtier,
                    "--server",
                    "127.0.0.1",
                    "--port",
                    "7499",
                    "--protocol",
                    "redis",
                    "--threads",
                    "1",
                    "--clients",
                    "5",
                    "--ratio",
                    "0:1",
                    "--key-pattern",
                    "R:R",
                    "--key-minimum",
                    "1",
                    "--key-maximum",
                    "1000",
                    "--data-size",
                    "64",
                    "--test-time",
                    "3",
                    "--json-out-file",
                    json_path,
                    "--hide-histogram",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"memtier failed: {result.stderr}"

            data = json.loads(Path(json_path).read_text())
            all_stats = data["ALL STATS"]

            # Real contract: Totals has Time-Serie with numeric keys
            assert "Totals" in all_stats, "Missing 'Totals' in ALL STATS"
            totals = all_stats["Totals"]
            assert "Time-Serie" in totals, (
                "Missing 'Time-Serie' in Totals -- parse_memtier_json_intervals will return []. "
                f"Totals keys: {list(totals.keys())}"
            )
            time_serie = totals["Time-Serie"]
            assert len(time_serie) > 0, "Time-Serie is empty"

            # Each entry must have Count
            for key, entry in time_serie.items():
                assert key.isdigit(), f"Time-Serie key '{key}' is not a numeric second index"
                assert "Count" in entry, (
                    f"Missing 'Count' in Time-Serie[{key}] -- parse_memtier_json_intervals will skip it. "
                    f"Entry keys: {list(entry.keys())}"
                )
                assert isinstance(entry["Count"], (int, float))
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_totals_entry_in_json(self, local_server):
        """'Totals' entry must exist with 'Ops/sec'.

        Consumer: parse_memtier_total_rps() from task_mixed.py (reads stdout Totals line,
        but JSON also has it for cross-validation).
        """
        memtier = _find_binary("memtier_benchmark")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            result = subprocess.run(
                [
                    memtier,
                    "--server",
                    "127.0.0.1",
                    "--port",
                    "7499",
                    "--protocol",
                    "redis",
                    "--threads",
                    "1",
                    "--clients",
                    "5",
                    "--ratio",
                    "1:1",
                    "--test-time",
                    "2",
                    "--json-out-file",
                    json_path,
                    "--hide-histogram",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"memtier failed: {result.stderr}"

            data = json.loads(Path(json_path).read_text())
            all_stats = data["ALL STATS"]
            assert "Totals" in all_stats, "Missing 'Totals' in ALL STATS"
            assert "Ops/sec" in all_stats["Totals"]
            assert all_stats["Totals"]["Ops/sec"] > 0
        finally:
            Path(json_path).unlink(missing_ok=True)


class TestValkeyBenchmarkContract:
    """Contract: valkey-benchmark -q -l streaming output format.

    Parser: task_perf_benchmark.py:__collect_metrics
    Depends on: lines containing 'overall' with format:
        'CMD: rps=NNNN.N (overall: NNNN.N) avg_msec=N.NNN (overall: N.NNN)'
    Parsing logic: line.split("rps=")[1].split()[0] -> float
    """

    def test_q_output_contains_overall_lines(self, local_server):
        """valkey-benchmark -q must produce lines with 'overall' and 'rps=' fields.

        Consumer: __collect_metrics() filters on 'overall' in line, then splits on 'rps='.
        """
        bench = _find_binary("valkey-benchmark")
        result = subprocess.run(
            [
                bench,
                "-h",
                "127.0.0.1",
                "-p",
                "7499",
                "-c",
                "10",
                "-n",
                "1000",
                "-q",
                "-t",
                "set,get",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"valkey-benchmark failed: {result.stderr}"

        # -q (non-loop) produces summary lines like:
        # "SET: 142857.14 requests per second, p50=0.103 msec"
        # But -q -l (loop mode, used by Conductress) produces:
        # "GET: rps=140328.0 (overall: 141165.2) avg_msec=0.193 (overall: 0.191)"
        # Since we can't run -l indefinitely in a test, verify the -q non-loop format
        # which __collect_metrics also encounters at final summary.
        stdout = result.stdout
        assert len(stdout.strip()) > 0, "No output from valkey-benchmark -q"

        # Verify at least one line has "requests per second" (non-loop summary)
        rps_lines = [l for l in stdout.splitlines() if "requests per second" in l]
        assert len(rps_lines) >= 1, f"No 'requests per second' summary lines in -q output:\n{stdout}"

    def test_q_loop_output_rps_format(self, local_server):
        """valkey-benchmark -q -l produces streaming lines parseable by rps= split.

        Consumer: __collect_metrics() does line.split("rps=")[1].split()[0]
        Run with -l but kill after a few seconds to get streaming output.
        """
        bench = _find_binary("valkey-benchmark")
        # Prefill keys so GET has data to return (avoids 0 rps warmup artifact)
        subprocess.run(
            [bench, "-h", "127.0.0.1", "-p", "7499", "-n", "1000", "-t", "set", "-d", "64"],
            capture_output=True,
            timeout=10,
        )
        proc = subprocess.Popen(
            [
                bench,
                "-h",
                "127.0.0.1",
                "-p",
                "7499",
                "-c",
                "10",
                "-n",
                "100000",
                "-q",
                "-l",
                "-t",
                "get",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(3)
        proc.terminate()
        stdout, _ = proc.communicate(timeout=5)

        # Look for streaming format lines with 'overall' and 'rps='
        overall_lines = [l for l in stdout.splitlines() if "overall" in l]
        assert len(overall_lines) >= 1, f"No 'overall' streaming lines from -q -l output:\n{stdout[:500]}"

        # Validate the parsing contract: split("rps=")[1].split()[0] must be a float
        # Filter out warmup lines where rps=0.0 (startup artifact before first interval completes)
        valid_lines = [l for l in overall_lines if "rps=0.0 " not in l]
        assert len(valid_lines) >= 1, f"All streaming lines had rps=0.0 (startup artifact):\n{stdout[:500]}"
        for line in valid_lines:
            assert "rps=" in line, f"Line has 'overall' but no 'rps=': {line}"
            rps_str = line.split("rps=")[1].split()[0]
            rps_val = float(rps_str)  # Must not raise
            assert rps_val > 0, f"Parsed RPS is not positive: {rps_val} from line: {line}"

    def test_overlay_rps_format(self, local_server):
        """valkey-benchmark non-streaming output has 'requests per second' line.

        Consumer: task_scenario.py:_parse_overlay_rps
        Pattern: re.search(r"([\\d.]+)\\s+requests per second", line)
        """
        bench = _find_binary("valkey-benchmark")
        result = subprocess.run(
            [
                bench,
                "-h",
                "127.0.0.1",
                "-p",
                "7499",
                "-c",
                "8",
                "-n",
                "500",
                "-t",
                "set",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"valkey-benchmark failed: {result.stderr}"

        # _parse_overlay_rps expects: "NNNN.NN requests per second"
        pattern = re.compile(r"([\d.]+)\s+requests per second")
        matches = [pattern.search(l) for l in result.stdout.splitlines()]
        matches = [m for m in matches if m is not None]
        assert len(matches) >= 1, (
            f"No 'NNN requests per second' line found -- _parse_overlay_rps will return None.\n"
            f"Output:\n{result.stdout[:500]}"
        )
        for m in matches:
            rps = float(m.group(1))
            assert rps > 0


class TestMemtierStdoutContract:
    """Contract: memtier_benchmark stdout Totals line.

    Parser: task_mixed.py:parse_memtier_total_rps
    Depends on: a line where split()[0] == 'Totals' and split()[1] is a float (ops/sec).
    """

    def test_stdout_totals_line(self, local_server):
        """memtier stdout must have a 'Totals' line with numeric ops/sec.

        Consumer: parse_memtier_total_rps() -> parts[0] == "Totals", float(parts[1])
        """
        memtier = _find_binary("memtier_benchmark")
        result = subprocess.run(
            [
                memtier,
                "--server",
                "127.0.0.1",
                "--port",
                "7499",
                "--protocol",
                "redis",
                "--threads",
                "1",
                "--clients",
                "5",
                "--ratio",
                "1:1",
                "--test-time",
                "2",
                "--hide-histogram",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"memtier failed: {result.stderr}"

        # Find Totals line
        totals_line = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Totals":
                totals_line = line
                break

        assert totals_line is not None, (
            f"No 'Totals' line in memtier stdout -- parse_memtier_total_rps will return None.\n"
            f"Output:\n{result.stdout[:500]}"
        )
        parts = totals_line.split()
        ops_sec = float(parts[1])  # Must not raise
        assert ops_sec > 0, f"Totals ops/sec is not positive: {ops_sec}"

    def test_stdout_progress_lines(self, local_server):
        """memtier stdout must emit periodic progress lines with ops/sec.

        Consumer: parse_memtier_stdout_intervals() (fallback for timeseries)
        Pattern: [RUN #N ...] ... NNN (avg: NNN) ops/sec ...
        """
        memtier = _find_binary("memtier_benchmark")
        result = subprocess.run(
            [
                memtier,
                "--server",
                "127.0.0.1",
                "--port",
                "7499",
                "--protocol",
                "redis",
                "--threads",
                "1",
                "--clients",
                "5",
                "--ratio",
                "0:1",
                "--key-pattern",
                "R:R",
                "--key-minimum",
                "1",
                "--key-maximum",
                "1000",
                "--data-size",
                "64",
                "--test-time",
                "3",
                "--hide-histogram",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"memtier failed: {result.stderr}"

        from conductress.tasks.task_scenario import parse_memtier_stdout_intervals

        intervals = parse_memtier_stdout_intervals(result.stdout)
        assert len(intervals) >= 1, (
            f"No progress lines parsed from stdout -- parse_memtier_stdout_intervals returned [].\n"
            f"Output (first 500 chars):\n{result.stdout[:500]}"
        )
        for val in intervals:
            assert val > 0, f"Progress line ops/sec not positive: {val}"
