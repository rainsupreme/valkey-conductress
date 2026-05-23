"""Tests for perf stat parsing logic."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from conductress.server import Server

SAMPLE_PERF_STAT_OUTPUT = """\
# started on Sat May 17 08:00:00 2026

 Performance counter stats for process id '12345':

       508,723,261      L1-icache-load-misses:u          #   13.56% of all L1-icache accesses   (87.51%)
     3,751,580,667      L1-icache-loads:u                                                       (87.48%)
        90,629,557      L1-dcache-load-misses:u          #    1.42% of all L1-dcache accesses   (87.50%)
     6,381,942,664      L1-dcache-loads:u                                                       (87.51%)
    20,391,313,680      instructions:u                   #    2.40  insn per cycle              (75.03%)
     8,483,437,256      cycles:u                                                                (87.51%)
         2,371,185      branch-misses:u                  #    0.05% of all branches             (87.51%)
     4,571,483,740      branches:u                                                              (87.47%)
       123,456,789      stalled-cycles-frontend:u                                               (87.50%)
        98,765,432      stalled-cycles-backend:u                                                (87.50%)

      15.001109872 seconds time elapsed

"""


def test_parse_perf_stat_basic():
    """Test parsing a typical perf stat output."""
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_PERF_STAT_OUTPUT)
        f.flush()
        result = Server.parse_perf_stat(Path(f.name))

    assert result["L1-icache-load-misses"] == 508723261
    assert result["L1-icache-loads"] == 3751580667
    assert result["L1-dcache-load-misses"] == 90629557
    assert result["L1-dcache-loads"] == 6381942664
    assert result["instructions"] == 20391313680
    assert result["cycles"] == 8483437256
    assert result["branch-misses"] == 2371185
    assert result["branches"] == 4571483740
    assert result["stalled-cycles-frontend"] == 123456789
    assert result["stalled-cycles-backend"] == 98765432


def test_parse_perf_stat_empty():
    """Test parsing an empty/missing file."""
    result = Server.parse_perf_stat(Path("/nonexistent/path"))
    assert result == {}


def test_parse_perf_stat_ipc():
    """Test that we can derive IPC from parsed results."""
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_PERF_STAT_OUTPUT)
        f.flush()
        result = Server.parse_perf_stat(Path(f.name))

    ipc = result["instructions"] / result["cycles"]
    assert 2.3 < ipc < 2.5  # ~2.40 from the sample


def test_parse_perf_stat_icache_miss_rate():
    """Test that we can derive I-cache miss rate from parsed results."""
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_PERF_STAT_OUTPUT)
        f.flush()
        result = Server.parse_perf_stat(Path(f.name))

    miss_rate = result["L1-icache-load-misses"] / result["L1-icache-loads"]
    assert 0.13 < miss_rate < 0.14  # ~13.56%


def test_parse_perf_stat_suffix_stripping():
    """Verify :u/:k suffixes are stripped as substrings, not character sets.

    Regression test: rstrip(":k") strips all trailing chars in {':','k'},
    mangling 'cpu-clock:k' → 'cpu-cloc'. removesuffix() correctly yields 'cpu-clock'.
    """
    content = """\
     123,456,789      stalled-cycles-frontend:u
   1,000,000,000      cpu-clock:k
     500,000,000      task-clock:u
"""
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        result = Server.parse_perf_stat(Path(f.name))

    assert "stalled-cycles-frontend" in result
    assert "cpu-clock" in result, f"Got keys: {list(result.keys())}"
    assert "task-clock" in result, f"Got keys: {list(result.keys())}"
    assert result["cpu-clock"] == 1_000_000_000
