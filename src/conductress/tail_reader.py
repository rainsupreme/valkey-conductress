"""Bounded reverse-line reader for JSONL files.

Reads the last N *complete* records from a JSONL file by seeking to the end
and reading backwards in fixed-size chunks. Handles no-final-newline and
very-large-line edge cases gracefully.

This replaces whole-file reads (``path.read_text().splitlines()``) that
caused 45-77s latency per ``build_status`` on large output.jsonl files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

# Default chunk size for reverse reading.  64 KiB is large enough to capture
# several typical JSONL records (~500-2000 bytes each) per read syscall.
DEFAULT_CHUNK_SIZE = 64 * 1024

# Byte cap for the reverse task-id search.  A v3 cachecannon row with perf-stat
# enabled carries flamegraph stacks (``data.cpu_stacks_main`` / ``cpu_stacks_io``)
# and measured 1.6-2.7 MB in the wild (Sep 2026 P1-confirm cells: 1.6 MB intel,
# 2.06 MB bench, 2.47 MB g4bench, 2.7 MB armbench).  64 MiB leaves ~24x headroom
# over the largest observed row while still bounding the scan of a multi-hundred-MB
# output.jsonl.  Rows older than this window are treated as not found (return None).
FIND_RECORD_MAX_BYTES = 64 * 1024 * 1024


def find_record_by_task_id(
    path: Union[str, Path],
    task_id: str,
    *,
    max_bytes: int = FIND_RECORD_MAX_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Optional[dict[str, Any]]:
    """Return the newest complete JSONL record whose ``task_id`` matches.

    Seeks to end-of-file and reads backwards in ``chunk_size`` chunks, decoding
    each complete line and returning the first (newest) record whose top-level
    ``task_id`` equals *task_id*.  Reads at most *max_bytes* from the tail; a
    matching row older than that window is not found and ``None`` is returned
    (matching the pre-fix fixed-window behavior for genuinely stale rows).

    Unlike a fixed tail window, this keeps scanning until the row is found or
    the byte cap is hit, so a single very large row (megabytes of flamegraph
    stacks) that no fixed window could ever contain in full is still located.

    Behavior:
    - A row straddling two chunk boundaries is reassembled, not dropped.
    - A file that ends without a trailing newline still has its final fragment
      considered.
    - Undecodable / non-JSON lines are skipped.
    - Returns ``None`` when the file is missing, empty, or holds no match within
      the byte cap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_bytes <= 0:
        return None

    path = Path(path)
    try:
        file_size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return None
    if file_size == 0:
        return None

    remainder = b""
    first_chunk = True
    bytes_read = 0

    with path.open("rb") as handle:
        pos = file_size
        while pos > 0 and bytes_read < max_bytes:
            read_size = min(chunk_size, pos, max_bytes - bytes_read)
            pos -= read_size
            bytes_read += read_size
            handle.seek(pos)
            chunk = handle.read(read_size)

            # Right-to-left assembly: the fragment carried from the previous
            # (more-recent) chunk belongs to the END of this chunk's data.
            data = chunk + remainder
            parts = data.split(b"\n")

            # On the very first (rightmost) chunk, a trailing newline yields an
            # empty final part that is not a real line; drop it.
            if first_chunk and parts and parts[-1] == b"":
                parts = parts[:-1]
            first_chunk = False

            # The leftmost fragment may be an incomplete line split at the chunk
            # boundary; carry it forward for the next (earlier) chunk.  The one
            # exception is when we have reached the start of the file (pos == 0)
            # or exhausted the byte cap, where it is a complete line.
            at_start = pos == 0 or bytes_read >= max_bytes
            complete = parts if at_start else parts[1:]
            remainder = b"" if at_start else parts[0]

            for part in reversed(complete):
                if not part:
                    continue
                try:
                    record = json.loads(part)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(record, dict) and record.get("task_id") == task_id:
                    return record

    return None


def tail_lines(
    path: Union[str, Path],
    n: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Return the last *n* complete lines from *path*, newest last.

    Behavior:
    - Returns up to *n* lines. If the file has fewer, returns all of them.
    - A trailing newline at end-of-file does NOT produce an extra empty line
      (matches ``str.splitlines()`` semantics, not ``str.split('\\n')``).
    - A file that ends without ``\\n`` still has its final fragment counted
      as a complete line.
    - Empty lines (blank ``\\n`` runs in the middle) are included in the count.
    - Binary-safe: reads in binary mode and decodes as UTF-8 with
      ``errors='replace'`` to tolerate partially-written lines.
    - Returns ``[]`` when the file is missing or empty.
    - Very large lines (bigger than *chunk_size*) are assembled correctly
      across chunk boundaries.

    Complexity:
        Reads at most ``ceil(n * avg_line_len / chunk_size) + 1`` chunks
        from the end of the file. For typical JSONL workloads (n=200,
        avg_line ~1 KiB, chunk 64 KiB) this is 4-5 read syscalls vs
        reading the entire (often 50-200 MiB) file.
    """
    if n <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    path = Path(path)
    try:
        file_size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return []
    if file_size == 0:
        return []

    collected: list[str] = []
    remainder = b""
    first_chunk = True

    with path.open("rb") as f:
        pos = file_size
        while pos > 0 and len(collected) < n + 1:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)

            # Prepend to remainder from last iteration (right-to-left assembly)
            data = chunk + remainder
            parts = data.split(b"\n")

            # On the very first chunk (rightmost read), if the file ends with
            # a newline the split produces a trailing empty byte-string that
            # does not correspond to a real line.  Drop it so we match
            # str.splitlines() semantics.
            if first_chunk and parts and parts[-1] == b"":
                parts = parts[:-1]
                first_chunk = False
            else:
                first_chunk = False

            # The leftmost fragment may be incomplete (split mid-line at chunk
            # boundary). Carry it forward as the remainder for the next chunk.
            remainder = parts[0]
            # All subsequent parts are complete lines.
            for part in reversed(parts[1:]):
                collected.append(part.decode("utf-8", errors="replace"))
                if len(collected) >= n + 1:
                    break

    # If there is a remaining fragment (beginning of file), include it.
    if remainder and len(collected) < n + 1:
        collected.append(remainder.decode("utf-8", errors="replace"))

    # collected is newest-first; reverse so newest is last (matches file order).
    collected.reverse()
    # We collected up to n+1 to handle the partial-first-line edge; trim.
    return collected[-n:] if len(collected) > n else collected
