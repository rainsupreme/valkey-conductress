"""Bounded reverse-line reader for JSONL files.

Reads the last N *complete* records from a JSONL file by seeking to the end
and reading backwards in fixed-size chunks. Handles no-final-newline and
very-large-line edge cases gracefully.

This replaces whole-file reads (``path.read_text().splitlines()``) that
caused 45-77s latency per ``build_status`` on large output.jsonl files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# Default chunk size for reverse reading.  64 KiB is large enough to capture
# several typical JSONL records (~500-2000 bytes each) per read syscall.
DEFAULT_CHUNK_SIZE = 64 * 1024


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
