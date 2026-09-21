"""Tests for the bounded reverse-line reader (tail_reader module)."""

import json
import os
from pathlib import Path

import pytest

from conductress.tail_reader import find_record_by_task_id, tail_lines


class TestTailLinesBasic:
    """Core correctness: result count, ordering, edge cases."""

    def test_missing_file_returns_empty(self, tmp_path):
        assert tail_lines(tmp_path / "nope.jsonl", 10) == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_bytes(b"")
        assert tail_lines(f, 10) == []

    def test_zero_or_negative_line_count_returns_empty(self, tmp_path):
        f = tmp_path / "lines.jsonl"
        f.write_text("a\nb\n")
        assert tail_lines(f, 0) == []
        assert tail_lines(f, -1) == []

    def test_nonpositive_chunk_size_is_rejected(self, tmp_path):
        f = tmp_path / "lines.jsonl"
        f.write_text("a\n")
        with pytest.raises(ValueError, match="chunk_size"):
            tail_lines(f, 1, chunk_size=0)

    def test_single_line_no_newline(self, tmp_path):
        f = tmp_path / "one.jsonl"
        f.write_bytes(b'{"x": 1}')
        assert tail_lines(f, 5) == ['{"x": 1}']

    def test_single_line_with_newline(self, tmp_path):
        f = tmp_path / "one.jsonl"
        f.write_bytes(b'{"x": 1}\n')
        result = tail_lines(f, 5)
        # Trailing newline does not produce extra empty line (splitlines semantics)
        assert result == ['{"x": 1}']

    def test_fewer_lines_than_requested(self, tmp_path):
        f = tmp_path / "few.jsonl"
        f.write_text("a\nb\n")
        result = tail_lines(f, 100)
        assert "a" in result
        assert "b" in result

    def test_exact_n_lines(self, tmp_path):
        f = tmp_path / "exact.jsonl"
        lines = [f'{{"i": {i}}}' for i in range(5)]
        f.write_text("\n".join(lines) + "\n")
        result = tail_lines(f, 5)
        # Should contain all 5 lines (trailing empty excluded from content)
        json_lines = [r for r in result if r.strip()]
        assert len(json_lines) == 5
        # Newest-last (file order preserved)
        assert json.loads(json_lines[-1])["i"] == 4

    def test_more_lines_than_n_returns_last_n(self, tmp_path):
        f = tmp_path / "many.jsonl"
        lines = [f'{{"i": {i}}}' for i in range(20)]
        f.write_text("\n".join(lines) + "\n")
        result = tail_lines(f, 5)
        # Should be exactly the last 5 records
        ids = [json.loads(l)["i"] for l in result]
        assert ids == [15, 16, 17, 18, 19]

    def test_preserves_file_order(self, tmp_path):
        f = tmp_path / "order.jsonl"
        lines = [f'{{"i": {i}}}' for i in range(10)]
        f.write_text("\n".join(lines) + "\n")
        result = tail_lines(f, 10)
        json_lines = [r for r in result if r.strip()]
        ids = [json.loads(l)["i"] for l in json_lines]
        assert ids == sorted(ids)  # ascending = file order


class TestTailLinesChunkBoundary:
    """Verify correctness when lines span chunk boundaries."""

    def test_very_small_chunk_size(self, tmp_path):
        """Lines longer than chunk_size are assembled correctly."""
        f = tmp_path / "big.jsonl"
        # Each line is ~50 bytes; chunk_size=16 forces many chunks
        lines = [f'{{"index": {i}, "pad": "{"x" * 30}"}}' for i in range(5)]
        f.write_text("\n".join(lines) + "\n")
        result = tail_lines(f, 3, chunk_size=16)
        json_lines = [r for r in result if r.strip()]
        assert len(json_lines) == 3
        # They should be the last 3
        ids = [json.loads(l)["index"] for l in json_lines]
        assert ids == [2, 3, 4]

    def test_line_exactly_at_chunk_boundary(self, tmp_path):
        """A newline at exactly the chunk boundary position."""
        f = tmp_path / "boundary.jsonl"
        # Make line lengths sum to exactly chunk_size
        line = "a" * 63  # 63 bytes + \n = 64 = chunk_size
        f.write_text(line + "\n" + "b" * 10 + "\n")
        result = tail_lines(f, 5, chunk_size=64)
        text_lines = [r for r in result if r.strip()]
        assert len(text_lines) == 2

    def test_large_single_line_no_newline(self, tmp_path):
        """One very large line with no trailing newline."""
        f = tmp_path / "huge.jsonl"
        huge = "x" * 200_000
        f.write_text(huge)
        result = tail_lines(f, 1, chunk_size=1024)
        assert len(result) == 1
        assert result[0] == huge


class TestTailLinesNoFinalNewline:
    """Files that don't end with \\n."""

    def test_multi_line_no_final_newline(self, tmp_path):
        f = tmp_path / "nolf.jsonl"
        f.write_bytes(b"line1\nline2\nline3")
        result = tail_lines(f, 3)
        text = [r for r in result if r.strip()]
        assert text == ["line1", "line2", "line3"]

    def test_request_fewer_than_available_no_final_newline(self, tmp_path):
        f = tmp_path / "nolf2.jsonl"
        f.write_bytes(b"a\nb\nc\nd")
        result = tail_lines(f, 2)
        text = [r for r in result if r.strip()]
        assert text == ["c", "d"]


class TestTailLinesBlankLines:
    """Empty lines in the middle/end are counted."""

    def test_blank_lines_counted(self, tmp_path):
        f = tmp_path / "blanks.jsonl"
        f.write_text("a\n\nb\n\nc\n")
        result = tail_lines(f, 3)
        # Should get the last 3 items including blanks
        assert len(result) == 3


class TestTailLinesPerformanceContract:
    """The reader must NOT read the whole file — verify via seek position."""

    def test_bounded_read_on_large_file(self, tmp_path):
        """With n=5 and ~100-byte lines, reading at most ~5KB, not the whole file."""
        f = tmp_path / "large.jsonl"
        # Write 10,000 lines (~1 MB)
        with f.open("w") as fh:
            for i in range(10_000):
                fh.write(json.dumps({"task_id": f"t{i}", "score": i}) + "\n")

        file_size = f.stat().st_size
        assert file_size > 100_000  # sanity: file is large enough to matter

        result = tail_lines(f, 5)
        json_lines = [r for r in result if r.strip()]
        assert len(json_lines) == 5
        # Verify last entry
        last = json.loads(json_lines[-1])
        assert last["task_id"] == "t9999"


class TestFindRecordByTaskId:
    """Byte-capped reverse search for a completed task's row."""

    def test_missing_file_returns_none(self, tmp_path):
        assert find_record_by_task_id(tmp_path / "nope.jsonl", "t1") is None

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_bytes(b"")
        assert find_record_by_task_id(f, "t1") is None

    def test_finds_only_row(self, tmp_path):
        f = tmp_path / "one.jsonl"
        f.write_text(json.dumps({"task_id": "t1", "score": 42}) + "\n")
        rec = find_record_by_task_id(f, "t1")
        assert rec is not None and rec["score"] == 42

    def test_finds_row_with_no_trailing_newline(self, tmp_path):
        f = tmp_path / "nolf.jsonl"
        f.write_bytes(json.dumps({"task_id": "t1", "score": 7}).encode())
        rec = find_record_by_task_id(f, "t1")
        assert rec is not None and rec["score"] == 7

    def test_finds_when_several_later_rows_follow(self, tmp_path):
        f = tmp_path / "many.jsonl"
        with f.open("w") as fh:
            fh.write(json.dumps({"task_id": "target", "score": 99}) + "\n")
            for i in range(50):
                fh.write(json.dumps({"task_id": f"later{i}", "score": i}) + "\n")
        rec = find_record_by_task_id(f, "target")
        assert rec is not None and rec["score"] == 99

    def test_returns_newest_match_when_task_id_repeats(self, tmp_path):
        f = tmp_path / "dup.jsonl"
        f.write_text(
            json.dumps({"task_id": "t1", "score": 1}) + "\n" + json.dumps({"task_id": "t1", "score": 2}) + "\n"
        )
        rec = find_record_by_task_id(f, "t1")
        assert rec is not None and rec["score"] == 2

    def test_finds_row_larger_than_one_megabyte(self, tmp_path):
        """A ~2.5 MB perf-stat row is found even though a 1 MB window never held it."""
        f = tmp_path / "huge.jsonl"
        big = {"task_id": "t1", "score": 5, "data": {"cpu_stacks_main": "x" * (2_500_000)}}
        f.write_bytes((json.dumps(big) + "\n").encode())
        rec = find_record_by_task_id(f, "t1")
        assert rec is not None and rec["score"] == 5
        assert len(rec["data"]["cpu_stacks_main"]) == 2_500_000

    def test_returns_none_when_match_lies_beyond_byte_cap(self, tmp_path):
        f = tmp_path / "capped.jsonl"
        with f.open("w") as fh:
            fh.write(json.dumps({"task_id": "old", "score": 1}) + "\n")
            # Push the "old" row well outside a tiny byte cap with filler rows.
            for i in range(200):
                fh.write(json.dumps({"task_id": f"new{i}", "pad": "y" * 100}) + "\n")
        assert find_record_by_task_id(f, "old", max_bytes=2048) is None

    def test_row_straddling_chunk_boundary_is_reassembled(self, tmp_path):
        f = tmp_path / "straddle.jsonl"
        # Row well larger than the tiny chunk size, plus later rows, forces the
        # target to span multiple reverse-read chunks.
        target = {"task_id": "t1", "pad": "z" * 500}
        with f.open("w") as fh:
            fh.write(json.dumps(target) + "\n")
            for i in range(10):
                fh.write(json.dumps({"task_id": f"n{i}", "pad": "w" * 50}) + "\n")
        rec = find_record_by_task_id(f, "t1", chunk_size=16)
        assert rec is not None and rec["pad"] == "z" * 500

    def test_zero_chunk_size_rejected(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text(json.dumps({"task_id": "t1"}) + "\n")
        with pytest.raises(ValueError, match="chunk_size"):
            find_record_by_task_id(f, "t1", chunk_size=0)

    def test_skips_non_json_lines(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text("not json\n" + json.dumps({"task_id": "t1", "score": 3}) + "\ngarbage\n")
        rec = find_record_by_task_id(f, "t1")
        assert rec is not None and rec["score"] == 3
