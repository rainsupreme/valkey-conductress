"""Minimal RESP encoder and an incremental reply parser.

Only the surface a storm client needs: encode a command as a RESP array of
bulk strings, and parse one reply that may arrive split across several socket
reads. Supported reply types cover a HELLO 3 handshake reply and ordinary
command replies: simple string (``+``), error (``-``), integer (``:``), bulk
string (``$``, including the null bulk ``$-1``), null array (``*-1``), array
(``*``) and RESP3 map (``%``), null (``_``), boolean (``#``), double (``,``)
and big number (``(``). Aggregate replies (arrays and maps) recurse.

The parser is fed bytes incrementally with :meth:`ReplyParser.feed`; each call
to :meth:`ReplyParser.try_parse` returns one complete reply or ``None`` when
more bytes are still needed. It never blocks and never reads a socket itself,
which is what makes it straightforward to fixture-test with replies chopped at
arbitrary byte boundaries.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

CRLF = b"\r\n"


def encode_command(*args: str) -> bytes:
    """Encode a command as a RESP array of bulk strings.

    ``encode_command("GET", "key")`` -> ``b"*2\\r\\n$3\\r\\nGET\\r\\n$3\\r\\nkey\\r\\n"``.
    """
    if not args:
        raise ValueError("a RESP command needs at least one argument")
    parts: List[bytes] = [b"*", str(len(args)).encode(), CRLF]
    for arg in args:
        encoded = arg.encode()
        parts += [b"$", str(len(encoded)).encode(), CRLF, encoded, CRLF]
    return b"".join(parts)


class ProtocolError(Exception):
    """The byte stream is not valid RESP."""


class ReplyParser:
    """Incremental RESP reply parser.

    Feed it bytes as they arrive and call :meth:`try_parse` to pull complete
    replies. Leftover bytes are retained for the next reply.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        """Append received bytes to the internal buffer."""
        self._buf.extend(data)

    def try_parse(self) -> Tuple[bool, Any]:
        """Attempt to parse one reply from the buffer.

        Returns ``(True, value)`` when a complete reply was consumed, or
        ``(False, None)`` when more bytes are needed. An error reply (``-``)
        is returned as its value; the caller decides how to treat it.
        """
        ok, value, consumed = self._parse_at(0)
        if not ok:
            return False, None
        del self._buf[:consumed]
        return True, value

    def _read_line(self, start: int) -> Optional[Tuple[bytes, int]]:
        """Return (line-without-CRLF, index-after-CRLF) at ``start`` or None if incomplete."""
        idx = self._buf.find(CRLF, start)
        if idx == -1:
            return None
        return bytes(self._buf[start:idx]), idx + 2

    def _parse_at(self, start: int) -> Tuple[bool, Any, int]:
        """Parse one value at offset ``start``. Returns (complete, value, next_offset)."""
        if start >= len(self._buf):
            return False, None, start
        marker = self._buf[start : start + 1]
        line = self._read_line(start + 1)
        if line is None:
            return False, None, start
        payload, after = line

        if marker in (b"+", b"-", b":", b"#", b",", b"("):
            return True, self._scalar(marker, payload), after
        if marker == b"_":  # RESP3 null
            return True, None, after
        if marker == b"$":
            return self._bulk(payload, after)
        if marker in (b"*", b"~", b">"):  # array, set, push
            return self._aggregate(payload, after, mapping=False)
        if marker == b"%":  # map
            return self._aggregate(payload, after, mapping=True)
        raise ProtocolError(f"unknown RESP type marker {marker!r}")

    @staticmethod
    def _scalar(marker: bytes, payload: bytes) -> Any:
        text = payload.decode()
        if marker == b":":
            return int(text)
        if marker == b",":
            return float(text)
        if marker == b"(":
            return int(text)
        if marker == b"#":
            return text == "t"
        # simple string (+) and error (-) both surface as text
        return text

    def _bulk(self, payload: bytes, after: int) -> Tuple[bool, Any, int]:
        length = int(payload.decode())
        if length == -1:
            return True, None, after
        end = after + length
        if end + 2 > len(self._buf):
            return False, None, after
        data = bytes(self._buf[after:end])
        return True, data.decode(), end + 2

    def _aggregate(self, payload: bytes, after: int, mapping: bool) -> Tuple[bool, Any, int]:
        count = int(payload.decode())
        if count == -1:
            return True, None, after
        elements = count * 2 if mapping else count
        offset = after
        parsed: List[Any] = []
        for _ in range(elements):
            complete, value, offset = self._parse_at(offset)
            if not complete:
                return False, None, after
            parsed.append(value)
        if mapping:
            return True, {parsed[i]: parsed[i + 1] for i in range(0, len(parsed), 2)}, offset
        return True, parsed, offset
