"""Unit tests for the connection-storm generator's pure modules.

Policy parsing/delays, RESP encode/parse (including replies split across
reads), the netstat counter parser, and the metric reductions on synthetic
events with hand-computed answers. No sockets, no server.
"""

import random

import pytest

from conductress.stormgen import metrics, netstat
from conductress.stormgen.policy import ExponentialPolicy, FixedPolicy, ImmediatePolicy, parse_policy
from conductress.stormgen.resp import ProtocolError, ReplyParser, encode_command

# --------------------------------------------------------------------------- policy


def test_parse_immediate():
    policy = parse_policy("immediate")
    assert isinstance(policy, ImmediatePolicy)
    assert policy.next_delay(0) == 0.0 and policy.next_delay(9) == 0.0
    assert policy.describe() == "immediate"


def test_parse_fixed_delay_is_constant_in_seconds():
    policy = parse_policy("fixed:250")
    assert isinstance(policy, FixedPolicy)
    assert policy.next_delay(0) == pytest.approx(0.25)
    assert policy.next_delay(5) == pytest.approx(0.25)
    assert policy.describe() == "fixed:250"


def test_parse_exponential_doubles_and_caps():
    policy = parse_policy("exp:100:800")
    assert isinstance(policy, ExponentialPolicy)
    assert policy.next_delay(0) == pytest.approx(0.1)
    assert policy.next_delay(1) == pytest.approx(0.2)
    assert policy.next_delay(2) == pytest.approx(0.4)
    assert policy.next_delay(3) == pytest.approx(0.8)
    assert policy.next_delay(4) == pytest.approx(0.8)  # capped
    assert policy.next_delay(50) == pytest.approx(0.8)  # no overflow at large index
    assert policy.describe() == "exp:100:800"


def test_exponential_jitter_stays_within_bounds_and_is_seeded():
    policy = parse_policy("exp:100:800:jitter", rng=random.Random(1234))
    for attempt in range(6):
        capped = min(0.8, 0.1 * (2**attempt))
        delay = policy.next_delay(attempt)
        assert 0.0 <= delay <= capped + 1e-9
    assert policy.describe() == "exp:100:800:jitter"


@pytest.mark.parametrize(
    "spec",
    ["", "bogus", "fixed", "fixed:abc", "exp:100", "exp:100:50", "exp:1:2:nope", "immediate:5"],
)
def test_parse_policy_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_policy(spec)


# --------------------------------------------------------------------------- resp encode


def test_encode_command_get():
    assert encode_command("GET", "key") == b"*2\r\n$3\r\nGET\r\n$3\r\nkey\r\n"


def test_encode_command_requires_args():
    with pytest.raises(ValueError):
        encode_command()


# --------------------------------------------------------------------------- resp parse


def _parse_one(data: bytes):
    parser = ReplyParser()
    parser.feed(data)
    complete, value = parser.try_parse()
    assert complete, "expected a complete reply"
    return value


def test_parse_simple_string_error_integer():
    assert _parse_one(b"+OK\r\n") == "OK"
    assert _parse_one(b"-ERR bad\r\n") == "ERR bad"
    assert _parse_one(b":42\r\n") == 42


def test_parse_bulk_and_null_bulk():
    assert _parse_one(b"$5\r\nhello\r\n") == "hello"
    assert _parse_one(b"$-1\r\n") is None


def test_parse_array_and_null_array():
    assert _parse_one(b"*2\r\n$3\r\nfoo\r\n:7\r\n") == ["foo", 7]
    assert _parse_one(b"*-1\r\n") is None


def test_parse_resp3_map_like_hello_reply():
    # %2 map: {"server": "valkey", "proto": 3}
    payload = b"%2\r\n$6\r\nserver\r\n$6\r\nvalkey\r\n$5\r\nproto\r\n:3\r\n"
    assert _parse_one(payload) == {"server": "valkey", "proto": 3}


def test_parse_resp3_scalars():
    assert _parse_one(b"_\r\n") is None
    assert _parse_one(b"#t\r\n") is True
    assert _parse_one(b"#f\r\n") is False
    assert _parse_one(b",3.14\r\n") == pytest.approx(3.14)


def test_incomplete_reply_needs_more_bytes():
    parser = ReplyParser()
    parser.feed(b"$5\r\nhel")
    complete, value = parser.try_parse()
    assert not complete and value is None


def test_reply_split_across_reads_is_reassembled():
    parser = ReplyParser()
    for chunk in (b"$5\r", b"\nhe", b"llo", b"\r\n"):
        parser.feed(chunk)
        complete, value = parser.try_parse()
        if complete:
            assert value == "hello"
            break
    else:
        pytest.fail("reply never completed across chunked feeds")


def test_map_split_across_reads():
    payload = b"%1\r\n$3\r\nkey\r\n:9\r\n"
    parser = ReplyParser()
    result = None
    for i in range(len(payload)):
        parser.feed(payload[i : i + 1])
        complete, value = parser.try_parse()
        if complete:
            result = value
            break
    assert result == {"key": 9}


def test_two_replies_in_one_buffer():
    parser = ReplyParser()
    parser.feed(b"+FIRST\r\n:2\r\n")
    assert parser.try_parse() == (True, "FIRST")
    assert parser.try_parse() == (True, 2)
    assert parser.try_parse() == (False, None)


def test_unknown_marker_raises():
    parser = ReplyParser()
    parser.feed(b"!oops\r\n")
    with pytest.raises(ProtocolError):
        parser.try_parse()


# --------------------------------------------------------------------------- netstat


_NETSTAT_FIXTURE = (
    "TcpExt: SyncookiesSent SyncookiesRecv ListenOverflows ListenDrops TCPHPHits\n"
    "TcpExt: 0 0 12 15 999\n"
    "IpExt: InNoRoutes InTruncatedPkts\n"
    "IpExt: 1 2\n"
)


def test_parse_netstat_reads_listen_counters():
    counters = netstat.parse_netstat(_NETSTAT_FIXTURE)
    assert counters.listen_overflows == 12
    assert counters.listen_drops == 15


def test_parse_netstat_missing_fields_default_to_zero():
    counters = netstat.parse_netstat("TcpExt: SyncookiesSent\nTcpExt: 3\n")
    assert counters.listen_overflows == 0 and counters.listen_drops == 0


def test_counter_delta_computes_difference():
    before = netstat.parse_netstat(_NETSTAT_FIXTURE)
    after = netstat.parse_netstat(
        "TcpExt: SyncookiesSent SyncookiesRecv ListenOverflows ListenDrops TCPHPHits\n" "TcpExt: 0 0 50 60 999\n"
    )
    delta = netstat.counter_delta(before, after)
    assert delta == {"ListenOverflows": 38, "ListenDrops": 45}


def test_counter_delta_is_none_when_a_snapshot_is_missing():
    some = netstat.parse_netstat(_NETSTAT_FIXTURE)
    assert netstat.counter_delta(None, some) == {"ListenOverflows": None, "ListenDrops": None}
    assert netstat.counter_delta(some, None) == {"ListenOverflows": None, "ListenDrops": None}


def test_read_counters_missing_path_returns_none(tmp_path):
    assert netstat.read_counters(str(tmp_path / "does-not-exist")) is None


# --------------------------------------------------------------------------- metrics


def _ev(client_id, attempt, t_start, t_end, outcome):
    return {"client_id": client_id, "attempt": attempt, "t_start": t_start, "t_end": t_end, "outcome": outcome}


def _storm_events():
    # 2 clients. Client 0: one connect_timeout then connected. Client 1: connected first try.
    # Times chosen to sit mid-bucket (not on 100ms edges) so float division is unambiguous.
    return [
        _ev(0, 0, 0.02, 0.55, "connect_timeout"),
        _ev(0, 1, 0.72, 1.05, "connected"),
        _ev(1, 0, 0.12, 0.22, "connected"),
    ]


def test_totals_amplification():
    result = metrics.totals(_storm_events(), client_count=2)
    assert result["clients"] == 2
    assert result["attempts"] == 3
    assert result["amplification"] == pytest.approx(1.5)
    assert result["connected_clients"] == 2


def test_outcome_counts_has_every_known_outcome():
    counts = metrics.outcome_counts(_storm_events())
    assert counts["connected"] == 2
    assert counts["connect_timeout"] == 1
    for outcome in metrics.OUTCOMES:
        assert outcome in counts
    assert counts["refused"] == 0


def test_time_to_quiescence_from_both_origins():
    q = metrics.time_to_quiescence(_storm_events(), client_count=2, storm_start=0.0, stall_end=0.6)
    assert q["from_start"] == pytest.approx(1.05)  # last t_end
    assert q["from_stall_end"] == pytest.approx(0.45)  # 1.05 - 0.6


def test_time_to_quiescence_none_when_a_client_never_connects():
    events = [_ev(0, 0, 0.0, 0.5, "connected"), _ev(1, 0, 0.0, 0.5, "reset")]
    q = metrics.time_to_quiescence(events, client_count=2, storm_start=0.0, stall_end=None)
    assert q["from_start"] is None
    assert q["from_stall_end"] is None


def test_time_to_quiescence_no_stall_leaves_stall_origin_none():
    q = metrics.time_to_quiescence(_storm_events(), client_count=2, storm_start=0.0, stall_end=None)
    assert q["from_start"] == pytest.approx(1.05)
    assert q["from_stall_end"] is None


def test_attempt_latency_percentiles():
    lat = metrics.attempt_latency_percentiles(_storm_events())
    # durations (ms): 530, 330, 100 -> sorted 100,330,530
    assert lat["max_ms"] == pytest.approx(530.0)
    assert lat["p50_ms"] == pytest.approx(330.0)


def test_timeline_buckets_started_connected_timeouts():
    tl = metrics.timeline(_storm_events(), bucket_ms=100, storm_start=0.0)
    by_t = {b["t_ms"]: b for b in tl}
    # client1 started at 0.12 (bucket 100) connected at 0.22 (bucket 200)
    assert by_t[100]["started"] == 1
    assert by_t[200]["connected"] == 1
    # client0 connect_timeout ended at 0.55 (bucket 500)
    assert by_t[500]["timeouts"] == 1
    # client0 connected attempt ended at 1.05 (bucket 1000)
    assert by_t[1000]["connected"] == 1


def test_timeline_rejects_bad_bucket():
    with pytest.raises(ValueError):
        metrics.timeline(_storm_events(), bucket_ms=0)


def test_timeline_empty_events():
    assert metrics.timeline([], bucket_ms=100) == []


def test_summarize_assembles_all_sections():
    doc = metrics.summarize(_storm_events(), client_count=2, stall_end=0.6)
    assert set(doc) == {"totals", "outcomes", "attempt_latency", "time_to_quiescence", "timeline"}
    assert doc["totals"]["amplification"] == pytest.approx(1.5)
