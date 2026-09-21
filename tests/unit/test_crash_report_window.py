"""crash_report_window picks the LAST crash report, START through END.

Falsified against the previous ``tail -1 | cut -d: -f1`` selection, which
matched the END marker of the last report and printed from there to EOF: the
runner log then carried everything AFTER the report and none of it (seen on
g4bench Sep 21 2026, eight assertion crashes, every capture empty of the
stack trace).
"""

from conductress.server import CRASH_REPORT_CONTEXT_LINES, crash_report_window

# grep -n output for a logfile that has seen two crashes (the g4bench shape:
# one logfile per port, appended across every instance that ran on it).
TWO_REPORTS = "\n".join(
    [
        "2049:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",
        "3027:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
        "12624:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",
        "13612:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
    ]
)


def test_picks_last_report_start_through_end():
    assert crash_report_window(TWO_REPORTS) == (12624 - CRASH_REPORT_CONTEXT_LINES, 13612)


def test_old_selection_would_have_anchored_on_end():
    # The behaviour this replaces: last matching line is the END marker.
    last = int(TWO_REPORTS.splitlines()[-1].split(":", 1)[0])
    start, end = crash_report_window(TWO_REPORTS)
    assert last == 13612 and start < 12624 and end == last


def test_open_report_prints_to_eof():
    # Process died mid-report (or the log was read before END was flushed).
    markers = TWO_REPORTS + "\n20066:=== VALKEY BUG REPORT START: Cut & paste starting from here ==="
    assert crash_report_window(markers) == (20066 - CRASH_REPORT_CONTEXT_LINES, None)


def test_signal_line_before_start_widens_the_window():
    markers = "\n".join(
        [
            "100:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",
            "400:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
            "900:1234:M 21 Sep 2026 16:15:35.455 # Valkey 8.1.0 crashed by signal: 11, si_code: 1",
            "903:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",
            "1500:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
        ]
    ).replace("crashed by signal", "CRASHED BY SIGNAL")
    assert crash_report_window(markers) == (900 - CRASH_REPORT_CONTEXT_LINES, 1500)


def test_signal_from_an_earlier_report_is_not_attached():
    markers = "\n".join(
        [
            "100:# CRASHED BY SIGNAL 11",
            "103:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",
            "400:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
            "903:=== VALKEY BUG REPORT START: Cut & paste starting from here ===",  # assertion: no signal line
            "1500:=== VALKEY BUG REPORT END. Make sure to include from START to END. ===",
        ]
    )
    assert crash_report_window(markers) == (903 - CRASH_REPORT_CONTEXT_LINES, 1500)


def test_signal_without_report_still_anchors():
    assert crash_report_window("77:# CRASHED BY SIGNAL 6") == (77 - CRASH_REPORT_CONTEXT_LINES, None)


def test_start_is_clamped_to_line_one():
    assert crash_report_window("3:=== VALKEY BUG REPORT START: Cut & paste starting from here ===") == (1, None)


def test_no_markers():
    assert crash_report_window("") is None
    assert crash_report_window("garbage without a line number\n") is None
