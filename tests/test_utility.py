import pytest

from src.utility import (
    HumanByte,
    HumanNumber,
    HumanTime,
    calc_percentile_averages,
    print_centered_text,
    print_inline_header,
    print_pretty_divider,
)

TEST_CONSOLE_WIDTH = 50


class TestHumanNumber:
    def test_basic_conversion(self):
        assert HumanNumber.to_human(1500) == "1.5K"
        assert HumanNumber.to_human(1000000) == "1M"

    def test_decimal_places(self):
        assert HumanNumber.to_human(1234, decimals=0) == "1K"
        assert HumanNumber.to_human(1234, decimals=2) == "1.23K"

    def test_from_human(self):
        result = HumanNumber.from_human("1.5K")
        assert result == 1500
        result = HumanNumber.from_human("3m")
        assert result == 3_000_000
        result = HumanNumber.from_human("3M")
        assert result == 3_000_000

    def test_integer_display(self):
        assert HumanNumber.to_human(1000) == "1K"
        assert HumanNumber.to_human(2000.0) == "2K"


class TestHumanByte:
    def test_byte_conversion(self):
        assert HumanByte.to_human(1024) == "1KB"
        assert HumanByte.to_human(1024 * 1024) == "1MB"
        assert HumanByte.to_human(1024 * 1024 * 1024) == "1GB"

    def test_partial_units(self):
        assert HumanByte.to_human(1536) == "1.5KB"
        assert HumanByte.to_human(1536, decimals=2) == "1.50KB"

    def test_from_human_bytes(self):
        assert HumanByte.from_human("1.5KB") == 1024 * 1.5
        assert HumanByte.from_human("1MB") == 1024 * 1024


class TestHumanTime:
    def test_time_conversion(self):
        assert HumanTime.to_human(30) == "30s"
        assert HumanTime.to_human(60) == "1m"
        assert HumanTime.to_human(3600) == "1h"
        assert HumanTime.to_human(86400) == "1d"
        assert HumanTime.to_human(604800) == "1w"

    def test_partial_time_units(self):
        assert HumanTime.to_human(90) == "1.5m"
        assert HumanTime.to_human(5400) == "1.5h"

    def test_threshold_behavior(self):
        assert HumanTime.to_human(59) == "59s"
        assert HumanTime.to_human(60) == "1m"

    def test_from_human_time(self):
        assert HumanTime.from_human("30s") == 30
        assert HumanTime.from_human("1h") == 3600
        assert HumanTime.from_human("1.5m") == 90
        assert HumanTime.from_human("1d") == 86400
        assert HumanTime.from_human("1w") == 604800

        assert HumanTime.from_human("1D") == 86400
        assert HumanTime.from_human("1W") == 604800

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (45, "45s"),
            (60, "1m"),
            (90, "1.5m"),
            (3600, "1h"),
            (86400, "1d"),
            (604800, "1w"),
        ],
    )
    def test_various_times(self, seconds, expected):
        assert HumanTime.to_human(seconds) == expected


def test_error_cases():
    with pytest.raises(ValueError):
        HumanNumber.from_human("invalid")

    with pytest.raises(ValueError):
        HumanNumber.from_human("1X")


class TestPrintFunctions:

    @pytest.fixture
    def mock_console_width(self, monkeypatch):
        monkeypatch.setattr("src.utility.get_console_width", lambda: TEST_CONSOLE_WIDTH)
        return TEST_CONSOLE_WIDTH

    @pytest.fixture
    def captured_print(self, capsys):
        # pytest's built-in capture of stdout
        def get_printed():
            captured = capsys.readouterr()
            return captured.out[:-1]  # Remove the trailing newline

        return get_printed

    def test_inline_header(self, mock_console_width, captured_print):
        print_inline_header("Test")
        printed_text = captured_print()
        assert len(printed_text) == mock_console_width
        assert "Test" in printed_text

    def test_pretty_divider(self, mock_console_width, captured_print):
        print_pretty_divider()
        printed_text = captured_print()
        assert abs(len(printed_text) - mock_console_width) <= 1  # Allow for odd lengths
        assert printed_text == printed_text[::-1]  # Check if it's symmetric

    def test_centered_text(self, mock_console_width, captured_print):
        test_text = "Center Me"
        print_centered_text(test_text)
        printed_text = captured_print()

        # Verify length
        assert len(printed_text) == mock_console_width

        # Verify centering
        left_space = printed_text.index(test_text)
        right_space = mock_console_width - (left_space + len(test_text))
        assert abs(left_space - right_space) <= 1  # Allow for odd lengths

    @pytest.mark.parametrize(
        "text,width",
        [
            ("Short", 50),
            ("This is a longer piece of text", 50),
            ("X", 50),
            ("", 50),
        ],
    )
    def test_various_text_lengths(self, monkeypatch, captured_print, text, width):
        monkeypatch.setattr("src.utility.get_console_width", lambda: width)
        print_centered_text(text)
        printed_text = captured_print()
        assert len(printed_text) == width
        assert text in printed_text


class TestPercentileAverages:
    def test_basic_percentile_avg(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # top 20% should be [9, 10], average = 9.5
        # top 50% should be [6, 7, 8, 9, 10], average = 8
        result = calc_percentile_averages(data, [20, 50])
        assert result == [9.5, 8.0]

    def test_lowest_vals(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # bottom 20% should be [1, 2], average = 1.5
        # bottom 50% should be [1, 2, 3, 4, 5], average = 3
        result = calc_percentile_averages(data, [20, 50], lowest_vals=True)
        assert result == [1.5, 3.0]

    def test_single_percentage(self):
        data = [1, 2, 3, 4, 5]
        # top 40% should be [4, 5], average = 4.5
        result = calc_percentile_averages(data, [40])
        assert result == [4.5]

    def test_empty_list(self):
        with pytest.raises(ZeroDivisionError):
            calc_percentile_averages([], [50])

    def test_original_list_unchanged(self):
        original = [3, 1, 4, 1, 5]
        data_copy = original.copy()
        calc_percentile_averages(original, [50])
        assert original == data_copy  # original should be unchanged

    @pytest.mark.parametrize(
        "data,percentages,lowest,expected",
        [
            ([1, 2, 3, 4], [50], False, [3.5]),  # top 50% of 4 numbers
            ([1, 2, 3, 4], [50], True, [1.5]),  # bottom 50% of 4 numbers
            ([1, 2, 3, 4, 5], [20], False, [5.0]),  # top 20% of 5 numbers
            ([1, 2, 3, 4, 5], [20, 40], False, [5.0, 4.5]),  # multiple percentages
        ],
    )
    def test_various_scenarios(self, data, percentages, lowest, expected):
        result = calc_percentile_averages(data, percentages, lowest_vals=lowest)
        assert result == expected

    def test_float_values(self):
        data = [1.5, 2.5, 3.5, 4.5]
        result = calc_percentile_averages(data, [50])
        assert result == [4.0]

    def test_negative_values(self):
        data = [-4, -3, -2, -1]
        result = calc_percentile_averages(data, [50])
        assert result == [-1.5]
