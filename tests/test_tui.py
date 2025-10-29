import pytest

from src.tui import (
    CommaSeparatedIntsValidator,
    RangeListValidator,
    SingleNumberValidator,
    SourceSpeciferValidator,
)
from src.utility import HumanByte, HumanNumber, HumanTime


class TestSingleNumberValidator:
    @pytest.mark.parametrize(
        "number_type,input_str,expected_result,expected_error",
        [
            (HumanNumber, "1", 1, None),
            (HumanNumber, "10K", 10_000, None),
            (HumanNumber, "1M", 1_000_000, None),
            (HumanNumber, "0", 0, None),
            (HumanNumber, "-1", None, "Number must be positive"),
            (HumanNumber, "", None, "Input cannot be empty"),
            (HumanNumber, "1.5", None, "Number must be an integer"),
            (HumanNumber, "abc", None, "could not convert string to float"),
            (HumanNumber, "12badunit", None, "Invalid unit"),
            (HumanByte, "1KB", 1024, None),
            (HumanByte, "2MB", 2 * 1024 * 1024, None),
            (HumanByte, "1.5KB", 1536, None),
            (HumanByte, "-1KB", None, "Number must be positive"),
            (HumanTime, "5m", 300, None),
            (HumanTime, "1h", 3600, None),
            (HumanTime, "30s", 30, None),
        ],
    )
    def test_parse_int(self, number_type, input_str, expected_result, expected_error):
        validator = SingleNumberValidator(number_type)
        if expected_error is None:
            assert validator.parse_int(input_str) == expected_result
        else:
            with pytest.raises(ValueError) as excinfo:
                validator.parse_int(input_str)
            assert expected_error in str(excinfo.value)

    @pytest.mark.parametrize(
        "number_type,input_str,expected_valid",
        [
            (HumanNumber, "1", True),
            (HumanNumber, "10K", True),
            (HumanNumber, "1.5", False),
            (HumanNumber, "-1", False),
            (HumanNumber, "", False),
            (HumanNumber, "abc", False),
            (HumanByte, "1KB", True),
            (HumanByte, "1.5KB", True),
            (HumanByte, "-1KB", False),
            (HumanTime, "5m", True),
            (HumanTime, "1h", True),
        ],
    )
    def test_validate(self, number_type, input_str, expected_valid):
        validator = SingleNumberValidator(number_type)
        result = validator.validate(input_str)
        assert result.is_valid == expected_valid


class TestCommaSeparatedIntsValidator:
    @pytest.mark.parametrize(
        "number_type,input_str,expected_result,expected_error",
        [
            # HumanNumber tests
            (HumanNumber, "1", [1], None),
            (HumanNumber, "1,2,3", [1, 2, 3], None),
            (HumanNumber, "10K,20K", [10_000, 20_000], None),
            (HumanNumber, "1M", [1_000_000], None),
            (HumanNumber, "0", [0], None),
            (HumanNumber, "1,2,-3", None, "All numbers must be positive"),
            (HumanNumber, "", None, "Input cannot be empty"),
            (HumanNumber, "1.5", None, "All numbers must be integers"),
            (HumanNumber, "abc", None, "could not convert string to float"),
            # HumanByte tests
            (HumanByte, "1KB", [1024], None),
            (HumanByte, "2MB,512B", [2 * 1024 * 1024, 512], None),
            (HumanByte, "1.5KB", [1536], None),
            (HumanByte, "-1KB", None, "All numbers must be positive"),
        ],
    )
    def test_parse_ints(self, number_type, input_str, expected_result, expected_error):
        validator = CommaSeparatedIntsValidator(number_type)
        if expected_error is None:
            assert validator.parse_ints(input_str) == expected_result
        else:
            with pytest.raises(ValueError) as excinfo:
                validator.parse_ints(input_str)
            assert expected_error in str(excinfo.value)

    @pytest.mark.parametrize(
        "number_type,input_str,expected_valid",
        [
            (HumanNumber, "1,2,3", True),
            (HumanNumber, "10K,20K", True),
            (HumanNumber, "1.5", False),
            (HumanNumber, "-1", False),
            (HumanNumber, "", False),
            (HumanNumber, "abc", False),
            (HumanByte, "1KB,2MB", True),
            (HumanByte, "1.5KB", True),
            (HumanByte, "-1KB", False),
        ],
    )
    def test_validate(self, number_type, input_str, expected_valid):
        validator = CommaSeparatedIntsValidator(number_type)
        result = validator.validate(input_str)
        assert result.is_valid == expected_valid


class TestRangeListValidator:

    @pytest.mark.parametrize(
        "number_type,input_str,expected_result,expected_error",
        [
            # HumanNumber tests
            (HumanNumber, "1", [1], None),
            (HumanNumber, "1,2,3", [1, 2, 3], None),
            (HumanNumber, "1:5:2", [1, 3, 5], None),
            (HumanNumber, "10K:20K:5K", [10_000, 15_000, 20_000], None),
            (HumanNumber, "1,2:6:2", [1, 2, 4, 6], None),
            (
                HumanNumber,
                "1:3",
                None,
                "ranges are of the format value, or start:end:step",
            ),
            (
                HumanNumber,
                "1:5:0",
                None,
                "range step (start:end:step) value must not be zero",
            ),
            (HumanNumber, "", [], "input cannot be empty"),
            (HumanNumber, "abc", [], "could not convert string to float"),
            (HumanNumber, "1:abc:2", [], "could not convert string to float"),
            # HumanByte tests
            (HumanByte, "1KB", [1024], None),
            (HumanByte, "1KB:3KB:1KB", [1024, 2048, 3072], None),
            (HumanByte, "1KB,2KB:4KB:1KB", [1024, 2048, 3072, 4096], None),
            (
                HumanByte,
                "1KB:2KB",
                None,
                "ranges are of the format value, or start:end:step",
            ),
            (HumanByte, "-1KB", [-1024], None),
            (HumanByte, "0.5MB:128KB:-128KB", [524288, 393216, 262144, 131072], None),
        ],
    )
    def test_parse_range_list(
        self, number_type, input_str, expected_result, expected_error
    ):
        validator = RangeListValidator(number_type)
        result, error = validator.parse_range_list(input_str)
        if expected_error is None:
            assert error is None
            assert result == expected_result
        else:
            assert result == []
            assert error is not None
            assert expected_error in error

    @pytest.mark.parametrize(
        "number_type,input_str,expected_valid",
        [
            (HumanNumber, "1", True),
            (HumanNumber, "1,2,3", True),
            (HumanNumber, "1:5:2", True),
            (HumanNumber, "1:3", False),
            (HumanNumber, "", False),
            (HumanNumber, "abc", False),
            (HumanByte, "1KB", True),
            (HumanByte, "1KB:3KB:1KB", True),
            (HumanByte, "1KB:2KB", False),
        ],
    )
    def test_validate(self, number_type, input_str, expected_valid):
        validator = RangeListValidator(number_type)
        result = validator.validate(input_str)
        assert result.is_valid == expected_valid


@pytest.mark.parametrize(
    "input_str,expected_result,expected_error",
    [
        # Valid cases
        ("repo1:abc", [("repo1", "abc")], None),
        ("repo1:abc,repo2:def", [("repo1", "abc"), ("repo2", "def")], None),
        ("repo1:abc , repo2:def", [("repo1", "abc"), ("repo2", "def")], None),
        (
            "manually_uploaded:local",
            [("manually_uploaded", "local")],
            None,
        ),
        # Invalid cases
        ("", [], "list cannot be empty"),
        ("repo1", [], "Invalid item format: repo1"),
        ("repo1:", [], "Invalid specifier: "),
        (":abc", [], "Invalid source: "),
        ("repoX:abc", [], "Invalid source: repoX"),
        ("repo1:abc,repo2", [], "Invalid item format: repo2"),
        ("repo1:abc,repo2:", [], "Invalid specifier: "),
        ("repo1:abc,repoX:def", [], "Invalid source: repoX"),
    ],
)
def test_parse_source_specifier_list(
    input_str, expected_result, expected_error, monkeypatch
):
    # Patch config.REPO_NAMES and config.MANUALLY_UPLOADED for test
    class DummyConfig:
        REPO_NAMES = ["repo1", "repo2"]
        MANUALLY_UPLOADED = "manually_uploaded"

    monkeypatch.setattr("src.tui.config", DummyConfig)

    result, error = SourceSpeciferValidator.parse_source_specifier_list(input_str)
    if expected_error is None:
        assert error is None
        assert result == expected_result
    else:
        assert not result
        assert error is not None
        assert expected_error in error
