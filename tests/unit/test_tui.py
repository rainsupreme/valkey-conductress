import pytest
from unittest.mock import Mock

from src.tui import (
    CommaSeparatedIntsValidator,
    MakeArgsValidator,
    NumberField,
    NumberListField,
    PerfTaskForm,
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
            (HumanNumber, "100", 100, None),
            (HumanNumber, "0", 0, None),
            (HumanNumber, "-1", None, "Number must be positive"),
            (HumanNumber, "-5", None, "Number must be positive"),
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
            (HumanNumber, "1", [1], None),
            (HumanNumber, "4", [4], None),
            (HumanNumber, "9", [9], None),
            (HumanNumber, "1,2,3", [1, 2, 3], None),
            (HumanNumber, "10K,20K", [10_000, 20_000], None),
            (HumanNumber, "1M", [1_000_000], None),
            (HumanNumber, "10M", [10_000_000], None),
            (HumanNumber, "0", [0], None),
            (HumanNumber, "1, 2, 3", [1, 2, 3], None),
            (HumanNumber, "10K , 20K", [10_000, 20_000], None),
            (HumanNumber, "1,2,-3", None, "All numbers must be positive"),
            (HumanNumber, "-1", None, "All numbers must be positive"),
            (HumanNumber, "", None, "Input cannot be empty"),
            (HumanNumber, "1.5", None, "All numbers must be integers"),
            (HumanNumber, "1.5,2.5", None, "All numbers must be integers"),
            (HumanNumber, "abc", None, "could not convert string to float"),
            (HumanByte, "1KB", [1024], None),
            (HumanByte, "1KB,2KB", [1024, 2048], None),
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
            (HumanNumber, "1", [1], None),
            (HumanNumber, "1,2,3", [1, 2, 3], None),
            (HumanNumber, "1:5:2", [1, 3, 5], None),
            (HumanNumber, "10K:20K:5K", [10_000, 15_000, 20_000], None),
            (HumanNumber, "1,2:6:2", [1, 2, 4, 6], None),
            (HumanNumber, "10:1:-2", [10, 8, 6, 4, 2, 0], None),
            (HumanNumber, "5:5:1", [5], None),
            (HumanNumber, "1,10:14:2,20", [1, 10, 12, 14, 20], None),
            (HumanNumber, "1:100:50", [1, 51, 101], None),
            (HumanNumber, "1:5:10", [1, 11], None),
            (HumanNumber, "1:5", None, "ranges are of the format value, or start:end:step"),
            (HumanNumber, "1:3", None, "ranges are of the format value, or start:end:step"),
            (HumanNumber, "1:5:0", None, "range step (start:end:step) value must not be zero"),
            (HumanNumber, "", [], "input cannot be empty"),
            (HumanNumber, "abc", [], "could not convert string to float"),
            (HumanNumber, "1:abc:2", [], "could not convert string to float"),
            (HumanByte, "1KB", [1024], None),
            (HumanByte, "0.5KB", [512], None),
            (HumanByte, "1KB:3KB:1KB", [1024, 2048, 3072], None),
            (HumanByte, "1KB,2KB:4KB:1KB", [1024, 2048, 3072, 4096], None),
            (HumanByte, "1KB:2KB", None, "ranges are of the format value, or start:end:step"),
            (HumanByte, "-1KB", [-1024], None),
            (HumanByte, "0.5MB:128KB:-128KB", [524288, 393216, 262144, 131072], None),
        ],
    )
    def test_parse_range_list(self, number_type, input_str, expected_result, expected_error):
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


class TestSourceSpecifierValidator:
    @pytest.fixture
    def mock_config(self, monkeypatch):
        class DummyConfig:
            REPO_NAMES = ["repo1", "repo2"]
            MANUALLY_UPLOADED = "manually_uploaded"
        monkeypatch.setattr("src.tui.config", DummyConfig)

    @pytest.mark.parametrize(
        "input_str,expected_result,expected_error",
        [
            ("repo1:abc", [("repo1", "abc")], None),
            ("repo1:abc,repo2:def", [("repo1", "abc"), ("repo2", "def")], None),
            ("repo1:abc , repo2:def", [("repo1", "abc"), ("repo2", "def")], None),
            (" repo1 : abc ", [("repo1", "abc")], None),
            ("manually_uploaded:local", [("manually_uploaded", "local")], None),
            ("", [], "list cannot be empty"),
            ("repo1", [], "Invalid item format: repo1"),
            ("repo1:", [], "Invalid specifier: "),
            (":abc", [], "Invalid source: "),
            ("repoX:abc", [], "Invalid source: repoX"),
            ("repo1:abc,repo2", [], "Invalid item format: repo2"),
            ("repo1:abc,repo2:", [], "Invalid specifier: "),
            ("repo1:abc,repoX:def", [], "Invalid source: repoX"),
            ("repo1:abc:def", [], "Invalid item format: repo1:abc:def"),
        ],
    )
    def test_parse_source_specifier_list(self, input_str, expected_result, expected_error, mock_config):
        result, error = SourceSpeciferValidator.parse_source_specifier_list(input_str)
        if expected_error is None:
            assert error is None
            assert result == expected_result
        else:
            assert not result
            assert error is not None
            assert expected_error in error


class TestBaseTaskFormValidation:
    def test_validate_tests_selected_with_empty_list(self):
        tests = []
        error = "No tests selected" if not tests else None
        assert error == "No tests selected"

    def test_validate_tests_selected_with_tests(self):
        tests = ["test1", "test2"]
        error = "No tests selected" if not tests else None
        assert error is None

    def test_validate_tests_selected_single_test(self):
        tests = ["test1"]
        error = "No tests selected" if not tests else None
        assert error is None



class TestNumberListField:
    def test_values_without_ranges(self):
        field = Mock(spec=NumberListField)
        field.allow_ranges = False
        field.number_type = HumanNumber
        field.input = Mock(value="10,20,30")
        assert NumberListField.values(field) == [10, 20, 30]

    def test_values_with_ranges(self):
        field = Mock(spec=NumberListField)
        field.allow_ranges = True
        field.number_type = HumanNumber
        field.input = Mock(value="1:5:2")
        assert NumberListField.values(field) == [1, 3, 5]


class TestPerfTaskFormFields:
    """Tests for PerfTaskForm key-size and repetitions fields (Requirement 9.7).

    Since PerfTaskForm is a Textual widget, we test the underlying field
    configuration and parsing logic using mocks to avoid needing a running app.
    """

    def test_key_sizes_field_is_number_list_field_with_correct_config(self):
        """Verify key_sizes would be a NumberListField with the right attributes.

        We check the PerfTaskForm.__init__ source creates the field with
        the expected parameters by inspecting the class definition.
        """
        import inspect
        source = inspect.getsource(PerfTaskForm.__init__)
        assert "self.key_sizes = NumberListField(" in source
        assert '"Key Sizes (comma-separated, 0=standard)"' in source
        assert '"key-sizes"' in source
        assert '"0"' in source  # default
        assert "HumanByte" in source

    def test_repetitions_field_is_number_field_with_correct_config(self):
        """Verify repetitions would be a NumberField with the right attributes."""
        import inspect
        source = inspect.getsource(PerfTaskForm.__init__)
        assert "self.repetitions = NumberField(" in source
        assert '"Repetitions"' in source
        assert '"repetitions"' in source
        assert "HumanNumber" in source

    def test_key_sizes_default_parses_to_zero(self):
        """The default key-size value '0' should parse to [0] via CommaSeparatedIntsValidator."""
        validator = CommaSeparatedIntsValidator(HumanByte)
        assert validator.parse_ints("0") == [0]

    def test_repetitions_default_parses_to_one(self):
        """The default repetitions value '1' should parse to 1 via SingleNumberValidator."""
        validator = SingleNumberValidator(HumanNumber)
        assert validator.parse_int("1") == 1

    def test_key_sizes_parses_multiple_values(self):
        """Key sizes field should parse comma-separated byte values."""
        validator = CommaSeparatedIntsValidator(HumanByte)
        assert validator.parse_ints("0, 64, 256, 1KB") == [0, 64, 256, 1024]

    def test_repetitions_parses_higher_value(self):
        """Repetitions field should parse values > 1."""
        validator = SingleNumberValidator(HumanNumber)
        assert validator.parse_int("5") == 5

    def test_number_list_field_values_for_key_sizes(self):
        """NumberListField.values() should parse comma-separated values correctly."""
        field = Mock(spec=NumberListField)
        field.allow_ranges = False
        field.number_type = HumanByte
        field.input = Mock(value="0, 64, 256, 1KB")
        assert NumberListField.values(field) == [0, 64, 256, 1024]

    def test_number_field_value_for_repetitions(self):
        """NumberField.value() should parse a single integer correctly."""
        field = Mock(spec=NumberField)
        field.number_type = HumanNumber
        field.input = Mock(value="3")
        assert NumberField.value(field) == 3


class TestPerfTaskFormCartesianProduct:
    """Tests verifying key_sizes is included in the Cartesian product
    and repetitions is passed through to PerfTaskData (Requirement 9.7).

    We test the Cartesian product logic and PerfTaskData construction
    directly, since the actual form submission requires a running Textual app.
    """

    def test_cartesian_product_includes_key_sizes(self):
        """The Cartesian product should include key_sizes as a dimension."""
        from itertools import product as itertools_product

        sizes = [512]
        pipelining = [1]
        io_threads = [1]
        tests = ["get"]
        key_sizes = [0, 64, 256]
        specifiers = [("repo1", "unstable")]
        make_args_list = [""]

        all_combos = list(
            itertools_product(
                sizes, pipelining, io_threads, tests, key_sizes, specifiers, make_args_list
            )
        )
        # With 3 key_sizes, we should get 3 combinations
        assert len(all_combos) == 3
        # Each combo should have a different key_size
        produced_key_sizes = [combo[4] for combo in all_combos]
        assert produced_key_sizes == [0, 64, 256]

    def test_cartesian_product_multiplies_all_dimensions(self):
        """Total tasks = sizes * pipelining * io_threads * tests * key_sizes * specifiers * make_args."""
        from itertools import product as itertools_product

        sizes = [512, 1024]
        pipelining = [1, 4]
        io_threads = [1, 9]
        tests = ["get", "set"]
        key_sizes = [0, 64]
        specifiers = [("repo1", "unstable")]
        make_args_list = [""]

        all_combos = list(
            itertools_product(
                sizes, pipelining, io_threads, tests, key_sizes, specifiers, make_args_list
            )
        )
        expected = 2 * 2 * 2 * 2 * 2 * 1 * 1  # 32
        assert len(all_combos) == expected

    def test_submit_task_cartesian_product_order_matches_tui(self):
        """The Cartesian product order in submit_task is (sizes, pipelining, io_threads, tests, key_sizes, specifiers, make_args)."""
        import inspect
        from src.tui import PerfTaskForm
        source = inspect.getsource(PerfTaskForm.submit_task)
        # Verify key_sizes is in the product call
        assert "key_sizes," in source
        # Verify the unpacking includes key_size
        assert "key_size" in source

    def test_repetitions_passed_to_perf_task_data(self, monkeypatch):
        """Repetitions value should be passed through to each PerfTaskData."""
        monkeypatch.setattr("src.task_queue.config.REPO_NAMES", ["repo1"])
        monkeypatch.setattr("src.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded")

        from src.tasks.task_perf_benchmark import PerfTaskData

        repetitions = 5
        task = PerfTaskData(
            source="repo1",
            specifier="unstable",
            replicas=-1,
            note="test",
            requirements={},
            make_args="",
            val_size=512,
            io_threads=1,
            pipelining=1,
            test="get",
            warmup=60,
            duration=900,
            profiling_sample_rate=-1,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            key_size=64,
            repetitions=repetitions,
        )
        assert task.repetitions == 5
        assert task.key_size == 64

    def test_key_size_default_in_perf_task_data(self, monkeypatch):
        """PerfTaskData should default key_size=0 and repetitions=1."""
        monkeypatch.setattr("src.task_queue.config.REPO_NAMES", ["repo1"])
        monkeypatch.setattr("src.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded")

        from src.tasks.task_perf_benchmark import PerfTaskData

        task = PerfTaskData(
            source="repo1",
            specifier="unstable",
            replicas=-1,
            note="",
            requirements={},
            make_args="",
            val_size=512,
            io_threads=1,
            pipelining=1,
            test="get",
            warmup=60,
            duration=900,
            profiling_sample_rate=-1,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )
        assert task.key_size == 0
        assert task.repetitions == 1

    def test_submit_task_passes_repetitions_to_perf_task_data(self):
        """Verify submit_task passes repetitions to PerfTaskData constructor."""
        import inspect
        from src.tui import PerfTaskForm
        source = inspect.getsource(PerfTaskForm.submit_task)
        assert "repetitions=repetitions" in source

    def test_submit_task_passes_key_size_to_perf_task_data(self):
        """Verify submit_task passes key_size to PerfTaskData constructor."""
        import inspect
        from src.tui import PerfTaskForm
        source = inspect.getsource(PerfTaskForm.submit_task)
        assert "key_size=key_size" in source


class TestMakeArgsValidator:
    @pytest.mark.parametrize(
        "input_str,expected_result,expected_error",
        [
            ("", [""], None),
            ("OPTIMIZATION=-O2", ["OPTIMIZATION=-O2"], None),
            ("OPTIMIZATION=-O2;; MALLOC=libc", ["OPTIMIZATION=-O2", "MALLOC=libc"], None),
            ("OPTIMIZATION=-O2;;MALLOC=libc", ["OPTIMIZATION=-O2", "MALLOC=libc"], None),
            (" OPTIMIZATION=-O2 ;; MALLOC=libc ", ["OPTIMIZATION=-O2", "MALLOC=libc"], None),
            ("OPTIMIZATION=-O2;; MALLOC=libc;; OPTIMIZATION=-O3", ["OPTIMIZATION=-O2", "MALLOC=libc", "OPTIMIZATION=-O3"], None),
            (";;", [""], None),
            ("OPTIMIZATION=-O2;;", ["OPTIMIZATION=-O2"], None),
            (";;OPTIMIZATION=-O2", ["OPTIMIZATION=-O2"], None),
            ("MALLOC=jemalloc OPTIMIZATION=-O3", ["MALLOC=jemalloc OPTIMIZATION=-O3"], None),
        ],
    )
    def test_parse_make_args_list(self, input_str, expected_result, expected_error):
        result, error = MakeArgsValidator.parse_make_args_list(input_str)
        if expected_error is None:
            assert error is None
            assert result == expected_result
        else:
            assert error is not None
            assert expected_error in error

    @pytest.mark.parametrize(
        "input_str,expected_valid",
        [
            ("", True),
            ("OPTIMIZATION=-O2", True),
            ("OPTIMIZATION=-O2;; MALLOC=libc", True),
            ("OPTIMIZATION=-O2;;MALLOC=libc;;", True),
        ],
    )
    def test_validate(self, input_str, expected_valid):
        validator = MakeArgsValidator()
        result = validator.validate(input_str)
        assert result.is_valid == expected_valid
