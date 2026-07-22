"""Property-based and unit tests for the CLI module.

Tests cover:
- Property 4: Cartesian product task count (Requirement 3.3)
- Property 5: Source validation (Requirements 3.4, 3.5)
- Unit tests for argument parsing, validation, and task submission (Requirement 9.3)
"""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from conductress import config
from conductress.cli import generate_task_combinations, main, validate_source

# Patch config for tests (consistent with other test modules)
config.MANUALLY_UPLOADED = "manual"
config.REPO_NAMES = ["repo1", "repo2"]


class TestCartesianProductProperties:
    """Property-based tests for Cartesian product task count."""

    @settings(max_examples=100)
    @given(
        tests=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        sizes=st.lists(
            st.integers(min_value=1, max_value=1_000_000),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        io_threads=st.lists(
            st.integers(min_value=1, max_value=32),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        pipelining=st.lists(
            st.integers(min_value=1, max_value=64),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        key_sizes=st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    def test_cartesian_product_task_count(self, tests, sizes, io_threads, pipelining, key_sizes):
        """**Validates: Requirements 3.3**

        Feature: benchmark-tooling-enhancements, Property 4: Cartesian product task count

        For any non-empty lists of tests, sizes, io_threads, pipelining, and key_sizes,
        the number of generated task combinations must equal the product of all list lengths,
        and every unique combination of (test, size, io_thread, pipeline, key_size) must
        appear exactly once.
        """
        result = generate_task_combinations(tests, sizes, io_threads, pipelining, key_sizes)

        # Verify count equals product of lengths
        expected_count = len(tests) * len(sizes) * len(io_threads) * len(pipelining) * len(key_sizes)
        assert len(result) == expected_count, f"Expected {expected_count} combinations, got {len(result)}"

        # Verify every unique combination appears exactly once (no duplicates)
        assert len(result) == len(
            set(result)
        ), f"Found duplicate combinations: {len(result)} total vs {len(set(result))} unique"

        # Verify every combination is a valid tuple from the input lists
        for combo in result:
            test, size, io_thread, pipeline, key_size = combo
            assert test in tests, f"Test '{test}' not in input tests"
            assert size in sizes, f"Size {size} not in input sizes"
            assert io_thread in io_threads, f"IO thread {io_thread} not in input io_threads"
            assert pipeline in pipelining, f"Pipeline {pipeline} not in input pipelining"
            assert key_size in key_sizes, f"Key size {key_size} not in input key_sizes"


class TestSourceValidationProperties:
    """Property-based tests for source validation."""

    @settings(max_examples=100)
    @given(s=st.text(min_size=0, max_size=50))
    def test_source_validation(self, s):
        """**Validates: Requirements 3.4, 3.5**

        Feature: benchmark-tooling-enhancements, Property 5: Source validation

        For any string s, validate_source(s) must return True if and only if
        s is in config.REPO_NAMES or s equals config.MANUALLY_UPLOADED.
        All other strings must be rejected.
        """
        valid_set = set(config.REPO_NAMES) | {config.MANUALLY_UPLOADED}
        expected = s in valid_set
        result = validate_source(s)
        assert result == expected, (
            f"validate_source({s!r}) returned {result}, expected {expected}. " f"Valid sources: {valid_set}"
        )


# ---------------------------------------------------------------------------
# Unit tests for CLI (Requirement 9.3)
# ---------------------------------------------------------------------------


class TestGenerateTaskCombinations:
    """Unit tests for generate_task_combinations()."""

    def test_single_combination(self):
        """A single value per parameter produces exactly 1 combination."""
        result = generate_task_combinations(["get"], [512], [1], [1], [0])
        assert len(result) == 1
        assert result[0] == ("get", 512, 1, 1, 0)

    def test_full_cartesian_product(self):
        """Multiple values per parameter produce the full Cartesian product."""
        result = generate_task_combinations(["get", "set"], [512, 1024], [1, 9], [1, 4], [0, 64])
        assert len(result) == 2 * 2 * 2 * 2 * 2  # 32 combinations


class TestValidateSource:
    """Unit tests for validate_source()."""

    def test_valid_repo_name(self):
        assert validate_source("repo1") is True

    def test_valid_manually_uploaded(self):
        assert validate_source("manual") is True

    def test_invalid_source(self):
        assert validate_source("nonexistent") is False

    def test_empty_string(self):
        assert validate_source("") is False


class TestQueueAddSubcommand:
    """Unit tests for the 'queue add' subcommand via main()."""

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_correct_number_of_tasks(self, mock_queue_cls):
        """main(["queue", "add", ...]) with 2 tests, 2 sizes, 2 io-threads, 2 pipelining, 1 key-size
        should queue 2*2*2*2*1 = 16 tasks."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "repo1",
                "--specifier",
                "unstable",
                "--tests",
                "get,set",
                "--sizes",
                "512,1KB",
                "--io-threads",
                "1,9",
                "--pipelining",
                "1,4",
            ]
        )

        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 16

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_invalid_source_returns_exit_code_1(self, mock_queue_cls):
        """main() with an invalid source should return exit code 1."""
        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "invalid_source",
                "--tests",
                "get",
            ]
        )

        assert exit_code == 1
        mock_queue_cls.return_value.submit_task.assert_not_called()

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_empty_tests_returns_exit_code_1(self, mock_queue_cls):
        """main() with empty --tests should return exit code 1."""
        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "repo1",
                "--tests",
                "",
            ]
        )

        assert exit_code == 1
        mock_queue_cls.return_value.submit_task.assert_not_called()

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_repetitions_zero_returns_exit_code_1(self, mock_queue_cls):
        """main() with --repetitions 0 should return exit code 1."""
        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "repo1",
                "--tests",
                "get",
                "--repetitions",
                "0",
            ]
        )

        assert exit_code == 1
        mock_queue_cls.return_value.submit_task.assert_not_called()

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_key_sizes_produces_tasks_with_correct_values(self, mock_queue_cls):
        """main() with --key-sizes "0,64" should produce tasks with key_size 0 and 64."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "repo1",
                "--tests",
                "get",
                "--key-sizes",
                "0,64",
            ]
        )

        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 2

        submitted_key_sizes = sorted(call.args[0].key_size for call in mock_queue.submit_task.call_args_list)
        assert submitted_key_sizes == [0, 64]

    @patch("conductress.cli.TaskQueue")
    def test_queue_add_defaults(self, mock_queue_cls):
        """Defaults: source=valkey, specifier=unstable, sizes=512, io-threads=9, pipeline=10, reps=5."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            [
                "queue",
                "add",
                "--source",
                "repo1",
                "--tests",
                "set",
            ]
        )

        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 1
        task = mock_queue.submit_task.call_args[0][0]
        assert task.source == "repo1"
        assert task.specifier == "unstable"
        assert task.val_size == 512
        assert task.io_threads == 9
        assert task.pipelining == 10
        assert task.repetitions == 3
        assert task.make_args == ""


class TestQueueListSubcommand:
    """Unit tests for the 'queue list' subcommand via main()."""

    @patch("conductress.cli.TaskQueue")
    def test_queue_list_returns_exit_code_0(self, mock_queue_cls):
        """main(["queue", "list"]) should return exit code 0."""
        mock_queue = MagicMock()
        mock_queue.get_all_tasks.return_value = []
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "list"])
        assert exit_code == 0

    @patch("conductress.cli.TaskQueue")
    def test_queue_no_subcommand_defaults_to_list(self, mock_queue_cls):
        """main(["queue"]) with no subcommand should default to list."""
        mock_queue = MagicMock()
        mock_queue.get_all_tasks.return_value = []
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue"])
        assert exit_code == 0

    def test_no_subcommand_returns_exit_code_1(self):
        """main([]) with no subcommand should return exit code 1."""
        exit_code = main([])
        assert exit_code == 1


class TestQueueRemoveSubcommand:
    """Unit tests for the 'queue remove' subcommand."""

    @patch("conductress.cli.TaskQueue")
    def test_queue_remove_success(self, mock_queue_cls):
        mock_queue = MagicMock()
        mock_queue.remove_task.return_value = True
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "remove", "2026.05.17_10.08.33.876407"])
        assert exit_code == 0
        mock_queue.remove_task.assert_called_once_with("2026.05.17_10.08.33.876407")

    @patch("conductress.cli.TaskQueue")
    def test_queue_remove_not_found(self, mock_queue_cls):
        mock_queue = MagicMock()
        mock_queue.remove_task.return_value = False
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "remove", "nonexistent"])
        assert exit_code == 1


class TestQueueClearSubcommand:
    """Unit tests for the 'queue clear' subcommand."""

    @patch("conductress.cli.TaskQueue")
    def test_queue_clear_removes_all(self, mock_queue_cls):
        mock_queue = MagicMock()
        mock_task1 = MagicMock()
        mock_task1.task_id = "task1"
        mock_task2 = MagicMock()
        mock_task2.task_id = "task2"
        mock_queue.get_all_tasks.return_value = [mock_task1, mock_task2]
        mock_queue.remove_task.return_value = True
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "clear"])
        assert exit_code == 0
        assert mock_queue.remove_task.call_count == 2


class TestQueueAddMemorySubcommand:
    """Unit tests for the 'queue add-memory' subcommand."""

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_defaults(self, mock_queue_cls):
        """All 4 types queued without --expire gives 4 tasks."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "add-memory", "--source", "repo1", "--specifier", "abc123"])
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 4

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_with_expire(self, mock_queue_cls):
        """--expire adds expire variants (5 total for default types)."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "add-memory", "--source", "repo1", "--specifier", "abc123", "--expire"])
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 5

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_single_type(self, mock_queue_cls):
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "set"])
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 1

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_manually_uploaded(self, mock_queue_cls):
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(["queue", "add-memory", "--source", "manual", "--specifier", "/path/to/binary"])
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 4

    def test_add_memory_invalid_source(self):
        exit_code = main(["queue", "add-memory", "--source", "bogus", "--specifier", "x"])
        assert exit_code == 1

    def test_add_memory_invalid_type(self):
        exit_code = main(["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "badtype"])
        assert exit_code == 1

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_custom_sizes_task_count(self, mock_queue_cls):
        """--sizes queues one task per type per size."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            [
                "queue",
                "add-memory",
                "--source",
                "repo1",
                "--specifier",
                "x",
                "--types",
                "zadd,sadd",
                "--sizes",
                "8,20,64",
            ]
        )
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 6

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_custom_sizes_user_data_bytes(self, mock_queue_cls):
        """user_data_bytes is derived per type at each custom size (zadd adds score bytes)."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            ["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "zadd", "--sizes", "40"]
        )
        assert exit_code == 0
        task = mock_queue.submit_task.call_args[0][0]
        assert task.val_sizes == [40]
        assert task.user_data_bytes == 40 + config.MEM_TEST_SCORE_SIZE

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_custom_sizes_set_includes_key(self, mock_queue_cls):
        """set user data at a custom size includes the workload's key size."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            ["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "set", "--sizes", "100"]
        )
        assert exit_code == 0
        task = mock_queue.submit_task.call_args[0][0]
        assert task.user_data_bytes == task.key_size + 100

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_custom_sizes_with_expire(self, mock_queue_cls):
        """--sizes composes with --expire (expire variants also get each size)."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            [
                "queue",
                "add-memory",
                "--source",
                "repo1",
                "--specifier",
                "x",
                "--types",
                "set",
                "--sizes",
                "16,64",
                "--expire",
            ]
        )
        assert exit_code == 0
        assert mock_queue.submit_task.call_count == 4

    @patch("conductress.cli.TaskQueue")
    def test_add_memory_custom_sizes_human_units(self, mock_queue_cls):
        """--sizes accepts human-readable byte units like 1KB."""
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue

        exit_code = main(
            ["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "sadd", "--sizes", "1KB"]
        )
        assert exit_code == 0
        task = mock_queue.submit_task.call_args[0][0]
        assert task.val_sizes == [1024]
        assert task.user_data_bytes == 1024

    def test_add_memory_invalid_sizes(self):
        exit_code = main(
            ["queue", "add-memory", "--source", "repo1", "--specifier", "x", "--types", "zadd", "--sizes", "abc"]
        )
        assert exit_code == 1
