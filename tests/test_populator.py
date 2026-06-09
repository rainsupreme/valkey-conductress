"""Tests for the memory workload populator module."""

from unittest.mock import MagicMock, call, patch

import pytest

from conductress.sweep.memory_coordinator import MemoryWorkload
from conductress.sweep.populator import BATCH_SIZE, _pad_to_size, populate


class TestPadToSize:
    """Test the padding helper."""

    def test_exact_size(self):
        result = _pad_to_size("key", 0, 16)
        assert len(result) == 16

    def test_larger_size(self):
        result = _pad_to_size("m", 42, 64)
        assert len(result) == 64
        assert result.startswith("m:0000042:")

    def test_small_target_raises(self):
        with pytest.raises(ValueError, match="too small for unique items"):
            _pad_to_size("key", 0, 5)

    def test_various_sizes(self):
        for size in [16, 20, 64, 128, 256]:
            result = _pad_to_size("x", 999999, size)
            assert len(result) == size, f"Failed for size={size}"


class TestPopulateSet:
    """Test SET workload population."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_populates_correct_count(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(command="set", key_size=16, value_size=64, label="set-v64", user_data_bytes=80)
        # Use small count for test
        small_workload = MemoryWorkload(
            command="set", key_size=16, value_size=64, label="set-v64", user_data_bytes=80, item_count=100
        )
        populate("127.0.0.1", 6379, small_workload)

        assert mock_pipe.set.call_count == 100
        mock_client.close.assert_called_once()

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_key_and_value_sizes(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="set", key_size=16, value_size=64, label="set-v64", user_data_bytes=80, item_count=10
        )
        populate("127.0.0.1", 6379, workload)

        # Check first call's key and value sizes
        first_call = mock_pipe.set.call_args_list[0]
        key, value = first_call[0]
        assert len(key) == 16
        assert len(value) == 64


class TestPopulateZadd:
    """Test ZADD workload population."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_populates_single_sorted_set(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="zadd", key_size=0, value_size=64, label="zadd-m64", user_data_bytes=72, item_count=100
        )
        populate("127.0.0.1", 6379, workload)

        assert mock_pipe.zadd.call_count == 100
        # All calls should use "myzset" as the key
        for c in mock_pipe.zadd.call_args_list:
            assert c[0][0] == "myzset"

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_member_size_is_exact(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="zadd", key_size=0, value_size=64, label="zadd-m64", user_data_bytes=72, item_count=5
        )
        populate("127.0.0.1", 6379, workload)

        # Check member size in the mapping dict
        first_call = mock_pipe.zadd.call_args_list[0]
        mapping = first_call[0][1]  # second positional arg is the mapping
        member = list(mapping.keys())[0]
        assert len(member) == 64

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_scores_are_random_integers(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="zadd", key_size=0, value_size=64, label="zadd-m64", user_data_bytes=72, item_count=10
        )
        populate("127.0.0.1", 6379, workload)

        scores = []
        for c in mock_pipe.zadd.call_args_list:
            mapping = c[0][1]
            scores.append(list(mapping.values())[0])

        # Scores should be integers
        assert all(isinstance(s, int) for s in scores)
        # Scores should not be sequential (random)
        assert scores != sorted(scores)

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_scores_are_reproducible(self, mock_valkey_cls):
        """Seeded RNG should produce same scores every time."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="zadd", key_size=0, value_size=64, label="zadd-m64", user_data_bytes=72, item_count=10
        )
        populate("127.0.0.1", 6379, workload)
        scores_1 = [list(c[0][1].values())[0] for c in mock_pipe.zadd.call_args_list]

        mock_pipe.reset_mock()
        populate("127.0.0.1", 6379, workload)
        scores_2 = [list(c[0][1].values())[0] for c in mock_pipe.zadd.call_args_list]

        assert scores_1 == scores_2


class TestPopulateSadd:
    """Test SADD workload population."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_populates_single_set(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="sadd", key_size=0, value_size=64, label="sadd-m64", user_data_bytes=64, item_count=50
        )
        populate("127.0.0.1", 6379, workload)

        assert mock_pipe.sadd.call_count == 50
        for c in mock_pipe.sadd.call_args_list:
            assert c[0][0] == "myset"
            assert len(c[0][1]) == 64


class TestPopulateHset:
    """Test HSET workload population."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_populates_single_hash(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="hset",
            key_size=0,
            value_size=64,
            field_size=64,
            label="hset-f64-v64",
            user_data_bytes=128,
            item_count=50,
        )
        populate("127.0.0.1", 6379, workload)

        assert mock_pipe.hset.call_count == 50
        for c in mock_pipe.hset.call_args_list:
            assert c[0][0] == "myhash"
            field = c[0][1]
            value = c[0][2]
            assert len(field) == 64
            assert len(value) == 64


class TestPopulateExpire:
    """Test EXPIRE application."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_expire_applied_for_set(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="set",
            key_size=16,
            value_size=64,
            has_expire=True,
            label="set-v64-expire",
            user_data_bytes=80,
            item_count=10,
        )
        populate("127.0.0.1", 6379, workload)

        # Should have both SET and EXPIRE calls
        assert mock_pipe.set.call_count == 10
        assert mock_pipe.expire.call_count == 10

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_expire_skipped_for_non_set(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="zadd",
            key_size=0,
            value_size=64,
            has_expire=True,
            label="zadd-m64-expire",
            user_data_bytes=72,
            item_count=10,
        )
        populate("127.0.0.1", 6379, workload)

        assert mock_pipe.zadd.call_count == 10
        assert mock_pipe.expire.call_count == 0


class TestPipelineBatching:
    """Test that pipeline is flushed in batches."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_flushes_at_batch_size(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        item_count = BATCH_SIZE * 2 + 100  # 2 full batches + remainder
        workload = MemoryWorkload(
            command="sadd", key_size=0, value_size=20, label="sadd-m20", user_data_bytes=20, item_count=item_count
        )
        populate("127.0.0.1", 6379, workload)

        # 2 full batch flushes + 1 remainder flush = 3 execute calls
        assert mock_pipe.execute.call_count == 3


class TestUnsupportedCommand:
    """Test error handling for unsupported commands."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_raises_for_unknown_command(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_valkey_cls.return_value = mock_client

        workload = MemoryWorkload(
            command="lpush", key_size=0, value_size=64, label="lpush-v64", user_data_bytes=64, item_count=10
        )
        with pytest.raises(ValueError, match="Unsupported command: lpush"):
            populate("127.0.0.1", 6379, workload)
