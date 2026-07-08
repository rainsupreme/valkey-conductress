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


class TestOperations:
    """Test the pure _operations generator across modes and command types."""

    def _replay(self, ops):
        """Replay op tuples into a live-item set, returning (live_set, n_add, n_del)."""
        live: set = set()
        n_add = n_del = 0
        for op in ops:
            item = op[1]
            if op[0] == "add":
                live.add(item)
                n_add += 1
            else:
                live.discard(item)
                n_del += 1
        return live, n_add, n_del

    def _ops(self, command, count, mode, value_size=20, key_size=16, field_size=16):
        from conductress.sweep.populator import _operations

        return list(_operations(command, count, key_size, value_size, field_size, mode))

    def test_zadd_sequential_ascending_scores_exact_count(self):
        ops = self._ops("zadd", 100, "sequential")
        assert len(ops) == 100 and all(op[0] == "add" for op in ops)
        assert [op[2] for op in ops] == list(range(100))  # dense ascending scores
        live, _, n_del = self._replay(ops)
        assert len(live) == 100 and n_del == 0

    def test_zadd_random_exact_count_random_scores(self):
        from conductress.sweep.populator import MAX_SCORE

        ops = self._ops("zadd", 100, "random")
        assert len(ops) == 100 and all(op[0] == "add" for op in ops)
        scores = [op[2] for op in ops]
        assert all(isinstance(s, int) and 0 <= s <= MAX_SCORE for s in scores)
        assert scores != sorted(scores)  # scattered, not monotonic
        live, _, n_del = self._replay(ops)
        assert len(live) == 100 and n_del == 0  # unique members -> exact count

    def test_deterministic(self):
        assert self._ops("zadd", 50, "random") == self._ops("zadd", 50, "random")
        assert self._ops("sadd", 50, "churn") == self._ops("sadd", 50, "churn")

    @pytest.mark.parametrize("command", ["set", "sadd", "hset", "zadd"])
    def test_all_modes_supported_for_all_types(self, command):
        # No mode is rejected for any type (uniform support, no guard).
        for mode in ("random", "sequential", "churn"):
            ops = self._ops(command, 100, mode)
            assert len(ops) >= 100  # churn adds extra ops
            assert all(op[0] in ("add", "del") for op in ops)

    @pytest.mark.parametrize("command", ["set", "sadd", "hset", "zadd"])
    def test_churn_exercises_deletes_for_all_types(self, command):
        count = 200
        ops = self._ops(command, count, "churn")
        live, n_add, n_del = self._replay(ops)
        assert n_del > 0, f"churn should issue deletes for {command}"
        assert n_add > count, f"churn should add beyond the initial fill for {command}"
        assert 0 < len(live) < 2 * count
        assert abs(len(live) - count) <= count // 2

    def test_churn_never_deletes_absent_item(self):
        present: set = set()
        for op in self._ops("zadd", 200, "churn"):
            if op[0] == "add":
                present.add(op[1])
            else:
                assert op[1] in present, "churn deleted an item that was not live"
                present.discard(op[1])

    def test_sequential_and_random_equivalent_membership_for_hashtable_types(self):
        # For set/sadd/hset, order doesn't change the resident item set (memory is
        # order-independent); both modes produce the same 100 unique items.
        seq_live, _, _ = self._replay(self._ops("sadd", 100, "sequential"))
        rnd_live, _, _ = self._replay(self._ops("sadd", 100, "random"))
        assert seq_live == rnd_live and len(seq_live) == 100

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown populate_mode"):
            self._ops("zadd", 10, "bogus")


class TestChurnAllTypes:
    """Churn (and every mode) is supported uniformly for all types — no guard."""

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_sadd_churn_runs_and_issues_srem(self, mock_valkey_cls):
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_valkey_cls.return_value = mock_client

        wl = MemoryWorkload(command="sadd", key_size=0, value_size=20, populate_mode="churn", item_count=100)
        populate("localhost", 6379, wl)  # must NOT raise
        assert mock_pipe.sadd.call_count > 100  # initial fill + churn adds
        assert mock_pipe.srem.call_count > 0  # churn deletes

    @patch("conductress.sweep.populator.valkey.Valkey")
    def test_churn_with_expire_rejected(self, mock_valkey_cls):
        # churn + expire is genuinely incompatible (churn changes which keys exist).
        mock_valkey_cls.return_value = MagicMock()
        wl = MemoryWorkload(command="set", key_size=16, value_size=64, has_expire=True, populate_mode="churn")
        with pytest.raises(ValueError, match="incompatible with has_expire"):
            populate("localhost", 6379, wl)
