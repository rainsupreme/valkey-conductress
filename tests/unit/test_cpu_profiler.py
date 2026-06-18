"""Tests for CPU profile categorization."""

from conductress.cpu_profiler import CPU_CATEGORIES_IO, CPU_CATEGORIES_MAIN, categorize_cpu_stacks


class TestCategorizeCpuStacks:
    """Test categorize_cpu_stacks with real-world stack patterns."""

    def test_main_thread_categorization(self):
        """Test categorization with realistic main thread stacks from ARM prototype."""
        stacks = [
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;hashtableIncrementalFindStep", 5000],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;processInputBuffer", 3000],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;processCommand;call", 2000],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;addReplyBulk", 1500],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;zmalloc", 800],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;memcpy", 500],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;commandlogPushEntryIfNeeded", 300],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;lookupKey", 400],
            ["valkey-server;main;aeMain;beforeSleep;processClientsCommandsBatch;resetClient", 200],
            ["valkey-server;main;aeMain;epoll_wait", 100],
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)

        # Check key categories are present
        assert "hash_lookup" in result
        assert "command_parse" in result
        assert "command_overhead" in result
        assert "reply_build" in result
        assert "memory_alloc" in result
        assert "string_ops" in result
        assert "commandlog" in result
        assert "key_access" in result
        assert "cleanup" in result

        # Verify percentages sum to ~100
        total = sum(result.values())
        assert abs(total - 100.0) < 0.01

        # Verify dominant categories
        assert result["hash_lookup"] > 30  # 5000/13800 = 36%
        assert result["command_parse"] > 20  # 3000/13800 = 22%

    def test_io_thread_categorization(self):
        """Test IO thread categorization with mutex spin patterns."""
        stacks = [
            ["io_thd_1;IOThreadMain;pthread_mutex_lock;__aarch64_cas4_acq", 10000],
            ["io_thd_1;IOThreadMain;writeToClient;connSocketWrite", 3000],
            ["io_thd_1;IOThreadMain;zmalloc", 1000],
            ["io_thd_1;IOThreadMain;spmcDequeue", 500],
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_IO)

        assert result["synchronization"] > 60  # mutex dominates
        assert result["networking_io"] > 15
        assert result["memory_alloc"] > 5
        total = sum(result.values())
        assert abs(total - 100.0) < 0.01

    def test_empty_stacks(self):
        """Empty input returns 100% other."""
        result = categorize_cpu_stacks([], CPU_CATEGORIES_MAIN)
        assert result == {"other": 100.0}

    def test_all_unmatched(self):
        """Stacks with no matching patterns go to other."""
        stacks = [
            ["some_unknown_function", 100],
            ["another;deeply;nested;mystery_func", 200],
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)
        assert result["other"] == 100.0

    def test_leaf_extraction(self):
        """Verify leaf function is correctly extracted from deep stacks."""
        stacks = [
            ["a;b;c;d;e;f;hashtableIncrementalFindStep", 100],
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)
        assert "hash_lookup" in result
        assert result["hash_lookup"] == 100.0

    def test_first_match_wins(self):
        """Categories are checked in order, first match wins."""
        # "sds" matches string_ops but also appears in function names with other patterns
        stacks = [
            ["valkey-server;sdslen", 100],  # matches string_ops (sds pattern)
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)
        assert "string_ops" in result

    def test_string_count_handling(self):
        """Counts passed as strings are handled correctly."""
        stacks = [
            ["some;stack;hashtableFind", "500"],
            ["other;stack;zmalloc", "300"],
        ]
        result = categorize_cpu_stacks(stacks, CPU_CATEGORIES_MAIN)
        assert abs(result["hash_lookup"] - 62.5) < 0.1
        assert abs(result["memory_alloc"] - 37.5) < 0.1
