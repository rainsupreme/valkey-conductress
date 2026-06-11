"""Tests for the heap_profiler module."""

from unittest.mock import AsyncMock, patch

import pytest

from conductress.heap_profiler import (
    CATEGORIES,
    CATEGORY_NAMES,
    _categorize_stack,
    _is_jemalloc_frame,
    _parse_stacks,
    _resolve_addresses,
    categorize_heap_dump,
    cleanup_heap_dumps,
    collect_heap_profile,
    recategorize_from_stacks,
)

# =============================================================================
# Unit tests for pure functions
# =============================================================================


class TestIsJemallocFrame:
    """Verify jemalloc/allocator frames are correctly identified for skipping."""

    def test_jemalloc_internals_are_skipped(self):
        jemalloc_funcs = [
            "je_malloc_default",
            "prof_backtrace_impl",
            "imalloc_body",
            "ztrymalloc_usable_internal",
            "zcalloc",
            "valkey_malloc",
        ]
        for func in jemalloc_funcs:
            assert _is_jemalloc_frame(func) is True, f"{func} should be jemalloc"

    def test_application_frames_are_not_skipped(self):
        app_funcs = [
            "createEmbeddedStringObject",
            "dictInsertAtPosition",
            "setCommand",
            "aeMain",
            "sdsnewlen",
        ]
        for func in app_funcs:
            assert _is_jemalloc_frame(func) is False, f"{func} should NOT be jemalloc"

    def test_generic_skip_funcs(self):
        for func in ("??", "_start", "main", "__libc_start_call_main"):
            assert _is_jemalloc_frame(func) is True


class TestCategorizeStack:
    """Verify stacks are assigned to the correct category based on function patterns."""

    @pytest.mark.parametrize(
        "stack,expected",
        [
            (
                ["je_malloc_default", "ztrymalloc_usable_internal", "createEmbeddedStringObject", "setKey"],
                "robj_embval",
            ),
            (["zcalloc", "hashtableExpandIfNeeded", "hashtableAdd", "dbAddInternal"], "hashtable"),
            (["zmalloc", "sdsnewlen", "sdsdup", "processMultibulkBuffer"], "sds"),
            (["zmalloc", "zslCreateNode", "zslInsert", "zsetAdd"], "skiplist"),
            (["zmalloc", "dictExpand", "dictAdd", "dbAdd"], "dict"),
            (["zcalloc", "aeCreateEventLoop", "initServer", "main"], "server_infra"),
            (["zmalloc", "createObject", "createStringObject", "setCommand"], "robj"),
            (
                ["zmalloc", "createUnembeddedObjectWithKeyAndExpire", "objectSetKeyAndExpire", "dbAdd"],
                "robj_embkey",
            ),
            (["zmalloc", "hashTypeCreateEntry", "hashTypeSet", "hsetCommand"], "hash_entry"),
            # Post-PR#3366: entryConstruct called via hashTypeSet without hashtableInsertAtPosition
            (
                ["ztrymalloc_usable_internal", "entryConstruct", "entryCreate", "hashTypeSet", "hsetCommand"],
                "hash_entry",
            ),
            (["je_malloc_default", "imalloc", "??", "_start"], "other"),
        ],
    )
    def test_category_assignment(self, stack, expected):
        assert _categorize_stack(stack) == expected

    def test_specificity_order(self):
        """Earlier categories in CATEGORIES list take priority."""
        # skiplist appears before dict, so zslCreateNode wins even if dictAdd is also present
        stack = ["zmalloc", "zslCreateNode", "dictAdd"]
        assert _categorize_stack(stack) == "skiplist"


class TestParseStacks:
    """Verify heap file parsing extracts correct (addresses, bytes) tuples."""

    def test_parses_multiple_stacks(self):
        heap = (
            "heap_v2/1\n"
            "  t*: 100: 6400 [0: 0]\n"
            "  t0: 100: 6400 [0: 0]\n"
            "@ 0xA 0xB 0xC\n"
            "  t*: 50: 3200 [0: 0]\n"
            "  t0: 50: 3200 [0: 0]\n"
            "@ 0xA 0xB 0xD\n"
            "  t*: 25: 1600 [0: 0]\n"
            "  t0: 25: 1600 [0: 0]\n"
        )
        stacks = _parse_stacks(heap)
        assert len(stacks) == 2  # first t* has no preceding @
        assert stacks[0] == (["0xA", "0xB", "0xC"], 3200)
        assert stacks[1] == (["0xA", "0xB", "0xD"], 1600)

    def test_empty_content(self):
        assert _parse_stacks("") == []

    def test_header_only(self):
        assert _parse_stacks("heap_v2/1\n") == []


class TestResolveAddresses:
    """Verify addr2line output is correctly mapped to addresses."""

    def test_basic_resolution(self):
        addrs = ["0x1000", "0x2000", "0x3000"]
        output = "funcA\nfile1:10\nfuncB\nfile2:20\nfuncC\nfile3:30\n"
        resolved = _resolve_addresses(addrs, output)
        assert resolved == {"0x1000": "funcA", "0x2000": "funcB", "0x3000": "funcC"}

    def test_partial_output(self):
        """Handles addr2line returning fewer results than addresses."""
        addrs = ["0x1000", "0x2000", "0x3000"]
        output = "funcA\nfile1:10\nfuncB\nfile2:20\n"
        resolved = _resolve_addresses(addrs, output)
        assert resolved == {"0x1000": "funcA", "0x2000": "funcB"}
        assert "0x3000" not in resolved


class TestCategorizeHeapDump:
    """Integration test for the full parse+categorize pipeline."""

    def test_categorizes_correctly(self):
        heap = (
            "heap_v2/1\n"
            "@ 0x1 0x2 0x3\n"
            "  t*: 100: 5000 [0: 0]\n"
            "  t0: 100: 5000 [0: 0]\n"
            "@ 0x1 0x2 0x4\n"
            "  t*: 50: 2000 [0: 0]\n"
            "  t0: 50: 2000 [0: 0]\n"
        )
        # 4 unique addresses: 0x1, 0x2, 0x3, 0x4
        addr2line = (
            "je_malloc_default\n/je.c:1\n"
            "ztrymalloc_usable_internal\n/zmalloc.c:1\n"
            "createEmbeddedStringObject\n/object.c:1\n"
            "hashtableAdd\n/hashtable.c:1\n"
        )
        result = categorize_heap_dump(heap, addr2line)
        assert result is not None
        assert result["robj_embval"] == 5000
        assert result["hashtable"] == 2000
        assert result["sds"] == 0

    def test_returns_none_for_empty(self):
        assert categorize_heap_dump("", "") is None


# =============================================================================
# Async tests for SSH-based collection
# =============================================================================


@pytest.mark.asyncio
class TestCollectHeapProfile:
    """Test the async collect_heap_profile with mocked SSH."""

    HEAP_CONTENT = (
        "heap_v2/1\n"
        "@ 0x1000 0x2000 0x3000\n"
        "  t*: 1000: 48000 [0: 0]\n"
        "  t0: 1000: 48000 [0: 0]\n"
        "@ 0x1000 0x2000 0x4000\n"
        "  t*: 500: 12000 [0: 0]\n"
        "  t0: 500: 12000 [0: 0]\n"
    )

    ADDR2LINE_OUTPUT = (
        "je_malloc_default\n/je.c:1\n"
        "ztrymalloc_usable_internal\n/zmalloc.c:1\n"
        "createEmbeddedStringObject\n/object.c:169\n"
        "hashtableAdd\n/hashtable.c:1374\n"
    )

    async def test_successful_collection(self):
        ssh = AsyncMock()
        ssh.run_host_command = AsyncMock(
            side_effect=[
                ("/tmp/valkey-heap.123.heap", ""),  # ls -t
                (self.HEAP_CONTENT, ""),  # cat
                (self.ADDR2LINE_OUTPUT, ""),  # addr2line
            ]
        )

        result = await collect_heap_profile(ssh, "/path/to/valkey-server", num_keys=1000)

        assert result is not None
        assert result.breakdown["robj_embval"] == 48.0  # 48000 / 1000
        assert result.breakdown["hashtable"] == 12.0  # 12000 / 1000
        assert result.raw_stacks is not None
        assert len(result.raw_stacks) > 0

    async def test_no_heap_dump_returns_none(self):
        ssh = AsyncMock()
        ssh.run_host_command = AsyncMock(return_value=("", ""))

        result = await collect_heap_profile(ssh, "/path/to/binary", num_keys=1000)
        assert result is None

    async def test_invalid_heap_content_returns_none(self):
        ssh = AsyncMock()
        ssh.run_host_command = AsyncMock(
            side_effect=[
                ("/tmp/valkey-heap.123.heap", ""),
                ("not a heap file", ""),
            ]
        )

        result = await collect_heap_profile(ssh, "/path/to/binary", num_keys=1000)
        assert result is None

    async def test_zero_keys_returns_none(self):
        ssh = AsyncMock()
        ssh.run_host_command = AsyncMock(
            side_effect=[
                ("/tmp/valkey-heap.123.heap", ""),
                (self.HEAP_CONTENT, ""),
                (self.ADDR2LINE_OUTPUT, ""),
            ]
        )

        result = await collect_heap_profile(ssh, "/path/to/binary", num_keys=0)
        assert result is None


@pytest.mark.asyncio
class TestCleanupHeapDumps:
    async def test_calls_rm(self):
        ssh = AsyncMock()
        ssh.run_host_command = AsyncMock(return_value=("", ""))
        await cleanup_heap_dumps(ssh)
        ssh.run_host_command.assert_called_once_with("rm -f /tmp/valkey-heap*.heap", check=False)


# =============================================================================
# Category metadata tests
# =============================================================================


class TestCategoryNames:
    def test_has_all_expected_categories(self):
        expected = {
            "robj_embval",
            "robj_embkey",
            "sds",
            "hashtable",
            "hash_entry",
            "skiplist",
            "robj",
            "listpack",
            "dict",
            "server_infra",
            "other",
        }
        assert set(CATEGORY_NAMES) == expected

    def test_stacking_order(self):
        """server_infra at bottom (first), skiplist at top (last)."""
        assert CATEGORY_NAMES[0] == "server_infra"
        assert CATEGORY_NAMES[-1] == "skiplist"

    def test_count(self):
        assert len(CATEGORY_NAMES) == 11

    def test_categories_and_names_in_sync(self):
        """Every category in CATEGORIES must appear in CATEGORY_NAMES and vice versa."""
        pattern_cats = {name for name, _ in CATEGORIES}
        # robj_embkey is synthesized from robj + marker check, not in CATEGORIES directly
        name_cats = set(CATEGORY_NAMES) - {"other", "robj_embkey"}
        assert pattern_cats == name_cats, (
            f"CATEGORIES and CATEGORY_NAMES out of sync.\n"
            f"  In CATEGORIES but not CATEGORY_NAMES: {pattern_cats - name_cats}\n"
            f"  In CATEGORY_NAMES but not CATEGORIES: {name_cats - pattern_cats}"
        )

    def test_no_duplicate_category_names(self):
        """No duplicates in CATEGORY_NAMES."""
        assert len(CATEGORY_NAMES) == len(set(CATEGORY_NAMES))

    def test_hash_entry_before_hashtable(self):
        """hash_entry must precede hashtable (entryConstruct calls hashtableInsertAtPosition)."""
        cat_order = [name for name, _ in CATEGORIES]
        assert cat_order.index("hash_entry") < cat_order.index("hashtable")

    def test_robj_embval_before_robj(self):
        """robj_embval must precede robj (createEmbeddedString contains createString)."""
        cat_order = [name for name, _ in CATEGORIES]
        assert cat_order.index("robj_embval") < cat_order.index("robj")

    def test_sds_before_hashtable_and_dict(self):
        """sds must precede hashtable/dict (SDS allocs appear in dict/hashtable stacks)."""
        cat_order = [name for name, _ in CATEGORIES]
        assert cat_order.index("sds") < cat_order.index("hashtable")
        assert cat_order.index("sds") < cat_order.index("dict")


class TestRealStackCategorization:
    """Verify categorization of real stacks captured from jemalloc heap profiles.

    These are actual stacks from ARM Graviton 3 (armbench) across multiple workloads
    and Valkey eras (dict-based, kvstore, new hashtable). They serve as regression tests
    ensuring categorization changes don't silently break existing correct classifications.
    """

    @pytest.mark.parametrize(
        "stack,expected,description",
        [
            # === SET workload — dict era (commit idx ~163, pre-kvstore) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "dictInsertAtPosition",
                    "dbAddInternal",
                    "dbAdd",
                    "setGenericCommand",
                    "exitExecutionUnit",
                    "processCommand",
                    "processCommandAndResetClient",
                    "readQueryFromClient",
                ],
                "dict",
                "SET dict-era: per-key dict entry allocation via dictInsertAtPosition",
                id="set-dict-era-entry",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "_dictExpand",
                    "_dictExpand",
                    "dictAddRaw",
                    "dbAddInternal",
                    "dbAdd",
                    "setGenericCommand",
                ],
                "dict",
                "SET dict-era: hash table expansion via _dictExpand",
                id="set-dict-era-expand",
            ),
            # === SET workload — kvstore era (commit idx ~645) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "_dictResize",
                    "dictExpandIfNeeded",
                    "_dictExpandIfNeeded",
                    "dictAddRaw",
                    "kvstoreDictAddRaw",
                    "dbAddInternal",
                    "dbAdd",
                    "setGenericCommand",
                ],
                "dict",
                "SET kvstore-era: dict resize within kvstore layer",
                id="set-kvstore-era-resize",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createEmbeddedEntry",
                    "kvstoreDictAddRaw",
                    "dbAddInternal",
                    "dbAdd",
                    "setGenericCommand",
                ],
                "dict",
                "SET kvstore-era: dict entry with embedded key SDS (not new hashtable)",
                id="set-kvstore-era-embedded-entry",
            ),
            # === SET workload — new hashtable era (commit idx ~2010) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "bucketConvertToChained",
                    "insert",
                    "dbAddInternal",
                    "setKey",
                    "setGenericCommand",
                    "setCommand",
                ],
                "hashtable",
                "SET hashtable-era: bucket chain conversion during insert",
                id="set-hashtable-era-bucket",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "resize",
                    "rehashStepOnWriteIfNeeded",
                    "dbAddInternal",
                    "setKey",
                    "setGenericCommand",
                ],
                "hashtable",
                "SET hashtable-era: table resize during rehash step",
                id="set-hashtable-era-resize",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createEmbeddedStringObjectWithKeyAndExpire",
                    "createStringObjectWithKeyAndExpire",
                    "dbAddInternal",
                    "setKey",
                    "setGenericCommand",
                    "setCommand",
                ],
                "robj_embval",
                "SET hashtable-era: embedded string object (value + key + expiry in one alloc)",
                id="set-hashtable-era-embval",
            ),
            # === SET workload — robj_embkey era (commit idx ~1093) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createObjectWithKeyAndExpire",
                    "objectSetKeyAndExpire",
                    "dbAddInternal",
                    "setKey",
                    "setGenericCommand",
                ],
                "robj_embkey",
                "SET embkey-era: robj with embedded key (objectSetKeyAndExpire marker)",
                id="set-robj-embkey",
            ),
            # === SET-EXPIRE workload — hashtable era (commit idx ~2010) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "bucketConvertToChained",
                    "rehashEntry",
                    "rehashStepOnReadIfNeeded",
                    "hashtableAddOrFind",
                    "setExpire",
                    "expireGenericCommand",
                ],
                "hashtable",
                "SET-EXPIRE: expire hashtable bucket conversion",
                id="set-expire-hashtable-bucket",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createEmbeddedStringObjectWithKeyAndExpire",
                    "createStringObjectWithKeyAndExpire",
                    "objectSetExpire",
                    "expireGenericCommand",
                ],
                "robj_embval",
                "SET-EXPIRE: embedded string for expire entry",
                id="set-expire-embval",
            ),
            # === HSET workload — dict era (commit idx ~163) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "dictInsertAtPosition",
                    "hashTypeSet",
                    "hsetCommand",
                    "exitExecutionUnit",
                    "processCommand",
                ],
                "dict",
                "HSET dict-era: per-field dict entry via dictInsertAtPosition",
                id="hset-dict-era-entry",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "_dictExpand",
                    "_dictExpand",
                    "dictAddRaw",
                    "hashTypeSet",
                    "hsetCommand",
                ],
                "dict",
                "HSET dict-era: hash field table expansion",
                id="hset-dict-era-expand",
            ),
            # === HSET workload — hashtable era (commit idx ~1093, hashTypeCreateEntry) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "hashTypeCreateEntry",
                    "hashtableInsertAtPosition",
                    "hsetCommand",
                    "call",
                    "processCommand",
                ],
                "hash_entry",
                "HSET hashtable-era: per-field entry via hashTypeCreateEntry (the key fix)",
                id="hset-hashtable-era-hashTypeCreateEntry",
            ),
            # === HSET workload — entryWrite era (commit idx ~1425) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "entryWrite",
                    "hashtableInsertAtPosition",
                    "hsetCommand",
                    "exitExecutionUnit",
                    "processCommand",
                ],
                "hash_entry",
                "HSET entryWrite-era: entry allocation via entryWrite",
                id="hset-entryWrite-era",
            ),
            # === HSET workload — newest (post-PR#3366, entryConstruct) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "entryConstruct",
                    "entryCreate",
                    "hashTypeSet",
                    "hsetCommand",
                    "call",
                    "processCommand",
                ],
                "hash_entry",
                "HSET newest: entry via entryConstruct (no hashtableInsertAtPosition)",
                id="hset-newest-entryConstruct",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "resize",
                    "rehashStepOnWriteIfNeeded",
                    "hashTypeSet",
                    "hsetCommand",
                ],
                "hashtable",
                "HSET newest: hash field table resize (infrastructure, not entries)",
                id="hset-newest-resize",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "bucketConvertToChained",
                    "hashtableFindPositionForInsert",
                    "hashTypeSet",
                    "hsetCommand",
                ],
                "hashtable",
                "HSET newest: bucket chain conversion (infrastructure)",
                id="hset-newest-bucket",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "_sdsnewlen.constprop.0",
                    "hashTypeIgnoreTTL",
                    "hsetCommand",
                    "call",
                    "processCommand",
                ],
                "sds",
                "HSET newest: SDS allocation for field/value strings",
                id="hset-newest-sds",
            ),
            # === ZADD workload ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "zslCreateNode",
                    "zsetAdd",
                    "zaddGenericCommand",
                    "call",
                    "processCommand",
                ],
                "skiplist",
                "ZADD: skiplist node allocation",
                id="zadd-skiplist-node",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "zslInsert",
                    "zsetAdd",
                    "zaddGenericCommand",
                    "exitExecutionUnit",
                    "processCommand",
                ],
                "skiplist",
                "ZADD dict-era: skiplist insert (level array allocation)",
                id="zadd-skiplist-insert",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "resize",
                    "rehashStepOnWriteIfNeeded",
                    "zsetAdd",
                    "zaddGenericCommand",
                ],
                "hashtable",
                "ZADD hashtable-era: member hashtable resize",
                id="zadd-hashtable-resize",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "bucketConvertToChained",
                    "highBits",
                    "zsetAdd",
                    "zaddGenericCommand",
                ],
                "hashtable",
                "ZADD hashtable-era: bucket conversion",
                id="zadd-hashtable-bucket",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "_sdsnewlen.constprop.0",
                    "zsetAdd",
                    "zaddGenericCommand",
                    "exitExecutionUnit",
                    "processCommand",
                ],
                "sds",
                "ZADD: SDS allocation for member string",
                id="zadd-sds-member",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "_dictExpand",
                    "_dictExpand",
                    "dictAddRaw",
                    "zsetAdd",
                    "zaddGenericCommand",
                ],
                "dict",
                "ZADD dict-era: member dict expansion",
                id="zadd-dict-expand",
            ),
            # === SADD workload ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "resize",
                    "rehashStepOnWriteIfNeeded",
                    "setTypeAddAux",
                    "saddCommand",
                ],
                "hashtable",
                "SADD hashtable-era: set hashtable resize",
                id="sadd-hashtable-resize",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "bucketConvertToChained",
                    "hashtableFindPositionForInsert",
                    "setTypeAddAux",
                    "saddCommand",
                ],
                "hashtable",
                "SADD hashtable-era: bucket conversion",
                id="sadd-hashtable-bucket",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "_sdsnewlen.constprop.0",
                    "setTypeAddAux",
                    "saddCommand",
                    "call",
                    "processCommand",
                ],
                "sds",
                "SADD: SDS for member string",
                id="sadd-sds-member",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createEntryNoValue",
                    "setTypeAddAux",
                    "saddCommand",
                    "exitExecutionUnit",
                    "processCommand",
                ],
                "dict",
                "SADD dict-era: no-value dict entry for set members",
                id="sadd-dict-era-entry",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrycalloc_usable_internal",
                    "_dictExpand",
                    "_dictExpand",
                    "setTypeAddAux",
                    "saddCommand",
                ],
                "dict",
                "SADD dict-era: set dict expansion",
                id="sadd-dict-era-expand",
            ),
            # === SADD — legacy dict rehash (was incorrectly categorized as hashtable) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "rehashEntriesInBucketAtIndex",
                    "dictRehash",
                    "_dictRehashStep",
                    "setTypeAddAux",
                    "saddCommand",
                ],
                "dict",
                "SADD kvstore-era: legacy dict rehash entry migration (not new hashtable)",
                id="sadd-dict-rehash-migration",
            ),
            # === Input buffer allocations (all workloads) ===
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "createObject",
                    "processMultibulkBuffer",
                    "parseCommand",
                    "readQueryFromClient",
                ],
                "robj",
                "Input parsing: robj for command argument",
                id="input-robj",
            ),
            pytest.param(
                [
                    "prof_backtrace_impl",
                    "tsd_post_reentrancy_raw",
                    "je_prof_tctx_create",
                    "prof_alloc_prep",
                    "ztrymalloc_usable_internal",
                    "_sdsnewlen.constprop.0",
                    "sdsnewlen",
                    "processMultibulkBuffer",
                    "parseCommand",
                    "readQueryFromClient",
                ],
                "sds",
                "Input parsing: SDS for command argument string",
                id="input-sds",
            ),
        ],
    )
    def test_real_stack(self, stack, expected, description):
        """Categorize a real stack from jemalloc heap profile."""
        assert _categorize_stack(stack) == expected, f"Failed: {description}"


class TestRecategorizeFromStacks:
    """Verify re-categorization from saved stacks produces same results."""

    def test_recategorize_matches_original(self):
        """Recategorizing saved stacks should produce the same breakdown."""
        raw_stacks = [
            [["je_malloc", "createEmbeddedStringObject", "setCommand"], 48000],
            [["je_malloc", "hashtableExpand", "dbAdd"], 12000],
            [["je_malloc", "sdsnewlen", "catAppendOnlyGenericCommand"], 5000],
        ]
        result = recategorize_from_stacks(raw_stacks, num_keys=1000)
        assert result["robj_embval"] == 48.0
        assert result["hashtable"] == 12.0
        assert result["sds"] == 5.0

    def test_recategorize_empty_stacks(self):
        """Empty stacks should return all-zero breakdown."""
        result = recategorize_from_stacks([], num_keys=1000)
        assert all(v == 0.0 for v in result.values())
        assert "other" in result
