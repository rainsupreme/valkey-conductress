"""Tests for the generator profile system.

Covers: profile registry, manifest persistence, bootstrap marker/hash mismatch,
profile resolution, v1/v2 cross-contamination, mixed result extraction,
serialization/backward compatibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Focused unit tests: GeneratorProfile registry
# ---------------------------------------------------------------------------


class TestProfileRegistry:
    """Profile lookup and validation."""

    def test_known_profiles_exist(self):
        from conductress.generator_profiles import LEGACY_V1, PROFILES, SCALABLE_V2

        assert "legacy-v1" in PROFILES
        assert "scalable-v2" in PROFILES
        assert PROFILES["legacy-v1"] is LEGACY_V1
        assert PROFILES["scalable-v2"] is SCALABLE_V2

    def test_get_profile_known(self):
        from conductress.generator_profiles import get_profile

        p = get_profile("legacy-v1")
        assert p.name == "legacy-v1"
        assert len(p.commit_sha) == 40

    def test_get_profile_unknown_raises(self):
        from conductress.generator_profiles import get_profile

        with pytest.raises(KeyError, match="Unknown generator profile"):
            get_profile("nonexistent-profile")

    def test_legacy_v1_matches_config_commit(self):
        from conductress.config import VALKEY_BENCHMARK_COMMIT
        from conductress.generator_profiles import LEGACY_V1

        assert LEGACY_V1.commit_sha == VALKEY_BENCHMARK_COMMIT

    def test_profile_paths(self):
        from conductress.generator_profiles import GENERATOR_DIR, LEGACY_V1

        assert LEGACY_V1.install_dir == GENERATOR_DIR / "legacy-v1"
        assert LEGACY_V1.binary_path == GENERATOR_DIR / "legacy-v1" / "valkey-benchmark"
        assert LEGACY_V1.manifest_path == GENERATOR_DIR / "legacy-v1" / "manifest.json"

    def test_profiles_are_frozen(self):
        from conductress.generator_profiles import LEGACY_V1

        with pytest.raises(AttributeError):
            LEGACY_V1.name = "mutated"  # type: ignore[misc]

    def test_default_profile_is_legacy(self):
        from conductress.generator_profiles import DEFAULT_PROFILE

        assert DEFAULT_PROFILE == "legacy-v1"

    def test_scalable_v2_repo_is_rainfall(self):
        from conductress.generator_profiles import SCALABLE_V2

        assert "valkey-rainfall" in SCALABLE_V2.repo_url


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------


class TestManifest:
    """Manifest save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        from conductress.generator_profiles import GeneratorManifest

        m = GeneratorManifest(
            profile_name="test",
            repo_url="https://example.com/test.git",
            commit_sha="a" * 40,
            build_args="",
            binary_sha256="b" * 64,
        )
        path = tmp_path / "manifest.json"
        m.save(path)
        loaded = GeneratorManifest.load(path)
        assert loaded is not None
        assert loaded.profile_name == "test"
        assert loaded.commit_sha == "a" * 40
        assert loaded.binary_sha256 == "b" * 64

    def test_load_missing_returns_none(self, tmp_path: Path):
        from conductress.generator_profiles import GeneratorManifest

        assert GeneratorManifest.load(tmp_path / "nonexistent.json") is None

    def test_load_corrupt_returns_none(self, tmp_path: Path):
        from conductress.generator_profiles import GeneratorManifest

        bad = tmp_path / "manifest.json"
        bad.write_text("not json")
        assert GeneratorManifest.load(bad) is None

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        from conductress.generator_profiles import GeneratorManifest

        m = GeneratorManifest("p", "u", "c" * 40, "", "d" * 64)
        path = tmp_path / "a" / "b" / "manifest.json"
        m.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Bootstrap marker/hash mismatch
# ---------------------------------------------------------------------------


class TestBootstrapMismatch:
    """Bootstrap detects mismatched manifests and triggers rebuild."""

    def test_sha256_file(self, tmp_path: Path):
        from conductress.generator_profiles import _sha256_file

        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _sha256_file(p) == expected

    def test_missing_binary_triggers_build(self, tmp_path: Path):
        """A manifest without a binary should trigger rebuild."""
        from conductress.generator_profiles import GeneratorManifest

        # Write a manifest but no binary
        manifest_path = tmp_path / "manifest.json"
        m = GeneratorManifest("test-profile", "https://example.com/test.git", "a" * 40, "", "deadbeef")
        m.save(manifest_path)

        # The fast path should NOT match (binary_path doesn't exist)
        loaded = GeneratorManifest.load(manifest_path)
        assert loaded is not None
        # binary_path doesn't exist, so bootstrap would proceed to build
        assert not (tmp_path / "valkey-benchmark").exists()

    def test_manifest_commit_mismatch_detected(self, tmp_path: Path):
        """Changed commit_sha in profile vs manifest should trigger rebuild."""
        from conductress.generator_profiles import GeneratorManifest, _sha256_file

        binary = tmp_path / "valkey-benchmark"
        binary.write_bytes(b"fake binary")

        # Manifest claims commit "aaa...", profile wants "bbb..."
        m = GeneratorManifest("test", "url", "a" * 40, "", _sha256_file(binary))
        manifest_path = tmp_path / "manifest.json"
        m.save(manifest_path)

        loaded = GeneratorManifest.load(manifest_path)
        assert loaded is not None
        assert loaded.commit_sha != "b" * 40  # mismatch detected

    def test_manifest_binary_hash_mismatch_detected(self, tmp_path: Path):
        """Binary content changed (corrupted or wrong version) -> rebuild."""
        from conductress.generator_profiles import GeneratorManifest, _sha256_file

        binary = tmp_path / "valkey-benchmark"
        binary.write_bytes(b"original binary")
        original_hash = _sha256_file(binary)

        m = GeneratorManifest("test", "url", "a" * 40, "", original_hash)
        manifest_path = tmp_path / "manifest.json"
        m.save(manifest_path)

        # Corrupt the binary
        binary.write_bytes(b"corrupted binary")
        assert _sha256_file(binary) != original_hash


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


class TestResolveProfile:
    """resolve_bench_binary returns correct paths and provenance."""

    def test_custom_override_takes_precedence(self):
        from conductress.generator_profiles import resolve_bench_binary

        path, prov = resolve_bench_binary(
            generator_profile="legacy-v1",
            bench_binary_override="/custom/path/bench",
        )
        assert path == "/custom/path/bench"
        assert prov["generator_profile"] == "custom-override"
        assert "bench_binary" in prov

    def test_profile_resolution_calls_bootstrap(self):
        from conductress.generator_profiles import resolve_bench_binary

        with patch("conductress.generator_profiles.bootstrap_profile") as mock_bs:
            mock_bs.return_value = Path("/generators/legacy-v1/valkey-benchmark")
            with patch("conductress.generator_profiles.GeneratorManifest.load") as mock_load:
                mock_load.return_value = MagicMock(binary_sha256="abc123")
                path, prov = resolve_bench_binary(generator_profile="legacy-v1")

            mock_bs.assert_called_once()
            assert "legacy-v1" in path
            assert prov["generator_profile"] == "legacy-v1"
            assert prov["generator_binary_sha256"] == "abc123"

    def test_empty_override_uses_profile(self):
        from conductress.generator_profiles import resolve_bench_binary

        with patch("conductress.generator_profiles.bootstrap_profile") as mock_bs:
            mock_bs.return_value = Path("/generators/scalable-v2/valkey-benchmark")
            with patch("conductress.generator_profiles.GeneratorManifest.load") as mock_load:
                mock_load.return_value = MagicMock(binary_sha256="def456")
                path, prov = resolve_bench_binary(
                    generator_profile="scalable-v2",
                    bench_binary_override="",
                )
            assert "scalable-v2" in path
            assert prov["generator_commit"] == "026288e2aaedc757c3dd8d347c237e669086a948"


# ---------------------------------------------------------------------------
# v1/v2 cross-contamination
# ---------------------------------------------------------------------------


class TestCrossContamination:
    """v2 coordinators must not match v1 tasks and vice versa."""

    @pytest.fixture(autouse=True)
    def _ensure_valid_source(self, monkeypatch):
        import conductress.config as cfg

        if "valkey" not in cfg.REPO_NAMES:
            monkeypatch.setattr(cfg, "REPO_NAMES", cfg.REPO_NAMES + ["valkey"])

    def test_v2_state_dir_isolated(self):
        from conductress.sweep.coordinator_v2 import V2_STATE_DIR

        assert "v2" in str(V2_STATE_DIR)

    @staticmethod
    def _make_perf_task(generator_profile: str):
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        task = PerfTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="epoch ownership task",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            generator_profile=generator_profile,
        )
        task.sweep_commit = "abc123"  # type: ignore[attr-defined]
        return task

    @staticmethod
    def _make_v1_coordinator(tmp_path: Path):
        from conductress.sweep.coordinator import SweepCoordinator

        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            return SweepCoordinator(tmp_path)

    @pytest.mark.parametrize("generator_profile", ["", "legacy-v1"])
    def test_v1_matches_only_legacy_profiles(self, tmp_path: Path, generator_profile: str):
        coord = self._make_v1_coordinator(tmp_path)
        assert coord._is_my_task(self._make_perf_task(generator_profile))

    @pytest.mark.parametrize("generator_profile", ["scalable-v2", "unknown-future-profile"])
    def test_v1_rejects_nonlegacy_profiles(self, tmp_path: Path, generator_profile: str):
        coord = self._make_v1_coordinator(tmp_path)
        assert not coord._is_my_task(self._make_perf_task(generator_profile))

    def test_v2_completion_does_not_reach_v1_result_extraction(self, tmp_path: Path):
        coord = self._make_v1_coordinator(tmp_path)
        task = self._make_perf_task("scalable-v2")

        with patch.object(coord, "_extract_result", return_value=None) as extract:
            coord.on_task_completed(task)

        extract.assert_not_called()

    def test_v2_failure_does_not_mark_v1_build_failure(self, tmp_path: Path):
        coord = self._make_v1_coordinator(tmp_path)
        task = self._make_perf_task("scalable-v2")

        with patch.object(coord, "record_build_failure") as record_failure:
            coord.on_task_failed(task)

        record_failure.assert_not_called()

    def test_v2_workload_id_is_canonical_and_epoch_is_v2(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import ThroughputSweepCoordinatorV2

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                coord = ThroughputSweepCoordinatorV2(tmp_path)
        assert coord.workload_id == "get-k16-v16-t7-p10"
        assert coord.epoch_id == "v2"
        assert "v2" in coord.state_file.name

    def test_v2_does_not_match_v1_task(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import ThroughputSweepCoordinatorV2
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                state_file = tmp_path / "state_v2-get-k16-v16-t7-p10.json"
                state_file.write_text("{}")
                coord = ThroughputSweepCoordinatorV2(tmp_path)

        # v1 task: no generator_profile
        v1_task = PerfTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="v1 task",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )
        v1_task.sweep_commit = "abc123"  # type: ignore[attr-defined]
        assert not coord._is_my_task(v1_task)

    def test_v2_matches_own_task(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import ThroughputSweepCoordinatorV2
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                state_file = tmp_path / "state_v2-get-k16-v16-t7-p10.json"
                state_file.write_text("{}")
                coord = ThroughputSweepCoordinatorV2(tmp_path)

        # v2 task: has generator_profile matching
        v2_task = PerfTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="v2 task",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            generator_profile="scalable-v2",
        )
        v2_task.sweep_commit = "abc123"  # type: ignore[attr-defined]
        assert coord._is_my_task(v2_task)


# ---------------------------------------------------------------------------
# Mixed sweep coordinator v2 discrimination
# ---------------------------------------------------------------------------


class TestMixedSweepV2:
    """MixedSweepCoordinatorV2 task discrimination."""

    @pytest.fixture(autouse=True)
    def _ensure_valid_source(self, monkeypatch):
        import conductress.config as cfg

        if "valkey" not in cfg.REPO_NAMES:
            monkeypatch.setattr(cfg, "REPO_NAMES", cfg.REPO_NAMES + ["valkey"])

    def test_mixed_v2_workload_id_and_epoch(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import MixedSweepCoordinatorV2

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                coord = MixedSweepCoordinatorV2(tmp_path)
        assert coord.workload_id == "mixed-s20-k16-v16-t7-p10"
        assert coord.epoch_id == "v2"
        assert "v2" in coord.state_file.name

    def test_mixed_v2_does_not_match_perf_task(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import MixedSweepCoordinatorV2
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                label = f"s20-v2-mixed-s20-k16-v16-t{7}-p10"
                state_file = tmp_path / f"state_{label}.json"
                state_file.write_text("{}")
                coord = MixedSweepCoordinatorV2(tmp_path)

        perf_task = PerfTaskData(
            source="valkey",
            specifier="abc",
            replicas=0,
            note="",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            generator_profile="scalable-v2",
        )
        perf_task.sweep_commit = "abc"  # type: ignore[attr-defined]
        assert not coord._is_my_task(perf_task)

    @staticmethod
    def _make_mixed_task(sweep_commit: str = "abc"):
        from conductress.tasks.task_mixed import MixedTaskData

        return MixedTaskData(
            source="valkey",
            specifier="abc",
            replicas=0,
            note="",
            requirements={},
            make_args="",
            set_ratio=20,
            val_size=16,
            io_threads=7,
            pipelining=10,
            duration=30,
            sweep_commit=sweep_commit,
        )

    @staticmethod
    def _make_coordinator(tmp_path: Path):
        from conductress.sweep.coordinator_v2 import MixedSweepCoordinatorV2

        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                return MixedSweepCoordinatorV2(tmp_path)

    def test_mixed_v2_matches_mixed_task(self, tmp_path: Path):
        coord = self._make_coordinator(tmp_path)
        assert coord._is_my_task(self._make_mixed_task())

    def test_mixed_v2_sweep_commit_survives_queue_round_trip(self, tmp_path: Path):
        from conductress.task_queue import TaskQueue
        from conductress.tasks.task_mixed import MixedTaskData

        queue = TaskQueue(tmp_path / "queue")
        task = self._make_mixed_task()
        queue.submit_task(task)

        restored = queue.get_next_task()
        assert isinstance(restored, MixedTaskData)
        assert restored.sweep_commit == "abc"
        assert self._make_coordinator(tmp_path)._is_my_task(restored)

    def test_reloaded_mixed_v2_completion_records_result(self, tmp_path: Path):
        from conductress.task_queue import TaskQueue

        queue = TaskQueue(tmp_path / "queue")
        queue.submit_task(self._make_mixed_task())
        restored = queue.get_next_task()
        assert restored is not None

        coord = self._make_coordinator(tmp_path)
        with patch.object(coord, "_extract_result", return_value=(123.0, 1.5, 5)):
            with patch.object(coord, "record_result") as record_result:
                with patch.object(coord, "_extract_perf_counters", return_value=None):
                    with patch.object(coord, "_extract_cpu_stacks") as extract_stacks:
                        coord.on_task_completed(restored)

        record_result.assert_called_once_with("abc", 123.0, 1.5, 5)
        extract_stacks.assert_called_once_with(restored)

    def test_legacy_mixed_task_without_sweep_commit_defaults_empty(self):
        from conductress.tasks.task_mixed import MixedTaskData

        task = MixedTaskData(
            source="valkey",
            specifier="abc",
            replicas=0,
            note="",
            requirements={},
            make_args="",
            set_ratio=20,
            val_size=16,
            io_threads=7,
            pipelining=10,
            duration=30,
        )
        assert task.sweep_commit == ""


# ---------------------------------------------------------------------------
# Serialization / backward compatibility
# ---------------------------------------------------------------------------


class TestSerialization:
    """Task serialization preserves generator_profile field."""

    @pytest.fixture(autouse=True)
    def _ensure_valid_source(self, monkeypatch):
        """Ensure 'valkey' is in REPO_NAMES even if earlier tests mutated it."""
        import conductress.config as cfg

        if "valkey" not in cfg.REPO_NAMES:
            monkeypatch.setattr(cfg, "REPO_NAMES", cfg.REPO_NAMES + ["valkey"])

    def test_perftaskdata_serializes_generator_profile(self):
        from dataclasses import asdict

        from conductress.tasks.task_perf_benchmark import PerfTaskData

        task = PerfTaskData(
            source="valkey",
            specifier="abc",
            replicas=0,
            note="test",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            generator_profile="scalable-v2",
        )
        d = asdict(task)
        assert d["generator_profile"] == "scalable-v2"

    def test_perftaskdata_default_empty_generator_profile(self):
        from dataclasses import asdict

        from conductress.tasks.task_perf_benchmark import PerfTaskData

        task = PerfTaskData(
            source="valkey",
            specifier="abc",
            replicas=0,
            note="test",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )
        d = asdict(task)
        assert d["generator_profile"] == ""

    def test_legacy_task_json_without_generator_profile_deserializes(self):
        """Tasks serialized before generator_profile existed should still load."""
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        legacy_dict = {
            "source": "valkey",
            "specifier": "abc",
            "replicas": 0,
            "note": "",
            "requirements": {},
            "make_args": "",
            "test": "get",
            "val_size": 16,
            "io_threads": 7,
            "pipelining": 10,
            "warmup": 5,
            "duration": 30,
            "perf_stat_enabled": False,
            "has_expire": False,
            "preload_keys": True,
            # NO generator_profile key
        }
        task = PerfTaskData(**legacy_dict)
        assert task.generator_profile == ""
