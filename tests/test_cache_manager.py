"""Cache identity + provenance tests (review blocker 2).

- compute_hash folds in cheap provenance context (object-index content,
  raw dataset names/sizes) so a stale cache can never be silently reused.
- save() persists a full audit manifest: per-file SHA-256, git commit,
  schema, split task names — and writes human-readable split files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _data_cfg(suite_dir: Path | None = None, index_path: str | None = None):
    cfg = {
        "state_dim": 151,
        "action_dim": 7,
        "state_keys": [{"key": "robot0_joint_pos", "dim": 7}],
        "object_state": {"enabled": True, "k_slots": 16, "dim_per_object": 7},
        "libero": {"phase_labeler": {"num_phases": 6}},
    }
    if index_path is not None:
        cfg["object_state"]["index_path"] = index_path
    if suite_dir is not None:
        cfg["libero"]["suite"] = suite_dir.name
    return OmegaConf.create(cfg)


def _write_dummy_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy-hdf5-content")


@pytest.fixture
def fake_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PHASEFORGE_DATA_DIR at a tmp root so suite paths are local."""
    monkeypatch.setenv("PHASEFORGE_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Cache identity (compute_hash)
# ---------------------------------------------------------------------------


def test_hash_changes_when_object_index_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = tmp_path / "object_index.json"
    idx.write_text(json.dumps({"version": 1, "k_slots": 16, "tasks": {}}))
    cfg = _data_cfg(index_path=str(idx))

    from phaseforge.data.ingestion.cache_manager import CacheManager

    h1 = CacheManager.compute_hash(cfg)
    idx.write_text(json.dumps({"version": 1, "k_slots": 8, "tasks": {}}))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "object-index change must invalidate the cache"


def test_hash_changes_when_raw_dataset_changes(
    tmp_path: Path, fake_data_root: Path
) -> None:
    suite_dir = fake_data_root / "raw" / "libero" / "libero_90"
    _write_dummy_hdf5(suite_dir / "TASK_A_demo.hdf5")
    cfg = _data_cfg(suite_dir=suite_dir)

    from phaseforge.data.ingestion.cache_manager import CacheManager

    h1 = CacheManager.compute_hash(cfg)
    _write_dummy_hdf5(suite_dir / "TASK_B_demo.hdf5")
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "raw dataset change must invalidate the cache"


def test_hash_portable_across_machine_mtimes_when_manifest_present(
    fake_data_root: Path,
) -> None:
    """With MANIFEST.json present the identity is the pinned dataset commit
    — different download mtimes must NOT change the cache key (this is what
    makes the HF-uploaded cache usable on other machines)."""
    import os
    import time

    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import libero_manifest_path

    suite_dir = fake_data_root / "raw" / "libero" / "libero_90"
    demo = suite_dir / "TASK_A_demo.hdf5"
    _write_dummy_hdf5(demo)
    cfg = _data_cfg(suite_dir=suite_dir)
    manifest = libero_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"source": "huggingface", "commit_sha": "f13aa24a" * 5},
            indent=2,
        )
    )

    h1 = CacheManager.compute_hash(cfg)
    # Simulate a re-download on another machine: same content, new mtime.
    os.utime(demo, (time.time() - 86400, time.time() - 86400))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 == h2, "mtimes must not enter the key when the manifest exists"


def test_hash_sensitive_to_manifest_commit(fake_data_root: Path) -> None:
    """A different pinned dataset revision must invalidate the cache."""
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import libero_manifest_path

    suite_dir = fake_data_root / "raw" / "libero" / "libero_90"
    _write_dummy_hdf5(suite_dir / "TASK_A_demo.hdf5")
    cfg = _data_cfg(suite_dir=suite_dir)
    manifest = libero_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest.write_text(json.dumps({"commit_sha": "a" * 40}, indent=2))
    h1 = CacheManager.compute_hash(cfg)
    manifest.write_text(json.dumps({"commit_sha": "b" * 40}, indent=2))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "dataset revision change must invalidate the cache"


def test_hash_sensitive_to_mtime_without_manifest(
    fake_data_root: Path,
) -> None:
    """Without MANIFEST.json the legacy mtime fingerprint still applies."""
    import os
    import time

    from phaseforge.data.ingestion.cache_manager import CacheManager

    suite_dir = fake_data_root / "raw" / "libero" / "libero_90"
    demo = suite_dir / "TASK_A_demo.hdf5"
    _write_dummy_hdf5(demo)
    cfg = _data_cfg(suite_dir=suite_dir)

    h1 = CacheManager.compute_hash(cfg)
    os.utime(demo, (time.time() - 86400, time.time() - 86400))
    h2 = CacheManager.compute_hash(cfg)

    assert h1 != h2, "no-manifest fallback must stay mtime-sensitive"


def test_hash_stable_without_data(tmp_path: Path) -> None:
    cfg = _data_cfg(index_path=str(tmp_path / "missing_index.json"))
    from phaseforge.data.ingestion.cache_manager import CacheManager

    assert CacheManager.compute_hash(cfg) == CacheManager.compute_hash(cfg)


# ---------------------------------------------------------------------------
# Manifest provenance (save)
# ---------------------------------------------------------------------------


def test_save_manifest_records_full_provenance(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import CacheManager

    cfg = _data_cfg()
    mgr = CacheManager(tmp_path)
    config_hash = CacheManager.compute_hash(cfg)

    provenance = {
        "dataset_repo": "yifengzhu-hf/LIBERO-datasets",
        "dataset_commit": None,
        "code_git_commit": "deadbeef",
        "raw_files": [
            {"name": "TASK_A_demo.hdf5", "size": 123, "sha256": "a" * 64}
        ],
        "object_index_sha256": "b" * 64,
        "state_schema": {
            "keys": [{"key": "robot0_joint_pos", "dim": 7}],
            "state_dim": 151,
            "action_dim": 7,
        },
        "split_task_names": {
            "train": ["TASK_A_demo"],
            "val": ["TASK_B_demo"],
            "eval": [],
        },
    }
    mgr.save(
        config_hash=config_hash,
        trajectories=[],
        norm_stats={},
        splits={"train": [], "val": [], "eval": []},
        task_index={"TASK_A_demo": 0, "TASK_B_demo": 1},
        provenance=provenance,
    )

    cache_dir = mgr.cache_dir(config_hash)
    assert mgr.cache_exists(config_hash)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["provenance"]["raw_files"][0]["sha256"] == "a" * 64
    assert manifest["provenance"]["code_git_commit"] == "deadbeef"

    train_tasks = (cache_dir / "train_tasks.txt").read_text().splitlines()
    val_tasks = (cache_dir / "validation_tasks.txt").read_text().splitlines()
    assert train_tasks == ["TASK_A_demo"]
    assert val_tasks == ["TASK_B_demo"]


def test_save_without_provenance_still_works(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import CacheManager

    mgr = CacheManager(tmp_path)
    config_hash = CacheManager.compute_hash(_data_cfg())
    mgr.save(
        config_hash=config_hash,
        trajectories=[],
        norm_stats={},
        splits={"train": [], "val": [], "eval": []},
    )
    assert mgr.cache_exists(config_hash)


def test_git_commit_helper_returns_sha(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import git_commit

    commit = git_commit()
    # Inside a git repo this is a 40-hex SHA; elsewhere it degrades to "".
    assert commit == "" or len(commit) == 40


def test_sha256_file_streams_large_content(tmp_path: Path) -> None:
    from phaseforge.data.ingestion.cache_manager import sha256_file

    data = b"phaseforge-provenance" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected
