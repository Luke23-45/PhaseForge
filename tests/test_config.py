"""CPU-only tests for config utilities (checkpoint discovery, hashing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig

from phaseforge.utils.config import (
    config_hash,
    find_latest_checkpoint,
    scan_checkpoints,
    write_run_meta,
)


def _make_run(base: Path, model: str, stage: int, name: str, seed: int | None) -> None:
    run_dir = base / model / f"stage{stage}" / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "checkpoint_best.pt").write_text("dummy")
    meta = {"seed": seed}
    if seed is not None:
        (run_dir / "run_meta.json").write_text(json.dumps(meta))


def test_find_latest_checkpoint_prefers_requested_seed(tmp_path: Path) -> None:
    # Newest run is seed 43; seed 42 runs are older. Requesting seed 42
    # must pick the newest seed-42 run, not the globally newest run.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-02_10-00-00_aaaa0002", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43)

    ckpt = find_latest_checkpoint("bc", stage=1, base=tmp_path, resolve_alias=False, seed=42)
    assert ckpt is not None
    assert "aaaa0002" in str(ckpt)


def test_find_latest_checkpoint_ignores_seed_when_none_requested(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    _make_run(tmp_path, "bc", 1, "2026-08-03_10-00-00_aaaa0003", seed=43)

    ckpt = find_latest_checkpoint("bc", stage=1, base=tmp_path, resolve_alias=False)
    assert ckpt is not None
    assert "aaaa0003" in str(ckpt)


def test_find_latest_checkpoint_legacy_run_fallback(tmp_path: Path) -> None:
    # No run_meta.json (legacy run): seed lookup must fall back to the
    # newest run instead of returning None.
    run_dir = tmp_path / "bc" / "stage1" / "2026-08-01_10-00-00_aaaa0001"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "checkpoint_best.pt").write_text("dummy")

    ckpt = find_latest_checkpoint("bc", stage=1, base=tmp_path, resolve_alias=False, seed=42)
    assert ckpt is not None
    assert "aaaa0001" in str(ckpt)


def test_find_latest_checkpoint_require_seed_fails_hard(tmp_path: Path) -> None:
    # Only a seed-42 run exists; requesting seed 43 with require_seed=True
    # must raise instead of silently falling back to a different seed (the
    # multi-seed protocol never mixes seeds between stages).
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)

    with pytest.raises(FileNotFoundError, match="seed 43"):
        find_latest_checkpoint(
            "bc", stage=1, base=tmp_path, resolve_alias=False,
            seed=43, require_seed=True,
        )


def test_scan_checkpoints_reports_seed(tmp_path: Path) -> None:
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)
    infos = scan_checkpoints("bc", stage=1, base=tmp_path)
    assert len(infos) == 1
    assert infos[0].seed == 42


def test_resolve_alias_looks_in_source_model_dir(tmp_path: Path) -> None:
    # warmstart_moe shares BC's Stage 1 checkpoint.
    _make_run(tmp_path, "bc", 1, "2026-08-01_10-00-00_aaaa0001", seed=42)

    ckpt = find_latest_checkpoint(
        "warmstart_moe", stage=1, base=tmp_path, resolve_alias=True, seed=42
    )
    assert ckpt is not None
    assert "aaaa0001" in str(ckpt)
    assert (
        find_latest_checkpoint(
            "warmstart_moe", stage=1, base=tmp_path, resolve_alias=False
        )
        is None
    )


def test_config_hash_is_deterministic() -> None:
    cfg1 = DictConfig({"a": 1, "b": {"c": [1, 2, 3]}})
    cfg2 = DictConfig({"a": 1, "b": {"c": [1, 2, 3]}})
    assert config_hash(cfg1) == config_hash(cfg2)
    assert len(config_hash(cfg1)) == 16

    cfg3 = DictConfig({"a": 1, "b": {"c": [1, 2, 4]}})
    assert config_hash(cfg1) != config_hash(cfg3)


def _meta_cfg(stage: int) -> DictConfig:
    return DictConfig(
        {
            "models": {"name": "phaseforge", "_target_": "phaseforge.models.moe"},
            "train": {"stage": stage},
            "project": {"seed": 42, "device": "cuda", "tag": None},
        }
    )


def test_write_run_meta_records_explicit_stage(tmp_path: Path) -> None:
    # Eval runs: the stage restored from the checkpoint wins over cfg.train.stage.
    write_run_meta(tmp_path, _meta_cfg(1), stage=2)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["stage"] == 2
    assert meta["model_name"] == "phaseforge"
    assert meta["seed"] == 42


def test_write_run_meta_defaults_to_train_stage(tmp_path: Path) -> None:
    write_run_meta(tmp_path, _meta_cfg(2))
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["stage"] == 2
