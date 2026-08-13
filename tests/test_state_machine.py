"""Tests for the data pipeline FSM's phase-label consumption guard.

The phase-count guard distinguishes models that consume phase labels
(phase_head cross-entropy / privileged oracle routing / centroid bootstrap)
from label-free rows (BC pilot, scratch/warm-start MoE). Degenerate labels
must fail loudly for the former and only warn for the latter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from phaseforge.data.ingestion.state_machine import (
    DataPipelineStateMachine,
    PipelineError,
)

PHASE_CONSUMING_MODELS = {
    "phaseforge",
    "teacher_forced",
    "oracle_moe",
    "plain_encoder_phase_bootstrap",
}
LABEL_FREE_MODELS = {
    "bc",
    "scratch_moe",
    "warmstart_moe",
    "phase_pretrain_random_router",
}


def _fsm_for_model(model: str) -> DataPipelineStateMachine:
    data_cfg = OmegaConf.load("phaseforge/config/data/common.yaml")
    path = (
        "phaseforge/config/models/phaseforge.yaml"
        if model == "phaseforge"
        else f"phaseforge/config/models/baselines/{model}.yaml"
    )
    return DataPipelineStateMachine(
        DictConfig({"models": OmegaConf.load(path), "data": data_cfg})
    )


def test_phase_consuming_models_are_detected() -> None:
    for name in PHASE_CONSUMING_MODELS:
        assert _fsm_for_model(name)._model_uses_phase_labels(), name


def test_label_free_models_are_not_phase_consuming() -> None:
    for name in LABEL_FREE_MODELS:
        assert not _fsm_for_model(name)._model_uses_phase_labels(), name


class _FakeIngester:
    """Minimal ingester returning degenerate (all-zero) phase labels."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    def ingest(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        traj: dict[str, Any] = {
            "state": np.zeros((10, 19), dtype=np.float32),
            "action": np.zeros((10, 7), dtype=np.float32),
            "phase": np.zeros(10, dtype=np.int64),
            "task_id": 0,
        }
        return [traj], {"lift": 0}


def _fsm_with_fake_ingester(model: str) -> DataPipelineStateMachine:
    fsm = _fsm_for_model(model)
    fsm.data_cfg.ingester = DictConfig(
        {"_target_": _FakeIngester}, flags={"allow_objects": True}
    )
    fsm._raw_dir = Path("fake-raw-dir")
    return fsm


def test_degenerate_phases_fail_loud_for_phase_consuming_models() -> None:
    fsm = _fsm_with_fake_ingester("phaseforge")
    with pytest.raises(PipelineError, match="no samples for phase"):
        fsm._ingest_source()


def test_degenerate_phases_only_warn_for_label_free_models(caplog) -> None:
    fsm = _fsm_with_fake_ingester("bc")
    with caplog.at_level(
        logging.WARNING, logger="phaseforge.data.ingestion.state_machine"
    ):
        fsm._ingest_source()
    assert "no samples for phase" in caplog.text
    assert fsm._trajectories