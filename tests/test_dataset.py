"""CPU-only tests for StateOnlyDataset sequence-length handling."""

from __future__ import annotations

import pytest
import torch

from phaseforge.data.common.dataset import StateOnlyDataset


def _traj(T: int = 4, state_dim: int = 3, action_dim: int = 2) -> dict:
    return {
        "state": torch.randn(T, state_dim),
        "action": torch.randn(T, action_dim),
        "phase": torch.randint(0, 2, (T,)),
        "task_id": 0,
    }


def test_sequence_length_one_yields_flat_samples() -> None:
    dataset = StateOnlyDataset([_traj(T=4)], sequence_length=1)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["state"].ndim == 1
    assert sample["action"].ndim == 1
    assert sample["phase"].ndim == 0


def test_sequence_length_greater_than_one_rejected() -> None:
    with pytest.raises(ValueError, match="sequence_length=3"):
        StateOnlyDataset([_traj()], sequence_length=3)
