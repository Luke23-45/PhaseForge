"""StateOnlyDataset: flat (state, action, phase, task_id) dataset."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class StateOnlyDataset(Dataset):
    """Dataset of pre-normalized (state, action, phase, task_id) tuples.

    All trajectories are already normalized and converted to tensors
    before this class is constructed.

    Args:
        trajectories: List of trajectory dicts, each containing:
            - ``"state"``:   Tensor (T, state_dim)
            - ``"action"``:  Tensor (T, action_dim)
            - ``"phase"``:   Tensor (T,) int64
            - ``"task_id"``: int
        sequence_length: Number of consecutive timesteps per sample.
            If 1, each timestep is a single sample (squeezes time dim).
        stride: Step between consecutive samples within a trajectory.
    """

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        sequence_length: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.stride = stride
        self._index_map = self._build_index_map()

    def _build_index_map(self) -> list[tuple[int, int]]:
        """Build (traj_idx, start_t) index pairs for all valid windows."""
        index_map = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = traj["state"].shape[0]
            for start_t in range(0, T - self.sequence_length + 1, self.stride):
                index_map.append((traj_idx, start_t))
        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        traj_idx, start_t = self._index_map[idx]
        traj = self.trajectories[traj_idx]
        end_t = start_t + self.sequence_length

        state = traj["state"][start_t:end_t]    # (seq_len, S)
        action = traj["action"][start_t:end_t]   # (seq_len, A)
        phase = traj["phase"][start_t:end_t]     # (seq_len,)
        task_id = torch.tensor(traj["task_id"], dtype=torch.long)

        if self.sequence_length == 1:
            # Squeeze the time dimension for single-step training
            return {
                "state": state.squeeze(0),   # (S,)
                "action": action.squeeze(0), # (A,)
                "phase": phase.squeeze(0),   # scalar
                "task_id": task_id,
            }
        return {
            "state": state,
            "action": action,
            "phase": phase,
            "task_id": task_id,
        }
