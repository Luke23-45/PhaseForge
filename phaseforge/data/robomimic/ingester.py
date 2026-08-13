"""Strict HDF5 ingester for robomimic low-dimensional datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_ALIASES: dict[str, tuple[str, ...]] = {
    "robot0_eef_pos": ("robot0_eef_pos", "eef_pos"),
    "robot0_eef_quat": ("robot0_eef_quat", "eef_quat"),
    "robot0_gripper_qpos": ("robot0_gripper_qpos", "gripper_qpos"),
    "robot0_joint_pos": ("robot0_joint_pos", "joint_pos"),
    "robot0_joint_vel": ("robot0_joint_vel", "joint_vel"),
    "object": ("object", "object-state"),
    "goal": ("goal",),
}


class RobomimicHDF5Ingester:
    """Convert one task directory of low-dimensional HDF5 files.

    Each file is expected to contain ``data/demo_*/obs/*`` and
    ``data/demo_*/actions``. Images are never opened because only the
    configured low-dimensional keys are read.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        state_keys: list[Any],
        action_key: str = "actions",
        action_dim: int = 7,
        phase_labeler: Any = None,
        task_names: list[str] | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.action_key = str(action_key)
        self.action_dim = int(action_dim)
        self.phase_labeler = phase_labeler
        self.task_names = set(task_names or [])
        self.state_specs = [
            (str(entry["key"]), int(entry["dim"]))
            for entry in state_keys
        ]
        if not self.state_specs:
            raise ValueError("state_keys must contain at least one low-dimensional key")
        if phase_labeler is None:
            raise ValueError("phase_labeler is required for PhaseForge training")

    def ingest(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        files = sorted(self.raw_dir.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in {self.raw_dir}")
        selected = [f for f in files if not self.task_names or f.stem in self.task_names]
        if not selected:
            raise ValueError(
                f"task_names={sorted(self.task_names)} matched no files in {self.raw_dir}"
            )
        task_index = {path.stem: i for i, path in enumerate(selected)}
        trajectories: list[dict[str, Any]] = []
        for path in selected:
            trajectories.extend(self._read_task(path, task_index[path.stem]))
        return trajectories, task_index

    def _read_task(self, path: Path, task_id: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        with h5py.File(path, "r") as h5:
            data = h5.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError(f"{path} has no HDF5 data group")
            demo_keys = sorted(k for k in data.keys() if k.startswith("demo_"))
            if not demo_keys:
                raise ValueError(f"{path} contains no demo_* trajectories")
            for demo_key in demo_keys:
                demo = data[demo_key]
                obs = demo.get("obs")
                if not isinstance(obs, h5py.Group):
                    raise ValueError(f"{path}:{demo_key} has no obs group")

                arrays = [
                    self._read_obs(obs, key, dim, path, demo_key)
                    for key, dim in self.state_specs
                ]
                lengths = {arr.shape[0] for arr in arrays}
                if len(lengths) != 1 or next(iter(lengths)) == 0:
                    raise ValueError(f"{path}:{demo_key} has inconsistent or empty observations")
                state = np.concatenate(arrays, axis=-1).astype(np.float32)

                if self.action_key not in demo:
                    raise ValueError(f"{path}:{demo_key} missing {self.action_key!r}")
                action = np.asarray(demo[self.action_key][:], dtype=np.float32)
                if action.ndim != 2 or action.shape[0] != state.shape[0]:
                    raise ValueError(f"{path}:{demo_key} action/state length mismatch")
                if action.shape[1] != self.action_dim:
                    raise ValueError(
                        f"{path}:{demo_key} action_dim={action.shape[1]} != {self.action_dim}"
                    )
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"{path}:{demo_key} contains non-finite values")

                traj: dict[str, Any] = {
                    "state": state,
                    "action": action,
                    "task_id": task_id,
                    "task_name": path.stem,
                    "source_file": str(path),
                    "demo_key": demo_key,
                }
                traj["phase"] = np.asarray(self.phase_labeler.label(traj), dtype=np.int64)
                result.append(traj)
        logger.info("%s: ingested %d trajectories", path.name, len(result))
        return result

    @staticmethod
    def _read_obs(
        obs: h5py.Group,
        key: str,
        expected_dim: int,
        path: Path,
        demo_key: str,
    ) -> np.ndarray:
        candidates = _ALIASES.get(key, (key,))
        dataset = next((obs[name] for name in candidates if name in obs), None)
        if dataset is None:
            raise ValueError(f"{path}:{demo_key} missing low-dimensional observation {key!r}")
        value = np.asarray(dataset[:], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != expected_dim:
            raise ValueError(
                f"{path}:{demo_key} observation {key!r} has shape {value.shape}; "
                f"expected (T, {expected_dim})"
            )
        return value
