"""Vision stripper: extract state-only data from LIBERO HDF5 files.

This is the ONLY module in the codebase that opens image keys in HDF5.
After this module runs, no downstream code ever encounters image arrays.
Images are never loaded into RAM — we skip them entirely at the HDF5 level.

Schema agnosticism
------------------
The LIBERO dataset has been released with different HDF5 key naming
conventions across versions. This module auto-detects the convention
used by each file and resolves the configured state keys accordingly.

Known schemas:

  "robosuite"  — keys match the raw robosuite observation dict:
      obs/robot0_joint_pos, obs/robot0_joint_vel, obs/robot0_eef_pos,
      obs/robot0_eef_quat, obs/robot0_gripper_qpos

  "flattened"  — keys use flattened naming common in HF releases:
      obs/joint_states, obs/ee_pos, obs/ee_ori, obs/gripper_states
      Joint velocity is derived via finite differences.
      End-effector quaternion is extracted from demo/robot_states[:, 5:9].

Detection is per-file, so a mixed corpus works without errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)

# Keys that are definitively vision data — never read into memory
_VISION_KEY_PATTERNS = ("rgb", "depth", "image", "pixel", "video", "wrist", "eye")


def _is_vision_key(key: str) -> bool:
    k = key.lower()
    return any(pat in k for pat in _VISION_KEY_PATTERNS)


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

_ROBOSUITE_SENTINEL = "robot0_joint_pos"
_FLATTENED_SENTINEL = "joint_states"


def _detect_obs_schema(obs_group: h5py.Group) -> str:
    """Return ``"robosuite"`` or ``"flattened"`` based on available keys.

    Raises ``ValueError`` if neither schema is recognised.
    """
    if _ROBOSUITE_SENTINEL in obs_group:
        return "robosuite"
    elif _FLATTENED_SENTINEL in obs_group:
        return "flattened"
    else:
        available = sorted(obs_group.keys())
        raise ValueError(
            f"Cannot detect HDF5 obs schema: expected either "
            f"'{_ROBOSUITE_SENTINEL}' (robosuite) or "
            f"'{_FLATTENED_SENTINEL}' (flattened) in obs/. "
            f"Available keys: {available}"
        )


# ---------------------------------------------------------------------------
# Schema-specific key resolvers
# ---------------------------------------------------------------------------


def _resolve_robosuite(
    obs_group: h5py.Group,
    demo_group: h5py.Group,
    key: str,
) -> np.ndarray | None:
    """Resolve ``key`` under the robosuite naming convention.

    All keys are read directly from ``obs_group``.
    """
    if key == "robot0_joint_vel":
        return obs_group["robot0_joint_vel"][:].astype(np.float32)
    if key == "robot0_joint_pos":
        return obs_group["robot0_joint_pos"][:].astype(np.float32)
    if key == "robot0_eef_pos":
        return obs_group["robot0_eef_pos"][:].astype(np.float32)
    if key == "robot0_eef_quat":
        return obs_group["robot0_eef_quat"][:].astype(np.float32)
    if key == "robot0_gripper_qpos":
        return obs_group["robot0_gripper_qpos"][:].astype(np.float32)
    return None


def _resolve_flattened(
    obs_group: h5py.Group,
    demo_group: h5py.Group,
    key: str,
) -> np.ndarray | None:
    """Resolve ``key`` under the flattened naming convention.

    Some keys come from ``obs_group``, some from ``demo_group``,
    and ``robot0_joint_vel`` is derived via finite differences.
    """
    if key == "robot0_joint_pos":
        # Canonical: joint_states in obs/
        if "joint_states" in obs_group:
            return obs_group["joint_states"][:].astype(np.float32)
        return None

    if key == "robot0_joint_vel":
        # Derive from joint_states via finite differences
        src = "joint_states"
        if src not in obs_group:
            return None
        arr = obs_group[src][:].astype(np.float32)  # (T, 7)
        vel = np.diff(arr, axis=0, prepend=arr[:1])
        return vel

    if key == "robot0_eef_pos":
        # Try ee_pos, fall back to ee_states[:, :3]
        if "ee_pos" in obs_group:
            return obs_group["ee_pos"][:].astype(np.float32)
        if "ee_states" in obs_group:
            return obs_group["ee_states"][:, :3].astype(np.float32)
        return None

    if key == "robot0_eef_quat":
        # Extract from robot_states at demo root (9-dim: gripper+ee_pos+ee_quat)
        if "robot_states" in demo_group:
            rs = demo_group["robot_states"][:]  # (T, 9)
            if rs.shape[-1] >= 9:
                return rs[:, 5:9].astype(np.float32)
        return None

    if key == "robot0_gripper_qpos":
        if "gripper_states" in obs_group:
            return obs_group["gripper_states"][:].astype(np.float32)
        return None

    return None


# Dispatch table: schema_name → resolver function
_SCHEMA_RESOLVERS = {
    "robosuite": _resolve_robosuite,
    "flattened": _resolve_flattened,
}


# ---------------------------------------------------------------------------
# VisionStripper
# ---------------------------------------------------------------------------


class VisionStripper:
    """Parse a LIBERO HDF5 file and return state-only trajectory dicts.

    Auto-detects the HDF5 schema (robosuite vs. flattened naming) per file
    and resolves the configured state keys accordingly.

    Args:
        state_keys: List of DictConfig entries with ``key`` and ``dim`` fields,
                    or a list of plain strings.
        task_index: ``{task_name: int_id}`` mapping produced by
            :func:`phaseforge.data.libero.task_index.build_task_index`.
    """

    def __init__(
        self,
        state_keys: list[Any],
        task_index: dict[str, int] | None = None,
    ) -> None:
        self._key_specs: list[tuple[str, int | None]] = []
        for entry in state_keys:
            if isinstance(entry, (DictConfig, dict)):
                self._key_specs.append((entry["key"], entry.get("dim")))
            else:
                self._key_specs.append((str(entry), None))
        if task_index is None:
            raise ValueError(
                "task_index is required. It must be built via "
                "phaseforge.data.libero.task_index.build_task_index() so that "
                "task_id is deterministic across processes. The previous "
                "hash()-based scheme produced different ids per run."
            )
        self._task_index = task_index

    def strip(self, hdf5_path: Path) -> list[dict[str, np.ndarray]]:
        """Extract state-only data from one HDF5 file.

        Args:
            hdf5_path: Path to a LIBERO ``.hdf5`` file.

        Returns:
            List of trajectory dicts. Each dict contains:
            - ``"state"``:   np.ndarray (T, state_dim)
            - ``"action"``:  np.ndarray (T, action_dim)
            - ``"task_id"``: int (derived from filename)
            - ``"traj_id"``: int (demo index)
        """
        task_id = self._task_id_from_path(hdf5_path)
        trajectories: list[dict[str, np.ndarray]] = []

        with h5py.File(hdf5_path, "r") as f:
            data_group = f.get("data")
            if data_group is None:
                raise ValueError(f"HDF5 file {hdf5_path} has no 'data' group.")

            demo_keys = sorted(data_group.keys())
            logger.debug(f"  {hdf5_path.name}: {len(demo_keys)} demos")

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                obs = demo.get("obs")
                if obs is None:
                    logger.warning(f"  Demo {demo_key} has no 'obs'. Skipping.")
                    continue

                # Auto-detect schema for this file (once per file, cached below)
                if not hasattr(self, "_schema"):
                    schema = _detect_obs_schema(obs)
                    logger.info(
                        "  %s: detected schema '%s'", hdf5_path.name, schema
                    )
                    self._schema = schema
                resolver = _SCHEMA_RESOLVERS[self._schema]

                # Build state array
                state_arrays: list[np.ndarray] = []
                missing_keys: list[str] = []

                for key, expected_dim in self._key_specs:
                    arr = resolver(obs, demo, key)
                    if arr is None:
                        missing_keys.append(key)
                        continue
                    # Safety: skip if it looks like vision data
                    if _is_vision_key(key):
                        logger.error(
                            f"State key '{key}' looks like a vision key! Skipping."
                        )
                        continue
                    if expected_dim is not None and arr.shape[-1] != expected_dim:
                        logger.warning(
                            f"Key '{key}' dim mismatch: "
                            f"expected {expected_dim}, got {arr.shape[-1]}. Using actual."
                        )
                    state_arrays.append(arr)

                if missing_keys:
                    logger.debug(
                        f"  Demo {demo_key}: missing keys {missing_keys}. "
                        "They will be absent from state vector."
                    )

                if not state_arrays:
                    logger.warning(
                        f"  Demo {demo_key}: no state arrays found. Skipping."
                    )
                    continue

                state = np.concatenate(state_arrays, axis=-1)  # (T, state_dim)

                action_key = "actions"
                if action_key not in demo:
                    logger.warning(
                        f"  Demo {demo_key}: no 'actions'. Skipping."
                    )
                    continue
                action = demo[action_key][:].astype(np.float32)

                traj_id = int(
                    demo_key.replace("demo_", "").lstrip("0") or "0"
                )

                trajectories.append(
                    {
                        "state": state,
                        "action": action,
                        "task_id": task_id,
                        "traj_id": traj_id,
                    }
                )

        # Reset schema cache for next file
        if hasattr(self, "_schema"):
            del self._schema

        logger.info(
            "  %s: extracted %d trajectories, "
            "state_dim=%s",
            hdf5_path.name,
            len(trajectories),
            trajectories[0]["state"].shape[-1] if trajectories else "N/A",
        )
        return trajectories

    def _task_id_from_path(self, path: Path) -> int:
        from phaseforge.data.libero.task_index import task_id_for

        return task_id_for(path.stem, self._task_index)
