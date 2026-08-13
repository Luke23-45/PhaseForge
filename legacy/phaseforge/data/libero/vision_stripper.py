"""Vision stripper: extract state-only data from LIBERO HDF5 files.

This is the ONLY module in the codebase that opens image keys in HDF5.
After this module runs, no downstream code ever encounters image arrays.
Images are never loaded into RAM — we skip them entirely at the HDF5 level.

Schema contract (flattened naming only)
---------------------------------------
The mirror used by this project is pinned to ``yifengzhu-hf/LIBERO-datasets``
(see ``phaseforge/data/scripts/download_libero.py``), which uses the
flattened key convention:

    obs/joint_states       (T, 7)   — robot arm joint positions
    obs/ee_pos             (T, 3)   — end-effector position
    obs/ee_ori             (T, 3)   — end-effector axis-angle (unused)
    obs/gripper_states     (T, 2)   — gripper finger qpos
    demo/robot_states      (T, 9)   — [gripper(2), eef_pos(3), eef_quat(4)]
    demo/states            (T, S)   — flattened MuJoCo sim state (qpos+qvel);
                                      used only by the object-state decoder

Joint velocity is derived from ``joint_states`` via finite differences.
The end-effector quaternion is read from ``demo/robot_states[:, 5:9]``.

The config's ``state_keys`` use robosuite naming (``robot0_joint_pos`` etc.);
the resolver below maps them onto the flattened keys. Only this one schema is
supported — the auto-detection of the old "robosuite"-style releases was
removed because the mirror is pinned (Decision 2, 2026-08-07).

Object-state channel (P-Stage 1)
--------------------------------
When the pipeline passes an :class:`ObjectIndex`
(``phaseforge.data.libero.object_state``), the stripper additionally decodes
per-object world poses from ``demo/states`` (pure numpy, no physics
stepping) and appends them to the state vector:

    [ proprio (23) | object_block (k_slots*7) | mask (k_slots) ]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig

from phaseforge.data.libero.object_state import ObjectIndex

logger = logging.getLogger(__name__)

# Keys that are definitively vision data — never read into memory
_VISION_KEY_PATTERNS = ("rgb", "depth", "image", "pixel", "video", "wrist", "eye")


def _is_vision_key(key: str) -> bool:
    k = key.lower()
    return any(pat in k for pat in _VISION_KEY_PATTERNS)


# ---------------------------------------------------------------------------
# Flattened-schema key resolver
# ---------------------------------------------------------------------------


def _resolve_key(
    obs_group: h5py.Group,
    demo_group: h5py.Group,
    key: str,
) -> np.ndarray | None:
    """Resolve a robosuite-named ``key`` against the flattened schema.

    Some keys come from ``obs_group``, some from ``demo_group``, and
    ``robot0_joint_vel`` is derived via finite differences. Returns ``None``
    when the key cannot be produced (logged by the caller).
    """
    if key == "robot0_joint_pos":
        if "joint_states" in obs_group:
            return obs_group["joint_states"][:].astype(np.float32)
        return None

    if key == "robot0_joint_vel":
        if "joint_states" not in obs_group:
            return None
        arr = obs_group["joint_states"][:].astype(np.float32)  # (T, 7)
        return np.diff(arr, axis=0, prepend=arr[:1])

    if key == "robot0_eef_pos":
        if "ee_pos" in obs_group:
            return obs_group["ee_pos"][:].astype(np.float32)
        if "ee_states" in obs_group:
            return obs_group["ee_states"][:, :3].astype(np.float32)
        return None

    if key == "robot0_eef_quat":
        # robot_states at demo root is 9-dim: [gripper(2), eef_pos(3), eef_quat(4)]
        if "robot_states" in demo_group:
            rs = demo_group["robot_states"][:]
            if rs.shape[-1] >= 9:
                return rs[:, 5:9].astype(np.float32)
        return None

    if key == "robot0_gripper_qpos":
        if "gripper_states" in obs_group:
            return obs_group["gripper_states"][:].astype(np.float32)
        return None

    return None


# ---------------------------------------------------------------------------
# VisionStripper
# ---------------------------------------------------------------------------


class VisionStripper:
    """Parse a LIBERO HDF5 file (flattened schema) and return state-only dicts.

    Args:
        state_keys: List of DictConfig entries with ``key`` and ``dim`` fields,
                    or a list of plain strings. Robosuite naming.
        task_index: ``{task_name: int_id}`` mapping produced by
            :func:`phaseforge.data.libero.task_index.build_task_index`.
        object_index: Optional :class:`ObjectIndex` enabling the P-Stage 1
            object-state channel. When set, each trajectory's state is
            ``[proprio (23) | object_block | mask]`` (the mask is omitted
            when the index has ``include_mask=False``).
        action_dim: Expected action width (default 7). Mismatches are
            fatal — a silently resized action space would corrupt training.
    """

    def __init__(
        self,
        state_keys: list[Any],
        task_index: dict[str, int] | None = None,
        object_index: ObjectIndex | None = None,
        action_dim: int = 7,
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
        self.object_index = object_index
        self._action_dim = int(action_dim)

    def strip(self, hdf5_path: Path) -> list[dict[str, np.ndarray]]:
        """Extract state-only data from one HDF5 file.

        Args:
            hdf5_path: Path to a LIBERO ``.hdf5`` file (flattened schema).

        Returns:
            List of trajectory dicts. Each dict contains:
            - ``"state"``:      np.ndarray (T, state_dim)
            - ``"action"``:     np.ndarray (T, action_dim)
            - ``"task_id"``:    int (derived from filename)
            - ``"traj_id"``:    int (demo index)
            - ``"task_name"``:  str (HDF5 filename stem)
            - ``"source_file"``: str (HDF5 path, for audit)
            - ``"demo_key"``:   str (e.g. "demo_3")

        Validation is STRICT: a malformed demo (missing key, dimension
        mismatch, length mismatch, non-finite value, wrong action width)
        raises instead of being silently dropped. Silently omitting even
        one demo changes the dataset distribution and biases the experiment.
        """
        task_name = hdf5_path.stem
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
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: missing 'obs' group."
                    )

                # Build the proprioceptive state vector
                state_arrays: list[np.ndarray] = []
                for key, expected_dim in self._key_specs:
                    arr = _resolve_key(obs, demo, key)
                    if arr is None:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: missing required state key "
                            f"{key!r}."
                        )
                    if _is_vision_key(key):
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: state key {key!r} looks like a "
                            "vision key; the state schema must not contain "
                            "image data."
                        )
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: key {key!r} must have shape "
                            f"(T, dim), observed {arr.shape}."
                        )
                    if expected_dim is not None and arr.shape[-1] != expected_dim:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: key {key!r} dim mismatch: "
                            f"expected {expected_dim}, observed "
                            f"{arr.shape[-1]}."
                        )
                    state_arrays.append(arr)

                if len({arr.shape[0] for arr in state_arrays}) != 1:
                    shapes = {key: arr.shape for key, arr in zip(self._key_specs, state_arrays)}
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: state arrays have inconsistent "
                        f"timesteps: {shapes}. All proprio keys must have "
                        "the same T."
                    )
                if state_arrays[0].shape[0] < 1:
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: empty trajectory (T=0). A zero-length "
                        "demo would pollute the cache with a sample-less "
                        "trajectory — re-download or drop the file."
                    )

                state = np.concatenate(state_arrays, axis=-1)  # (T, 23)

                # Object-state channel: decode from demo/states (numpy-only)
                if self.object_index is not None:
                    if "states" not in demo:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: object_state is enabled but "
                            "'states' is missing at the demo root. "
                            "Re-download the dataset."
                        )
                    states = demo["states"][:].astype(np.float32)  # (T, S)
                    if states.ndim != 2:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: 'states' must have shape (T, S), "
                            f"observed {states.shape}."
                        )
                    if states.shape[0] != state.shape[0]:
                        raise ValueError(
                            f"Task '{task_name}' ({hdf5_path.name}), demo "
                            f"{demo_key}: 'states' has T={states.shape[0]} "
                            f"but proprio has T={state.shape[0]}."
                        )
                    obj_block, obj_mask = self.object_index.decode(task_name, states)
                    # obj_mask is (T, k_slots) when the index has
                    # include_mask=True, else (T, 0) — a no-op on
                    # concatenation. The eval env honors the same flag.
                    state = np.concatenate([state, obj_block, obj_mask], axis=-1)

                if "actions" not in demo:
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: missing 'actions'."
                    )
                action = demo["actions"][:].astype(np.float32)
                if action.ndim != 2:
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: 'actions' must have shape (T, A), "
                        f"observed {action.shape}."
                    )
                if action.shape[0] != state.shape[0]:
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: actions T={action.shape[0]} but state "
                        f"T={state.shape[0]}."
                    )
                if action.shape[-1] != self._action_dim:
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: action dim {action.shape[-1]} != "
                        f"expected {self._action_dim}."
                    )

                if not np.isfinite(state).all():
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: non-finite values in state vector."
                    )
                if not np.isfinite(action).all():
                    raise ValueError(
                        f"Task '{task_name}' ({hdf5_path.name}), demo "
                        f"{demo_key}: non-finite values in actions."
                    )

                traj_id = int(
                    demo_key.replace("demo_", "").lstrip("0") or "0"
                )

                trajectories.append(
                    {
                        "state": state,
                        "action": action,
                        "task_id": task_id,
                        "traj_id": traj_id,
                        "task_name": task_name,
                        "source_file": str(hdf5_path),
                        "demo_key": demo_key,
                    }
                )

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
