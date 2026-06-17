"""Vision stripper: extract state-only data from LIBERO HDF5 files.

This is the ONLY module in the codebase that opens image keys in HDF5.
After this module runs, no downstream code ever encounters image arrays.
Images are never loaded into RAM — we skip them entirely at the HDF5 level.
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


class VisionStripper:
    """Parse a LIBERO HDF5 file and return state-only trajectory dicts.

    Args:
        state_keys: List of DictConfig entries with ``key`` and ``dim`` fields,
                    or a list of plain strings.
    """

    def __init__(self, state_keys: list[Any]) -> None:
        # Normalize to list of (key, dim) pairs
        self._key_specs: list[tuple[str, int | None]] = []
        for entry in state_keys:
            if isinstance(entry, (DictConfig, dict)):
                self._key_specs.append((entry["key"], entry.get("dim")))
            else:
                self._key_specs.append((str(entry), None))

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

                # Build state array from specified keys only
                state_arrays: list[np.ndarray] = []
                missing_keys: list[str] = []

                for key, expected_dim in self._key_specs:
                    if key in obs:
                        arr = obs[key][:]  # Load this key only
                        if _is_vision_key(key):
                            # Safety: should never happen, but skip if so
                            logger.error(
                                f"State key '{key}' looks like a vision key! Skipping."
                            )
                            continue
                        if expected_dim is not None and arr.shape[-1] != expected_dim:
                            logger.warning(
                                f"Key '{key}' dim mismatch: "
                                f"expected {expected_dim}, got {arr.shape[-1]}. Using actual."
                            )
                        state_arrays.append(arr.astype(np.float32))
                    else:
                        missing_keys.append(key)

                if missing_keys:
                    logger.debug(
                        f"  Demo {demo_key}: missing keys {missing_keys}. "
                        "They will be absent from state vector."
                    )

                if not state_arrays:
                    logger.warning(f"  Demo {demo_key}: no state arrays found. Skipping.")
                    continue

                state = np.concatenate(state_arrays, axis=-1)  # (T, state_dim)

                action_key = "actions"
                if action_key not in demo:
                    logger.warning(f"  Demo {demo_key}: no 'actions'. Skipping.")
                    continue
                action = demo[action_key][:].astype(np.float32)  # (T, action_dim)

                traj_id = int(demo_key.replace("demo_", "").lstrip("0") or "0")

                trajectories.append(
                    {
                        "state": state,
                        "action": action,
                        "task_id": task_id,
                        "traj_id": traj_id,
                    }
                )

        logger.info(
            f"  {hdf5_path.name}: extracted {len(trajectories)} trajectories, "
            f"state_dim={trajectories[0]['state'].shape[-1] if trajectories else 'N/A'}"
        )
        return trajectories

    @staticmethod
    def _task_id_from_path(path: Path) -> int:
        """Derive a numeric task ID from the filename (hash-based)."""
        return abs(hash(path.stem)) % (10**6)
