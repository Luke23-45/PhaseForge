"""Strict-validation tests for VisionStripper (review blocker 3).

A malformed demo must fail loudly with task/file/demo context instead of
being silently dropped: missing keys, wrong dims, length mismatches,
non-finite values, and wrong action widths all raise.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from phaseforge.data.libero.vision_stripper import VisionStripper

T = 8
STATE_KEYS = [
    {"key": "robot0_joint_pos", "dim": 7},
    {"key": "robot0_joint_vel", "dim": 7},
    {"key": "robot0_eef_pos", "dim": 3},
    {"key": "robot0_eef_quat", "dim": 4},
    {"key": "robot0_gripper_qpos", "dim": 2},
]
TASK_INDEX = {"TASK_A_demo": 0}


def _write_valid_file(path: Path, n_demos: int = 2) -> None:
    """A well-formed flattened-schema file: 2 demos x T steps."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for d in range(n_demos):
            grp = data.create_group(f"demo_{d}")
            obs = grp.create_group("obs")
            obs["joint_states"] = rng.normal(0, 0.5, (T, 7)).astype(np.float32)
            obs["ee_pos"] = rng.normal(0, 0.5, (T, 3)).astype(np.float32)
            obs["gripper_states"] = np.full((T, 2), 0.06, np.float32)
            quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (T, 1))
            grp["robot_states"] = np.concatenate(
                [obs["gripper_states"][:], obs["ee_pos"][:], quat], axis=-1
            ).astype(np.float32)
            grp["states"] = np.zeros((T, 30), np.float32)
            grp["actions"] = rng.normal(0, 0.05, (T, 7)).astype(np.float32)


def _open_demo(path: Path, demo_key: str = "demo_0"):
    """Context manager returning (h5py.File, obs_group, demo_group)."""
    f = h5py.File(path, "r+")
    demo = f["data"][demo_key]
    return f, demo


def _strip(path: Path, **kwargs) -> list[dict]:
    stripper = VisionStripper(
        state_keys=STATE_KEYS, task_index=TASK_INDEX, **kwargs
    )
    return stripper.strip(path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_extract_adds_audit_metadata(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path, n_demos=2)
    trajs = _strip(path)

    assert len(trajs) == 2
    for traj in trajs:
        assert traj["task_name"] == "TASK_A_demo"
        assert traj["source_file"] == str(path)
        assert traj["task_id"] == 0
        assert traj["demo_key"] in ("demo_0", "demo_1")
        assert traj["state"].shape == (T, 23)
        assert traj["action"].shape == (T, 7)


# ---------------------------------------------------------------------------
# Strict failure modes (all must RAISE, never skip)
# ---------------------------------------------------------------------------


def test_missing_required_state_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["obs"]["joint_states"]

    with pytest.raises(ValueError, match="robot0_joint_pos"):
        _strip(path)


def test_state_dim_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        ds = f["data"]["demo_0"]["obs"]["joint_states"]
        bad = ds[:, :6]
        del f["data"]["demo_0"]["obs"]["joint_states"]
        f["data"]["demo_0"]["obs"]["joint_states"] = bad

    with pytest.raises(ValueError, match="dim mismatch"):
        _strip(path)


def test_missing_actions_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["actions"]

    with pytest.raises(ValueError, match="missing 'actions'"):
        _strip(path)


def test_action_dim_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["actions"]
        f["data"]["demo_0"]["actions"] = np.zeros((T, 6), np.float32)

    with pytest.raises(ValueError, match="action dim 6"):
        _strip(path)


def test_state_action_length_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["actions"]
        f["data"]["demo_0"]["actions"] = np.zeros((T - 2, 7), np.float32)

    with pytest.raises(ValueError, match="actions T=6 but state T=8"):
        _strip(path)


def test_nonfinite_state_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        ds = f["data"]["demo_0"]["obs"]["joint_states"]
        arr = ds[:]
        arr[3, 0] = np.nan
        del f["data"]["demo_0"]["obs"]["joint_states"]
        f["data"]["demo_0"]["obs"]["joint_states"] = arr

    with pytest.raises(ValueError, match="non-finite"):
        _strip(path)


def test_missing_obs_group_raises(tmp_path: Path) -> None:
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["obs"]

    with pytest.raises(ValueError, match="missing 'obs' group"):
        _strip(path)


def test_zero_length_trajectory_raises(tmp_path: Path) -> None:
    """T=0 demos must be rejected loudly: a sample-less trajectory would
    silently pollute the cache and skew the dataset distribution."""
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path, n_demos=1)
    with h5py.File(path, "r+") as f:
        demo = f["data"]["demo_0"]
        del demo["obs"]
        del demo["robot_states"]
        del demo["states"]
        del demo["actions"]
        obs = demo.create_group("obs")
        obs["joint_states"] = np.zeros((0, 7), np.float32)
        obs["ee_pos"] = np.zeros((0, 3), np.float32)
        obs["gripper_states"] = np.zeros((0, 2), np.float32)
        demo["robot_states"] = np.zeros((0, 9), np.float32)
        demo["states"] = np.zeros((0, 30), np.float32)
        demo["actions"] = np.zeros((0, 7), np.float32)

    with pytest.raises(ValueError, match="T=0"):
        _strip(path)


def test_joint_velocity_derived_via_finite_difference(tmp_path: Path) -> None:
    """Parity contract (E3) ingest side: robot0_joint_vel must be the same
    finite difference the eval env computes — np.diff(pos, prepend=pos[:1]).
    t=0 is zeros, then vel[t] = pos[t] - pos[t-1]. Never the raw simulator
    qvel."""
    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path, n_demos=1)
    pos = (
        np.linspace(0.0, 1.0, T).reshape(-1, 1) * np.arange(1, 8)
    ).astype(np.float32)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["obs"]["joint_states"]
        f["data"]["demo_0"]["obs"]["joint_states"] = pos

    state = _strip(path)[0]["state"]  # (T, 23): pos(7) | vel(7) | eef(3) ...
    expected = np.diff(pos, axis=0, prepend=pos[:1])
    np.testing.assert_allclose(state[:, 7:14], expected, rtol=0, atol=0)
    np.testing.assert_allclose(state[0, 7:14], 0.0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Object-state channel strictness
# ---------------------------------------------------------------------------


def test_object_state_enabled_but_states_missing_raises(
    tmp_path: Path,
) -> None:
    from phaseforge.data.libero.object_state import (
        ObjectEntry,
        ObjectIndex,
        TaskObjectTable,
    )

    path = tmp_path / "TASK_A_demo.hdf5"
    _write_valid_file(path)
    with h5py.File(path, "r+") as f:
        del f["data"]["demo_0"]["states"]

    table = TaskObjectTable(
        task_name="TASK_A_demo",
        nq=30,
        objects=[
            ObjectEntry(
                name="bowl_0", joint_type="free", qpos_start=9, qpos_len=7
            )
        ],
    )
    index = ObjectIndex(tasks={"TASK_A_demo": table}, k_slots=16)

    with pytest.raises(ValueError, match="'states' is missing"):
        _strip(path, object_index=index)
