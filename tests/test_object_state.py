"""Tests for the P-Stage 1 object-state channel.

Covers the ``ObjectIndex.decode`` layout (free/hinge joints, mask, k_slots
policies) and the config <-> index handshake validation in
``DataPipelineStateMachine._load_object_index``.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from phaseforge.data.ingestion.state_machine import DataPipelineStateMachine
from phaseforge.data.libero.object_state import (
    JOINT_FREE,
    JOINT_HINGE,
    ObjectEntry,
    ObjectIndex,
    TaskObjectTable,
)

T = 5
NQ = 23  # 9 robot qpos + 2 free objects x 7
K_SLOTS = 4
DIM_PER_OBJECT = 7
PROPRIO_DIM = 23
STATE_DIM = PROPRIO_DIM + K_SLOTS * DIM_PER_OBJECT + K_SLOTS  # 55 with mask


def make_entry(
    name: str,
    qpos_start: int = 0,
    qpos_len: int = DIM_PER_OBJECT,
    joint_type: str = JOINT_FREE,
) -> ObjectEntry:
    return ObjectEntry(
        name=name,
        joint_type=joint_type,
        qpos_start=qpos_start,
        qpos_len=qpos_len,
        anchor_world=np.array([0.0, 0.0, 0.0]),
        axis_world=np.array([0.0, 0.0, 1.0]),
        rest_pos=np.array([1.0, 0.0, 0.0]),
        rest_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def make_index(include_mask: bool = True, k_slots: int = K_SLOTS) -> ObjectIndex:
    """Index for a single task ``TASK_A`` with 2 free-joint objects."""
    table = TaskObjectTable(
        task_name="TASK_A",
        nq=NQ,
        objects=[
            make_entry("cube_0", qpos_start=9),
            make_entry("cube_1", qpos_start=16),
        ],
    )
    return ObjectIndex(
        tasks={"TASK_A": table},
        k_slots=k_slots,
        dim_per_object=DIM_PER_OBJECT,
        include_mask=include_mask,
    )


def make_states(nq: int = NQ, t: int = T) -> np.ndarray:
    """A (t, nq + nv) flattened sim state with deterministic content."""
    rng = np.random.default_rng(0)
    qpos = rng.normal(0, 1, (t, nq)).astype(np.float32)
    qpos[:, 9 + 3] = 1.0  # first object quat w — keep normalized
    qpos[:, 9 + 4 : 9 + 7] = 0.0
    qpos[:, 16 + 3] = 1.0
    qpos[:, 16 + 4 : 16 + 7] = 0.0
    qvel = rng.normal(0, 1, (t, nq - 6)).astype(np.float32)  # nv arbitrary
    return np.concatenate([qpos, qvel], axis=-1)


# ---------------------------------------------------------------------------
# ObjectIndex.decode
# ---------------------------------------------------------------------------


def test_decode_free_joint_copies_qpos_and_fills_mask() -> None:
    index = make_index(include_mask=True)
    states = make_states()
    block, mask = index.decode("TASK_A", states)

    assert block.shape == (T, K_SLOTS * DIM_PER_OBJECT)
    assert mask.shape == (T, K_SLOTS)
    # Free-joint qpos IS the world pose: [x, y, z, qw, qx, qy, qz].
    np.testing.assert_allclose(block[:, 0:7], states[:, 9:16], rtol=0, atol=0)
    np.testing.assert_allclose(block[:, 7:14], states[:, 16:23], rtol=0, atol=0)
    # Unfilled slots stay zero-padded.
    np.testing.assert_allclose(block[:, 14:], np.zeros((T, 14)), rtol=0, atol=0)
    # Mask marks exactly the two filled slots.
    np.testing.assert_allclose(mask, np.tile([1.0, 1.0, 0.0, 0.0], (T, 1)))


def test_decode_include_mask_false_omits_mask_dims() -> None:
    index = make_index(include_mask=False)
    states = make_states()
    block, mask = index.decode("TASK_A", states)

    assert mask.shape == (T, 0)
    assert block.shape == (T, K_SLOTS * DIM_PER_OBJECT)
    # Concatenating the empty mask is a no-op: the resulting state vector
    # contains the object block but no mask dims.
    state = np.concatenate([np.zeros((T, PROPRIO_DIM)), block, mask], axis=-1)
    assert state.shape == (T, PROPRIO_DIM + K_SLOTS * DIM_PER_OBJECT)
    assert index.state_block_size == K_SLOTS * DIM_PER_OBJECT


def test_state_block_size_respects_include_mask() -> None:
    assert make_index(include_mask=True).state_block_size == 28 + 4
    assert make_index(include_mask=False).state_block_size == 28


def test_decode_rejects_objects_exceeding_k_slots() -> None:
    index = make_index(k_slots=1)  # 2 objects, 1 slot
    with pytest.raises(ValueError, match="k_slots"):
        index.decode("TASK_A", make_states())


def test_decode_rejects_states_narrower_than_nq() -> None:
    index = make_index()
    with pytest.raises(ValueError, match="nq"):
        index.decode("TASK_A", np.zeros((T, NQ - 2), dtype=np.float32))


def test_decode_hinge_screw_rotation() -> None:
    """90 deg about world +z at the origin maps rest_pos (1,0,0) to (0,1,0)."""
    table = TaskObjectTable(
        task_name="TASK_B",
        nq=16,
        objects=[
            ObjectEntry(
                name="drawer_0",
                joint_type=JOINT_HINGE,
                qpos_start=9,
                qpos_len=1,
                anchor_world=np.array([0.0, 0.0, 0.0]),
                axis_world=np.array([0.0, 0.0, 1.0]),
                rest_pos=np.array([1.0, 0.0, 0.0]),
                rest_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        ],
    )
    index = ObjectIndex(tasks={"TASK_B": table}, k_slots=1)
    states = np.zeros((T, 22), dtype=np.float32)
    states[:, 9] = np.pi / 2  # hinge angle
    block, _ = index.decode("TASK_B", states)

    np.testing.assert_allclose(block[:, :3], np.tile([0.0, 1.0, 0.0], (T, 1)), atol=1e-6)
    expected_quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(block[:, 3:], np.tile(expected_quat, (T, 1)), atol=1e-6)


# ---------------------------------------------------------------------------
# Config <-> index handshake (DataPipelineStateMachine._load_object_index)
# ---------------------------------------------------------------------------


def _make_fsm(
    object_state: dict, state_dim: int = STATE_DIM, index_path: str = ""
) -> DataPipelineStateMachine:
    data_cfg = OmegaConf.create(
        {
            "object_state": object_state,
            "state_keys": [
                {"key": "robot0_joint_pos", "dim": 7},
                {"key": "robot0_joint_vel", "dim": 7},
                {"key": "robot0_eef_pos", "dim": 3},
                {"key": "robot0_eef_quat", "dim": 4},
                {"key": "robot0_gripper_qpos", "dim": 2},
            ],
            "state_dim": state_dim,
            "libero": {"phase_labeler": {"num_phases": 6}},
        }
    )
    return DataPipelineStateMachine(OmegaConf.create({"data": data_cfg}))


def _object_state_cfg(index_path: str, **overrides) -> dict:
    cfg = {
        "enabled": True,
        "k_slots": K_SLOTS,
        "dim_per_object": DIM_PER_OBJECT,
        "include_mask": True,
        "index_path": index_path,
    }
    cfg.update(overrides)
    return cfg


def test_load_object_index_passes_with_matching_config(tmp_path) -> None:
    index_path = str(tmp_path / "object_index.json")
    make_index().save(index_path)
    fsm = _make_fsm(_object_state_cfg(index_path))

    index = fsm._load_object_index()
    assert index is not None
    assert index.k_slots == K_SLOTS


def test_load_object_index_rejects_k_slots_mismatch(tmp_path) -> None:
    index_path = str(tmp_path / "object_index.json")
    make_index().save(index_path)
    fsm = _make_fsm(_object_state_cfg(index_path, k_slots=8))

    with pytest.raises(ValueError, match="k_slots"):
        fsm._load_object_index()


def test_load_object_index_rejects_dim_per_object_mismatch(tmp_path) -> None:
    index_path = str(tmp_path / "object_index.json")
    make_index().save(index_path)
    fsm = _make_fsm(_object_state_cfg(index_path, dim_per_object=6))

    with pytest.raises(ValueError, match="dim_per_object"):
        fsm._load_object_index()


def test_load_object_index_rejects_include_mask_mismatch(tmp_path) -> None:
    index_path = str(tmp_path / "object_index.json")
    make_index().save(index_path)
    fsm = _make_fsm(_object_state_cfg(index_path, include_mask=False))

    with pytest.raises(ValueError, match="include_mask"):
        fsm._load_object_index()


def test_load_object_index_rejects_state_dim_mismatch(tmp_path) -> None:
    index_path = str(tmp_path / "object_index.json")
    make_index().save(index_path)
    fsm = _make_fsm(_object_state_cfg(index_path), state_dim=99)

    with pytest.raises(ValueError, match="state_dim"):
        fsm._load_object_index()


def test_load_object_index_disabled_returns_none(tmp_path) -> None:
    fsm = _make_fsm({"enabled": False})
    assert fsm._load_object_index() is None


def test_object_mask_dims_none_when_index_has_no_mask(tmp_path) -> None:
    """Without mask dims there is nothing to exclude from normalization."""
    index_path = str(tmp_path / "object_index.json")
    index = make_index(include_mask=False)
    index.save(index_path)
    # With the mask omitted the object layout is proprio 23 + block 28 = 51.
    fsm = _make_fsm(
        _object_state_cfg(index_path, include_mask=False), state_dim=51
    )

    loaded = fsm._load_object_index()
    assert fsm._object_mask_dims(loaded) is None

    index_with_mask = make_index(include_mask=True)
    assert fsm._object_mask_dims(index_with_mask) == set(range(51, 55))
