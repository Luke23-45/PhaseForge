"""CPU-only regression tests for the real-data object-index census helpers."""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.data.scripts.build_object_index import (
    _infer_offset_candidates,
    _object_entries_from_env,
    _run_b6_gate,
    _select_qpos_offset,
)


class _FakeData:
    body_xpos = np.zeros((2, 3))
    body_xquat = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


class _FakeModel:
    nq = 16
    nv = 6
    body_names = ["robot0", "cube"]
    body_parentid = np.array([0, 0])
    body_jntadr = np.array([0, 1])  # cube's joint is joint 1
    jnt_type = np.array([0, 0])  # free joints
    jnt_qposadr = np.array([0, 9])  # cube's free-joint qpos starts at 9
    jnt_pos = np.zeros((2, 3))
    jnt_axis = np.zeros((2, 3))


class _FakeSim:
    model = _FakeModel()
    data = _FakeData()


class _FakeEnv:
    """Minimal environment exposing the B6 gate's required API.

    The live observations always reflect the TRUE qpos block of the
    current state (as robosuite's ``set_state_from_flattened`` would):
    the cube's free-joint pose is read from the qpos block at
    ``[qpos_offset + 9, qpos_offset + 16)``, and the quat is reported in
    xyzw (the robosuite observable convention — the gate converts it back
    to wxyz before comparing with the decode).
    """

    sim = _FakeSim()

    def __init__(self, states: np.ndarray, qpos_offset: int = 1) -> None:
        self.states = states
        self.qpos_offset = qpos_offset
        self.current_state = states[0]

    def set_init_state(self, state: np.ndarray) -> None:
        self.current_state = state

    def _get_observations(self) -> dict[str, np.ndarray]:
        off = self.qpos_offset
        pose = self.current_state[off + 9 : off + 16].copy()
        quat_wxyz = pose[3:]
        return {
            "cube_pos": pose[:3],
            "cube_quat": np.concatenate((quat_wxyz[1:], quat_wxyz[:1])),
        }


def _states(offset: int = 1) -> np.ndarray:
    """Mirror-style states with the cube's free-joint pose in the qpos block.

    ``offset=1``: the canonical LIBERO layout ``[time, qpos, qvel]`` —
    sim time in column 0, pose at true qpos columns [10, 17), width
    nq+nv+1 = 23. ``offset=0``: plain ``[qpos, qvel]``, pose at columns
    [9, 16), width nq+nv = 22.
    """
    states = np.zeros((5, 22 + offset), dtype=np.float32)
    if offset == 1:
        states[:, 0] = np.arange(5, dtype=np.float32)  # sim time — must be skipped
    states[:, offset + 9 : offset + 12] = np.arange(15, dtype=np.float32).reshape(5, 3)
    states[:, offset + 12] = 1.0  # unit quaternion [1, 0, 0, 0]
    return states


def _entry() -> dict:
    return {
        "name": "cube",
        "joint_type": "free",
        "qpos_start": 9,
        "qpos_len": 7,
    }


# ---------------------------------------------------------------------------
# qpos-block offset inference
# ---------------------------------------------------------------------------


def test_infer_offset_candidates() -> None:
    # Plain [qpos, qvel]: the only candidate is offset 0.
    assert _infer_offset_candidates(22, 16, 6) == (0,)
    # Canonical LIBERO [time, qpos, qvel]: 0 and 1, gate arbitrates.
    assert _infer_offset_candidates(23, 16, 6) == (0, 1)
    # Below nq+nv cannot be a flattened sim state — must fail loudly.
    with pytest.raises(ValueError, match="not the flattened sim state"):
        _infer_offset_candidates(21, 16, 6)
    # Above nq+nv+1 is an unknown layout — never guess.
    with pytest.raises(ValueError, match="unknown flattened-state layout"):
        _infer_offset_candidates(24, 16, 6)


# ---------------------------------------------------------------------------
# B6 gate (decode == live obs)
# ---------------------------------------------------------------------------


def test_b6_gate_passes_with_time_first_offset() -> None:
    """The canonical LIBERO layout: [time, qpos, qvel], decode at offset 1."""
    states = _states(offset=1)
    env = _FakeEnv(states, qpos_offset=1)

    # The second demo is deliberately different; the helper must validate
    # both demos, not only the first one.
    second = states.copy()
    second[:, 11] += 10.0

    error = _run_b6_gate(
        env,
        "TASK_A",  # canonical name (census strips the ``_demo`` stem suffix)
        [_entry()],
        [states, second],
        qpos_offset=1,
        max_demos=2,
        steps_per_demo=3,
    )
    assert error == pytest.approx(0.0)


def test_b6_gate_fails_when_qpos_block_is_misaligned() -> None:
    """The real mirror bug, pinned: states are ``[time, qpos, qvel]``
    (offset 1) but a decode that reads the first ``nq`` columns (offset 0)
    shifts the whole qpos block by one — the time scalar becomes qpos[0]
    and every object pose/quaternion is wrong. Live obs reflect the TRUE
    physics, so the gate must fail this with a large mismatch."""
    states = _states(offset=1)
    env = _FakeEnv(states, qpos_offset=1)  # live obs = true qpos

    with pytest.raises(ValueError, match="B6 gate FAILED"):
        _run_b6_gate(
            env,
            "TASK_A",
            [_entry()],
            [states],
            qpos_offset=0,
            max_demos=1,
            steps_per_demo=1,
        )


def test_b6_gate_passes_with_plain_offset_zero() -> None:
    """Mirrors without the time scalar decode at offset 0."""
    states = _states(offset=0)
    env = _FakeEnv(states, qpos_offset=0)

    error = _run_b6_gate(
        env,
        "TASK_A",
        [_entry()],
        [states],
        qpos_offset=0,
        max_demos=1,
        steps_per_demo=3,
    )
    assert error == pytest.approx(0.0)


def test_b6_gate_rejects_non_unit_quaternion() -> None:
    # Corrupt the X component (qpos column 13, i.e. states column 14 with
    # offset 1) — NOT the w component (column 12/13): the w component sits
    # at exactly the position a single-component check would read, so a
    # w-only test would pass even with the old buggy validation. This
    # pins the fix: every quaternion must be validated on all four
    # components.
    states = _states(offset=1)
    states[:, 14] = 2.0
    env = _FakeEnv(states, qpos_offset=1)

    with pytest.raises(ValueError, match="quaternion.*norm"):
        _run_b6_gate(
            env,
            "TASK_A",
            [_entry()],
            [states],
            qpos_offset=1,
            max_demos=1,
            steps_per_demo=1,
        )


# ---------------------------------------------------------------------------
# Candidate arbitration (_select_qpos_offset)
# ---------------------------------------------------------------------------


def test_select_qpos_offset_arbitrates_time_first() -> None:
    """Real mirror data (time-first): candidate 0 misaligns the qpos block
    and fails the gate, candidate 1 matches live FK and wins."""
    states = _states(offset=1)
    env = _FakeEnv(states, qpos_offset=1)

    offset, err = _select_qpos_offset(
        env,
        "TASK_A",
        [_entry()],
        [states],
        (0, 1),
        max_demos=1,
        steps_per_demo=2,
    )
    assert offset == 1
    assert err == pytest.approx(0.0)


def test_select_qpos_offset_picks_zero_for_plain_states() -> None:
    """Mirror without the time scalar: only offset 0 is offered and wins."""
    states = _states(offset=0)
    env = _FakeEnv(states, qpos_offset=0)

    offset, err = _select_qpos_offset(
        env,
        "TASK_A",
        [_entry()],
        [states],
        (0,),
        max_demos=1,
        steps_per_demo=2,
    )
    assert offset == 0
    assert err == pytest.approx(0.0)


def test_select_qpos_offset_fails_when_no_candidate_matches() -> None:
    """A corpus whose stored states match NEITHER offset must fail the
    census with every candidate's error attached — never guess."""
    states = _states(offset=1)
    states[:, 14] = 2.0  # corrupted quat: both candidates trip a check
    env = _FakeEnv(states, qpos_offset=1)

    with pytest.raises(ValueError, match="every candidate qpos offset"):
        _select_qpos_offset(
            env,
            "TASK_A",
            [_entry()],
            [states],
            (0, 1),
            max_demos=1,
            steps_per_demo=1,
        )


# ---------------------------------------------------------------------------
# Entry derivation
# ---------------------------------------------------------------------------


def test_object_entries_from_env() -> None:
    env = _FakeEnv(_states())
    entries = _object_entries_from_env(env, "TASK_A")

    assert [e["name"] for e in entries] == ["cube"]
    assert entries[0]["joint_type"] == "free"
    assert entries[0]["qpos_start"] == 9
    assert entries[0]["qpos_len"] == 7
