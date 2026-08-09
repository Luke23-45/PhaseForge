"""CPU-only regression tests for the real-data object-index census helpers."""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.data.scripts.build_object_index import (
    _object_entries_from_env,
    _run_b6_gate,
)


class _FakeData:
    body_xpos = np.zeros((2, 3))
    body_xquat = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


class _FakeModel:
    nq = 16
    nv = 6
    body_names = ["robot0", "cube"]
    body_parentid = np.array([0, 0])
    body_jntadr = np.array([0, 0])
    jnt_type = np.array([0, 0])  # free joints
    jnt_qposadr = np.array([0, 9])
    jnt_pos = np.zeros((2, 3))
    jnt_axis = np.zeros((2, 3))


class _FakeSim:
    model = _FakeModel()
    data = _FakeData()


class _FakeEnv:
    """Minimal environment exposing the B6 gate's required API."""

    sim = _FakeSim()

    def __init__(self, states: np.ndarray) -> None:
        self.states = states
        self.current_state = states[0]

    def set_init_state(self, state: np.ndarray) -> None:
        self.current_state = state

    def _get_observations(self) -> dict[str, np.ndarray]:
        pose = self.current_state[9:16].copy()
        return {
            "cube_pos": pose[:3],
            "cube_quat": pose[3:],
        }


def _states() -> np.ndarray:
    states = np.zeros((5, 22), dtype=np.float32)
    states[:, 9:12] = np.arange(15, dtype=np.float32).reshape(5, 3)
    states[:, 12] = 1.0  # unit quaternion [1, 0, 0, 0]
    return states


def _entry() -> dict:
    return {
        "name": "cube",
        "joint_type": "free",
        "qpos_start": 9,
        "qpos_len": 7,
    }


def test_b6_gate_accepts_multiple_demos_and_timesteps() -> None:
    states = _states()
    env = _FakeEnv(states)

    # The second demo is deliberately different; the helper must validate
    # both demos, not only the first one.
    second = states.copy()
    second[:, 9] += 10.0

    error = _run_b6_gate(
        env,
        "TASK_A",  # canonical name (census strips the ``_demo`` stem suffix)
        [_entry()],
        [states, second],
        max_demos=2,
        steps_per_demo=3,
    )
    assert error == pytest.approx(0.0)


def test_b6_gate_rejects_non_unit_quaternion() -> None:
    # Corrupt the X component (state index 13) — NOT the w component (12):
    # the w component sits at exactly the position the previous buggy
    # slice ``decoded[0, 3::7]`` read, so a w-only test would pass even
    # with the single-component bug. This pins the fix: every quaternion
    # must be validated on all four components.
    states = _states()
    states[:, 13] = 2.0
    env = _FakeEnv(states)

    with pytest.raises(ValueError, match="quaternion.*norm"):
        _run_b6_gate(
            env,
            "TASK_A",
            [_entry()],
            [states],
            max_demos=1,
            steps_per_demo=1,
        )


def test_states_width_tolerates_extra_trailing_dims() -> None:
    """The mirror stores nq+nv+1-wide states in KITCHEN scenes; the census
    must accept widths >= nq+nv (the B6 gate arbitrates the qpos block)."""
    env = _FakeEnv(_states())
    entries = _object_entries_from_env(env, "TASK_A", np.zeros((3, 22), dtype=np.float32))
    assert [e["name"] for e in entries] == ["cube"]

    # nq(16) + nv(6) + 1 trailing dim — must pass like the real mirror.
    env = _FakeEnv(_states())
    entries = _object_entries_from_env(env, "TASK_A", np.zeros((3, 23), dtype=np.float32))
    assert [e["name"] for e in entries] == ["cube"]

    # Below nq+nv cannot be a flattened sim state — must fail loudly.
    with pytest.raises(ValueError, match="not the flattened sim state"):
        _object_entries_from_env(env, "TASK_A", np.zeros((3, 21), dtype=np.float32))
