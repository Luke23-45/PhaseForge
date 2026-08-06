"""CPU-only tests for the state-only LIBERO env wrapper.

Runs without the real ``libero``/``robosuite`` packages: the wrapper's
``__init__`` (which imports LIBERO) is bypassed via ``object.__new__`` and
a fake robosuite-style env is injected. This verifies the state vector
construction, the official reset sequence (seed -> reset -> set_init_state
-> dummy steps), and the gymnasium-style step mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.evaluations.envs.libero_env import (
    SUITE_BENCHMARK_NAMES,
    SUITE_MAX_STEPS,
    StateOnlyLiberoEnv,
)

STATE_DIM = 23
KEY_DIMS = {
    "robot0_joint_pos": 7,
    "robot0_joint_vel": 7,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
}


def make_fake_obs() -> dict[str, np.ndarray]:
    """Observation dict in LIBERO's robosuite naming; values are distinct."""
    obs: dict[str, np.ndarray] = {}
    v = 1.0
    for key, dim in KEY_DIMS.items():
        obs[key] = np.arange(v, v + dim, dtype=np.float64)
        v += dim
    return obs


def expected_state_vector() -> np.ndarray:
    """The 23-dim concatenation in the training-data key order."""
    parts = []
    v = 1.0
    for dim in KEY_DIMS.values():
        parts.append(np.arange(v, v + dim, dtype=np.float32))
        v += dim
    return np.concatenate(parts).astype(np.float32)


class FakeRobosuiteEnv:
    """Minimal robosuite-style env: step -> (obs, reward, done, info)."""

    def __init__(self, done: bool = False, success: bool = False) -> None:
        self.calls: list[tuple] = []
        self.obs = make_fake_obs()
        self.done = done
        self.success = success

    def seed(self, seed: int) -> None:
        self.calls.append(("seed", seed))

    def reset(self) -> None:
        self.calls.append(("reset",))

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        self.calls.append(("set_init_state", state))
        return self.obs

    def step(self, action: np.ndarray):
        self.calls.append(("step", action))
        return self.obs, 0.0, self.done, {}

    def check_success(self) -> bool:
        self.calls.append(("check_success",))
        return self.success

    def close(self) -> None:
        self.calls.append(("close",))


def _make_env(
    env: FakeRobosuiteEnv,
    num_steps_wait: int = 10,
    seed: int = 42,
    num_init_states: int = 5,
) -> StateOnlyLiberoEnv:
    """Construct a StateOnlyLiberoEnv without running its LIBERO __init__."""
    obj = object.__new__(StateOnlyLiberoEnv)
    obj._env = env
    obj.seed = seed
    obj.num_steps_wait = num_steps_wait
    obj._elapsed_steps = 0
    obj._init_states = [np.zeros(STATE_DIM, dtype=np.float64) for _ in range(num_init_states)]
    obj._task = type("Task", (), {"language": "fake task"})()
    return obj


# ---------------------------------------------------------------------------
# State vector construction
# ---------------------------------------------------------------------------


def test_state_vector_ordering_and_dims() -> None:
    env = _make_env(FakeRobosuiteEnv())
    state = env._extract_state(make_fake_obs())
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state, expected_state_vector(), rtol=0, atol=0)


def test_state_vector_matches_training_state_keys() -> None:
    """The concatenation order must match config/data/common.yaml state_keys."""
    env = _make_env(FakeRobosuiteEnv())
    state = env._extract_state(make_fake_obs())
    # joint_pos (7) + joint_vel (7) + eef_pos (3) + eef_quat (4) + gripper (2)
    assert state[:7].tolist() == list(range(1, 8))
    assert state[7:14].tolist() == list(range(8, 15))
    assert state[14:17].tolist() == list(range(15, 18))
    assert state[17:21].tolist() == list(range(18, 22))
    assert state[21:23].tolist() == list(range(22, 24))


# ---------------------------------------------------------------------------
# Reset sequence (official LIBERO protocol)
# ---------------------------------------------------------------------------


def test_reset_follows_official_sequence() -> None:
    fake = FakeRobosuiteEnv()
    env = _make_env(fake, num_steps_wait=3)
    state = env.reset(episode_idx=2)

    kinds = [c[0] for c in fake.calls]
    # seed -> reset -> set_init_state -> num_steps_wait dummy steps
    assert kinds == ["seed", "reset", "set_init_state"] + ["step"] * 3

    assert fake.calls[0] == ("seed", 42)
    init_state_idx_2 = env._init_states[2]
    assert fake.calls[2] == ("set_init_state", init_state_idx_2)
    for call in fake.calls[3:]:
        action = call[1]
        assert action.shape == (7,)
        assert action.dtype == np.float32
        assert np.allclose(action, np.zeros(7))

    assert env._elapsed_steps == 0
    np.testing.assert_allclose(state, expected_state_vector())


def test_reset_episode_idx_out_of_range() -> None:
    env = _make_env(FakeRobosuiteEnv(), num_init_states=5)
    with pytest.raises(IndexError, match="out of range"):
        env.reset(episode_idx=5)


# ---------------------------------------------------------------------------
# step() gymnasium-style mapping
# ---------------------------------------------------------------------------


def test_step_success_terminates() -> None:
    fake = FakeRobosuiteEnv(done=False, success=True)
    env = _make_env(fake)
    state, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert env._elapsed_steps == 1
    assert state.shape == (STATE_DIM,)


def test_step_timeout_terminates_without_success() -> None:
    fake = FakeRobosuiteEnv(done=True, success=False)
    env = _make_env(fake)
    _, _, terminated, _, info = env.step(np.zeros(7))
    assert terminated is True
    assert info["is_success"] is False


def test_step_continues_mid_episode() -> None:
    fake = FakeRobosuiteEnv(done=False, success=False)
    env = _make_env(fake)
    _, _, terminated, truncated, _ = env.step(np.zeros(7))
    assert terminated is False
    assert truncated is False


def test_close_calls_underlying_env() -> None:
    fake = FakeRobosuiteEnv()
    env = _make_env(fake)
    env.close()
    assert fake.calls[-1] == ("close",)


# ---------------------------------------------------------------------------
# Suite constants (must match the official LIBERO benchmark protocol)
# ---------------------------------------------------------------------------


def test_max_steps_match_official_protocol() -> None:
    # Values used by the LeRobot/OpenVLA LIBERO evaluation protocol.
    assert SUITE_MAX_STEPS == {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }


def test_benchmark_names_are_official_keys() -> None:
    assert SUITE_BENCHMARK_NAMES["libero_spatial"] == "libero_spatial"
    assert SUITE_BENCHMARK_NAMES["libero_object"] == "libero_object"
    assert SUITE_BENCHMARK_NAMES["libero_goal"] == "libero_goal"
    assert SUITE_BENCHMARK_NAMES["libero_10"] == "libero_10"
    assert SUITE_BENCHMARK_NAMES["libero_90"] == "libero_90"
