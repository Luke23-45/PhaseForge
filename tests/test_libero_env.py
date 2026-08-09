"""CPU-only tests for the state-only LIBERO env wrapper.

Runs without the real ``libero``/``robosuite`` packages: the wrapper's
``__init__`` (which imports LIBERO) is bypassed via ``object.__new__`` and
a fake robosuite-style env is injected. This verifies the state vector
construction, the official reset sequence (seed -> reset -> set_init_state
-> dummy steps), and the gymnasium-style step mapping.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import phaseforge.evaluations.envs.libero_env as le
from phaseforge.evaluations.envs.libero_env import (
    SUITE_BENCHMARK_NAMES,
    SUITE_MAX_STEPS,
    StateOnlyLiberoEnv,
)

# State layout: [proprio 23 | objects k_slots*7 | mask k_slots] => 151.
# The fixture uses 2 objects of 16 slots to exercise zero-padding.
STATE_DIM = 151
OBJECT_K_SLOTS = 16
OBJECT_DIM = 7
OBJECT_NAMES = ("world_bowl", "world_cup")
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
    for name in OBJECT_NAMES:
        obs[f"{name}_pos"] = np.arange(v, v + 3, dtype=np.float64)
        v += 3
        obs[f"{name}_quat"] = np.arange(v, v + 4, dtype=np.float64)
        v += 4
    return obs


def expected_state_vector() -> np.ndarray:
    """The 151-dim concatenation in the training-data key order."""
    parts = []
    v = 1.0
    for dim in KEY_DIMS.values():
        parts.append(np.arange(v, v + dim, dtype=np.float32))
        v += dim
    # Object block: proprio ends at value 24, so bowl = 24..30, cup = 31..37.
    obj_block = np.zeros(OBJECT_K_SLOTS * OBJECT_DIM, dtype=np.float32)
    obj_block[: len(OBJECT_NAMES) * OBJECT_DIM] = np.arange(
        24.0, 24.0 + len(OBJECT_NAMES) * OBJECT_DIM
    )
    parts.append(obj_block)
    mask = np.zeros(OBJECT_K_SLOTS, dtype=np.float32)
    mask[: len(OBJECT_NAMES)] = 1.0
    parts.append(mask)
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
    object_names: list[str] | None = None,
) -> StateOnlyLiberoEnv:
    """Construct a StateOnlyLiberoEnv without running its LIBERO __init__."""
    obj = object.__new__(StateOnlyLiberoEnv)
    obj._env = env
    obj.seed = seed
    obj.num_steps_wait = num_steps_wait
    obj._elapsed_steps = 0
    obj._success_via_done = None
    obj._init_states = [np.zeros(STATE_DIM, dtype=np.float64) for _ in range(num_init_states)]
    obj._task = type("Task", (), {"language": "fake task"})()
    obj._object_names = object_names if object_names is not None else list(OBJECT_NAMES)
    obj._object_k_slots = OBJECT_K_SLOTS
    obj._object_dim = OBJECT_DIM
    obj._object_include_mask = True
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


def test_object_state_appended_with_zero_pad_and_mask() -> None:
    """Objects go into slot 0..2 (world_bowl=24..30, world_cup=31..37),
    slots 2..15 are zero-padded, and the mask marks filled slots."""
    env = _make_env(FakeRobosuiteEnv())
    state = env._extract_state(make_fake_obs())
    obj_start, obj_end = 23, 23 + OBJECT_K_SLOTS * OBJECT_DIM
    mask_start = obj_end
    assert state.shape == (151,)
    np.testing.assert_allclose(
        state[obj_start:obj_start + 7], np.arange(24.0, 31.0), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        state[obj_start + 7:obj_start + 14], np.arange(31.0, 38.0), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        state[obj_start + 14:obj_end], np.zeros(14 * 7), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        state[mask_start:], [1.0, 1.0] + [0.0] * 14, rtol=0, atol=0
    )


def test_no_object_state_when_not_configured() -> None:
    """Without object names the state vector stays at the 23-dim proprio layout."""
    env = _make_env(FakeRobosuiteEnv(), object_names=[])
    state = env._extract_state(make_fake_obs())
    assert state.shape == (23,)
    np.testing.assert_allclose(state, expected_state_vector()[:23], rtol=0, atol=0)


def test_train_decode_matches_eval_extract_state() -> None:
    """B7 parity lock: ObjectIndex.decode (ingest/train side) must produce
    exactly the object block the eval env reads from live robosuite obs.

    Both sides consume the same census-built table, so slot order,
    zero-padding and mask must agree row-for-row for every joint type
    (free + hinge here). If either side ever reorders/slices differently,
    this test fails before any rollout can be trusted.
    """
    from phaseforge.data.libero.object_state import (
        ObjectEntry,
        ObjectIndex,
        TaskObjectTable,
    )

    task_name = "fake_task"
    table = TaskObjectTable(
        task_name=task_name,
        nq=14,
        objects=[
            ObjectEntry(
                name="world_bowl", joint_type="free", qpos_start=0, qpos_len=7
            ),
            ObjectEntry(
                name="world_cup",
                joint_type="hinge",
                qpos_start=7,
                qpos_len=1,
                anchor_world=np.array([0.3, 0.4, 0.5]),
                axis_world=np.array([0.0, 0.0, 1.0]),
                rest_pos=np.array([0.6, 0.7, 0.5]),
                rest_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        ],
    )
    index = ObjectIndex(
        tasks={task_name: table},
        k_slots=OBJECT_K_SLOTS,
        dim_per_object=OBJECT_DIM,
        include_mask=True,
    )

    qpos = np.array(
        [
            [0.5, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0, 1.2],   # t0: bowl pose, hinge angle
            [-0.3, 0.8, 0.4, 0.0, 0.0, 0.0, 1.0, -0.6],  # t1
        ],
        dtype=np.float64,
    )
    # decode() reads states[:, :nq]; pad to the table's full nq width.
    states = np.zeros((2, table.nq), dtype=np.float64)
    states[:, : qpos.shape[1]] = qpos
    block, mask = index.decode(task_name, states)
    assert block.shape == (2, OBJECT_K_SLOTS * OBJECT_DIM)
    assert mask.shape == (2, OBJECT_K_SLOTS)

    env = _make_env(FakeRobosuiteEnv(), object_names=list(OBJECT_NAMES))

    for t in (0, 1):
        obs = make_fake_obs()
        # Robosuite would report exactly these values for the same qpos.
        for i, name in enumerate(OBJECT_NAMES):
            start = i * OBJECT_DIM
            obs[f"{name}_pos"] = block[t, start : start + 3]
            obs[f"{name}_quat"] = block[t, start + 3 : start + 7]

        eval_state = env._extract_state(obs)
        train_state = np.concatenate(
            [expected_state_vector()[:23], block[t], mask[t]]
        ).astype(np.float32)
        np.testing.assert_allclose(eval_state, train_state, rtol=1e-6, atol=1e-6)


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


def test_step_probes_done_success_once_then_skips_check() -> None:
    """LIBERO semantics (done == _check_success): the first step probes the
    equivalence, and afterwards check_success() is NOT called per step —
    the predicate runs exactly once per control step (inside step())."""
    fake = FakeRobosuiteEnv(done=True, success=True)
    env = _make_env(fake)
    env.step(np.zeros(7))
    env.step(np.zeros(7))

    assert fake.calls.count(("check_success",)) == 1  # probe only
    assert env._success_via_done is True


def test_step_mismatch_falls_back_to_check_success(caplog) -> None:
    """Non-LIBERO env (done != predicate): probe detects the mismatch once,
    warns, and calls check_success() on every subsequent step."""
    fake = FakeRobosuiteEnv(done=True, success=False)
    env = _make_env(fake)
    with caplog.at_level(logging.WARNING, logger="phaseforge.evaluations.envs.libero_env"):
        _, _, terminated, _, info = env.step(np.zeros(7))
        env.step(np.zeros(7))

    assert env._success_via_done is False
    assert fake.calls.count(("check_success",)) == 3  # probe + 2 fallback steps
    assert terminated is True
    assert info["is_success"] is False
    assert "falling back" in caplog.text


def test_close_calls_underlying_env() -> None:
    fake = FakeRobosuiteEnv()
    env = _make_env(fake)
    env.close()
    assert fake.calls[-1] == ("close",)


# ---------------------------------------------------------------------------
# Observable pruning (_disable_unused_observables)
# ---------------------------------------------------------------------------


class FakeObservable:
    """Minimal robosuite-style observable: modality + set_enabled/set_active."""

    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_active(self, enabled: bool) -> None:
        self.enabled = enabled


class FakeObservableEnv:
    def __init__(self, with_observables: bool = True) -> None:
        self._observables = (
            {
                "agentview_image": FakeObservable("image"),
                "eye_in_hand_image": FakeObservable("image"),
                "robot0_joint_pos": FakeObservable("state"),
                "robot0_joint_vel": FakeObservable("state"),
                "robot0_eef_pos": FakeObservable("state"),
                "robot0_eef_quat": FakeObservable("state"),
                "robot0_gripper_qpos": FakeObservable("state"),
                "world_bowl_pos": FakeObservable("object"),
                "world_bowl_quat": FakeObservable("object"),
                "world_cup_pos": FakeObservable("object"),
                "world_cup_quat": FakeObservable("object"),
                "world_plate_pos": FakeObservable("object"),
            }
            if with_observables
            else None
        )


class WrappedObservableEnv:
    """LIBERO-style composition wrapper: real env lives at self._env."""

    def __init__(self) -> None:
        self._env = FakeObservableEnv()


def _enabled_names(observables: dict) -> list[str]:
    return sorted(name for name, obs in observables.items() if obs.enabled)


def test_pruning_keeps_only_state_vector_observables() -> None:
    env = FakeObservableEnv()
    le._disable_unused_observables(env)
    # Exactly the five 23-DoF keys survive; images and object sensors go.
    assert _enabled_names(env._observables) == [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ]


def test_pruning_keeps_extra_object_observables() -> None:
    """With the object-state channel enabled, the listed object observables
    (and only those) survive pruning alongside the proprio keys."""
    env = FakeObservableEnv()
    le._disable_unused_observables(
        env, keep_extra=("world_bowl_pos", "world_bowl_quat", "world_cup_pos", "world_cup_quat")
    )
    assert _enabled_names(env._observables) == [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
        "world_bowl_pos",
        "world_bowl_quat",
        "world_cup_pos",
        "world_cup_quat",
    ]


def test_pruning_finds_wrapped_env() -> None:
    """OffScreenRenderEnv stores the robosuite env at _env — the fix for
    the 'NO-OP' warnings seen in the Colab eval logs."""
    env = WrappedObservableEnv()
    le._disable_unused_observables(env)
    assert _enabled_names(env._env._observables) == [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ]


def test_pruning_warns_on_missing_observables(caplog) -> None:
    env = FakeObservableEnv(with_observables=False)
    with caplog.at_level(logging.WARNING, logger="phaseforge.evaluations.envs.libero_env"):
        le._disable_unused_observables(env)
    assert "Observable pruning NO-OP" in caplog.text


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
