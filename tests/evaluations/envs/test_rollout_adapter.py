"""Tests for the robosuite state adapter (observation extraction + action contract)."""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    PolicyInvalidActionError,
    StateSchemaError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
from tests.rollout_helpers import make_meta


class TestStateSpec:
    def test_dims_and_index(self) -> None:
        spec = StateSpec(keys=("a", "b"), dims=(3, 4))
        assert spec.dim == 7
        assert spec.index_of("b") == (3, 7)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(keys=("a",), dims=(3, 4))

    def test_unknown_key(self) -> None:
        spec = StateSpec(keys=("a",), dims=(3,))
        with pytest.raises(KeyError):
            spec.index_of("nope")


class TestExtractState:
    def setup_method(self) -> None:
        self.spec = StateSpec(keys=("robot0_eef_pos", "object"), dims=(3, 10))

    def test_concatenation_and_order(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        obs = {
            "robot0_eef_pos": np.ones(3),
            "object-state": np.arange(10, dtype=np.float32),
        }
        state = adapter.extract_state(obs)
        assert state.shape == (13,)
        assert np.allclose(state[:3], 1.0)
        assert np.allclose(state[3:], np.arange(10))

    def test_object_alias(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        state = adapter.extract_state({"robot0_eef_pos": np.zeros(3), "object": np.zeros(10)})
        assert state.shape == (13,)

    def test_missing_key(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        with pytest.raises(StateSchemaError, match="robot0_eef_pos"):
            adapter.extract_state({"object": np.zeros(10)})

    def test_wrong_dim(self) -> None:
        adapter = _adapter_with_spec(self.spec)
        with pytest.raises(StateSchemaError, match="dimension"):
            adapter.extract_state({"robot0_eef_pos": np.zeros(5), "object": np.zeros(10)})


class TestValidateAction:
    def setup_method(self) -> None:
        self.spec = StateSpec(keys=("robot0_eef_pos",), dims=(3,))
        self.adapter = _adapter_with_spec(self.spec, action_dim=7)

    def test_valid_action_passthrough(self) -> None:
        out = self.adapter.validate_action(np.ones(7) * 0.5)
        assert out.dtype == np.float64

    def test_batch_shape_accepted(self) -> None:
        out = self.adapter.validate_action(np.zeros((1, 7)))
        assert out.shape == (7,)

    def test_nan_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="non-finite"):
            self.adapter.validate_action(np.array([np.nan] * 7))

    def test_inf_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="non-finite"):
            self.adapter.validate_action(np.array([np.inf] * 7))

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="outside"):
            self.adapter.validate_action(np.ones(7) * 1.5)

    def test_wrong_shape_rejected(self) -> None:
        with pytest.raises(PolicyInvalidActionError, match="shape"):
            self.adapter.validate_action(np.zeros(6))

    def test_tolerance_extension(self) -> None:
        out = self.adapter.validate_action(np.ones(7) * 1.00005, tolerance=1e-4)
        assert out.shape == (7,)


class TestConstruction:
    def test_action_dim_mismatch_fails_closed(self, monkeypatch) -> None:
        import sys

        from phaseforge.evaluations.envs import robosuite_adapter as mod

        class _FakeRobosuite:
            def make(self, *args, **kwargs):
                class _Env:
                    action_spec = (np.zeros((5,)), np.zeros((5,)))
                    horizon = 500

                return _Env()

        monkeypatch.setitem(sys.modules, "robosuite", _FakeRobosuite())
        monkeypatch.setattr(mod, "robosuite", _FakeRobosuite(), raising=False)
        with pytest.raises(EnvParityError, match="Action dimension mismatch"):
            RobosuiteStateAdapter(
                make_meta(),
                StateSpec(keys=("a",), dims=(3,)),
                action_dim=7,
            )

    def test_missing_robosuite(self, monkeypatch) -> None:
        import builtins

        original = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "robosuite":
                raise ImportError("no robosuite")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(EnvParityError, match="not installed"):
            RobosuiteStateAdapter(make_meta(), StateSpec(keys=("a",), dims=(3,)), action_dim=3)


def _adapter_with_spec(spec: StateSpec, action_dim: int = 7) -> RobosuiteStateAdapter:
    """Build an adapter without touching robosuite (bypass __init__)."""
    adapter = object.__new__(RobosuiteStateAdapter)
    adapter.meta = make_meta()
    adapter.state_spec = spec
    adapter.action_dim = action_dim
    adapter.action_low = -1.0
    adapter.action_high = 1.0
    # Matches the __init__ default; the bypass skips attribute initialization.
    adapter.action_tolerance = 1e-4
    return adapter


class _FakePartController:
    def __init__(self, joint_pos: np.ndarray) -> None:
        self.joint_pos = np.asarray(joint_pos, dtype=float)
        self.calls: list[tuple] = []
        self.initial_joint: np.ndarray | None = None

    def update(self, force: bool = False) -> None:
        self.calls.append(("update", bool(force)))

    def update_initial_joints(self, joints: np.ndarray) -> None:
        self.calls.append(("update_initial_joints", np.asarray(joints).copy()))
        self.initial_joint = np.asarray(joints).copy()


class _FakeComposite:
    def __init__(self, parts: dict) -> None:
        self.part_controllers = parts


class _FakeRobot:
    def __init__(self, parts: dict) -> None:
        self.composite_controller = _FakeComposite(parts)


class _FakeData:
    def __init__(self) -> None:
        self.qacc_warmstart = np.full(9, 17.1)


class _FakeSim:
    def __init__(self) -> None:
        self.data = _FakeData()
        self.forward_calls = 0

    def set_state_from_flattened(self, states) -> None:
        self.last_states = np.asarray(states)

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeEnv:
    def __init__(self, parts: dict) -> None:
        self.sim = _FakeSim()
        self.robots = [_FakeRobot(parts)]
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def _get_observations(self, force_update: bool = False) -> dict:
        self.last_force_update = force_update
        return {"robot0_eef_pos": np.zeros(3), "object": np.zeros(10)}


class TestCanonicalReset:
    """Phase 8c: hidden-state canonicalization at episode restore."""

    def test_forced_kwargs_pin_hard_reset_false(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import _FORCED_ENV_KWARGS

        assert _FORCED_ENV_KWARGS["hard_reset"] is False
        assert _FORCED_ENV_KWARGS["has_renderer"] is False
        assert _FORCED_ENV_KWARGS["has_offscreen_renderer"] is False
        assert _FORCED_ENV_KWARGS["use_camera_obs"] is False

    def test_reset_to_canonicalizes_solver_and_controller_state(self) -> None:
        arm = _FakePartController(np.array([0.1, 0.2, 0.3]))
        gripper = _FakePartController(np.array([0.0, 0.0]))
        env = _FakeEnv({"right": arm, "right_gripper": gripper})
        adapter = _adapter_with_spec(StateSpec(keys=("robot0_eef_pos", "object"), dims=(3, 10)))
        adapter.env = env

        states = np.zeros(1 + 20 + 20)
        adapter.reset_to(states)

        # Solver warm-start is zeroed and the sim is forwarded.
        assert np.all(env.sim.data.qacc_warmstart == 0.0)
        assert env.sim.forward_calls >= 1
        # Every part controller is force-refreshed BEFORE its (refreshed)
        # joint_pos is read for initial-joint canonicalization.
        for part in (arm, gripper):
            kinds = [c[0] for c in part.calls]
            assert kinds.index("update") < kinds.index("update_initial_joints")
            assert part.calls[0] == ("update", True)
            np.testing.assert_array_equal(part.initial_joint, part.joint_pos)
        # Observations are force-refreshed so no cached sensor value from
        # the previous episode leaks into the policy's first input.
        assert env.last_force_update is True

    def test_construction_is_rng_position_independent(self, monkeypatch) -> None:
        """Two constructions at different global-RNG positions build the
        same environment under the same deterministic construction seed,
        and the caller's RNG stream is left untouched."""
        import sys

        from phaseforge.evaluations.envs import robosuite_adapter as mod

        draws_seen: list[float] = []

        class _FakeRobosuite:
            def make(self, name, **kwargs):
                class _Env:
                    action_spec = (np.zeros((3,)), np.zeros((3,)))

                draws_seen.append(float(np.random.uniform()))
                return _Env()

        fake = _FakeRobosuite()
        monkeypatch.setitem(sys.modules, "robosuite", fake)
        monkeypatch.setattr(mod, "robosuite", fake, raising=False)

        np.random.seed(5)
        np.random.uniform(size=11)  # shift the caller's RNG position
        before = np.random.get_state()[1].copy()
        RobosuiteStateAdapter(make_meta(), StateSpec(keys=("a",), dims=(3,)), action_dim=3)
        after = np.random.get_state()[1].copy()
        np.testing.assert_array_equal(before, after)  # caller stream untouched

        np.random.seed(77)
        np.random.uniform(size=3)  # different position entirely
        RobosuiteStateAdapter(make_meta(), StateSpec(keys=("a",), dims=(3,)), action_dim=3)
        assert draws_seen[0] == draws_seen[1]  # identical construction draw
