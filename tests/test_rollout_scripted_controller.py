"""Tests for the scripted Lift controller (phase machine + closed-loop solve)."""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.rollout.scripted_controller import (
    GRASP_HOLD_STEPS,
    LIFT_GRASP_Z_OFFSET,
    POSITION_SCALE,
    ScriptedCanController,
    ScriptedLiftConfig,
    ScriptedLiftController,
)
from tests.rollout_helpers import (
    SUCCESS_Z,
    FakeLiftSim,
    lift_state_spec,
    state_from_parts,
)


def _controller() -> ScriptedLiftController:
    return ScriptedLiftController(lift_state_spec())


def _eef_of(action: np.ndarray) -> np.ndarray:
    return action[0:3] * POSITION_SCALE


class TestPhases:
    def test_approach_moves_toward_above_cube(self) -> None:
        ctrl = _controller()
        state = state_from_parts(np.array([0.05, 0.05, 0.87]), np.array([-0.1, 0.02, 0.8]))
        action = ctrl.act(state, t=0)
        # The approach target is above the cube and should require positive z.
        assert _eef_of(action)[2] > 0
        assert action[6] == -1.0

    def test_descend_target(self) -> None:
        ctrl = _controller()
        state = state_from_parts(np.array([-0.1, 0.02, 1.08]), np.array([-0.1, 0.02, 0.8]))
        action = ctrl.act(state, t=0)
        assert _eef_of(action)[2] < 0

    def test_grasp_closes_gripper_after_descend(self) -> None:
        ctrl = _controller()
        sim = FakeLiftSim()
        # Walk the controller through APPROACH→DESCEND→GRASP via the
        # kinematic FakeLiftSim (which matches the controller's expected
        # step sizes closely enough to converge on the descend target).
        saw_grasp = False
        for _ in range(200):
            action = ctrl.act(sim.state, sim.t)
            sim.step(action)
            if action[6] == 1.0:
                saw_grasp = True
            if sim.success:
                break
        assert saw_grasp, "controller never entered the GRASP phase"

    def test_lift_after_grasp_hold(self) -> None:
        ctrl = _controller()
        sim = FakeLiftSim()
        saw_close = saw_lift = False
        for _ in range(500):
            action = ctrl.act(sim.state, sim.t)
            sim.step(action)
            if action[6] == 1.0 and _eef_of(action)[2] > 0:
                saw_lift = True
            if action[6] == 1.0:
                saw_close = True
            if sim.success:
                break
        assert saw_close and saw_lift, f"saw_close={saw_close} saw_lift={saw_lift}"
        assert sim.success

    def test_success_holds_closed_for_lift(self) -> None:
        ctrl = _controller()
        state = state_from_parts(np.array([0.0, 0.0, 0.95]), np.array([0.0, 0.0, SUCCESS_Z + 0.01]))
        action = ctrl.act(state, t=100)
        assert action[6] == 1.0

    def test_stall_watchdog_abandons(self) -> None:
        cfg = ScriptedLiftConfig(stall_steps=3, stall_progress=0.005)
        ctrl = ScriptedLiftController(lift_state_spec(), config=cfg)
        state = state_from_parts(np.array([0.0, 0.0, 0.80]), np.array([0.5, 0.5, 0.8]))
        for t in range(10):
            action = ctrl.act(state, t)
            assert np.all(np.isfinite(action))
            assert np.all(np.abs(action) <= 1.0)
        action = ctrl.act(state, 10)
        assert np.allclose(action[0:3], 0.0)
        assert ctrl.stalled_from_phase == "APPROACH"


class TestClosedLoop:
    def test_solves_all_seeded_starts(self) -> None:
        for seed in range(5):
            sim = FakeLiftSim(np.random.default_rng(seed))
            ctrl = _controller()
            ok = False
            for t in range(500):
                sim.step(ctrl.act(sim.state, t))
                if sim.success:
                    ok = True
                    break
            assert ok, f"controller failed from rng seed {seed}"
            assert sim.cube[2] > SUCCESS_Z

    def test_real_env_guard_does_not_claim_unverified_grasp(self) -> None:
        class NoGraspEnv:
            cube = object()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return False

        ctrl = ScriptedLiftController(lift_state_spec(), env=NoGraspEnv())
        approach_state = state_from_parts(
            np.array([0.0, 0.0, 0.8 + 0.12]),
            np.array([0.0, 0.0, 0.8]),
        )
        ctrl.act(approach_state, 0)
        grasp_state = state_from_parts(
            np.array([0.0, 0.0, 0.8 + LIFT_GRASP_Z_OFFSET]),
            np.array([0.0, 0.0, 0.8]),
        )
        for t in range(1, GRASP_HOLD_STEPS + 2):
            action = ctrl.act(grasp_state, t)
            assert action[6] == 1.0
        assert ctrl.phase_name == "GRASP"

    def test_can_resolves_indexed_robosuite_object_for_grasp_guard(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        can_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class CanObject:
            contact_geoms = ["can_geom"]

        class IndexedCanEnv:
            objects = [object(), object(), object(), CanObject()]
            object_id = 3
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                assert object_geoms == ["can_geom"]
                return False

        ctrl = ScriptedCanController(can_spec, env=IndexedCanEnv())
        assert ctrl._native_grasp_status() is False
