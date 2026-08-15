"""Tests for the scripted Lift controller (phase machine + closed-loop solve)."""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.rollout.scripted_controller import (
    POSITION_SCALE,
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
        # target = cube + (0,0,0.28) = (-0.1, 0.02, 1.08); delta positive in z
        assert _eef_of(action)[2] > 0
        assert action[6] == 1.0

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
            if action[6] == -1.0:
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
            if action[6] == -1.0 and _eef_of(action)[2] > 0:
                saw_lift = True
            if action[6] == -1.0:
                saw_close = True
            if sim.success:
                break
        assert saw_close and saw_lift, f"saw_close={saw_close} saw_lift={saw_lift}"
        assert sim.success

    def test_success_holds_closed_for_lift(self) -> None:
        ctrl = _controller()
        state = state_from_parts(np.array([0.0, 0.0, 0.95]), np.array([0.0, 0.0, SUCCESS_Z + 0.01]))
        action = ctrl.act(state, t=100)
        assert action[6] == -1.0

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
