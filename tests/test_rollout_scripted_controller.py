"""Tests for the scripted Lift controller (phase machine + closed-loop solve)."""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.rollout.scripted_controller import (
    GRASP_HOLD_STEPS,
    GRIPPER_OPEN,
    LIFT_GRASP_Z_OFFSET,
    POSITION_SCALE,
    SQUARE_GRASP_LOSS_CONFIRM_STEPS,
    SQUARE_LIFT_ACTION_LIMIT,
    SQUARE_LIFT_SETTLE_STEPS,
    ScriptedCanController,
    ScriptedControllerConfig,
    ScriptedLiftConfig,
    ScriptedLiftController,
    ScriptedSquareController,
    ScriptedToolHangController,
    ScriptedTransportController,
    _Phase,
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
    def test_place_settling_is_not_aborted_by_progress_watchdog(self) -> None:
        ctrl = ScriptedCanController(
            lift_state_spec(),
            config=ScriptedControllerConfig(stall_steps=1, stall_progress=1.0),
        )
        ctrl._phase = _Phase.PLACE
        ctrl._place_started_at = 0
        eef = np.array([0.0, 0.0, 1.0])
        target = np.array([0.5, 0.5, 0.8])

        for t in range(5):
            action = ctrl._track(target, 1.0, eef, t)
            assert np.all(np.isfinite(action))

        assert ctrl.phase_name == "PLACE"
        assert ctrl.stalled_from_phase is None

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

    def test_square_resolves_indexed_robosuite_nut_for_grasp_guard(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]
            name = "SquareNut"

        class IndexedSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.4]]),
                            "body_xpos": np.array([[0.2, 0.3, 0.85]]),
                        },
                    )(),
                },
            )()
            peg1_body_id = 0
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                assert object_geoms == ["square_nut_geom"]
                return False

        ctrl = ScriptedSquareController(square_spec, env=IndexedSquareEnv())
        assert ctrl._native_grasp_status() is False
        state = np.zeros(square_spec.dim, dtype=np.float32)
        assert np.allclose(ctrl.grasp_pos(state), [0.2, 0.3, 0.4])
        state[0:3] = [0.1, 0.2, 0.9]
        state[9 + 7 : 9 + 10] = [0.3, 0.4, 0.8]
        assert np.allclose(
            ctrl.placement_target(state), [0.0, 0.1, 0.96], atol=1e-6
        )

    def test_square_refreshes_held_placement_from_current_body_offset(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]
            name = "SquareNut"

        class HeldSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            peg1_body_id = 1
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.4]]),
                            "body_xpos": np.array(
                                [[0.0, 0.0, 0.0], [0.2, 0.3, 0.85]]
                            ),
                        },
                    )(),
                },
            )()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return True

        ctrl = ScriptedSquareController(square_spec, env=HeldSquareEnv())
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.2, 1.1]
        state[9 + 7 : 9 + 10] = [0.1, 0.2, 0.9]
        ctrl._phase = _Phase.TRANSPORT
        ctrl._placement_snapshot = np.array([99.0, 99.0, 99.0])

        ctrl.act(state, t=0)

        # Peg target [0.2, 0.3, 0.86] plus the current eef-to-body offset
        # [-0.1, 0.0, 0.2].
        assert np.allclose(ctrl._placement_snapshot, [0.1, 0.3, 1.06])

        ready_state = state.copy()
        ready_state[0:3] = [0.1, 0.3, 0.96]
        ready_state[9 + 7 : 9 + 10] = [0.2, 0.3, 0.86]
        assert ctrl._placement_release_ready(ready_state)

        not_ready_state = ready_state.copy()
        not_ready_state[9 + 7] = 0.24
        assert not ctrl._placement_release_ready(not_ready_state)

    def test_square_debounces_transient_native_grasp_loss(self) -> None:
        """A one-step contact flicker must not trigger a re-approach."""
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]

        class DroppedSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            peg1_body_id = 1
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.84]]),
                            "body_xpos": np.array(
                                [[0.0, 0.0, 0.0], [0.2, 0.3, 0.85]]
                            ),
                        },
                    )(),
                },
            )()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return False

        ctrl = ScriptedSquareController(square_spec, env=DroppedSquareEnv())
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 1.0]
        state[9 + 7 : 9 + 10] = [0.1, 0.2, 0.83]
        ctrl._phase = _Phase.LIFT
        ctrl._placement_snapshot = np.array([99.0, 99.0, 99.0])

        action = ctrl.act(state, t=37)

        assert ctrl.phase_name == "LIFT"
        assert action[6] == 1.0

    def test_square_reapproaches_after_confirmed_native_grasp_loss(self) -> None:
        """A persistent drop must not transport using a stale target."""
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]

        class DroppedSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            peg1_body_id = 1
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.84]]),
                            "body_xpos": np.array(
                                [[0.0, 0.0, 0.0], [0.2, 0.3, 0.85]]
                            ),
                        },
                    )(),
                },
            )()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return False

        ctrl = ScriptedSquareController(square_spec, env=DroppedSquareEnv())
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 1.0]
        state[9 + 7 : 9 + 10] = [0.1, 0.2, 0.83]
        ctrl._phase = _Phase.LIFT
        ctrl._placement_snapshot = np.array([99.0, 99.0, 99.0])

        for t in range(SQUARE_GRASP_LOSS_CONFIRM_STEPS):
            action = ctrl.act(state, t=37 + t)

        assert ctrl.phase_name == "APPROACH"
        assert ctrl._placement_snapshot is None
        assert ctrl._grasp_started_at is None
        assert action[6] == GRIPPER_OPEN

    def test_square_reapproaches_if_contact_is_lost_during_place(self) -> None:
        """A lost pre-release contact must be recovered, not held forever."""
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]

        class ReleasedSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            peg1_body_id = 1
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.84]]),
                            "body_xpos": np.array(
                                [[0.0, 0.0, 0.0], [0.2, 0.3, 0.85]]
                            ),
                        },
                    )(),
                },
            )()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return False

        ctrl = ScriptedSquareController(square_spec, env=ReleasedSquareEnv())
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 1.0]
        state[9 + 7 : 9 + 10] = [0.1, 0.2, 0.83]
        ctrl._phase = _Phase.PLACE
        ctrl._place_started_at = 0
        ctrl._placement_snapshot = np.array([0.2, 0.3, 0.86])

        action = ctrl.act(state, t=10)

        assert ctrl.phase_name == "APPROACH"
        assert ctrl._placement_snapshot is None
        assert ctrl._grasp_started_at is None
        assert action[6] == GRIPPER_OPEN

    def test_square_lift_uses_small_vertical_increment(self) -> None:
        """Square's thin nut receives a bounded first lift command."""
        ctrl = ScriptedSquareController(lift_state_spec())
        ctrl._phase = _Phase.LIFT
        action = ctrl._normalized_action(
            np.array([0.0, 0.0, 1.0]),
            gripper=1.0,
        )
        assert np.isclose(action[2], SQUARE_LIFT_ACTION_LIMIT)

    def test_square_aligns_gripper_yaw_to_rotated_nut(self) -> None:
        """A rotated SquareNut must not be approached with zero yaw action."""
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )
        ctrl = ScriptedSquareController(square_spec)
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 1.0]
        state[3:7] = [0.0, 0.0, 0.0, 1.0]
        state[9 + 7 : 9 + 10] = [0.2, 0.2, 0.83]
        yaw = 1.0
        state[9 + 10 : 9 + 14] = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

        action = ctrl.act(state, t=0)

        assert abs(action[5]) > 0.1
        assert np.all(np.abs(action) <= 1.0 + 1e-6)

    def test_square_settles_before_first_lift_command(self) -> None:
        """Square holds the confirmed grasp before applying upward motion."""
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        square_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 14),
        )

        class SquareNut:
            contact_geoms = ["square_nut_geom"]

        class HeldSquareEnv:
            nuts = [SquareNut()]
            nut_id = 0
            object_site_ids = [0]
            peg1_body_id = 1
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array([[0.2, 0.3, 0.84]]),
                            "body_xpos": np.array(
                                [[0.0, 0.0, 0.0], [0.2, 0.3, 0.85]]
                            ),
                        },
                    )(),
                },
            )()
            robots = [type("Robot", (), {"gripper": object()})()]

            @staticmethod
            def _check_grasp(*, gripper, object_geoms) -> bool:  # noqa: ARG004
                return True

        ctrl = ScriptedSquareController(square_spec, env=HeldSquareEnv())
        state = np.zeros(square_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.0, 0.84]
        state[9 + 7 : 9 + 10] = [0.0, 0.0, 0.83]
        ctrl._phase = _Phase.GRASP
        ctrl._grasp_started_at = 0

        action = ctrl.act(state, t=ctrl.config.grasp_hold_steps)

        assert ctrl.phase_name == "LIFT"
        assert ctrl._square_lift_started_at == ctrl.config.grasp_hold_steps
        assert action[6] == 1.0
        assert np.isclose(action[2], 0.0)
        assert SQUARE_LIFT_SETTLE_STEPS > 0

    def test_tool_hang_placement_preserves_eef_to_tool_offset(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        tool_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 44),
        )

        class ToolHangEnv:
            obj_site_id = {"frame_hang_site": 0, "tool_hole1_center": 1}
            obj_body_id = {"tool": 0}
            sim = type(
                "Sim",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "site_xpos": np.array(
                                [[0.2, 0.3, 1.0], [0.1, 0.3, 0.9]]
                            ),
                            "body_xpos": np.array([[0.0, 0.3, 0.9]]),
                        },
                    )(),
                },
            )()

        ctrl = ScriptedToolHangController(tool_spec, env=ToolHangEnv())
        state = np.zeros(tool_spec.dim, dtype=np.float32)
        state[0:3] = [0.0, 0.2, 1.1]
        state[9 + 35 : 9 + 38] = [0.0, 0.3, 0.9]
        # Desired tool body target is hook - (hole - body) = (0.1, 0.3, 1.0);
        # add the current eef-to-body offset (0, -0.1, 0.2).
        assert np.allclose(
            ctrl.placement_target(state), [0.1, 0.2, 1.2], atol=1e-6
        )

    def test_transport_uses_full_two_arm_state_and_action_contract(self) -> None:
        from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

        transport_spec = StateSpec(
            keys=(
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
                "object",
            ),
            dims=(3, 4, 2, 3, 4, 2, 41),
        )
        ctrl = ScriptedTransportController(transport_spec)
        state = np.zeros(transport_spec.dim, dtype=np.float32)
        object_start, _ = transport_spec.index_of("object")
        state[object_start : object_start + 3] = [0.1, -0.2, 0.82]
        state[object_start + 7 : object_start + 10] = [0.2, 0.2, 0.82]
        state[object_start + 14 : object_start + 17] = [0.1, -0.2, 0.95]
        state[object_start + 21 : object_start + 24] = [-0.2, 0.2, 0.8]
        state[object_start + 24 : object_start + 27] = [0.2, 0.3, 0.8]

        action = ctrl.act(state, 0)
        assert action.shape == (14,)
        assert np.all(np.isfinite(action))
        assert np.all(np.abs(action) <= 1.0)
        assert action[6] == GRIPPER_OPEN
        assert action[13] == GRIPPER_OPEN
        assert ctrl.phase_name == "LID_APPROACH"
