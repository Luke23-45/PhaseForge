"""Parametrized tests for the per-task scripted controllers.

The five benchmark tasks each get their own scripted oracle subclass
(:class:`ScriptedLiftController`, :class:`ScriptedCanController`,
:class:`ScriptedSquareController`, :class:`ScriptedToolHangController`,
:class:`ScriptedTransportController`). The controllers share a common
base phase machine (:class:`ScriptedController`) but specialize the
success predicate and (for the four placement tasks) the receptacle
target. These tests verify each controller can solve its kinematic fake
sim from a representative initial state.
"""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.evaluations.rollout.scripted_controller import (
    ScriptedCanController,
    ScriptedController,
    ScriptedLiftController,
    ScriptedSquareController,
    ScriptedToolHangController,
    ScriptedTransportController,
)
from tests.rollout_helpers import FakeLiftSim, lift_state_spec, state_from_parts

_CONTROLLERS: dict[str, type[ScriptedController]] = {
    "Lift": ScriptedLiftController,
    "Can": ScriptedCanController,
    "Square": ScriptedSquareController,
    "ToolHang": ScriptedToolHangController,
    "Transport": ScriptedTransportController,
}


@pytest.mark.parametrize("task", sorted(_CONTROLLERS))
def test_controller_subclass_returns_proper_action_shape(task: str) -> None:
    """Every controller emits a 7-dim normalized action on every step."""
    controller_cls = _CONTROLLERS[task]
    ctrl = controller_cls(lift_state_spec())
    sim = FakeLiftSim(task=task)
    state = sim.state
    action = ctrl.act(state, 0)
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))
    assert float(action.min()) >= -1.0 - 1e-6
    assert float(action.max()) <= 1.0 + 1e-6


@pytest.mark.parametrize("task", sorted(_CONTROLLERS))
def test_controller_solves_kinematic_sim(task: str) -> None:
    """The per-task scripted controller solves the kinematic fake sim."""
    controller_cls = _CONTROLLERS[task]
    rng = np.random.default_rng(0)
    sim = FakeLiftSim(rng=rng, task=task)
    ctrl = controller_cls(lift_state_spec())
    for t in range(500):
        action = ctrl.act(sim.state, t)
        sim.step(action)
        if sim.success:
            break
    assert sim.success, (
        f"Scripted{controller_cls.__name__} failed to solve its kinematic sim within 500 steps."
    )


@pytest.mark.parametrize("task", sorted(_CONTROLLERS))
def test_controller_resets_cleanly(task: str) -> None:
    """Calling reset() returns the controller to APPROACH phase."""
    controller_cls = _CONTROLLERS[task]
    ctrl = controller_cls(lift_state_spec())
    sim = FakeLiftSim(task=task)
    # Walk through several phases.
    for t in range(100):
        ctrl.act(sim.state, t)
        sim.step(np.zeros(7))
    ctrl.reset()
    assert ctrl._phase.value == 1  # APPROACH = 1
    assert ctrl._approach_done is False
    assert ctrl._grasp_started_at is None
    assert ctrl._stall_since is None


@pytest.mark.parametrize("task", sorted(_CONTROLLERS))
def test_controller_invalid_state_key_raises(task: str) -> None:
    """Constructing with an incompatible state spec raises ValueError."""
    from phaseforge.evaluations.envs.robosuite_adapter import StateSpec

    bad_spec = StateSpec(keys=("only_eef",), dims=(3,))
    controller_cls = _CONTROLLERS[task]
    with pytest.raises((KeyError, ValueError)):
        controller_cls(bad_spec)


@pytest.mark.parametrize("task", sorted(_CONTROLLERS))
def test_controller_constructs_with_real_task_state_spec(task: str) -> None:
    """Each controller must construct cleanly with the task's canonical state spec.

    Regression guard: Transport's published state uses the full two-arm
    low-dimensional schema (object: 41, plus both arm pose/gripper keys).
    The controller must construct against that schema without silently
    treating Transport as a single-arm observation.
    """
    from phaseforge.evaluations.envs.robosuite_adapter import StateSpec
    from phaseforge.evaluations.envs.task_registry import TaskSpec

    spec = TaskSpec.from_protocol(task)
    real_spec = StateSpec(keys=spec.state_keys, dims=spec.state_dims)
    controller_cls = _CONTROLLERS[task]
    # Must construct without raising.
    controller_cls(real_spec)


def test_placement_target_distinct_per_task() -> None:
    """The four placement tasks each return a non-None placement target;
    the Lift controller returns None (no placement phase)."""
    assert ScriptedLiftController(lift_state_spec()).placement_target(np.zeros(19)) is None
    for cls in (
        ScriptedCanController,
        ScriptedSquareController,
        ScriptedToolHangController,
        ScriptedTransportController,
    ):
        ctrl = cls(lift_state_spec())
        target = ctrl.placement_target(state_from_parts(np.zeros(3), np.zeros(3)))
        assert target is not None
        assert target.shape == (3,)
        assert target[2] > 0.0  # placement is above the table
