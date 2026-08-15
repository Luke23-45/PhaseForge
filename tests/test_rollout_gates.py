"""Tests for the rollout validation gates (all with fakes, no robosuite)."""

from __future__ import annotations

import numpy as np

from phaseforge.evaluations.envs.errors import InfrastructureError
from phaseforge.evaluations.rollout.gates import (
    gate_action_contract,
    gate_demo_replay,
    gate_env_schema,
    gate_random_noop_sanity,
    gate_scripted_controller,
)
from tests.rollout_helpers import (
    FakeAdapter,
    FakeLiftSim,
    lift_state_spec,
    make_bank,
)


class TestEnvSchema:
    def test_passes(self) -> None:
        result = gate_env_schema(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            expected_state_dim=19,
            expected_action_dim=7,
        )
        assert result.status == "PASS", result.detail

    def test_wrong_state_dim_fails(self) -> None:
        result = gate_env_schema(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            expected_state_dim=23,
            expected_action_dim=7,
        )
        assert result.status == "FAIL"


class TestDemoReplay:
    def test_skipped_when_hdf5_missing(self, tmp_path) -> None:
        result = gate_demo_replay(FakeAdapter(), tmp_path / "missing.hdf5", num_demos=1)  # type: ignore[arg-type]
        assert result.status == "SKIPPED"
        assert "not present" in result.detail

    def test_passes_on_matching_demo(self, tmp_path) -> None:
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        path = tmp_path / "lift.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail

    def test_passes_on_v15_equal_length_demo(self, tmp_path) -> None:
        """The v1.5 PH files store T states and T actions.

        The published v1.5 states are post-action states, so state[0] is
        already the result of action[0]. The gate must replay actions[1:]
        against states[1:].
        """
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        path = tmp_path / "lift_v15.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states[1:])
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail
        assert result.metrics["compared"] == 4

    def test_fails_on_divergent_demo(self, tmp_path) -> None:
        import h5py

        adapter = FakeAdapter(horizon=500)  # type: ignore[arg-type]
        states, actions = _linear_demo(adapter, steps=5)
        states[2, 1] += 0.01  # perturb the stored post-step state
        path = tmp_path / "lift.hdf5"
        with h5py.File(path, "w") as h5:
            data = h5.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=actions)

        result = gate_demo_replay(adapter, path, num_demos=1, tolerance=1e-6)  # type: ignore[arg-type]
        assert result.status == "FAIL", result.detail
        assert result.metrics["mismatches"] >= 1


def _linear_demo(adapter: FakeAdapter, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A fake demo the kinematic sim reproduces exactly.

    World-state layout: ``[time, eef(3), cube(3)]``. Actions move the eef
    linearly; the cube never moves (gripper stays open).
    """
    states = []
    adapter.sim.reset()
    for i in range(steps):
        state = np.asarray(adapter.sim.get_state(), dtype=np.float64)
        states.append(state)
        action = np.zeros(7, dtype=np.float64)
        action[0:3] = np.array([0.01, 0.02, -0.01]) / 0.05
        adapter.sim.step(action)
    states.append(np.asarray(adapter.sim.get_state(), dtype=np.float64))
    actions = np.tile(np.array([0.2, 0.4, -0.2, 0.0, 0.0, 0.0, 1.0]), (steps, 1))
    return np.stack(states), actions


class TestActionContract:
    def test_passes(self) -> None:
        result = gate_action_contract(FakeAdapter(), trials=5)  # type: ignore[arg-type]
        assert result.status == "PASS", result.detail


class TestScriptedController:
    def test_solves_bank_at_threshold(self) -> None:
        result = gate_scripted_controller(
            FakeAdapter(horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            lift_state_spec(),
            threshold=1.0,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["successes"] == 3
        assert result.metrics["infra_failures"] == 0

    def test_fails_on_infra(self) -> None:
        adapter = FakeAdapter(FakeLiftSim(), fail_step_with=InfrastructureError("boom"))
        result = gate_scripted_controller(
            adapter,  # type: ignore[arg-type]
            make_bank(3),
            lift_state_spec(),
            threshold=1.0,
        )
        assert result.status == "FAIL"
        assert result.metrics["infra_failures"] > 0


class TestRandomNoop:
    def test_no_successes_no_infra(self) -> None:
        # Tight grasp window: random actions can never align well enough.
        sim = FakeLiftSim(grasp_z_window=0.0005)
        result = gate_random_noop_sanity(
            FakeAdapter(sim, horizon=500),  # type: ignore[arg-type]
            make_bank(3),
            num_cases=3,
            horizon=200,
            max_success_rate=0.05,
        )
        assert result.status == "PASS", result.detail
        assert result.metrics["successes"] == 0
        assert result.metrics["infra_failures"] == 0


class TestCheckpointSmoke:
    def test_skipped_without_checkpoint(self) -> None:
        from phaseforge.evaluations.rollout.gates import gate_checkpoint_smoke

        class _Cfg:
            train = {"stage1_ckpt_path": ""}
            project = {"device": "cpu"}

        result = gate_checkpoint_smoke(_Cfg(), FakeAdapter(), make_bank(3), num_episodes=2)  # type: ignore[arg-type]
        assert result.status == "SKIPPED"
