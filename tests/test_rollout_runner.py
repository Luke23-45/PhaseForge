"""Tests for the rollout runner: strict-metric episode classification."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from phaseforge.data.common.normalizer import FrozenNormalizer
from phaseforge.evaluations.envs.errors import InfrastructureError
from phaseforge.evaluations.rollout.runner import (
    FAILURE_POLICY_EXCEPTION,
    FAILURE_POLICY_INVALID_ACTION,
    FAILURE_TIMEOUT,
    RolloutEvaluator,
    RolloutRunInvalid,
    resolve_robosuite_requirement,
)
from phaseforge.outputs_writer.episodes import read_episode_records
from tests.rollout_helpers import FakeAdapter, make_bank


def test_task_source_robosuite_pin_overrides_eval_default() -> None:
    from omegaconf import DictConfig

    cfg = DictConfig(
        {
            "data": {"source": {"robosuite_requirement": "==1.5.0"}},
            "eval": {"env": {"robosuite_requirement": "==1.5.1"}},
        }
    )
    assert resolve_robosuite_requirement(cfg) == "==1.5.0"


def test_legacy_eval_robosuite_pin_is_fallback() -> None:
    from omegaconf import DictConfig

    cfg = DictConfig(
        {
            "data": {"source": {}},
            "eval": {"env": {"robosuite_requirement": "==1.5.1"}},
        }
    )
    assert resolve_robosuite_requirement(cfg) == "==1.5.1"


class _ZeroModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 7)


class _NanModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full((state.shape[0], 7), float("nan"))


class _RaisingModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom in the policy")


class _RangeModel(torch.nn.Module):
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full((state.shape[0], 7), 2.0)


class _Ticker:
    """Policy hook wrapper that advances an underlying policy's step counter."""

    def __init__(self, policy):
        self.policy = policy
        self.t = 0

    def reset(self) -> None:
        self.policy.reset()
        self.t = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action = self.policy.act(state, self.t)
        self.t += 1
        return action


class _MarkerPolicy:
    """Policy hook that delegates to a ticker unless ``state[2]`` is a marker."""

    def __init__(self, ticker: _Ticker):
        self._ticker = ticker

    def reset(self) -> None:
        self._ticker.reset()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state[2] > 4.0:
            return np.full(7, np.nan)
        return self._ticker(state)


def _evaluator(tmp_path, adapter, model=None, normalizer=None, horizon=500, policy=None):
    return RolloutEvaluator(
        cfg=None,  # type: ignore[arg-type]
        policy=policy,
        adapter=adapter,  # type: ignore[arg-type]
        bank=make_bank(),
        normalizer=normalizer or FrozenNormalizer(torch.zeros(19), torch.ones(19)),
        model=model,
        output_dir=tmp_path,
        run_id="testrun",
        model_name="phaseforge",
        training_seed=42,
        task="Lift",
        checkpoint_sha256="deadbeef",
        horizon=horizon,
    )


def _lift_controller():
    """Trivial valid-action policy used by the runner unit tests.

    Returns a small callable that emits a valid 7-dim neutral action every
    step. It is intentionally simple — these runner tests only need a
    policy that produces in-range actions, not one that solves the task.
    The fake simulator never reaches success with this policy (the cube
    does not move and the gripper does not close), so the affected tests
    assert ``successes == 0``.
    """
    return _ZeroPolicy()


class _ZeroPolicy:
    """A trivial valid-action policy (zero OSC delta, gripper stays open).

    Exposes ``act(state, t)`` and ``reset()`` so the ``_Ticker`` wrapper
    can drive it without changes.
    """

    def reset(self) -> None:  # noqa: D401 — runner calls reset() if present
        pass

    def act(self, state: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG002
        action = np.zeros(7, dtype=np.float64)
        action[-1] = -1.0  # gripper open (matches FakeAdapter convention)
        return action

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.act(state, 0)


class _SuccessAdapter(FakeAdapter):
    """Adapter wrapper that teleports the cube above the success threshold.

    Forces the simulator's success predicate (``cube.z > SUCCESS_Z``) to
    fire on the first step of every episode. Used to exercise the runner's
    success path without a hand-coded controller. The runner's success
    reporting is what we want to test; the geometric mechanism of *how*
    success is reached is the fake's responsibility, not the policy's.
    """

    def reset_to(self, states, *, xml=None, ep_meta=None) -> np.ndarray:
        super().reset_to(states, xml=xml, ep_meta=ep_meta)
        # Lift the cube above the success threshold (0.84 + epsilon).
        self.sim.cube = np.array([self.sim.cube[0], self.sim.cube[1], 0.90])
        return self.extract_state(self.sim.state)


class TestEpisodeOutcomes:
    def test_zero_policy_times_out(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_ZeroModel(), horizon=500)
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 0
        assert results["eval/rollout/valid_episodes"] == 3
        rows = read_episode_records(tmp_path)
        assert all(r["timed_out"] for r in rows)
        assert all(r["failure_category"] == FAILURE_TIMEOUT for r in rows)

    def test_policy_nan_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_NanModel(), horizon=500)
        results = evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["valid_episode"] for r in rows)
        assert all(r["success"] is False for r in rows)
        assert all(r["failure_category"] == FAILURE_POLICY_INVALID_ACTION for r in rows)
        assert results["eval/rollout/policy_failures"] == 3
        assert results["eval/rollout/success_rate"] == 0.0

    def test_policy_out_of_range_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(tmp_path, FakeAdapter(horizon=500), model=_RangeModel(), horizon=500)
        evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["failure_category"] == FAILURE_POLICY_INVALID_ACTION for r in rows)

    def test_policy_exception_is_valid_failure(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path, FakeAdapter(horizon=500), model=_RaisingModel(), horizon=500
        )
        evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(r["valid_episode"] for r in rows)
        assert all(r["failure_category"] == FAILURE_POLICY_EXCEPTION for r in rows)
        assert "boom in the policy" in rows[0]["exception"]

    def test_infrastructure_failure_invalidates_run(self, tmp_path) -> None:
        adapter = FakeAdapter(fail_step_with=InfrastructureError("sim exploded"))
        evaluator = _evaluator(tmp_path, adapter, model=_ZeroModel(), horizon=500)
        with pytest.raises(RolloutRunInvalid, match="infrastructure"):
            evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(not r["valid_episode"] for r in rows)
        assert all("sim exploded" in r["exception"] for r in rows)
        assert all("success" not in r for r in rows)
        assert all("failure_category" not in r for r in rows)

    def test_reset_failure_invalidates_run(self, tmp_path) -> None:
        adapter = FakeAdapter(fail_reset_with=InfrastructureError("no reset"))
        evaluator = _evaluator(tmp_path, adapter, model=_ZeroModel(), horizon=500)
        with pytest.raises(RolloutRunInvalid):
            evaluator.run()
        rows = read_episode_records(tmp_path)
        assert all(not r["valid_episode"] for r in rows)

    def test_mixed_outcomes(self, tmp_path) -> None:
        """Case 0 (marker eef z=5) → NaN → policy failure; others time out cleanly."""
        controller = _lift_controller()
        ticker = _Ticker(controller)
        policy = _MarkerPolicy(ticker)

        evaluator = _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            policy=policy,
            horizon=500,
        )
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 0
        assert results["eval/rollout/policy_failures"] == 1
        rows = read_episode_records(tmp_path)
        by_index = {r["episode_index"]: r for r in rows}
        assert by_index[0]["failure_category"] == FAILURE_POLICY_INVALID_ACTION
        assert by_index[1]["failure_category"] == FAILURE_TIMEOUT
        assert by_index[2]["failure_category"] == FAILURE_TIMEOUT

    def test_success_recorded_with_correct_metadata(self, tmp_path) -> None:
        """A successful episode records ``success=True``, ``termination_reason='success'``,
        ``timed_out=False``, and the evaluator stops at the success step rather than
        running to the horizon. Uses a ``_SuccessAdapter`` that forces the fake's
        success predicate to fire on the first step; no hand-coded controller is
        involved.
        """
        evaluator = _evaluator(
            tmp_path,
            _SuccessAdapter(horizon=500),
            policy=_Ticker(_lift_controller()),
            horizon=500,
        )
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 3
        assert results["eval/rollout/valid_episodes"] == 3
        assert results["eval/rollout/success_rate"] == 1.0
        rows = read_episode_records(tmp_path)
        assert len(rows) == 3
        for row in rows:
            assert row["valid_episode"] is True
            assert row["success"] is True
            assert row["timed_out"] is False
            assert row["termination_reason"] == "success"
            assert "failure_category" not in row
            # Stops at success (step 1), well before the 500-step horizon.
            assert row["steps"] <= 2

    def test_summary_json_written(self, tmp_path) -> None:
        _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            policy=_Ticker(_lift_controller()),
            horizon=500,
        ).run()
        summary = json.loads((tmp_path / "rollout_summary.json").read_text())
        assert summary["run_id"] == "testrun"
        assert summary["reset_bank"] == "testbank"
        assert "metrics" in summary
        assert summary["metrics"]["eval/rollout/valid_episodes"] == 3
        assert summary["metrics"]["eval/rollout/successes"] == 0

    def test_run_id_and_tag_in_rows(self, tmp_path) -> None:
        adapter = FakeAdapter(horizon=500)
        evaluator = RolloutEvaluator(
            cfg=None,  # type: ignore[arg-type]
            policy=_Ticker(_lift_controller()),
            adapter=adapter,  # type: ignore[arg-type]
            bank=make_bank(1),
            normalizer=None,
            model=None,
            output_dir=tmp_path,
            run_id="runabc",
            model_name="bc",
            training_seed=43,
            task="Lift",
            checkpoint_sha256="cafe",
            tag="robot_only",
            horizon=500,
        )
        evaluator.run()
        row = read_episode_records(tmp_path)[0]
        assert row["run_id"] == "runabc"
        assert row["model"] == "bc"
        assert row["training_seed"] == 43
        assert row["checkpoint_sha256"] == "cafe"
        assert row["tag"] == "robot_only"
        assert row["reset_seed"] == 2026
