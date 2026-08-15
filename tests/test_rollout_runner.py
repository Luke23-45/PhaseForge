"""Tests for the rollout runner: strict-metric episode classification."""

from __future__ import annotations

import json
import math

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
)
from phaseforge.outputs_writer.episodes import read_episode_records
from tests.rollout_helpers import FakeAdapter, make_bank


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
    """Policy hook wrapper that advances the controller's step counter."""

    def __init__(self, controller):
        self.controller = controller
        self.t = 0

    def reset(self) -> None:
        self.controller.reset()
        self.t = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action = self.controller.act(state, self.t)
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
    from phaseforge.evaluations.rollout.scripted_controller import (
        ScriptedLiftController,
    )
    from tests.rollout_helpers import lift_state_spec

    return ScriptedLiftController(lift_state_spec())


class TestEpisodeOutcomes:
    def test_scripted_policy_success_recorded(self, tmp_path) -> None:
        evaluator = _evaluator(
            tmp_path,
            FakeAdapter(horizon=500),
            policy=_Ticker(_lift_controller()),
            horizon=500,
        )
        results = evaluator.run()
        assert results["eval/rollout/successes"] == 3
        assert results["eval/rollout/valid_episodes"] == 3
        assert math.isnan(results["eval/action_mse"])
        rows = read_episode_records(tmp_path)
        assert len(rows) == 3
        assert all(r["valid_episode"] and r["success"] for r in rows)
        assert [r["episode_index"] for r in rows] == [0, 1, 2]

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
        """Case 0 (marker eef z=5) → NaN → policy failure; others succeed."""
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
        assert results["eval/rollout/successes"] == 2
        assert results["eval/rollout/policy_failures"] == 1
        rows = read_episode_records(tmp_path)
        by_index = {r["episode_index"]: r for r in rows}
        assert by_index[0]["failure_category"] == FAILURE_POLICY_INVALID_ACTION
        assert by_index[1]["success"] and by_index[2]["success"]

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
        assert summary["metrics"]["eval/rollout/successes"] == 3

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
