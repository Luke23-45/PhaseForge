"""CPU-only simulated tests for the rollout evaluator.

These tests run WITHOUT the real ``libero``/``robosuite`` packages and
WITHOUT a GPU. They exercise the full evaluation logic — cache-based
normalizer loading, state normalization before inference, raw action
passthrough, per-task/per-suite success aggregation, multi-suite
averaging, and graceful error messages — against fake environments.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from phaseforge.data.ingestion.cache_manager import CacheManager
from phaseforge.evaluations.runners import rollout_evaluator as re_mod
from phaseforge.evaluations.runners.rollout_evaluator import RolloutEvaluator

STATE_DIM = 151
ACTION_DIM = 7

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "phaseforge" / "config")


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def make_cfg(
    suites: list[str] | None = None,
    num_episodes: int = 3,
    seed: int = 42,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {"state_dim": STATE_DIM, "action_dim": ACTION_DIM, "batch_size": 1},
            "project": {"seed": seed},
            "eval": {
                "mode": "rollout",
                "environment": {
                    "suites": suites or ["libero_90"],
                    "num_steps_wait": 2,
                    "num_workers": 1,  # force serial path with fake envs
                },
                "evaluation": {"num_episodes_per_task": num_episodes},
            },
        }
    )


class RecordingModel(torch.nn.Module):
    """Fake policy: records every normalized state it sees, returns a fixed action."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_states: list[torch.Tensor] = []
        self.constant_action = torch.full((1, ACTION_DIM), 5.0)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        self.seen_states.append(state.detach().cpu().clone())
        return self.constant_action


class FakeStateEnv:
    """Task 0 always succeeds; any other task never does.

    Every episode terminates after a single step so the loop stays fast
    even though SUITE_MAX_STEPS is in the hundreds.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        seed: int,
        num_steps_wait: int = 10,
        render_observations: bool = False,
        hard_reset: bool = True,
        object_state_cfg: object | None = None,
    ) -> None:
        self.suite_name = suite_name
        self.task_id = task_id
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.hard_reset = hard_reset
        self.task_description = f"task_{suite_name}_{task_id}"
        self.num_init_states = 50
        self.closed = False
        self.actions_received: list[np.ndarray] = []

    def reset(self, episode_idx: int = 0) -> np.ndarray:
        return np.full(STATE_DIM, 10.0, dtype=np.float32)

    def step(self, action: np.ndarray):
        self.actions_received.append(action)
        success = self.task_id == 0
        return (
            np.full(STATE_DIM, 10.0, dtype=np.float32),
            0.0,
            True,
            False,
            {"is_success": bool(success)},
        )

    def close(self) -> None:
        self.closed = True


class FakeSuite:
    def __init__(self, n_tasks: int) -> None:
        self.n_tasks = n_tasks

    def get_task(self, task_id: int):
        return types.SimpleNamespace(language=f"task_{task_id}")

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        return [np.zeros(STATE_DIM) for _ in range(50)]


class FakeBenchmark:
    """Task counts must match the protocol's SUITE_N_TASKS — the evaluator
    asserts the installed benchmark's n_tasks against the protocol (E9)."""

    def get_benchmark_dict(self) -> dict[str, callable]:
        return {
            name: (lambda name=name: FakeSuite(re_mod.SUITE_N_TASKS[name]))
            for name in SUITE_NAMES
        }


SUITE_NAMES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]


@pytest.fixture
def fake_libero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``libero.libero`` module so no real LIBERO is needed."""
    parent = types.ModuleType("libero")
    child = types.ModuleType("libero.libero")
    child.benchmark = FakeBenchmark()
    monkeypatch.setitem(sys.modules, "libero", parent)
    monkeypatch.setitem(sys.modules, "libero.libero", child)


@pytest.fixture
def block_libero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate LIBERO being uninstalled (None in sys.modules -> ImportError)."""
    monkeypatch.setitem(sys.modules, "libero", None)
    monkeypatch.setitem(sys.modules, "libero.libero", None)


def _seed_cache(tmp_path: Path, data_cfg) -> str:
    """Write a minimal processed cache with a known train-frozen normalizer."""
    cache_root = tmp_path / "cache"
    mgr = CacheManager(cache_root)
    config_hash = CacheManager.compute_hash(data_cfg)
    mgr.save(
        config_hash=config_hash,
        trajectories=[],
        norm_stats={
            "mean": torch.zeros(STATE_DIM),
            "std": torch.full((STATE_DIM,), 2.0),
        },
        splits={"train": [], "val": [], "eval": []},
    )
    return config_hash


def _make_evaluator(
    tmp_path: Path,
    cfg: OmegaConf,
    model: torch.nn.Module | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> RolloutEvaluator:
    if monkeypatch is not None:
        monkeypatch.setattr(re_mod, "processed_cache_root", lambda: tmp_path / "cache")
    _seed_cache(tmp_path, cfg.data)
    return RolloutEvaluator(cfg=cfg, model=model or RecordingModel(), device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Normalizer loading
# ---------------------------------------------------------------------------


def test_load_normalizer_from_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg()
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)
    assert torch.allclose(evaluator.normalizer.mean, torch.zeros(STATE_DIM))
    assert torch.allclose(evaluator.normalizer.std, torch.full((STATE_DIM,), 2.0))


def test_load_normalizer_missing_cache_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg()
    monkeypatch.setattr(re_mod, "processed_cache_root", lambda: tmp_path / "empty")
    with pytest.raises(RuntimeError, match="No cached dataset found"):
        RolloutEvaluator(cfg=cfg, model=RecordingModel(), device=torch.device("cpu"))


def test_missing_libero_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_libero
) -> None:
    cfg = make_cfg()
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError, match="libero package not installed"):
        evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=2)


# ---------------------------------------------------------------------------
# Rollout loop, normalization, and aggregation
# ---------------------------------------------------------------------------


def test_evaluate_suite_aggregates_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    cfg = make_cfg(num_episodes=3)
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)
    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", FakeStateEnv)

    results = evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=3)

    # 10 tasks x 3 episodes; task 0 always succeeds, the rest never do.
    assert results["eval/success_rate/libero_spatial"] == pytest.approx(0.1)
    assert results["eval/total_episodes/libero_spatial"] == 30
    assert results["eval/total_successes/libero_spatial"] == 3
    per_task = results["eval/per_task/libero_spatial"]
    assert set(per_task) == {str(t) for t in range(10)}  # numeric task-id keys
    assert per_task["0"]["description"] == "task_libero_spatial_0"
    assert per_task["0"]["success_rate"] == pytest.approx(1.0)
    assert per_task["0"]["successes"] == 3
    assert per_task["1"]["success_rate"] == pytest.approx(0.0)
    assert per_task["1"]["successes"] == 0


def test_state_normalized_before_inference_and_action_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    """Env state (10.0) must arrive normalized (mean 0, std 2 -> 5.0) to the
    model, and the model's raw action must go to the env untouched."""
    cfg = make_cfg(num_episodes=2)
    model = RecordingModel()
    evaluator = _make_evaluator(tmp_path, cfg, model=model, monkeypatch=monkeypatch)
    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", FakeStateEnv)

    evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=2)

    assert len(model.seen_states) == 20  # 10 tasks x 2 episodes
    for seen in model.seen_states:
        assert seen.shape == (1, STATE_DIM)
        assert torch.allclose(seen, torch.full((1, STATE_DIM), 5.0), atol=1e-5)


def test_actions_passed_to_env_are_raw_float64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    cfg = make_cfg(num_episodes=1)
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    received: list[np.ndarray] = []

    class RecordingFakeEnv(FakeStateEnv):
        def step(self, action: np.ndarray):
            received.append(action)
            return super().step(action)

    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", RecordingFakeEnv)
    evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=1)

    assert len(received) == 10  # one step per episode, 10 tasks
    for action in received:
        assert action.shape == (ACTION_DIM,)
        assert action.dtype == np.float64
        assert np.allclose(action, 5.0)


def test_envs_are_closed_after_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    cfg = make_cfg(num_episodes=1)
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    created: list[FakeStateEnv] = []

    def factory(
        suite_name: str,
        task_id: int,
        seed: int,
        num_steps_wait: int = 10,
        render_observations: bool = False,
        hard_reset: bool = True,
        object_state_cfg: object | None = None,
    ) -> FakeStateEnv:
        env = FakeStateEnv(suite_name, task_id, seed, num_steps_wait, render_observations)
        created.append(env)
        return env

    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", factory)
    evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=1)

    assert len(created) == 10  # one env per task
    assert all(env.closed for env in created)


def test_run_multi_suite_averages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(suites=["libero_spatial", "libero_object"], num_episodes=2)
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    def fake_suite(self, suite_name: str, num_episodes_per_task: int) -> dict:
        rate = 1.0 if suite_name == "libero_spatial" else 0.25
        return {
            f"eval/success_rate/{suite_name}": rate,
            f"eval/per_task/{suite_name}": {"t": {"success_rate": rate}},
            f"eval/total_episodes/{suite_name}": 10,
            f"eval/total_successes/{suite_name}": int(10 * rate),
        }

    evaluator.evaluate_suite = types.MethodType(fake_suite, evaluator)
    results = evaluator.run()

    assert results["eval/success_rate"] == pytest.approx(0.625)
    assert results["eval/success_rate/libero_spatial"] == pytest.approx(1.0)
    assert results["eval/success_rate/libero_object"] == pytest.approx(0.25)
    assert results["eval/seed"] == 42
    assert results["eval/num_episodes_per_task"] == 2
    assert results["eval/suites"] == ["libero_spatial", "libero_object"]


def test_rollout_mode_without_environment_keys_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting eval.mode=rollout without the rollout.yaml groups must not crash."""
    cfg = make_cfg()
    del cfg.eval["environment"]
    del cfg.eval["evaluation"]
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)
    assert evaluator.suites == ["libero_90"]  # Decision 2 (A2): ID core only
    assert evaluator.num_episodes_per_task == 50
    assert evaluator.num_steps_wait == 10
    assert evaluator.render_observations is False
    assert evaluator.hard_reset is True
    assert evaluator.num_workers == 0  # auto: one worker per logical CPU


def test_unsupported_suite_names_rejected_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suites without a full protocol definition (e.g. the ``libero_long``
    legacy alias) must be rejected up front — not KeyError mid-evaluation."""
    cfg = make_cfg(suites=["libero_long"])
    with pytest.raises(ValueError, match="Unknown or unsupported suite name"):
        _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    bogus = make_cfg(suites=["not_a_suite"])
    with pytest.raises(ValueError, match="Unknown or unsupported suite name"):
        _make_evaluator(tmp_path, bogus, monkeypatch=monkeypatch)

    good = make_cfg(suites=["libero_10", "libero_90"])
    evaluator = _make_evaluator(tmp_path, good, monkeypatch=monkeypatch)
    assert evaluator.suites == ["libero_10", "libero_90"]


def test_env_factory_receives_hard_reset_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    """hard_reset: false in the config must reach the env constructor."""
    cfg = make_cfg(num_episodes=1)
    cfg.eval.environment.hard_reset = False
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    received: list[bool] = []

    def factory(
        suite_name: str,
        task_id: int,
        seed: int,
        num_steps_wait: int = 10,
        render_observations: bool = False,
        hard_reset: bool = True,
        object_state_cfg: object | None = None,
    ) -> FakeStateEnv:
        received.append(hard_reset)
        return FakeStateEnv(suite_name, task_id, seed, num_steps_wait)

    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", factory)
    evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=1)

    assert received == [False] * 10


# ---------------------------------------------------------------------------
# Parallel episode sharding and aggregation
# ---------------------------------------------------------------------------


def test_split_episode_shards_round_robin_and_cover_all() -> None:
    shards = re_mod.split_episode_shards(10, 3)
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(i for s in shards for i in s) == list(range(10))


def test_split_episode_shards_more_workers_than_episodes() -> None:
    assert re_mod.split_episode_shards(3, 8) == [[0], [1], [2]]


def test_split_episode_shards_rejects_zero_workers() -> None:
    with pytest.raises(ValueError):
        re_mod.split_episode_shards(10, 0)


def test_resolve_num_workers_caps_to_episodes() -> None:
    assert re_mod._resolve_num_workers(8, 3) == 3
    assert re_mod._resolve_num_workers(2, 50) == 2
    assert re_mod._resolve_num_workers(4, 0) == 1
    assert re_mod._resolve_num_workers(1, 50) == 1


def test_resolve_num_workers_auto_uses_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(re_mod.os, "cpu_count", lambda: 4)
    assert re_mod._resolve_num_workers(0, 50) == 4
    assert re_mod._resolve_num_workers(-1, 50) == 4
    assert re_mod._resolve_num_workers(0, 2) == 2  # capped by episodes
    assert re_mod._resolve_num_workers(0, 0) == 1  # nothing to parallelize

    monkeypatch.setattr(re_mod.os, "cpu_count", lambda: None)
    assert re_mod._resolve_num_workers(0, 50) == 1  # unknown CPU count


def test_merge_worker_results_matches_serial_aggregation() -> None:
    """Two workers split 50 episodes; task 0 succeeds only in worker 0's
    half — the merged result must equal the serial 25/50. This is exactly
    the chain evaluate_suite's parallel branch runs: merge -> finalize.
    Task IDs are numeric keys (E9); the merged dict carries the description
    per task."""
    worker_results = [
        {
            "worker_idx": 0,
            "task_results": {
                "0": {"successes": 25, "episodes": 25},
                "1": {"successes": 0, "episodes": 25},
            },
        },
        {
            "worker_idx": 1,
            "task_results": {
                "0": {"successes": 0, "episodes": 25},
                "1": {"successes": 0, "episodes": 25},
            },
        },
    ]
    merged = re_mod._merge_worker_results(worker_results)
    assert merged == {
        0: {"description": "", "successes": 25},
        1: {"description": "", "successes": 0},
    }

    results = re_mod._finalize_suite_results(
        "libero_spatial", merged, 50, expected_tasks=2
    )
    assert results["eval/success_rate/libero_spatial"] == pytest.approx(0.25)
    assert results["eval/total_episodes/libero_spatial"] == 100
    assert results["eval/total_successes/libero_spatial"] == 25
    assert results["eval/per_task/libero_spatial"]["0"]["successes"] == 25
    assert results["eval/per_task/libero_spatial"]["0"]["success_rate"] == pytest.approx(0.5)


def test_render_observations_flag_plumbed_to_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_libero
) -> None:
    cfg = make_cfg(num_episodes=1)
    cfg.eval.environment.render_observations = True
    evaluator = _make_evaluator(tmp_path, cfg, monkeypatch=monkeypatch)

    kwargs_seen: list[dict] = []

    class RecordingFakeEnv(FakeStateEnv):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs_seen.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(re_mod, "StateOnlyLiberoEnv", RecordingFakeEnv)
    evaluator.evaluate_suite("libero_spatial", num_episodes_per_task=1)

    assert len(kwargs_seen) == 10  # one env per task
    assert all(kw.get("render_observations") is True for kw in kwargs_seen)


# ---------------------------------------------------------------------------
# Hydra config wiring
# ---------------------------------------------------------------------------


def test_eval_config_groups_compose() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        rollout_cfg = compose(config_name="main", overrides=["eval=rollout"])
        assert rollout_cfg.eval.mode == "rollout"
        assert list(rollout_cfg.eval.environment.suites) == [
            "libero_90",
            "libero_10",
        ]
        assert rollout_cfg.eval.environment.num_steps_wait == 10
        assert rollout_cfg.eval.environment.render_observations is False
        assert rollout_cfg.eval.environment.num_workers == 0  # auto
        assert rollout_cfg.eval.evaluation.num_episodes_per_task == 50
        assert rollout_cfg.eval.environment.object_state.enabled is True
        assert rollout_cfg.eval.environment.object_state.k_slots == 16
        assert rollout_cfg.data.state_dim == 151

        default_cfg = compose(config_name="main")
        assert default_cfg.eval.mode == "offline"


def test_bc_model_builds_and_acts_on_state_vector() -> None:
    from hydra import compose, initialize_config_dir

    from phaseforge.utils.registry import build_model

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main", overrides=["models=baselines/bc"])
    model = build_model(cfg)
    model.eval()
    action = model.get_action(torch.randn(1, STATE_DIM))
    assert action.shape == (1, ACTION_DIM)
