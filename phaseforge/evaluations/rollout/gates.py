"""Validation gates for the rollout protocol (implementation plan §4.5).

The gates validate the simulator/adapter/bank/controller chain BEFORE real
policy rollouts, so a broken chain can never produce meaningless numbers:

* Gate 1 — environment schema self-test (needs robosuite).
* Gate 2 — demo replay against the raw HDF5 (skipped loudly when absent).
* Gate 3 — action-contract enforcement (in-range accepted, out-of-range
  rejected, gripper convention).
* Gate 4 — scripted state-oracle controller on the frozen bank.
* Gate 5 — random/no-op sanity (success must be 0, no infra failures).
* Gate 6 — checkpoint smoke run (skipped loudly when no checkpoint given).

Gates that need a missing input are SKIPPED with a loud warning (never a
silent pass); gates that need robosuite itself hard-fail. The gates exit 0
only when every required gate passed; skipped gates are reported as
warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phaseforge.evaluations.envs.env_metadata import (
    verify_environment_parity,
)
from phaseforge.evaluations.envs.errors import (
    EnvParityError,
    PolicyInvalidActionError,
    StateSchemaError,
)
from phaseforge.evaluations.envs.robosuite_adapter import (
    RobosuiteStateAdapter,
    StateSpec,
)
from phaseforge.evaluations.envs.task_registry import (
    TaskSpec,
    is_known_task,
)
from phaseforge.evaluations.rollout.reset_bank import ResetBank

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of one gate."""

    gate: str
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class GateFailure(RuntimeError):
    """A required gate failed; the protocol must not proceed."""


# ---------------------------------------------------------------------------
# Gate 1 — environment schema self-test
# ---------------------------------------------------------------------------


def gate_env_schema(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    expected_state_dim: int,
    expected_action_dim: int,
) -> GateResult:
    """Verify obs keys, dims, action spec, and a state restore + zero step."""
    problems: list[str] = []
    try:
        state = adapter.reset_to(
            bank.case(0).states, xml=bank.case(0).xml, ep_meta=bank.case(0).ep_meta
        )
        if state.shape != (expected_state_dim,):
            problems.append(f"restored state dim {state.shape[0]} != declared {expected_state_dim}")
        zero_action = np.zeros(expected_action_dim, dtype=np.float64)
        next_state, _done, success, _info = adapter.step(zero_action)
        if next_state.shape != (expected_state_dim,):
            problems.append(
                f"stepped state dim {next_state.shape[0]} != declared {expected_state_dim}"
            )
        if not isinstance(success, bool):
            problems.append(f"success predicate returned {type(success).__name__}")
    except (StateSchemaError, PolicyInvalidActionError, EnvParityError) as exc:
        problems.append(str(exc))
    except Exception as exc:  # noqa: BLE001 — gate must not crash the CLI
        problems.append(f"unexpected: {type(exc).__name__}: {exc}")

    status = "PASS" if not problems else "FAIL"
    return GateResult(
        gate="1_env_schema_selftest",
        status=status,
        detail="; ".join(problems)
        if problems
        else (
            f"state restore + zero step over ({expected_state_dim},) states and "
            f"({expected_action_dim},) actions OK"
        ),
        metrics={"state_dim": expected_state_dim, "action_dim": expected_action_dim},
    )


# ---------------------------------------------------------------------------
# Gate 2 — demo replay against the raw HDF5
# ---------------------------------------------------------------------------


def gate_demo_replay(
    adapter: RobosuiteStateAdapter,
    hdf5_path: Path,
    *,
    num_demos: int = 1,
    tolerance: float = 1e-3,
) -> GateResult:
    """Replay the first demo(s) against the raw HDF5 state trajectory.

    robomimic stores either ``T+1`` states with ``T`` actions (an initial state
    plus one post-action state per action) or ``T`` post-action states with
    ``T`` actions in the v1.5 files. In the latter representation the first
    action produced the stored initial state, so the check starts at
    ``states[0]`` and replays ``actions[1:]`` against ``states[1:]``. Both
    representations are valid; silently treating the equal-length v1.5 form
    as a malformed demo would incorrectly skip this gate.

    Skipped loudly when the raw HDF5 is absent.
    """
    import h5py

    if not hdf5_path.is_file():
        return GateResult(
            gate="2_demo_replay",
            status="SKIPPED",
            detail=(
                f"raw dataset {hdf5_path} not present — replay gate skipped "
                "(the evaluation machine must run it before real rollouts)"
            ),
        )
    try:
        mismatches = 0
        compared = 0
        checked = 0
        with h5py.File(hdf5_path, "r") as h5:
            data = h5["data"]
            demo_keys = sorted(k for k in data.keys() if k.startswith("demo_"))
            if not demo_keys:
                raise GateFailure("dataset has no demo_* trajectories")
            for demo_key in demo_keys[:num_demos]:
                demo = data[demo_key]
                if "states" not in demo:
                    return GateResult(
                        gate="2_demo_replay",
                        status="SKIPPED",
                        detail=(
                            f"{demo_key} has no per-step 'states' attribute — "
                            "replay cannot be validated"
                        ),
                    )
                states = np.asarray(demo["states"][:], dtype=np.float64)
                actions = np.asarray(demo["actions"][:], dtype=np.float64)
                if states.shape[0] == actions.shape[0] + 1:
                    replay_actions = actions
                    expected_states = states[1:]
                elif states.shape[0] == actions.shape[0] and states.shape[0] >= 2:
                    # The robosuite data-collection path records states after
                    # each action and removes the extra terminal state before
                    # writing the equal-length v1.5 HDF5. State[0] is thus
                    # already the result of action[0].
                    replay_actions = actions[1:]
                    expected_states = states[1:]
                else:
                    return GateResult(
                        gate="2_demo_replay",
                        status="SKIPPED",
                        detail=(
                            f"{demo_key} has incompatible trajectory lengths: "
                            f"states={states.shape[0]}, actions={actions.shape[0]}; "
                            "expected states=actions or states=actions+1"
                        ),
                    )
                adapter.reset_to(states[0], xml=None, ep_meta=None)
                for i in range(replay_actions.shape[0]):
                    adapter.step(replay_actions[i])
                    sim = np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float64)
                    if sim.shape != expected_states[i].shape:
                        mismatches += 1
                        break
                    compared += 1
                    if np.max(np.abs(sim[1:] - expected_states[i][1:])) > tolerance:
                        mismatches += 1
                checked += 1
    except GateFailure as exc:
        return GateResult(gate="2_demo_replay", status="SKIPPED", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="2_demo_replay",
            status="FAIL",
            detail=f"replay raised {type(exc).__name__}: {exc}",
        )

    if mismatches:
        return GateResult(
            gate="2_demo_replay",
            status="FAIL",
            detail=(
                f"{mismatches} of {compared} compared steps diverged from the "
                f"dataset (tolerance {tolerance}) — the adapter does not "
                "reproduce the collection simulator"
            ),
            metrics={"compared": compared, "mismatches": mismatches},
        )
    return GateResult(
        gate="2_demo_replay",
        status="PASS",
        detail=(f"{checked} demo(s), {compared} steps replayed within tolerance {tolerance}"),
        metrics={"compared": compared, "checked": checked},
    )


# ---------------------------------------------------------------------------
# Gate 3 — action contract
# ---------------------------------------------------------------------------


def gate_action_contract(
    adapter: RobosuiteStateAdapter,
    *,
    trials: int = 10,
    tolerance: float = 1e-4,
) -> GateResult:
    """In-range actions accepted; out-of-range/NaN rejected; gripper sign."""
    problems: list[str] = []
    rng = np.random.default_rng(0)

    for _ in range(trials):
        action = rng.uniform(-1.0, 1.0, size=adapter.action_dim)
        try:
            adapter.validate_action(action, tolerance=tolerance)
        except PolicyInvalidActionError as exc:
            problems.append(f"in-range action rejected: {exc}")
            break

    for bad in (
        np.full(adapter.action_dim, 1.5),
        np.full(adapter.action_dim, -1.5),
        np.array([np.nan] * adapter.action_dim),
        np.array([np.inf] * adapter.action_dim),
        np.zeros(adapter.action_dim + 1),
    ):
        try:
            adapter.validate_action(bad, tolerance=tolerance)
            problems.append(f"invalid action {np.asarray(bad).tolist()} was accepted")
        except PolicyInvalidActionError:
            pass

    if not problems:
        try:
            base_state = adapter.reset_to(
                np.asarray(adapter.env.sim.get_state().flatten(), dtype=np.float64),
                xml=None,
                ep_meta=None,
            )
            close_action = np.zeros(adapter.action_dim, dtype=np.float64)
            close_action[-1] = -1.0
            after_state, _done, _success, _info = adapter.step(close_action)
            # The gripper convention probe requires the
            # ``robot0_gripper_qpos`` key. Every robomimic v1.5 Panda task
            # exposes it, but skip the probe cleanly if a future schema
            # omits it rather than crashing the gate.
            try:
                start, _stop = adapter.state_spec.index_of("robot0_gripper_qpos")
            except KeyError:
                start = None
            if start is not None:
                delta = float(after_state[start + 1] - base_state[start + 1])
                if delta > 1e-4:
                    problems.append(
                        "gripper close action (-1) moved the finger gap the "
                        f"wrong way (delta {delta:.6g}) — action convention mismatch"
                    )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"gripper convention probe failed: {exc}")

    return GateResult(
        gate="3_action_contract",
        status="PASS" if not problems else "FAIL",
        detail="; ".join(problems)
        if problems
        else (
            "in-range accepted, NaN/Inf/out-of-range rejected, gripper -1 "
            "closes (dataset convention)"
        ),
    )


# ---------------------------------------------------------------------------
# Gate 4 — scripted controller on the bank
# ---------------------------------------------------------------------------


def gate_scripted_controller(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    state_spec: StateSpec,
    *,
    threshold: float = 1.0,
) -> GateResult:
    """The training-free controller must solve the frozen bank at
    ``threshold`` (default: all cases). Failures are task outcomes only.

    The scripted controller class is dispatched on ``bank.task`` via the
    task registry, so each of the five benchmark tasks gets its own
    oracle policy. Unknown tasks fail closed.
    """
    if not is_known_task(bank.task):
        return GateResult(
            gate="4_scripted_controller",
            status="FAIL",
            detail=(
                f"bank.task={bank.task!r} is not one of the five benchmark "
                f"tasks; cannot dispatch a scripted controller."
            ),
        )
    controller_cls = TaskSpec.from_protocol(bank.task).get_controller_class()

    successes = 0
    infra = 0
    timeouts = 0
    failures_detail: list[str] = []
    failure_phases: dict[str, int] = {}

    for case in bank.cases:
        # The oracle may read pinned simulator geometry (target-bin, peg,
        # hook, and transport-bin poses) but never images. Learned policies
        # remain restricted to the declared low-dimensional state vector.
        controller = controller_cls(state_spec, env=getattr(adapter, "env", None))
        try:
            state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
        except Exception as exc:  # noqa: BLE001
            infra += 1
            failures_detail.append(f"case {case.index}: reset failed: {exc}")
            continue
        ok = False
        for t in range(adapter.horizon):
            action = controller.act(state, t)
            try:
                state, _done, success, _info = adapter.step(action)
            except Exception as exc:  # noqa: BLE001
                infra += 1
                failures_detail.append(f"case {case.index}: step failed: {exc}")
                break
            if success:
                ok = True
                break
        if ok:
            successes += 1
        else:
            timeouts += 1
            phase = getattr(controller, "phase_name", "unknown")
            failure_phases[phase] = failure_phases.get(phase, 0) + 1
            failures_detail.append(f"case {case.index}: timed out in phase {phase}")

    rate = successes / len(bank.cases) if bank.cases else float("nan")
    passed = rate + 1e-9 >= threshold and infra == 0
    return GateResult(
        gate="4_scripted_controller",
        status="PASS" if passed else "FAIL",
        detail=(
            f"{successes}/{len(bank.cases)} cases solved ({rate:.3f}; "
            f"threshold {threshold}); {timeouts} timeouts; {infra} infra "
            + ("failures: " + "; ".join(failures_detail[:3]) if infra else "none")
        ),
        metrics={
            "successes": successes,
            "cases": len(bank.cases),
            "rate": rate,
            "timeouts": timeouts,
            "infra_failures": infra,
            "timeout_phases": failure_phases,
        },
    )


# ---------------------------------------------------------------------------
# Gate 5 — random / no-op sanity
# ---------------------------------------------------------------------------


def gate_random_noop_sanity(
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    num_cases: int,
    horizon: int,
    max_success_rate: float = 0.05,
) -> GateResult:
    """Neither no-op nor random actions may succeed or crash the simulator."""
    rng = np.random.default_rng(0)
    total_successes = 0
    total_infra = 0
    total_steps = 0
    for policy_name, policy in (
        ("noop", lambda: np.zeros(adapter.action_dim, dtype=np.float64)),
        (
            "random",
            lambda: rng.uniform(-1.0, 1.0, size=adapter.action_dim),
        ),
    ):
        for case in bank.cases[:num_cases]:
            try:
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
            except Exception:  # noqa: BLE001
                total_infra += 1
                continue
            ok = False
            for _t in range(horizon):
                try:
                    state, _done, success, _info = adapter.step(policy())
                except Exception:  # noqa: BLE001
                    total_infra += 1
                    break
                total_steps += 1
                if success:
                    ok = True
                    break
            if ok:
                total_successes += 1

    total = num_cases * 2
    rate = total_successes / total if total else 0.0
    passed = rate <= max_success_rate and total_infra == 0
    return GateResult(
        gate="5_random_noop_sanity",
        status="PASS" if passed else "FAIL",
        detail=(
            f"{total_successes}/{total} episodes 'succeeded' (max allowed "
            f"{max_success_rate}); {total_infra} infra failures; "
            f"{total_steps} steps"
        ),
        metrics={
            "successes": total_successes,
            "episodes": total,
            "rate": rate,
            "infra_failures": total_infra,
            "steps": total_steps,
        },
    )


# ---------------------------------------------------------------------------
# Gate 6 — checkpoint smoke run
# ---------------------------------------------------------------------------


def gate_checkpoint_smoke(
    cfg,
    adapter: RobosuiteStateAdapter,
    bank: ResetBank,
    *,
    num_episodes: int,
    max_success_rate: float = 1.0,
) -> GateResult:
    """Run a few episodes with the configured checkpoint policy.

    Skipped loudly when no checkpoint is configured (the gates CLI is also
    used before training on a fresh machine).
    """
    ckpt_path = cfg.train.get("stage1_ckpt_path")
    if not ckpt_path:
        return GateResult(
            gate="6_checkpoint_smoke",
            status="SKIPPED",
            detail="no train.stage1_ckpt_path configured — smoke gate skipped",
        )
    try:
        import torch

        from phaseforge.cli import build_eval_model

        model = build_eval_model(cfg)
        requested_device = str(cfg.project.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        device = torch.device(requested_device)
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="6_checkpoint_smoke",
            status="FAIL",
            detail=f"model/checkpoint construction failed: {exc}",
        )
    if num_episodes > len(bank.cases):
        num_episodes = len(bank.cases)

    from phaseforge.data.common.normalizer import FrozenNormalizer
    from phaseforge.data.ingestion.cache_manager import CacheManager
    from phaseforge.data.paths import processed_cache_root

    try:
        hash_val = CacheManager.compute_hash(cfg.data)
        normalizer = FrozenNormalizer.load(processed_cache_root() / hash_val / "norm_stats.pt")
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="6_checkpoint_smoke",
            status="FAIL",
            detail=f"normalizer load failed: {exc}",
        )

    mean = normalizer.mean.to(device)
    std = normalizer.std.to(device)

    successes = 0
    infra = 0
    policy_failures = 0
    with torch.inference_mode():
        for case in bank.cases[:num_episodes]:
            try:
                state = adapter.reset_to(case.states, xml=case.xml, ep_meta=case.ep_meta)
            except Exception:  # noqa: BLE001
                infra += 1
                continue
            ok = False
            for _t in range(adapter.horizon):
                try:
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    normalized = ((state_tensor - mean) / std).unsqueeze(0)
                    action = model.get_action(normalized)  # type: ignore[operator]
                    action = np.asarray(action.detach().cpu().numpy()).reshape(-1)
                    action = adapter.validate_action(action)
                except PolicyInvalidActionError:
                    policy_failures += 1
                    break
                except Exception:  # noqa: BLE001
                    policy_failures += 1
                    break
                try:
                    state, _done, success, _info = adapter.step(action)
                except Exception:  # noqa: BLE001
                    infra += 1
                    break
                if success:
                    ok = True
                    break
            if ok:
                successes += 1

    rate = successes / num_episodes if num_episodes else 0.0
    passed = rate + 1e-9 >= max_success_rate and infra == 0
    return GateResult(
        gate="6_checkpoint_smoke",
        status="PASS" if passed else "FAIL",
        detail=(
            f"{successes}/{num_episodes} episodes solved ({rate:.3f}; "
            f"max expected {max_success_rate}); {policy_failures} policy "
            f"failures; {infra} infra failures"
        ),
        metrics={
            "successes": successes,
            "episodes": num_episodes,
            "rate": rate,
            "policy_failures": policy_failures,
            "infra_failures": infra,
        },
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_gates(cfg, *, bank: ResetBank | None = None) -> list[GateResult]:
    """Run every gate in order; returns results (raises only on setup errors).

    The bank is generated/loaded like the eval path does; robosuite must be
    installed or the gates fail (hard requirement, not skippable).
    """
    from phaseforge.evaluations.rollout.runner import (
        _adapter_from_config,
        resolve_pinned_metadata,
        state_spec_from_config,
    )

    meta = resolve_pinned_metadata(cfg)
    # The expected env_name is derived from the dataset's pinned metadata
    # by default; an explicit override in cfg.eval.env wins. This makes the
    # gate work uniformly across the five benchmark tasks.
    expected_env_name = cfg.eval.env.get("expected_env_name") or meta.env_name
    verify_environment_parity(
        meta,
        expected_env_name=str(expected_env_name),
        robosuite_requirement=str(cfg.eval.env.get("robosuite_requirement", "==1.5.1")),
        mujoco_requirement=str(cfg.eval.env.get("mujoco_requirement", "==3.2.7")),
    )
    spec = state_spec_from_config(cfg)
    adapter = _adapter_from_config(cfg, meta)
    try:
        if bank is None:
            from phaseforge.evaluations.rollout.runner import load_or_generate_bank

            bank = load_or_generate_bank(cfg, meta)

        gates = cfg.eval.get("gates", {})
        results = [
            gate_env_schema(
                adapter,
                bank,
                expected_state_dim=int(cfg.data.state_dim),
                expected_action_dim=int(cfg.data.action_dim),
            ),
            gate_demo_replay(
                adapter,
                _resolve_raw_hdf5(cfg),
                num_demos=int(gates.get("replay_demos", 1)),
                tolerance=float(gates.get("replay_tolerance", 1e-3)),
            ),
            gate_action_contract(
                adapter,
                tolerance=float(cfg.eval.episodes.get("action_tolerance", 1e-4)),
            ),
            gate_scripted_controller(
                adapter,
                bank,
                spec,
                threshold=float(gates.get("scripted_threshold", 1.0)),
            ),
            gate_random_noop_sanity(
                adapter,
                bank,
                num_cases=int(gates.get("random_sanity_episodes", 20)),
                horizon=int(gates.get("random_sanity_horizon", 200)),
                max_success_rate=float(gates.get("random_sanity_success_max", 0.05)),
            ),
            gate_checkpoint_smoke(
                cfg,
                adapter,
                bank,
                num_episodes=int(gates.get("smoke_episodes", 10)),
            ),
        ]
        return results
    finally:
        adapter.close()


def _resolve_raw_hdf5(cfg) -> Path:
    raw_dir = cfg.data.source.get("dir")
    if not raw_dir:
        return Path("")
    raw_path = Path(str(raw_dir))
    files = sorted(raw_path.glob("*.hdf5")) if raw_path.is_dir() else []
    return files[0] if files else Path("")


__all__ = [
    "GateResult",
    "GateFailure",
    "run_all_gates",
    "gate_env_schema",
    "gate_demo_replay",
    "gate_action_contract",
    "gate_scripted_controller",
    "gate_random_noop_sanity",
    "gate_checkpoint_smoke",
]
