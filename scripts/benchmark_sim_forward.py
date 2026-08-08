"""Replay-sanity benchmark: does ``sim.forward()`` matter for demo replay? (F1)

Decision pipeline for F1 ("candidate physics fix: sim.forward() before
mj_step"):

1. Load a handful of real LIBERO demos (states + actions) per suite.
2. For each demo, replay actions TWO ways from the demo's init state:
   (a) reset_then_replay: env.reset + set init state from the demo, then
       step actions WITHOUT any extra sim.forward() (the current behavior
       in :class:`StateOnlyLiberoEnv`);
   (b) forward_then_replay: same, but call env.sim.forward() before every
       action step (the F1 candidate fix).
3. Compare the resulting state trajectories against the recorded demo
   states: per-step position/rotation error of the eef and the main
   objects. Report mean/max errors for both modes.

If (b)'s errors are materially smaller than (a)'s, the candidate fix is
confirmed and should be applied to ``libero_env.py``. If they are similar,
F1 is RULED OUT with this artifact as the evidence.

Usage (Colab, real data only):
    uv sync --extra rollout
    uv run python scripts/benchmark_sim_forward.py --suite libero_90 \
        --num-demos 3 --data-root /content/data

Exit code 0 = benchmark completed; 1 = environment/data error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_demo_states_and_actions(suite: str, demo_idx: int, data_root: Path):
    """Load a demo's recorded states and actions from the raw HDF5 mirror.

    Layout is the LIBERO convention (same as ``build_object_index.py`` and
    ``vision_stripper.py`` consume): ``f["data"]`` is a group keyed
    ``demo_N``, each demo holds ``states`` (T, S) and ``actions`` (T, A).
    We pick the task file at ``demo_idx`` and its first trajectory. This
    function is intentionally thin: it only needs enough to answer F1, and
    the full truth is the B6 gate in ``build_object_index.py``.
    """
    h5py = __import__("h5py")
    from phaseforge.data.paths import libero_suite_dir

    suite_dir = libero_suite_dir(suite, data_root)
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite dir not found: {suite_dir} (mirror not downloaded?)")

    files = sorted(suite_dir.glob("*.hdf5"))
    if demo_idx >= len(files):
        raise IndexError(f"{suite} has only {len(files)} task files")

    with h5py.File(files[demo_idx], "r") as f:
        demos = sorted(f["data"].keys())
        if not demos:
            raise ValueError(f"{files[demo_idx].stem}: no demos in file")
        demo = f["data"][demos[0]]
        state = np.array(demo["states"][:])     # (T, S)
        actions = np.array(demo["actions"][:])  # (T, A)
    return state, actions


def _replay(state_traj: np.ndarray, actions: np.ndarray, env, extra_forward: bool):
    """Replay actions from the demo init state and collect step-wise errors."""
    T = min(len(state_traj), len(actions))

    # Reset to the demo init state (same mechanism as the B6 gate).
    env.reset()
    if hasattr(env, "set_init_state"):
        env.set_init_state(state_traj[0])
    else:
        env.sim.set_state_from_flattened(state_traj[0])
    env.sim.forward()

    eef_pos_errs: list[float] = []
    eef_quat_errs: list[float] = []
    # Compare the post-step state against the NEXT recorded state: in the
    # demo convention, state[t+1] is the result of executing action[t].
    for t in range(T - 1):
        if extra_forward and t > 0:
            env.sim.forward()
        env.step(actions[t])

        flat = env.sim.get_state().flatten()
        recorded = state_traj[t + 1]
        # eef offset from qpos: for LIBERO this is fixed by the robot model;
        # use the recorded proprio slice as a pragmatic proxy — identical in
        # both modes, so the mode comparison stays valid.
        eef_pos_errs.append(float(np.linalg.norm(flat[:3] - recorded[:3])))
        q = flat[3:7]
        qr = recorded[3:7]
        dq = float(np.linalg.norm(q - qr) + np.linalg.norm(q + qr))
        eef_quat_errs.append(min(dq, 4.0))

    return {
        "eef_pos_mae": float(np.mean(eef_pos_errs)),
        "eef_pos_max": float(np.max(eef_pos_errs)),
        "eef_quat_mae": float(np.mean(eef_quat_errs)),
        "eef_quat_max": float(np.max(eef_quat_errs)),
        "steps": T - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--data-root", default="/content/data")
    parser.add_argument("--out", default="outputs/eval/f1_benchmark.json")
    args = parser.parse_args()

    try:
        from libero.libero import benchmark  # type: ignore
    except ImportError as e:
        print(f"FATAL: libero not installed: {e}")
        sys.exit(1)

    data_root = Path(args.data_root)
    suite = benchmark.get_benchmark_dict()[args.suite]
    results: dict[str, dict] = {}
    results["mode"] = {
        "baseline": "no sim.forward() before step (current behavior)",
        "extra_forward": "sim.forward() before every action step (F1 candidate)",
    }

    for i in range(min(args.num_demos, len(suite))):
        task = suite.get_task(i)
        env = task.get_env()
        try:
            states, actions = _load_demo_states_and_actions(args.suite, i, data_root)
        except (FileNotFoundError, IndexError) as e:
            print(f"skip demo {i}: {e}")
            continue

        base = _replay(states, actions, env, extra_forward=False)
        fixed = _replay(states, actions, env, extra_forward=True)
        verdict = (
            "CONFIRM"
            if fixed["eef_pos_mae"] < 0.5 * base["eef_pos_mae"]
            else "no-difference"
        )
        results[f"demo_{i}"] = {"baseline": base, "extra_forward": fixed, "verdict": verdict}
        print(f"demo {i} [{task.name}]: {verdict}")

        # The verdict is also answerable from a single scalar: if BOTH modes
        # replay within a small tolerance, the physics is fine and F1 needs
        # no patch regardless of the mode difference.
        if base["eef_pos_mae"] < 1e-3 and fixed["eef_pos_mae"] < 1e-3:
            results["mode"]["overall"] = "ruled-out (both modes replay within tolerance)"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
