"""Patch-0 census: build the per-task object decode tables (the B6 gate).

Run ONCE on the machine that has the LIBERO mirror AND the ``libero`` /
``robosuite`` packages installed (e.g. the Colab box used for evaluation):

    python -m phaseforge.data.scripts.build_object_index --suites libero_90 libero_10

What it does
------------
For every task HDF5 file the script:
  1. Reads the demo ``states`` array and verifies its width is at least
     the task's MuJoCo ``nq + nv`` (the flattened-sim-state convention;
     LIBERO issues #26/#71). The mirror may append extra trailing dims
     (observed nq+nv+1 in KITCHEN scenes) — the qpos block (first ``nq``
     columns) is what the decode reads, and the B6 gate verifies it
     against live observations.
  2. Loads the task's real environment, reads the LIVE observation dict,
     and derives the object set from the ``{name}_pos`` / ``{name}_quat``
     keys (the BDDL-defined observation list — exactly what the eval side
     will consume).
  3. Maps each object to its body and joint, recording the ``qpos`` slice
     (free: 7; hinge/slide: 1; fixed: none) plus, for constrained joints,
     the world-frame constants needed for the closed-form numpy decode:
     anchor point, axis direction, and rest pose (captured at reset).
  4. Runs the B6 gate on sampled timesteps across MULTIPLE demos:
     ``set_init_state(states[t])`` then compares the decoded pose against
     the LIVE robosuite observations at that exact sim state, and verifies
     every decoded quaternion is normalized. Any mismatch beyond float
     rounding fails the task (and the census), because it would silently
     corrupt every ingested trajectory. Coverage is tunable with
     ``--b6-max-demos`` (default 3) and ``--b6-steps-per-demo`` (default 7).

Output
------
``{data_root}/raw/libero/object_index.json`` (default; override with
``--out``). ``k_slots`` is the max object count observed; if it exceeds
``object_state.k_slots`` in ``config/data/common.yaml``, the script exits
with an error (the decoder fails loud too, but catching it here avoids a
useless 66GB re-ingest).

The B6 gate MUST pass before the P-Stage 1 re-ingest (E2 -> B6 -> E1
ordering, resolved plan Rev. 2).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_object_index")

# ---------------------------------------------------------------------------
# Constants (MuJoCo mjtJoint, matching object_state.py)
# ---------------------------------------------------------------------------

FREEJOINT, BALL, SLIDE, HINGE = 0, 1, 2, 3
_JOINT_TYPE_NAMES = {FREEJOINT: "free", BALL: "ball", SLIDE: "slide", HINGE: "hinge"}

POS_QUAT_DIM = 7
_POS_SUFFIX = "_pos"
_QUAT_SUFFIX = "_quat"
_ROBOT_PREFIX = "robot0"


# ---------------------------------------------------------------------------
# Small numpy helpers (w-first quaternions, MuJoCo convention)
# ---------------------------------------------------------------------------


def _rot_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    u = q[1:4]
    return v + 2.0 * (np.cross(u, np.cross(u, v)) + q[0] * np.cross(u, v))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"cannot normalize near-zero vector {v}")
    return v / n


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def _model_arrays(env) -> dict[str, np.ndarray]:
    model = env.sim.model
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "body_names": [str(n) for n in model.body_names],
        "body_parentid": np.asarray(model.body_parentid).reshape(-1),
        "body_jntadr": np.asarray(model.body_jntadr).reshape(-1),
        "jnt_type": np.asarray(model.jnt_type).reshape(-1),
        "jnt_qposadr": np.asarray(model.jnt_qposadr).reshape(-1),
        "jnt_pos": np.asarray(model.jnt_pos),
        "jnt_axis": np.asarray(model.jnt_axis),
        "body_xpos": np.asarray(env.sim.data.body_xpos),
        "body_xquat": np.asarray(env.sim.data.body_xquat),
    }


def _live_observations(env) -> dict[str, np.ndarray]:
    """Read the live observation dict across supported LIBERO wrappers.

    LIBERO versions expose the robosuite observation method differently:
    some put ``_get_observations`` on ``OffScreenRenderEnv`` itself, while
    others keep it on the wrapped robosuite environment at ``._env`` (and
    older builds used ``.env``). The census must use the same live
    observation source in both the object census and B6 parity gate.
    """
    candidates = [env, getattr(env, "_env", None), getattr(env, "env", None)]
    for candidate in candidates:
        getter = getattr(candidate, "_get_observations", None)
        if callable(getter):
            observations = getter()
            if not isinstance(observations, dict):
                raise TypeError(
                    f"{type(candidate).__name__}._get_observations() returned "
                    f"{type(observations).__name__}, expected dict"
                )
            return observations
    raise AttributeError(
        "Could not locate _get_observations() on OffScreenRenderEnv or its "
        "wrapped environment (._env/.env). Check the installed LIBERO/"
        "robosuite version and wrapper structure."
    )


def _resolve_body_id(m: dict, name: str) -> int | None:
    """Find the MuJoCo body id for an object name, defensively."""
    names = m["body_names"]
    by_name = {n: i for i, n in enumerate(names)}
    if name in by_name:
        return by_name[name]
    # LIBERO/robosuite sometimes suffix body names (e.g. "_body").
    for cand in (name + "_body", f"{name}_collision", name.split("_")[0]):
        if cand in by_name:
            return by_name[cand]
    prefix_matches = [i for i, n in enumerate(names) if n.startswith(name + "_")]
    if prefix_matches:
        return prefix_matches[0]
    return None


def _is_relative_object_observable(name: str) -> bool:
    """Return whether an observable is relative to the robot, not world pose.

    Robosuite exposes paired sensors such as
    ``akita_black_bowl_1_to_robot0_eef_pos`` and ``..._quat``. These are
    derived relative-coordinate sensors and have no corresponding MuJoCo
    body, so they must never become entries in the world-object index.
    """
    return "_to_robot" in name


def _object_entries_from_env(env, task_name: str, states: np.ndarray) -> list[dict]:
    """Build the ObjectEntry dicts for a task from its LIVE obs + model.

    Returns the list of entry dicts. Raises ValueError with a clear message
    if an object cannot be resolved (the census must be all-or-nothing).
    """
    m = _model_arrays(env)
    nq, nv = m["nq"], m["nv"]
    width = states.shape[-1]
    # The mirror's ``states`` can carry extra trailing dims beyond the
    # flattened sim state (qpos + qvel) — observed nq+nv+1 in the KITCHEN
    # scenes. The decode only ever reads the first ``nq`` columns (the
    # qpos block), and the B6 gate validates that assumption against live
    # robosuite observations, so trailing extras are safe. A width BELOW
    # nq+nv cannot be the flattened sim state and is fatal.
    if width < nq + nv:
        raise ValueError(
            f"{task_name}: states width {width} < nq({nq}) + nv({nv}) — "
            f"the 'states' key is not the flattened sim state for this "
            f"mirror revision."
        )
    if width != nq + nv:
        logger.info(
            "  %s: states width %d vs nq+nv=%d (%d extra trailing dim(s)); "
            "the qpos block is still validated by the B6 gate",
            task_name, width, nq + nv, width - (nq + nv),
        )

    obs = _live_observations(env)
    object_names = sorted(
        {
            key[: -len(_POS_SUFFIX)]
            for key in obs
            if key.endswith(_POS_SUFFIX)
            and not key.startswith(_ROBOT_PREFIX)
            and not _is_relative_object_observable(key[: -len(_POS_SUFFIX)])
            and key[: -len(_POS_SUFFIX)] + _QUAT_SUFFIX in obs
        }
    )
    if not object_names:
        logger.warning(f"  {task_name}: no object-state observables found in live obs")

    entries: list[dict] = []
    for name in object_names:
        body_id = _resolve_body_id(m, name)
        if body_id is None:
            raise ValueError(
                f"{task_name}: object {name!r} has observables but no body in "
                "the model — revise the name matching."
            )
        jid = int(m["body_jntadr"][body_id])
        if jid < 0:
            joint_type = "fixed"
            qpos_start, qpos_len = 0, 0
            constants: dict = {
                "rest_pos": m["body_xpos"][body_id].tolist(),
                "rest_quat": m["body_xquat"][body_id].tolist(),
            }
        else:
            jtype = int(m["jnt_type"][jid])
            qpos_start = int(m["jnt_qposadr"][jid])
            if jtype == FREEJOINT:
                joint_type = "free"
                qpos_len = POS_QUAT_DIM
                constants = {}
            elif jtype in (HINGE, SLIDE):
                joint_type = _JOINT_TYPE_NAMES[jtype]
                qpos_len = 1
                parent_id = int(m["body_parentid"][body_id])
                p_pos = m["body_xpos"][parent_id]
                p_quat = m["body_xquat"][parent_id]
                axis_world = _normalize(_rot_vec_by_quat(m["jnt_axis"][jid], p_quat))
                constants = {
                    "axis_world": axis_world.tolist(),
                    "rest_pos": m["body_xpos"][body_id].tolist(),
                    "rest_quat": m["body_xquat"][body_id].tolist(),
                }
                if jtype == HINGE:
                    anchor_world = p_pos + _rot_vec_by_quat(m["jnt_pos"][jid], p_quat)
                    constants["anchor_world"] = anchor_world.tolist()
            else:
                raise ValueError(
                    f"{task_name}: object {name!r} has unsupported joint type "
                    f"{jtype} ({_JOINT_TYPE_NAMES.get(jtype, '?')})"
                )
        entries.append(
            {
                "name": name,
                "joint_type": joint_type,
                "qpos_start": qpos_start,
                "qpos_len": qpos_len,
                **constants,
            }
        )
        logger.info(
            "  %s: %s -> %s (qpos [%d, %d))",
            task_name, name, joint_type, qpos_start, qpos_start + qpos_len,
        )
    return entries


# ---------------------------------------------------------------------------
# B6 gate: decode(states[t]) == live robosuite obs at the same sim state
# ---------------------------------------------------------------------------


def _run_b6_gate(
    env,
    task_name: str,
    table_entries: list[dict],
    demo_states: list[np.ndarray],
    max_demos: int = 3,
    steps_per_demo: int = 7,
) -> float:
    """Verify the numpy decode against live observations.

    Checks ``max_demos`` demonstrations (all of them when fewer exist) at
    ``steps_per_demo`` evenly spaced timesteps each (including t=0 and
    t=T-1). Also verifies every decoded quaternion is normalized, because
    the free-joint decoder copies the stored qpos quaternion verbatim and
    would silently propagate a non-unit quaternion otherwise.

    Returns the max absolute error across all (demo, timestep) checks.
    Raises ValueError on any mismatch beyond float rounding.
    """
    if max_demos <= 0:
        raise ValueError(f"max_demos must be >= 1, got {max_demos}")
    if steps_per_demo <= 0:
        raise ValueError(f"steps_per_demo must be >= 1, got {steps_per_demo}")
    if not demo_states:
        raise ValueError(f"{task_name}: B6 gate received no demo states")

    from phaseforge.data.libero.object_state import (
        ObjectEntry,
        ObjectIndex,
        TaskObjectTable,
    )

    table = TaskObjectTable(
        task_name=task_name,
        nq=env.sim.model.nq,
        objects=[ObjectEntry(**e) for e in table_entries],
    )
    if not table.objects:
        return 0.0
    probe = ObjectIndex(
        {task_name: table},
        k_slots=len(table.objects),
        include_mask=False,
    )

    max_err = 0.0
    n_checks = 0
    for demo_idx, states in enumerate(demo_states[:max_demos]):
        T = states.shape[0]
        steps = sorted(
            set(int(round(i)) for i in np.linspace(0, T - 1, steps_per_demo))
        )
        for t in steps:
            # LIBERO's set_init_state() returns the freshly materialized
            # observation dict. Use that return value when available: some
            # OffScreenRenderEnv versions update sim.data but do not refresh
            # the wrapped observation cache until the next public call.
            state_obs = env.set_init_state(states[t])
            live_obs = (
                state_obs
                if isinstance(state_obs, dict)
                else _live_observations(env)
            )
            # Materialize the sequence before passing it to NumPy.  Recent
            # NumPy versions reject a generator as the first argument to
            # concatenate (and this gate must fail on a real mismatch, not
            # because of the audit code itself).
            reference = np.concatenate(
                [
                    np.concatenate(
                        [
                            live_obs[f"{e['name']}_pos"],
                            live_obs[f"{e['name']}_quat"],
                        ]
                    )
                    for e in table_entries
                ]
            )
            decoded, _ = probe.decode(task_name, states[t : t + 1])
            err = float(np.max(np.abs(decoded[0] - reference)))
            max_err = max(max_err, err)
            n_checks += 1
            if err > 1e-4:
                raise ValueError(
                    f"B6 gate FAILED for {task_name} (demo {demo_idx}, "
                    f"t={t}): max |decode - live obs| = {err:.3e} (> 1e-4). "
                    "The decode is NOT identical to robosuite FK for this "
                    "task; do not ingest."
                )
            # Quaternion normalization check: free-joint decode copies the
            # stored qpos quaternion verbatim, so a non-unit stored quat
            # would corrupt the state channel silently.
            # ``decoded`` is a flattened [pos(3), quat(4)] block per object;
            # reshape first so each quaternion contains all four components.
            decoded_quats = decoded[0].reshape(-1, POS_QUAT_DIM)[:, 3:]
            for name, quat in zip(
                [e["name"] for e in table_entries], decoded_quats
            ):
                if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-4:
                    raise ValueError(
                        f"B6 gate FAILED for {task_name} (demo {demo_idx}, "
                        f"t={t}): decoded quaternion of {name!r} has norm "
                        f"{float(np.linalg.norm(quat)):.6f} (must be 1 within "
                        "1e-4). The stored qpos quaternion is not normalized."
                    )
    logger.info(
        "  %s: B6 gate OK (%d checks across %d demo(s), "
        "max |decode - live| = %.2e)",
        task_name, n_checks, min(max_demos, len(demo_states)), max_err,
    )
    return max_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _task_id_by_name(task_suite, task_name: str) -> tuple[int, Any]:
    """Resolve a task id by its benchmark name.

    Task lookup is BY NAME, not by position: the mirror files are sorted
    alphabetically while the benchmark's task order is not guaranteed to
    match, so index-based loading would silently pair each HDF5 file with
    the wrong environment (and the B6 gate would then fail confusingly).
    """
    try:
        names = task_suite.get_task_names()
    except (AttributeError, TypeError):
        names = None
    if names is not None:
        for i, name in enumerate(names):
            if str(name) == task_name:
                return i, task_suite.get_task(i)
        known = list(names)[:5]
        raise KeyError(
            f"Task {task_name!r} not found in suite {task_suite.name!r} "
            f"(known names e.g. {known}). The mirror filenames must match "
            "the benchmark task names."
        )

    n_tasks = getattr(task_suite, "n_tasks", None)
    if n_tasks is None:
        n_tasks = len(getattr(task_suite, "task_list", []))
    for i in range(int(n_tasks)):
        task = task_suite.get_task(i)
        if task.name == task_name:
            return i, task
    raise KeyError(
        f"Task {task_name!r} not found in suite {task_suite.name!r} "
        "(task_suite exposes neither get_task_names() nor a usable "
        "task count; match filenames to the benchmark task list)."
    )


def _load_env(task_suite_name: str, task_name: str):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task_id, task = _task_id_by_name(task_suite, task_name)
    bddl_file = str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
    env.reset()
    return env, task


def build_index(
    suites: list[str],
    data_root: str | None = None,
    out_path: str | None = None,
    b6_max_demos: int = 3,
    b6_steps_per_demo: int = 7,
) -> Path:
    """Run the census and write the object index JSON."""
    if b6_max_demos <= 0:
        raise ValueError(f"b6_max_demos must be >= 1, got {b6_max_demos}")
    if b6_steps_per_demo <= 0:
        raise ValueError(
            f"b6_steps_per_demo must be >= 1, got {b6_steps_per_demo}"
        )

    from phaseforge.data.libero.object_state import (
        DEMO_SUFFIX,
        ObjectEntry,
        ObjectIndex,
        TaskObjectTable,
    )
    from phaseforge.data.paths import libero_object_index_path

    if out_path is None:
        out_path = str(libero_object_index_path(data_root))

    all_tables: dict[str, dict] = {}
    b6_errors: dict[str, float] = {}
    nq_values: list[int] = []

    for suite in suites:
        from phaseforge.data.paths import libero_suite_dir

        suite_dir = libero_suite_dir(suite, data_root)
        if not suite_dir.exists():
            raise FileNotFoundError(
                f"Suite dir not found: {suite_dir} (mirror not downloaded?)"
            )
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        logger.info("Suite %s: %d task file(s)", suite, len(hdf5_files))

        for path in hdf5_files:
            # The mirror filenames carry the ``_demo`` suffix, but the
            # benchmark task names (and the eval env's lookups) do not.
            # Strip it so both the benchmark resolution and the stored
            # ObjectIndex keys use the canonical name (object_state.py
            # accepts both forms defensively).
            task_name = path.stem.removesuffix(DEMO_SUFFIX)
            with h5py.File(path, "r") as f:
                demos = sorted(f["data"].keys())
                if not demos:
                    raise ValueError(f"{task_name}: no demos in file")
                # B6 gate now samples MULTIPLE demos per task (not just
                # demo_0) so decode parity is validated across the variety
                # of initial states the mirror actually contains.
                demo_states = [
                    f["data"][key]["states"][:] for key in demos[: b6_max_demos]
                ]

            env, task = _load_env(suite, task_name)
            if task.name != task_name:
                raise ValueError(
                    f"Filename/task mismatch: {task_name!r} resolved to "
                    f"benchmark task {task.name!r} — mirror and benchmark "
                    "task names must agree."
                )
            try:
                entries = _object_entries_from_env(env, task_name, demo_states[0])
                max_err = _run_b6_gate(
                    env,
                    task_name,
                    entries,
                    demo_states,
                    max_demos=b6_max_demos,
                    steps_per_demo=b6_steps_per_demo,
                )
                nq = int(env.sim.model.nq)
            finally:
                env.close()

            nq_values.append(nq)
            all_tables[task_name] = {"nq": nq, "objects": entries}
            b6_errors[task_name] = max_err

    k_slots = max((len(t["objects"]) for t in all_tables.values()), default=1)
    index = ObjectIndex(
        tasks={
            name: TaskObjectTable(
                task_name=name,
                nq=t["nq"],
                objects=[ObjectEntry(**e) for e in t["objects"]],
            )
            for name, t in all_tables.items()
        },
        k_slots=k_slots,
        dim_per_object=POS_QUAT_DIM,
        include_mask=True,
    )
    index.save(
        out_path,
        extra_meta={
            "created_by": "phaseforge.data.scripts.build_object_index",
            "created_at": time.time(),
            "suites": suites,
            "k_slots_census": k_slots,
            "max_states_width": max(nq_values) if nq_values else 0,
            "b6_max_errors": b6_errors,
        },
    )

    _warn_if_config_k_slots_mismatch(k_slots, out_path)
    logger.info(
        "Object index written to %s (%d tasks, k_slots=%d)",
        out_path, len(all_tables), k_slots,
    )
    return Path(out_path)


def _warn_if_config_k_slots_mismatch(census_k_slots: int, out_path: str) -> None:
    """Loudly flag k_slots mismatches with config/data/common.yaml."""
    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[3]
    cfg_path = repo / "phaseforge" / "config" / "data" / "common.yaml"
    if not cfg_path.exists():
        return
    cfg = OmegaConf.load(str(cfg_path))
    cfg_k = int(cfg.object_state.k_slots)
    if census_k_slots > cfg_k:
        raise RuntimeError(
            f"Census found {census_k_slots} objects for a task but "
            f"object_state.k_slots={cfg_k}. Raise k_slots in "
            "config/data/common.yaml (+ model input_dims) BEFORE re-ingesting."
        )
    if census_k_slots < cfg_k:
        logger.warning(
            "Census max objects (%d) < configured k_slots (%d). State keeps "
            "the configured width (zero-padded); the extra %d mask dims are "
            "always 0 — harmless but wasteful. Consider tightening k_slots.",
            census_k_slots, cfg_k, cfg_k - census_k_slots,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites", nargs="+", default=["libero_90", "libero_10"],
        help="LIBERO suites to census (default: libero_90 libero_10 — the "
             "eval set of rollout.yaml; the index must cover every task "
             "the eval env will run).",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Data root (default: PHASEFORGE_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: {data_root}/raw/libero/object_index.json)",
    )
    parser.add_argument(
        "--b6-max-demos", type=int, default=3,
        help="Max demos per task sampled by the B6 parity gate (default 3).",
    )
    parser.add_argument(
        "--b6-steps-per-demo", type=int, default=7,
        help="Timesteps sampled per demo by the B6 parity gate, evenly "
             "spaced incl. endpoints (default 7).",
    )
    args = parser.parse_args()
    try:
        build_index(
            suites=args.suites,
            data_root=args.data_root,
            out_path=args.out,
            b6_max_demos=args.b6_max_demos,
            b6_steps_per_demo=args.b6_steps_per_demo,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Census FAILED: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
