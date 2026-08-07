"""Atomic doc patches for F4 (pins), D4 (LAR-MoE citation), B3 (oracle footnote),
D3 (Report #2 revision). Fails loudly without writing anything if any target
line is missing (guards against concurrent edits).
"""

from pathlib import Path


def apply(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = []
    for old, new in replacements:
        if old not in text:
            failures.append(old[:90])
            continue
        text = text.replace(old, new, 1)
    if failures:
        raise SystemExit(
            f"{path}: {len(failures)} target(s) missing, no writes performed:\n"
            + "\n".join(f"  - {f!r}" for f in failures)
        )
    path.write_text(text, encoding="utf-8")
    print(f"OK {path}: {len(replacements)} replacements")


# ---------------------------------------------------------------------------
# F4 — evaluation_plan.md dependency table -> exact resolved pins
# ---------------------------------------------------------------------------
eval_plan = Path("docs/dev/evaluation_plan.md")
apply(
    eval_plan,
    [
        (
            "| `robosuite` | ≥1.4 | Franka Panda simulation | `uv add robosuite` |",
            "| `robosuite` | ==1.4.0 | Franka Panda simulation | `uv sync --extra rollout` |",
        ),
        (
            "| `mujoco` | ≥3.1 | Physics engine (robosuite dep) | installed with robosuite |",
            "| `mujoco` | ==3.3.1 | Physics engine (robosuite dep) | installed with robosuite |",
        ),
        (
            "| `gymnasium` | ≥0.29 | Environment interface | installed with robosuite |",
            "| `gymnasium` | 1.3.0 (resolved via robosuite, uv.lock) | Environment interface | installed with robosuite |",
        ),
        (
            "| `libero` | latest | Task definitions, initial states, benchmark | `pip install libero` (or fork) |",
            "| `libero` | ==0.1.1 (resolved in uv.lock) | Task definitions, initial states, benchmark | `uv add libero` |",
        ),
        (
            "```\npip install robosuite  # includes mujoco\npip install libero     # includes benchmark definitions\n```",
            "```\nuv sync --extra rollout  # robosuite==1.4.0, mujoco==3.3.1, libero==0.1.1 (uv.lock)\n```",
        ),
    ],
)

# ---------------------------------------------------------------------------
# D4 — novelty_claim.md LAR-MoE citation (verified via arXiv)
# ---------------------------------------------------------------------------
novelty = Path("docs/notes/novelty_claim.md")
apply(
    novelty,
    [
        (
            '4. "LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Manipulation", arXiv:2603.08476 (authors unverified — confirm before citing).',
            '4. A. Rodriguez, C. Li, L. Mazza, R. Younis, O. Hellig, S. Bodenstedt, M. Wagner, S. Speidel, "LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning", arXiv:2603.08476, 2026-03-09 (verified; same group as MoE-ACT).',
        ),
    ],
)

# ---------------------------------------------------------------------------
# B3 — experiment_report.md (Final): oracle footnote in success tables
# ---------------------------------------------------------------------------
final = Path("docs/dev/experiment_report.md")
apply(
    final,
    [
        (
            "| **Oracle MoE** (upper bound) | **4.54%** | ≈0 | ≈0 | 0.833 | 1.000 |",
            "| **Oracle MoE** (GT routing; signature-only bound) | **4.54%** | ≈0 | ≈0 | 0.833 | 1.000 |",
        ),
        (
            "*(ES = early stopping triggered)*",
            "*(ES = early stopping triggered)*\n\n*Oracle MoE footnotes (E7): receives privileged GT phase labels at inference — non-deployable; its only claim is the routing signature (NMI=1.0, entropy≈0), not success. The July matrix is superseded by the 8-cell Batch A/B protocol (C1/E7/E8: `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge`, `oracle_moe`, `teacher_forced`, `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap`); rollout success (B1) is goal-predicate based, not this L2 proxy.*",
        ),
    ],
)

# ---------------------------------------------------------------------------
# D3 — REPORT_to_professor_2.md: A2 finding, oracle relabel, 8-cell matrix,
#      task-pool decision, timeline
# ---------------------------------------------------------------------------
report2 = Path("docs/reports/REPORT_to_professor_2.md")
apply(
    report2,
    [
        (
            "**Subject:** Response to your analysis of the 0% rollout results — our objectives, the verified failure analysis, and our revised three-stage plan",
            "**Subject:** Response to your analysis of the 0% rollout results — our objectives, the verified failure analysis, and our revised three-stage plan\n\n> **Revision (round 3, 2026-08-07):** this version corrects two claims of the round-2 text: (i) the 0% result is now attributed to **two** confirmed causes — A1 observability (as before) **and A2 task-pool mismatch** (the evaluated spatial/object/goal suites are zero-shot task pools; see §2.4); (ii) the oracle baseline is relabeled **signature-only / non-deployable** (E7) and a **teacher-forced cell** (E8, privileged-training, label-free inference) joins the matrix, which grew from 5 models to **8 cells** (Batch A/B). The full-length training claim is now true: early stopping is explicitly disabled in the protocol runners (`643674a`); tests: 113/113.",
        ),
        (
            "The experimental matrix is five models — `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge` (proposed), and `oracle_moe` (ground-truth phase routing, intended as the specialization upper bound) — trained on LIBERO-90 and evaluated with the accepted protocol (simulator rollouts, goal predicates, per-suite breakdowns, 50 episodes per task, multiple seeds).",
            "The experimental matrix (round-2 dry run: five models — `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge` (proposed), and `oracle_moe`). Round 3 extended it to eight cells: the five above, `teacher_forced` (E8; GT-partitioned experts, predicted-phase routing at inference — privileged *training*, label-free *inference*, deployable after training, footnoted), `phase_pretrain_random_router`, and `plain_encoder_phase_bootstrap` (2×2 completion). The oracle is relabeled a **signature-only, non-deployable bound** (its claim is the routing signature, not success). Evaluation: simulator rollouts, goal predicates, per-task breakdowns, 50 episodes/task on the in-distribution suite, 3 seeds.",
        ),
        (
            "On the actual simulator, results are **0% success: libero_spatial 0/500, libero_object 0/500** (libero_goal was in progress at last check), with episodes running the full horizon.",
            "On the actual simulator, results are **0% success: libero_spatial 0/500, libero_object 0/500** (libero_goal was in progress at last check), with episodes running the full horizon. Round-3 analysis (A2) added a second cause: those suites are **zero-shot task pools** — their tasks were never in the training set — so 0% is also consistent with task-pool mismatch, not observability alone (see §2.4).",
        ),
        (
            "| Undertrained / **under-observable** policy | 23-DoF proprioception, zero object information | **Consistent with all evidence** |",
            "| Undertrained / **under-observable** policy | 23-DoF proprioception, zero object information | **Consistent with all evidence** |\n| **Zero-shot task pool (A2)** | spatial/object/goal suites are unseen tasks; eval was zero-shot, never labeled as such | **Consistent (added round 3)** |",
        ),
        (
            "The remaining explanation is the information ceiling: our 23-DoF state (`robot0_joint_pos/vel`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`) describes the arm and gripper only. Object placement is drawn fresh every episode and is not a function of joint angles. A policy that cannot know where the bowl is cannot pick it up — 0/500 is the signature of that, exactly as you describe.",
            "The primary explanation remains the information ceiling: our 23-DoF state (`robot0_joint_pos/vel`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`) describes the arm and gripper only. Object placement is drawn fresh every episode and is not a function of joint angles. A policy that cannot know where the bowl is cannot pick it up.\n\n### 2.4 Second confirmed cause: task-pool mismatch (A2)\n\nRound-3 analysis showed the 0% was **over-attributed to observability alone**. The raw LIBERO-90 pool mixes feature/spatial (2×), object (2×) and goal (1×) tasks, and our dem100 training subset was drawn with a 5:1 object:spatial ratio — a different distribution from the eval suites. The spatial/object suites are **zero-shot** evaluation: their tasks were never trained on. 0/500 is therefore consistent with two stacked causes, not one.\n\n**Decision 2 (locked):** evaluation is restricted to `libero_90` as the in-distribution core (90 tasks × 50 eps × 3 seeds, per-task breakdowns) plus `libero_10` as a labeled zero-shot row (10 tasks × 10 eps); spatial/object/goal suites are dropped from the protocol. Results JSONs declare `eval/suite_roles` (in-distribution / zero-shot) so the split can never be silently conflated again.",
        ),
        (
            "3. **Agreed: no further work on the 23-dim arm-only setup.** We will not attempt to rescue it with a larger or cleverer MoE. It is retired as an evaluation configuration.",
            "3. **Agreed: no further work on the 23-dim arm-only setup.** We will not attempt to rescue it with a larger or cleverer MoE. It is retired as an evaluation configuration. Evaluation scope is locked by Decision 2 (§2.4): `libero_90` in-distribution + `libero_10` labeled zero-shot only.",
        ),
        (
            "  - *Training side:* extend the state schema (`phaseforge/config/data/common.yaml`, `state_keys`) and the ingestion stripper so the object-state keys enter the training HDF5 features; the cache hash changes, triggering re-ingestion; retrain stage 1 + stage 2 for all five models.",
            "  - *Training side:* extend the state schema (`phaseforge/config/data/common.yaml`, `state_keys`) and the ingestion stripper so the object-state keys enter the training HDF5 features; the cache hash changes, triggering re-ingestion; retrain stage 1 + stage 2 for all eight cells (Batch A/B matrix, §1.1).",
        ),
        (
            "  2. Per-suite success rates clear the floor, with per-task breakdowns and 3 seeds.",
            "  2. In-distribution success on `libero_90` (per-task breakdowns, 3 seeds) clears the floor; the `libero_10` zero-shot row is reported labeled.",
        ),
        (
            "| Stage 1 | Schema + env changes, re-ingestion, retrain 5 models, rollout eval (all suites, 3 seeds) | ~3–5 days |",
            "| Stage 1 | Schema + env changes (done), re-ingestion, retrain 8 cells, rollout eval (`libero_90` + `libero_10`, 3 seeds) | ~38 h compute (F3 estimate); 2–4 weeks wall on free Colab |",
        ),
    ],
)

print("ALL DOC PATCHES APPLIED")
