# docs/dev — Development Records

This directory documents the development process: what was tried, what was
decided, and what the current direction is. The files here are progress
records, not the current specification — the authoritative direction lives in
`docs/plan/` (protocol, reports, runbooks) and `docs/op/` (implementation plan,
positioning).

## Active

| File | Role |
|---|---|
| `final_run_plan.md` | Operational runbook for the final five-task sweep: environment provisioning, runner commands, and table generation (steps 1–9 + main-method baselines). |

## Legacy (historical progress records)

| File | Date | Records |
|---|---|---|
| `legacy/final_plan.md` | 2026-08-19 | Supervisor consultation on the surgical checkpoint-selection direction (epoch sweep 1,2,4,8,16,30,50,100,200; router-noise and forced-utilization mechanism tests). Drove the surgical-cpu-analysis workstream, which was superseded by the warm-start analysis and the locked-in `phaseforge_r50` direction. |
| `legacy/lift_pilot_offline_report.md` | 2026-08-14 | Offline Lift pilot: 9-method × 3-seed action-MSE/NMI diagnostic matrix (pre-rollout, pre-fix). Baseline anchor for the rollout results in `docs/plan/lift_rollout_eval_report.md`. |
| `legacy/sweep_review_implementation_plan.md` | 2026-08-14 | Sweep-output review findings and fix plan with dated statuses: P1 `bc`/`bc_robot_only` tag merge (fixed 2026-08-14), stage-1 monitor restoration `val/loss_total` → `val/loss_action` with supervisor decision (2026-08-18), wall-time definition, provenance notes. |

Nothing here is deleted: these are the decision and audit records behind the
current implementation and the paper's "lessons learned" narrative.