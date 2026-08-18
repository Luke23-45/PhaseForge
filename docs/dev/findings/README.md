# Findings Register — PhaseForge surgical analysis (branch: surgical-cpu-analysis)

All experiments here run on the dedicated surgical-analysis branch
(`surgical-cpu-analysis`) at the current commit. Each experiment writes its
raw JSON under `outputs/surgical/_findings/` and a rendered report in this
directory. The register tracks status and the one-line conclusion.

Run the whole sequence on the cloud with:

    !uv run python scripts/experiments/run_all.py --waves A B C --seeds 42,43,44

## Wave A — absolutely first (from docs/dev/final_plan.md)

| # | Experiment | Script | Status | Conclusion |
|---|------------|--------|--------|-----------|
| A1 | Checkpoint sweep (SR vs epoch, same 50 paired episodes) | `scripts/experiments/checkpoint_sweep.py` | written — cloud pending | — |
| A2 | SR vs val-MSE correlation across checkpoints | `scripts/experiments/sr_val_corr.py` | written — cloud pending | — |
| A3 | Multi validation-bank checkpoint ranking test | `scripts/experiments/validation_bank.py` | written — cloud pending | — |
| A4 | Phase×Expert specialization matrix M_{z,e} | `scripts/experiments/specialization_matrix.py` | written — cloud pending | — |
| A5 | Routing counterfactuals: learned / oracle_true / oracle_pred / uniform / random (offline action-MSE) | `scripts/experiments/routing_counterfactuals.py` | written — cloud pending | — |

## Wave B — mechanism tests

| # | Experiment | Script | Status |
|---|------------|--------|--------|
| B1 | Four-way router×expert init matrix | `scripts/experiments/four_way_init.py` | written — cloud pending |
| B2 | Expert divergence trajectory D_ij(t) at t=1,5,20,200 | `scripts/experiments/expert_diversity.py` | written — cloud pending |
| B3 | balance_coeff ∈ {0.0, 0.01, 0.1} | `scripts/experiments/ablation_grid.py` | written — cloud pending |
| B4 | router noise σ ∈ {0.0, 0.1, 0.5} | `scripts/experiments/ablation_grid.py` | written — cloud pending |

## Wave C — representation + failure analysis

| # | Experiment | Script | Status |
|---|------------|--------|--------|
| C1 | Latent geometry / centroid-margin Δ_i + silhouette | `scripts/experiments/latent_geometry.py` | written — cloud pending |
| C2 | Failure-by-phase z_failure | `scripts/experiments/failure_phase.py` | written — cloud pending (artifact-only: per-step phases are NOT stored in rollout artifacts, so z_failure requires an instrumented re-run; script reports steps-to-failure + categories) |
| C3 | Routing on success vs failed episodes | merged into A5 note | requires instrumented rollouts — not deliverable from current artifacts |
| C4 | K=3/6/12, corruption, FT (full matrix rerun) | covered by real matrix (outputs/part4, part5) | complete |

## Raw data layout

- Training runs: `outputs/surgical/phaseforge/stage{1,2}/seed{S}/...`
- Eval runs (per checkpoint): `outputs/surgical/eval/phaseforge/seed{S}/...`
- Aggregated findings JSON: `outputs/surgical/_findings/*.json`
- Wave-specific training runs: `outputs/surgical/phaseforge_b1_*/`, `outputs/surgical/phaseforge_bc_*/`, `outputs/surgical/phaseforge_vbank_*/` (each under `stage2/seed{S}/`)