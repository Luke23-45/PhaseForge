# Findings Register — PhaseForge surgical analysis (branch: surgical-cpu-analysis)

All experiments here run on the local optimized CPU at the current commit.
Each experiment writes its raw JSON under `outputs/cpu_sweep/_findings/` and a
rendered report in this directory. The register tracks status and the
one-line conclusion.

## Wave A — absolutely first (from docs/dev/final_plan.md)

| # | Experiment | Script | Status | Conclusion |
|---|------------|--------|--------|-----------|
| A1 | Checkpoint sweep (SR vs epoch, same 50 paired episodes) | `scripts/experiments/checkpoint_sweep.py` | pending | — |
| A2 | SR vs val-MSE correlation across checkpoints | `scripts/experiments/sr_val_corr.py` | pending | — |
| A3 | Multi validation-bank checkpoint ranking test | planned | pending | — |
| A4 | Phase×Expert specialization matrix M_{z,e} | planned | pending | — |
| A5 | Routing counterfactuals: learned / centroid / oracle / uniform / random | planned | pending | — |

## Wave B — mechanism tests (planned)

| # | Experiment | Status |
|---|------------|--------|
| B1 | Four-way router×expert init matrix | pending |
| B2 | Expert divergence trajectory D_ij(t) at t=1,5,20,200 | pending |
| B3 | balance_coeff ∈ {0, .001, .01, .03} | pending |
| B4 | router noise σ ∈ {0, .05, .1} | pending |

## Wave C — representation + failure analysis (planned)

| # | Experiment | Status |
|---|------------|--------|
| C1 | Latent geometry / centroid-margin Δ_i | pending |
| C2 | Failure-by-phase z_failure | pending |
| C3 | Routing on success vs failed episodes | pending |
| C4 | K=3/6/12, corruption, FT (full matrix rerun) | pending |

## Raw data layout

- Training runs: `outputs/cpu_sweep/phaseforge/stage{1,2}/seed{S}/...`
- Eval runs (per checkpoint): `outputs/cpu_sweep/eval/phaseforge/seed{S}/...`
- Aggregated findings JSON: `outputs/cpu_sweep/_findings/*.json`