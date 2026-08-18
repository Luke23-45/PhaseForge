# Experiment scripts — surgical analysis (branch: surgical-cpu-analysis)

Every script is standalone and uv-compatible:

    !uv run python scripts/experiments/<name>.py [--seeds 42,43,44] [--outputs outputs/surgical]

Each writes its raw findings JSON to `outputs/surgical/_findings/` and a
rendered report to `docs/dev/findings/`. Run order matters: the Wave A1
sweep (`checkpoint_sweep.py`) produces the per-epoch stage-2 checkpoints the
static analyzers consume; Waves A3/B1/B3_B4 train their own runs on top of
the shared stage-1 checkpoint from Wave A1. `run_all.py` orchestrates the
whole sequence and skips any script whose findings JSON already exists.

| Script | Wave | Purpose |
|--------|------|---------|
| `checkpoint_sweep.py` | A1 | Train phaseforge stage-1+2 (3 seeds) with per-epoch checkpoints; rollout-eval SR at epochs {1,2,4,8,16,30,50,100,200,best} on the same 50-case bank; overlay val-loss/NMI/entropy/balance per epoch. |
| `sr_val_corr.py` | A2 | Pearson corr(val action MSE, rollout SR) across checkpoints per seed + pooled. |
| `validation_bank.py` | A3 | 4 fixed val-bank seeds × 3 training seeds; SR spread vs checkpoint-selection noise. |
| `specialization_matrix.py` | A4 | Offline M_{z,e} = mean P(expert e | phase z) contingency from validation latents. |
| `routing_counterfactuals.py` | A5 | Offline action-MSE of learned vs oracle_true vs oracle_pred vs uniform vs random router distributions. |
| `four_way_init.py` | B1 | 4 cells (router_init × expert_init) × 3 seeds; best-epoch rollout eval. |
| `expert_diversity.py` | B2 | Offline expert-output divergence D_ij(t) at t=1,5,20,200 from Wave A1 checkpoints. |
| `ablation_grid.py` | B3+B4 | balance_coeff {0.0, 0.01, 0.1} × noise_std {0.0, 0.1, 0.5} grid; best-epoch rollout eval. |
| `latent_geometry.py` | C1 | Offline per-phase centroid/intra/inter distances + silhouette on validation latents. |
| `failure_phase.py` | C2 | Artifact-only failure analysis (success rate, steps-to-failure, failure categories) over any eval dirs. |
| `run_all.py` | — | Master runner: `--waves A B C` executes everything in order, skip-if-done, `--force` to rerun. |

All rollout evals reuse the frozen reset bank (seed 2026, 50 cases) so
episodes are paired across checkpoints. Offline analyzers load checkpoints
via `_model_utils.py` (Hydra compose + `build_model` + `_load_state_dict_checked`,
matching the training CLI exactly).