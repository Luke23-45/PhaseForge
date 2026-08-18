# Experiment scripts — surgical analysis (CPU, branch: surgical-cpu-analysis)

Every script is standalone (`python scripts/experiments/<name>.py`) and writes
its raw findings JSON to `outputs/cpu_sweep/_findings/` and a rendered report
to `docs/dev/findings/`.

| Script | Purpose |
|--------|---------|
| `checkpoint_sweep.py` | Train phaseforge stage-1+2 (3 seeds) on CPU with per-epoch checkpoints; rollout-eval SR at epochs {1,2,4,8,16,30,50,100,200,best} on the same 50-case bank; overlay val-loss/NMI/entropy/balance per epoch (Wave A1). |
| `sr_val_corr.py` | Pearson corr(val action MSE, rollout SR) across checkpoints per seed + pooled; stratified per-epoch table (Wave A2). |
| `validation_bank.py` | planned — multiple fixed val banks, checkpoint ranking correlation with SR ranking (Wave A3). |
| `specialization_matrix.py` | planned — M_{z,e} = MSE(expert e on phase z), Case A/B diagnosis (Wave A4). |
| `routing_counterfactuals.py` | planned — learned vs frozen-centroid vs oracle vs uniform vs random router rollouts (Wave A5). |
| `expert_diversity.py` | planned — D_ij(t) expert-output divergence at t=1,5,20,200 (Wave B2). |
| `failure_phase.py` | planned — failure-by-phase analysis from episodes (Wave C2). |

Run order: `checkpoint_sweep.py` first (produces the checkpointed stage-2
runs every other script consumes), then the static analyzers. All rollout
evals reuse the frozen reset bank (seed 2026, 50 cases) so episodes are
paired across checkpoints.