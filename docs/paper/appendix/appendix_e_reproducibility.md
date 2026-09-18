# Appendix E — Reproducibility

This appendix documents the code provenance, dependency environment, compute requirements, artifact directory layout, and the deterministic reproduction pipeline.

---

## E.1 Final Manifests, Data-Generating Commit, and Artifact Layout

### Code Provenance and Git Commits

All reported experimental sweeps, data ingestion artifacts, model checkpoints, and evaluation logs were generated using the following tracked codebase commits:
- **Primary Data-Generating Commits:** `e948b73`, `f83d096`.
- **Primary Sweep Manifest:** `experiments/final_causal_matrix.json`.
- **Repository Organization:**
  ```text
  PhaseForge/
  ├── phaseforge/                 # Core library
  │   ├── config/                 # Hydra configuration files
  │   ├── data/                   # Ingestion, state machine, and PELT discovery
  │   ├── models/                 # MoE architecture, routers, and baselines
  │   ├── runner/                 # Execution runner and pre-flight validation
  │   └── trains/                 # Stage 1 and Stage 2 training loops
  ├── experiments/                # Sweep matrices and execution plans
  ├── studies/analysis/           # Audit scripts, tables, and figure generators
  └── docs/paper/                 # Complete paper and appendix manuscripts
  ```

### Artifact Directory Layout

Training artifacts and evaluation logs are persisted in the following deterministic directory structure:
```text
outputs/
├── processed/
│   └── cache/                   # SHA-256 verified processed Robomimic trajectories
├── topo/
│   └── {task_name}/             # Discovered PELT change-points and regime artifacts
├── checkpoints/
│   ├── stage1/                  # Pre-trained encoder and auxiliary heads
│   └── stage2/                  # Jointly fine-tuned MoE models
└── evaluations/
    └── {task_name}/             # Rollout traces, success booleans, and NMI evaluations
```

---

## E.2 Dependency Versions and Execution Environment

### Pinned Software Stack and Platform Record (Table A10)

Experiments were executed within an isolated Python virtual environment on AWS Linux instances. The protocol, platform environment, and artifact provenance are recorded in Table A10:

| Item | Value |
| :--- | :--- |
| **Seeds (matrix / ablation)** | 42, 43, 44 / 42, 43, 44 |
| **Reset banks** | `310d9cfd3fa5e843`, `a7d3953c0afcf560`, `c6683cf0dbb23876`, `db5b4c2a5e6519d0`, `e16288589f5f69c2` |
| **Reset seeds** | 2026 |
| **Evaluation router modes** | learned |
| **Training commits** | `e948b73`, `f83d096` |
| **Dropped-neuron hashes** | 39 recorded (e.g., `final_aligned_softmax_top1@seed42`: `9113226cdcef0f09c1d2bf8a9507d1fcf528e2d5d6b13b695a14f00139d60d54`) |
| **Pinned stack** | `numpy==2.4.6`, `torch==2.13.0+cu130`, `torchvision==0.18.0+cu130` |
| **Scientific stack** | `scipy==1.13.1`, `scikit-learn==1.4.2` |
| **Benchmark stack** | `robomimic==0.3.0`, `robosuite==1.4.1`, `mujoco==3.1.5` |
| **Config & logging** | `hydra-core==1.3.2`, `omegaconf==2.3.0` |
| **Platform** | `Linux-6.8.0-1063-aws-x86_64-with-glibc2.39` (`Python 3.10.14`) |
| **Evaluation to Checkpoint SHA links** | 165 / 180 verified |
| **Rollout horizon** | 500 steps ($25.0\,\mathrm{s}$ at $20\,\mathrm{Hz}$) |

Hardware provenance records confirm Linux AWS execution under the pinned stack above; specific GPU microarchitectures are not recorded in the artifact provenance record.

---

## E.3 Reproduction Commands and Compute Costs

### Reproduction Workflow

All data processing, representation pre-training, topology discovery, modular fine-tuning, and closed-loop rollouts are orchestrated through the unified runner entry point using `uv`:

1. **Pre-Flight Gate and Dry-Run Verification:**
   Before launching training runs, verify all manifest gates, provider orderings, and command contracts:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final \
     --verify-gates
   ```

2. **Dry-Run Inspection:**
   Inspect the complete multi-stage execution plan without executing commands:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final \
     --expect-steps 330 \
     --dry-run
   ```

3. **Full Experimental Sweep Execution:**
   Execute all training and closed-loop evaluation cells in the manifest:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final
   ```

### Computational Cost and Memory Footprint (Table A9)

Table A9 reports the average wall-clock training times, training throughput, and peak GPU memory consumption per matrix cell (mean across tasks and seeds, extracted directly from `timings.json` and training curve efficiency fields):

| Method | Stage-1 Wall (s) | Stage-2 Wall (s) | Steps/s | Peak GPU MB |
| :--- | :---: | :---: | :---: | :---: |
| **PhaseForge** | 501.7 | 1153.7 | 33.9 | 24.4 |
| **Monolithic BC** | 395.1 | -- | 50.9 | 21.5 |
| **Softmax Top-1** | -- | 1172.6 | 33.7 | 24.6 |
| **Phase-Random** | -- | 1135.5 | 33.9 | 24.4 |
| **Plain Encoder** | -- | 1184.8 | 33.5 | 25.5 |
| **Scratch MoE** | -- | 1174.1 | 34.1 | 24.4 |
| **Static Rule** | 473.0 | 1127.8 | 34.7 | 24.4 |
| **Factorial Floor** | -- | 1145.1 | 35.0 | 27.1 |
| **Teacher-Forced** | -- | 1024.4 | 39.2 | 24.2 |
| **Oracle (Offline)** | -- | 1006.0 | 38.9 | 24.1 |

*Accounting Summary:*
- Total training time for the proposed PhaseForge pipeline averages approximately $27.6\,\mathrm{minutes}$ per task-seed ($501.7\,\mathrm{s}$ for Stage 1, plus $1153.7\,\mathrm{s}$ for Stage 2).
- Training throughput averages $\sim 34\,\mathrm{steps/second}$ during Stage 2 modular fine-tuning.
- Peak GPU memory utilization remains under $30\,\mathrm{MB}$ for low-dimensional proprioceptive states across all evaluated conditions.
