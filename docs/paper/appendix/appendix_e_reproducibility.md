# Appendix E — Reproducibility

This appendix documents the code provenance, dependency environment, hardware infrastructure, compute requirements, artifact directory layout, and the deterministic reproduction pipeline.

---

## E.1 Final Manifests, Data-Generating Commit, and Artifact Layout

### Code Provenance and Git Commits

All reported experimental sweeps, data ingestion artifacts, model checkpoints, and evaluation logs were generated using the following tracked codebase commits:
- **Primary Data-Generating Commits:** `e948b73`, `f83d096`.
- **Repository Organization:**
  ```text
  PhaseForge/
  ├── phaseforge/                 # Core library
  │   ├── config/                 # Hydra configuration files
  │   ├── data/                   # Ingestion, state machine, and PELT discovery
  │   ├── models/                 # MoE architecture, routers, and baselines
  │   ├── runner/                 # CLI entry points and sweep runner
  │   └── trains/                 # Stage 1 and Stage 2 training loops
  ├── experiments/                # Sweep matrices and evaluation plans
  ├── studies/analysis/           # Audit scripts, tables, and figure generators
  └── docs/paper/                 # Complete paper and appendix manuscripts
  ```

### Artifact Directory Layout

Training artifacts and evaluation logs are persisted in a deterministic directory structure:
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

## E.2 Dependency Versions and Hardware

### Pinned Software Stack

Experiments were executed within an isolated Python virtual environment with pinned dependencies:
- **Operating System:** Ubuntu Linux 24.04 LTS (`Linux-6.8.0-1063-aws-x86_64` with `glibc 2.39`).
- **Python Version:** `Python 3.10.14`.
- **Core Deep Learning Framework:** `torch==2.13.0+cu130`, `torchvision==0.18.0+cu130`.
- **Numerical and Scientific Libraries:** `numpy==2.4.6`, `scipy==1.13.1`, `scikit-learn==1.4.2`.
- **Robot Manipulation Benchmark:** `robomimic==0.3.0`, `robosuite==1.4.1`, `mujoco==3.1.5`.
- **Configuration & Logging:** `hydra-core==1.3.2`, `omegaconf==2.3.0`.

### Hardware Infrastructure

- **Compute Platform:** AWS EC2 Accelerated Computing GPU instances (`g5.xlarge` and `g5.2xlarge`).
- **Accelerator:** NVIDIA A10G Tensor Core GPU (24 GB VRAM, PCIe).
- **Host Processor:** AMD EPYC 7R32 CPU (4 vCPUs on `g5.xlarge`, 8 vCPUs on `g5.2xlarge`).
- **System Memory:** 16 GB to 32 GB RAM.

---

## E.3 Artifact Checksums, Reproduction Commands, and Compute Costs

### Reproduction Commands

The full experimental workflow can be executed deterministically via the unified CLI entry points:

1. **Data Ingestion and Task-Variable Extraction:**
   ```bash
   python -m phaseforge.runner.cli ingest \
       --task can \
       --dataset-path datasets/can/ph/low_dim.hdf5 \
       --output-dir outputs/processed/cache
   ```

2. **Stage 1 Representation Pre-Training:**
   ```bash
   python -m phaseforge.runner.cli train_stage1 \
       --task can \
       --config phaseforge/config/models/phaseforge_stage1.yaml \
       --seed 42
   ```

3. **Unsupervised Trajectory Regime Discovery (PELT + Clustering):**
   ```bash
   python -m phaseforge.runner.cli discover_topo \
       --task can \
       --penalty 10.0 \
       --min-length 5 \
       --num-regimes 6 \
       --output-dir outputs/topo/can
   ```

4. **Stage 2 Joint Modular Fine-Tuning:**
   ```bash
   python -m phaseforge.runner.cli train_stage2 \
       --task can \
       --stage1-checkpoint outputs/checkpoints/stage1/can_seed42.pt \
       --regime-artifact outputs/topo/can/regimes.pt \
       --config phaseforge/config/models/phaseforge.yaml \
       --seed 42
   ```

5. **Closed-Loop Rollout Evaluation:**
   ```bash
   python -m phaseforge.runner.cli evaluate \
       --task can \
       --checkpoint outputs/checkpoints/stage2/can_seed42.pt \
       --num-episodes 50 \
       --reset-bank-seed 2026 \
       --output-dir outputs/evaluations/can/seed42
   ```

### Computational Cost and Memory Footprint (Table A9)

Table A9 reports the average wall-clock training times, training throughput, and peak GPU memory consumption per matrix cell (averaged across tasks and seeds):

| Method | Stage-1 Wall Time (s) | Stage-2 Wall Time (s) | Training Throughput (steps/s) | Peak GPU Memory (MB) |
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

*Efficiency Summary:* 
- Total training time for the proposed PhaseForge pipeline is approximately $27.6\,\mathrm{minutes}$ per task-seed on a single NVIDIA A10G GPU ($8.4\,\mathrm{minutes}$ for Stage 1, plus $19.2\,\mathrm{minutes}$ for Stage 2).
- Peak GPU VRAM utilization remains under $30\,\mathrm{MB}$ for low-dimensional states across all conditions.
- Inference latency during closed-loop simulation rollouts averages $1.1\,\mathrm{ms}$ per forward pass ($> 900\,\mathrm{Hz}$ throughput), well within the $20\,\mathrm{Hz}$ ($50\,\mathrm{ms}$) operational control period.
