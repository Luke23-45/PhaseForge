# Appendix D — Extended Experimental Results

This appendix reports the complete, untruncated empirical data tables generated across all experimental sweeps, including the full five-task benchmark, per-seed raw rollouts, the matched Can/Square router-initialization ablation, and paired statistical significance tests.

---

## D.1 Full Five-Task and Per-Seed Rollout Success

### Five-Task Benchmark Rollout Success (Table T1)

Table T1 reports closed-loop rollout success rates across all five Robomimic manipulation tasks. Brackets denote 95% Wilson score confidence intervals over the pooled rollout episodes ($N = 150$ per cell across 3 seeds).

| Method | Lift | Can | Square | ToolHang | Transport |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PhaseForge (Proposed)** | **1.00** [0.98, 1.00] | **0.78** [0.71, 0.84] | **0.41** [0.33, 0.49] | **0.00** [0.00, 0.02] | **0.01** [0.00, 0.04] |
| **Monolithic BC** | 1.00 [0.98, 1.00] | 0.63 [0.55, 0.70] | 0.34 [0.27, 0.42] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] |
| **Softmax Top-1** | 1.00 [0.98, 1.00] | 0.69 [0.61, 0.76] | 0.39 [0.31, 0.47] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] |
| **Phase-Random** | 1.00 [0.98, 1.00] | 0.62 [0.54, 0.69] | 0.39 [0.31, 0.47] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] |
| **Plain Encoder** | 1.00 [0.98, 1.00] | 0.36 [0.29, 0.44] | 0.50 [0.42, 0.58] | 0.00 [0.00, 0.02] | 0.01 [0.00, 0.04] |
| **Scratch MoE** | 0.93 [0.88, 0.96] | 0.67 [0.59, 0.74] | 0.31 [0.24, 0.39] | 0.00 [0.00, 0.02] | 0.01 [0.00, 0.04] |
| **Static Rule** | 1.00 [0.98, 1.00] | 0.53 [0.45, 0.60] | 0.12 [0.08, 0.18] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] |
| **Factorial Floor** | 0.97 [0.92, 0.99] | 0.38 [0.31, 0.46] | 0.31 [0.24, 0.39] | 0.00 [0.00, 0.02] | 0.01 [0.00, 0.05] |
| *Teacher-Forced (Diagnostic)* | 0.41 [0.33, 0.49] | 0.01 [0.00, 0.04] | 0.02 [0.01, 0.06] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] |

*Observations:*
- **Saturation and Floors:** Lift is saturated at 100% across all standard methods. ToolHang is completely unsolved (0% success across all methods), and Transport yields at most 0–2 successful episodes out of 150 trials across all methods. Neither task provides discriminative evidence.
- **Can:** PhaseForge records 78% observed success, leading BC (63%), Softmax (69%), and Phase-Random (62%).
- **Square:** Plain Encoder records the highest observed success rate at 50%, followed by PhaseForge at 41%, and Softmax / Phase-Random at 39%.

### Per-Seed Raw Rollout Success Rates (Table A1)

Table A1 documents the exact per-seed trial counts and success fractions across all three seeds (`seed 42`, `seed 43`, `seed 44`) for all 10 evaluated methods (50 rollout episodes per seed):

| Task | Method | Seed 42 | Seed 43 | Seed 44 | Mean Success |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lift** | PhaseForge | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | BC | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Softmax Top-1 | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Phase-Random | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Plain Encoder | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Scratch MoE | 0.94 (47/50) | 0.96 (48/50) | 0.90 (45/50) | 0.93 |
| Lift | Static Rule | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Factorial Floor | 0.92 (46/50) | 0.98 (49/50) | 1.00 (50/50) | 0.97 |
| Lift | Teacher-Forced | 0.08 (4/50) | 0.54 (27/50) | 0.60 (30/50) | 0.41 |
| Lift | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Can** | PhaseForge | 0.76 (38/50) | 0.80 (40/50) | 0.78 (39/50) | 0.78 |
| Can | BC | 0.78 (39/50) | 0.68 (34/50) | 0.42 (21/50) | 0.63 |
| Can | Softmax Top-1 | 0.58 (29/50) | 0.72 (36/50) | 0.76 (38/50) | 0.69 |
| Can | Phase-Random | 0.58 (29/50) | 0.66 (33/50) | 0.62 (31/50) | 0.62 |
| Can | Plain Encoder | 0.22 (11/50) | 0.56 (28/50) | 0.30 (15/50) | 0.36 |
| Can | Scratch MoE | 0.56 (28/50) | 0.74 (37/50) | 0.70 (35/50) | 0.67 |
| Can | Static Rule | 0.48 (24/50) | 0.58 (29/50) | 0.52 (26/50) | 0.53 |
| Can | Factorial Floor | 0.34 (17/50) | 0.46 (23/50) | 0.34 (17/50) | 0.38 |
| Can | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.02 (1/50) | 0.01 |
| Can | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Square** | PhaseForge | 0.46 (23/50) | 0.30 (15/50) | 0.46 (23/50) | 0.41 |
| Square | BC | 0.38 (19/50) | 0.20 (10/50) | 0.44 (22/50) | 0.34 |
| Square | Softmax Top-1 | 0.42 (21/50) | 0.44 (22/50) | 0.30 (15/50) | 0.39 |
| Square | Phase-Random | 0.40 (20/50) | 0.46 (23/50) | 0.30 (15/50) | 0.39 |
| Square | Plain Encoder | 0.52 (26/50) | 0.46 (23/50) | 0.52 (26/50) | 0.50 |
| Square | Scratch MoE | 0.44 (22/50) | 0.20 (10/50) | 0.30 (15/50) | 0.31 |
| Square | Static Rule | 0.14 (7/50) | 0.10 (5/50) | 0.12 (6/50) | 0.12 |
| Square | Factorial Floor | 0.30 (15/50) | 0.18 (9/50) | 0.46 (23/50) | 0.31 |
| Square | Teacher-Forced | 0.06 (3/50) | 0.00 (0/50) | 0.00 (0/50) | 0.02 |
| Square | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **ToolHang** | PhaseForge | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | BC | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Softmax Top-1 | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Phase-Random | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Plain Encoder | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Scratch MoE | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Static Rule | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Factorial Floor | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Transport** | PhaseForge | 0.02 (1/50) | 0.00 (0/50) | 0.00 (0/50) | 0.01 |
| Transport | BC | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Softmax Top-1 | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Phase-Random | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Plain Encoder | 0.00 (0/50) | 0.02 (1/50) | 0.00 (0/50) | 0.01 |
| Transport | Scratch MoE | 0.00 (0/50) | 0.00 (0/50) | 0.02 (1/50) | 0.01 |
| Transport | Static Rule | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Factorial Floor | 0.00 (0/50) | 0.02 (1/50) | 0.02 (1/50) | 0.01 |
| Transport | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |

---

## D.2 Matched Can/Square Router-Initialization Ablation

### Core Ablation Summary (Table T2)

Table T2 reports the primary router-initialization ablation on Can and Square. All arms operate under identical Stage-2 objectives with margin loss disabled ($\lambda_m = 0$), identical architectures, and identical optimizers:

| Initialization Arm | Can Success (Pooled) | Square Success (Pooled) | Validation NMI (Can / Sq) | Switch Rate (Can / Sq) | Expert Collapse Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime-Derived Init (Proposed)** | **76.0%** (114/150) | **26.7%** (40/150) | **0.67 / 0.51** | **0.04 / 0.07** | **0.0%** |
| **Rule-Based Centroid Init** | 68.7% (103/150) | 31.3% (47/150) | 0.08 / 0.09 | 0.11 / 0.10 | 0.0% |
| **Random Prototype Init** | 73.3% (110/150) | 28.0% (42/150) | 0.07 / 0.09 | 0.11 / 0.10 | 0.0% |
| *Plain Encoder (Diagnostic)* | 67.3% (101/150) | 32.0% (48/150) | 0.40 / 0.47 | 0.06 / 0.06 | 0.0% |
| *Softmax Top-1 (Diagnostic)* | 73.3% (110/150) | 35.3% (53/150) | 0.72 / 0.61 | 0.04 / 0.05 | 0.0% |

### Full Ablation Suite Per-Seed Breakdown (Table A4)

Table A4 provides the unaggregated per-seed success rates and validation metrics across the ablation suite:

| Task | Condition | Role | Mean SR | Per-Seed Success (42, 43, 44) | Final NMI | Switch Rate | Collapse Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Can** | Random Init | Matched prototype control | 0.733 | 0.68, 0.70, 0.82 | 0.071 | 0.109 | 0.00 |
| Can | Rule-Based Init | Matched prototype control | 0.687 | 0.68, 0.74, 0.64 | 0.083 | 0.107 | 0.00 |
| Can | **Regime Init (PF)** | Proposed initialization | **0.760** | 0.66, 0.82, 0.80 | **0.674** | **0.040** | 0.00 |
| Can | Plain Encoder | Representation diagnostic | 0.673 | 0.68, 0.74, 0.60 | 0.405 | 0.060 | 0.00 |
| Can | Softmax Top-1 | Architectural diagnostic | 0.733 | 0.66, 0.74, 0.80 | 0.721 | 0.037 | 0.00 |
| **Square** | Random Init | Matched prototype control | 0.280 | 0.14, 0.40, 0.30 | 0.086 | 0.096 | 0.00 |
| Square | Rule-Based Init | Matched prototype control | 0.313 | 0.24, 0.36, 0.34 | 0.091 | 0.100 | 0.00 |
| Square | **Regime Init (PF)** | Proposed initialization | **0.267** | 0.22, 0.28, 0.30 | **0.506** | **0.068** | 0.00 |
| Square | Plain Encoder | Representation diagnostic | 0.320 | 0.42, 0.26, 0.28 | 0.466 | 0.064 | 0.00 |
| Square | Softmax Top-1 | Architectural diagnostic | 0.353 | 0.32, 0.38, 0.36 | 0.613 | 0.054 | 0.00 |

---

## D.3 Paired Differences and Statistical-Test Results (Table A15)

Table A15 reports paired within-seed success differences ($\Delta = \text{Success}_{\mathrm{PhaseForge}} - \text{Success}_{\mathrm{Baseline}}$) evaluated across identical reset states. Significance is evaluated using exact two-sided sign tests with step-down Holm-Bonferroni correction over the pre-declared primary comparison family:

| Task | Condition A | Condition B | Mean $\Delta$ | $\Delta$ Std (Seeds) | $p$ (Exact Sign) | $p$ (Holm-Adjusted) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Lift** | PhaseForge | BC | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | BC | +0.153 | 0.192 | 1.000 | **1.000** |
| **Square** | PhaseForge | BC | +0.067 | 0.042 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | BC | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | BC | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Softmax Top-1 | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Softmax Top-1 | +0.093 | 0.081 | 0.250 | **1.000** |
| **Square** | PhaseForge | Softmax Top-1 | +0.020 | 0.151 | 1.000 | **1.000** |
| **ToolHang** | PhaseForge | Softmax Top-1 | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Softmax Top-1 | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Plain Encoder | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Plain Encoder | +0.420 | 0.159 | 0.250 | **1.000** |
| **Square** | PhaseForge | Plain Encoder | -0.093 | 0.058 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Plain Encoder | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Plain Encoder | +0.000 | 0.020 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Phase-Random | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Phase-Random | +0.160 | 0.020 | 0.250 | **1.000** |
| **Square** | PhaseForge | Phase-Random | +0.020 | 0.164 | 1.000 | **1.000** |
| **ToolHang** | PhaseForge | Phase-Random | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Phase-Random | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Scratch MoE | +0.067 | 0.031 | 0.250 | **1.000** |
| **Can** | PhaseForge | Scratch MoE | +0.113 | 0.076 | 0.250 | **1.000** |
| **Square** | PhaseForge | Scratch MoE | +0.093 | 0.070 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Scratch MoE | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Scratch MoE | +0.000 | 0.020 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Static Rule | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Static Rule | +0.253 | 0.031 | 0.250 | **1.000** |
| **Square** | PhaseForge | Static Rule | +0.287 | 0.076 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Static Rule | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Static Rule | +0.007 | 0.012 | 1.000 | **1.000** |

*Statistical Note:* With three independently trained seeds, all paired comparisons yield Holm-adjusted $p$-values of $1.000$. None of the observed performance differences establish statistical significance at the population level; all reported differences must be interpreted as descriptive properties of the observed training runs.
