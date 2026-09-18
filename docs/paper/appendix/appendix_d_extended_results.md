# Appendix D — Extended Experimental Results

This appendix reports the complete, unaggregated empirical data tables generated across all experimental sweeps, including the per-seed raw rollouts across all ten methods, the matched Can/Square router-initialization ablation breakdown, and paired statistical significance tests.

---

## D.1 Per-Seed Raw Rollout Success Rates (Table A1)

Table A1 documents the exact per-seed rollout episode counts and success fractions across all three independently trained seeds (`seed 42`, `seed 43`, `seed 44`) for all 10 evaluated methods (50 rollout episodes per seed, $N = 150$ total trials per cell). The benchmark results summarized in main-paper Table 1 are derived directly from these per-seed evaluations:

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

## D.2 Matched Can/Square Ablation Breakdown (Table A4)

Table A4 provides the unaggregated per-seed success rates, validation NMI, routing-switch rates, and expert collapse rates across the matched Can/Square ablation suite (where margin loss is disabled, $\lambda_m = 0$):

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

### Definition of Validation Collapse Metric

The expert collapse rate reported in Table A4 corresponds strictly to the recorded validation routing metric `val/expert_collapse_rate`. This metric evaluates the fraction of available experts ($E = 6$) that receive fewer than $1 / (2E) = 1/12 \approx 8.3\%$ of total routing assignments across the 20 held-out validation demonstration trajectories.

Across all evaluated ablation conditions and seeds, the recorded validation collapse rate is exactly $0.00$, indicating that all six experts receive substantial routing assignments on the validation distribution. This finding is a empirical property of the validation trajectories under the trained models, and should not be construed as a general guarantee that expert starvation cannot occur on out-of-distribution inputs.

---

## D.3 Paired Differences and Statistical-Test Results (Table A15)

Table A15 reports paired within-seed success differences ($\Delta = \text{Success}_{\mathrm{PhaseForge}} - \text{Success}_{\mathrm{Baseline}}$) evaluated across identical reset states. Multiplicity-adjusted hypothesis testing is conducted using exact two-sided sign tests with step-down Holm-Bonferroni correction:

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

*Statistical Note:* Across three independently trained seeds, all paired comparisons yield Holm-adjusted $p$-values of $1.000$. None of the observed performance differences achieve statistical significance at the population level; all reported differences must be interpreted as descriptive properties of the observed training runs.
