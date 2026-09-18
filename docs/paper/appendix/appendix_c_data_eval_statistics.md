# Appendix C — Data, Evaluation, and Statistics

This appendix details the dataset splits, structured observation schemas, reset-bank evaluation protocol, metric aggregation procedures, and statistical hypothesis-testing methods.

---

## C.1 Dataset Splits and Observation Schemas

All evaluations are conducted on five robot manipulation tasks from the Robomimic benchmark using the Proficient-Human (`ph`) demonstration datasets. The tasks are executed on a simulated 7-DOF Franka Panda arm under operational-space impedance control at a control frequency of $20\,\mathrm{Hz}$.

### Demonstration Splits and Leakage Prevention

Each task provides exactly 200 human demonstrations:
- **Training split:** 180 trajectories ($90\%$).
- **Validation split:** 20 trajectories ($10\%$).

All data splits are partitioned strictly at the **trajectory level**, rather than by shuffling individual timesteps. This ensures that the validation demonstrations contain completely disjoint motion sequences with zero temporal leakage into training batches. Normalization statistics (mean and standard deviation) are computed exclusively from the 180 training trajectories and frozen for all subsequent data preprocessing. The 20 validation demonstrations are used strictly for offline routing evaluation (NMI and switch rates) and are never used for checkpoint selection.

### Observation Schemas (Table A13)

Observations $x_t \in \mathbb{R}^D$ consist of low-dimensional, proprioceptive and object state vectors. Table A13 summarizes the schema, dimensions, and sequence statistics across the benchmark suite:

| Task | State Dim $D$ | Action Dim $A$ | Action Contract | Demonstration Count (Train / Val) | Mean Episode Length | Primary State Components |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Lift** | 19 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 124.5 steps | Robot arm joint positions/velocities, end-effector pose, cube position/quaternion |
| **Can** | 23 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 198.2 steps | Robot state, can position/quaternion, bin receptacle bounds |
| **Square** | 23 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 165.4 steps | Robot state, square peg pose, hole fixture coordinate frame |
| **ToolHang** | 53 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 387.1 steps | Robot state, tool base/hook geometries, frame attachment points |
| **Transport** | 59 | 14 | Bimanual $\Delta$ EEF Poses ($2 \times 6\mathrm{D}$) + Grippers ($2 \times 1\mathrm{D}$) | 180 / 20 | 442.8 steps | Dual-arm states, payload transport box, target table locations |

---

## C.2 Reset-Bank Provenance and Task-Specific Rollout Horizons

### Deterministic Reset Banks

To eliminate environment reset variance across model comparisons, rollout evaluations are initialized from a pre-generated bank of frozen environment states.
- **Master Reset Seed:** `2026`.
- **Per-Task Reset Hashes:** Verified by SHA-256 provenance manifests (`310d9cfd3fa5e843`, `a7d3953c0afcf560`, `c6683cf0dbb23876`, `db5b4c2a5e6519d0`, `e16288589f5f69c2`).
- **Trial Accounting:** Exactly 50 rollout episodes are executed per independently trained seed. Across the 3 evaluated training seeds (`seeds = [42, 43, 44]`), each cell in the evaluation matrix corresponds to exactly $N = 150$ closed-loop rollout trials.
- **Rollout Horizon:** The maximum episode horizon is fixed at $H = 500$ timesteps ($25.0\,\mathrm{seconds}$ of simulated execution) for all tasks. If the native environment success predicate is satisfied at any timestep $t \le H$, the episode terminates immediately and is scored as a success ($1$); otherwise, upon reaching $t = H$, it is scored as a failure ($0$).

---

## C.3 Success, NMI, and Routing-Switch-Rate Aggregation

Performance and routing metrics are aggregated across experimental seeds under the following protocols:

### Task Success Rate
Closed-loop control quality is measured as the empirical success fraction over the 150 rollout episodes:
\[
\text{Success Rate} = \frac{1}{150} \sum_{s=1}^3 \sum_{e=1}^{50} \mathbf{1}\bigl[\text{Episode } e \text{ of seed } s \text{ succeeded}\bigr].
\]

### Phase-Expert Alignment (NMI)
Phase-expert alignment is evaluated offline on the 20 held-out validation demonstrations using Normalized Mutual Information ($\operatorname{NMI}$). Let $k_t^* \in \{0, \dots, E-1\}$ denote the top-1 assigned expert at timestep $t$, and let $y_t^{\mathrm{phase}} \in \{0, \dots, K-1\}$ denote the canonical rule-derived phase label.

For each training seed $s \in \{42, 43, 44\}$, all validation timesteps across the 20 validation demonstrations are concatenated into a single sample array, and NMI is computed over these concatenated validation samples:

\[
\operatorname{NMI}_s(k^*, y^{\mathrm{phase}}) = \frac{2 \, I(k^*;\, y^{\mathrm{phase}})}{H(k^*) + H(y^{\mathrm{phase}})}
\]

where $I(k^*; y^{\mathrm{phase}})$ is mutual information and $H(\cdot)$ denotes Shannon entropy under arithmetic average normalization. The reported benchmark NMI is the arithmetic mean across the three independently trained seeds:
\[
\operatorname{NMI} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \operatorname{NMI}_s.
\]
$\operatorname{NMI} \in [0, 1]$, where $1$ indicates perfect bijective alignment between experts and behavioral phases, and $0$ indicates statistical independence.

### Step-to-Step Routing-Switch Rate
The routing-switch rate quantifies temporal stability across adjacent timesteps on the validation demonstration distribution. A step pair $(t, t+1)$ is defined as adjacent if and only if both samples belong to the same trajectory and their positions satisfy $\text{pos}_{t+1} = \text{pos}_t + 1$. Step pairs spanning across trajectory boundaries are strictly excluded.

For each training seed $s$, the switch rate is computed over all concatenated adjacent step pairs in the validation split:

\[
\text{Switch Rate}_s = \frac{\sum_{i=1}^{N_{\mathrm{val}}} \sum_{t=1}^{T_i - 1} \mathbf{1}\bigl[ k_{i, t+1}^* \ne k_{i, t}^* \bigr]}{\sum_{i=1}^{N_{\mathrm{val}}} (T_i - 1)}.
\]

The reported benchmark switch rate is the arithmetic mean across the three seeds:
\[
\text{Switch Rate} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \text{Switch Rate}_s.
\]

---

## C.4 Paired Comparisons, Wilson Intervals, and Holm Correction

### Wilson Score Confidence Intervals

Uncertainty on rollout success rates is reported using 95% Wilson score intervals, which provide calibrated binomial coverage without Gaussian normality assumptions. For $S$ successes out of $N = 150$ trials with observed success fraction $\hat{p} = S / N$ and critical value $z = 1.96$:

\[
\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}.
\]

Wilson score intervals quantify binomial sampling uncertainty over pooled rollout episodes; they do not represent variance across independently trained seeds.

### Paired Within-Seed Differences

Because all methods are evaluated on identical initial reset states, comparisons between methods $A$ and $B$ are computed as paired within-seed differences:
\[
\Delta_{s} = \text{Success}_A(s) - \text{Success}_B(s), \qquad \bar{\Delta} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \Delta_s.
\]

### Exact Sign Tests with Step-Down Holm-Bonferroni Correction

To test the null hypothesis that method $A$ is no better than method $B$, we perform exact two-sided sign tests on paired per-episode outcomes across identical reset states:
- **Test Statistic:** Under the null hypothesis $H_0: P(\text{Outcome}_A > \text{Outcome}_B) = 0.5$.
- **Exact Significance:** Computed using the exact binomial distribution for discordant episode pairs.
- **Multiplicity Correction:** Multiplicity correction across baseline comparisons is performed using the step-down Holm-Bonferroni method. Given sorted raw $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$:
  \[
  p_{(i)}^{\mathrm{Holm}} = \min\left(1.0,\; \max_{j \le i} \bigl\{ (M - j + 1) \, p_{(j)} \bigr\}\right).
  \]
As tabulated in Appendix D, after Holm adjustment, all paired differences across the three seeds yield $p = 1.000$, confirming that observed differences in success rate represent sample properties of the three training runs rather than resolved population-level effects.
