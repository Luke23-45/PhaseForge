# Research Report: MoE Routing Organization and Closed-Loop Task Success

**To:** Research Director / Principal Investigator (Professor)  
**From:** PhaseForge Project Team & Antigravity Research Group  
**Date:** September 16, 2026
**Subject:** Scientific re-evaluation of PhaseForge routing structure and task performance in continuous robot manipulation
**Authority Reference:** `outputs_router_ablation_can_square` (Revision `f83d096`), valid PhaseForge final runs (Revision `e948b73`), and Square repair audit (Revision `76685b9`)

---

## Executive Summary & Scientific Repositioning

Following a review of the locked 5-task benchmark matrix (150 evaluation runs),
the focused Can/Square router-initialization ablation (30 Stage 2 runs), and
the subsequent Square repair runs, this report provides the final evidence
assessment of PhaseForge.

### The Honest Scientific Reality
If PhaseForge is positioned as an algorithmic contribution claiming a
**universally superior robotic manipulation policy**, the current evidence
does not support that claim:
1. On the 5-task benchmark, two tasks (`ToolHang`, `Transport`) remain at
   **0.00%** across the reported memoryless models, one task (`Lift`) is
   saturated at **100.0%**, and on `Square` (peg insertion), plain unsegmented
   Behavioral Cloning outperforms PhaseForge (**50.0% vs. 40.7%**).
2. In the controlled ablation, standard learned Softmax gating outperforms hard prototype routing pooled across Can and Square (**54.3% vs. 51.3%**), winning on every seed of Square.
3. Random prototype initialization is close to topological initialization on
   Can (73.3% vs. 76.0%) and higher on Square (28.0% vs. 26.7%). No
   significance test is claimed for these aggregate differences.

### The True Research Contribution
However, the mixed success results do not eliminate the possibility of a
useful structural contribution.

In the broader machine-learning MoE literature, routing stability is an
important design concern. Within the focused ablation, topology initialization
is associated with higher phase-expert NMI (from `0.07` to `0.67` on Can and
from `0.09` to `0.51` on Square) and lower measured switch rates (from `0.11`
to `0.04` on Can and from `0.10` to `0.07` on Square) than the random and
phase-initialized arms. These are structural routing observations, not proof
that the method solves routing instability.

The results support a more limited decoupling observation: better routing
organization did not guarantee higher closed-loop success in this benchmark.
The trace audit also observes larger commanded-action changes at expert
switches. Those changes are a plausible mechanism, but the available traces
do not contain force, contact, or actual end-effector velocity fields, so a
specific physical failure mechanism cannot be identified from these data.

This report outlines the evidence and the limits of a possible paper on the
mechanics and limitations of structured MoE routing in robot manipulation.

---

## 1. Disentangling "Stability" in MoE Architectures

To establish scientific precision, we must distinguish between four separate definitions of "stability" across the MoE and control literature:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         FOUR DIMENSIONS OF MoE STABILITY                        │
├──────────────────────────────────────┬───────────────────────────────────────────┤
│ 1. Training-Routing Stability        │ Does the router assign identical inputs to│
│    (StableMoE, ACL 2022)             │ the same expert throughout training?      │
├──────────────────────────────────────┼───────────────────────────────────────────┤
│ 2. Expert-Utilization Stability      │ Does the router avoid dead capacity and   │
│    (Switch Transformers, JMLR 2022)  │ complete expert collapse?                 │
├──────────────────────────────────────┼───────────────────────────────────────────┤
│ 3. Temporal Routing Coherence        │ Do adjacent timesteps in a physical       │
│    (Continuous Control MoE)          │ rollout avoid rapid, noisy chattering?    │
├──────────────────────────────────────┼───────────────────────────────────────────┤
│ 4. Closed-Loop Control Stability     │ Does the physical system converge to the  │
│    (Nonlinear / Switched Systems)    │ goal state under environmental dynamics?  │
└──────────────────────────────────────┴───────────────────────────────────────────┘
```

### Empirical Assessment of PhaseForge Against These Dimensions:
- **Dimension 1 (Training Stability):** Partially observed. Centroid initialization from topological change-points (`phase_topo`) changes the initial prototype geometry, but the available results do not isolate or measure all early-training fluctuations.
- **Dimension 2 (Utilization Stability):** Limited evidence only. No top-1 collapse was observed in the focused matrix, but a zero collapse rate alone does not establish healthy utilization or general utilization stability.
- **Dimension 3 (Temporal Routing Coherence):** Observed conditionally. The topology arm has higher NMI and lower measured switch rates than the random and phase-initialized arms in the focused ablation. The softmax control also has strong routing diagnostics, so topology is not shown to be uniquely most stable.
- **Dimension 4 (Closed-Loop Control Stability):** Not established. Higher measured routing organization did not guarantee higher task success in this benchmark.

**The Reframed Thesis:**  
> *"Topological prototype initialization can produce more phase-aligned and lower-switch routing in a memoryless hard-routing MoE. In the tested benchmark, this routing organization did not guarantee higher closed-loop success. Commanded-action changes at expert transitions are a plausible contributor, but the available traces do not establish that they induce the physical failure mechanism."*

---

## 2. Complete Empirical Evidence

### 2.1 The Full 5-Task Benchmark Matrix

The full matrix evaluated 10 method identities across 5 Robosuite tasks (50 paired evaluation episodes × 3 seeds = 150 episodes per cell, 500-step horizon for Lift/Can/Square/ToolHang, 700 for Transport).

| Task | BC Floor | Plain Encoder Control | Softmax Top-1 Control | Phase-Random Router | **PhaseForge (Proposed)** | Characteristic Phenomenon |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lift** | 150/150 (100%) | 150/150 (100%) | 150/150 (100%) | 150/150 (100%) | **150/150 (100%)** | Saturated Ceiling (No signal) |
| **Can** | 94/150 (62.7%) | 54/150 (36.0%) | 103/150 (68.7%) | 93/150 (62.0%) | **117/150 (78.0%)** | Structured Routing Win (+15.3% vs BC) |
| **Square** | 51/150 (34.0%) | **75/150 (50.0%)** | 58/150 (38.7%) | 58/150 (38.7%) | **61/150 (40.7%)** | Plain BC Beats Structured MoE (-9.3%) |
| **ToolHang** | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | **0/150 (0.0%)** | Markovian Memoryless Floor |
| **Transport** | 0/150 (0.0%) | 1/150 (0.7%) | 0/150 (0.0%) | 0/150 (0.0%) | **1/150 (0.7%)** | Markovian Memoryless Floor |

### 2.2 The Focused Router-Initialization Ablation (`outputs_router_ablation_can_square`)

To isolate the causal effect of prototype initialization without objective-level confounds, the Stage 2 margin loss was disabled across all arms:

| Method Token | Model Description | Can Success (s42 / s43 / s44) | Square Success (s42 / s43 / s44) | Pooled Success |
| :--- | :--- | :---: | :---: | :---: |
| `router_init_random` | Random Prototype Initialization | 68% / 70% / **82%** (73.3%) | 14% / **40%** / 30% (28.0%) | 50.7% (152/300) |
| `router_init_phase` | Rule-Based Centroid Initialization | 68% / 74% / 64% (68.7%) | 24% / 36% / 34% (31.3%) | 50.0% (150/300) |
| `router_init_topology` | **Topological Prototype Initialization** | 66% / **82%** / 80% (**76.0%**) | 22% / 28% / 30% (**26.7%**) | 51.3% (154/300) |
| `representation_bc` | Plain BC Latent Representation | 68% / 74% / 60% (67.3%) | **42%** / 26% / 28% (32.0%) | 49.7% (149/300) |
| `routing_softmax_top1` | Learned Softmax Gating Control | 66% / 74% / 80% (73.3%) | 32% / 38% / **36%** (**35.3%**) | **54.3%** (163/300) |

### 2.3 Routing Diagnostics: The Evidence of Structural Organization

Despite the mixed task success rates, the internal routing metrics show that
topological initialization substantially changes the measured latent
partition:

| Method | Can Phase-Expert NMI | Square Phase-Expert NMI | Can Switch Rate | Square Switch Rate | Top-1 Collapse Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `router_init_random` | 0.07 | 0.09 | 0.11 | 0.10 | **0.0%** |
| `router_init_phase` | 0.08 | 0.09 | 0.11 | 0.10 | **0.0%** |
| `router_init_topology` | **0.67** | **0.51** | **0.04** | **0.07** | **0.0%** |
| `representation_bc` | 0.41 | 0.47 | 0.06 | 0.06 | **0.0%** |
| `routing_softmax_top1` | 0.72 | 0.61 | 0.04 | 0.05 | **0.0%** |

**Empirical Takeaway:** Within this focused ablation, topology initialization is
associated with substantially higher phase alignment and lower measured switch
rates than the random and phase-initialized arms. Yet, on Square, it achieves
lower success (26.7%) than random initialization (28.0%) and plain
unsegmented BC (50.0%). The routing metrics therefore do not predict task
success by themselves.

---

### 2.4 Protocol audit of the Square repair runs

The later runs in `debug_run/new_runs` must not be treated as a replication of
the original PhaseForge anchor. The valid original PhaseForge Square protocol
used `train.margin.enabled=true` with `train.margin.lambda_margin=0.05`. The
`square_regression_repairs.json` manifest disabled that objective for the
anchor, top-2, beta, and phase-CE arms. The repair anchor therefore changed a
training objective in addition to serving as a control.

The original `debug_run/phaseforge_square` PhaseForge results were `23/50`,
`16/50`, and `24/50` for seeds 42, 43, and 44, respectively (pooled
`63/150 = 42.0%`). The later valid v10 final matrix independently reports
`23/50`, `15/50`, and `23/50` (pooled `61/150 = 40.7%`). On seed 42,
the repair results were 10/50 for the anchor, 11/50 with phase CE enabled,
17/50 for top-2, and 17/50 for beta 0.1. None exceeded the valid original
PhaseForge result of 23/50 on that reset bank.

The original seed-42 Stage 2 summary also reported phase-expert NMI `0.927`
and routing switch rate `0.0259`; the repair anchor reported NMI `0.485` and
switch rate `0.0680`. This is evidence that the repair protocol did not
preserve the original routing behavior. It is not an isolated test of the
repair mechanisms. No additional cloud runs of the current repair manifest
are recommended.

## 3. Commanded-action discontinuity at expert switches

Why can a more organized router still fail on precision contact tasks?

To answer this, we performed a step-by-step physical audit of the recorded rollout traces (`trace.jsonl`), measuring the instantaneous L2 norm of the commanded action difference:
$$\Delta a_t = \|a_t - a_{t-1}\|_2$$

We separated timesteps into **non-switch steps** ($e_t = e_{t-1}$) and **switch steps** ($e_t \neq e_{t-1}$):

| Method | Task | Overall Switch Rate | Mean Jump at Switch ($\overline{\Delta a}_{\text{switch}}$) | Mean Jump at Non-Switch ($\overline{\Delta a}_{\text{nonswitch}}$) | **Action Discontinuity Ratio** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PhaseForge (Topology Top-1)** | Can | 3.85% | **0.2322** | 0.0406 | **5.71×** |
| **PhaseForge (Topology Top-1)** | Square | 3.70% | **0.1503** | 0.0225 | **6.68×** |
| **Phase-Random Router** | Can | 4.68% | 0.2062 | 0.0418 | 4.94× |
| **Phase-Random Router** | Square | 3.79% | 0.1364 | 0.0237 | 5.76× |
| **Softmax Top-1 Control** | Can | 3.12% | 0.3072 | 0.0459 | 6.69× |
| **Softmax Top-1 Control** | Square | 2.43% | 0.2147 | 0.0236 | **9.09×** |

```
                       ACTION DISCONTINUITY AT SWITCHES
    Action Jump
      Norm
       ▲
  0.25 ┼                           ┌───────────┐ (0.2322: 5.7x Jump)
       │                           │  SWITCH   │
  0.20 ┼                           │   STEP    │
       │                           │           │
  0.15 ┼                           │           │
       │                           │           │
  0.10 ┼                           │           │
       │   ┌───────────┐           │           │
  0.05 ┼───│ NON-SWITCH│───────────│           │────────────────────────
       │   │  (0.0406) │           │           │
  0.00 ┴───┴───────────┴───────────┴───────────┴───────────────────────►
                                                              Timestep
```

### What the discontinuity does and does not establish:
1. **The vector-field discontinuity:** In a memoryless hard-routing MoE, different experts can produce different actions at a router boundary. The measured action jump is therefore a valid commanded-output observation.
2. **The causal limit:** The traces do not contain force, contact, or actual end-effector velocity fields. In addition, the softmax control has a larger reported jump ratio yet higher Square success. Thus the measured jump is not sufficient evidence for contact jamming or for a unique Square failure mechanism.
3. **The defensible interpretation:** Plain BC may benefit from having no discrete expert boundary, but the present results cannot determine whether its Square advantage comes from smoother actions, state coverage, action direction, or another difference in the learned policy.

---

## 4. Analyses required for a stronger mechanistic paper

The following analyses would be needed to substantiate a stronger mechanistic
claim. They are not all completed in the current artifact.

### Analysis 1: Switch Partitioning (Within-Phase vs. Phase-Boundary)
- **Concept:** In an ideal phase-aligned policy, switches should occur **only** at genuine semantic transitions (e.g., approach $\to$ grasp), with **zero** switches during steady-state execution within a phase.
- **Method:** Using offline PELT changepoints $\tau_k$, categorize every rollout timestep as boundary-adjacent ($|t - \tau_k| \le \delta$) or within-phase ($|t - \tau_k| > \delta$).
- **Evidence status:** Not completed. The current documents must not state an
  expected percentage as an observed result.

### Analysis 2: Action Discontinuity Profiling (commanded output only)
- **Method:** Document the measured commanded-action jump ratio.
- **Evidence status:** The jump profiling is completed for recorded traces,
  but it does not include physical contact or end-effector dynamics. A figure
  may show trace timing and timeout association, but it must not label the
  result as mechanical jamming without additional physical measurements.

### Analysis 3: Router Lipschitz & Perturbation Sensitivity
- **Concept:** A stable router must not alter its gating decision under small observation noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.
- **Method:** Inject calibrated Gaussian perturbations into states along successful trajectories and measure the **Router Flip Probability**:
  $$P_{\text{flip}}(\sigma) = \mathbb{P}\left(\arg\min_k \|E(x_t + \epsilon) - c_k\| \neq \arg\min_k \|E(x_t) - c_k\|\right)$$
- **Evidence status:** Not run. The expected finding is a testable prediction,
  not a result. It must not be included as evidence until perturbation
  experiments are executed and analyzed.

---

## 5. Defensible paper framing

### Proposed Working Title:
> **"Decoupling Routing Organization from Closed-Loop Success: An Empirical Study of Structured Mixture-of-Experts in Robotic Manipulation"**

### Positioning:
PhaseForge should be framed as a structured MoE study with task-dependent
performance, not as a universally superior manipulation algorithm. The paper
can investigate whether routing organization and closed-loop success are
separable properties.

### Venue note:
Venue suitability should be decided after the manuscript and evidence are
complete. The current results alone do not justify predicting acceptance at a
specific venue.

---

## 6. Decision Points for the Supervisor

To finalize this research, the team requests the supervisor's approval on the following decisions:

1. **D1 — Scientific scope:** Use the conditional hypothesis: topology-derived
   initialization can organize routing, but its control benefit is task-
   dependent and is not established as universally superior.
2. **D2 — Manuscript evidence:** Include the completed routing and commanded-
   action analyses only with their stated limits. Treat switch partitioning
   and perturbation sensitivity as future work unless they are actually run.
3. **D3 — Model description:** Do not describe the beta-zero model as a
   feedback or impedance controller. Use the exact implementation name and
   state that `beta=0.0` removes the residual branch.

The recorded metrics are preserved, but the copied training checkpoints are
not present locally. The archive is therefore reviewable but not currently
independently reloadable. Any reproducibility statement must acknowledge
that limitation.
