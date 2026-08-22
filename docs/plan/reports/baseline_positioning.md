# PhaseForge — Baseline Coverage Positioning (D9 record)

**Status:** decision record + paper positioning material. Survey date:
2026-08-22. Companion to `../specs/state_only_rollout_implementation_plan.md`
(§5 primary comparison family) and `../specs/research_definition.md`.

## Decision

**No additional trained baseline methods are added to the final five-task
matrix.** The published comparison set is:

1. **Primary family (multiplicity-corrected):** PhaseForge vs BC-MLP,
   Scratch MoE, Warm-Start MoE, Phase-Pretrain Random-Router,
   Plain-Encoder Phase-Bootstrap.
2. **Context rows (intervals reported, outside the corrected family):**
   BC-Large (per-task deployed parameter match +0.8%…+1.8%), BC-RNN
   (strongest temporal comparator; 5.6× BC parameters, not history-matched
   to the single-state contract — disclosed).
3. **Diagnostics (never deployable-performance evidence):** robot-only BC,
   teacher-forced, oracle routing (pending D10).

The strength of the design is the exactly-matched mechanism factorial: the
proposed method and all H1–H4 controls share deployed capacity (382,646
parameters at Lift dims) and, at equal training seed, the **identical
dropped-neuron index set** in the 50% partial warm-start — verified
end-to-end through the registry in implementation ledger Phase 3.

## Literature context (NOT re-run comparators)

The following published numbers appear on the same benchmark family but
under **different evaluation protocols** (episode counts, reset
distributions, dataset variants, action conventions). They are cited as
context only and must be clearly labeled as such in the paper. They are
never pooled with, or plotted against, this project's own results.

| Source | Methods reported | Relevance |
|---|---|---|
| Mandlekar et al., CoRL 2021 (robomimic study; arXiv:2108.03298) | BC, BC-RNN, IBC, BCQ, CQL on Lift/Can/Square/ToolHang/Transport (PH/MH) | The benchmark's own baseline suite; establishes BC-RNN as the strongest IL baseline on human demonstrations |
| Chi et al. 2023 (Diffusion Policy) and follow-ups applying low-dim DP to robomimic | DP vs LSTM-GMM, IBC, BET (low-dim state track) | The modern generative comparator family; context for where state-only IL stands |

## Prepared defenses for "why no X?" questions

**Why no offline RL (CQL/IQL/BCQ)?**
The benchmark's own study concludes that offline RL underperforms BC-RNN on
human-demonstration data (e.g., Can MH: BCQ 62.7%, CQL 22.0%, vs BC-RNN
100%). This project's datasets are proficient-human (PH) demonstrations
(`dataset_type: "ph"`), squarely in the regime where imitation is the
appropriate family. Exclusion is cited, not assumed.

**Why no Diffusion Policy?**
The declared protocol holds the policy class fixed — deterministic,
single-step, state-only — across every compared method so the comparison
isolates the routing/initialization factor. Diffusion policies require
observation history and action chunking (a different observation/action
contract); BC-RNN already fills the "stronger, non-contract-matched
temporal comparator" slot with disclosure. Published low-dim DP numbers are
provided as literature context (table above).

**Why no GMM / stochastic heads or BC-Transformer?**
Stochastic policy classes would vary the output distribution family in
addition to the studied factor; BC-RNN already represents the temporal
family at 5.6× BC capacity with full disclosure. The uniform policy class
is part of the registered protocol.

**Why is this baseline set sufficient?**
It contains the benchmark study's strongest IL baseline (BC-RNN), a
parameter-matched dense capacity control (BC-Large, verified per task in
preflight), an architecture control (Scratch MoE), an upcycling control
(Warm-Start MoE), and an exactly-matched four-way mechanism factorial for
the causal claims (H1–H4). The paper's claim is a mechanism claim under a
controlled protocol, not a leaderboard claim.

## Contingency

If a reviewer explicitly demands a modern generative baseline, the
pre-agreed fallback is a low-dim Diffusion Policy added as a clearly-labeled
non-contract-matched reference (same disclosure status as BC-RNN). This is
a post-review revision decision (ledger D9), not part of the registered
final sweep.
