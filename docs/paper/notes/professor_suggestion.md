The paper now has a defensible scientific story. The remaining risks are mostly internal precision, not the core claim. I reviewed only the pasted paper text; I did not re-check code, data, or citations.

## Must fix before freezing content

1. **Resolve the NMI-label inconsistency.**  
   The paper says NMI aligns experts with rule-derived `phase` labels in the Introduction and contributions, but §4.4 defines NMI against \(r_t\), which §3.2 defines as the trajectory-derived `phase_topo` regime label. These cannot both be true. Based on the facts you provided, change §4.4 to use the rule-derived label, e.g. \(y_t^{\text{phase}}\), and state this explicitly.

2. **Correct “cluster centers” in the Abstract and Introduction.**  
   The router is initialized from **latent centroids of Stage-1 representations grouped by `phase_topo`**, not directly from K-means centers in task-variable space. “Resulting cluster centers initialize prototypes” is technically misleading.

3. **Remove the phrase “ground-truth regime labels.”**  
   In Teacher-Forced, the labels are offline-generated labels, not environment ground truth. Name the exact field used (`phase` or `phase_topo`) and call them “offline labels.”

4. **Fix the notation collision.**  
   \(\beta\) denotes both the residual-feedback coefficient in §3.1 and the change-point penalty in §3.2. Rename one, preferably the segmentation penalty to \(\lambda_{\mathrm{cp}}\).

5. **Do not call all five ablation conditions “matched initialization conditions.”**  
   Only the three prototype arms are matched. Plain-BC changes representation; Softmax changes routing architecture. §5.2 must say this consistently.

6. **Fix the Can result sentence.**  
   “Regime initialization leads the prototype-based arms, with random and softmax…” is internally wrong: Softmax is not prototype-based. Separate the three matched prototype arms from the diagnostic Softmax control.

7. **Remove the unsupported Can trend claim.**  
   “More phase-aligned conditions tend to succeed more often” is not supported: Softmax has the highest NMI but is below regime initialization on Can. Report the ordering without inferring a trend.

## Abstract

- Replace “cluster centers” with “Stage-1 latent centroids grouped by trajectory-derived regime labels.”
- State that NMI measures alignment with **rule-derived phase labels**, if that is the confirmed target.
- Clarify that Rule/Random are the matched prototype controls; Softmax is a separate routing-mechanism diagnostic.
- State whether NMI and switch-rate values are means across seeds. Use consistent rounding: either 76% throughout or 76.0% throughout.
- The final sentence is strong and appropriate; retain its restricted scope.

## Introduction

- The research question and overall narrative are now good.
- Contribution 1 needs the same latent-centroid correction.
- Replace “controlled causal ablation” in Contribution 3 with “matched initialization ablation.” You can say it isolates initialization **within the stated configuration**, but avoid broad causal language.
- Change “establishing” to “showing an empirical dissociation in the evaluated setting.”
- If the title is not handled elsewhere, add one that names the scientific object rather than the software system.

## Related Work

- The section is structurally sound.
- Do not retain “to the best of our knowledge” until the literature search and bibliography are complete.
- Every citation must later be checked for relevance, especially the claims about MoE initialization path dependence.
- Add a stronger comparison to the closest prior work on modular imitation policies or trajectory-segmented policy learning once the reference audit begins.

## Method

- Explain how hard top-1 dispatch is optimized. The paper currently defines an \(\arg\min\) router but does not clearly state whether dispatch is treated as non-differentiable, uses a surrogate, or how prototypes receive gradients from each loss.
- State that `phase` is rule-derived and `phase_topo` is segmentation-and-clustering-derived everywhere they first appear. The distinction is central to the paper.
- Add one sentence noting that the full proposed configuration uses rule-derived phase supervision in Stage 1 and the Stage-2 margin term; the matched ablation disables the margin term. This prevents readers from incorrectly interpreting the method as unsupervised.
- Specify how the task-variable signal is scaled before a squared-Euclidean segmentation cost. Position, quaternion, gripper, and object variables have different units; the paper must state normalization or weighting.
- “Stationary kinematic behavior” is too strong for trajectories represented by position. Prefer “locally homogeneous task-variable statistics.”
- Give the observability-probe threshold, occupancy threshold, and validation protocol in the appendix.
- Define the expert reinitialization proportion and which layers are perturbed in the appendix.

## Experimental Setup

- In §4.1, change “approach, grasp, transport, insertion” for both Can and Square. Can is placement, not insertion. Use “approach, grasp, transport, and task-specific terminal placement or insertion.”
- §4.2 says the five-task sweep is in “§4.2,” which is a self-reference. It should point to §5.1 or simply say “the five-task benchmark.”
- Static Rule needs the actual thresholds in an appendix table.
- In §4.4, make the NMI target label exact; do not call `phase` a regime label if `phase_topo` is the regime label.
- Clarify whether the 20 validation demonstrations are used for checkpoint selection. If they are, call them a validation split, not an independent held-out test set.
- State how NMI and switch rate are aggregated across demonstrations and seeds.
- State that Wilson intervals describe pooled rollout episodes, not uncertainty over independently trained seeds.
- Define switch rate as transitions **within a trajectory**, excluding trajectory boundaries.
- Action continuity is not reported as evidence. Either move it to an appendix/future-diagnostic paragraph or state explicitly that it was not analyzed in the present results.

## Results

- Explicitly warn readers that the full-suite values (e.g., Can 78%, Square 41%) and focused-ablation values (Can 76%, Square 27%) come from different configurations. Otherwise they look contradictory.
- Replace “outperforms” and “advantage” in the five-task sweep with “records a higher observed success rate,” because the three-seed comparisons are not inferentially resolved.
- Replace “beyond the capability frontier” with “unresolved under the evaluated memoryless policies and training budget.”
- Remove the macro-average unless its uncertainty source is clearly defined and it serves a concrete argument. It is currently dominated by saturated and floor tasks, as the paper itself admits.
- Change “reliably shapes” to “is associated with” or “produces in the observed runs.”
- Keep Softmax explicitly diagnostic throughout; do not let its high NMI become evidence about prototype initialization.

## Discussion and Limitations

- Add a limitation that the representation is already trained with rule-derived `phase` supervision. The study isolates the **additional effect of prototype initialization** under that representation; it does not show that trajectory-derived regimes alone produce routing organization.
- Add a limitation that \(K=E=6\) is fixed rather than selected adaptively.
- If image observations are excluded, add this as a scope limitation only. Do not claim it explains Square.
- Replace “hard top-1 routing produces action discontinuities” with “can produce action discontinuities.”
- Replace “a structured partition concentrates transitions at phase boundaries” with “may concentrate transitions…” unless you measured boundary locations directly.
- Keep the Square mechanisms explicitly labeled hypotheses.
- Separate `phase` and `phase_topo` in the phase-label-validity limitation; the current wording still risks conflation.

## Conclusion

- It is strong overall.
- Replace “routing organization and task success respond to the same intervention” with “within the matched prototype ablation, both outcomes are evaluated under changes to initialization geometry.”
- Keep the final paragraph as an open question; do not convert the suggested wrong-expert mechanism into a conclusion.
- Prefer “does not establish uniform improvement in closed-loop control” over “the answer is no.” It is more precise for two informative tasks and three seeds.

## Required later, but not for this content revision

- Bibliography and citation audit.
- Tables, figures, captions, and appendix.
- Exact software/data versions, checkpoint protocol, reset-bank provenance, and hyperparameters.

No edits were made.