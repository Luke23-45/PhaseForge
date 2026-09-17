# Research-Paper Writing Guidelines — Reviewed & Finalized

Every point below was checked individually against the original. Most were already sound and are kept close to verbatim. A handful were tightened for precision, a few merged to remove overlap, and ten new points were added to close real gaps — grounded against established scientific-writing sources (full list at the end: Mensh & Kording 2017; Gopen & Swan 1990; Henderson et al. 2018; Agarwal et al. 2021; Kobak et al. 2024). Tags mark what changed: _(refined)_, _(merged)_, _(new)_. No tag = essentially unchanged.

**One flag, not a change:** **B3** encodes this paper's specific conceptual chain (behavioral structure → expert partition → topology as a prior → consequences). That's project-specific content with no independent way for me to verify — kept as written, worth re-confirming it still matches the current argument.

---

## A. Core Idea & Information Hierarchy

**A1. Write the scientific object, not the implementation.** The paper should describe the phenomenon, question, mechanism, and findings under study — not how the project was assembled. Rough test: if a sentence would read the same in an internal engineering doc, it belongs in the appendix, not the narrative.

**A2. Identify the one-sentence core idea before writing.** Reduce the paper to one sentence: what is this work trying to establish, reveal, or understand? Everything else — every section, figure, paragraph — must serve that sentence, or it moves to supplementary material.

**A3. Establish an explicit hierarchy of rhetorical weight.** _(refined + merged)_ Roughly: (1) the scientific contribution, (2) the question it answers, (3) the evidence, (4) the interpretation, (5) experimental controls, (6) implementation details. This ranks emphasis, not order of appearance on the page — the question is often posed just before the contribution is revealed. Don't let a lower level borrow a higher level's weight: a design restriction, architectural setting, training setting, or evaluation choice is not automatically a research idea, even when it took real engineering effort to get right.

**A4. Don't let an absence become the finding unless it is the finding.** _(refined)_ An experimental condition defined by removing something ("memoryless," "no recurrence," "single-step") may just describe the condition under study, not the contribution. Ask directly: is the absence itself the finding, or is it a control that makes a different finding measurable?

---

## B. Opening the Paper

**B1. Don't turn the introduction into a textbook.** Don't spend space on generic concepts your target audience already knows, unless the definition is load-bearing for your specific argument.

**B2. Begin with the research problem, not the architecture.** Establish why the phenomenon matters before describing the machinery used to study it.

**B3. Introduce the scientific object before the experimental apparatus.** _(flagged — see note above)_ Here: behavioral structure → expert partition → topology as a prior → consequences of that prior.

**B4. Motivate the proposed mechanism conceptually before its implementation.** Explain why it could matter scientifically, not merely how it's built.

**B5. State the central hypothesis or question once.** Let the experiment and results test it — don't keep re-announcing it.

**B6. Don't repeatedly announce what the paper is doing.** Replace recurring "we investigate whether…" with substantive statements about the underlying problem wherever possible.

**B7. Write the abstract as a complete, compressed story.** _(new)_ Context and the specific gap → the approach, in one sentence → the main result → what it means. A reader who stops after the abstract should have the finding, not just the topic. (Mensh & Kording's Rule 5: for most readers, the abstract is the only part of the paper they will read.)

**B8. Let the title point at the finding, not just the system's name.** _(new)_ A title built only from an internal project name or a generic "X: A Framework for Y" tells the reader what was built, not what was found. Mensh & Kording treat the title as the single most important sentence in the paper — worth returning to and re-honing throughout writing, not fixing early and forgetting.

---

## C. Section-Level Architecture

**C1. Keep method, results, and discussion in their lanes.** Method says what was done; results say what happened; discussion says what it means.

**C2. Don't put interpretation into the method just because it explains why you configured something.** State the configuration; save what it means for later.

**C3. Don't put raw implementation bookkeeping into the discussion.** If it's not shaping the reader's interpretation, it belongs in the protocol or supplement.

**C4. Each section should answer a scientific question.** Avoid sections whose purpose is simply "here is another part of the implementation."

---

## D. Constraints, Controls, and Method Framing

**D1. Explain constraints by what they isolate, not by what they remove.** "We removed X" tells the reader what changed; it doesn't tell them why that change is informative.

**D2. State rationale as fact, not as narrated decision-making.** _(refined)_ Avoid narrating the authors' decision journey ("we decided to…," "we initially tried…," "we wanted to…") unless the decision itself is part of the scientific story. This differs from stating a methodological fact — "we use X because it satisfies property P" is not a narrated decision and should stay. Test: does removing the sentence remove information the reader needs to interpret the results? If not, cut it.

**D3. Don't write for an imagined reviewer.** Avoid language that exists only to pre-empt criticism ("to be fair," "admittedly"). Address a real limitation once, in the right place — don't scatter pre-emptive hedges through the narrative.

**D4. Treat controls as controls.** Explain why a control is necessary, then move on.

**D5. State a control's role once, clearly, and don't let it upstage the treatment.** _(merged)_

**D6. Use technical detail — including equations — only when it clarifies the argument.** Equations should define the object being studied, not serve as implementation documentation.

**D7. Push reproducibility bookkeeping down the hierarchy.** Seeds, horizons, reset banks, coefficients, optimizer settings, objective variants, configuration names, and artifact paths belong in the experimental protocol or supplementary material — unless one of them directly determines how a reader should interpret a result.

**D8. Don't let internal configuration names become prose.** Backticks, filenames, branch names, and code terminology shouldn't dominate the narrative.

**D9. Use the experimental protocol to earn comparability — not to become the story.** Its job is convincing the reader the comparison is fair.

---

## E. Causal Rigor in Comparisons

**E1. Keep causal comparisons explicit.** Clearly distinguish what an experiment actually isolates from what it merely correlates with.

**E2. Never merge different experimental questions into one result.** A run measuring initialization effects and a run characterizing the final system play different evidentiary roles — present them as such.

**E3. Don't treat unmatched experiments as causal comparisons, and say so when a comparison isn't matched.** _(refined)_ A result from a different objective, configuration, or protocol shouldn't be read as if it isolates the same variable as your main comparison — and if it doesn't, say that explicitly rather than letting the reader assume otherwise.

**E4. Report variability, not just a point estimate, wherever a conclusion depends on the size of a difference.** _(new)_ This matters most under stochastic training (seeds, initialization, sampling): Henderson et al. (2018) showed run-to-run variance in RL is often larger than the effect being claimed, and Agarwal et al. (2021) is now the standard reference for reporting interval estimates rather than a single mean or median. If two conditions' variability ranges overlap substantially, don't call the difference clear or robust.

---

## F. Metrics & Measurement

**F1. Separate structure from performance when they're different scientific quantities.** A better internal organization doesn't automatically imply a better controller.

**F2. Define what each metric actually measures.** NMI, switch rate, success rate, action jumps, etc. should keep distinct interpretations.

**F3. Don't use one metric as a proxy for a different property without evidence.** Routing coherence is not physical stability; phase alignment is not task success; an action discontinuity is not automatically a failure mechanism.

**F4. Establish a metric's construct validity before leaning on it.** _(new)_ If you introduce a metric, or repurpose one, show it measures what you say it measures — an independent check, a known relationship to an established measure, or an explicit argument for why face validity suffices here. A metric's name is not evidence of what it measures.

---

## G. Reporting Results

**G1. Write results around observations, not around tables.** The prose should tell the reader what matters in the table, not repeat its entries.

**G2. Highlight the contrasts that are scientifically informative.** A discrepancy between routing organization and task success can matter more than either number alone.

**G3. State the result plainly; keep its evidentiary boundary in its own sentence.** _(merged — combines "state claims directly," "don't bury under caveats," and "avoid defensive formulations")_ One clean sentence, with the verb calibrated to what the evidence supports ("suggests," "is consistent with," "demonstrates" — whichever is true). That's different from wrapping the finding in commentary about how important it is, and different from folding a defensive qualifier ("this does not prove…") into the same sentence as the finding. If a hedge is genuinely necessary, give it its own sentence, placed once, near — not inside — the result it qualifies.

**G4. Distinguish observation from explanation.** Don't speculate beyond the measured evidence, especially for mechanism claims.

**G5. When evidence is incomplete, say exactly what is unmeasured.** Don't infer a physical explanation from an indirect trace.

**G6. Treat negative or mixed results as information, not as failures needing rhetorical defense.**

**G7. Actively look for separations, contradictions, and non-equivalences.** They often carry the real contribution.

**G8. Ask of every result: what does this teach about the underlying object, not just this configuration?**

**G9. Make each figure and table self-contained.** _(new)_ A reader should get the point from the figure and caption alone. Mensh & Kording are specific here: the figure's title should state the conclusion, the legend should explain how it was obtained — many readers go straight from the abstract to the figures and never read the body text.

---

## H. Limitations & Conclusion

**H1. Put limitations after establishing the result, not folded into every sentence that states it.**

**H2. State each caveat once.** Repeating it across abstract, results, interpretation, limitations, and conclusion reads as defensiveness, not rigor.

**H3. Don't write the conclusion as an apology.** A bounded result is still a result.

**H4. The conclusion answers "what did we learn," not "what did we implement."**

**H5. State the contribution so it would still hold if the project's name disappeared.** _(merged)_ The system demonstrates the idea; it is not the idea, and the two shouldn't become synonymous.

---

## I. Literature & Related Work

**I1. Use literature to establish real intellectual context, not to manufacture motivation.**

**I2. Don't import a concern from an adjacent field merely because it sounds related.** A language-model MoE result is not automatic motivation for a robotics claim.

**I3. Build the literature argument as a chain.** Prior concept → the specific unresolved issue → your specific question → your intervention → your evidence.

**I4. State the actual gap.** "Prior work motivates our question" is a placeholder, not an argument, until the gap is named.

**I5. Organize related work around the dimensions of your argument, not as an annotated list.** _(new)_ "X did A. Y did B. Z did C." tells the reader what happened in the field, not how it bears on your question. Group prior work by what it agrees with, contradicts, or leaves open relative to your specific claim.

---

## J. Prose & Sentence-Level Craft

**J1. Individual transition words are fine; mechanical, repeated use to narrate the paper is the problem.** "However," "therefore," "this motivates," "we investigate" are each fine once — the failure mode is using them as the connective tissue of the whole paper.

**J2. Avoid phrasing that reads as unedited AI output.** _(new)_ Kobak et al. (2024) documented a measurable post-2022 spike in words like "delve," "underscores," and "showcasing" across millions of PubMed abstracts — these are now recognizable tells, not neutral vocabulary. If a sentence would read identically in a paper about an unrelated topic, rewrite it in terms specific to your object of study.

**J3. Optimize for conceptual compression, not vocabulary sophistication.** Professional writing requires knowing what information can be omitted, not complicated words.

**J4. Prefer one precise sentence over three explanatory ones.**

**J5. Don't explain an implication that's already obvious from the evidence just given.** Trust the reader.

**J6. Don't explain every logical step at the same granularity.** Important ideas get detail; routine facts don't.

**J7. Use paragraphs to make an argument, not to store information.** Each paragraph should have one purpose: establish a problem, introduce an idea, report an observation, or interpret it.

**J8. Put a sentence's point at the end; use the beginning to connect to what the reader already knows.** _(new)_ Gopen & Swan's classic analysis: readers expect old, familiar information in the "topic position" (the start of a sentence) and new, important information in the "stress position" (the end). When a passage is hard to follow, check whether the key idea is stranded in the middle of a sentence instead of landing at the end — that's often a structural problem, not a content problem.

**J9. Prefer active, attributed statements over passive constructions that hide who's claiming what.** _(new)_ "We find that…" over "It was found that…" — Nature and Science both explicitly prefer active voice for this reason. Passive still earns its place when the object of study, not the authors, should be the sentence's subject — e.g., "the gripper was actuated at 10 Hz," where the gripper is what matters, not "we."

---

## K. From Project to Paper

**K1. Don't write the paper in the order the experiments were run.** Chronology is not scientific narrative.

**K2. Don't preserve the internal documentation's structure when converting a project into a paper.** Documentation optimizes for reproducibility and maintenance; a paper optimizes for understanding.

**K3. Before finalizing, cut anything that exists only to prove the experiment was carefully engineered.** Let the protocol demonstrate rigor; don't let rigor become the argument.

---

## L. Self-Checks Before Submission

**L1. Would this paragraph still matter if the implementation changed?** If not, it belongs in the experimental details, not the narrative.

**L2. Is this sentence telling the reader what we built, or what we learned?** Prefer the latter.

**L3. Read only the first sentence of each paragraph, in order.** _(new)_ If those sentences alone don't add up to a coherent version of the argument, either the paragraphs are misordered or one of them is missing a real topic sentence.

**L4. Final test.** After only the abstract and introduction, a reader should be able to answer: What is the scientific problem? What is the proposed idea? What is genuinely new or informative? What evidence tests it? What was learned? If they'd remember the implementation before the answer to any of these, revise.

---

## Further Reading

- Mensh, B. & Kording, K. (2017). "Ten simple rules for structuring papers." _PLOS Computational Biology_, 13(9): e1005619.
- Gopen, G. D. & Swan, J. A. (1990). "The Science of Scientific Writing." _American Scientist_, 78(6): 550–558.
- Henderson, P. et al. (2018). "Deep Reinforcement Learning that Matters." _AAAI_.
- Agarwal, R. et al. (2021). "Deep Reinforcement Learning at the Edge of the Statistical Precipice." _NeurIPS_ (Outstanding Paper Award).
- Kobak, D. et al. (2024). "Delving into ChatGPT usage in academic writing through excess vocabulary." arXiv:2406.07016.