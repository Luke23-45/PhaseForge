# Publication Directory

Assembles everything that becomes the paper.

```
docs/publication/
├── paper/          # LaTeX sources, style files, and the compiled manuscript
│                   # (NeurIPS template for now — swap styles at submission)
└── analysis_docs/  # figure & table plans, analysis decision records, and the
                    # mapping from experiment artifacts to publication assets
```

Authoritative inputs live elsewhere and are **never duplicated here**:

- Experiment definitions: `experiments/five_task.json`, `experiments/lift_ablation.json`
- Run artifacts: `outputs_final/` (main matrix), `outputs_ablation/` (ablations)
- Summary statistics: `scripts/analysis/*` → `outputs_final/_summaries/`
- Protocol & claims rules: `docs/dev/final_baselines_plan.md` §9–§10,
  `docs/plan/specs/research_definition.md` §3–§4

Start with `analysis_docs/figures_tables_plan.md` — the complete publication
asset plan (figures, tables, encodings, main-vs-appendix split, and the
claims-to-evidence map).
