# Paper Sources

LaTeX manuscript sources go here (NeurIPS style until a venue is chosen;
adjust the class file at submission time — the content plan in
`../analysis_docs/figures_tables_plan.md` is venue-agnostic).

Expected layout once drafting starts:

```
paper/
├── main.tex            # single-column NeurIPS preprint build
├── neurips_*.sty, .bst # venue style (frozen copy)
├── sections/           # intro, related, method, experiments, discussion
├── figures/            # only final exported PDFs (generated, not hand-drawn)
└── tables/             # generated .tex table includes
```

Figures and tables are **generated** from `outputs_final/` by the plotting
scripts planned in `../analysis_docs/figures_tables_plan.md`; nothing in
`figures/` or `tables/` is edited by hand.
