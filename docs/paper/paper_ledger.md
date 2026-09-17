# 📄 Paper Writing Ledger

> **Last Updated:** 2026-09-17
> **Status Key:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

## Section Order & Writing Status

| #  | Section                        | Folder Name                  | Status | Priority |
|----|--------------------------------|------------------------------|--------|----------|
| 1  | Abstract                       | `abatract`*                  | 🟡     | HIGH     |
| 2  | Introduction                   | `1.introduction`             | 🟡     | HIGH     |
| 3  | Related Work                   | `2.Related_Work`             | 🟡     | MEDIUM   |
| 4  | Method                         | `3.method`                   | 🟡     | HIGH     |
| 5  | Experimental Setup             | `4.experimental_setup`       | 🟡     | MEDIUM   |
| 6  | Results                        | `6.Results`                  | 🟡     | HIGH     |
| 7  | Discussion & Limitations       | `7.discussion_limitations`   | 🟡     | MEDIUM   |
| 8  | Conclusion                     | `8.conclusion`               | 🟡     | LOW      |

---

## ⚠️ Notes

- **Typo:** Folder `abatract` → should be `abstract`
- **Missing #5:** No folder `5.*` exists (numbering jumps from 4 → 6)
- **Recommended writing order** (bottom-up):
  1. Method → 2. Experimental Setup → 3. Results → 4. Discussion → 5. Related Work → 6. Introduction → 7. Conclusion → 8. Abstract

---

## Suggested Writing Sequence

```text
Phase 1 – Core Content
  [/] 3.method              ← first draft complete (method.md)
  [/] 4.experimental_setup  ← first draft complete (experimental_setup.md)
  [/] 6.Results              ← first draft complete (results.md)

Phase 2 – Context & Analysis
  [/] 2.Related_Work        ← first draft complete (related_work.md)
  [/] 7.discussion_limitations ← first draft complete (discussion.md)

Phase 3 – Framing
  [/] 1.introduction          ← first draft complete (introduction.md)
  [/] 8.conclusion            ← first draft complete (conclusion.md)

Phase 4 – Final
  [/] abatract               ← first draft complete (abstract.md)
```