"""Wave A2: SR vs val-MSE correlation across checkpoints.

Consumes the Wave A1 checkpoint sweep JSON and reports the correlation
between the model-selection criterion (val action MSE) and the actual
control objective (rollout SR) per seed and pooled, plus the early-epoch
(1..50) and late-epoch (100..200) sub-correlations.

Outputs:
    outputs/cpu_sweep/_findings/sr_val_corr.json
    docs/dev/findings/sr_val_corr.md

Usage:
    python scripts/experiments/sr_val_corr.py [--findings outputs/cpu_sweep/_findings/checkpoint_sweep.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    if den == 0:
        return None
    return num / den


def _corr_over(rows: dict, early: bool) -> float | None:
    xs, ys = [], []
    for label, r in rows.items():
        if label == "best":
            continue
        epoch = int(label)
        if early and epoch > 50:
            continue
        if not early and epoch < 100:
            continue
        if r.get("val_loss") is not None and r.get("sr") is not None:
            xs.append(r["val_loss"])
            ys.append(r["sr"])
    return _pearson(xs, ys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--findings", default="outputs/cpu_sweep/_findings/checkpoint_sweep.json")
    args = parser.parse_args(argv)

    src = (PROJECT_ROOT / args.findings).resolve()
    if not src.is_file():
        print(f"[FAIL] {src} does not exist — run checkpoint_sweep.py first")
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))

    out: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": str(src), "seeds": {}}
    for seed in payload["seeds"]:
        rows = payload["evals"].get(str(seed), {})
        xs = [r["val_loss"] for r in rows.values() if r.get("val_loss") is not None and r.get("sr") is not None]
        ys = [r["sr"] for r in rows.values() if r.get("val_loss") is not None and r.get("sr") is not None]
        out["seeds"][str(seed)] = {
            "corr_all": _pearson(xs, ys),
            "corr_early_1_50": _corr_over(rows, early=True),
            "corr_late_100_200": _corr_over(rows, early=False),
            "sr_at_selected": rows.get("best", {}).get("sr"),
            "peak_sr": max((r["sr"] for r in rows.values() if r.get("sr") is not None), default=None),
        }
    all_x, all_y = [], []
    early_x, early_y = [], []
    for seed_rows in payload["evals"].values():
        for label, r in seed_rows.items():
            if r.get("val_loss") is None or r.get("sr") is None:
                continue
            all_x.append(r["val_loss"])
            all_y.append(r["sr"])
            if label != "best" and int(label) <= 50:
                early_x.append(r["val_loss"])
                early_y.append(r["sr"])
    out["pooled"] = {
        "corr_all": _pearson(all_x, all_y),
        "corr_early_1_50": _pearson(early_x, early_y),
    }

    findings = PROJECT_ROOT / "outputs/cpu_sweep/_findings/sr_val_corr.json"
    findings.parent.mkdir(parents=True, exist_ok=True)
    findings.write_text(json.dumps(out, indent=2), encoding="utf-8")

    report = PROJECT_ROOT / "docs/dev/findings/sr_val_corr.md"
    lines = [
        "# Wave A2 — SR vs val-MSE correlation",
        "",
        f"Created {out['created']}. Source: {src.name}.",
        "",
        "| seed | corr(all) | corr(1..50) | corr(100..200) | SR@selected | peak SR |",
        "|---|---|---|---|---|---|",
    ]
    for seed, d in out["seeds"].items():
        def fmt(x):
            return "-" if x is None else f"{x:.3f}"
        lines.append(
            f"| {seed} | {fmt(d['corr_all'])} | {fmt(d['corr_early_1_50'])} | "
            f"{fmt(d['corr_late_100_200'])} | {fmt(d['sr_at_selected'])} | {fmt(d['peak_sr'])} |"
        )
    lines += [
        "",
        "| pooled | corr(all) | corr(1..50) |",
        "|---|---|---|",
        f"| - | {fmt(out['pooled']['corr_all'])} | {fmt(out['pooled']['corr_early_1_50'])} |",
        "",
        "## Interpretation",
        "",
        "- If corr is weak/negative → `val/loss_action` is a poor proxy for control success;",
        "  the model-selection protocol needs changing (better monitor, larger val split, or",
        "  selection by a rollout-validated metric).",
        "- If corr is strong → checkpoint selection is aligned and the spread comes from",
        "  elsewhere (expert specialization, routing, representation).",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] -> {findings}\n[ok] -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())