"""Wave A4: specialization matrix M_{z,e}.

Offline diagnostic. For each trained stage-2 model (best checkpoint):
- encode validation states
- compute the full softmax gate distribution p(e|z) over the router
- compute M_{z,e} = E_latents_in_phase_z[p(e)]
- also report M_{z_pred,e} using the predicted phase (oracle-like)

Outputs:
    outputs/surgical/_findings/specialization_matrix.json
    docs/dev/findings/specialization_matrix.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiments._model_utils import (  # noqa: E402
    build_model_and_load,
    build_val_loader,
    device_from,
    gate_probs,
    iter_val_batches,
    stage2_ckpt,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/specialization_matrix.md")


def _contingency(outputs: Path, seed: int, max_samples: int, device) -> dict:
    ckpt = stage2_ckpt(outputs, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint_best.pt missing at {ckpt}")
    model, cfg = build_model_and_load(ckpt, device)
    val_loader = build_val_loader(cfg, max_samples=max_samples)
    counts = torch.zeros((cfg.models.phase_head.num_phases, cfg.models.router.num_experts), dtype=torch.float64)
    counts_pred = torch.zeros_like(counts)
    n_per_phase = torch.zeros(cfg.models.phase_head.num_phases, dtype=torch.long)
    n_total = 0
    for batch in iter_val_batches(model, val_loader, device):
        p = gate_probs(model, batch["latent"].to(device)).cpu().double()
        z = batch["phase_true"].long()
        zp = batch["phase_pred"].long()
        counts.index_add_(0, z, p)
        counts_pred.index_add_(0, zp, p)
        n_per_phase += torch.bincount(z, minlength=counts.shape[0])
        n_total += p.shape[0]
    M = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    M_pred = counts_pred / counts_pred.sum(dim=1, keepdim=True).clamp(min=1)
    diag = torch.diagonal(M).mean().item()
    diag_pred = torch.diagonal(M_pred).mean().item()
    dominant = M.argmax(dim=1).tolist()
    return {
        "M": M.tolist(),
        "M_pred": M_pred.tolist(),
        "diag_mean_true": diag,
        "diag_mean_pred": diag_pred,
        "dominant_expert_per_phase_true": dominant,
        "n_per_phase": n_per_phase.tolist(),
        "n_total": n_total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--outputs", default="outputs/surgical")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    device = device_from("auto")
    per_seed: dict[str, dict] = {}
    pooled_count: torch.Tensor | None = None
    pooled_count_pred: torch.Tensor | None = None
    n_pooled = 0

    for seed in seeds:
        try:
            res = _contingency(outputs, seed, args.max_samples, device)
        except FileNotFoundError as exc:
            print(f"[spec] seed {seed}: {exc}")
            per_seed[str(seed)] = {"missing": str(exc)}
            continue
        per_seed[str(seed)] = res
        n_per_phase = torch.tensor(res["n_per_phase"], dtype=torch.double)
        weighted = torch.tensor(res["M"], dtype=torch.double) * n_per_phase.unsqueeze(1)
        weighted_pred = torch.tensor(res["M_pred"], dtype=torch.double) * n_per_phase.unsqueeze(1)
        if pooled_count is None:
            pooled_count = weighted
            pooled_count_pred = weighted_pred
        else:
            pooled_count = pooled_count + weighted
            pooled_count_pred = pooled_count_pred + weighted_pred
        n_pooled += int(res["n_total"])
        print(f"[spec] seed {seed}: diag(true)={res['diag_mean_true']:.3f} "
              f"diag(pred)={res['diag_mean_pred']:.3f} n={res['n_total']}")

    pooled_M = (pooled_count / pooled_count.sum(dim=1, keepdim=True).clamp(min=1)).tolist() if pooled_count is not None else None
    pooled_M_pred = (pooled_count_pred / pooled_count_pred.sum(dim=1, keepdim=True).clamp(min=1)).tolist() if pooled_count_pred is not None else None
    pooled = {
        "M": pooled_M,
        "M_pred": pooled_M_pred,
        "diag_mean_true": sum(torch.diagonal(torch.tensor(pooled_M)).tolist()) / len(pooled_M) if pooled_M else None,
        "diag_mean_pred": sum(torch.diagonal(torch.tensor(pooled_M_pred)).tolist()) / len(pooled_M_pred) if pooled_M_pred else None,
        "n_total": n_pooled,
    }

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "max_samples": args.max_samples,
        "per_seed": per_seed,
        "pooled": pooled,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "specialization_matrix.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[spec] done -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave A4 — Specialization matrix M_{z,e}",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        "Diagonal mean is the average P(e=phase) along the main diagonal of the",
        "row-normalized contingency; 1/E (=0.167 for E=6) is random routing.",
        "",
        "| seed | diag(true) | diag(pred) | n |",
        "|---|---|---|---|",
    ]
    for s in payload["seeds"]:
        r = payload["per_seed"].get(str(s), {})
        if "missing" in r:
            lines.append(f"| {s} | MISSING | MISSING | - |")
            continue
        lines.append(f"| {s} | {r['diag_mean_true']:.3f} | {r['diag_mean_pred']:.3f} | {r['n_total']} |")
    p = payload["pooled"]
    pooled_row = (
        f"| pooled | {p['diag_mean_true']:.3f} | {p['diag_mean_pred']:.3f} | {p['n_total']} |"
        if p["diag_mean_true"] is not None
        else "| pooled | - | - | 0 |"
    )
    lines += [
        pooled_row,
        "",
        "## Pooled M (true phase)",
        "",
    ]
    if p["M"]:
        lines.append("| phase \\ expert | " + " | ".join(str(e) for e in range(len(p["M"][0]))) + " |")
        lines.append("|" + "---|" * (len(p["M"][0]) + 1))
        for z, row in enumerate(p["M"]):
            lines.append(f"| {z} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    lines += ["", "## Interpretation", "", "- diag >> 1/E means experts specialize on the matching phase.", "- diag ≈ 1/E means routing ignores phase information.", ""]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
