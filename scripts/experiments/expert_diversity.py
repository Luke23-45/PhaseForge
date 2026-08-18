"""Wave B2: expert-output divergence over training time.

For each seed, loads the per-epoch checkpoints saved by Wave A1
(``outputs/surgical/phaseforge/stage2/seed{S}/checkpoints/checkpoint_epoch_{t:04d}.pt``)
at t in {1, 5, 20, 200}. Encodes a fixed batch of validation states
(capped at 8192), passes them through every expert, and reports the
pairwise cosine distance between the resulting per-expert mean
activation vectors: ``D_ij(t) = 1 - cos(μ_i(t), μ_j(t))``.

Outputs:
    outputs/surgical/_findings/expert_diversity.json
    docs/dev/findings/expert_diversity.md
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
    expert_outputs,
    iter_val_batches,
    stage2_ckpt,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/expert_diversity.md")

EPOCHS = [10, 30, 100, 200]


def _collect_latents(model, val_loader, device, max_samples: int) -> torch.Tensor:
    chunks = []
    seen = 0
    for batch in iter_val_batches(model, val_loader, device):
        chunks.append(batch["latent"])
        seen += batch["latent"].shape[0]
        if seen >= max_samples:
            break
    return torch.cat(chunks, dim=0)[:max_samples]


def _cos_distance(mean_i: torch.Tensor, mean_j: torch.Tensor) -> float:
    ni = torch.nn.functional.normalize(mean_i, dim=-1)
    nj = torch.nn.functional.normalize(mean_j, dim=-1)
    return float(1.0 - torch.dot(ni, nj).item())


def _diversity(outputs: Path, seed: int, epoch: int, max_samples: int, device) -> dict:
    ckpt = stage2_ckpt(outputs, seed, epoch)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found")
    model, cfg = build_model_and_load(ckpt, device)
    val_loader = build_val_loader(cfg)
    latents = _collect_latents(model, val_loader, device, max_samples)
    if latents.shape[0] < 2:
        return {"epoch": epoch, "error": "too few latents"}
    outs = expert_outputs(model, latents.to(device))
    means = outs.mean(dim=0)
    E = means.shape[0]
    D = torch.zeros((E, E))
    for i in range(E):
        for j in range(E):
            D[i, j] = _cos_distance(means[i], means[j])
    iu = torch.triu_indices(E, E, offset=1)
    off_diag = D[iu[0], iu[1]]
    return {
        "epoch": epoch,
        "D": D.tolist(),
        "off_diag_mean": float(off_diag.mean().item()),
        "off_diag_min": float(off_diag.min().item()),
        "off_diag_max": float(off_diag.max().item()),
        "n_samples": int(latents.shape[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-samples", type=int, default=8192)
    parser.add_argument("--outputs", default="outputs/surgical")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    device = device_from("auto")
    per_seed: dict[str, dict[int, dict]] = {}
    summary = {e: [] for e in EPOCHS}

    for seed in seeds:
        per_seed[str(seed)] = {}
        for epoch in EPOCHS:
            try:
                res = _diversity(outputs, seed, epoch, args.max_samples, device)
            except FileNotFoundError as exc:
                print(f"[div] seed {seed} epoch {epoch}: {exc}")
                per_seed[str(seed)][str(epoch)] = {"missing": str(exc)}
                continue
            per_seed[str(seed)][str(epoch)] = res
            if "off_diag_mean" in res:
                summary[epoch].append(res["off_diag_mean"])
            print(f"[div] seed {seed} epoch {epoch}: off-diag mean = {res.get('off_diag_mean')}")

    overall = {
        str(epoch): {
            "mean_off_diag": (sum(v) / len(v)) if v else None,
            "min": min(v) if v else None,
            "max": max(v) if v else None,
            "n_seeds": len(v),
        }
        for epoch, v in summary.items()
    }

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "epochs": EPOCHS,
        "per_seed": per_seed,
        "summary": overall,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "expert_diversity.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[div] done -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave B2 — Expert-output divergence over time",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        "``D_ij(t) = 1 - cos(μ_i(t), μ_j(t))`` where μ_e is the mean expert output",
        "over a fixed batch of validation latents. Lower D = experts produce",
        "more similar outputs.",
        "",
        "| epoch | mean off-diag D | min | max | n seeds |",
        "|---|---|---|---|---|",
    ]
    for epoch in payload["epochs"]:
        s = payload["summary"][str(epoch)]
        if s["mean_off_diag"] is None:
            lines.append(f"| {epoch} | - | - | - | 0 |")
            continue
        lines.append(
            f"| {epoch} | {s['mean_off_diag']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n_seeds']} |"
        )
    lines += [
        "",
        "## Per-seed D_ij at t=200",
        "",
    ]
    for s in payload["seeds"]:
        r = payload["per_seed"].get(str(s), {}).get("200", {})
        if "D" not in r:
            lines.append(f"### Seed {s}: missing")
            continue
        D = r["D"]
        n = len(D)
        lines.append(f"### Seed {s}")
        lines.append("")
        lines.append("|       | " + " | ".join(f"e{e}" for e in range(n)) + " |")
        lines.append("|---" + "|---" * (n + 1))
        for i, row in enumerate(D):
            lines.append(f"| e{i} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
        lines.append("")
    lines += ["## Interpretation", "", "- Rising off-diag D over time = experts diverge (specialization).", "- Flat D = experts remain interchangeable.", ""]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
