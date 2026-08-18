"""Wave C1: latent geometry by phase.

Offline diagnostic on the trained stage-2 best checkpoint. For each
phase ``z``:
- per-phase centroid ``c_z`` (mean latent)
- mean intra-phase distance ``d_intra(z) = E_x ||x - c_z|| / σ``
- pairwise between-phase centroid distance ``d_inter(z, z') = ||c_z - c_z'|| / σ``
- silhouette coefficient (sklearn-free closed-form)

``σ`` is the global latent std. Distances normalized by ``σ`` make the
"compactness vs separation" claim scale-free.

Outputs:
    outputs/surgical/_findings/latent_geometry.json
    docs/dev/findings/latent_geometry.md
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
    iter_val_batches,
    stage2_ckpt,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/latent_geometry.md")


def _silhouette_simple(latents: torch.Tensor, labels: torch.Tensor) -> dict:
    by_phase = {}
    for x, l in zip(latents, labels):
        by_phase.setdefault(int(l.item()), []).append(x)
    means = {z: torch.stack(xs, dim=0).mean(dim=0) for z, xs in by_phase.items() if xs}
    if len(by_phase) < 2:
        return {"mean": None, "per_phase": {}}
    pairs = sorted(by_phase.keys())
    per_phase = {}
    for z in pairs:
        xs = torch.stack(by_phase[z], dim=0)
        a = xs.sub(means[z]).norm(dim=-1).mean().item()
        b_vals = []
        for zp in pairs:
            if zp == z:
                continue
            d = xs.sub(means[zp]).norm(dim=-1).mean().item()
            b_vals.append(d)
        b = min(b_vals) if b_vals else 0.0
        per_phase[str(z)] = {"a": a, "b": b, "s": (b - a) / max(a, b, 1e-12)}
    overall = sum(p["s"] for p in per_phase.values()) / len(per_phase)
    return {"mean": overall, "per_phase": per_phase}


def _geometry(outputs: Path, seed: int, max_samples: int, device) -> dict:
    ckpt = stage2_ckpt(outputs, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint_best.pt missing at {ckpt}")
    model, cfg = build_model_and_load(ckpt, device)
    val_loader = build_val_loader(cfg, max_samples=max_samples)
    chunks_lat, chunks_z = [], []
    seen = 0
    for batch in iter_val_batches(model, val_loader, device):
        chunks_lat.append(batch["latent"])
        chunks_z.append(batch["phase_true"])
        seen += batch["latent"].shape[0]
        if seen >= max_samples:
            break
    latents = torch.cat(chunks_lat, dim=0)[:max_samples]
    labels = torch.cat(chunks_z, dim=0)[:max_samples].long()
    sigma = latents.std(dim=0).mean().item() + 1e-12
    by_phase = {}
    for x, l in zip(latents, labels):
        by_phase.setdefault(int(l.item()), []).append(x)
    centroids = {z: torch.stack(xs, dim=0).mean(dim=0) for z, xs in by_phase.items() if xs}
    intra = {}
    for z, xs in by_phase.items():
        xs = torch.stack(xs, dim=0)
        intra[str(z)] = float(xs.sub(centroids[z]).norm(dim=-1).mean().item() / sigma)
    inter = {}
    phases = sorted(centroids.keys())
    for i, z in enumerate(phases):
        for zp in phases[i + 1:]:
            inter[f"{z}-{zp}"] = float(centroids[z].sub(centroids[zp]).norm().item() / sigma)
    sil = _silhouette_simple(latents, labels)
    n_per = {str(z): len(xs) for z, xs in by_phase.items()}
    return {
        "centroids": {str(z): centroids[z].tolist() for z in phases},
        "intra": intra,
        "inter": inter,
        "silhouette": sil,
        "n_per_phase": n_per,
        "sigma": sigma,
        "n_total": int(latents.shape[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--outputs", default="outputs/surgical")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    device = device_from("auto")
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        try:
            res = _geometry(outputs, seed, args.max_samples, device)
        except FileNotFoundError as exc:
            print(f"[geom] seed {seed}: {exc}")
            per_seed[str(seed)] = {"missing": str(exc)}
            continue
        per_seed[str(seed)] = res
        inter_mean = sum(res["inter"].values()) / max(len(res["inter"]), 1)
        intra_mean = sum(res["intra"].values()) / max(len(res["intra"]), 1)
        sil = res["silhouette"].get("mean")
        sil_s = f"{sil:.3f}" if sil is not None else "-"
        print(f"[geom] seed {seed}: inter_mean={inter_mean:.3f} intra_mean={intra_mean:.3f} "
              f"silhouette={sil_s} n={res['n_total']}")

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "per_seed": per_seed,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "latent_geometry.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[geom] done -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave C1 — Latent geometry by phase",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        "Distances are normalized by the global latent std ``σ``. A silhouette",
        "score of +1 means clusters are well-separated from neighbours.",
        "",
        "| seed | intra mean | inter mean | ratio inter/intra | silhouette |",
        "|---|---|---|---|---|",
    ]
    for s in payload["seeds"]:
        r = payload["per_seed"].get(str(s), {})
        if "missing" in r:
            lines.append(f"| {s} | MISSING | MISSING | - | - |")
            continue
        inter_mean = sum(r["inter"].values()) / max(len(r["inter"]), 1)
        intra_mean = sum(r["intra"].values()) / max(len(r["intra"]), 1)
        ratio = inter_mean / intra_mean if intra_mean > 0 else None
        sil = r["silhouette"].get("mean")
        sil_s = f"{sil:.3f}" if sil is not None else "-"
        lines.append(
            f"| {s} | {intra_mean:.3f} | {inter_mean:.3f} | {ratio:.2f} | {sil_s} |"
        )
    lines += ["", "## Interpretation", "", "- High silhouette and high inter/intra ratio: phase geometry is structured.", "- Low silhouette: latents do not cluster by phase, regardless of routing.", ""]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
