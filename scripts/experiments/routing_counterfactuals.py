"""Wave A5: routing counterfactuals.

Offline diagnostic. For each validation sample we replace the router
distribution with five variants and measure the resulting action-MSE
against the ground-truth action. Variants:
- learned:     softmax(gate_linear(latent))
- oracle_true: one-hot of the *true* phase label
- oracle_pred: one-hot of the *predicted* phase label
- uniform:     1/E
- random:      fixed Dirichlet(1) draws, constant across samples
               (averaged over 4 RNG seeds for stability)

The script is offline (no rollouts). Each variant reports action MSE on
the held-out validation trajectories and the per-phase breakdown.

Outputs:
    outputs/surgical/_findings/routing_counterfactuals.json
    docs/dev/findings/routing_counterfactuals.md
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
    gate_probs,
    iter_val_batches,
    stage2_ckpt,
)

FINDINGS_DIR = Path("outputs/surgical/_findings")
REPORT_PATH = Path("docs/dev/findings/routing_counterfactuals.md")


def _variant_outputs(model, latents: torch.Tensor, phases: torch.Tensor, actions: torch.Tensor, n_random: int = 4) -> dict:
    device = latents.device
    expert_out = expert_outputs(model, latents)
    p_learned = gate_probs(model, latents)
    num_experts = p_learned.shape[-1]
    num_phases = int(phases.max().item()) + 1

    p_oracle_true = torch.zeros_like(p_learned)
    p_oracle_true.scatter_(1, phases.long().unsqueeze(1), 1.0)
    p_oracle_pred = torch.zeros_like(p_learned)
    p_oracle_pred.scatter_(1, model.phase_head(latents).argmax(dim=-1).unsqueeze(1), 1.0)
    p_uniform = torch.ones_like(p_learned) / num_experts
    p_random_avg = torch.zeros_like(p_learned)
    dist = torch.distributions.Dirichlet(torch.ones(num_experts, device="cpu"))
    for k in range(n_random):
        torch.manual_seed(1000 + k)
        p = dist.sample().squeeze(0).to(device)
        p_random_avg = p_random_avg + p.unsqueeze(0).expand_as(p_learned)
    p_random_avg = p_random_avg / n_random

    variants = {
        "learned":     (p_learned.unsqueeze(-1) * expert_out).sum(dim=1),
        "oracle_true": (p_oracle_true.unsqueeze(-1) * expert_out).sum(dim=1),
        "oracle_pred": (p_oracle_pred.unsqueeze(-1) * expert_out).sum(dim=1),
        "uniform":     (p_uniform.unsqueeze(-1) * expert_out).sum(dim=1),
        "random":      (p_random_avg.unsqueeze(-1) * expert_out).sum(dim=1),
    }
    out = {}
    for name, pred in variants.items():
        err = (pred - actions).pow(2).mean(dim=-1)
        out[name] = {
            "action_mse": float(err.mean().item()),
            "mse_per_phase": {
                str(z): (float(err[phases == z].mean().item()) if (phases == z).any() else None)
                for z in range(num_phases)
            },
        }
    return out


def _process_seed(outputs: Path, seed: int, max_samples: int, device) -> dict:
    ckpt = stage2_ckpt(outputs, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint_best.pt missing at {ckpt}")
    model, cfg = build_model_and_load(ckpt, device)
    val_loader = build_val_loader(cfg)
    chunks_lat, chunks_act, chunks_z = [], [], []
    seen = 0
    for batch in iter_val_batches(model, val_loader, device):
        chunks_lat.append(batch["latent"])
        chunks_act.append(batch["action"])
        chunks_z.append(batch["phase_true"])
        seen += batch["latent"].shape[0]
        if seen >= max_samples:
            break
    latents = torch.cat(chunks_lat, dim=0)[:max_samples].to(device)
    actions = torch.cat(chunks_act, dim=0)[:max_samples].to(device)
    phases = torch.cat(chunks_z, dim=0)[:max_samples].to(device)
    return _variant_outputs(model, latents, phases, actions)


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
    for seed in seeds:
        try:
            res = _process_seed(outputs, seed, args.max_samples, device)
        except FileNotFoundError as exc:
            print(f"[counterfact] seed {seed}: {exc}")
            per_seed[str(seed)] = {"missing": str(exc)}
            continue
        per_seed[str(seed)] = res
        summary = " ".join(f"{k}={v['action_mse']:.4f}" for k, v in res.items())
        print(f"[counterfact] seed {seed}: {summary}")

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": seeds,
        "per_seed": per_seed,
        "note": "action MSE against the ground-truth validation actions; "
                "every variant shares the same trained experts and only the "
                "router distribution differs.",
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    findings_path = PROJECT_ROOT / FINDINGS_DIR / "routing_counterfactuals.json"
    findings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _render_report(payload)
    print(f"[counterfact] done -> {findings_path}")
    return 0


def _render_report(payload: dict) -> None:
    lines = [
        "# Wave A5 — Routing counterfactuals (offline)",
        "",
        f"Created {payload['created']} on branch `surgical-cpu-analysis`.",
        "",
        payload["note"],
        "",
        "Lower action MSE = better action fit on the validation trajectories.",
        "``oracle_true`` is the upper bound when the true phase label is used",
        "to select a single expert; ``random``/``uniform`` are the no-information",
        "baselines.",
        "",
        "| seed | learned | oracle_true | oracle_pred | uniform | random |",
        "|---|---|---|---|---|---|",
    ]
    for s in payload["seeds"]:
        r = payload["per_seed"].get(str(s), {})
        if "missing" in r:
            lines.append(f"| {s} | MISSING | MISSING | MISSING | MISSING | MISSING |")
            continue
        vals = {k: r.get(k, {}).get("action_mse") for k in ("learned", "oracle_true", "oracle_pred", "uniform", "random")}
        lines.append(
            f"| {s} | {vals['learned']:.4f} | {vals['oracle_true']:.4f} | "
            f"{vals['oracle_pred']:.4f} | {vals['uniform']:.4f} | {vals['random']:.4f} |"
        )
    lines += ["", "## Interpretation", "", "- If `learned` ≈ `oracle_true`, the router already exploits phase information.", "- If `learned` ≈ `uniform`, the router adds nothing over the mean expert.", "- The gap `oracle_true - learned` is the routing-improvement headroom.", ""]
    report = PROJECT_ROOT / REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())