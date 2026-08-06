"""Offline evaluator runner."""

from __future__ import annotations

import logging
import math

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from phaseforge.evaluations.metrics import (
    expert_utilization,
    phase_alignment,
    routing_stability,
    task_metrics,
)
from phaseforge.models.base import BaseManipulationModel

logger = logging.getLogger(__name__)


class OfflineEvaluator:
    """Runs a comprehensive offline evaluation over a validation/test dataset.

    Computes all enabled metrics defined in config/eval/metrics.yaml.
    """

    def __init__(
        self, cfg: DictConfig, model: BaseManipulationModel, dataloader: DataLoader
    ) -> None:
        self.cfg = cfg
        self.metrics_cfg = cfg.eval
        self.device = torch.device(cfg.project.get("device", "cuda"))
        self.model = model.to(self.device)
        self.dataloader = dataloader

    @torch.no_grad()
    def run(self) -> dict[str, float]:
        """Execute the evaluation loop and return aggregated metrics."""
        self.model.eval()
        
        all_action_preds = []
        all_action_targets = []
        all_phases = []
        all_routing_weights = []
        all_expert_indices = []
        all_gate_logits = []
        all_masks = []
        
        # 1. Collect all outputs
        for batch in self.dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            out = self.model(batch)
            
            all_action_preds.append(out.action_pred.detach().cpu())
            all_action_targets.append(batch["action"].cpu())
            all_phases.append(batch["phase"].cpu())
            
            mask = batch.get("padding_mask")
            if mask is not None:
                all_masks.append(mask.cpu())
            
            if out.routing_weights is not None:
                all_routing_weights.append(out.routing_weights.detach().cpu())
            if out.expert_indices is not None:
                all_expert_indices.append(out.expert_indices.detach().cpu())
            if out.gate_logits is not None:
                all_gate_logits.append(out.gate_logits.detach().cpu())
                
        # 2. Concatenate (flattening batch dim if needed, or keeping it depending on metrics)
        action_preds = torch.cat(all_action_preds, dim=0)
        action_targets = torch.cat(all_action_targets, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # 2b. Decisive signal first, logged IMMEDIATELY: raw action error. If a
        #     later metric is slow (e.g. the routing metrics) and the run gets
        #     interrupted, this line is what distinguishes "the model cannot
        #     reproduce the training actions" (0% LIBERO success is expected)
        #     from "the eval path is broken" (low MSE but 0% rollouts).
        mask = torch.cat(all_masks, dim=0) if all_masks else None
        mse = task_metrics.action_mse(action_preds, action_targets, mask)
        logger.info(
            "  eval/action_mse: %.6f (RMSE %.4f)",
            mse, math.sqrt(max(mse, 0.0)),
        )
        results = {"eval/action_mse": mse}

        is_moe = len(all_expert_indices) > 0
        if is_moe:
            expert_indices = torch.cat(all_expert_indices, dim=0)
            gate_logits = torch.cat(all_gate_logits, dim=0)
        else:
            expert_indices = None
            gate_logits = None

        # 3. Compute Metrics
        # (results already contains eval/action_mse from step 2b)
        
        # Task Metrics
        if self.metrics_cfg.task.success_rate.enabled:
            threshold = self.metrics_cfg.task.success_rate.l2_threshold
            results["eval/success_rate"] = task_metrics.success_rate(
                action_preds, action_targets, threshold
            )
            logger.info(
                "  eval/success_rate (L2 <= %.2f): %.4f",
                threshold, results["eval/success_rate"],
            )
            
        if self.metrics_cfg.task.boundary_smoothness.enabled:
            window = self.metrics_cfg.task.boundary_smoothness.boundary_window
            val = task_metrics.boundary_smoothness(action_preds, phases, window)
            if not torch.isnan(torch.tensor(val)):
                results["eval/boundary_smoothness"] = val
                
        # MoE Metrics
        if is_moe:
            num_experts = gate_logits.size(-1)
            
            if self.metrics_cfg.mechanism.routing_entropy.enabled:
                results["eval/routing_entropy"] = (
                    routing_stability.routing_entropy(gate_logits, normalize=True).item()
                )

            if self.metrics_cfg.mechanism.routing_entropy_variance.enabled:
                window = self.metrics_cfg.mechanism.routing_entropy_variance.get(
                    "window_size", 100
                )
                results["eval/routing_entropy_variance"] = (
                    routing_stability.routing_entropy_variance(
                        gate_logits, window_size=window
                    ).item()
                )

            if self.metrics_cfg.mechanism.time_to_stable_routing.enabled:
                window = self.metrics_cfg.mechanism.routing_entropy_variance.get(
                    "window_size", 100
                )
                var_threshold = (
                    self.metrics_cfg.mechanism.time_to_stable_routing.get(
                        "variance_threshold", 0.001
                    )
                )
                consecutive = (
                    self.metrics_cfg.mechanism.time_to_stable_routing.get(
                        "consecutive_windows", 5
                    )
                )
                results["eval/time_to_stable_routing"] = (
                    routing_stability.time_to_stable_routing(
                        gate_logits,
                        window_size=window,
                        variance_threshold=var_threshold,
                        consecutive_windows=consecutive,
                    ).item()
                )

            if self.metrics_cfg.mechanism.expert_utilization.enabled:
                fractions = expert_utilization.expert_utilization(
                    expert_indices, num_experts
                )

                results["eval/balance_score"] = (
                    expert_utilization.expert_utilization_balance(fractions)
                )

                if self.metrics_cfg.mechanism.collapse_rate.enabled:
                    factor = self.metrics_cfg.mechanism.collapse_rate.threshold_factor
                    results["eval/collapse_rate"] = (
                        expert_utilization.collapse_rate(fractions, factor)
                    )

            if self.metrics_cfg.mechanism.phase_expert_nmi.enabled:
                results["eval/phase_expert_nmi"] = (
                    phase_alignment.phase_expert_nmi(phases, expert_indices)
                )

        return results
