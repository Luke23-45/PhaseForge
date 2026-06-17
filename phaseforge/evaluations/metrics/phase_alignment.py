"""Phase-Expert alignment metrics."""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def phase_expert_nmi(phases: Tensor, expert_indices: Tensor) -> float:
    """Calculate Normalized Mutual Information (NMI) between phases and experts.

    NMI measures the alignment between the assigned phase labels and the 
    router's expert choices. 
    1.0 = Perfect one-to-one mapping (experts specialize perfectly by phase).
    0.0 = Completely independent (router ignores phase structure).

    Args:
        phases: Tensor of shape (B,) or (B, T) containing integer phase labels.
        expert_indices: Tensor of shape (B, K) or (B, T, K) containing top-K experts.
            Only the top-1 expert (K=0) is used for NMI calculation.

    Returns:
        Float NMI score in [0, 1].
    """
    # Flatten inputs
    p_flat = phases.view(-1).cpu().numpy()
    
    # Use top-1 expert for alignment
    if expert_indices.ndim == phases.ndim + 1:
        # e.g. phases is (B,) and expert_indices is (B, K)
        e_top1 = expert_indices[..., 0].view(-1).cpu().numpy()
    else:
        # Assumed flat
        e_top1 = expert_indices.view(-1).cpu().numpy()
        
    if len(p_flat) == 0 or len(e_top1) == 0:
        return 0.0
        
    if len(p_flat) != len(e_top1):
        raise ValueError(f"Shape mismatch: phases has {len(p_flat)} elements, experts has {len(e_top1)}")

    # Calculate NMI using scikit-learn
    # We use 'arithmetic' average method as standard
    return float(normalized_mutual_info_score(p_flat, e_top1, average_method='arithmetic'))


def build_contingency_matrix(phases: Tensor, expert_indices: Tensor, num_phases: int, num_experts: int) -> Tensor:
    """Build a P x E contingency matrix (heatmap) of phase-expert assignments.
    
    Args:
        phases: Tensor of shape (B,) containing integer phase labels.
        expert_indices: Tensor of shape (B, K) containing top-K experts. Uses top-1.
        num_phases: Total number of phases (P).
        num_experts: Total number of experts (E).
        
    Returns:
        Tensor of shape (num_phases, num_experts) containing normalized counts.
        Each row (phase) sums to 1.0.
    """
    p_flat = phases.view(-1)
    e_top1 = expert_indices[..., 0].view(-1)
    
    matrix = torch.zeros((num_phases, num_experts), dtype=torch.float32, device=phases.device)
    
    for p, e in zip(p_flat, e_top1):
        if 0 <= p < num_phases and 0 <= e < num_experts:
            matrix[p, e] += 1.0
            
    # Row normalize so each phase's distribution sums to 1
    row_sums = matrix.sum(dim=1, keepdim=True)
    
    # Avoid division by zero
    row_sums = torch.clamp(row_sums, min=1.0)
    
    return matrix / row_sums
