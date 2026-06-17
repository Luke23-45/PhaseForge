"""Expert utilization and load balancing metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def expert_utilization(expert_indices: Tensor, num_experts: int) -> Tensor:
    """Compute the fraction of items routed to each expert.

    Args:
        expert_indices: Tensor of shape (B, K) containing chosen expert indices.
        num_experts: Total number of experts (E).

    Returns:
        Tensor of shape (E,) representing the fraction [0, 1] of routing assignments.
    """
    # Flatten indices to 1D
    indices_flat = expert_indices.view(-1)
    
    # Bincount to get absolute usage, minlength ensures output shape is (E,)
    counts = torch.bincount(indices_flat, minlength=num_experts).float()
    
    # Normalize to fractions
    total_assignments = max(1, len(indices_flat))
    fractions = counts / total_assignments
    
    return fractions


def expert_utilization_balance(fractions: Tensor) -> float:
    """Compute the balance score (normalized entropy) of expert usage.

    Score of 1.0 means perfectly uniform usage across all E experts.
    Score of 0.0 means complete collapse (all items routed to 1 expert).

    Args:
        fractions: Tensor of shape (E,) summing to 1.0.

    Returns:
        Float score in [0, 1].
    """
    E = fractions.size(0)
    if E <= 1:
        return 1.0
        
    # Clamp for numerical stability
    probs = fractions.clamp(min=1e-8)
    
    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs))
    
    # Normalize by log(E)
    normalized_entropy = entropy / torch.log(torch.tensor(E, dtype=torch.float32, device=fractions.device))
    
    return normalized_entropy.item()


def collapse_rate(fractions: Tensor, threshold_factor: float = 5.0) -> float:
    """Calculate the percentage of "collapsed" (unused or rarely used) experts.

    An expert is considered collapsed if its usage fraction is less than
    1/(threshold_factor * E).

    Args:
        fractions: Tensor of shape (E,) summing to 1.0.
        threshold_factor: Factor controlling strictness of collapse definition.

    Returns:
        Float rate in [0, 1]. E.g., if 3 out of 6 experts are collapsed, returns 0.5.
    """
    E = fractions.size(0)
    if E == 0:
        return 0.0
        
    threshold = 1.0 / (threshold_factor * E)
    collapsed_count = (fractions < threshold).sum().item()
    
    return collapsed_count / E
