"""Routing stability metrics: entropy, variance, and convergence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def routing_entropy(gate_logits: Tensor, normalize: bool = True) -> Tensor:
    """Calculate the Shannon entropy of the routing distribution.

    Entropy measures how "certain" the router is. 
    High entropy = uniform routing (uncertain).
    Low entropy = peaked routing (certain).

    Args:
        gate_logits: Raw gating logits of shape (B, E).
        normalize: If True, divides by log(E) so output is in [0, 1].

    Returns:
        Scalar tensor containing the mean entropy across the batch.
    """
    E = gate_logits.size(-1)
    
    # Softmax probabilities (B, E)
    probs = F.softmax(gate_logits, dim=-1)
    
    # log_probs, clamped for numerical stability
    log_probs = torch.log(probs.clamp(min=1e-8))
    
    # Entropy: -sum(p * log(p)) over experts
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B,)
    
    mean_entropy = entropy.mean()
    
    if normalize and E > 1:
        mean_entropy = mean_entropy / torch.log(torch.tensor(E, dtype=torch.float32, device=gate_logits.device))
        
    return mean_entropy


class RoutingEntropyTracker:
    """Tracks routing entropy over a sliding window to compute variance."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.history: list[float] = []

    def update(self, entropy_val: float) -> None:
        """Add a new entropy value to the history."""
        self.history.append(entropy_val)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def current_variance(self) -> float:
        """Compute the variance of the current window."""
        if len(self.history) < 2:
            return 0.0
        # Sample variance
        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / (len(self.history) - 1)
        return var


class TimeToStableRouting:
    """Detects when routing has converged to a stable state."""

    def __init__(self, variance_threshold: float = 0.001, consecutive_windows: int = 5) -> None:
        self.variance_threshold = variance_threshold
        self.consecutive_windows = consecutive_windows
        self.stable_count = 0
        self.is_stable = False
        self.stable_step = -1

    def update(self, step: int, variance: float) -> bool:
        """Update stability status.
        
        Returns:
            True if newly stabilized on this step, False otherwise.
        """
        if self.is_stable:
            return False
            
        if variance < self.variance_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
            
        if self.stable_count >= self.consecutive_windows:
            self.is_stable = True
            self.stable_step = step
            return True
            
        return False
