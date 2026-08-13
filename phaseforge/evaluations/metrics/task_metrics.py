"""Task-level offline performance metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def success_rate(
    predicted_actions: Tensor, target_actions: Tensor, l2_threshold: float = 0.05
) -> float:
    """Offline proxy for success rate based on action L2 error threshold.

    Since offline metrics cannot directly measure task success (which requires an environment),
    we use the percentage of actions that fall within a strict L2 error bound as a proxy.

    Args:
        predicted_actions: Tensor of shape (B, A) or (B, T, A).
        target_actions: Tensor of shape (B, A) or (B, T, A).
        l2_threshold: Maximum allowed L2 distance to be considered "successful".

    Returns:
        Float success rate in [0, 1].
    """
    if predicted_actions.numel() == 0:
        return 0.0
        
    # Calculate pairwise L2 distances
    l2_errors = torch.norm(predicted_actions - target_actions, p=2, dim=-1)
    
    # Count how many are below threshold
    successes = (l2_errors <= l2_threshold).sum().item()
    total = l2_errors.numel()
    
    return successes / total


def action_mse(
    predicted_actions: Tensor, target_actions: Tensor, mask: Tensor | None = None
) -> float:
    """Mean squared error between predicted and target actions.

    The single most informative offline diagnostic: if a policy that scores
    0% rollout success cannot even reproduce the demonstration actions
    (high MSE), the rollout result reflects model capability rather than an
    evaluation bug.

    Args:
        predicted_actions: (B, A) or (B, T, A).
        target_actions: Same shape as ``predicted_actions``.
        mask: Optional (B, T) (or (B,)) boolean/0-1 padding mask. When given,
            only the masked-in entries contribute (variable-length batches).

    Returns:
        Scalar MSE over the (masked) action entries, or ``float('nan')`` when
        no entries remain.
    """
    if mask is not None:
        m = mask.bool()
        while m.ndim < predicted_actions.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(predicted_actions)
        predicted_actions = predicted_actions[m]
        target_actions = target_actions[m]

    if predicted_actions.numel() == 0:
        return float("nan")
    return float(((predicted_actions - target_actions) ** 2).mean())


def boundary_action_smoothness(
    predicted_actions: Tensor, phases: Tensor, boundary_window: int = 3
) -> float:
    """Measure the temporal smoothness of predicted actions at phase boundaries.

    .. note::
       Despite its historical name ("boundary smoothness"), this metric
       does NOT measure an error against target actions. It reports the
       mean per-step temporal change (L2 of ``pred[t] - pred[t-1]``) of
       the *predicted* actions restricted to a window around detected
       phase transitions. A target-based boundary error requires
       trajectory-aligned targets and is not implemented.

    Phase transitions are the hardest parts of long-horizon tasks. This metric
    isolates the temporal change of the predicted actions immediately
    surrounding a phase boundary.

    Args:
        predicted_actions: Tensor of shape (B, T, A).
        phases: Tensor of shape (B, T).
        boundary_window: Number of timesteps before and after the boundary to include.

    Returns:
        Mean L2 temporal difference at the boundaries, or float('nan') if no
        boundaries exist.

    Raises:
        ValueError: If ``boundary_window < 0``, inputs are empty, or the
            tensors contain non-finite values.
    """
    if int(boundary_window) < 0:
        raise ValueError(f"boundary_window must be >= 0, got {boundary_window}")
    if predicted_actions.ndim != 3 or phases.ndim != 2:
        # Require sequence dimension to detect boundaries
        return float('nan')
    if predicted_actions.numel() == 0 or phases.numel() == 0:
        raise ValueError("predicted_actions/phases must not be empty")
    if not torch.isfinite(predicted_actions).all():
        raise ValueError("predicted_actions contains non-finite values")
    if not torch.isfinite(phases.to(torch.float32)).all():
        raise ValueError("phases contains non-finite values")

    B, T, _ = predicted_actions.shape
    if T < 2:
        return float('nan')
        
    # Detect transitions: phases[t] != phases[t-1]
    # mask is (B, T-1)
    transitions = (phases[:, 1:] != phases[:, :-1])
    
    boundary_mask = torch.zeros((B, T), dtype=torch.bool, device=predicted_actions.device)
    
    has_boundary = False
    
    for b in range(B):
        # Indices where a transition occurs (offset by 1 because diff)
        transition_idxs = torch.where(transitions[b])[0] + 1
        
        for t_idx in transition_idxs:
            has_boundary = True
            start = max(0, int(t_idx.item()) - boundary_window)
            end = min(T, int(t_idx.item()) + boundary_window + 1)
            boundary_mask[b, start:end] = True
            
    if not has_boundary:
        return float('nan')
        
    # We don't have the ground truth actions here directly to compute error against,
    # so we measure the smoothness (temporal difference) of the predicted actions
    # at the boundaries. 
    # High smoothness means the model doesn't jerk violently at transitions.
    
    diffs = predicted_actions[:, 1:] - predicted_actions[:, :-1]
    diffs_l2 = torch.norm(diffs, p=2, dim=-1) # (B, T-1)
    
    # Apply mask (excluding the last element which we diff'd against)
    valid_diffs = diffs_l2[boundary_mask[:, :-1]]
    
    if valid_diffs.numel() == 0:
        return float('nan')
        
    return valid_diffs.mean().item()
