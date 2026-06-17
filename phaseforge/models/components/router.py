"""TopKRouter: Sparse top-k expert router with load balancing."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RouterOutput(NamedTuple):
    """Standardized output from the TopKRouter."""
    weights: Tensor         # (B, K) normalized top-k weights
    indices: Tensor         # (B, K) selected expert indices
    gate_logits: Tensor     # (B, E) raw logits over all experts
    balance_loss: Tensor    # scalar, auxiliary load-balancing loss


class TopKRouter(nn.Module):
    """Sparse top-k expert router with auxiliary load-balancing loss.

    Follows the Switch Transformer routing mechanism:
    1. Compute gate logits via linear projection.
    2. Add scaled Gaussian noise during training for exploration.
    3. Select the top-k experts.
    4. Normalize the top-k gate values via softmax.
    5. Compute a load-balancing loss to encourage equal expert utilization.

    Args:
        latent_dim: Dimension of the input latent vector.
        num_experts: Total number of experts (E) to route to.
        top_k: Number of experts to select per input (K).
        noise_std: Standard deviation of the routing noise added during training.
            If 0.0, routing is purely deterministic.
        balance_coeff: Multiplier for the auxiliary balance loss.
    """

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.balance_coeff = balance_coeff

        self.gate_linear = nn.Linear(latent_dim, num_experts)
        
        # Linear layer to scale the noise per-input, following standard MoE practices
        if self.noise_std > 0.0:
            self.noise_linear = nn.Linear(latent_dim, num_experts)
        else:
            self.noise_linear = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize routing weights.
        
        Using normal initialization with a small std dev helps prevent 
        all inputs from collapsing to a single expert at the start.
        """
        nn.init.normal_(self.gate_linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_linear.bias)
        
        if self.noise_linear is not None:
            nn.init.normal_(self.noise_linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.noise_linear.bias)

    def forward(self, latent: Tensor) -> RouterOutput:
        """Route inputs to top-k experts and compute balance loss.

        Args:
            latent: Tensor of shape (B, latent_dim).

        Returns:
            RouterOutput containing weights, indices, logits, and balance loss.
        """
        # (B, E) raw gating logits
        gate_logits = self.gate_linear(latent)

        # Add exploration noise during training
        if self.training and self.noise_std > 0.0 and self.noise_linear is not None:
            noise_logits = self.noise_linear(latent)
            # softplus ensures noise scaling is positive
            noise_scale = self.noise_std * F.softplus(noise_logits)
            # standard normal noise
            noise = torch.randn_like(gate_logits)
            gate_logits = gate_logits + noise_scale * noise

        # Get routing probabilities (B, E)
        routing_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        # values: (B, K), indices: (B, K)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Normalize top-k values to sum to 1
        # Re-compute softmax over just the top-k elements so that sum(weights) == 1
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Compute auxiliary balance loss
        # We compute this even during eval so metrics can track it if desired
        balance_loss = self._compute_balance_loss(routing_probs, gate_logits)

        return RouterOutput(
            weights=top_k_weights,
            indices=top_k_indices,
            gate_logits=gate_logits,
            balance_loss=balance_loss * self.balance_coeff
        )

    def _compute_balance_loss(self, routing_probs: Tensor, gate_logits: Tensor) -> Tensor:
        """Compute the Switch Transformer load balancing loss.

        L_balance = E * sum(f_i * p_i) for i in 1..E
        where f_i is the fraction of items routed to expert i (based on top-1)
        and p_i is the mean routing probability for expert i.
        
        Args:
            routing_probs: (B, E) softmax probabilities
            gate_logits: (B, E) raw logits before softmax
            
        Returns:
            Scalar balance loss tensor.
        """
        B, E = gate_logits.shape
        
        # f_i: fraction of batch routed to each expert (based on primary choice)
        # We use top-1 for the balance loss calculation, as is standard.
        top_1_indices = gate_logits.argmax(dim=-1) # (B,)
        
        # One-hot encoding of expert assignments (B, E)
        expert_mask = F.one_hot(top_1_indices, num_classes=E).float()
        
        # Mean fraction of tokens routed to each expert: (E,)
        f_i = expert_mask.mean(dim=0)
        
        # Mean probability assigned to each expert: (E,)
        p_i = routing_probs.mean(dim=0)
        
        # The loss encourages f_i and p_i to be uniform (1/E)
        balance_loss = E * torch.sum(f_i * p_i)
        
        return balance_loss
