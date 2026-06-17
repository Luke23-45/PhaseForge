"""MoELayer: Encapsulates routing logic and multiple experts."""

from __future__ import annotations

import copy
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from phaseforge.models.components.router import TopKRouter, RouterOutput
from phaseforge.models.components.expert import ExpertMLP


class MoEOutput(NamedTuple):
    """Output from the MoE Layer."""
    combined_output: Tensor     # (B, output_dim) Final action prediction
    routing_weights: Tensor     # (B, K)
    expert_indices: Tensor      # (B, K)
    balance_loss: Tensor        # scalar
    gate_logits: Tensor         # (B, E)


class MoELayer(nn.Module):
    """Mixture-of-Experts Layer orchestrating the router and experts.

    This layer takes the latent representation, uses the TopKRouter to determine
    expert assignments and weights, dynamically dispatches inputs to the assigned
    experts, and combines their outputs via a weighted sum.

    Args:
        router: Instantiated TopKRouter.
        experts: A list (or nn.ModuleList) of instantiated ExpertMLP networks.
            If a single ExpertMLP is provided, it will be cloned `num_experts` times.
    """

    def __init__(
        self,
        router: TopKRouter,
        experts: ExpertMLP | nn.ModuleList | list[ExpertMLP],
    ) -> None:
        super().__init__()
        self.router = router
        
        # Handle expert instantiation
        if isinstance(experts, nn.ModuleList):
            self.experts = experts
        elif isinstance(experts, list):
            self.experts = nn.ModuleList(experts)
        elif isinstance(experts, ExpertMLP):
            # Clone the single expert template E times
            self.experts = nn.ModuleList([
                copy.deepcopy(experts) for _ in range(router.num_experts)
            ])
        else:
            raise TypeError("experts must be an ExpertMLP, list[ExpertMLP], or nn.ModuleList")

        if len(self.experts) != router.num_experts:
            raise ValueError(
                f"Number of experts ({len(self.experts)}) does not match "
                f"router.num_experts ({router.num_experts})"
            )

    def forward(self, latent: Tensor) -> MoEOutput:
        """Route latents to experts and combine their outputs.

        Args:
            latent: Tensor of shape (B, latent_dim).

        Returns:
            MoEOutput containing the combined predictions and routing metadata.
        """
        B, D = latent.shape
        E = self.router.num_experts
        K = self.router.top_k

        # 1. Route inputs
        router_out: RouterOutput = self.router(latent)
        weights = router_out.weights    # (B, K)
        indices = router_out.indices    # (B, K)

        # Retrieve the output dimension from the first expert to preallocate
        out_dim = self.experts[0].output_dim
        
        # Final combined output tensor: (B, out_dim)
        combined_output = torch.zeros(
            (B, out_dim), dtype=latent.dtype, device=latent.device
        )

        # 2. Dispatch to experts and combine
        # Implementation note: For small K (e.g., 2) and E (e.g., 6-8), iterating over 
        # experts is often faster than complex scatter/gather batched operations due to 
        # kernel launch overheads. We use the loop-over-experts approach.
        
        # Flatten indices and weights for easier masking
        # We need to find which items in the batch go to which expert
        
        for expert_idx, expert_net in enumerate(self.experts):
            # Find all locations where this expert was selected
            # match_mask: (B, K) boolean tensor
            match_mask = (indices == expert_idx)
            
            # Check if this expert was selected at all in this batch
            if not match_mask.any():
                continue
                
            # Find the batch indices that selected this expert
            # batch_idx: 1D tensor of batch indices
            # k_idx: 1D tensor of the k-th choice (0 to K-1)
            batch_idx, k_idx = torch.where(match_mask)
            
            # Gather the latents for this expert
            # expert_inputs: (N, D) where N is the number of items routed to this expert
            expert_inputs = latent[batch_idx]
            
            # Forward pass through the expert
            # expert_outputs: (N, out_dim)
            expert_outputs = expert_net(expert_inputs)
            
            # Gather the corresponding weights
            # expert_weights: (N, 1)
            expert_weights = weights[batch_idx, k_idx].unsqueeze(-1)
            
            # Accumulate into the combined output
            # combined_output[batch_idx] += expert_outputs * expert_weights
            # Note: We use scatter_add_ to safely handle cases where the same expert
            # might somehow be selected multiple times for the same batch item 
            # (though topk should prevent this, it's safer).
            weighted_outputs = expert_outputs * expert_weights
            combined_output.index_add_(0, batch_idx, weighted_outputs)

        return MoEOutput(
            combined_output=combined_output,
            routing_weights=weights,
            expert_indices=indices,
            balance_loss=router_out.balance_loss,
            gate_logits=router_out.gate_logits,
        )
