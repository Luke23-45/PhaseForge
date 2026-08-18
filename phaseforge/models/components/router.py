"""TopKRouter: Sparse top-k expert router with load balancing."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RouterOutput(NamedTuple):
    """Standardized output from the TopKRouter."""

    weights: Tensor  # (B, K) normalized top-k weights
    indices: Tensor  # (B, K) selected expert indices
    gate_logits: Tensor  # (B, E) raw logits over all experts
    balance_loss: Tensor  # scalar, auxiliary load-balancing loss


class TopKRouter(nn.Module):
    """Sparse top-k expert router with auxiliary load-balancing loss.

    Follows the Shazeer et al. (2017) / GShard-style top-k noisy-gating
    mechanism (not Switch Transformer, whose defining contribution is
    top-1 routing):
    1. Compute gate logits via linear projection.
    2. Add scaled Gaussian noise during training for exploration.
    3. Select the top-k experts.
    4. Normalize the top-k gate values via softmax.
    5. Compute a load-balancing loss to encourage equal expert utilization.

    Args:
        latent_dim: Dimension of the input latent vector.
        num_experts: Total number of experts (E) to route to.
        top_k: Number of experts to select per input (K). Must be in
            ``[1, num_experts]``.
        noise_std: Standard deviation of the routing noise added during training.
            If 0.0, routing is purely deterministic.
        balance_coeff: Multiplier for the auxiliary balance loss.
        normalize_input: If ``True``, L2-normalize the latent vector before
            the gate projection. Combined with unit-norm centroids loaded
            into ``gate_linear.weight`` (see the bootstrap), the gate logits
            become true cosine similarities between the latent and each
            centroid — the ``phaseforge`` / ``plain_encoder_phase_bootstrap``
            cells. ``False`` keeps raw dot products.
    """

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        balance_coeff: float = 0.01,
        normalize_input: bool = False,
        anchor: str | None = None,
        anchor_rank: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError(f"num_experts must be a positive int, got {num_experts!r}")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts}). "
                "The config routes to more experts than exist."
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0.0, got {noise_std}")
        if balance_coeff < 0.0:
            raise ValueError(f"balance_coeff must be >= 0.0, got {balance_coeff}")
        if anchor not in (None, "phase_head"):
            raise ValueError(
                f"anchor must be None or 'phase_head', got {anchor!r}"
            )
        if anchor is not None:
            if not isinstance(anchor_rank, int) or anchor_rank < 1:
                raise ValueError(
                    f"anchor_rank must be a positive int when anchor is set, "
                    f"got {anchor_rank!r}"
                )
            if anchor_rank > min(latent_dim, num_experts):
                raise ValueError(
                    f"anchor_rank ({anchor_rank}) exceeds "
                    f"min(latent_dim, num_experts) = {min(latent_dim, num_experts)}"
                )

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.balance_coeff = balance_coeff
        self.normalize_input = bool(normalize_input)
        self.anchor = anchor
        self.anchor_rank = anchor_rank

        self.gate_linear: nn.Linear | nn.Sequential
        self.noise_linear: nn.Linear | None

        # Low-rank anchored router (V6): gate_logits = anchor_linear(z) +
        # gate_linear(z), where anchor_linear is loaded from the stage-1 phase
        # head (frozen) and gate_linear is a ZERO-INITIALIZED low-rank
        # residual. The rank constraint makes the anchor structural (Lemma 3
        # in docs/research/phase_utilization_design.md): the router can only
        # deviate from the phase predictor in an anchor_rank-dimensional
        # subspace, so phase alignment persists by construction instead of
        # decaying like a plain initialization.
        self.anchor_linear: nn.Linear | None = None
        if anchor == "phase_head":
            assert isinstance(anchor_rank, int)  # validated above
            self.anchor_linear = nn.Linear(latent_dim, num_experts)
            self.gate_linear = nn.Sequential(
                nn.Linear(latent_dim, anchor_rank),
                nn.Linear(anchor_rank, num_experts),
            )
            self.gate_linear.apply(self._zero_init)
        else:
            self.gate_linear = nn.Linear(latent_dim, num_experts)

        # Linear layer to scale the noise per-input, following standard MoE practices
        if self.noise_std > 0.0:
            self.noise_linear = nn.Linear(latent_dim, num_experts)
        else:
            self.noise_linear = None

        self._init_weights()

    @staticmethod
    def _zero_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_weights(self) -> None:
        """Initialize routing weights.

        Using normal initialization with a small std dev helps prevent
        all inputs from collapsing to a single expert at the start.
        """
        if isinstance(self.gate_linear, nn.Linear):
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
        if self.normalize_input:
            # Unit-norm latents + unit-norm centroid weights (set by the
            # bootstrap) => gate logits are true cosine similarities.
            latent = F.normalize(latent, p=2, dim=-1)
        if self.anchor_linear is not None:
            # Anchored router (V6): the frozen stage-1 phase predictor is the
            # backbone; the zero-init low-rank residual adds corrections.
            gate_logits = self.anchor_linear(latent) + self.gate_linear(latent)
        else:
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
            balance_loss=balance_loss * self.balance_coeff,
        )

    def _compute_balance_loss(self, routing_probs: Tensor, gate_logits: Tensor) -> Tensor:
        """Compute the Switch Transformer auxiliary load-balancing loss.

        L_balance = E * sum(f_i * p_i) for i in 1..E
        where f_i is the fraction of items routed to expert i (based on top-1)
        and p_i is the mean routing probability for expert i.
        (The f_i * p_i auxiliary-loss formulation is Switch Transformer's;
        the surrounding top-k noisy-gating mechanism is Shazeer et al.
        (2017) / GShard lineage.)

        .. note:: Routing semantics — this loss is computed on the *top-1*
            hard assignment (``argmax`` of the gate logits, as in Switch
            Transformer), while the utilization diagnostics reported during
            validation/offline evaluation count *all* top-k assignments
            (see ``expert_utilization``). This is intentional: the loss
            penalizes the primary-choice imbalance that drives dead experts,
            whereas the reported metrics measure full dispatch load. The
            balance score therefore does NOT measure exactly what the
            optimization loss minimizes — interpret them side by side, not
            interchangeably.

        Args:
            routing_probs: (B, E) softmax probabilities
            gate_logits: (B, E) raw logits before softmax

        Returns:
            Scalar balance loss tensor.
        """
        B, E = gate_logits.shape

        # f_i: fraction of batch routed to each expert (based on primary choice)
        # We use top-1 for the balance loss calculation, as is standard.
        top_1_indices = gate_logits.argmax(dim=-1)  # (B,)

        # One-hot encoding of expert assignments (B, E)
        expert_mask = F.one_hot(top_1_indices, num_classes=E).float()

        # Mean fraction of tokens routed to each expert: (E,)
        f_i = expert_mask.mean(dim=0)

        # Mean probability assigned to each expert: (E,)
        p_i = routing_probs.mean(dim=0)

        # The loss encourages f_i and p_i to be uniform (1/E)
        balance_loss = E * torch.sum(f_i * p_i)

        return balance_loss
