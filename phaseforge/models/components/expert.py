"""ExpertMLP: A single specialized neural network expert."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Reuse the activations dictionary from encoder.py for consistency
_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class ExpertMLP(nn.Module):
    """A single expert network within the Mixture-of-Experts layer.

    All experts share identical architectures but maintain independent weights.
    The architecture is typically a small multi-layer perceptron mapping the
    latent representation to the action space.

    Args:
        input_dim: Dimension of the latent vector from the encoder.
        hidden_dims: List of hidden layer widths.
        output_dim: Dimension of the action prediction.
        activation: Activation function (e.g., "gelu").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_cls = _ACTIVATIONS.get(activation.lower(), nn.GELU)

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(in_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights.

        Using Kaiming initialization to preserve variance through the MLP.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def reset_parameters(self) -> None:
        """Re-initialize all weights with an independent fresh draw.

        Used to break symmetry when expert clones are created by
        ``copy.deepcopy`` (bit-identical clones make the combined MoE
        output invariant to the router's weights).
        """
        self._init_weights()

    def forward(self, latent: Tensor) -> Tensor:
        """Map latent vector to action prediction.

        Args:
            latent: Tensor of shape (..., input_dim)

        Returns:
            action_pred: Tensor of shape (..., output_dim)
        """
        h = self.hidden(latent)
        return self.output_proj(h)


def warm_start_experts_from_action_head(
    experts: nn.ModuleList,
    action_head: nn.Module,
    jitter_std: float = 0.02,
) -> None:
    """Copy the Stage 1 ActionHead into every expert — exactly and strictly.

    The ActionHead is a single-hidden-layer policy (``trunk`` =
    ``Linear -> GELU``, ``mean_head`` = output projection). The ExpertMLP
    must therefore be configured with a matching single-hidden-layer
    structure (``hidden_dims=[hidden_dim]`` with ``hidden_dim`` ==
    ``action_head.hidden_dim``). Any expert parameter left unfilled is a
    hard error: a partial warm start (e.g. a randomly initialized second
    expert hidden layer) is not a functional warm start and would invalidate
    the Stage 1 -> Stage 2 initialization claim.

    A small independent Gaussian jitter is added to every copied parameter
    so the experts are not bit-identical after bootstrapping. Identical
    experts make the combined output invariant to the router's weights
    (top-k softmax weights sum to 1), so the action loss can never shape
    routing; the jitter breaks that symmetry while remaining a true warm
    start.

    Args:
        experts: The MoE layer's ``nn.ModuleList`` of experts.
        action_head: The Stage 1 ``ActionHead`` whose weights are copied.
        jitter_std: Standard deviation of the symmetry-breaking noise.
            ``0.0`` disables the jitter (exact copy).
    """
    if jitter_std < 0.0:
        raise ValueError(f"jitter_std must be >= 0.0, got {jitter_std}")
    action_sd = action_head.state_dict()
    mapping = {
        "trunk.0.weight": "hidden.0.weight",
        "trunk.0.bias": "hidden.0.bias",
        "mean_head.weight": "output_proj.weight",
        "mean_head.bias": "output_proj.bias",
    }

    for expert in experts:
        expert_sd = expert.state_dict()

        new_dict: dict[str, Tensor] = {}
        for src_key, dst_key in mapping.items():
            if src_key not in action_sd:
                raise ValueError(
                    "Cannot warm-start experts from the ActionHead: it has no "
                    f"'{src_key}' (available keys: {sorted(action_sd)})."
                )
            if dst_key not in expert_sd:
                raise ValueError(
                    "Cannot warm-start experts: the ExpertMLP has no "
                    f"'{dst_key}' (available keys: {sorted(expert_sd)})."
                )
            new_dict[dst_key] = action_sd[src_key].clone()

        missing = sorted(set(expert_sd) - set(new_dict))
        if missing:
            raise ValueError(
                "Cannot warm-start experts: the ExpertMLP has parameters that the "
                f"ActionHead cannot fill (uninitialized keys: {missing}). The "
                "ExpertMLP must match the ActionHead's single-hidden-layer "
                "structure: use expert.hidden_dims=[<action_head.hidden_dim>]."
            )
        expert.load_state_dict(new_dict, strict=True)

        if jitter_std > 0.0:
            for param in expert.parameters():
                param.data.add_(torch.randn_like(param.data) * jitter_std)
