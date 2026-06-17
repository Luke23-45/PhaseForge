"""StateEncoder: MLP with residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class StateEncoder(nn.Module):
    """MLP encoder mapping proprioceptive state → latent representation.

    Args:
        input_dim:    Raw state dimension.
        hidden_dims:  List of hidden layer widths.
        latent_dim:   Output latent dimension.
        activation:   Activation function name (``"gelu"``, ``"relu"``, ``"silu"``).
        dropout:      Dropout rate applied after each hidden layer.
        use_residual: Add residual connections between layers of the same width.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        act_cls = _ACTIVATIONS.get(activation, nn.GELU)

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_proj = nn.Linear(in_dim, latent_dim)

        # Residual shortcut (only when input_dim == latent_dim, else project)
        self.use_residual = use_residual and (input_dim == latent_dim)
        if use_residual and input_dim != latent_dim:
            self.res_proj: nn.Module = nn.Linear(input_dim, latent_dim)
        else:
            self.res_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, state: Tensor) -> Tensor:
        """Encode state to latent vector.

        Args:
            state: (B, input_dim)

        Returns:
            latent: (B, latent_dim)
        """
        h = self.hidden(state)
        out = self.output_proj(h)
        if self.use_residual:
            out = out + self.res_proj(state)
        return out
