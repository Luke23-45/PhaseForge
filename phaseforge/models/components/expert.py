"""ExpertMLP: A single specialized neural network expert."""

from __future__ import annotations

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

    def forward(self, latent: Tensor) -> Tensor:
        """Map latent vector to action prediction.

        Args:
            latent: Tensor of shape (..., input_dim)

        Returns:
            action_pred: Tensor of shape (..., output_dim)
        """
        h = self.hidden(latent)
        return self.output_proj(h)
