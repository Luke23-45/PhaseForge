"""State-only recurrent behavior-cloning comparator."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder


class BehaviorCloningRNNModel(BaseManipulationModel):
    """BC-RNN over the same low-dimensional state schema as the policy.

    The encoder and action head are the same components used by the MLP BC
    control. The only added capacity is a unidirectional recurrent layer,
    making this a temporal-history comparator with the same observation and
    action representation.

    ``get_action`` supports complete ``(B, T, S)`` sequences and the
    streaming ``(B, S)`` calls used by the rollout runner. In streaming mode
    recurrent state is retained between calls and reset at every episode.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        rnn_name = str(rnn_type).lower()
        if rnn_name not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = encoder
        self.action_head = action_head
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn_type = rnn_name
        if rnn_name == "lstm":
            self.rnn: nn.Module = nn.LSTM(
                input_size=encoder.latent_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=encoder.latent_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
        self._hidden: Tensor | tuple[Tensor, Tensor] | None = None
        self._hidden_batch_size: int | None = None

    def reset(self) -> None:
        """Forget recurrent history before a new rollout episode."""
        self._hidden = None
        self._hidden_batch_size = None

    def _encode_sequence(self, state: Tensor) -> Tensor:
        if state.ndim != 3:
            raise ValueError(f"expected state shape (B,T,S), got {tuple(state.shape)}")
        batch, steps, state_dim = state.shape
        latent = self.encoder(state.reshape(batch * steps, state_dim))
        return latent.reshape(batch, steps, -1)

    def _actions_from_sequence(self, state: Tensor, hidden=None) -> tuple[Tensor, object]:
        latent = self._encode_sequence(state)
        recurrent, next_hidden = self.rnn(latent, hidden)
        actions = self.action_head(recurrent.reshape(-1, self.hidden_dim))
        return actions.reshape(state.shape[0], state.shape[1], -1), next_hidden

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        was_single = state.ndim == 2
        if was_single:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError("BC-RNN expects state with shape (B,S) or (B,T,S)")
        action_pred, _ = self._actions_from_sequence(state)
        if was_single:
            action_pred = action_pred[:, 0]
        return ModelOutput(action_pred=action_pred)

    def get_action(self, state: Tensor) -> Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(1)
            batch_size = state.shape[0]
            if self._hidden_batch_size != batch_size:
                self.reset()
            with torch.no_grad():
                action, hidden = self._actions_from_sequence(state, self._hidden)
            self._hidden = self._detach_hidden(hidden)
            self._hidden_batch_size = batch_size
            return action[:, 0]
        if state.ndim == 3:
            # A complete sequence is an independent inference request; it
            # must not inherit a previous episode's state.
            with torch.no_grad():
                action, _ = self._actions_from_sequence(state)
            return action
        raise ValueError("BC-RNN expects state with shape (B,S) or (B,T,S)")

    @staticmethod
    def _detach_hidden(hidden):
        if isinstance(hidden, tuple):
            return tuple(value.detach() for value in hidden)
        return hidden.detach()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BehaviorCloningRNNModel"]
