"""Online and frozen normalizers for state vectors.

Stats are computed ONLY on the training split using Welford's online algorithm
and then applied (frozen) to all splits.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor


class RunningStatNormalizer:
    """Online mean/std computation via Welford's algorithm.

    Call :meth:`update` with each training batch, then :meth:`finalize`
    to produce a :class:`FrozenNormalizer`.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None  # Sum of squared deviations

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a new batch.

        Args:
            batch: Array of shape (T, D) or (D,).
        """
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]  # (1, D)
        for sample in batch:
            self._count += 1
            if self._mean is None:
                self._mean = np.zeros_like(sample, dtype=np.float64)
                self._M2 = np.zeros_like(sample, dtype=np.float64)
            delta = sample.astype(np.float64) - self._mean
            self._mean += delta / self._count
            delta2 = sample.astype(np.float64) - self._mean
            self._M2 += delta * delta2

    def finalize(self, eps: float = 1e-6) -> "FrozenNormalizer":
        """Freeze the accumulated statistics.

        Args:
            eps: Small constant added to std to avoid division by zero.

        Returns:
            A :class:`FrozenNormalizer` ready for use.
        """
        if self._count == 0:
            raise RuntimeError("No data was passed to RunningStatNormalizer.update().")

        mean = torch.from_numpy(self._mean).float()
        if self._count < 2:
            std = torch.ones_like(mean)
        else:
            variance = self._M2 / (self._count - 1)
            std = torch.from_numpy(np.sqrt(variance) + eps).float()

        return FrozenNormalizer(mean=mean, std=std)


class FrozenNormalizer:
    """Immutable normalizer loaded from cache.

    Formula::

        normalize(x)   = (x - mean) / std
        denormalize(x) = x * std + mean
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    def normalize(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Normalize input to zero-mean unit-variance."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def denormalize(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Invert normalization."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean

    def save(self, path) -> None:
        import torch
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path) -> "FrozenNormalizer":
        import torch
        data = torch.load(path, weights_only=False)
        return cls(mean=data["mean"], std=data["std"])
