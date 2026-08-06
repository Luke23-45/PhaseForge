"""CPU-only tests for routing stability metrics.

Covers the scalar, per-window, and convergence metrics that power the
``routing_entropy_variance`` and ``time_to_stable_routing`` offline
evaluation keys.
"""

from __future__ import annotations

import pytest
import torch

from phaseforge.evaluations.metrics.routing_stability import (
    routing_entropy,
    routing_entropy_variance,
    time_to_stable_routing,
)


def _peak_logits(T: int, E: int) -> torch.Tensor:
    """Routing logits that start uniform and become peaky over time.

    Step t: logits = t * alpha in the first expert slot, 0 elsewhere, so
    entropy decreases monotonically and eventually concentrates on one
    expert (stable routing).
    """
    logits = torch.zeros(T, E)
    for t in range(T):
        logits[t, 0] = t * 0.05
    return logits


def _alternating_logits(T: int, E: int) -> torch.Tensor:
    """Routing logits whose entropy swings every step (unstable routing).

    Rows alternate between a peaked distribution (low entropy) and a near
    uniform one (high entropy), so per-step entropy oscillates strongly.
    """
    logits = torch.zeros(T, E)
    for t in range(T):
        if t % 2 == 0:
            logits[t, 0] = 5.0
        else:
            logits[t, :] = 0.2
    return logits


def test_routing_entropy_normalized_range() -> None:
    logits = torch.randn(64, 4)
    h = routing_entropy(logits, normalize=True)
    assert 0.0 <= h <= 1.0

    uniform = torch.zeros(64, 4)
    assert routing_entropy(uniform, normalize=True).item() == 1.0

    one_hot = torch.tensor([[50.0, 0.0, 0.0, 0.0]] * 64)
    assert routing_entropy(one_hot, normalize=True).item() < 1e-3


def test_routing_entropy_variance_zero_when_constant() -> None:
    logits = torch.zeros(500, 4)  # constant uniform routing
    var = routing_entropy_variance(logits, window_size=100)
    assert var.item() == pytest.approx(0.0, abs=1e-8)


def test_routing_entropy_variance_decreases_as_routing_stabilizes() -> None:
    unstable = routing_entropy_variance(_alternating_logits(500, 4), window_size=100)
    stabilized = routing_entropy_variance(
        torch.cat([_alternating_logits(100, 4), torch.zeros(400, 4)]),
        window_size=100,
    )
    assert stabilized.item() < unstable.item()


def test_routing_entropy_variance_single_step() -> None:
    var = routing_entropy_variance(torch.randn(1, 4), window_size=100)
    assert var.item() == 0.0


def test_time_to_stable_routing_detects_convergence() -> None:
    T, E = 1000, 4
    logits = _peak_logits(T, E)
    t = time_to_stable_routing(
        logits, window_size=100, variance_threshold=0.001, consecutive_windows=5
    )
    assert t.item() < T  # converged before the end


def test_time_to_stable_routing_never_stable_returns_T() -> None:
    logits = torch.randn(500, 4)  # constantly fluctuating
    T = logits.size(0)
    t = time_to_stable_routing(
        logits, window_size=100, variance_threshold=0.0001, consecutive_windows=5
    )
    assert t.item() == T


def test_time_to_stable_routing_short_sequence_returns_T() -> None:
    logits = torch.randn(10, 4)
    t = time_to_stable_routing(logits, window_size=100, consecutive_windows=5)
    assert t.item() == 10
