"""Viterbi-decoded routing (V5): temporal decoding of routing decisions.

The stage-1 phase predictor and the stage-2 router both emit per-step
logits, but the underlying phase process is temporally structured (Lift:
monotone non-decreasing phases, long segments). Viterbi decoding under the
empirical transition prior replaces isolated per-step flickers with
segment-consistent decisions — the phase *process* structure, which is
otherwise never used, becomes a decision-time channel
(docs/research/phase_utilization_design.md, Lemma 4).

Two emissions are supported:

1. ``phase_head`` — decode the phase sequence from the frozen stage-1 phase
   predictor's logits, then map phases to experts via a measured affinity
   matrix (the trained router's phase→expert usage, not an assumed
   bijection).
2. ``router`` — decode the expert sequence directly from the router's gate
   logits, with the expert-level transition prior induced from the phase
   transition prior through the affinity matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_transition_prior(phases: Tensor) -> Tensor:
    """Empirical row-normalized transition matrix from phase sequences.

    Args:
        phases: 1-D tensor of phase labels from one or more concatenated
            trajectories (or (N,) over a single trajectory). Transitions
            are counted between consecutive steps.

    Returns:
        (P, P) row-normalized transition probabilities, P = max+1.
    """
    if phases.ndim != 1:
        raise ValueError(f"phases must be 1-D, got shape {tuple(phases.shape)}")
    if phases.numel() < 2:
        num_states = int(phases.max().item()) + 1 if phases.numel() else 1
        return torch.eye(num_states)
    num_states = int(phases.max().item()) + 1
    # bincount (not ``counts[src, dst] += 1``, which drops duplicate
    # transitions with advanced indexing).
    flat = phases[:-1] * num_states + phases[1:]
    counts = torch.bincount(flat, minlength=num_states * num_states).float()
    counts = counts.reshape(num_states, num_states)
    row_sums = counts.sum(dim=-1, keepdim=True)
    row_sums = row_sums.clamp_min(1.0)
    return (counts / row_sums).float()


def viterbi_decode(
    log_emissions: Tensor,
    log_transition: Tensor,
    log_initial: Tensor | None = None,
) -> Tensor:
    """Standard Viterbi decode: MAP hidden-state sequence.

    Args:
        log_emissions: (T, S) log emission scores per step and state.
        log_transition: (S, S) log transition scores (from state i to j,
            row i, column j).
        log_initial: (S,) log initial-state scores; uniform when None.

    Returns:
        (T,) int64 MAP state indices.
    """
    if log_emissions.ndim != 2:
        raise ValueError(
            f"log_emissions must be 2-D (T, S), got {tuple(log_emissions.shape)}"
        )
    T, S = log_emissions.shape
    if log_transition.shape != (S, S):
        raise ValueError(
            f"log_transition must be ({S}, {S}), got {tuple(log_transition.shape)}"
        )
    if T == 0:
        return torch.empty((0,), dtype=torch.long, device=log_emissions.device)

    if log_initial is None:
        log_init = torch.zeros((S,), device=log_emissions.device)
    else:
        if log_initial.shape != (S,):
            raise ValueError(
                f"log_initial must be ({S},), got {tuple(log_initial.shape)}"
            )
        log_init = log_initial

    v = log_init + log_emissions[0]
    backptr = torch.zeros((T, S), dtype=torch.long, device=log_emissions.device)

    for t in range(1, T):
        # v_prev[j] + log_transition[j, i] for all (j, i) -> (S, S)
        scores = v.unsqueeze(1) + log_transition
        best_val, best_prev = scores.max(dim=0)  # (S,) values, then indices
        backptr[t] = best_prev
        v = best_val + log_emissions[t]

    path = torch.empty((T,), dtype=torch.long, device=log_emissions.device)
    path[-1] = v.argmax()
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def decode_phase_sequence(
    phase_logits: Tensor,
    transition: Tensor,
) -> Tensor:
    """Decode the MAP phase sequence from phase-head logits.

    Args:
        phase_logits: (T, P) per-step phase logits (the stage-1 predictor).
        transition: (P, P) transition prior (from build_transition_prior).

    Returns:
        (T,) int64 decoded phase indices.
    """
    return viterbi_decode(
        torch.log_softmax(phase_logits, dim=-1),
        transition.log(),
    )


def decode_router_sequence(
    gate_logits: Tensor,
    transition: Tensor,
    affinity: Tensor,
) -> Tensor:
    """Decode the MAP expert sequence from router logits.

    The expert-level transition prior is induced from the phase transition
    prior through the affinity matrix (phase p's probability mass on each
    expert): ``T_exp[j, i] ∝ Σ_{p,q} T_phase[q, p] · A[p, i] · A[q, j]``
    (row-normalized), where ``A[p, i]`` is the measured probability that a
    phase-p token is routed to expert i.

    Args:
        gate_logits: (T, E) per-step gate logits (the stage-2 router).
        transition: (P, P) phase transition prior.
        affinity: (P, E) phase→expert affinity matrix (row-normalized).

    Returns:
        (T,) int64 decoded expert indices.
    """
    P, E = affinity.shape
    if transition.shape != (P, P):
        raise ValueError(
            f"transition must be ({P}, {P}) to match affinity rows, got "
            f"{tuple(transition.shape)}"
        )
    # T_exp[j, i]: transition from expert i to expert j.
    t_exp = torch.zeros((E, E), dtype=torch.float64, device=affinity.device)
    t_phase = transition.to(torch.float64)
    a = affinity.to(torch.float64)
    for j in range(E):
        for i in range(E):
            t_exp[j, i] = (t_phase * a[:, i].unsqueeze(1) * a[:, j].unsqueeze(0)).sum()
    row_sums = t_exp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    t_exp = t_exp / row_sums
    return viterbi_decode(
        torch.log_softmax(gate_logits, dim=-1),
        t_exp.float().log(),
    )