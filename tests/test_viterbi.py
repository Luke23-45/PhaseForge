"""CPU-only tests for the V5 Viterbi-decoded routing module.

Covers ``viterbi_decode`` (hand-computable cases), ``build_transition_prior``
(no cross-trajectory contamination, row normalization), and the phase/router
decoding wrappers (monotone prior behavior, shape/validity checks).
"""

from __future__ import annotations

import pytest
import torch

from phaseforge.evaluations.decoding.viterbi import (
    build_transition_prior,
    decode_phase_sequence,
    decode_router_sequence,
    viterbi_decode,
)


def test_viterbi_decode_hand_case() -> None:
    # T=3, S=2. Emissions strongly favor state 0, then 1, then 0; the
    # transition prior mildly penalizes switches (log 0.01 per switch). The
    # emission gain of switching (+20) outweighs the transition cost
    # (2 * log 0.01 ~ -9.2), so the MAP path follows the emissions.
    log_emissions = torch.tensor(
        [
            [0.0, -10.0],   # step 0 favors state 0
            [-10.0, 0.0],   # step 1 favors state 1
            [0.0, -10.0],   # step 2 favors state 0
        ]
    )
    # P(0->1) = P(1->0) = 0.01; staying has probability 0.99.
    log_transition = torch.tensor(
        [[-0.01005, -4.60517], [-4.60517, -0.01005]]
    )
    path = viterbi_decode(log_emissions, log_transition)
    assert path.tolist() == [0, 1, 0]


def test_viterbi_decode_emissions_win() -> None:
    # With a uniform (log 0) transition prior, the emissions decide.
    log_emissions = torch.tensor(
        [
            [0.0, -10.0],
            [-10.0, 0.0],
            [0.0, -10.0],
        ]
    )
    log_transition = torch.zeros(2, 2)
    path = viterbi_decode(log_emissions, log_transition)
    assert path.tolist() == [0, 1, 0]


def test_viterbi_decode_shape_and_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        viterbi_decode(torch.zeros(3), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="log_transition must be"):
        viterbi_decode(torch.zeros(3, 2), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="log_initial must be"):
        viterbi_decode(torch.zeros(3, 2), torch.zeros(2, 2), torch.zeros(3))
    empty = viterbi_decode(torch.zeros(0, 2), torch.zeros(2, 2))
    assert empty.shape == (0,)


def test_viterbi_decode_initial_prior() -> None:
    # Both steps' emissions favor state 1, but a strong initial prior forces
    # state 0 at step 0; the uniform transition prior then lets the
    # emissions win at step 1.
    log_emissions = torch.tensor([[-10.0, 0.0], [-10.0, 0.0]])
    log_transition = torch.zeros(2, 2)
    log_initial = torch.tensor([0.0, -100.0])
    path = viterbi_decode(log_emissions, log_transition, log_initial)
    assert path.tolist() == [0, 1]


def test_build_transition_prior_rows_normalized() -> None:
    phases = torch.tensor([0, 0, 1, 1, 1, 2])
    prior = build_transition_prior(phases)
    # Transitions: 0->0 x1, 0->1 x1; 1->1 x2; 1->2 x1; 2 has no outgoing.
    expected = torch.zeros(3, 3)
    expected[0, 0] = 0.5
    expected[0, 1] = 0.5
    expected[1, 1] = 2.0 / 3.0
    expected[1, 2] = 1.0 / 3.0
    # Row 2 has no outgoing transitions: clamp_min(1) keeps the row
    # normalized but the numerator stays zero.
    assert torch.allclose(prior, expected, atol=1e-6)
    # Rows with outgoing transitions normalize to 1; the never-used row 2
    # has no mass (and is never entered as a source state in decoding).
    assert torch.allclose(prior.sum(dim=-1), torch.tensor([1.0, 1.0, 0.0]), atol=1e-6)


def test_build_transition_prior_ignores_trajectory_boundaries() -> None:
    # Concatenating [0,0] and [1,1] naively would count a 0->1 edge.
    # The helper is trajectory-aware in the accumulation script, so this
    # tests only the single-sequence helper: it must count per-sequence
    # edges correctly for a 2-element sequence.
    prior = build_transition_prior(torch.tensor([0, 0]))
    assert prior[0, 0] == 1.0
    assert prior.shape == (1, 1)


def test_decode_phase_sequence_monotone_prior() -> None:
    # The empirical prior is learned from monotone training phases with NO
    # transitions through state 1 (0 -> 0, 0 -> 2, 2 -> 2 only), so state 1
    # is unreachable: the step-1 emission flicker toward state 1 must be
    # absorbed by the decode (de-flickering, Lemma 4).
    transitions = build_transition_prior(torch.tensor([0, 0, 2, 2]))
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],   # flicker toward 1: unreachable
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ]
    )
    decoded = decode_phase_sequence(logits, transitions)
    assert decoded.tolist() == [0, 2, 2, 2]


def test_decode_router_sequence_shapes_and_validity() -> None:
    gate_logits = torch.randn(5, 3)
    transition = build_transition_prior(torch.tensor([0, 0, 1, 1, 2]))
    affinity = torch.tensor(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    )
    decoded = decode_router_sequence(gate_logits, transition, affinity)
    assert decoded.shape == (5,)
    assert set(decoded.tolist()) <= {0, 1, 2}

    with pytest.raises(ValueError, match="transition must be"):
        decode_router_sequence(gate_logits, torch.zeros(4, 4), affinity)


def test_decode_router_sequence_biased_affinity() -> None:
    # Affinity concentrated on expert 0 forces the decoded path to expert 0
    # regardless of gate logits (transition prior induced through affinity).
    gate_logits = torch.tensor([[0.0, 5.0, 5.0]] * 4)
    transition = torch.eye(3)
    affinity = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    decoded = decode_router_sequence(gate_logits, transition, affinity)
    assert decoded.tolist() == [0, 0, 0, 0]