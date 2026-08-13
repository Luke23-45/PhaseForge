"""Unit tests for the adaptive rule-based phase labeler.

The core regression: with fixed absolute thresholds (closed < 0.02,
open > 0.04) the robomimic Lift PH panda gripper signal (finger qpos in
[-0.02, 0.02], closed at the positive end) degenerated to only phases
2/3 — grasp fired at t=0 and release never fired. The labeler must now
populate all six phases for every common gripper convention.
"""

import numpy as np
import pytest

from phaseforge.data.robomimic.phase_labeler import RuleBasedPhaseLabeler


def _build_demo(open_aperture: float, closed_aperture: float, segments) -> dict:
    """Build a 9-dim state (eef_pos(3) + quat(4) + gripper_qpos(2)) demo.

    ``segments`` is a list of (num_samples, aperture, eef_speed) tuples; the
    eef position accumulates the per-sample speed so the velocity threshold
    is exercised.
    """
    states = []
    pos = 0.0
    for n, aperture, speed in segments:
        for _ in range(n):
            pos += speed
            states.append((pos, aperture))
    T = len(states)
    state = np.zeros((T, 9), dtype=np.float32)
    for t, (p, aperture) in enumerate(states):
        state[t, 0] = p
        state[t, 7:9] = aperture
    return {"state": state}


CANONICAL_SEGMENTS = [
    (60, None, 0.02),   # approach
    (20, None, 0.0),    # pre-grasp hold (open, stationary)
    (10, None, 0.002),  # grasp close
    (15, None, 0.0),    # grip hold
    (40, None, 0.03),   # transport
    (20, None, 0.003),  # place
    (10, None, 0.02),   # release + retract start
    (20, None, 0.02),   # retract
    (15, None, 0.0),    # settle
]


def _canonical(open_aperture: float, closed_aperture: float) -> dict:
    segments = [
        (n, closed_aperture if i in (2, 3, 4, 5) else open_aperture, speed)
        for i, (n, _unused, speed) in enumerate(CANONICAL_SEGMENTS)
    ]
    return _build_demo(open_aperture, closed_aperture, segments)


def _first_occurrence(phases: np.ndarray) -> dict[int, int]:
    first: dict[int, int] = {}
    for t, p in enumerate(phases):
        if p not in first:
            first[int(p)] = t
    return first


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (-0.02, 0.02),  # panda finger qpos, closed at the positive end
        (0.04, 0.0),    # magnitude convention, closed at the low end
        (0.0, 0.04),    # magnitude convention, closed at the high end
    ],
)
def test_all_six_phases_populated(open_aperture, closed_aperture) -> None:
    """Regression: fixed thresholds degenerated to {2, 3} for panda qpos."""
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))

    assert set(phases.tolist()) == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (-0.02, 0.02),
        (0.04, 0.0),
        (0.0, 0.04),
    ],
)
def test_phase_order_is_approach_to_retract(open_aperture, closed_aperture) -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))
    first = _first_occurrence(phases)

    assert first[0] == 0
    for p in range(5):
        assert first[p] < first[p + 1], f"phase {p} first occurs after phase {p + 1}"


@pytest.mark.parametrize(
    ("open_aperture", "closed_aperture"),
    [
        (-0.02, 0.02),
        (0.04, 0.0),
    ],
)
def test_post_grasp_phases_dominate_approach(open_aperture, closed_aperture) -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label(_canonical(open_aperture, closed_aperture))
    counts = np.bincount(phases, minlength=6)

    assert counts[2:].sum() > counts[0]


def test_constant_aperture_falls_back_to_absolute_thresholds() -> None:
    labeler = RuleBasedPhaseLabeler()
    segments = [
        (n, 0.04, speed) for n, _unused, speed in CANONICAL_SEGMENTS
    ]
    phases = labeler.label(_build_demo(0.04, 0.04, segments))

    assert set(phases.tolist()).issubset({0, 1})


def test_short_trajectory_no_crash() -> None:
    labeler = RuleBasedPhaseLabeler()
    state = np.zeros((3, 9), dtype=np.float32)
    state[:, 7:9] = 0.04
    phases = labeler.label({"state": state})
    assert phases.shape == (3,)
    assert set(phases.tolist()).issubset({0, 1})


def test_empty_trajectory() -> None:
    labeler = RuleBasedPhaseLabeler()
    phases = labeler.label({"state": np.zeros((0, 9), dtype=np.float32)})
    assert phases.shape == (0,)


def test_invalid_slices_raise() -> None:
    labeler = RuleBasedPhaseLabeler(eef_pos_slice=(0, 20))
    with pytest.raises(ValueError, match="invalid for state_dim"):
        labeler.label({"state": np.zeros((10, 9), dtype=np.float32)})


def test_num_phases_vocabulary_guard() -> None:
    with pytest.raises(ValueError, match="exactly 6 phases"):
        RuleBasedPhaseLabeler(num_phases=4)


def test_threshold_ordering_guard() -> None:
    with pytest.raises(ValueError, match="below gripper_open_threshold"):
        RuleBasedPhaseLabeler(gripper_closed_threshold=0.04, gripper_open_threshold=0.02)