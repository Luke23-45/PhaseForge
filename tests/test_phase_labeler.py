"""C4 gate: invariant tests for the rule-based phase labeler.

Covers the P-Stage 1 index-awareness fix (explicit ``eef_pos_slice`` /
``gripper_qpos_slice`` derived from ``state_keys``) plus the invariants
the bootstrap relies on: full 6-phase cycle, label range, length,
dtype, and minimum-duration enforcement.
"""

from __future__ import annotations

import numpy as np
import pytest

from phaseforge.data.libero.phase_labeler import (
    APPROACH,
    GRASP,
    PLACE,
    PRE_GRASP,
    RETRACT,
    TRANSPORT,
    RuleBasedPhaseLabeler,
)

# Canonical proprio layout inside the (T, 151) state:
#   [0:7] joint_pos  [7:14] joint_vel  [14:17] eef_pos
#   [17:21] eef_quat [21:23] gripper_qpos
#   [23:135] object_block (zeroed here)  [135:151] mask (zeroed here)
PROPRIO_DIM = 23
OBJECT_BLOCK_DIM = 16 * 7
MASK_DIM = 16
STATE_DIM = PROPRIO_DIM + OBJECT_BLOCK_DIM + MASK_DIM
EEF_SLICE = (14, 17)
GRIPPER_SLICE = (21, 23)

# Median filter must not erase the transient GRASP/PLACE phases (they are
# exactly min_phase_duration frames long), so size < min_phase_duration.
MIN_PHASE_DURATION = 5
MEDIAN_FILTER_SIZE = 3


def make_cycle_traj(
    T: int = 150,
    close_at: int = 50,
    open_at: int = 100,
    offset: int = 0,
) -> dict[str, np.ndarray]:
    """A grasp-release cycle: open -> close -> open, EEF always moving.

    The proprio block can be shifted by ``offset`` columns so tests can
    exercise explicit slices at non-canonical positions.
    """
    state = np.zeros((T, STATE_DIM + offset), dtype=np.float32)
    eef_start = EEF_SLICE[0] + offset
    gripper_start = GRIPPER_SLICE[0] + offset

    t = np.arange(T, dtype=np.float32)
    state[:, eef_start : eef_start + 3] = np.stack(
        [0.05 * t, np.zeros(T), np.zeros(T)], axis=-1
    )
    gripper = np.full(T, 0.06, dtype=np.float32)
    gripper[close_at:open_at] = 0.005
    state[:, gripper_start : gripper_start + 2] = np.stack(
        [gripper, gripper], axis=-1
    )
    return {"state": state}


def make_labeler(**kwargs) -> RuleBasedPhaseLabeler:
    defaults = {
        "min_phase_duration": MIN_PHASE_DURATION,
        "median_filter_size": MEDIAN_FILTER_SIZE,
    }
    defaults.update(kwargs)
    return RuleBasedPhaseLabeler(**defaults)


# ---------------------------------------------------------------------------
# Full-cycle detection
# ---------------------------------------------------------------------------


def test_full_six_phase_cycle_detected_in_order() -> None:
    """A single grasp-release cycle yields all six phases, monotonically."""
    traj = make_cycle_traj()
    labels = make_labeler().label(traj)

    assert labels.dtype == np.int64
    assert labels.shape == (traj["state"].shape[0],)
    assert set(labels.tolist()) == {
        APPROACH, PRE_GRASP, GRASP, TRANSPORT, PLACE, RETRACT,
    }
    assert np.all(np.diff(labels) >= 0), "phase order must be monotonic"


def test_min_duration_enforced_after_first_segment() -> None:
    """Every segment except possibly the first is >= min_phase_duration."""
    labels = make_labeler().label(make_cycle_traj())
    changes = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], changes, [len(labels)]])
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == 0:
            continue
        assert end - start >= MIN_PHASE_DURATION, (
            f"segment [{start}, {end}) of phase {labels[start]} is too short"
        )


def test_labels_always_in_range_on_random_states() -> None:
    """Invariant: labels stay in [0, num_phases) for arbitrary inputs."""
    rng = np.random.default_rng(0)
    labeler = make_labeler()
    for _ in range(10):
        T = rng.integers(1, 200)
        state = rng.normal(0, 1, (T, STATE_DIM))
        labels = labeler.label({"state": state})
        assert labels.shape == (T,)
        assert labels.min() >= 0
        assert labels.max() < labeler.num_phases


def test_empty_trajectory_returns_empty_labels() -> None:
    labels = make_labeler().label({"state": np.zeros((0, STATE_DIM))})
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# Index-aware slicing (P-Stage 1 fix)
# ---------------------------------------------------------------------------


def test_explicit_slices_match_canonical_fallback() -> None:
    """Explicit canonical slices produce identical labels to the fallback."""
    traj = make_cycle_traj()
    canonical = make_labeler().label(traj)
    sliced = make_labeler(
        eef_pos_slice=EEF_SLICE,
        gripper_qpos_slice=GRIPPER_SLICE,
    ).label(traj)
    np.testing.assert_array_equal(sliced, canonical)


def test_slices_honored_when_proprio_block_shifted() -> None:
    """With a shifted proprio block, configured slices beat hardcoded 14:17."""
    offset = 10
    shifted = make_cycle_traj(offset=offset)
    canonical = make_labeler().label(make_cycle_traj())
    shifted_labels = make_labeler(
        eef_pos_slice=(EEF_SLICE[0] + offset, EEF_SLICE[1] + offset),
        gripper_qpos_slice=(GRIPPER_SLICE[0] + offset, GRIPPER_SLICE[1] + offset),
    ).label(shifted)
    np.testing.assert_array_equal(shifted_labels, canonical)


def test_no_slices_on_extended_state_uses_canonical_branch() -> None:
    """Without slices, a 151-dim state falls back to the canonical 14:17/21:23."""
    labels = make_labeler().label(make_cycle_traj())
    assert set(labels.tolist()) == {
        APPROACH, PRE_GRASP, GRASP, TRANSPORT, PLACE, RETRACT,
    }


def test_short_state_fallback_does_not_crash() -> None:
    """A minimal 9-dim state still yields valid labels."""
    T = 60
    state = np.zeros((T, 9), dtype=np.float32)
    state[:, 0:3] = 0.05 * np.arange(T, dtype=np.float32)[:, None]
    state[:, 8] = 0.06
    state[20:40, 8] = 0.005
    labels = make_labeler().label({"state": state})
    assert labels.shape == (T,)
    assert labels.min() >= 0
    assert labels.max() < make_labeler().num_phases


@pytest.mark.parametrize(
    "slice_arg",
    [
        {"eef_pos_slice": (14, 17)},
        {"gripper_qpos_slice": (21, 23)},
    ],
)
def test_partial_slice_configuration_falls_back_to_canonical(slice_arg) -> None:
    """Only one slice configured -> canonical fallback (no partial use)."""
    traj = make_cycle_traj()
    labels = make_labeler(**slice_arg).label(traj)
    np.testing.assert_array_equal(
        labels, make_labeler().label(traj)
    )
