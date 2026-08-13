"""Rule-based phase labeler for LIBERO trajectories.

Derives integer phase labels from proprioceptive state signals.
No manual annotation required. Labels are fully deterministic given
the same config and input trajectory.

Phase vocabulary (default 6 phases):
    0  APPROACH    — EEF moving toward target, gripper open
    1  PRE_GRASP   — EEF decelerating, near object
    2  GRASP       — Gripper closing, contact event
    3  TRANSPORT   — EEF moving with object, gripper closed
    4  PLACE       — EEF decelerating at target, gripper opening
    5  RETRACT     — EEF moving away, gripper open
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

# Phase integer constants
APPROACH = 0
PRE_GRASP = 1
GRASP = 2
TRANSPORT = 3
PLACE = 4
RETRACT = 5


class RuleBasedPhaseLabeler:
    """Derive integer phase labels from raw (unnormalized) state trajectories.

    Args:
        num_phases:               Number of phase classes (default 6).
        gripper_closed_threshold: Gripper aperture below this → closed.
        gripper_open_threshold:   Gripper aperture above this → open.
        eef_velocity_threshold:   EEF speed below this → stationary.
        min_phase_duration:       Merge segments shorter than this (timesteps).
        median_filter_size:       Window for temporal smoothing.
    """

    def __init__(
        self,
        num_phases: int = 6,
        gripper_closed_threshold: float = 0.02,
        gripper_open_threshold: float = 0.04,
        eef_velocity_threshold: float = 0.01,
        min_phase_duration: int = 5,
        median_filter_size: int = 7,
        eef_pos_slice: tuple[int, int] | None = None,
        gripper_qpos_slice: tuple[int, int] | None = None,
    ) -> None:
        self.num_phases = num_phases
        self.gripper_closed_threshold = gripper_closed_threshold
        self.gripper_open_threshold = gripper_open_threshold
        if not self.gripper_closed_threshold < self.gripper_open_threshold:
            raise ValueError(
                f"gripper_closed_threshold={gripper_closed_threshold} must be "
                f"strictly below gripper_open_threshold={gripper_open_threshold} "
                "for the hysteresis band to exist."
            )
        self.eef_velocity_threshold = eef_velocity_threshold
        self.min_phase_duration = min_phase_duration
        self.median_filter_size = median_filter_size
        # Explicit slices into the state vector (end indices exclusive),
        # derived by the pipeline from the configured state_keys. When unset,
        # _extract_signals falls back to the canonical 23-dim layout.
        self.eef_pos_slice = (
            tuple(int(v) for v in eef_pos_slice) if eef_pos_slice is not None else None
        )
        self.gripper_qpos_slice = (
            tuple(int(v) for v in gripper_qpos_slice)
            if gripper_qpos_slice is not None
            else None
        )

    def label(self, traj: dict[str, Any]) -> np.ndarray:
        """Produce a (T,) integer phase label array from a trajectory dict.

        The input state is the raw (unnormalized) numpy array.
        Indices into the state vector are inferred from available dimensions.

        Args:
            traj: Dict with key ``"state"`` of shape (T, state_dim).

        Returns:
            np.ndarray of dtype int64, shape (T,), values in [0, num_phases).
        """
        state: np.ndarray = traj["state"]  # (T, state_dim)
        T = state.shape[0]

        if T == 0:
            return np.zeros(0, dtype=np.int64)

        # -------------------------------------------------------------------
        # Extract sub-signals
        # -------------------------------------------------------------------
        eef_pos, gripper_aperture = self._extract_signals(state)

        # EEF velocity: finite differences (prepend first row so shape stays T)
        eef_vel = np.linalg.norm(
            np.diff(eef_pos, axis=0, prepend=eef_pos[:1]), axis=-1
        )  # (T,)

        # Binary gripper state via HYSTERESIS: aperture below the closed
        # threshold -> closed, above the open threshold -> open, and between
        # the two -> hold the previous state (defaults to open on the first
        # frame). This gives BOTH configured thresholds a meaning and keeps
        # sensor noise around a single cutoff from flipping the state.
        gripper_closed = np.zeros(T, dtype=bool)
        prev_closed: bool | None = None
        for t in range(T):
            a = gripper_aperture[t]
            if a < self.gripper_closed_threshold:
                cur = True
            elif a > self.gripper_open_threshold:
                cur = False
            else:
                cur = prev_closed if prev_closed is not None else False
            gripper_closed[t] = cur
            prev_closed = cur

        # -------------------------------------------------------------------
        # Detect gripper events
        # -------------------------------------------------------------------
        gripper_state_int = gripper_closed.astype(np.int8)
        diff = np.diff(gripper_state_int, prepend=gripper_state_int[:1])
        grasp_events = diff > 0    # open → closed  (GRASP event)
        release_events = diff < 0  # closed → open  (PLACE/RELEASE event)

        # -------------------------------------------------------------------
        # Sweep through trajectory assigning phases
        # -------------------------------------------------------------------
        # NOTE on transient phases: GRASP and PLACE are entered at a single
        # event frame (gripper close / open). Without protection, the very
        # next frame's velocity check would immediately refine GRASP->TRANSPORT
        # and PLACE->RETRACT, leaving GRASP/PLACE as single-frame states that
        # the median filter then erases. Simulation proved this dropped both
        # to 0%. We therefore HOLD a transient phase for at least
        # `min_phase_duration` frames before allowing refinement, so every
        # phase survives post-processing.
        phases = np.zeros(T, dtype=np.int64)
        current_phase = APPROACH
        phase_entered_at = 0  # timestep at which current_phase began

        for t in range(T):
            held_long_enough = (t - phase_entered_at) >= self.min_phase_duration

            # Phase transitions driven by gripper events
            if grasp_events[t]:
                # Backfill recent slow frames as PRE_GRASP
                lookback = min(self.min_phase_duration * 2, t)
                for bt in range(t - lookback, t):
                    if phases[bt] == APPROACH:
                        phases[bt] = PRE_GRASP
                current_phase = GRASP
                phase_entered_at = t

            elif release_events[t]:
                current_phase = PLACE
                phase_entered_at = t

            else:
                # Refinement within GRASP -> TRANSPORT (only after holding GRASP)
                if current_phase == GRASP and held_long_enough:
                    if gripper_closed[t] and eef_vel[t] > self.eef_velocity_threshold:
                        current_phase = TRANSPORT
                        phase_entered_at = t
                # Refinement within PLACE -> RETRACT (only after holding PLACE)
                elif current_phase == PLACE and held_long_enough:
                    if (not gripper_closed[t]) and eef_vel[t] > self.eef_velocity_threshold:
                        current_phase = RETRACT
                        phase_entered_at = t
                # RETRACT -> APPROACH (begin next sub-task cycle)
                elif current_phase == RETRACT:
                    if eef_vel[t] < self.eef_velocity_threshold:
                        current_phase = APPROACH
                        phase_entered_at = t

            phases[t] = current_phase

        # -------------------------------------------------------------------
        # Post-processing: temporal smoothing + min duration enforcement
        # -------------------------------------------------------------------
        phases = self._smooth(phases)
        phases = self._enforce_min_duration(phases)

        # Clamp to valid range (safety)
        phases = np.clip(phases, 0, self.num_phases - 1)

        return phases.astype(np.int64)

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _extract_signals(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract EEF position and gripper aperture from the state vector.

        When explicit slices are configured (``eef_pos_slice`` /
        ``gripper_qpos_slice``, derived by the pipeline from the configured
        ``state_keys``), those are used verbatim. Otherwise the canonical
        state key order from common.yaml is assumed:
            [0:7]   joint_pos
            [7:14]  joint_vel
            [14:17] eef_pos       ← 3-dim
            [17:21] eef_quat      ← 4-dim
            [21:23] gripper_qpos  ← 2-dim (last two)

        If the state vector is shorter, we gracefully fall back to simpler signals.
        """
        S = state.shape[-1]

        if self.eef_pos_slice is not None and self.gripper_qpos_slice is not None:
            eef_pos = state[:, self.eef_pos_slice[0] : self.eef_pos_slice[1]]
            gripper_qpos = state[
                :, self.gripper_qpos_slice[0] : self.gripper_qpos_slice[1]
            ]
            gripper_aperture = gripper_qpos.mean(axis=-1)
        elif S >= 23:
            eef_pos = state[:, 14:17]
            gripper_qpos = state[:, 21:23]
            gripper_aperture = gripper_qpos.mean(axis=-1)
        elif S >= 9:
            # Minimal: 7 joint pos + 2 gripper
            eef_pos = state[:, :3]  # Approximate with first 3 dims
            gripper_aperture = state[:, -1]  # Last channel as proxy
        else:
            # Ultra-minimal fallback
            eef_pos = state[:, :min(3, S)]
            if eef_pos.shape[-1] < 3:
                eef_pos = np.pad(eef_pos, ((0, 0), (0, 3 - eef_pos.shape[-1])))
            gripper_aperture = np.zeros(state.shape[0])

        return eef_pos, gripper_aperture

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _smooth(self, phases: np.ndarray) -> np.ndarray:
        """Apply median filter to remove single-timestep noise."""
        if len(phases) <= self.median_filter_size:
            return phases
        # scipy median_filter preserves dtype for integer arrays
        return median_filter(phases.astype(np.float32), size=self.median_filter_size).astype(
            np.int64
        )

    def _enforce_min_duration(self, phases: np.ndarray) -> np.ndarray:
        """Merge segments shorter than min_phase_duration into neighbors."""
        if len(phases) == 0:
            return phases

        result = phases.copy()
        T = len(result)
        i = 0
        while i < T:
            current = result[i]
            j = i
            while j < T and result[j] == current:
                j += 1
            seg_len = j - i
            if seg_len < self.min_phase_duration and i > 0:
                # Merge: fill with the preceding phase
                result[i:j] = result[i - 1]
            i = j
        return result
