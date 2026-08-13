"""Deterministic auxiliary phase labels for low-dimensional demonstrations.

These labels are a training signal for the PhaseForge routing study. They are
not environment annotations and are never supplied to the policy at inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter


class RuleBasedPhaseLabeler:
    """Assign coarse approach/grasp/transport/place/retract phases.

    The labeler only reads the configured end-effector position and gripper
    slices from the unnormalized state. It deliberately does not inspect
    actions, future states, object state, or task metadata.
    """

    def __init__(
        self,
        num_phases: int = 6,
        gripper_closed_threshold: float = 0.02,
        gripper_open_threshold: float = 0.04,
        eef_velocity_threshold: float = 0.01,
        min_phase_duration: int = 5,
        median_filter_size: int = 7,
        eef_pos_slice: tuple[int, int] = (0, 3),
        gripper_qpos_slice: tuple[int, int] = (7, 9),
    ) -> None:
        if num_phases != 6:
            raise ValueError(
                "The current phase vocabulary has exactly 6 phases; use a "
                "different labeler before changing num_phases."
            )
        if not 0 < min_phase_duration:
            raise ValueError("min_phase_duration must be positive")
        if median_filter_size < 1 or median_filter_size % 2 == 0:
            raise ValueError("median_filter_size must be a positive odd integer")
        if not gripper_closed_threshold < gripper_open_threshold:
            raise ValueError(
                "gripper_closed_threshold must be below gripper_open_threshold"
            )
        self.num_phases = int(num_phases)
        self.closed_threshold = float(gripper_closed_threshold)
        self.open_threshold = float(gripper_open_threshold)
        self.velocity_threshold = float(eef_velocity_threshold)
        self.min_duration = int(min_phase_duration)
        self.filter_size = int(median_filter_size)
        self.eef_pos_slice = tuple(int(v) for v in eef_pos_slice)
        self.gripper_slice = tuple(int(v) for v in gripper_qpos_slice)

    def label(self, traj: dict[str, Any]) -> np.ndarray:
        state = np.asarray(traj["state"], dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"Expected state shape (T, D), got {state.shape}")
        T, D = state.shape
        e0, e1 = self.eef_pos_slice
        g0, g1 = self.gripper_slice
        if not (0 <= e0 < e1 <= D and 0 <= g0 < g1 <= D):
            raise ValueError(
                f"Phase slices eef={self.eef_pos_slice}, gripper={self.gripper_slice} "
                f"are invalid for state_dim={D}"
            )
        if T == 0:
            return np.zeros(0, dtype=np.int64)

        eef = state[:, e0:e1]
        aperture = state[:, g0:g1].mean(axis=1)
        speed = np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=1)

        closed = np.zeros(T, dtype=bool)
        previous = False
        for t, value in enumerate(aperture):
            if value < self.closed_threshold:
                previous = True
            elif value > self.open_threshold:
                previous = False
            closed[t] = previous

        transitions = np.diff(closed.astype(np.int8), prepend=0)
        grasp = transitions > 0
        release = transitions < 0

        # 0 approach, 1 pre-grasp, 2 grasp, 3 transport, 4 place, 5 retract.
        phases = np.zeros(T, dtype=np.int64)
        phase = 0
        entered = 0
        for t in range(T):
            held = t - entered >= self.min_duration
            if grasp[t]:
                lookback = min(2 * self.min_duration, t)
                phases[t - lookback:t][phases[t - lookback:t] == 0] = 1
                phase, entered = 2, t
            elif release[t]:
                phase, entered = 4, t
            elif phase == 2 and held and closed[t] and speed[t] > self.velocity_threshold:
                phase, entered = 3, t
            elif phase == 4 and held and not closed[t] and speed[t] > self.velocity_threshold:
                phase, entered = 5, t
            elif phase == 5 and speed[t] < self.velocity_threshold:
                phase, entered = 0, t
            phases[t] = phase

        if T > self.filter_size:
            phases = median_filter(phases.astype(np.float32), size=self.filter_size).astype(
                np.int64
            )
        phases = self._merge_short_segments(phases)
        return np.clip(phases, 0, self.num_phases - 1).astype(np.int64)

    def _merge_short_segments(self, phases: np.ndarray) -> np.ndarray:
        result = phases.copy()
        i = 0
        while i < len(result):
            j = i + 1
            while j < len(result) and result[j] == result[i]:
                j += 1
            if j - i < self.min_duration and i > 0:
                result[i:j] = result[i - 1]
            i = j
        return result
