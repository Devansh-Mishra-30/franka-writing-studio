"""Timed approach/write/lift/transfer/retreat motion planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from motion_profiles import (
    CartesianSetpoint,
    minimum_jerk_segment,
)


class WritingPhase(Enum):
    APPROACH = auto()
    LOWER = auto()
    WRITE = auto()
    LIFT = auto()
    TRANSFER = auto()
    RETREAT = auto()


@dataclass(frozen=True)
class CartesianSegment:
    """One timed Cartesian motion segment."""

    phase: WritingPhase
    start_m: np.ndarray
    end_m: np.ndarray
    duration_s: float


@dataclass(frozen=True)
class WritingSetpoint:
    """Desired Cartesian state and active writing phase."""

    phase: WritingPhase
    segment_index: int
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray


def classify_segment(
    start_m: np.ndarray,
    end_m: np.ndarray,
    *,
    write_height_m: float,
    z_tolerance_m: float = 1e-6,
) -> WritingPhase:
    """Classify SVG motion based on vertical position."""

    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)

    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            "start_m and end_m must both have shape (3,)"
        )

    dz = float(end[2] - start[2])

    if dz < -z_tolerance_m:
        return WritingPhase.LOWER

    if dz > z_tolerance_m:
        return WritingPhase.LIFT

    mean_z_m = 0.5 * float(
        start[2] + end[2]
    )

    if mean_z_m > write_height_m + z_tolerance_m:
        return WritingPhase.TRANSFER

    return WritingPhase.WRITE


def segment_duration(
    start_m: np.ndarray,
    end_m: np.ndarray,
    *,
    speed_m_s: float,
    minimum_duration_s: float,
) -> float:
    """Compute segment duration from distance and speed."""

    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")

    if minimum_duration_s <= 0.0:
        raise ValueError(
            "minimum_duration_s must be positive"
        )

    distance_m = float(
        np.linalg.norm(
            np.asarray(end_m, dtype=float)
            - np.asarray(start_m, dtype=float)
        )
    )

    return max(
        minimum_duration_s,
        distance_m / speed_m_s,
    )


class TimedWritingPlan:
    """Complete smooth Cartesian writing execution plan."""

    def __init__(
        self,
        *,
        initial_position_m: np.ndarray,
        svg_positions_m: np.ndarray,
        write_height_m: float,
        approach_speed_m_s: float = 0.10,
        write_speed_m_s: float = 0.04,
        transfer_speed_m_s: float = 0.10,
        vertical_speed_m_s: float = 0.05,
        retreat_clearance_m: float = 0.05,
        minimum_duration_s: float = 0.10,
        speed_scale: float = 1.0,
    ) -> None:
        initial = np.asarray(
            initial_position_m,
            dtype=float,
        )

        positions = np.asarray(
            svg_positions_m,
            dtype=float,
        )

        if initial.shape != (3,):
            raise ValueError(
                "initial_position_m must have shape (3,)"
            )

        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] < 2
        ):
            raise ValueError(
                "svg_positions_m must have shape (N, 3), N >= 2"
            )

        if (
            not np.isfinite(speed_scale)
            or speed_scale <= 0.0
        ):
            raise ValueError(
                "speed_scale must be finite and positive"
            )

        scaled_minimum_duration_s = (
            minimum_duration_s
            / speed_scale
        )

        self._segments: list[CartesianSegment] = []

        # 1. Physically move from the robot's actual initial
        # Cartesian position to the first retracted SVG point.
        first = positions[0].copy()

        self._segments.append(
            CartesianSegment(
                phase=WritingPhase.APPROACH,
                start_m=initial.copy(),
                end_m=first.copy(),
                duration_s=segment_duration(
                    initial,
                    first,
                    speed_m_s=(
                        approach_speed_m_s
                        * speed_scale
                    ),
                    minimum_duration_s=(
                        scaled_minimum_duration_s
                    ),
                ),
            )
        )

        # 2. Convert every SVG waypoint transition into an
        # explicit writing phase.
        for index in range(
            positions.shape[0] - 1
        ):
            start = positions[index].copy()
            end = positions[index + 1].copy()

            phase = classify_segment(
                start,
                end,
                write_height_m=write_height_m,
            )

            if phase == WritingPhase.WRITE:
                speed = (
                    write_speed_m_s
                    * speed_scale
                )
            elif phase == WritingPhase.TRANSFER:
                speed = (
                    transfer_speed_m_s
                    * speed_scale
                )
            else:
                speed = (
                    vertical_speed_m_s
                    * speed_scale
                )

            self._segments.append(
                CartesianSegment(
                    phase=phase,
                    start_m=start,
                    end_m=end,
                    duration_s=segment_duration(
                        start,
                        end,
                        speed_m_s=speed,
                        minimum_duration_s=(
                        scaled_minimum_duration_s
                    ),
                    ),
                )
            )

        # 3. Final retreat after the SVG parser's final pen lift.
        final = positions[-1].copy()

        retreat = final.copy()
        retreat[2] += retreat_clearance_m

        self._segments.append(
            CartesianSegment(
                phase=WritingPhase.RETREAT,
                start_m=final,
                end_m=retreat,
                duration_s=segment_duration(
                    final,
                    retreat,
                    speed_m_s=(
                        vertical_speed_m_s
                        * speed_scale
                    ),
                    minimum_duration_s=(
                        scaled_minimum_duration_s
                    ),
                ),
            )
        )

        self._segment_start_times_s: list[float] = []

        elapsed_s = 0.0

        for segment in self._segments:
            self._segment_start_times_s.append(
                elapsed_s
            )
            elapsed_s += segment.duration_s

        self._total_duration_s = elapsed_s

    @property
    def segments(self) -> tuple[CartesianSegment, ...]:
        return tuple(self._segments)

    @property
    def total_duration_s(self) -> float:
        return self._total_duration_s

    def sample(
        self,
        time_s: float,
    ) -> WritingSetpoint:
        """Sample the complete writing plan at simulation time."""

        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")

        if time_s < 0.0:
            raise ValueError(
                "time_s must be nonnegative"
            )

        # Hold final retreat point after plan completion.
        if time_s >= self._total_duration_s:
            final_segment = self._segments[-1]

            return WritingSetpoint(
                phase=final_segment.phase,
                segment_index=len(self._segments) - 1,
                position_m=final_segment.end_m.copy(),
                velocity_m_s=np.zeros(3),
                acceleration_m_s2=np.zeros(3),
            )

        segment_index = 0

        for index, start_time_s in enumerate(
            self._segment_start_times_s
        ):
            end_time_s = (
                start_time_s
                + self._segments[index].duration_s
            )

            if time_s < end_time_s:
                segment_index = index
                break

        segment = self._segments[segment_index]

        local_time_s = (
            time_s
            - self._segment_start_times_s[
                segment_index
            ]
        )

        sample: CartesianSetpoint = (
            minimum_jerk_segment(
                segment.start_m,
                segment.end_m,
                local_time_s,
                segment.duration_s,
            )
        )

        return WritingSetpoint(
            phase=segment.phase,
            segment_index=segment_index,
            position_m=sample.position_m,
            velocity_m_s=sample.velocity_m_s,
            acceleration_m_s2=sample.acceleration_m_s2,
        )
