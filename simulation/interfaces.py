"""Simulator-independent data structures and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobotState:
    """Robot state exposed by a simulator."""

    simulation_time_s: float

    joint_positions_rad: FloatArray
    joint_velocities_rad_s: FloatArray

    # Pose of fer_link8 in the world frame.
    tool_position_m: FloatArray
    tool_rotation_matrix: FloatArray


@dataclass(frozen=True)
class PenContactState:
    """Physical pen/writing-surface contact state."""

    active: bool

    contact_count: int

    # Minimum PyBullet contact distance.
    #
    # Negative values indicate penetration.
    # None means no contact exists.
    minimum_distance_m: float | None

    # Sum of normal forces across all pen/table contacts.
    normal_force_n: float


@dataclass(frozen=True)
class VelocityCommand:
    """Joint-velocity command."""

    joint_velocities_rad_s: FloatArray


class SimulatorInterface(Protocol):
    """Interface required by the writing experiment."""

    def configure(self) -> None:
        ...

    def reset(
        self,
        joint_positions_rad: FloatArray,
    ) -> None:
        ...

    def read_state(
        self,
        simulation_time_s: float,
    ) -> RobotState:
        ...

    def read_pen_contact_state(
        self,
    ) -> PenContactState:
        ...

    def apply_velocity_command(
        self,
        command: VelocityCommand,
    ) -> None:
        ...

    def step(self) -> None:
        ...

    def shutdown(self) -> None:
        ...
