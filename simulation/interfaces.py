"""Simulator-independent data structures and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobotState:
    """Timestamped robot state returned by a simulator."""

    simulation_time_s: float
    joint_positions_rad: FloatArray
    joint_velocities_rad_s: FloatArray

    # This is currently fer_link8, not yet the physical pen tip.
    tool_position_m: FloatArray


@dataclass(frozen=True)
class VelocityCommand:
    """Joint-velocity command sent to a simulator."""

    joint_velocities_rad_s: FloatArray


class SimulatorInterface(Protocol):
    """Minimal interface required by robot experiments."""

    def configure(self) -> None:
        """Connect, configure physics, and load models."""

    def reset(
        self,
        joint_positions_rad: FloatArray,
    ) -> None:
        """Reset the simulated robot state."""

    def read_state(
        self,
        simulation_time_s: float,
    ) -> RobotState:
        """Read the current timestamped robot state."""

    def apply_velocity_command(
        self,
        command: VelocityCommand,
    ) -> None:
        """Apply one joint-velocity command."""

    def step(self) -> None:
        """Advance physics by one configured timestep."""

    def shutdown(self) -> None:
        """Release simulator resources."""
