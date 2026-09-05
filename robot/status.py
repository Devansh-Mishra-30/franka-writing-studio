"""Observable robot application status."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot.commands import RobotCommand
from robot.faults import RobotFault
from robot.state import RobotState, StateTransition


@dataclass(frozen=True)
class RobotStatus:
    """Single authoritative status snapshot for the robot application."""

    state: RobotState
    active_command: RobotCommand | None = None

    task_progress: float = 0.0
    controller_mode: str = "IDLE"

    tcp_position_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    tracking_error_m: float = 0.0
    normal_force_n: float = 0.0
    cycle_time_s: float = 0.0

    fault: RobotFault | None = None
    last_transition: StateTransition | None = None

    def __post_init__(self) -> None:
        position = np.asarray(
            self.tcp_position_m,
            dtype=float,
        )

        if position.shape != (3,):
            raise ValueError(
                "tcp_position_m must have shape (3,)"
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                "tcp_position_m must be finite"
            )

        if not 0.0 <= self.task_progress <= 1.0:
            raise ValueError(
                "task_progress must be within [0, 1]"
            )

        for name, value in (
            ("tracking_error_m", self.tracking_error_m),
            ("normal_force_n", self.normal_force_n),
            ("cycle_time_s", self.cycle_time_s),
        ):
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if self.cycle_time_s < 0.0:
            raise ValueError(
                "cycle_time_s must be non-negative"
            )

        object.__setattr__(
            self,
            "tcp_position_m",
            position.copy(),
        )
