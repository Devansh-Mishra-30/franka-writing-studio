"""Standardized robot application faults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FaultCode(IntEnum):
    NONE = 0

    IK_FAILURE = 1001
    WORKSPACE_VIOLATION = 1002
    COLLISION_PRECHECK_FAILED = 1003

    TRAJECTORY_TIMEOUT = 2001
    CONTROLLER_FAILURE = 2002
    TRACKING_ERROR_EXCEEDED = 2003

    TOOL_UNAVAILABLE = 3001

    SIMULATION_DISCONNECTED = 4001

    COMMAND_REJECTED = 5001
    INVALID_STATE_TRANSITION = 5002

    SAFETY_INHIBIT = 6001


@dataclass(frozen=True)
class RobotFault:
    """Structured fault exposed to GUI, logs, tests, and future PLC."""

    code: FaultCode
    message: str
    source: str
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.code is FaultCode.NONE:
            raise ValueError(
                "RobotFault cannot use FaultCode.NONE"
            )

        if not self.message.strip():
            raise ValueError(
                "Fault message must not be empty"
            )

        if not self.source.strip():
            raise ValueError(
                "Fault source must not be empty"
            )
