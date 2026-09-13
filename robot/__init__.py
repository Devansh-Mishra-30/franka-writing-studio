"""Robot application-layer API."""

from robot.commands import RobotCommand
from robot.faults import (
    FaultCode,
    RobotFault,
)
from robot.interface import (
    CommandRejected,
    RobotInterface,
)
from robot.state import (
    RobotState,
    RobotStateMachine,
    StateTransition,
)
from robot.status import RobotStatus

__all__ = [
    "CommandRejected",
    "FaultCode",
    "RobotCommand",
    "RobotFault",
    "RobotInterface",
    "RobotState",
    "RobotStateMachine",
    "RobotStatus",
    "StateTransition",
]
