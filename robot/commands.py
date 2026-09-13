"""High-level commands accepted by the robot application layer."""

from enum import Enum


class RobotCommand(str, Enum):
    """Stable command API for GUI, CLI, tests, and future PLC adapters."""

    HOME = "HOME"
    START_WRITING = "START_WRITING"
    STOP = "STOP"
    RESET = "RESET"
