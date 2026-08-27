"""Common task lifecycle used by every workcell scenario."""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol


class TaskStatus(Enum):
    IDLE = auto()
    PLANNING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETE = auto()
    FAILED = auto()


class Task(Protocol):
    """Minimal interface implemented by all robot tasks."""

    @property
    def name(self) -> str:
        ...

    @property
    def status(self) -> TaskStatus:
        ...

    def plan(self) -> None:
        ...

    def validate(self) -> None:
        ...

    def run(self) -> None:
        ...

    def reset(self) -> None:
        ...
