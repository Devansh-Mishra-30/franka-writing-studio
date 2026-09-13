"""Authoritative robot application state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RobotState(str, Enum):
    INIT = "INIT"
    HOMING = "HOMING"
    READY = "READY"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    COMPLETE = "COMPLETE"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class StateTransition:
    previous: RobotState
    current: RobotState


_ALLOWED_TRANSITIONS: dict[RobotState, frozenset[RobotState]] = {
    RobotState.INIT: frozenset({
        RobotState.HOMING,
        RobotState.FAULT,
    }),
    RobotState.HOMING: frozenset({
        RobotState.READY,
        RobotState.STOPPED,
        RobotState.FAULT,
    }),
    RobotState.READY: frozenset({
        RobotState.HOMING,
        RobotState.PLANNING,
        RobotState.STOPPED,
        RobotState.FAULT,
    }),
    RobotState.PLANNING: frozenset({
        RobotState.VALIDATING,
        RobotState.STOPPED,
        RobotState.FAULT,
    }),
    RobotState.VALIDATING: frozenset({
        RobotState.EXECUTING,
        RobotState.STOPPED,
        RobotState.FAULT,
    }),
    RobotState.EXECUTING: frozenset({
        RobotState.COMPLETE,
        RobotState.STOPPED,
        RobotState.FAULT,
    }),
    RobotState.COMPLETE: frozenset({
        RobotState.READY,
        RobotState.HOMING,
        RobotState.FAULT,
    }),
    RobotState.STOPPED: frozenset({
        RobotState.READY,
        RobotState.HOMING,
        RobotState.FAULT,
    }),
    RobotState.FAULT: frozenset({
        RobotState.READY,
        RobotState.HOMING,
    }),
}


class RobotStateMachine:
    """Own and enforce legal robot application state transitions."""

    def __init__(self) -> None:
        self._state = RobotState.INIT
        self._last_transition: StateTransition | None = None

    @property
    def state(self) -> RobotState:
        return self._state

    @property
    def last_transition(self) -> StateTransition | None:
        return self._last_transition

    def can_transition(self, target: RobotState) -> bool:
        return target in _ALLOWED_TRANSITIONS[self._state]

    def transition(self, target: RobotState) -> StateTransition:
        if not isinstance(target, RobotState):
            raise TypeError("target must be a RobotState")

        if not self.can_transition(target):
            raise RuntimeError(
                "Illegal robot state transition: "
                f"{self._state.value} -> {target.value}"
            )

        transition = StateTransition(
            previous=self._state,
            current=target,
        )

        self._state = target
        self._last_transition = transition

        return transition
