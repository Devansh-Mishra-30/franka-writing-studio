"""High-level robot application interface.

This module is the boundary used by:
- GUI
- CLI
- automated tests
- future PLC / industrial protocol adapters

Lower-level planning, IK, control, and simulation must remain behind
this interface.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiment_config import ExperimentConfig
from experiments.writing_experiment import (
    ExperimentResult,
    ExperimentStopped,
)
from robot.commands import RobotCommand
from robot.faults import FaultCode, RobotFault
from robot.state import RobotState, RobotStateMachine
from robot.status import RobotStatus
from tasks.writing_studio import (
    WritingStudioTask,
    WritingValidationReport,
)


class CommandRejected(RuntimeError):
    """Raised when a command is not legal in the current robot state."""


class RobotInterface:
    """Authoritative application interface for the Writing Studio."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        task: WritingStudioTask | None = None,
        status_callback: (
            Callable[[RobotStatus], None] | None
        ) = None,
        telemetry_callback: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        telemetry_period_s: float = 0.05,
    ) -> None:
        self._lock = threading.RLock()

        self._state_machine = RobotStateMachine()

        self._active_command: RobotCommand | None = None
        self._fault: RobotFault | None = None

        self._status_callback = status_callback
        self._raw_telemetry_callback = telemetry_callback

        self._task = (
            task
            if task is not None
            else WritingStudioTask(
                config,
                telemetry_callback=self._on_telemetry,
                telemetry_period_s=telemetry_period_s,
            )
        )

        self._status = RobotStatus(
            state=RobotState.INIT,
        )

    # --------------------------------------------------------
    # Public state
    # --------------------------------------------------------

    @property
    def state(self) -> RobotState:
        return self._state_machine.state

    @property
    def status(self) -> RobotStatus:
        with self._lock:
            return self._status

    @property
    def task(self) -> WritingStudioTask:
        return self._task

    # --------------------------------------------------------
    # High-level command API
    # --------------------------------------------------------

    def execute(
        self,
        command: RobotCommand,
    ) -> ExperimentResult | RobotStatus | None:
        """Execute one high-level robot command."""

        if command is RobotCommand.HOME:
            return self.home()

        if command is RobotCommand.START_WRITING:
            return self.start_writing()

        if command is RobotCommand.STOP:
            return self.stop()

        if command is RobotCommand.RESET:
            return self.reset()

        raise CommandRejected(
            f"Unsupported command: {command}"
        )

    def home(self) -> RobotStatus:
        """Prepare the application at its validated home configuration.

        The current v1 simulator is created per experiment rather than
        kept alive continuously. Therefore HOME currently resets the task
        lifecycle and establishes the known home/ready application state.
        A future persistent simulator or hardware adapter can replace this
        implementation with commanded physical homing without changing
        the external RobotInterface contract.
        """

        with self._lock:
            current = self.state

            if current is RobotState.FAULT:
                raise CommandRejected(
                    "RESET is required before HOME from FAULT"
                )

            allowed = {
                RobotState.INIT,
                RobotState.READY,
                RobotState.COMPLETE,
                RobotState.STOPPED,
            }

            if current not in allowed:
                raise CommandRejected(
                    "HOME rejected while robot is "
                    f"{current.value}"
                )

            self._active_command = RobotCommand.HOME

        self._transition(
            RobotState.HOMING,
            controller_mode="HOMING",
            task_progress=0.0,
            cycle_time_s=0.0,
        )

        try:
            self._task.reset()

        except Exception as error:
            self._enter_fault(
                FaultCode.CONTROLLER_FAILURE,
                message=str(error),
                source="home",
            )
            raise

        self._transition(
            RobotState.READY,
            controller_mode="IDLE",
            task_progress=0.0,
            cycle_time_s=0.0,
        )

        with self._lock:
            self._active_command = None

        self._publish()

        return self.status

    def start_writing(self) -> ExperimentResult:
        """Plan, validate, and execute one writing cycle."""

        with self._lock:
            if self.state is not RobotState.READY:
                raise CommandRejected(
                    "START_WRITING requires READY state; "
                    f"current state is {self.state.value}"
                )

            self._fault = None
            self._active_command = (
                RobotCommand.START_WRITING
            )

        self._replace_status(
            task_progress=0.0,
            cycle_time_s=0.0,
            controller_mode="PLANNING",
        )

        try:
            self._transition(
                RobotState.PLANNING,
                controller_mode="PLANNING",
            )

            self._task.plan()

            self._transition(
                RobotState.VALIDATING,
                controller_mode="VALIDATING",
            )

            report = self._task.validate()

            if not report.success:
                code, message = (
                    self._validation_fault(
                        report
                    )
                )

                self._enter_fault(
                    code,
                    message=message,
                    source="validation",
                )

                raise RuntimeError(
                    message
                )

            self._transition(
                RobotState.EXECUTING,
                controller_mode="POSE_TRACKING",
            )

            result = self._task.run()

            self._transition(
                RobotState.COMPLETE,
                controller_mode="IDLE",
                task_progress=1.0,
            )

            with self._lock:
                self._active_command = None

            self._publish()

            return result

        except ExperimentStopped:
            with self._lock:
                if (
                    self.state
                    is RobotState.EXECUTING
                ):
                    self._state_machine.transition(
                        RobotState.STOPPED
                    )

                self._active_command = None

            self._replace_status(
                controller_mode="STOPPED",
            )

            raise

        except Exception as error:
            with self._lock:
                already_faulted = (
                    self.state
                    is RobotState.FAULT
                )

            if not already_faulted:
                code = self._classify_execution_error(
                    error
                )

                self._enter_fault(
                    code,
                    message=str(error),
                    source="execution",
                )

            raise

    def stop(self) -> RobotStatus:
        """Request cooperative interruption of active execution."""

        with self._lock:
            if self.state is RobotState.STOPPED:
                return self._status

            if self.state is not RobotState.EXECUTING:
                raise CommandRejected(
                    "STOP currently requires EXECUTING state; "
                    f"current state is {self.state.value}"
                )

            self._active_command = RobotCommand.STOP

        # Do not transition to STOPPED here.
        #
        # The execution thread owns acknowledgement of the stop.
        # WritingExperiment will observe the cooperative stop event,
        # exit the control loop, shut down PyBullet, and raise
        # ExperimentStopped. start_writing() then transitions to
        # STOPPED.
        self._task.request_stop()

        self._publish()

        return self.status

    def reset(self) -> RobotStatus:
        """Clear task/fault state after completion, stop, or fault."""

        with self._lock:
            current = self.state

            blocked = {
                RobotState.HOMING,
                RobotState.PLANNING,
                RobotState.VALIDATING,
                RobotState.EXECUTING,
            }

            if current in blocked:
                raise CommandRejected(
                    "RESET rejected while robot is "
                    f"{current.value}"
                )

            self._active_command = RobotCommand.RESET

        self._task.reset()

        with self._lock:
            self._fault = None

            if self.state in {
                RobotState.COMPLETE,
                RobotState.STOPPED,
                RobotState.FAULT,
            }:
                self._state_machine.transition(
                    RobotState.READY
                )

            self._active_command = None

        self._replace_status(
            task_progress=0.0,
            controller_mode="IDLE",
            cycle_time_s=0.0,
            tracking_error_m=0.0,
            normal_force_n=0.0,
        )

        return self.status

    # --------------------------------------------------------
    # Telemetry → authoritative status
    # --------------------------------------------------------

    def _on_telemetry(
        self,
        row: dict[str, Any],
    ) -> None:
        """Translate experiment telemetry into RobotStatus."""

        physical_tcp = np.array(
            [
                float(
                    row.get(
                        "physical_pen_x_m",
                        self._status.tcp_position_m[0],
                    )
                ),
                float(
                    row.get(
                        "physical_pen_y_m",
                        self._status.tcp_position_m[1],
                    )
                ),
                float(
                    row.get(
                        "physical_pen_z_m",
                        self._status.tcp_position_m[2],
                    )
                ),
            ],
            dtype=float,
        )

        simulation_time_s = float(
            row.get(
                "simulation_time_s",
                self._status.cycle_time_s,
            )
        )

        total_duration_s = None

        planning = getattr(
            self._task,
            "planning_result",
            None,
        )

        if planning is not None:
            total_duration_s = float(
                planning.writing_plan.total_duration_s
            )

        if (
            total_duration_s is not None
            and total_duration_s > 0.0
        ):
            progress = float(
                np.clip(
                    simulation_time_s
                    / total_duration_s,
                    0.0,
                    1.0,
                )
            )
        else:
            progress = self._status.task_progress

        force_control_active = bool(
            row.get(
                "force_control_active",
                0,
            )
        )

        controller_mode = (
            "HYBRID_XY_POSITION_Z_FORCE"
            if force_control_active
            else "POSE_TRACKING"
        )

        self._replace_status(
            task_progress=progress,
            controller_mode=controller_mode,
            tcp_position_m=physical_tcp,
            tracking_error_m=float(
                row.get(
                    "simulator_position_error_m",
                    self._status.tracking_error_m,
                )
            ),
            normal_force_n=float(
                row.get(
                    "pen_normal_force_n",
                    self._status.normal_force_n,
                )
            ),
            cycle_time_s=simulation_time_s,
        )

        callback = self._raw_telemetry_callback

        if callback is not None:
            callback(
                dict(row)
            )

    # --------------------------------------------------------
    # State / status helpers
    # --------------------------------------------------------

    def _transition(
        self,
        target: RobotState,
        **status_updates: Any,
    ) -> None:
        with self._lock:
            self._state_machine.transition(
                target
            )

        self._replace_status(
            **status_updates
        )

    def _replace_status(
        self,
        *,
        task_progress: float | None = None,
        controller_mode: str | None = None,
        tcp_position_m: np.ndarray | None = None,
        tracking_error_m: float | None = None,
        normal_force_n: float | None = None,
        cycle_time_s: float | None = None,
    ) -> None:
        with self._lock:
            previous = self._status

            self._status = RobotStatus(
                state=self.state,
                active_command=self._active_command,
                task_progress=(
                    previous.task_progress
                    if task_progress is None
                    else task_progress
                ),
                controller_mode=(
                    previous.controller_mode
                    if controller_mode is None
                    else controller_mode
                ),
                tcp_position_m=(
                    previous.tcp_position_m
                    if tcp_position_m is None
                    else tcp_position_m
                ),
                tracking_error_m=(
                    previous.tracking_error_m
                    if tracking_error_m is None
                    else tracking_error_m
                ),
                normal_force_n=(
                    previous.normal_force_n
                    if normal_force_n is None
                    else normal_force_n
                ),
                cycle_time_s=(
                    previous.cycle_time_s
                    if cycle_time_s is None
                    else cycle_time_s
                ),
                fault=self._fault,
                last_transition=(
                    self._state_machine.last_transition
                ),
            )

        self._publish()

    def _publish(self) -> None:
        callback = self._status_callback

        if callback is not None:
            callback(
                self.status
            )

    def _enter_fault(
        self,
        code: FaultCode,
        *,
        message: str,
        source: str,
        recoverable: bool = True,
    ) -> None:
        fault = RobotFault(
            code=code,
            message=(
                message
                if message.strip()
                else code.name
            ),
            source=source,
            recoverable=recoverable,
        )

        with self._lock:
            self._fault = fault
            self._active_command = None

            if (
                self.state
                is not RobotState.FAULT
            ):
                self._state_machine.transition(
                    RobotState.FAULT
                )

        self._replace_status(
            controller_mode="FAULT",
        )

    @staticmethod
    def _validation_fault(
        report: WritingValidationReport,
    ) -> tuple[FaultCode, str]:
        if not report.trajectory_finite:
            return (
                FaultCode.COMMAND_REJECTED,
                "Writing trajectory contains invalid values",
            )

        if not report.within_notebook_bounds:
            return (
                FaultCode.WORKSPACE_VIOLATION,
                "Writing trajectory exceeds notebook workspace",
            )

        if not report.sampled_reachability:
            return (
                FaultCode.IK_FAILURE,
                "Sampled writing waypoint is unreachable"
                + (
                    ""
                    if report.failed_waypoint_index
                    is None
                    else (
                        " at waypoint "
                        f"{report.failed_waypoint_index}"
                    )
                ),
            )

        if not report.sampled_collision_free:
            return (
                FaultCode.COLLISION_PRECHECK_FAILED,
                "Sampled collision preflight failed"
                + (
                    ""
                    if report.collision_waypoint_index
                    is None
                    else (
                        " at waypoint "
                        f"{report.collision_waypoint_index}"
                    )
                ),
            )

        return (
            FaultCode.COMMAND_REJECTED,
            "Writing validation failed",
        )

    @staticmethod
    def _classify_execution_error(
        error: Exception,
    ) -> FaultCode:
        message = str(error).lower()

        if (
            "safety limit" in message
            or "normal force exceeded" in message
        ):
            return FaultCode.SAFETY_INHIBIT

        if (
            "pybullet" in message
            or "client is unavailable" in message
            or "simulation" in message
        ):
            return FaultCode.SIMULATION_DISCONNECTED

        if "timeout" in message:
            return FaultCode.TRAJECTORY_TIMEOUT

        if "tracking" in message:
            return FaultCode.TRACKING_ERROR_EXCEEDED

        return FaultCode.CONTROLLER_FAILURE
