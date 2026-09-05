"""Qt worker for PLC-ready Writing Studio execution."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from experiment_config import ExperimentConfig
from experiments.writing_experiment import ExperimentStopped
from robot import (
    CommandRejected,
    RobotInterface,
)


class ExperimentWorker(QObject):
    """Run RobotInterface without blocking the Qt dashboard."""

    telemetry = Signal(dict)
    robot_status = Signal(object)

    completed = Signal(object)
    stopped = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        svg_file: Path,
        output_dir: Path,
        speed_scale: float,
        show_robot: bool,
        full_plan: bool,
        duration_s: float = 10.0,
        record_video: bool = False,
    ) -> None:
        super().__init__()

        self.svg_file = svg_file
        self.output_dir = output_dir
        self.speed_scale = speed_scale
        self.show_robot = show_robot
        self.full_plan = full_plan
        self.duration_s = duration_s
        self.record_video = record_video

        self.robot: RobotInterface | None = None

    @Slot()
    def run(self) -> None:
        try:
            mode = (
                "gui"
                if self.show_robot
                else "direct"
            )

            config = ExperimentConfig(
                mode=mode,
                duration_s=self.duration_s,
                timestep_s=0.001,
                svg_file=self.svg_file,
                output_dir=self.output_dir,
                realtime=self.show_robot,
                record_video=(
                    self.record_video
                    and self.show_robot
                ),
                speed_scale=self.speed_scale,
                full_plan=self.full_plan,
            ).validate()

            self.robot = RobotInterface(
                config,
                status_callback=(
                    self.robot_status.emit
                ),
                telemetry_callback=(
                    self.telemetry.emit
                ),
                telemetry_period_s=0.05,
            )

            # GUI, CLI, tests and future PLC all use
            # the same high-level application contract.
            self.robot.home()

            result = (
                self.robot.start_writing()
            )

            self.completed.emit(
                result
            )

        except ExperimentStopped:
            status = (
                self.robot.status
                if self.robot is not None
                else None
            )

            self.stopped.emit(
                status
            )

        except Exception:
            self.failed.emit(
                traceback.format_exc()
            )

    def request_stop(self) -> bool:
        """Thread-safe cooperative STOP request.

        This method is intentionally callable from the GUI thread.
        RobotInterface.stop() only updates protected application
        state and sets the experiment's threading.Event; it does not
        manipulate Qt widgets.
        """

        robot = self.robot

        if robot is None:
            return False

        try:
            robot.stop()

        except CommandRejected:
            return False

        return True
