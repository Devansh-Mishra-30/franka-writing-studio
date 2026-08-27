"""Qt worker for running writing experiments without blocking the dashboard."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from experiment_config import ExperimentConfig
from experiments.writing_experiment import WritingExperiment


class ExperimentWorker(QObject):
    telemetry = Signal(dict)
    completed = Signal(object)
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

            experiment = WritingExperiment(
                config,
                telemetry_callback=(
                    self.telemetry.emit
                ),
                telemetry_period_s=0.05,
            )

            result = experiment.run()

            self.completed.emit(result)

        except Exception:
            self.failed.emit(
                traceback.format_exc()
            )
