"""Live Qt command center for the Franka writing robot."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
)

from ui.command_center import CommandCenter
from ui.experiment_worker import ExperimentWorker


DEFAULT_SVG = Path(
    "svg/portfolio_writing1 (8).svg"
)


class LiveCommandCenter(CommandCenter):
    """Command center connected to a live WritingExperiment."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "Franka Writing Robot — LIVE Digital Twin Command Center"
        )

        # Disable historical replay started by the parent class.
        self.running = False
        self.data = None

        self.svg_file = DEFAULT_SVG

        self.worker_thread: QThread | None = None
        self.worker: ExperimentWorker | None = None

        self._configure_live_controls()
        self._clear_live_data()

        self.status_label.setText(
            f"LIVE MODE — SVG: {self.svg_file.name}"
        )

    def _configure_live_controls(self) -> None:
        # Repurpose the existing replay controls.
        self.run_button.clicked.disconnect(self._run)
        self.load_button.clicked.disconnect(self._choose_run)
        self.reset_button.clicked.disconnect(self._reset)

        self.run_button.setText(
            "RUN LIVE"
        )

        self.load_button.setText(
            "LOAD SVG"
        )

        self.run_button.clicked.connect(
            self._run_live
        )

        self.load_button.clicked.connect(
            self._choose_svg
        )

        self.reset_button.clicked.connect(
            self._reset_live
        )

        # Pause/stop require cooperative cancellation.
        # Keep them disabled for the Phase 2.5 MVP.
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        # We already validated 1.0x.
        # Allow slower experiments, but do not encourage
        # the degraded >1.0x regimes during the MVP.
        self.speed_slider.setRange(
            50,
            100,
        )
        self.speed_slider.setValue(
            100,
        )

        self.speed_value.setText(
            "1.00×"
        )

        self.force_slider.setEnabled(
            False
        )

        self.record_checkbox.setText(
            "Record PyBullet video"
        )

        experiment_group = (
            self.record_checkbox.parentWidget()
        )

        experiment_layout = (
            experiment_group.layout()
        )

        self.show_robot_checkbox = QCheckBox(
            "Show PyBullet robot window"
        )

        self.show_robot_checkbox.setChecked(
            False
        )

        self.full_plan_checkbox = QCheckBox(
            "Run complete writing plan"
        )

        self.full_plan_checkbox.setChecked(
            False
        )

        experiment_layout.addWidget(
            self.show_robot_checkbox,
            4,
            0,
            1,
            2,
        )

        experiment_layout.addWidget(
            self.full_plan_checkbox,
            5,
            0,
            1,
            2,
        )

    def _clear_live_data(self) -> None:
        self.live_time_s: list[float] = []

        self.live_force_n: list[float] = []
        self.live_target_force_n: list[float] = []
        self.live_xy_error_mm: list[float] = []

        self.live_reference_x_m: list[float] = []
        self.live_reference_y_m: list[float] = []

        self.live_ink_x_m: list[float] = []
        self.live_ink_y_m: list[float] = []

        self.write_samples = 0
        self.write_contact_samples = 0

        self.write_force_sum_n = 0.0
        self.write_xy_squared_sum_mm2 = 0.0

        self.reference_curve.setData(
            [],
            [],
        )

        self.actual_curve.setData(
            [],
            [],
        )

        self.pen_marker.setData(
            [],
            [],
        )

        self.force_curve.setData(
            [],
            [],
        )

        self.force_target_curve.setData(
            [],
            [],
        )

        self.xy_curve.setData(
            [],
            [],
        )

        self.phase_label.setText(
            "IDLE"
        )

        self.contact_label.setText(
            "NO"
        )

        self.force_label.setText(
            "0.000 N"
        )

        self.xy_error_label.setText(
            "0.000 mm"
        )

        self.time_label.setText(
            "0.000 s"
        )

        self.contact_rate_label.setText(
            "0.00 %"
        )

        self.mean_force_label.setText(
            "0.000 N"
        )

        self.xy_rmse_label.setText(
            "0.000 mm"
        )

        self.progress.setRange(
            0,
            1000,
        )

        self.progress.setValue(
            0
        )

    def _choose_svg(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select writing SVG",
            "svg",
            "SVG files (*.svg)",
        )

        if not filename:
            return

        self.svg_file = Path(
            filename
        )

        self.status_label.setText(
            f"SVG LOADED — {self.svg_file.name}"
        )

    def _reset_live(self) -> None:
        if (
            self.worker_thread is not None
            and self.worker_thread.isRunning()
        ):
            self.status_label.setText(
                "Cannot reset while experiment is running"
            )
            return

        self._clear_live_data()

        self.status_label.setText(
            f"LIVE READY — {self.svg_file.name}"
        )

    def _run_live(self) -> None:
        if (
            self.worker_thread is not None
            and self.worker_thread.isRunning()
        ):
            self.status_label.setText(
                "Experiment already running"
            )
            return

        if not self.svg_file.is_file():
            self.status_label.setText(
                "Selected SVG does not exist"
            )
            return

        self._clear_live_data()

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = Path(
            "artifacts/live_runs"
        ) / timestamp

        speed_scale = (
            self.speed_slider.value()
            / 100.0
        )

        show_robot = (
            self.show_robot_checkbox.isChecked()
        )

        full_plan = (
            self.full_plan_checkbox.isChecked()
        )

        record_video = (
            self.record_checkbox.isChecked()
        )

        self.worker_thread = QThread(
            self
        )

        self.worker = ExperimentWorker(
            svg_file=self.svg_file,
            output_dir=output_dir,
            speed_scale=speed_scale,
            show_robot=show_robot,
            full_plan=full_plan,
            duration_s=5.5,
            record_video=record_video,
        )

        self.worker.moveToThread(
            self.worker_thread
        )

        self.worker_thread.started.connect(
            self.worker.run
        )

        self.worker.telemetry.connect(
            self._on_telemetry
        )

        self.worker.completed.connect(
            self._on_completed
        )

        self.worker.failed.connect(
            self._on_failed
        )

        self.worker.completed.connect(
            self.worker_thread.quit
        )

        self.worker.failed.connect(
            self.worker_thread.quit
        )

        self.worker_thread.finished.connect(
            self._thread_finished
        )

        self.run_button.setEnabled(
            False
        )

        self.load_button.setEnabled(
            False
        )

        self.speed_slider.setEnabled(
            False
        )

        self.show_robot_checkbox.setEnabled(
            False
        )

        self.full_plan_checkbox.setEnabled(
            False
        )

        self.progress.setRange(
            0,
            0,
        )

        mode = (
            "PYBULLET GUI"
            if show_robot
            else "HEADLESS"
        )

        self.status_label.setText(
            f"LIVE EXPERIMENT RUNNING — {mode}"
        )

        self.worker_thread.start()

    def _on_telemetry(
        self,
        row: dict,
    ) -> None:
        time_s = float(
            row["simulation_time_s"]
        )

        phase = str(
            row["writing_phase"]
        )

        force_n = float(
            row["pen_normal_force_n"]
        )

        target_force_n = float(
            row[
                "desired_pen_normal_force_n"
            ]
        )

        xy_error_mm = (
            1000.0
            * float(
                row[
                    "xy_tracking_error_m"
                ]
            )
        )

        contact = bool(
            int(
                row[
                    "pen_contact_active"
                ]
            )
        )

        desired_x = float(
            row["desired_pen_x_m"]
        )

        desired_y = float(
            row["desired_pen_y_m"]
        )

        physical_x = float(
            row["physical_pen_x_m"]
        )

        physical_y = float(
            row["physical_pen_y_m"]
        )

        is_write = (
            phase == "WRITE"
        )

        self.live_time_s.append(
            time_s
        )

        self.live_force_n.append(
            force_n
        )

        self.live_target_force_n.append(
            target_force_n
        )

        # Only show tracking error during WRITE.
        # This avoids misleading pen-up/transfer spikes.
        self.live_xy_error_mm.append(
            xy_error_mm
            if is_write
            else np.nan
        )

        self.live_reference_x_m.append(
            desired_x
            if is_write
            else np.nan
        )

        self.live_reference_y_m.append(
            desired_y
            if is_write
            else np.nan
        )

        ink_active = (
            is_write
            and contact
            and force_n > 0.0
        )

        self.live_ink_x_m.append(
            physical_x
            if ink_active
            else np.nan
        )

        self.live_ink_y_m.append(
            physical_y
            if ink_active
            else np.nan
        )

        self.reference_curve.setData(
            self.live_reference_x_m,
            self.live_reference_y_m,
        )

        self.actual_curve.setData(
            self.live_ink_x_m,
            self.live_ink_y_m,
        )

        self.pen_marker.setData(
            [physical_x],
            [physical_y],
        )

        self.force_curve.setData(
            self.live_time_s,
            self.live_force_n,
        )

        self.force_target_curve.setData(
            self.live_time_s,
            self.live_target_force_n,
        )

        self.xy_curve.setData(
            self.live_time_s,
            self.live_xy_error_mm,
        )

        self.phase_label.setText(
            phase
        )

        self.contact_label.setText(
            "YES"
            if contact
            else "NO"
        )

        self.force_label.setText(
            f"{force_n:.3f} N"
        )

        self.force_value.setText(
            f"{target_force_n:.2f} N"
        )

        self.xy_error_label.setText(
            f"{xy_error_mm:.3f} mm"
        )

        self.time_label.setText(
            f"{time_s:.3f} s"
        )

        if is_write:
            self.write_samples += 1

            if contact:
                self.write_contact_samples += 1

            self.write_force_sum_n += (
                force_n
            )

            self.write_xy_squared_sum_mm2 += (
                xy_error_mm
                * xy_error_mm
            )

            contact_rate = (
                100.0
                * self.write_contact_samples
                / self.write_samples
            )

            mean_force = (
                self.write_force_sum_n
                / self.write_samples
            )

            xy_rmse = float(
                np.sqrt(
                    self.write_xy_squared_sum_mm2
                    / self.write_samples
                )
            )

            self.contact_rate_label.setText(
                f"{contact_rate:.2f} %"
            )

            self.mean_force_label.setText(
                f"{mean_force:.3f} N"
            )

            self.xy_rmse_label.setText(
                f"{xy_rmse:.3f} mm"
            )

        self.status_label.setText(
            f"LIVE — {phase} — "
            f"t={time_s:.2f} s"
        )

    def _on_completed(
        self,
        result: object,
    ) -> None:
        self.progress.setRange(
            0,
            1000,
        )

        self.progress.setValue(
            1000,
        )

        samples_path = getattr(
            result,
            "samples_path",
            None,
        )

        self.status_label.setText(
            "LIVE EXPERIMENT COMPLETE"
            + (
                f" — {samples_path}"
                if samples_path is not None
                else ""
            )
        )

    def _on_failed(
        self,
        traceback_text: str,
    ) -> None:
        self.progress.setRange(
            0,
            1000,
        )

        self.progress.setValue(
            0,
        )

        self.status_label.setText(
            "LIVE EXPERIMENT FAILED"
        )

        print(
            "\nLIVE EXPERIMENT FAILURE\n"
            "=======================\n"
            + traceback_text,
            file=sys.stderr,
        )

    def _thread_finished(
        self,
    ) -> None:
        self.run_button.setEnabled(
            True
        )

        self.load_button.setEnabled(
            True
        )

        self.speed_slider.setEnabled(
            True
        )

        self.show_robot_checkbox.setEnabled(
            True
        )

        self.full_plan_checkbox.setEnabled(
            True
        )

        if self.worker is not None:
            self.worker.deleteLater()

        if self.worker_thread is not None:
            self.worker_thread.deleteLater()

        self.worker = None
        self.worker_thread = None

    def closeEvent(
        self,
        event,
    ) -> None:
        if (
            self.worker_thread is not None
            and self.worker_thread.isRunning()
        ):
            self.status_label.setText(
                "Experiment is running — "
                "wait for completion before closing"
            )

            event.ignore()
            return

        super().closeEvent(
            event
        )


def main() -> int:
    app = QApplication(
        sys.argv
    )

    window = LiveCommandCenter()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
