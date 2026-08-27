"""Interactive command center for the Franka writing-robot digital twin."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


DEFAULT_RUN = Path(
    "artifacts/phase2_full_plan_1x/samples.csv"
)

GUI_UPDATE_MS = 50
DISPLAY_STRIDE = 10


@dataclass(frozen=True)
class ReplayData:
    time_s: np.ndarray
    phase: np.ndarray

    desired_x_m: np.ndarray
    desired_y_m: np.ndarray

    physical_x_m: np.ndarray
    physical_y_m: np.ndarray

    force_n: np.ndarray
    desired_force_n: np.ndarray

    xy_error_mm: np.ndarray

    contact: np.ndarray
    write: np.ndarray

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "ReplayData":
        with path.open(
            newline="",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

        if not rows:
            raise ValueError(
                f"No samples found in {path}"
            )

        def floats(
            column: str,
        ) -> np.ndarray:
            return np.asarray(
                [
                    float(row[column])
                    for row in rows
                ],
                dtype=float,
            )

        phase = np.asarray(
            [
                row["writing_phase"]
                for row in rows
            ],
            dtype=object,
        )

        contact = np.asarray(
            [
                bool(
                    int(
                        row[
                            "pen_contact_active"
                        ]
                    )
                )
                for row in rows
            ],
            dtype=bool,
        )

        write = phase == "WRITE"

        return cls(
            time_s=floats(
                "simulation_time_s"
            ),
            phase=phase,
            desired_x_m=floats(
                "desired_pen_x_m"
            ),
            desired_y_m=floats(
                "desired_pen_y_m"
            ),
            physical_x_m=floats(
                "physical_pen_x_m"
            ),
            physical_y_m=floats(
                "physical_pen_y_m"
            ),
            force_n=floats(
                "pen_normal_force_n"
            ),
            desired_force_n=floats(
                "desired_pen_normal_force_n"
            ),
            xy_error_mm=(
                1000.0
                * floats(
                    "xy_tracking_error_m"
                )
            ),
            contact=contact,
            write=write,
        )

    @property
    def duration_s(self) -> float:
        return float(
            self.time_s[-1]
        )


class CommandCenter(QMainWindow):
    """Single-window robotics experiment dashboard."""

    def __init__(self) -> None:
        super().__init__()

        self.data: ReplayData | None = None
        self.run_path: Path | None = None

        self.current_index = 0
        self.playback_time_s = 0.0

        self.running = False
        self.last_wall_time_s = (
            time.perf_counter()
        )

        self.timer = QTimer(self)
        self.timer.setInterval(
            GUI_UPDATE_MS
        )
        self.timer.timeout.connect(
            self._update_replay
        )

        self.setWindowTitle(
            "Franka Writing Robot — "
            "Digital Twin Command Center"
        )

        self.resize(
            1500,
            900,
        )

        self._build_ui()

        if DEFAULT_RUN.is_file():
            self._load_run(
                DEFAULT_RUN
            )
        else:
            self.status_label.setText(
                "READY — load experiment CSV"
            )

        self.timer.start()

    def _build_ui(
        self,
    ) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(
            root
        )

        title = QLabel(
            "FRANKA WRITING ROBOT — "
            "DIGITAL TWIN COMMAND CENTER"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 20px; "
            "font-weight: bold; "
            "padding: 8px;"
        )

        root_layout.addWidget(
            title
        )

        main_splitter = QSplitter(
            Qt.Horizontal
        )

        root_layout.addWidget(
            main_splitter,
            1,
        )

        main_splitter.addWidget(
            self._build_controls()
        )

        visual_splitter = QSplitter(
            Qt.Vertical
        )

        visual_splitter.addWidget(
            self._build_writing_view()
        )

        plot_splitter = QSplitter(
            Qt.Horizontal
        )

        plot_splitter.addWidget(
            self._build_force_plot()
        )

        plot_splitter.addWidget(
            self._build_xy_plot()
        )

        visual_splitter.addWidget(
            plot_splitter
        )

        visual_splitter.setSizes(
            [520, 300]
        )

        main_splitter.addWidget(
            visual_splitter
        )

        main_splitter.setSizes(
            [320, 1180]
        )

        root_layout.addWidget(
            self._build_status_bar()
        )

    def _panel(
        self,
    ) -> QFrame:
        panel = QFrame()

        panel.setFrameShape(
            QFrame.StyledPanel
        )

        return panel

    def _build_controls(
        self,
    ) -> QWidget:
        panel = self._panel()

        layout = QVBoxLayout(
            panel
        )

        heading = QLabel(
            "EXPERIMENT CONTROLS"
        )

        heading.setStyleSheet(
            "font-size: 16px; "
            "font-weight: bold;"
        )

        layout.addWidget(
            heading
        )

        parameters = QGroupBox(
            "Replay / Experiment Parameters"
        )

        form = QFormLayout(
            parameters
        )

        self.force_value = QLabel(
            "1.00 N"
        )

        self.force_slider = QSlider(
            Qt.Horizontal
        )

        self.force_slider.setRange(
            20,
            200,
        )

        self.force_slider.setValue(
            100
        )

        # Historical replay must remain reproducible.
        # Live force control gets connected next.
        self.force_slider.setEnabled(
            False
        )

        force_row = QWidget()

        force_layout = QHBoxLayout(
            force_row
        )

        force_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        force_layout.addWidget(
            self.force_slider
        )

        force_layout.addWidget(
            self.force_value
        )

        form.addRow(
            "Recorded target force:",
            force_row,
        )

        self.speed_value = QLabel(
            "1.00×"
        )

        self.speed_slider = QSlider(
            Qt.Horizontal
        )

        self.speed_slider.setRange(
            25,
            400,
        )

        self.speed_slider.setValue(
            100
        )

        self.speed_slider.valueChanged.connect(
            self._speed_changed
        )

        speed_row = QWidget()

        speed_layout = QHBoxLayout(
            speed_row
        )

        speed_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        speed_layout.addWidget(
            self.speed_slider
        )

        speed_layout.addWidget(
            self.speed_value
        )

        form.addRow(
            "Replay speed:",
            speed_row,
        )

        layout.addWidget(
            parameters
        )

        run_box = QGroupBox(
            "Experiment"
        )

        run_layout = QGridLayout(
            run_box
        )

        self.run_button = QPushButton(
            "RUN"
        )

        self.pause_button = QPushButton(
            "PAUSE"
        )

        self.stop_button = QPushButton(
            "STOP"
        )

        self.reset_button = QPushButton(
            "RESET"
        )

        self.load_button = QPushButton(
            "LOAD RUN CSV"
        )

        self.record_checkbox = QCheckBox(
            "Record experiment"
        )

        self.run_button.clicked.connect(
            self._run
        )

        self.pause_button.clicked.connect(
            self._pause
        )

        self.stop_button.clicked.connect(
            self._stop
        )

        self.reset_button.clicked.connect(
            self._reset
        )

        self.load_button.clicked.connect(
            self._choose_run
        )

        self.record_checkbox.stateChanged.connect(
            self._record_changed
        )

        run_layout.addWidget(
            self.run_button,
            0,
            0,
        )

        run_layout.addWidget(
            self.pause_button,
            0,
            1,
        )

        run_layout.addWidget(
            self.stop_button,
            1,
            0,
        )

        run_layout.addWidget(
            self.reset_button,
            1,
            1,
        )

        run_layout.addWidget(
            self.load_button,
            2,
            0,
            1,
            2,
        )

        run_layout.addWidget(
            self.record_checkbox,
            3,
            0,
            1,
            2,
        )

        layout.addWidget(
            run_box
        )

        statistics = QGroupBox(
            "Live Statistics"
        )

        stats = QFormLayout(
            statistics
        )

        self.phase_label = QLabel(
            "IDLE"
        )

        self.contact_label = QLabel(
            "NO"
        )

        self.force_label = QLabel(
            "0.000 N"
        )

        self.xy_error_label = QLabel(
            "0.000 mm"
        )

        self.time_label = QLabel(
            "0.000 s"
        )

        self.contact_rate_label = QLabel(
            "0.00 %"
        )

        self.mean_force_label = QLabel(
            "0.000 N"
        )

        self.xy_rmse_label = QLabel(
            "0.000 mm"
        )

        stats.addRow(
            "Phase:",
            self.phase_label,
        )

        stats.addRow(
            "Contact:",
            self.contact_label,
        )

        stats.addRow(
            "Normal force:",
            self.force_label,
        )

        stats.addRow(
            "XY error:",
            self.xy_error_label,
        )

        stats.addRow(
            "Simulation time:",
            self.time_label,
        )

        stats.addRow(
            "WRITE contact rate:",
            self.contact_rate_label,
        )

        stats.addRow(
            "WRITE mean force:",
            self.mean_force_label,
        )

        stats.addRow(
            "WRITE XY RMSE:",
            self.xy_rmse_label,
        )

        layout.addWidget(
            statistics
        )

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            1000,
        )

        layout.addWidget(
            self.progress
        )

        layout.addStretch(
            1
        )

        return panel

    def _build_writing_view(
        self,
    ) -> QWidget:
        panel = self._panel()

        layout = QVBoxLayout(
            panel
        )

        heading = QLabel(
            "LIVE WRITING / SVG TRAJECTORY"
        )

        heading.setStyleSheet(
            "font-size: 15px; "
            "font-weight: bold;"
        )

        layout.addWidget(
            heading
        )

        self.writing_plot = (
            pg.PlotWidget()
        )

        self.writing_plot.setLabel(
            "bottom",
            "X",
            units="m",
        )

        self.writing_plot.setLabel(
            "left",
            "Y",
            units="m",
        )

        self.writing_plot.showGrid(
            x=True,
            y=True,
            alpha=0.25,
        )

        self.writing_plot.setAspectLocked(
            True
        )

        self.reference_curve = (
            self.writing_plot.plot(
                [],
                [],
                pen=pg.mkPen(
                    (140, 140, 140),
                    width=2,
                ),
            )
        )

        self.actual_curve = (
            self.writing_plot.plot(
                [],
                [],
                pen=pg.mkPen(
                    (0, 180, 255),
                    width=3,
                ),
            )
        )

        self.pen_marker = (
            self.writing_plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush=(
                    255,
                    80,
                    80,
                ),
            )
        )

        layout.addWidget(
            self.writing_plot
        )

        return panel

    def _build_force_plot(
        self,
    ) -> QWidget:
        panel = self._panel()

        layout = QVBoxLayout(
            panel
        )

        heading = QLabel(
            "NORMAL FORCE"
        )

        heading.setStyleSheet(
            "font-weight: bold;"
        )

        layout.addWidget(
            heading
        )

        self.force_plot = (
            pg.PlotWidget()
        )

        self.force_plot.setLabel(
            "bottom",
            "Time",
            units="s",
        )

        self.force_plot.setLabel(
            "left",
            "Force",
            units="N",
        )

        self.force_plot.showGrid(
            x=True,
            y=True,
            alpha=0.25,
        )

        self.force_curve = (
            self.force_plot.plot(
                [],
                [],
                pen=pg.mkPen(
                    (0, 180, 255),
                    width=2,
                ),
            )
        )

        self.force_target_curve = (
            self.force_plot.plot(
                [],
                [],
                pen=pg.mkPen(
                    (240, 210, 0),
                    width=2,
                ),
            )
        )

        layout.addWidget(
            self.force_plot
        )

        return panel

    def _build_xy_plot(
        self,
    ) -> QWidget:
        panel = self._panel()

        layout = QVBoxLayout(
            panel
        )

        heading = QLabel(
            "XY TRACKING ERROR"
        )

        heading.setStyleSheet(
            "font-weight: bold;"
        )

        layout.addWidget(
            heading
        )

        self.xy_plot = (
            pg.PlotWidget()
        )

        self.xy_plot.setLabel(
            "bottom",
            "Time",
            units="s",
        )

        self.xy_plot.setLabel(
            "left",
            "XY error",
            units="mm",
        )

        self.xy_plot.showGrid(
            x=True,
            y=True,
            alpha=0.25,
        )

        self.xy_curve = (
            self.xy_plot.plot(
                [],
                [],
                pen=pg.mkPen(
                    (255, 150, 0),
                    width=2,
                ),
            )
        )

        layout.addWidget(
            self.xy_plot
        )

        return panel

    def _build_status_bar(
        self,
    ) -> QWidget:
        panel = self._panel()

        layout = QHBoxLayout(
            panel
        )

        self.status_label = QLabel(
            "SYSTEM READY"
        )

        self.recording_label = QLabel(
            "Recording: OFF"
        )

        layout.addWidget(
            self.status_label
        )

        layout.addStretch(
            1
        )

        layout.addWidget(
            self.recording_label
        )

        return panel

    def _speed_changed(
        self,
        value: int,
    ) -> None:
        scale = (
            value
            / 100.0
        )

        self.speed_value.setText(
            f"{scale:.2f}×"
        )

    def _record_changed(
        self,
    ) -> None:
        enabled = (
            self.record_checkbox.isChecked()
        )

        self.recording_label.setText(
            "Recording: ON"
            if enabled
            else "Recording: OFF"
        )

    def _choose_run(
        self,
    ) -> None:
        filename, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Load writing experiment",
                "artifacts",
                "CSV files (*.csv)",
            )
        )

        if filename:
            self._load_run(
                Path(filename)
            )

    def _load_run(
        self,
        path: Path,
    ) -> None:
        self.running = False

        self.data = ReplayData.load(
            path
        )

        self.run_path = path

        recorded_force = float(
            np.mean(
                self.data.desired_force_n[
                    self.data.write
                ]
            )
        )

        self.force_value.setText(
            f"{recorded_force:.2f} N"
        )

        self.force_slider.setValue(
            int(
                round(
                    recorded_force
                    * 100.0
                )
            )
        )

        reference_x = np.where(
            self.data.write,
            self.data.desired_x_m,
            np.nan,
        )

        reference_y = np.where(
            self.data.write,
            self.data.desired_y_m,
            np.nan,
        )

        self.reference_curve.setData(
            reference_x[
                ::DISPLAY_STRIDE
            ],
            reference_y[
                ::DISPLAY_STRIDE
            ],
        )

        self.writing_plot.autoRange()

        self._reset()

        self.status_label.setText(
            f"LOADED: {path.parent.name}"
        )

    def _run(
        self,
    ) -> None:
        if self.data is None:
            self.status_label.setText(
                "NO EXPERIMENT LOADED"
            )
            return

        if (
            self.current_index
            >= len(
                self.data.time_s
            )
            - 1
        ):
            self._reset()

        self.running = True

        self.last_wall_time_s = (
            time.perf_counter()
        )

        self.status_label.setText(
            "REPLAY RUNNING"
        )

    def _pause(
        self,
    ) -> None:
        self.running = False

        self.status_label.setText(
            "REPLAY PAUSED"
        )

    def _stop(
        self,
    ) -> None:
        self.running = False

        self.status_label.setText(
            "REPLAY STOPPED"
        )

    def _reset(
        self,
    ) -> None:
        self.running = False

        self.current_index = 0
        self.playback_time_s = 0.0

        self.actual_curve.setData(
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

        self.pen_marker.setData(
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

        self.progress.setValue(
            0
        )

        self.status_label.setText(
            "READY"
        )

    def _update_replay(
        self,
    ) -> None:
        if (
            not self.running
            or self.data is None
        ):
            return

        now = time.perf_counter()

        wall_dt = (
            now
            - self.last_wall_time_s
        )

        self.last_wall_time_s = now

        replay_scale = (
            self.speed_slider.value()
            / 100.0
        )

        self.playback_time_s += (
            wall_dt
            * replay_scale
        )

        index = int(
            np.searchsorted(
                self.data.time_s,
                self.playback_time_s,
                side="right",
            )
        )

        index = min(
            index,
            len(
                self.data.time_s
            )
            - 1,
        )

        self.current_index = index

        self._refresh_display(
            index
        )

        if (
            index
            >= len(
                self.data.time_s
            )
            - 1
        ):
            self.running = False

            self.status_label.setText(
                "REPLAY COMPLETE"
            )

    def _refresh_display(
        self,
        index: int,
    ) -> None:
        assert self.data is not None

        end = index + 1

        ink_mask = (
            self.data.write[:end]
            & self.data.contact[:end]
        )

        ink_x = np.where(
            ink_mask,
            self.data.physical_x_m[
                :end
            ],
            np.nan,
        )

        ink_y = np.where(
            ink_mask,
            self.data.physical_y_m[
                :end
            ],
            np.nan,
        )

        self.actual_curve.setData(
            ink_x[
                ::DISPLAY_STRIDE
            ],
            ink_y[
                ::DISPLAY_STRIDE
            ],
        )

        self.pen_marker.setData(
            [
                self.data.physical_x_m[
                    index
                ]
            ],
            [
                self.data.physical_y_m[
                    index
                ]
            ],
        )

        plot_slice = slice(
            0,
            end,
            DISPLAY_STRIDE,
        )

        self.force_curve.setData(
            self.data.time_s[
                plot_slice
            ],
            self.data.force_n[
                plot_slice
            ],
        )

        self.force_target_curve.setData(
            self.data.time_s[
                plot_slice
            ],
            self.data.desired_force_n[
                plot_slice
            ],
        )

        self.xy_curve.setData(
            self.data.time_s[
                plot_slice
            ],
            self.data.xy_error_mm[
                plot_slice
            ],
        )

        self.phase_label.setText(
            str(
                self.data.phase[
                    index
                ]
            )
        )

        self.contact_label.setText(
            "YES"
            if self.data.contact[
                index
            ]
            else "NO"
        )

        self.force_label.setText(
            f"{self.data.force_n[index]:.3f} N"
        )

        self.xy_error_label.setText(
            f"{self.data.xy_error_mm[index]:.3f} mm"
        )

        self.time_label.setText(
            f"{self.data.time_s[index]:.3f} s"
        )

        write_mask = (
            self.data.write[
                :end
            ]
        )

        write_count = int(
            np.count_nonzero(
                write_mask
            )
        )

        if write_count > 0:
            contact_rate = (
                100.0
                * np.count_nonzero(
                    self.data.contact[
                        :end
                    ]
                    & write_mask
                )
                / write_count
            )

            write_force = (
                self.data.force_n[
                    :end
                ][
                    write_mask
                ]
            )

            write_xy = (
                self.data.xy_error_mm[
                    :end
                ][
                    write_mask
                ]
            )

            mean_force = float(
                np.mean(
                    write_force
                )
            )

            xy_rmse = float(
                np.sqrt(
                    np.mean(
                        write_xy
                        * write_xy
                    )
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

        progress = int(
            1000.0
            * index
            / max(
                1,
                len(
                    self.data.time_s
                )
                - 1,
            )
        )

        self.progress.setValue(
            progress
        )


def main() -> int:
    app = QApplication(
        sys.argv
    )

    window = CommandCenter()

    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
