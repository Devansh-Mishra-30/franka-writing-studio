"""Live PyBullet instrumentation for the writing robot.

Visualization only. No control commands or dynamics are modified here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class LiveDebugSample:
    simulation_time_s: float
    phase: str

    desired_position_m: np.ndarray
    actual_position_m: np.ndarray

    contact_active: bool
    normal_force_n: float
    desired_force_n: float

    xy_tracking_error_m: float
    force_correction_velocity_m_s: float


class LiveWritingDebugVisualizer:
    """Ink, XY/XZ graphs, force graph, and live HUD."""

    def __init__(
        self,
        *,
        client_id: int,
        enabled: bool,
        surface_height_m: float,
        update_period_s: float = 0.05,
    ) -> None:
        self.client_id = int(client_id)
        self.enabled = bool(enabled)

        self.surface_height_m = float(
            surface_height_m
        )

        self.update_period_s = float(
            update_period_s
        )

        if self.update_period_s <= 0.0:
            raise ValueError(
                "update_period_s must be positive"
            )

        self._last_update_time_s = -np.inf

        self._previous_ink_position_m = None
        self._previous_xy_graph_point = None
        self._previous_xz_graph_point = None
        self._previous_force_graph_point = None

        self._hud_id = -1

        self._time_max_s = 1.0
        self._force_max_n = 1.5

        self._xy_min = np.zeros(2)
        self._xy_max = np.ones(2)

        self._xz_min = np.zeros(2)
        self._xz_max = np.ones(2)

        # Three vertical graph panels behind the writing surface.
        self._panel_y_m = 0.32
        self._panel_z_m = 0.69
        self._panel_width_m = 0.18
        self._panel_height_m = 0.13
        self._panel_gap_m = 0.035

        self._panel_x0 = (
            0.13,
            0.13
            + self._panel_width_m
            + self._panel_gap_m,
            0.13
            + 2.0
            * (
                self._panel_width_m
                + self._panel_gap_m
            ),
        )

        self._ink_z_offset_m = 0.0015

    def _panel_point(
        self,
        panel_index: int,
        u: float,
        v: float,
    ) -> list[float]:
        return [
            (
                self._panel_x0[panel_index]
                + float(u)
                * self._panel_width_m
            ),
            self._panel_y_m,
            (
                self._panel_z_m
                + float(v)
                * self._panel_height_m
            ),
        ]

    @staticmethod
    def _normalized(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        span = maximum - minimum

        if span <= 1e-12:
            return 0.5

        return float(
            np.clip(
                (value - minimum) / span,
                0.0,
                1.0,
            )
        )

    def _draw_line(
        self,
        start,
        end,
        color,
        *,
        width: float = 1.0,
        lifetime: float = 0.0,
    ) -> int:
        return p.addUserDebugLine(
            lineFromXYZ=start,
            lineToXYZ=end,
            lineColorRGB=color,
            lineWidth=width,
            lifeTime=lifetime,
            physicsClientId=self.client_id,
        )

    def _draw_panel(
        self,
        panel_index: int,
        title: str,
    ) -> None:
        corners = [
            self._panel_point(
                panel_index,
                0.0,
                0.0,
            ),
            self._panel_point(
                panel_index,
                1.0,
                0.0,
            ),
            self._panel_point(
                panel_index,
                1.0,
                1.0,
            ),
            self._panel_point(
                panel_index,
                0.0,
                1.0,
            ),
        ]

        for a, b in zip(
            corners,
            corners[1:] + corners[:1],
        ):
            self._draw_line(
                a,
                b,
                [0.15, 0.15, 0.15],
                width=1.5,
            )

        # Mid-grid lines.
        self._draw_line(
            self._panel_point(
                panel_index,
                0.0,
                0.5,
            ),
            self._panel_point(
                panel_index,
                1.0,
                0.5,
            ),
            [0.7, 0.7, 0.7],
            width=0.5,
        )

        self._draw_line(
            self._panel_point(
                panel_index,
                0.5,
                0.0,
            ),
            self._panel_point(
                panel_index,
                0.5,
                1.0,
            ),
            [0.7, 0.7, 0.7],
            width=0.5,
        )

        p.addUserDebugText(
            text=title,
            textPosition=(
                self._panel_point(
                    panel_index,
                    0.02,
                    1.07,
                )
            ),
            textColorRGB=[
                0.05,
                0.05,
                0.05,
            ],
            textSize=1.0,
            lifeTime=0.0,
            physicsClientId=self.client_id,
        )

    def draw_reference_path(
        self,
        positions_m: np.ndarray,
        *,
        total_duration_s: float,
        desired_force_n: float,
    ) -> None:
        """Initialize graphs and draw desired references."""

        if not self.enabled:
            return

        positions = np.asarray(
            positions_m,
            dtype=float,
        )

        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] < 2
        ):
            raise ValueError(
                "positions_m must have shape (N, 3), N >= 2"
            )

        self._time_max_s = max(
            float(total_duration_s),
            1e-6,
        )

        self._force_max_n = max(
            1.5 * float(desired_force_n),
            0.5,
        )

        # ------------------------------
        # Graph ranges
        # ------------------------------

        xy_min = np.min(
            positions[:, :2],
            axis=0,
        )
        xy_max = np.max(
            positions[:, :2],
            axis=0,
        )

        xy_padding = np.maximum(
            0.05 * (xy_max - xy_min),
            np.array(
                [0.005, 0.005],
                dtype=float,
            ),
        )

        self._xy_min = (
            xy_min - xy_padding
        )
        self._xy_max = (
            xy_max + xy_padding
        )

        xz = positions[:, [0, 2]]

        xz_min = np.min(
            xz,
            axis=0,
        )
        xz_max = np.max(
            xz,
            axis=0,
        )

        xz_padding = np.maximum(
            0.08 * (xz_max - xz_min),
            np.array(
                [0.005, 0.005],
                dtype=float,
            ),
        )

        self._xz_min = (
            xz_min - xz_padding
        )
        self._xz_max = (
            xz_max + xz_padding
        )

        # Plotting is handled by the dedicated Qt telemetry window.
        # PyBullet displays only the robot, workcell and ink trail.

    def update(
        self,
        sample: LiveDebugSample,
    ) -> None:
        if not self.enabled:
            return

        if (
            sample.simulation_time_s
            - self._last_update_time_s
            < self.update_period_s
        ):
            return

        self._last_update_time_s = (
            sample.simulation_time_s
        )

        actual = np.asarray(
            sample.actual_position_m,
            dtype=float,
        ).reshape(3)

        # ------------------------------
        # Actual ink on the paper
        # ------------------------------

        writing_contact = (
            sample.phase == "WRITE"
            and sample.contact_active
            and sample.normal_force_n > 0.0
        )

        if writing_contact:
            if (
                self._previous_ink_position_m
                is not None
            ):
                previous = (
                    self._previous_ink_position_m
                )

                self._draw_line(
                    [
                        float(previous[0]),
                        float(previous[1]),
                        (
                            self.surface_height_m
                            + self._ink_z_offset_m
                        ),
                    ],
                    [
                        float(actual[0]),
                        float(actual[1]),
                        (
                            self.surface_height_m
                            + self._ink_z_offset_m
                        ),
                    ],
                    [0.02, 0.02, 0.02],
                    width=3.0,
                )

            self._previous_ink_position_m = (
                actual.copy()
            )

        else:
            self._previous_ink_position_m = None
