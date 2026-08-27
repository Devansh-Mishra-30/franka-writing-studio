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

        # ------------------------------
        # Panels
        # ------------------------------

        self._draw_panel(
            0,
            "XY PATH   gray=desired blue=actual",
        )
        self._draw_panel(
            1,
            "XZ PROFILE   gray=desired blue=actual",
        )
        self._draw_panel(
            2,
            "NORMAL FORCE   orange=actual",
        )

        # ------------------------------
        # Desired XY reference
        # Only in graph panel.
        # ------------------------------

        for start, end in zip(
            positions[:-1],
            positions[1:],
        ):
            a = self._panel_point(
                0,
                self._normalized(
                    start[0],
                    self._xy_min[0],
                    self._xy_max[0],
                ),
                self._normalized(
                    start[1],
                    self._xy_min[1],
                    self._xy_max[1],
                ),
            )

            b = self._panel_point(
                0,
                self._normalized(
                    end[0],
                    self._xy_min[0],
                    self._xy_max[0],
                ),
                self._normalized(
                    end[1],
                    self._xy_min[1],
                    self._xy_max[1],
                ),
            )

            self._draw_line(
                a,
                b,
                [0.55, 0.55, 0.55],
                width=1.0,
            )

        # ------------------------------
        # Desired XZ reference
        # ------------------------------

        for start, end in zip(
            positions[:-1],
            positions[1:],
        ):
            a = self._panel_point(
                1,
                self._normalized(
                    start[0],
                    self._xz_min[0],
                    self._xz_max[0],
                ),
                self._normalized(
                    start[2],
                    self._xz_min[1],
                    self._xz_max[1],
                ),
            )

            b = self._panel_point(
                1,
                self._normalized(
                    end[0],
                    self._xz_min[0],
                    self._xz_max[0],
                ),
                self._normalized(
                    end[2],
                    self._xz_min[1],
                    self._xz_max[1],
                ),
            )

            self._draw_line(
                a,
                b,
                [0.55, 0.55, 0.55],
                width=1.0,
            )

        # Paper plane in XZ graph.
        paper_v = self._normalized(
            self.surface_height_m,
            self._xz_min[1],
            self._xz_max[1],
        )

        self._draw_line(
            self._panel_point(
                1,
                0.0,
                paper_v,
            ),
            self._panel_point(
                1,
                1.0,
                paper_v,
            ),
            [0.75, 0.2, 0.2],
            width=1.5,
        )

        # Desired force line.
        force_v = self._normalized(
            desired_force_n,
            0.0,
            self._force_max_n,
        )

        self._draw_line(
            self._panel_point(
                2,
                0.0,
                force_v,
            ),
            self._panel_point(
                2,
                1.0,
                force_v,
            ),
            [0.2, 0.55, 0.2],
            width=1.5,
        )

        p.addUserDebugText(
            text=f"target = {desired_force_n:.2f} N",
            textPosition=self._panel_point(
                2,
                0.03,
                force_v + 0.04,
            ),
            textColorRGB=[
                0.1,
                0.4,
                0.1,
            ],
            textSize=0.8,
            lifeTime=0.0,
            physicsClientId=self.client_id,
        )

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
        # Actual XY graph
        # ------------------------------

        xy_point = self._panel_point(
            0,
            self._normalized(
                actual[0],
                self._xy_min[0],
                self._xy_max[0],
            ),
            self._normalized(
                actual[1],
                self._xy_min[1],
                self._xy_max[1],
            ),
        )

        if (
            self._previous_xy_graph_point
            is not None
        ):
            self._draw_line(
                self._previous_xy_graph_point,
                xy_point,
                [0.1, 0.3, 0.95],
                width=2.0,
            )

        self._previous_xy_graph_point = (
            xy_point
        )

        # ------------------------------
        # Actual XZ graph
        # ------------------------------

        xz_point = self._panel_point(
            1,
            self._normalized(
                actual[0],
                self._xz_min[0],
                self._xz_max[0],
            ),
            self._normalized(
                actual[2],
                self._xz_min[1],
                self._xz_max[1],
            ),
        )

        if (
            self._previous_xz_graph_point
            is not None
        ):
            self._draw_line(
                self._previous_xz_graph_point,
                xz_point,
                [0.1, 0.3, 0.95],
                width=2.0,
            )

        self._previous_xz_graph_point = (
            xz_point
        )

        # ------------------------------
        # Force-vs-time graph
        # ------------------------------

        force_point = self._panel_point(
            2,
            self._normalized(
                sample.simulation_time_s,
                0.0,
                self._time_max_s,
            ),
            self._normalized(
                sample.normal_force_n,
                0.0,
                self._force_max_n,
            ),
        )

        if (
            self._previous_force_graph_point
            is not None
        ):
            self._draw_line(
                self._previous_force_graph_point,
                force_point,
                [0.9, 0.45, 0.1],
                width=2.0,
            )

        self._previous_force_graph_point = (
            force_point
        )

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

        # ------------------------------
        # HUD
        # ------------------------------

        contact = (
            "YES"
            if sample.contact_active
            else "NO"
        )

        hud = (
            "WRITING ROBOT\n"
            f"phase: {sample.phase}\n"
            f"contact: {contact}\n"
            f"force: {sample.normal_force_n:.3f} / "
            f"{sample.desired_force_n:.3f} N\n"
            f"XY error: "
            f"{1000.0 * sample.xy_tracking_error_m:.3f} mm\n"
            f"force vz: "
            f"{1000.0 * sample.force_correction_velocity_m_s:+.3f} mm/s\n"
            f"time: {sample.simulation_time_s:.2f} s"
        )

        self._hud_id = p.addUserDebugText(
            text=hud,
            textPosition=[
                0.16,
                -0.32,
                0.83,
            ],
            textColorRGB=[
                0.05,
                0.05,
                0.05,
            ],
            textSize=1.05,
            lifeTime=0.0,
            replaceItemUniqueId=self._hud_id,
            physicsClientId=self.client_id,
        )
