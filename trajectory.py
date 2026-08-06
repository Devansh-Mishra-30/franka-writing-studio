"""Trajectory generation for the writing robot."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from parse_svg_path import (
    parse_svg_for_paths,
    scale_coords_to_arena,
)


class SvgTrajectory:
    """Preprocessed, piecewise-constant legacy SVG trajectory.

    This class intentionally preserves the original trajectory behavior:
    one waypoint is selected every 0.1 seconds and the final waypoint is
    held after the path ends.

    Arc-length resampling and smooth time parameterization will be added
    in the trajectory-engineering phase.
    """

    def __init__(
        self,
        svg_path: Path | str,
        *,
        waypoint_interval_s: float = 0.1,
        dx_m: float = 0.4,
        dy_m: float = 0.4,
        x_min_m: float = 0.2,
        z_min_m: float = 0.635,
    ) -> None:
        self.svg_path = Path(svg_path)
        self.waypoint_interval_s = float(
            waypoint_interval_s
        )

        if not self.svg_path.is_file():
            raise FileNotFoundError(
                f"SVG file does not exist: {self.svg_path}"
            )

        if self.waypoint_interval_s <= 0.0:
            raise ValueError(
                "waypoint_interval_s must be greater than zero"
            )

        raw_coordinates = parse_svg_for_paths(
            str(self.svg_path)
        )

        if len(raw_coordinates) == 0:
            raise ValueError(
                f"No trajectory coordinates found in {self.svg_path}"
            )

        scaled_coordinates = scale_coords_to_arena(
            raw_coordinates,
            dx=dx_m,
            dy=dy_m,
            x_min=x_min_m,
            z_min=z_min_m,
        )

        self._positions_m = np.asarray(
            scaled_coordinates,
            dtype=float,
        )

        if self._positions_m.ndim != 2:
            raise ValueError(
                "scaled SVG coordinates must be two-dimensional"
            )

        if self._positions_m.shape[1] != 3:
            raise ValueError(
                "scaled SVG coordinates must contain x, y, z"
            )

        if not np.all(np.isfinite(self._positions_m)):
            raise ValueError(
                "scaled SVG coordinates contain invalid values"
            )

        # Preserve the legacy fixed downward orientation.
        self._orientation_rpy_rad = np.array(
            [-3.14, 0.0, 0.0],
            dtype=float,
        )

    @property
    def num_waypoints(self) -> int:
        return int(self._positions_m.shape[0])

    @property
    def final_waypoint_time_s(self) -> float:
        return (
            max(0, self.num_waypoints - 1)
            * self.waypoint_interval_s
        )

    def index_at(self, time_s: float) -> int:
        """Return the active waypoint index at simulation time."""

        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")

        if time_s < 0.0:
            raise ValueError("time_s must be nonnegative")

        index = int(
            math.floor(
                time_s / self.waypoint_interval_s
            )
        )

        return min(index, self.num_waypoints - 1)

    def sample(self, time_s: float) -> np.ndarray:
        """Return desired [x, y, z, roll, pitch, yaw]."""

        index = self.index_at(time_s)

        return np.concatenate(
            (
                self._positions_m[index].copy(),
                self._orientation_rpy_rad.copy(),
            )
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "legacy_piecewise_constant_svg",
            "svg_file": str(self.svg_path),
            "num_waypoints": self.num_waypoints,
            "waypoint_interval_s": (
                self.waypoint_interval_s
            ),
            "final_waypoint_time_s": (
                self.final_waypoint_time_s
            ),
        }


def circle_trajectory(time_s: float) -> np.ndarray:
    """Return the existing circular reference trajectory."""

    radius_m = 0.15
    angular_speed_rad_s = 0.50

    return np.array(
        [
            radius_m
            * math.cos(angular_speed_rad_s * time_s)
            + 0.4,
            radius_m
            * math.sin(angular_speed_rad_s * time_s),
            0.62,
            -3.14,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def point_trajectory(time_s: float) -> np.ndarray:
    """Return the existing fixed-point reference."""

    del time_s

    return np.array(
        [
            0.3,
            0.0,
            0.55,
            -3.14,
            0.0,
            0.0,
        ],
        dtype=float,
    )


@lru_cache(maxsize=16)
def _cached_svg_trajectory(
    svg_path: str,
) -> SvgTrajectory:
    """Cache preprocessed trajectories for legacy callers."""

    return SvgTrajectory(Path(svg_path))


def svg_trajectory(
    time_s: float,
    svg_path: Path | str,
) -> np.ndarray:
    """Backward-compatible SVG trajectory function."""

    return _cached_svg_trajectory(
        str(svg_path)
    ).sample(time_s)
