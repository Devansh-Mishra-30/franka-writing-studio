"""Common multi-stroke drawing representation.

This module defines the input-source-independent drawing model
shared by SVG import and interactive touchpad/mouse drawing.

Both sources produce a DrawingDocument, which is then converted
to Cartesian pen-tip waypoints for the existing writing-plan,
validation, and execution pipeline.

Coordinate contract
-------------------
- Stroke points are 2D display-space coordinates: X increases to the
  right and Y increases down. They remain in source units (pixels for
  the drawing pad, SVG user units for SVG import).
- No physical units are implied at the DrawingDocument layer.
- drawing_to_waypoints() maps strokes into 3D pen-tip
  waypoints. The planar axes are mapped to the established notebook
  convention and Z explicitly encodes pen-up / pen-down in metres.
- Downstream fit_xy_path_to_surface() handles scaling and
  centering onto the physical notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ------------------------------------------------------------------
# Domain model
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Stroke:
    """One continuous pen-down path in 2D drawing space.

    Points are in the source coordinate system.
    A valid stroke contains at least 2 finite points.
    """

    points: np.ndarray  # shape (M, 2), dtype float

    def __post_init__(self) -> None:
        pts = np.array(
            self.points,
            dtype=float,
            copy=True,
            order="C",
        )

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                "Stroke points must have shape (M, 2); "
                f"got {pts.shape}"
            )

        if pts.shape[0] < 2:
            raise ValueError(
                "A stroke must contain at least 2 points"
            )

        if not np.all(np.isfinite(pts)):
            raise ValueError(
                "Stroke points contain non-finite values"
            )

        # A frozen dataclass containing a mutable ndarray is not truly
        # immutable. Own the storage and make it read-only so a drawing
        # cannot change while a worker thread is planning it.
        pts.setflags(write=False)

        object.__setattr__(
            self,
            "points",
            pts,
        )


@dataclass(frozen=True)
class DrawingDocument:
    """Input-source-independent multi-stroke drawing.

    This is the common representation produced by both the
    SVG adapter and the interactive drawing pad.
    """

    strokes: tuple[Stroke, ...]
    source: str  # "svg", "touchpad", "api"
    canvas_size: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strokes, tuple):
            object.__setattr__(
                self,
                "strokes",
                tuple(self.strokes),
            )

        if len(self.strokes) == 0:
            raise ValueError(
                "DrawingDocument must contain "
                "at least one stroke"
            )

        if not all(
            isinstance(stroke, Stroke)
            for stroke in self.strokes
        ):
            raise TypeError(
                "DrawingDocument strokes must all be Stroke instances"
            )

        if not self.source:
            raise ValueError(
                "DrawingDocument source must "
                "not be empty"
            )

        if self.canvas_size is not None:
            canvas_size = np.asarray(
                self.canvas_size,
                dtype=float,
            )

            if (
                canvas_size.shape != (2,)
                or not np.all(np.isfinite(canvas_size))
                or np.any(canvas_size <= 0.0)
            ):
                raise ValueError(
                    "canvas_size must contain two positive finite values"
                )

            object.__setattr__(
                self,
                "canvas_size",
                (
                    float(canvas_size[0]),
                    float(canvas_size[1]),
                ),
            )

    @property
    def total_points(self) -> int:
        """Total number of points across all strokes."""

        return sum(
            s.points.shape[0]
            for s in self.strokes
        )

    @property
    def stroke_count(self) -> int:
        return len(self.strokes)

    @property
    def bounding_box(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xy, max_xy) across all strokes."""

        all_points = np.vstack(
            [s.points for s in self.strokes]
        )

        return (
            np.min(all_points, axis=0),
            np.max(all_points, axis=0),
        )

    def metadata(self) -> dict[str, Any]:
        """Return drawing metadata for experiment logs."""

        return {
            "type": "drawing_document",
            "source": self.source,
            "stroke_count": self.stroke_count,
            "total_points": self.total_points,
            "canvas_size": self.canvas_size,
            "coordinate_system": "display_xy_right_down",
            "coordinate_units": "source_units",
        }


# ------------------------------------------------------------------
# Waypoint generation
# ------------------------------------------------------------------


def drawing_to_waypoints(
    document: DrawingDocument,
    *,
    surface_height_m: float,
    lift_clearance_m: float = 0.05,
) -> np.ndarray:
    """Convert a DrawingDocument to (N, 3) pen-tip waypoints.

    The returned array maps display-space coordinates into the
    established notebook-plane convention used by the v1 SVG path:

    - notebook X = -display Y
    - notebook Y = -display X

    Planar values are still source units at this boundary. The
    downstream fit step converts them to metres. Pen state is encoded
    in Z:

    - Z = surface_height_m for pen-on-paper (WRITE)
    - Z = surface_height_m + lift_clearance_m for pen-up

    Downstream processing (fit_xy_path_to_surface) handles
    scaling and centering onto the physical notebook.

    Stroke ordering and pen-lift transitions between strokes
    are explicit.  The sequence for N strokes is:

        [lift above stroke-1 start]
        [lower to stroke-1 start]
        [stroke-1 points at write height]
        [lift from stroke-1 end]
        [transfer to stroke-2 start]
        [lower to stroke-2 start]
        [stroke-2 points at write height]
        ...
        [lift from stroke-N end]
    """

    if not np.isfinite(surface_height_m):
        raise ValueError(
            "surface_height_m must be finite"
        )

    if (
        not np.isfinite(lift_clearance_m)
        or lift_clearance_m <= 0.0
    ):
        raise ValueError(
            "lift_clearance_m must be finite and positive"
        )

    write_z = surface_height_m
    lift_z = surface_height_m + lift_clearance_m

    waypoints: list[np.ndarray] = []

    def waypoint(
        point: np.ndarray,
        z_m: float,
    ) -> np.ndarray:
        return np.array(
            [
                -point[1],
                -point[0],
                z_m,
            ],
            dtype=float,
        )

    for stroke_index, stroke in enumerate(
        document.strokes
    ):
        pts = stroke.points  # (M, 2)

        if stroke_index > 0:
            # Lift from previous stroke end.
            prev_last = (
                document.strokes[
                    stroke_index - 1
                ].points[-1]
            )

            waypoints.append(
                waypoint(prev_last, lift_z)
            )

        # Pen-up above this stroke's start.
        waypoints.append(
            waypoint(pts[0], lift_z)
        )

        # Lower to write height.
        waypoints.append(
            waypoint(pts[0], write_z)
        )

        # Write stroke points (first point already placed).
        for point_index in range(
            1,
            pts.shape[0],
        ):
            waypoints.append(
                waypoint(
                    pts[point_index],
                    write_z,
                )
            )

    # Final lift after last stroke.
    last_pt = (
        document.strokes[-1].points[-1]
    )

    waypoints.append(
        waypoint(last_pt, lift_z)
    )

    return np.array(
        waypoints,
        dtype=float,
    )


# ------------------------------------------------------------------
# Point filtering
# ------------------------------------------------------------------


def filter_stroke_points(
    points: np.ndarray,
    *,
    min_distance: float = 1.0,
) -> np.ndarray:
    """Remove points closer than min_distance to the previous kept point.

    Always preserves the first and last points.
    Useful for cleaning raw touchpad/mouse input before
    constructing a Stroke.
    """

    pts = np.asarray(
        points,
        dtype=float,
    )

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            "points must have shape (N, 2)"
        )

    if not np.all(np.isfinite(pts)):
        raise ValueError(
            "points contain non-finite values"
        )

    if (
        not np.isfinite(min_distance)
        or min_distance < 0.0
    ):
        raise ValueError(
            "min_distance must be finite and nonnegative"
        )

    if pts.shape[0] <= 2:
        return pts.copy()

    filtered = [pts[0]]

    for i in range(1, pts.shape[0] - 1):
        dist = float(
            np.linalg.norm(
                pts[i] - filtered[-1]
            )
        )

        if dist >= min_distance:
            filtered.append(pts[i])

    # Always keep last point.
    filtered.append(pts[-1])

    return np.array(
        filtered,
        dtype=float,
    )

