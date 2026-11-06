"""Adapter converting SVG files to DrawingDocument.

This preserves backward compatibility: existing SVG files produce
equivalent waypoints through the common DrawingDocument pathway
as they did through the legacy SvgTrajectory pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drawing import DrawingDocument, Stroke
from parse_svg_path import parse_svg_for_paths


# SVG pen-up is encoded as Z > 0 (typically 0.05).
_PEN_UP_Z_THRESHOLD = 0.01


def svg_to_drawing(
    svg_path: Path | str,
) -> DrawingDocument:
    """Parse an SVG file into a DrawingDocument.

    The legacy SVG parser produces (N, 3) coordinates with
    Z = 0 for pen-on-paper and Z = 0.05 for pen-up.

    This function splits on pen-up transitions to recover
    individual strokes and discards the Z column (pen state
    is represented structurally as separate Stroke objects).
    """

    svg_path = Path(svg_path)

    if not svg_path.is_file():
        raise FileNotFoundError(
            f"SVG file does not exist: {svg_path}"
        )

    raw = parse_svg_for_paths(
        str(svg_path)
    )

    if len(raw) == 0:
        raise ValueError(
            f"No path data found in {svg_path}"
        )

    strokes = _split_into_strokes(raw)

    if not strokes:
        raise ValueError(
            f"SVG file {svg_path} contains "
            "no valid strokes"
        )

    return DrawingDocument(
        strokes=tuple(strokes),
        source="svg",
    )


def _split_into_strokes(
    raw: np.ndarray,
) -> list[Stroke]:
    """Split raw SVG coordinates into strokes at pen-up transitions."""

    strokes: list[Stroke] = []
    current_points: list[np.ndarray] = []

    for i in range(len(raw)):
        z = float(raw[i, 2])
        xy = raw[i, :2].copy()

        if z > _PEN_UP_Z_THRESHOLD:
            # Pen is up: flush current stroke.
            if len(current_points) >= 2:
                strokes.append(
                    Stroke(
                        points=np.array(
                            current_points,
                            dtype=float,
                        )
                    )
                )

            current_points = []
        else:
            # Pen is down: accumulate.
            current_points.append(xy)

    # Flush final stroke.
    if len(current_points) >= 2:
        strokes.append(
            Stroke(
                points=np.array(
                    current_points,
                    dtype=float,
                )
            )
        )

    return strokes

