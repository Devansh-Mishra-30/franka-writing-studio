"""Pen tool used by the interactive writing workcell."""

from __future__ import annotations

import numpy as np

from .base import ToolSpec


PEN_TOOL = ToolSpec(
    name="pen",
    tcp_offset_m=np.array(
        [0.0, 0.0, 0.120],
        dtype=float,
    ),
    mass_kg=0.010,
    urdf_path="urdfs/pen_tool_pybullet.urdf",
)
