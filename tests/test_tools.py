"""Tests for reusable workcell tools."""

import numpy as np

from tools import PEN_TOOL


def test_pen_tool_definition() -> None:
    assert PEN_TOOL.name == "pen"
    assert PEN_TOOL.mass_kg == 0.010
    assert PEN_TOOL.urdf_path == "urdfs/pen_tool_pybullet.urdf"

    np.testing.assert_allclose(
        PEN_TOOL.tcp_offset_m,
        [0.0, 0.0, 0.120],
    )


def test_pen_tcp_is_immutable_copy() -> None:
    original = PEN_TOOL.tcp_offset_m.copy()

    original[2] = 999.0

    np.testing.assert_allclose(
        PEN_TOOL.tcp_offset_m,
        [0.0, 0.0, 0.120],
    )
