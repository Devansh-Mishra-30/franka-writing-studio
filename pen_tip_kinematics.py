"""Kinematics for a fixed pen tip attached to fer_link8."""

from __future__ import annotations

import numpy as np


# Physical design parameter:
# pen tip is 120 mm along the tool's local +Z axis.
PEN_TIP_OFFSET_TOOL_M = np.array(
    [0.0, 0.0, 0.120],
    dtype=float,
)

# The legacy trajectory used fer_link8 z=0.635 m while
# the writing surface is at z=0.525 m. This corresponds
# to an implicit 110 mm pen/tool offset.
LEGACY_IMPLICIT_PEN_LENGTH_M = 0.110


def _skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix for a 3-vector."""

    x, y, z = np.asarray(
        vector,
        dtype=float,
    )

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def pen_tip_position(
    *,
    tool_position_m: np.ndarray,
    tool_rotation: np.ndarray,
    offset_tool_m: np.ndarray = PEN_TIP_OFFSET_TOOL_M,
) -> np.ndarray:
    """Return pen-tip position in world coordinates."""

    p_tool = np.asarray(
        tool_position_m,
        dtype=float,
    )

    R = np.asarray(
        tool_rotation,
        dtype=float,
    )

    offset = np.asarray(
        offset_tool_m,
        dtype=float,
    )

    if p_tool.shape != (3,):
        raise ValueError(
            "tool_position_m must have shape (3,)"
        )

    if R.shape != (3, 3):
        raise ValueError(
            "tool_rotation must have shape (3, 3)"
        )

    if offset.shape != (3,):
        raise ValueError(
            "offset_tool_m must have shape (3,)"
        )

    return (
        p_tool
        + R @ offset
    )


def pen_tip_jacobian(
    *,
    tool_jacobian: np.ndarray,
    tool_rotation: np.ndarray,
    offset_tool_m: np.ndarray = PEN_TIP_OFFSET_TOOL_M,
) -> np.ndarray:
    """Return the 6x7 world-aligned Jacobian at the pen tip."""

    J = np.asarray(
        tool_jacobian,
        dtype=float,
    )

    R = np.asarray(
        tool_rotation,
        dtype=float,
    )

    offset = np.asarray(
        offset_tool_m,
        dtype=float,
    )

    if J.shape != (6, 7):
        raise ValueError(
            "tool_jacobian must have shape (6, 7)"
        )

    if R.shape != (3, 3):
        raise ValueError(
            "tool_rotation must have shape (3, 3)"
        )

    if offset.shape != (3,):
        raise ValueError(
            "offset_tool_m must have shape (3,)"
        )

    offset_world = (
        R @ offset
    )

    J_linear = J[:3, :]
    J_angular = J[3:, :]

    # v_tip =
    # v_tool + omega x r
    #
    # omega x r = -skew(r) omega
    J_tip_linear = (
        J_linear
        - _skew(offset_world)
        @ J_angular
    )

    return np.vstack(
        (
            J_tip_linear,
            J_angular,
        )
    )


def legacy_positions_to_pen_tip(
    positions_m: np.ndarray,
) -> np.ndarray:
    """Convert legacy flange-height SVG positions to pen-tip targets."""

    positions = np.asarray(
        positions_m,
        dtype=float,
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions_m must have shape (N, 3)"
        )

    converted = positions.copy()

    # Legacy write height:
    # 0.635 m -> physical paper height 0.525 m.
    #
    # Legacy lift height:
    # 0.685 m -> pen-tip height 0.575 m.
    converted[:, 2] -= (
        LEGACY_IMPLICIT_PEN_LENGTH_M
    )

    return converted
