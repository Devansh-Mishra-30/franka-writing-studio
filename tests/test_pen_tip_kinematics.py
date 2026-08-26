import unittest

import numpy as np

from franka_mechanics import FrankaMechanics
from pen_tip_kinematics import (
    LEGACY_IMPLICIT_PEN_LENGTH_M,
    PEN_TIP_OFFSET_TOOL_M,
    legacy_positions_to_pen_tip,
    pen_tip_jacobian,
    pen_tip_position,
)


Q_HOME = np.array(
    [
        0.23081834,
        -0.69476370,
        -0.12918779,
        -2.02518342,
        -0.08491880,
        1.33535219,
        0.11158610,
    ],
    dtype=float,
)


class PenTipKinematicsTests(unittest.TestCase):
    def test_pen_tip_is_fixed_offset_from_tool(self):
        mechanics = FrankaMechanics()

        tool_position = np.asarray(
            mechanics.solve_fk(Q_HOME)[:3]
        )

        R = mechanics.get_tool_rotation(
            Q_HOME
        )

        tip = pen_tip_position(
            tool_position_m=tool_position,
            tool_rotation=R,
        )

        expected = (
            tool_position
            + R @ PEN_TIP_OFFSET_TOOL_M
        )

        np.testing.assert_allclose(
            tip,
            expected,
            atol=1e-12,
        )

    def test_pen_tip_jacobian_matches_finite_difference(self):
        mechanics = FrankaMechanics()

        dq = np.array(
            [
                0.10,
                -0.08,
                0.06,
                -0.04,
                0.05,
                -0.03,
                0.02,
            ],
            dtype=float,
        )

        dt = 1e-7

        q0 = Q_HOME.copy()
        q1 = q0 + dt * dq

        p0_tool = np.asarray(
            mechanics.solve_fk(q0)[:3]
        )

        p1_tool = np.asarray(
            mechanics.solve_fk(q1)[:3]
        )

        R0 = mechanics.get_tool_rotation(q0)
        R1 = mechanics.get_tool_rotation(q1)

        p0_tip = pen_tip_position(
            tool_position_m=p0_tool,
            tool_rotation=R0,
        )

        p1_tip = pen_tip_position(
            tool_position_m=p1_tool,
            tool_rotation=R1,
        )

        measured_velocity = (
            p1_tip - p0_tip
        ) / dt

        J_tool = mechanics.get_jacobian(q0)

        J_tip = pen_tip_jacobian(
            tool_jacobian=J_tool,
            tool_rotation=R0,
        )

        predicted_velocity = (
            J_tip[:3, :] @ dq
        )

        np.testing.assert_allclose(
            predicted_velocity,
            measured_velocity,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_legacy_write_height_maps_to_table(self):
        legacy = np.array(
            [
                [0.3, 0.1, 0.635],
                [0.3, 0.1, 0.685],
            ]
        )

        converted = (
            legacy_positions_to_pen_tip(
                legacy
            )
        )

        self.assertAlmostEqual(
            converted[0, 2],
            0.525,
            places=12,
        )

        self.assertAlmostEqual(
            converted[1, 2],
            0.575,
            places=12,
        )

        self.assertAlmostEqual(
            LEGACY_IMPLICIT_PEN_LENGTH_M,
            0.110,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
