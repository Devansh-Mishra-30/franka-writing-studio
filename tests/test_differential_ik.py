import unittest

import numpy as np

from differential_ik import (
    solve_position_differential_ik,
)


class DifferentialIKTests(unittest.TestCase):
    def test_zero_error_zero_velocity_gives_zero_joint_velocity(self):
        J = np.zeros((6, 7))
        J[:3, :3] = np.eye(3)

        result = solve_position_differential_ik(
            jacobian=J,
            current_position_m=np.array([0.3, 0.0, 0.6]),
            desired_position_m=np.array([0.3, 0.0, 0.6]),
            desired_velocity_m_s=np.zeros(3),
        )

        np.testing.assert_allclose(
            result.joint_velocity_rad_s,
            np.zeros(7),
            atol=1e-12,
        )

    def test_position_error_generates_correct_direction(self):
        J = np.zeros((6, 7))
        J[:3, :3] = np.eye(3)

        result = solve_position_differential_ik(
            jacobian=J,
            current_position_m=np.array([0.0, 0.0, 0.0]),
            desired_position_m=np.array([0.1, 0.0, 0.0]),
            desired_velocity_m_s=np.zeros(3),
            position_gain_s_inv=4.0,
            damping=0.02,
        )

        self.assertGreater(
            result.joint_velocity_rad_s[0],
            0.0,
        )

        self.assertAlmostEqual(
            result.joint_velocity_rad_s[1],
            0.0,
            delta=1e-12,
        )

        self.assertAlmostEqual(
            result.joint_velocity_rad_s[2],
            0.0,
            delta=1e-12,
        )

    def test_feedforward_velocity_is_used(self):
        J = np.zeros((6, 7))
        J[:3, :3] = np.eye(3)

        desired_velocity = np.array(
            [0.05, -0.02, 0.01]
        )

        result = solve_position_differential_ik(
            jacobian=J,
            current_position_m=np.zeros(3),
            desired_position_m=np.zeros(3),
            desired_velocity_m_s=desired_velocity,
            position_gain_s_inv=4.0,
            damping=0.02,
        )

        np.testing.assert_allclose(
            result.commanded_cartesian_velocity_m_s,
            desired_velocity,
            atol=1e-12,
        )

    def test_bad_jacobian_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_position_differential_ik(
                jacobian=np.zeros((3, 7)),
                current_position_m=np.zeros(3),
                desired_position_m=np.zeros(3),
                desired_velocity_m_s=np.zeros(3),
            )

    def test_nonpositive_damping_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_position_differential_ik(
                jacobian=np.zeros((6, 7)),
                current_position_m=np.zeros(3),
                desired_position_m=np.zeros(3),
                desired_velocity_m_s=np.zeros(3),
                damping=0.0,
            )


if __name__ == "__main__":
    unittest.main()
