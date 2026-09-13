import unittest

import numpy as np

from differential_ik import (
    solve_pose_differential_ik,
)


def rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class PoseDifferentialIKTests(unittest.TestCase):
    def test_zero_pose_error_gives_zero_joint_velocity(self):
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)

        result = solve_pose_differential_ik(
            jacobian=J,
            current_position_m=np.zeros(3),
            desired_position_m=np.zeros(3),
            desired_linear_velocity_m_s=np.zeros(3),
            current_rotation=np.eye(3),
            desired_rotation=np.eye(3),
        )

        np.testing.assert_allclose(
            result.joint_velocity_rad_s,
            np.zeros(7),
            atol=1e-12,
        )

        np.testing.assert_allclose(
            result.orientation_error_rad,
            np.zeros(3),
            atol=1e-12,
        )

    def test_position_error_generates_linear_command(self):
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)

        result = solve_pose_differential_ik(
            jacobian=J,
            current_position_m=np.zeros(3),
            desired_position_m=np.array(
                [0.1, 0.0, 0.0]
            ),
            desired_linear_velocity_m_s=np.zeros(3),
            current_rotation=np.eye(3),
            desired_rotation=np.eye(3),
            position_gain_s_inv=4.0,
        )

        np.testing.assert_allclose(
            result.commanded_linear_velocity_m_s,
            np.array([0.4, 0.0, 0.0]),
            atol=1e-12,
        )

        self.assertGreater(
            result.joint_velocity_rad_s[0],
            0.0,
        )

    def test_orientation_error_generates_angular_command(self):
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)

        angle = 0.10

        result = solve_pose_differential_ik(
            jacobian=J,
            current_position_m=np.zeros(3),
            desired_position_m=np.zeros(3),
            desired_linear_velocity_m_s=np.zeros(3),
            current_rotation=np.eye(3),
            desired_rotation=rotation_z(angle),
            orientation_gain_s_inv=4.0,
        )

        np.testing.assert_allclose(
            result.orientation_error_rad,
            np.array([0.0, 0.0, angle]),
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result.commanded_angular_velocity_rad_s,
            np.array([0.0, 0.0, 0.4]),
            atol=1e-10,
        )

        self.assertGreater(
            result.joint_velocity_rad_s[5],
            0.0,
        )

    def test_desired_angular_velocity_is_feedforward(self):
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)

        desired_omega = np.array(
            [0.01, -0.02, 0.03]
        )

        result = solve_pose_differential_ik(
            jacobian=J,
            current_position_m=np.zeros(3),
            desired_position_m=np.zeros(3),
            desired_linear_velocity_m_s=np.zeros(3),
            current_rotation=np.eye(3),
            desired_rotation=np.eye(3),
            desired_angular_velocity_rad_s=desired_omega,
        )

        np.testing.assert_allclose(
            result.commanded_angular_velocity_rad_s,
            desired_omega,
            atol=1e-12,
        )

    def test_invalid_rotation_is_rejected(self):
        bad_rotation = np.eye(3)
        bad_rotation[0, 0] = 2.0

        with self.assertRaises(ValueError):
            solve_pose_differential_ik(
                jacobian=np.zeros((6, 7)),
                current_position_m=np.zeros(3),
                desired_position_m=np.zeros(3),
                desired_linear_velocity_m_s=np.zeros(3),
                current_rotation=bad_rotation,
                desired_rotation=np.eye(3),
            )

    def test_invalid_jacobian_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_pose_differential_ik(
                jacobian=np.zeros((3, 7)),
                current_position_m=np.zeros(3),
                desired_position_m=np.zeros(3),
                desired_linear_velocity_m_s=np.zeros(3),
                current_rotation=np.eye(3),
                desired_rotation=np.eye(3),
            )


if __name__ == "__main__":
    unittest.main()
