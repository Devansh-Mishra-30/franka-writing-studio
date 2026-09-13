import unittest

from force_control import (
    compute_normal_force_velocity,
)


class NormalForceControlTests(unittest.TestCase):
    def test_low_force_commands_downward_motion(self):
        result = compute_normal_force_velocity(
            measured_normal_force_n=0.5
        )

        self.assertGreater(
            result.force_error_n,
            0.0,
        )

        self.assertLess(
            result.commanded_world_z_velocity_m_s,
            0.0,
        )

    def test_high_force_commands_upward_motion(self):
        result = compute_normal_force_velocity(
            measured_normal_force_n=1.5
        )

        self.assertLess(
            result.force_error_n,
            0.0,
        )

        self.assertGreater(
            result.commanded_world_z_velocity_m_s,
            0.0,
        )

    def test_target_force_commands_zero_velocity(self):
        result = compute_normal_force_velocity(
            measured_normal_force_n=1.0
        )

        self.assertAlmostEqual(
            result.force_error_n,
            0.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.commanded_world_z_velocity_m_s,
            0.0,
            places=12,
        )

    def test_velocity_is_saturated(self):
        result = compute_normal_force_velocity(
            measured_normal_force_n=100.0
        )

        self.assertTrue(
            result.saturated
        )

        self.assertLessEqual(
            abs(
                result.commanded_world_z_velocity_m_s
            ),
            0.005,
        )

    def test_negative_force_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_normal_force_velocity(
                measured_normal_force_n=-0.1
            )


if __name__ == "__main__":
    unittest.main()
