import unittest

import numpy as np

from motion_profiles import minimum_jerk_segment


class MinimumJerkTests(unittest.TestCase):
    def test_start_boundary(self):
        start = np.array([0.1, 0.2, 0.3])
        end = np.array([0.4, -0.1, 0.6])

        sample = minimum_jerk_segment(
            start,
            end,
            time_s=0.0,
            duration_s=2.0,
        )

        np.testing.assert_allclose(
            sample.position_m,
            start,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sample.velocity_m_s,
            np.zeros(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sample.acceleration_m_s2,
            np.zeros(3),
            atol=1e-12,
        )

    def test_end_boundary(self):
        start = np.array([0.1, 0.2, 0.3])
        end = np.array([0.4, -0.1, 0.6])

        sample = minimum_jerk_segment(
            start,
            end,
            time_s=2.0,
            duration_s=2.0,
        )

        np.testing.assert_allclose(
            sample.position_m,
            end,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sample.velocity_m_s,
            np.zeros(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sample.acceleration_m_s2,
            np.zeros(3),
            atol=1e-12,
        )

    def test_halfway_position(self):
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 2.0, 3.0])

        sample = minimum_jerk_segment(
            start,
            end,
            time_s=1.0,
            duration_s=2.0,
        )

        np.testing.assert_allclose(
            sample.position_m,
            0.5 * end,
            atol=1e-12,
        )

    def test_invalid_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            minimum_jerk_segment(
                np.zeros(3),
                np.ones(3),
                time_s=0.0,
                duration_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
