import unittest
from pathlib import Path

import numpy as np

from trajectory import SvgTrajectory


class SvgTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = SvgTrajectory(
            Path(
                "svg/portfolio_writing1 (8).svg"
            )
        )

    def test_trajectory_has_valid_waypoints(self):
        self.assertGreater(
            self.trajectory.num_waypoints,
            0,
        )

        sample = self.trajectory.sample(0.0)

        self.assertEqual(sample.shape, (6,))
        self.assertTrue(np.all(np.isfinite(sample)))

    def test_final_waypoint_is_held(self):
        final_sample = self.trajectory.sample(
            self.trajectory.final_waypoint_time_s
        )

        later_sample = self.trajectory.sample(
            self.trajectory.final_waypoint_time_s
            + 100.0
        )

        np.testing.assert_allclose(
            final_sample,
            later_sample,
        )

    def test_metadata_identifies_svg_as_waypoint_source(self):
        metadata = self.trajectory.metadata()

        self.assertEqual(
            metadata["type"],
            "svg_cartesian_waypoint_source",
        )
        self.assertEqual(
            metadata["num_waypoints"],
            self.trajectory.num_waypoints,
        )

    def test_negative_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.trajectory.sample(-0.1)


if __name__ == "__main__":
    unittest.main()
