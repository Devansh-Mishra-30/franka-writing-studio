import unittest

import numpy as np

from writing_plan import TimedWritingPlan


class WritingSpeedScaleTests(unittest.TestCase):
    def test_double_speed_halves_plan_duration(self):
        initial = np.array(
            [0.30, 0.0, 0.60],
            dtype=float,
        )

        positions = np.array(
            [
                [0.30, 0.0, 0.575],
                [0.30, 0.0, 0.525],
                [0.40, 0.0, 0.525],
                [0.40, 0.0, 0.575],
            ],
            dtype=float,
        )

        normal = TimedWritingPlan(
            initial_position_m=initial,
            svg_positions_m=positions,
            write_height_m=0.525,
            speed_scale=1.0,
        )

        fast = TimedWritingPlan(
            initial_position_m=initial,
            svg_positions_m=positions,
            write_height_m=0.525,
            speed_scale=2.0,
        )

        self.assertAlmostEqual(
            fast.total_duration_s,
            0.5 * normal.total_duration_s,
            places=12,
        )

    def test_nonpositive_speed_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            TimedWritingPlan(
                initial_position_m=np.array(
                    [0.3, 0.0, 0.6]
                ),
                svg_positions_m=np.array(
                    [
                        [0.3, 0.0, 0.575],
                        [0.3, 0.0, 0.525],
                    ]
                ),
                write_height_m=0.525,
                speed_scale=0.0,
            )


if __name__ == "__main__":
    unittest.main()
