import unittest

import numpy as np

from writing_plan import (
    WritingPhase,
    classify_segment,
    segment_duration,
)


class WritingPlanTests(unittest.TestCase):
    def test_lower_is_detected(self):
        phase = classify_segment(
            np.array([0.3, 0.0, 0.685]),
            np.array([0.3, 0.0, 0.635]),
            write_height_m=0.635,
        )

        self.assertEqual(
            phase,
            WritingPhase.LOWER,
        )

    def test_lift_is_detected(self):
        phase = classify_segment(
            np.array([0.3, 0.0, 0.635]),
            np.array([0.3, 0.0, 0.685]),
            write_height_m=0.635,
        )

        self.assertEqual(
            phase,
            WritingPhase.LIFT,
        )

    def test_same_height_is_write(self):
        phase = classify_segment(
            np.array([0.3, 0.0, 0.635]),
            np.array([0.31, 0.02, 0.635]),
            write_height_m=0.635,
        )

        self.assertEqual(
            phase,
            WritingPhase.WRITE,
        )

    def test_pen_up_horizontal_motion_is_transfer(self):
        phase = classify_segment(
            np.array([0.3, 0.0, 0.685]),
            np.array([0.4, 0.1, 0.685]),
            write_height_m=0.635,
        )

        self.assertEqual(
            phase,
            WritingPhase.TRANSFER,
        )

    def test_duration_uses_distance_and_speed(self):
        duration = segment_duration(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            speed_m_s=0.05,
            minimum_duration_s=0.1,
        )

        self.assertAlmostEqual(
            duration,
            2.0,
        )

    def test_duration_respects_minimum(self):
        duration = segment_duration(
            np.zeros(3),
            np.array([0.001, 0.0, 0.0]),
            speed_m_s=0.05,
            minimum_duration_s=0.1,
        )

        self.assertAlmostEqual(
            duration,
            0.1,
        )


if __name__ == "__main__":
    unittest.main()