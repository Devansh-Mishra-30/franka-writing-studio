import unittest

import numpy as np

from writing_plan import (
    TimedWritingPlan,
    WritingPhase,
)


class TimedWritingPlanTests(unittest.TestCase):
    def setUp(self):
        self.initial_position = np.array(
            [0.20, 0.00, 0.75],
            dtype=float,
        )

        # Mimics the structure produced by our SVG parser:
        #
        # retracted point
        #       ↓ lower
        # writing point
        #       ↓ write
        # writing point
        #       ↓ lift
        # retracted point
        #       ↓ transfer
        # retracted point
        #       ↓ lower
        # writing point
        self.svg_positions = np.array(
            [
                [0.30, 0.00, 0.685],
                [0.30, 0.00, 0.635],
                [0.35, 0.02, 0.635],
                [0.35, 0.02, 0.685],
                [0.40, 0.05, 0.685],
                [0.40, 0.05, 0.635],
            ],
            dtype=float,
        )

        self.plan = TimedWritingPlan(
            initial_position_m=self.initial_position,
            svg_positions_m=self.svg_positions,
            write_height_m=0.635,
            approach_speed_m_s=0.10,
            write_speed_m_s=0.04,
            transfer_speed_m_s=0.10,
            vertical_speed_m_s=0.05,
            retreat_clearance_m=0.05,
            minimum_duration_s=0.10,
        )

    def test_expected_phase_sequence(self):
        phases = [
            segment.phase
            for segment in self.plan.segments
        ]

        self.assertEqual(
            phases,
            [
                WritingPhase.APPROACH,
                WritingPhase.LOWER,
                WritingPhase.WRITE,
                WritingPhase.LIFT,
                WritingPhase.TRANSFER,
                WritingPhase.LOWER,
                WritingPhase.RETREAT,
            ],
        )

    def test_approach_starts_at_actual_robot_position(self):
        first_segment = self.plan.segments[0]

        self.assertEqual(
            first_segment.phase,
            WritingPhase.APPROACH,
        )

        np.testing.assert_allclose(
            first_segment.start_m,
            self.initial_position,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            first_segment.end_m,
            self.svg_positions[0],
            atol=1e-12,
        )

    def test_segments_are_position_continuous(self):
        segments = self.plan.segments

        for index in range(len(segments) - 1):
            np.testing.assert_allclose(
                segments[index].end_m,
                segments[index + 1].start_m,
                atol=1e-12,
            )

    def test_sample_at_zero_starts_smoothly(self):
        sample = self.plan.sample(0.0)

        self.assertEqual(
            sample.phase,
            WritingPhase.APPROACH,
        )

        np.testing.assert_allclose(
            sample.position_m,
            self.initial_position,
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

    def test_each_segment_boundary_is_continuous(self):
        elapsed_s = 0.0
        segments = self.plan.segments

        for index in range(len(segments) - 1):
            elapsed_s += segments[index].duration_s

            sample = self.plan.sample(elapsed_s)

            np.testing.assert_allclose(
                sample.position_m,
                segments[index].end_m,
                atol=1e-10,
            )

            np.testing.assert_allclose(
                sample.velocity_m_s,
                np.zeros(3),
                atol=1e-10,
            )

            np.testing.assert_allclose(
                sample.acceleration_m_s2,
                np.zeros(3),
                atol=1e-10,
            )

    def test_retreat_moves_upward(self):
        final_segment = self.plan.segments[-1]

        self.assertEqual(
            final_segment.phase,
            WritingPhase.RETREAT,
        )

        self.assertAlmostEqual(
            final_segment.end_m[0],
            final_segment.start_m[0],
        )

        self.assertAlmostEqual(
            final_segment.end_m[1],
            final_segment.start_m[1],
        )

        self.assertAlmostEqual(
            final_segment.end_m[2]
            - final_segment.start_m[2],
            0.05,
        )

    def test_plan_has_positive_duration(self):
        self.assertGreater(
            self.plan.total_duration_s,
            0.0,
        )

        for segment in self.plan.segments:
            self.assertGreater(
                segment.duration_s,
                0.0,
            )

    def test_final_pose_is_held_after_completion(self):
        final_segment = self.plan.segments[-1]

        sample = self.plan.sample(
            self.plan.total_duration_s + 100.0
        )

        self.assertEqual(
            sample.phase,
            WritingPhase.RETREAT,
        )

        np.testing.assert_allclose(
            sample.position_m,
            final_segment.end_m,
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

    def test_negative_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan.sample(-0.001)


if __name__ == "__main__":
    unittest.main()
