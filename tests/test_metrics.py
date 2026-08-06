import math
import unittest

from metrics import (
    build_experiment_summary,
    summarize_values,
)


class MetricsTests(unittest.TestCase):
    def test_value_summary(self):
        summary = summarize_values(
            [1.0, 2.0, 3.0, 4.0],
            name="errors",
        )

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(
            summary["rmse"],
            math.sqrt(7.5),
        )
        self.assertAlmostEqual(summary["maximum"], 4.0)

    def test_experiment_summary_computes_rtf(self):
        summary = build_experiment_summary(
            position_errors_m=[0.1, 0.2],
            joint_errors_rad=[0.3, 0.4],
            loop_durations_s=[0.01, 0.01],
            ik_durations_s=[0.005, 0.005],
            timestep_s=0.1,
            wall_duration_s=0.4,
        )

        execution = summary["execution"]

        self.assertEqual(execution["steps"], 2)
        self.assertAlmostEqual(
            execution["simulated_duration_s"],
            0.2,
        )
        self.assertAlmostEqual(
            execution["real_time_factor"],
            0.5,
        )

    def test_mismatched_sequences_are_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_summary(
                position_errors_m=[0.1, 0.2],
                joint_errors_rad=[0.3],
                loop_durations_s=[0.01, 0.01],
                ik_durations_s=[0.005, 0.005],
                timestep_s=0.1,
                wall_duration_s=0.4,
            )


if __name__ == "__main__":
    unittest.main()
