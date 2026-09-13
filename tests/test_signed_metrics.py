import unittest

from metrics import (
    summarize_signed_values,
)


class SignedMetricsTests(unittest.TestCase):
    def test_signed_values_accept_negative_samples(self):
        summary = summarize_signed_values(
            [-0.002, 0.0, 0.003],
            name="signed_signal",
        )

        self.assertEqual(
            summary["count"],
            3,
        )

        self.assertAlmostEqual(
            summary["minimum"],
            -0.002,
        )

        self.assertAlmostEqual(
            summary["maximum"],
            0.003,
        )

        self.assertAlmostEqual(
            summary["maximum_absolute"],
            0.003,
        )

    def test_signed_values_reject_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            summarize_signed_values(
                [0.0, float("nan")],
                name="signed_signal",
            )


if __name__ == "__main__":
    unittest.main()
