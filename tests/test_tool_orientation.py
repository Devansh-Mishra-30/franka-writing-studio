import unittest

import numpy as np

from franka_mechanics import FrankaMechanics


Q_HOME = np.array(
    [
        0.23081834,
        -0.69476370,
        -0.12918779,
        -2.02518342,
        -0.08491880,
        1.33535219,
        0.11158610,
    ],
    dtype=float,
)


class ToolOrientationTests(unittest.TestCase):
    def test_rotation_matrix_is_valid(self):
        mechanics = FrankaMechanics()

        R = mechanics.get_tool_rotation(
            Q_HOME
        )

        self.assertEqual(
            R.shape,
            (3, 3),
        )

        np.testing.assert_allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-10,
        )

        self.assertAlmostEqual(
            np.linalg.det(R),
            1.0,
            delta=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
