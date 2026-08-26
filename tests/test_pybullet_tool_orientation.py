import unittest

import numpy as np

from simulation.pybullet_adapter import (
    PyBulletAdapter,
    PyBulletSettings,
)


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


class PyBulletToolOrientationTests(unittest.TestCase):
    def test_tool_rotation_is_valid(self):
        sim = PyBulletAdapter(
            PyBulletSettings(
                mode="direct",
                timestep_s=0.001,
                output_dir="artifacts/test",
            )
        )

        try:
            sim.configure()
            sim.reset(Q_HOME)

            state = sim.read_state(0.0)

            R = state.tool_rotation_matrix

            self.assertEqual(
                R.shape,
                (3, 3),
            )

            np.testing.assert_allclose(
                R.T @ R,
                np.eye(3),
                atol=1e-6,
            )

            self.assertAlmostEqual(
                np.linalg.det(R),
                1.0,
                delta=1e-6,
            )

        finally:
            sim.shutdown()


if __name__ == "__main__":
    unittest.main()
