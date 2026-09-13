import unittest

import numpy as np
import pinocchio as pin

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


class JacobianConsistencyTests(unittest.TestCase):
    def test_local_world_aligned_jacobian_matches_finite_difference(self):
        mechanics = FrankaMechanics()

        dq = np.array(
            [
                0.10,
                -0.08,
                0.06,
                -0.04,
                0.05,
                -0.03,
                0.02,
            ],
            dtype=float,
        )

        dt = 1e-7

        q0 = Q_HOME.copy()
        q1 = q0 + dt * dq

        J = mechanics.get_jacobian(q0)

        predicted_linear = (
            J[:3, :] @ dq
        )

        predicted_angular = (
            J[3:, :] @ dq
        )

        pose0 = mechanics.solve_fk(q0)
        pose1 = mechanics.solve_fk(q1)

        p0 = np.asarray(
            pose0[:3],
            dtype=float,
        )

        p1 = np.asarray(
            pose1[:3],
            dtype=float,
        )

        measured_linear = (
            p1 - p0
        ) / dt

        R0 = mechanics.get_tool_rotation(q0)
        R1 = mechanics.get_tool_rotation(q1)

        # LOCAL_WORLD_ALIGNED angular velocity
        # is expressed in world-aligned coordinates.
        measured_angular = (
            np.asarray(
                pin.log3(
                    R1 @ R0.T
                ),
                dtype=float,
            ).reshape(3)
            / dt
        )

        np.testing.assert_allclose(
            predicted_linear,
            measured_linear,
            rtol=1e-4,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            predicted_angular,
            measured_angular,
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
