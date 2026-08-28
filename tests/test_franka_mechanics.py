import unittest

import numpy as np

from franka_mechanics import FrankaMechanics


INITIAL_Q = np.array(
    [
        0.0,
        -0.785,
        0.0,
        -2.355,
        0.0,
        1.57,
        0.785,
    ],
    dtype=float,
)


class FrankaMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mechanics = FrankaMechanics()

    def test_official_model_and_named_frame(self):
        self.assertEqual(
            self.mechanics.model.nq,
            7,
        )
        self.assertEqual(
            self.mechanics.model.nv,
            7,
        )
        self.assertEqual(
            self.mechanics.tool_frame_name,
            "fer_link8",
        )

    def test_fk_and_jacobian_are_finite(self):
        pose = self.mechanics.solve_fk(INITIAL_Q)
        jacobian = self.mechanics.get_jacobian(
            INITIAL_Q
        )

        self.assertEqual(pose.shape, (6,))
        self.assertEqual(jacobian.shape, (6, 7))

        self.assertTrue(
            np.all(np.isfinite(pose))
        )
        self.assertTrue(
            np.all(np.isfinite(jacobian))
        )

    def test_tool_position_matches_fk_position(self):
        position = self.mechanics.get_tool_position(
            INITIAL_Q
        )
        pose = self.mechanics.solve_fk(
            INITIAL_Q
        )

        self.assertEqual(
            position.shape,
            (3,),
        )

        np.testing.assert_allclose(
            position,
            pose[:3],
            rtol=0.0,
            atol=1e-12,
        )

    def test_official_limits_are_available(self):
        self.assertEqual(
            self.mechanics
            .joint_position_lower_limits
            .shape,
            (7,),
        )
        self.assertEqual(
            self.mechanics
            .joint_position_upper_limits
            .shape,
            (7,),
        )
        self.assertEqual(
            self.mechanics
            .joint_velocity_limits
            .shape,
            (7,),
        )
        self.assertEqual(
            self.mechanics
            .joint_effort_limits
            .shape,
            (7,),
        )


if __name__ == "__main__":
    unittest.main()
