import unittest

import numpy as np
import pybullet as p

from pen_tip_kinematics import (
    pen_tip_position,
)
from simulation.pybullet_adapter import (
    PEN_GRIPPER_JOINT_POSITION_M,
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


class PhysicalPenTests(unittest.TestCase):
    def setUp(self):
        self.sim = PyBulletAdapter(
            PyBulletSettings(
                mode="direct",
                timestep_s=0.001,
                output_dir="artifacts/test",
            )
        )

        self.sim.configure()
        self.sim.reset(Q_HOME)

    def tearDown(self):
        self.sim.shutdown()

    def test_contact_state_is_clear_at_home(self):
        contact = (
            self.sim.read_pen_contact_state()
        )

        self.assertFalse(contact.active)
        self.assertEqual(
            contact.contact_count,
            0,
        )
        self.assertIsNone(
            contact.minimum_distance_m
        )
        self.assertEqual(
            contact.normal_force_n,
            0.0,
        )

    def test_contact_force_is_nonnegative(self):
        contact = (
            self.sim.read_pen_contact_state()
        )

        self.assertGreaterEqual(
            contact.normal_force_n,
            0.0,
        )

    def test_physical_pen_exists(self):
        self.assertIsNotNone(
            self.sim.pen_id
        )

        self.assertIsNotNone(
            self.sim.pen_constraint_id
        )

    def test_physical_pen_has_collision_geometry(self):
        collision_shapes = (
            p.getCollisionShapeData(
                self.sim.pen_id,
                -1,
                physicsClientId=(
                    self.sim.client_id
                ),
            )
        )

        self.assertGreater(
            len(collision_shapes),
            0,
        )

    def test_physical_tip_matches_kinematic_tip(self):
        state = self.sim.read_state(0.0)

        expected_tip = pen_tip_position(
            tool_position_m=(
                state.tool_position_m
            ),
            tool_rotation=(
                state.tool_rotation_matrix
            ),
        )

        physical_tip = (
            self.sim.read_pen_tip_position()
        )

        np.testing.assert_allclose(
            physical_tip,
            expected_tip,
            atol=1e-6,
        )

    def test_gripper_matches_pen_diameter(self):
        states = p.getJointStates(
            self.sim.robot_id,
            list(
                self.sim.finger_joint_indices
            ),
            physicsClientId=(
                self.sim.client_id
            ),
        )

        positions = np.asarray(
            [
                state[0]
                for state in states
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            positions,
            [
                PEN_GRIPPER_JOINT_POSITION_M,
                PEN_GRIPPER_JOINT_POSITION_M,
            ],
            atol=1e-9,
        )

    def test_pen_does_not_collide_with_robot(self):
        p.performCollisionDetection(
            physicsClientId=(
                self.sim.client_id
            )
        )

        contacts = p.getContactPoints(
            bodyA=self.sim.robot_id,
            bodyB=self.sim.pen_id,
            physicsClientId=(
                self.sim.client_id
            ),
        )

        self.assertEqual(
            len(contacts),
            0,
        )

    def test_pen_does_not_contact_table_at_home(self):
        p.performCollisionDetection(
            physicsClientId=(
                self.sim.client_id
            )
        )

        contacts = p.getContactPoints(
            bodyA=self.sim.pen_id,
            bodyB=self.sim.table_id,
            physicsClientId=(
                self.sim.client_id
            ),
        )

        self.assertEqual(
            len(contacts),
            0,
        )

    def test_force_is_zero_without_contact(self):
        self.assertAlmostEqual(
            self.sim.read_pen_normal_force_n(),
            0.0,
            delta=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
