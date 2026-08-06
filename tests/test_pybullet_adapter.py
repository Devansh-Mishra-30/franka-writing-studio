import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulation.interfaces import VelocityCommand
from simulation.pybullet_adapter import (
    ARM_JOINT_NAMES,
    PyBulletAdapter,
    PyBulletSettings,
)


INITIAL_Q = np.array(
    [0.0, -0.785, 0.0, -2.355, 0.0, 1.57, 0.785],
    dtype=float,
)


class PyBulletAdapterTests(unittest.TestCase):
    def test_official_named_model(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = PyBulletAdapter(
                PyBulletSettings(
                    mode="direct",
                    timestep_s=0.001,
                    output_dir=Path(directory),
                )
            )

            try:
                adapter.configure()
                adapter.reset(INITIAL_Q)
                state = adapter.read_state(0.0)

                self.assertEqual(adapter.joint_count, 7)
                self.assertEqual(
                    adapter.arm_joint_names,
                    ARM_JOINT_NAMES,
                )
                self.assertEqual(
                    adapter.tool_frame_name,
                    "fer_link8",
                )
                self.assertEqual(
                    len(adapter.finger_joint_indices),
                    2,
                )
                self.assertEqual(
                    state.joint_positions_rad.shape,
                    (7,),
                )
                self.assertEqual(
                    state.joint_velocities_rad_s.shape,
                    (7,),
                )
                self.assertEqual(
                    state.tool_position_m.shape,
                    (3,),
                )

                np.testing.assert_allclose(
                    state.joint_positions_rad,
                    INITIAL_Q,
                    atol=1e-12,
                )

                adapter.apply_velocity_command(
                    VelocityCommand(
                        joint_velocities_rad_s=np.zeros(7)
                    )
                )
                adapter.step()

            finally:
                adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
