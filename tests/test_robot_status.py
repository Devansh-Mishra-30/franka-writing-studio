"""Tests for robot faults and status snapshots."""

import unittest

import numpy as np

from robot import (
    FaultCode,
    RobotCommand,
    RobotFault,
    RobotState,
    RobotStatus,
)


class RobotStatusTests(unittest.TestCase):
    def test_default_ready_status(self):
        status = RobotStatus(
            state=RobotState.READY
        )

        self.assertEqual(
            status.state,
            RobotState.READY,
        )

        self.assertIsNone(
            status.active_command
        )

        self.assertEqual(
            status.task_progress,
            0.0,
        )

        np.testing.assert_allclose(
            status.tcp_position_m,
            np.zeros(3),
        )

    def test_fault_contains_structured_information(self):
        fault = RobotFault(
            code=FaultCode.IK_FAILURE,
            message="IK failed at waypoint 42",
            source="preflight",
            recoverable=True,
        )

        self.assertEqual(
            fault.code,
            FaultCode.IK_FAILURE,
        )

        self.assertEqual(
            fault.source,
            "preflight",
        )

        self.assertTrue(
            fault.recoverable
        )

    def test_fault_rejects_none_code(self):
        with self.assertRaises(ValueError):
            RobotFault(
                code=FaultCode.NONE,
                message="invalid",
                source="test",
            )

    def test_status_can_expose_active_fault(self):
        fault = RobotFault(
            code=FaultCode.COLLISION_PRECHECK_FAILED,
            message="Collision detected",
            source="validation",
        )

        status = RobotStatus(
            state=RobotState.FAULT,
            active_command=RobotCommand.START_WRITING,
            fault=fault,
        )

        self.assertIs(
            status.fault,
            fault,
        )

    def test_progress_is_bounded(self):
        with self.assertRaises(ValueError):
            RobotStatus(
                state=RobotState.EXECUTING,
                task_progress=1.1,
            )

    def test_tcp_position_must_have_three_values(self):
        with self.assertRaises(ValueError):
            RobotStatus(
                state=RobotState.READY,
                tcp_position_m=np.zeros(2),
            )

    def test_cycle_time_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            RobotStatus(
                state=RobotState.READY,
                cycle_time_s=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
