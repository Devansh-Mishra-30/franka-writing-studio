"""Tests for the authoritative robot application state machine."""

import unittest

from robot import (
    RobotCommand,
    RobotState,
    RobotStateMachine,
)


class RobotStateMachineTests(unittest.TestCase):
    def test_initial_state_is_init(self):
        machine = RobotStateMachine()

        self.assertEqual(
            machine.state,
            RobotState.INIT,
        )

        self.assertIsNone(
            machine.last_transition,
        )

    def test_nominal_writing_cycle(self):
        machine = RobotStateMachine()

        expected_states = (
            RobotState.HOMING,
            RobotState.READY,
            RobotState.PLANNING,
            RobotState.VALIDATING,
            RobotState.EXECUTING,
            RobotState.COMPLETE,
            RobotState.READY,
        )

        for state in expected_states:
            machine.transition(state)

        self.assertEqual(
            machine.state,
            RobotState.READY,
        )

    def test_illegal_transition_is_rejected(self):
        machine = RobotStateMachine()

        with self.assertRaisesRegex(
            RuntimeError,
            "INIT -> EXECUTING",
        ):
            machine.transition(
                RobotState.EXECUTING
            )

        self.assertEqual(
            machine.state,
            RobotState.INIT,
        )

    def test_stop_during_execution(self):
        machine = RobotStateMachine()

        for state in (
            RobotState.HOMING,
            RobotState.READY,
            RobotState.PLANNING,
            RobotState.VALIDATING,
            RobotState.EXECUTING,
            RobotState.STOPPED,
        ):
            machine.transition(state)

        self.assertEqual(
            machine.state,
            RobotState.STOPPED,
        )

    def test_fault_can_be_entered_from_execution(self):
        machine = RobotStateMachine()

        for state in (
            RobotState.HOMING,
            RobotState.READY,
            RobotState.PLANNING,
            RobotState.VALIDATING,
            RobotState.EXECUTING,
            RobotState.FAULT,
        ):
            machine.transition(state)

        self.assertEqual(
            machine.state,
            RobotState.FAULT,
        )

    def test_last_transition_is_recorded(self):
        machine = RobotStateMachine()

        transition = machine.transition(
            RobotState.HOMING
        )

        self.assertEqual(
            transition.previous,
            RobotState.INIT,
        )
        self.assertEqual(
            transition.current,
            RobotState.HOMING,
        )
        self.assertIs(
            machine.last_transition,
            transition,
        )

    def test_high_level_command_contract(self):
        self.assertEqual(
            {command.value for command in RobotCommand},
            {
                "HOME",
                "START_WRITING",
                "STOP",
                "RESET",
            },
        )


if __name__ == "__main__":
    unittest.main()
