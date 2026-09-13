"""Tests for the PLC-ready robot application interface."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment_config import ExperimentConfig
from experiments.writing_experiment import ExperimentStopped
from robot import (
    CommandRejected,
    FaultCode,
    RobotInterface,
    RobotState,
)


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[str] = []

        self.planning_result = None

        self.validation_report = SimpleNamespace(
            success=True,
            trajectory_finite=True,
            within_notebook_bounds=True,
            sampled_reachability=True,
            sampled_collision_free=True,
            failed_waypoint_index=None,
            collision_waypoint_index=None,
        )

        self.run_started = threading.Event()
        self.stop_requested = threading.Event()

        self.block_execution = False

        self.result = object()

    def reset(self) -> None:
        self.calls.append("reset")
        self.planning_result = None

    def plan(self):
        self.calls.append("plan")

        self.planning_result = SimpleNamespace(
            writing_plan=SimpleNamespace(
                total_duration_s=10.0
            )
        )

        return self.planning_result

    def validate(self):
        self.calls.append("validate")
        return self.validation_report

    def run(self):
        self.calls.append("run")
        self.run_started.set()

        if self.block_execution:
            if not self.stop_requested.wait(
                timeout=2.0
            ):
                raise RuntimeError(
                    "test stop timeout"
                )

            raise ExperimentStopped(
                "operator stop"
            )

        return self.result

    def request_stop(self) -> None:
        self.calls.append("request_stop")
        self.stop_requested.set()


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        mode="direct",
        duration_s=1.0,
        timestep_s=0.001,
        svg_file=Path("svg/hey.svg"),
        speed_scale=1.0,
    )


class RobotInterfaceTests(unittest.TestCase):
    def make_robot(
        self,
        task: FakeTask | None = None,
    ) -> tuple[RobotInterface, FakeTask]:
        fake = (
            task
            if task is not None
            else FakeTask()
        )

        robot = RobotInterface(
            make_config(),
            task=fake,
        )

        return robot, fake

    def test_initial_state_is_init(self):
        robot, _ = self.make_robot()

        self.assertEqual(
            robot.state,
            RobotState.INIT,
        )

    def test_home_transitions_to_ready(self):
        robot, task = self.make_robot()

        robot.home()

        self.assertEqual(
            robot.state,
            RobotState.READY,
        )

        self.assertEqual(
            task.calls,
            ["reset"],
        )

    def test_start_writing_requires_ready(self):
        robot, _ = self.make_robot()

        with self.assertRaises(
            CommandRejected
        ):
            robot.start_writing()

        self.assertEqual(
            robot.state,
            RobotState.INIT,
        )

    def test_successful_writing_cycle(self):
        robot, task = self.make_robot()

        robot.home()

        result = robot.start_writing()

        self.assertIs(
            result,
            task.result,
        )

        self.assertEqual(
            robot.state,
            RobotState.COMPLETE,
        )

        self.assertEqual(
            task.calls,
            [
                "reset",
                "plan",
                "validate",
                "run",
            ],
        )

        self.assertEqual(
            robot.status.task_progress,
            1.0,
        )

    def test_workspace_failure_enters_fault(self):
        task = FakeTask()

        task.validation_report = SimpleNamespace(
            success=False,
            trajectory_finite=True,
            within_notebook_bounds=False,
            sampled_reachability=True,
            sampled_collision_free=True,
            failed_waypoint_index=None,
            collision_waypoint_index=None,
        )

        robot, _ = self.make_robot(task)

        robot.home()

        with self.assertRaises(
            RuntimeError
        ):
            robot.start_writing()

        self.assertEqual(
            robot.state,
            RobotState.FAULT,
        )

        self.assertEqual(
            robot.status.fault.code,
            FaultCode.WORKSPACE_VIOLATION,
        )

    def test_reset_recovers_from_fault(self):
        task = FakeTask()

        task.validation_report = SimpleNamespace(
            success=False,
            trajectory_finite=True,
            within_notebook_bounds=False,
            sampled_reachability=True,
            sampled_collision_free=True,
            failed_waypoint_index=None,
            collision_waypoint_index=None,
        )

        robot, _ = self.make_robot(task)

        robot.home()

        with self.assertRaises(
            RuntimeError
        ):
            robot.start_writing()

        robot.reset()

        self.assertEqual(
            robot.state,
            RobotState.READY,
        )

        self.assertIsNone(
            robot.status.fault
        )

    def test_stop_is_cooperatively_acknowledged(self):
        task = FakeTask()
        task.block_execution = True

        robot, _ = self.make_robot(task)

        robot.home()

        stopped = []

        def run_cycle():
            try:
                robot.start_writing()
            except ExperimentStopped:
                stopped.append(True)

        worker = threading.Thread(
            target=run_cycle
        )

        worker.start()

        self.assertTrue(
            task.run_started.wait(
                timeout=2.0
            )
        )

        self.assertEqual(
            robot.state,
            RobotState.EXECUTING,
        )

        robot.stop()

        worker.join(
            timeout=2.0
        )

        self.assertFalse(
            worker.is_alive()
        )

        self.assertEqual(
            stopped,
            [True],
        )

        self.assertEqual(
            robot.state,
            RobotState.STOPPED,
        )

        self.assertIn(
            "request_stop",
            task.calls,
        )

    def test_telemetry_updates_authoritative_status(self):
        robot, task = self.make_robot()

        robot.home()
        task.plan()

        robot._on_telemetry(
            {
                "simulation_time_s": 5.0,
                "physical_pen_x_m": 0.40,
                "physical_pen_y_m": 0.10,
                "physical_pen_z_m": 0.52,
                "simulator_position_error_m": 0.0002,
                "pen_normal_force_n": 1.01,
                "force_control_active": 1,
            }
        )

        status = robot.status

        self.assertAlmostEqual(
            status.task_progress,
            0.5,
        )

        self.assertAlmostEqual(
            status.cycle_time_s,
            5.0,
        )

        self.assertAlmostEqual(
            status.normal_force_n,
            1.01,
        )

        self.assertEqual(
            status.controller_mode,
            "HYBRID_XY_POSITION_Z_FORCE",
        )

        self.assertAlmostEqual(
            status.tcp_position_m[0],
            0.40,
        )

    def test_reset_from_complete_returns_ready(self):
        robot, _ = self.make_robot()

        robot.home()
        robot.start_writing()
        robot.reset()

        self.assertEqual(
            robot.state,
            RobotState.READY,
        )


if __name__ == "__main__":
    unittest.main()
