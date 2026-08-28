"""Tests for the Writing Studio task lifecycle."""

from pathlib import Path
import unittest
from unittest.mock import patch

from experiment_config import ExperimentConfig
from tasks.base import TaskStatus
from tasks.writing_studio import WritingStudioTask


class WritingStudioTaskTests(unittest.TestCase):
    def make_task(self) -> WritingStudioTask:
        return WritingStudioTask(
            ExperimentConfig(
                mode="direct",
                duration_s=1.0,
                timestep_s=0.001,
                svg_file=Path("svg/hey.svg"),
                speed_scale=1.0,
            )
        )

    def test_plan_transitions_task_to_ready(self):
        task = self.make_task()

        self.assertEqual(
            task.status,
            TaskStatus.IDLE,
        )

        planning = task.plan()

        self.assertEqual(
            task.status,
            TaskStatus.READY,
        )

        self.assertIs(
            task.planning_result,
            planning,
        )

        self.assertGreater(
            planning.writing_plan.total_duration_s,
            0.0,
        )

    def test_validate_checks_model_reachability(self):
        task = self.make_task()

        task.plan()
        report = task.validate()

        self.assertTrue(report.success)
        self.assertTrue(report.trajectory_finite)
        self.assertTrue(report.within_notebook_bounds)
        self.assertTrue(report.sampled_reachability)

        self.assertGreater(
            report.checked_waypoints,
            0,
        )

        self.assertIsNone(
            report.failed_waypoint_index
        )

        self.assertLess(
            report.max_position_error_m,
            1e-4,
        )

        self.assertLess(
            report.max_orientation_error_rad,
            1e-4,
        )

    def test_validate_requires_plan(self):
        task = self.make_task()

        with self.assertRaises(RuntimeError):
            task.validate()


    def test_run_requires_plan(self):
        task = self.make_task()

        with self.assertRaises(RuntimeError):
            task.run()

        self.assertEqual(
            task.status,
            TaskStatus.IDLE,
        )

    def test_run_requires_validation(self):
        task = self.make_task()

        task.plan()

        with self.assertRaises(RuntimeError):
            task.run()

        self.assertEqual(
            task.status,
            TaskStatus.READY,
        )

    def test_successful_run_transitions_to_complete(self):
        task = self.make_task()

        task.plan()
        report = task.validate()
        self.assertTrue(report.success)

        expected_result = object()

        with patch.object(
            task._experiment,
            "run",
            return_value=expected_result,
        ):
            result = task.run()

        self.assertIs(
            result,
            expected_result,
        )

        self.assertEqual(
            task.status,
            TaskStatus.COMPLETE,
        )

    def test_execution_failure_transitions_to_failed(self):
        task = self.make_task()

        task.plan()
        report = task.validate()
        self.assertTrue(report.success)

        with patch.object(
            task._experiment,
            "run",
            side_effect=RuntimeError(
                "simulated execution failure"
            ),
        ):
            with self.assertRaises(RuntimeError):
                task.run()

        self.assertEqual(
            task.status,
            TaskStatus.FAILED,
        )

    def test_reset_clears_plan(self):
        task = self.make_task()

        task.plan()
        task.reset()

        self.assertEqual(
            task.status,
            TaskStatus.IDLE,
        )

        self.assertIsNone(
            task.planning_result
        )


if __name__ == "__main__":
    unittest.main()
