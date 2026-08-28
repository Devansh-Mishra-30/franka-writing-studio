"""Tests for the Writing Studio task lifecycle."""

from pathlib import Path
import unittest

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
