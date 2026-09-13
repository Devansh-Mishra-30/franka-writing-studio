"""Tests for simulator-independent writing planning."""

import unittest
from pathlib import Path

import numpy as np

from experiment_config import ExperimentConfig
from experiments.writing_experiment import (
    INITIAL_JOINT_POSITIONS_RAD,
    WritingExperiment,
)
from pen_tip_kinematics import pen_tip_position
from workcells import WRITING_STUDIO


class WritingPlanningTests(unittest.TestCase):
    def test_build_plan_without_simulator(self):
        config = ExperimentConfig(
            mode="direct",
            duration_s=1.0,
            timestep_s=0.001,
            svg_file=Path("svg/hey.svg"),
            speed_scale=1.0,
        )

        experiment = WritingExperiment(config)

        tool_position_m = (
            experiment.mechanics.get_tool_position(
                INITIAL_JOINT_POSITIONS_RAD
            )
        )

        tool_rotation = (
            experiment.mechanics.get_tool_rotation(
                INITIAL_JOINT_POSITIONS_RAD
            )
        )

        initial_pen_tip_position_m = pen_tip_position(
            tool_position_m=tool_position_m,
            tool_rotation=tool_rotation,
        )

        planning = experiment.build_plan(
            initial_pen_tip_position_m
        )

        self.assertEqual(
            planning.nominal_pen_tip_waypoints_m.shape[1],
            3,
        )

        self.assertEqual(
            planning.pen_tip_waypoints_m.shape,
            planning.nominal_pen_tip_waypoints_m.shape,
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    planning.pen_tip_waypoints_m
                )
            )
        )

        self.assertAlmostEqual(
            planning.contact_parameters.surface_height_m,
            WRITING_STUDIO.notebook.top_height_m,
        )

        self.assertLess(
            planning.write_height_m,
            planning.nominal_write_height_m,
        )

        self.assertGreater(
            planning.writing_plan.total_duration_s,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
