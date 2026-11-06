"""Input-selection tests for the Qt experiment worker."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from drawing import DrawingDocument, Stroke
from ui.experiment_worker import ExperimentWorker


def make_drawing() -> DrawingDocument:
    return DrawingDocument(
        strokes=(
            Stroke(
                np.array(
                    [
                        [0.0, 0.0],
                        [10.0, 10.0],
                    ]
                )
            ),
        ),
        source="touchpad",
        canvas_size=(100.0, 100.0),
    )


class FakeRobot:
    configs = []

    def __init__(self, config, **kwargs):
        del kwargs
        self.configs.append(config)
        self.status = object()

    def home(self):
        return None

    def start_writing(self):
        return object()


class ExperimentWorkerTests(unittest.TestCase):
    def test_requires_one_input(self):
        with self.assertRaises(ValueError):
            ExperimentWorker(
                output_dir=Path("artifacts/test"),
                speed_scale=1.0,
                show_robot=False,
                full_plan=True,
            )

    def test_in_memory_drawing_reaches_config(self):
        drawing = make_drawing()
        worker = ExperimentWorker(
            drawing=drawing,
            output_dir=Path("artifacts/test"),
            speed_scale=1.0,
            show_robot=False,
            full_plan=True,
        )

        FakeRobot.configs.clear()

        with patch(
            "ui.experiment_worker.RobotInterface",
            FakeRobot,
        ):
            worker.run()

        self.assertEqual(len(FakeRobot.configs), 1)
        self.assertIs(
            FakeRobot.configs[0].drawing,
            drawing,
        )

    def test_svg_input_remains_supported(self):
        svg_path = Path("svg/hey.svg")
        worker = ExperimentWorker(
            svg_file=svg_path,
            output_dir=Path("artifacts/test"),
            speed_scale=1.0,
            show_robot=False,
            full_plan=True,
        )

        FakeRobot.configs.clear()

        with patch(
            "ui.experiment_worker.RobotInterface",
            FakeRobot,
        ):
            worker.run()

        self.assertEqual(len(FakeRobot.configs), 1)
        self.assertEqual(
            FakeRobot.configs[0].svg_file,
            svg_path,
        )
        self.assertIsNone(
            FakeRobot.configs[0].drawing
        )


if __name__ == "__main__":
    unittest.main()
