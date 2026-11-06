import tempfile
import unittest
from pathlib import Path

import numpy as np

from drawing import DrawingDocument, Stroke
from experiment_config import ExperimentConfig


class ExperimentConfigTests(unittest.TestCase):
    def make_svg(self, directory: str) -> Path:
        svg_file = Path(directory) / "trajectory.svg"
        svg_file.write_text(
            "<svg></svg>",
            encoding="utf-8",
        )
        return svg_file

    def test_valid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                duration_s=2.0,
                timestep_s=0.001,
                svg_file=self.make_svg(directory),
            ).validate()

            self.assertEqual(config.num_steps, 2000)
            self.assertAlmostEqual(
                config.simulated_duration_s,
                2.0,
            )
    def test_serialized_full_plan_duration_is_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                duration_s=5.5,
                timestep_s=0.001,
                svg_file=self.make_svg(directory),
                full_plan=True,
            ).validate()

            payload = config.to_dict()

            self.assertEqual(
                payload["requested_duration_s"],
                5.5,
            )
            self.assertEqual(
                payload["requested_num_steps"],
                5500,
            )
            self.assertEqual(
                payload["requested_simulated_duration_s"],
                5.5,
            )
            self.assertEqual(
                payload["execution_duration_source"],
                "writing_plan",
            )
            self.assertNotIn("duration_s", payload)
            self.assertNotIn("num_steps", payload)
            self.assertNotIn(
                "simulated_duration_s",
                payload,
            )

    def test_nonpositive_duration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                duration_s=0.0,
                svg_file=self.make_svg(directory),
            )

            with self.assertRaises(ValueError):
                config.validate()

    def test_missing_svg_is_rejected(self):
        config = ExperimentConfig(
            svg_file=Path("missing-trajectory.svg")
        )

        with self.assertRaises(FileNotFoundError):
            config.validate()

    def test_drawing_does_not_require_svg_file(self):
        drawing = DrawingDocument(
            strokes=(
                Stroke(
                    np.array(
                        [
                            [0.0, 0.0],
                            [1.0, 1.0],
                        ]
                    )
                ),
            ),
            source="api",
        )

        config = ExperimentConfig(
            svg_file=Path("missing-trajectory.svg"),
            drawing=drawing,
        ).validate()

        self.assertIs(config.drawing, drawing)
        self.assertEqual(
            config.to_dict()["drawing"]["source"],
            "api",
        )

    def test_invalid_drawing_type_is_rejected(self):
        with self.assertRaises(TypeError):
            ExperimentConfig(
                drawing=object(),
            ).validate()

    def test_direct_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                mode="direct",
                record_video=True,
                svg_file=self.make_svg(directory),
            )

            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
