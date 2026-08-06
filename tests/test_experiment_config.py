import tempfile
import unittest
from pathlib import Path

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
