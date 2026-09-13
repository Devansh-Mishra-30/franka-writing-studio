import csv
import json
import tempfile
import unittest
from pathlib import Path

from logging_utils import (
    build_run_provenance,
    write_csv,
    write_json,
)


class LoggingUtilsTests(unittest.TestCase):
    def test_json_and_csv_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "summary.json"
            csv_path = root / "samples.csv"

            write_json(
                json_path,
                {"status": "passed"},
            )

            write_csv(
                csv_path,
                [
                    {
                        "time_s": 0.0,
                        "error_m": 0.1,
                    },
                    {
                        "time_s": 0.1,
                        "error_m": 0.2,
                    },
                ],
            )

            with json_path.open(
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

            self.assertEqual(
                payload["status"],
                "passed",
            )

            with csv_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0]["time_s"],
                "0.0",
            )

    def test_run_provenance_records_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "robot.urdf"
            model_path.write_text(
                "robot-model",
                encoding="utf-8",
            )

            provenance = build_run_provenance(
                repo_root=root,
                model_path=model_path,
            )

            self.assertIn("created_at_utc", provenance)
            self.assertEqual(
                provenance["model"]["path"],
                "robot.urdf",
            )
            self.assertEqual(
                len(provenance["model"]["sha256"]),
                64,
            )
            self.assertIn(
                "python_version",
                provenance["runtime"],
            )
            self.assertIn(
                "numpy",
                provenance["runtime"]["packages"],
            )
            self.assertIsNone(
                provenance["git"]["commit"],
            )
            self.assertIsNone(
                provenance["git"]["dirty"],
            )

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "empty.csv"

            with self.assertRaises(ValueError):
                write_csv(output, [])


if __name__ == "__main__":
    unittest.main()
