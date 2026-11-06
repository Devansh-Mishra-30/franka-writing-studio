"""Three-run E2E acceptance gate for dashboard-style drawings."""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drawing import DrawingDocument, Stroke
from experiment_config import ExperimentConfig
from robot import RobotInterface, RobotState
from scripts.e2e_writing import (
    MAX_FORCE_SATURATION_COUNT,
    MAX_MEAN_FORCE_ERROR_N,
    MAX_WRITE_FORCE_RMS_ERROR_N,
    MAX_WRITE_XY_ERROR_M,
    MAX_WRITE_XY_RMSE_M,
    MIN_WRITE_CONTACT_RATE,
    load_write_metrics,
)


def stroke(points: list[tuple[float, float]]) -> Stroke:
    return Stroke(
        points=np.asarray(
            points,
            dtype=float,
        )
    )


def acceptance_drawings() -> tuple[DrawingDocument, ...]:
    return (
        DrawingDocument(
            strokes=(
                stroke([(20, 20), (20, 180)]),
                stroke([(120, 20), (120, 180)]),
                stroke([(20, 100), (120, 100)]),
                stroke([(180, 20), (180, 180)]),
            ),
            source="touchpad",
            canvas_size=(240.0, 200.0),
        ),
        DrawingDocument(
            strokes=(
                stroke([(20, 180), (20, 20), (90, 40), (110, 100), (90, 160), (20, 180)]),
                stroke([(140, 180), (140, 20), (190, 100), (240, 20), (240, 180)]),
            ),
            source="touchpad",
            canvas_size=(260.0, 200.0),
        ),
        DrawingDocument(
            strokes=(
                stroke(
                    [
                        (130, 100),
                        (125, 80),
                        (105, 65),
                        (75, 70),
                        (55, 95),
                        (60, 130),
                        (95, 155),
                        (140, 145),
                        (170, 110),
                        (165, 65),
                    ]
                ),
                stroke([(205, 40), (205, 165)]),
            ),
            source="touchpad",
            canvas_size=(240.0, 200.0),
        ),
    )


def run_one(
    drawing: DrawingDocument,
    output_dir: Path,
) -> dict[str, float | int]:
    config = ExperimentConfig(
        mode="direct",
        duration_s=5.5,
        timestep_s=0.001,
        output_dir=output_dir,
        realtime=False,
        record_video=False,
        speed_scale=1.0,
        full_plan=True,
        drawing=drawing,
    ).validate()

    robot = RobotInterface(config)
    robot.home()
    result = robot.start_writing()

    if robot.state is not RobotState.COMPLETE:
        raise RuntimeError(
            f"Robot ended in {robot.state.value}"
        )

    if result.summary.get("status") != "completed":
        raise RuntimeError(
            "Experiment summary is not completed"
        )

    metadata = result.summary.get("trajectory", {})

    if metadata.get("stroke_count") != drawing.stroke_count:
        raise RuntimeError(
            "Drawing stroke count was not preserved"
        )

    return load_write_metrics(
        result.samples_path
    )


def metrics_pass(
    metrics: dict[str, float | int],
) -> bool:
    return all(
        (
            int(metrics["write_samples"]) > 0,
            float(metrics["write_contact_rate"])
            >= MIN_WRITE_CONTACT_RATE,
            float(metrics["write_xy_rmse_m"])
            <= MAX_WRITE_XY_RMSE_M,
            float(metrics["write_xy_max_m"])
            <= MAX_WRITE_XY_ERROR_M,
            float(metrics["write_force_rms_error_n"])
            <= MAX_WRITE_FORCE_RMS_ERROR_N,
            abs(
                float(metrics["mean_write_force_n"])
                - float(metrics["mean_target_force_n"])
            )
            <= MAX_MEAN_FORCE_ERROR_N,
            int(metrics["force_saturation_count"])
            <= MAX_FORCE_SATURATION_COUNT,
        )
    )


def main() -> int:
    run_root = (
        Path("artifacts/e2e_drawings")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    print()
    print("============================================")
    print(" FRANKA WRITING STUDIO — DRAWING E2E (3×)")
    print("============================================")

    passed = True

    for index, drawing in enumerate(
        acceptance_drawings(),
        start=1,
    ):
        name = f"drawing_{index}"
        metrics = run_one(
            drawing,
            run_root / name,
        )
        run_passed = metrics_pass(metrics)
        passed = passed and run_passed

        print(
            f"{name}: {'PASS' if run_passed else 'FAIL'} | "
            f"strokes={drawing.stroke_count} | "
            f"contact={100.0 * float(metrics['write_contact_rate']):.2f}% | "
            f"xy_rmse={1000.0 * float(metrics['write_xy_rmse_m']):.4f} mm | "
            f"force_rms={float(metrics['write_force_rms_error_n']):.4f} N"
        )

    print("--------------------------------------------")
    print(
        "DRAWING E2E RESULT: "
        + ("PASS" if passed else "FAIL")
    )
    print(f"Artifacts: {run_root}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
