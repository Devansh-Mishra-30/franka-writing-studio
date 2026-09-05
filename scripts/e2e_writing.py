"""End-to-end acceptance test for the Franka Writing Studio."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import ExperimentConfig
from robot import RobotInterface, RobotState


# These are regression limits, not claims about the
# theoretical capability of the robot.
MAX_WRITE_XY_RMSE_M = 0.000150
MAX_WRITE_XY_ERROR_M = 0.000500
MIN_WRITE_CONTACT_RATE = 0.99

MAX_WRITE_FORCE_RMS_ERROR_N = 0.150
MAX_MEAN_FORCE_ERROR_N = 0.100

MAX_FORCE_SATURATION_COUNT = 0


def rms(values: list[float]) -> float:
    if not values:
        return math.inf

    return math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def load_write_metrics(
    samples_path: Path,
) -> dict[str, float | int]:
    with samples_path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["writing_phase"] == "WRITE"
        ]

    if not rows:
        raise RuntimeError(
            "No WRITE samples were recorded"
        )

    xy_errors = [
        float(row["xy_tracking_error_m"])
        for row in rows
    ]

    forces = [
        float(row["pen_normal_force_n"])
        for row in rows
    ]

    desired_forces = [
        float(row["desired_pen_normal_force_n"])
        for row in rows
    ]

    force_errors = [
        float(row["force_error_n"])
        for row in rows
        if row.get("force_error_n")
        not in (None, "", "None")
    ]

    contact_count = sum(
        int(row["pen_contact_active"])
        for row in rows
    )

    saturation_count = sum(
        int(row["force_control_saturated"])
        for row in rows
    )

    return {
        "write_samples": len(rows),
        "write_contact_rate": (
            contact_count / len(rows)
        ),
        "write_xy_rmse_m": rms(
            xy_errors
        ),
        "write_xy_max_m": max(
            xy_errors
        ),
        "mean_write_force_n": (
            sum(forces) / len(forces)
        ),
        "mean_target_force_n": (
            sum(desired_forces)
            / len(desired_forces)
        ),
        "write_force_rms_error_n": rms(
            force_errors
        ),
        "force_saturation_count": (
            saturation_count
        ),
    }


def report(
    name: str,
    passed: bool,
    detail: str = "",
) -> bool:
    result = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{name:<28} {result:<5}"
        + (
            f"  {detail}"
            if detail
            else ""
        )
    )

    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Franka Writing Studio "
            "end-to-end acceptance test."
        )
    )

    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("svg/hey.svg"),
        help=(
            "SVG used for the acceptance cycle. "
            "Default: svg/hey.svg"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            Path("artifacts/e2e")
            / datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    config = ExperimentConfig(
        mode="direct",
        duration_s=5.5,
        timestep_s=0.001,
        svg_file=args.svg,
        output_dir=output_dir,
        realtime=False,
        record_video=False,
        speed_scale=1.0,
        full_plan=True,
    ).validate()

    print()
    print(
        "============================================"
    )
    print(
        " FRANKA WRITING STUDIO — E2E ACCEPTANCE"
    )
    print(
        "============================================"
    )
    print(
        f"SVG:       {args.svg}"
    )
    print(
        f"Artifacts: {output_dir}"
    )
    print()

    checks: list[bool] = []

    try:
        robot = RobotInterface(
            config
        )

        checks.append(
            report(
                "INITIAL STATE",
                robot.state
                is RobotState.INIT,
                robot.state.value,
            )
        )

        robot.home()

        checks.append(
            report(
                "HOME / READY",
                robot.state
                is RobotState.READY,
                robot.state.value,
            )
        )

        result = (
            robot.start_writing()
        )

        checks.append(
            report(
                "TASK EXECUTION",
                robot.state
                is RobotState.COMPLETE,
                robot.state.value,
            )
        )

        summary_status = (
            result.summary.get(
                "status"
            )
        )

        checks.append(
            report(
                "EXPERIMENT SUMMARY",
                summary_status
                == "completed",
                str(summary_status),
            )
        )

        full_plan = bool(
            result.summary.get(
                "writing_plan",
                {},
            ).get(
                "full_plan",
                False,
            )
        )

        checks.append(
            report(
                "FULL PLAN",
                full_plan,
            )
        )

        metrics = load_write_metrics(
            result.samples_path
        )

        write_samples = int(
            metrics[
                "write_samples"
            ]
        )

        contact_rate = float(
            metrics[
                "write_contact_rate"
            ]
        )

        xy_rmse_m = float(
            metrics[
                "write_xy_rmse_m"
            ]
        )

        xy_max_m = float(
            metrics[
                "write_xy_max_m"
            ]
        )

        mean_force_n = float(
            metrics[
                "mean_write_force_n"
            ]
        )

        target_force_n = float(
            metrics[
                "mean_target_force_n"
            ]
        )

        force_rms_n = float(
            metrics[
                "write_force_rms_error_n"
            ]
        )

        saturation_count = int(
            metrics[
                "force_saturation_count"
            ]
        )

        checks.append(
            report(
                "WRITE SAMPLES",
                write_samples > 0,
                str(write_samples),
            )
        )

        checks.append(
            report(
                "WRITE CONTACT",
                contact_rate
                >= MIN_WRITE_CONTACT_RATE,
                f"{100.0 * contact_rate:.2f} %",
            )
        )

        checks.append(
            report(
                "WRITE XY RMSE",
                xy_rmse_m
                <= MAX_WRITE_XY_RMSE_M,
                f"{1000.0 * xy_rmse_m:.4f} mm",
            )
        )

        checks.append(
            report(
                "WRITE MAX XY ERROR",
                xy_max_m
                <= MAX_WRITE_XY_ERROR_M,
                f"{1000.0 * xy_max_m:.4f} mm",
            )
        )

        checks.append(
            report(
                "MEAN WRITE FORCE",
                abs(
                    mean_force_n
                    - target_force_n
                )
                <= MAX_MEAN_FORCE_ERROR_N,
                (
                    f"{mean_force_n:.4f} N "
                    f"(target {target_force_n:.4f} N)"
                ),
            )
        )

        checks.append(
            report(
                "FORCE RMS ERROR",
                force_rms_n
                <= MAX_WRITE_FORCE_RMS_ERROR_N,
                f"{force_rms_n:.4f} N",
            )
        )

        checks.append(
            report(
                "FORCE SATURATION",
                saturation_count
                <= MAX_FORCE_SATURATION_COUNT,
                str(saturation_count),
            )
        )

        # v1 HOME establishes the known application
        # home/ready state. The simulator is instantiated
        # per experiment; this is not a claim of persistent
        # physical-hardware homing.
        robot.home()

        checks.append(
            report(
                "RETURN HOME / READY",
                robot.state
                is RobotState.READY,
                robot.state.value,
            )
        )

    except Exception as error:
        print()
        print(
            f"E2E EXCEPTION: {type(error).__name__}: {error}"
        )

        print()
        print(
            "E2E RESULT: FAIL"
        )

        return 1

    print()
    print(
        "--------------------------------------------"
    )

    if all(checks):
        print(
            "E2E RESULT: PASS"
        )
        return 0

    print(
        "E2E RESULT: FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
