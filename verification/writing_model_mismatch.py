"""Evaluate controlled dynamics mismatch on the validated writing trajectory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from franka_mechanics import FrankaMechanics
from verification.model_mismatch_experiment import (
    INERTIA_SCALE,
    PERTURBED_JOINT,
    build_perturbed_model,
)


INPUT_CSV = Path(
    "artifacts/phase2_full_plan_1x/samples.csv"
)

OUTPUT_DIR = Path(
    "artifacts/phase5_model_mismatch/writing_trajectory"
)

OUTPUT_CSV = OUTPUT_DIR / "writing_residuals.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"


def load_trajectory() -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    times: list[float] = []
    phases: list[str] = []
    q_rows: list[list[float]] = []
    dq_rows: list[list[float]] = []

    with INPUT_CSV.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            times.append(
                float(row["simulation_time_s"])
            )

            phases.append(
                row["writing_phase"]
            )

            q_rows.append(
                [
                    float(row[f"q{j}_rad"])
                    for j in range(1, 8)
                ]
            )

            dq_rows.append(
                [
                    float(row[f"dq{j}_rad_s"])
                    for j in range(1, 8)
                ]
            )

    return (
        np.asarray(times, dtype=float),
        phases,
        np.asarray(q_rows, dtype=float),
        np.asarray(dq_rows, dtype=float),
    )


def estimate_acceleration(
    time_s: np.ndarray,
    dq: np.ndarray,
) -> np.ndarray:
    if len(time_s) < 3:
        raise ValueError(
            "Need at least three trajectory samples."
        )

    dt = np.diff(time_s)

    if np.any(dt <= 0.0):
        raise ValueError(
            "Simulation timestamps must be strictly increasing."
        )

    # np.gradient uses central differences internally and
    # one-sided differences at the trajectory boundaries.
    ddq = np.empty_like(dq)

    for j in range(7):
        ddq[:, j] = np.gradient(
            dq[:, j],
            time_s,
            edge_order=2,
        )

    return ddq


def rms(values: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.square(values)))
    )


def summarize(
    residuals: np.ndarray,
) -> dict[str, object]:
    residual_norm = np.linalg.norm(
        residuals,
        axis=1,
    )

    return {
        "samples": int(len(residuals)),
        "joint_residual_rms_nm": {
            f"joint_{j + 1}": rms(
                residuals[:, j]
            )
            for j in range(7)
        },
        "joint_residual_max_abs_nm": {
            f"joint_{j + 1}": float(
                np.max(
                    np.abs(
                        residuals[:, j]
                    )
                )
            )
            for j in range(7)
        },
        "mean_residual_norm_nm": float(
            np.mean(residual_norm)
        ),
        "rms_residual_norm_nm": rms(
            residual_norm
        ),
        "max_residual_norm_nm": float(
            np.max(residual_norm)
        ),
    }


def main() -> None:
    nominal = FrankaMechanics()
    perturbed = build_perturbed_model()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        time_s,
        phases,
        q,
        dq,
    ) = load_trajectory()

    ddq = estimate_acceleration(
        time_s,
        dq,
    )

    num_samples = len(time_s)

    residuals = np.zeros(
        (num_samples, 7),
        dtype=float,
    )

    tau_nominal_all = np.zeros_like(
        residuals
    )

    tau_perturbed_all = np.zeros_like(
        residuals
    )

    print("Phase 5 Writing-Trajectory Model Mismatch")
    print("=========================================")
    print(f"Input:             {INPUT_CSV}")
    print(f"Samples:           {num_samples}")
    print(f"Perturbed joint:   {PERTURBED_JOINT}")
    print(f"Inertia scale:     {INERTIA_SCALE:.3f}")
    print()
    print("Evaluating dynamics...")

    for i in range(num_samples):
        tau_nominal = nominal.get_tau(
            q[i],
            dq[i],
            ddq[i],
        )

        tau_perturbed = perturbed.get_tau(
            q[i],
            dq[i],
            ddq[i],
        )

        residual = (
            tau_perturbed
            - tau_nominal
        )

        tau_nominal_all[i] = tau_nominal
        tau_perturbed_all[i] = tau_perturbed
        residuals[i] = residual

        if (
            (i + 1) % 25000 == 0
            or i + 1 == num_samples
        ):
            print(
                f"  {i + 1}/{num_samples}"
            )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "sample",
            "simulation_time_s",
            "writing_phase",
        ]

        for j in range(1, 8):
            fieldnames.extend(
                [
                    f"q{j}_rad",
                    f"dq{j}_rad_s",
                    f"ddq{j}_estimated_rad_s2",
                    f"tau{j}_nominal_nm",
                    f"tau{j}_perturbed_nm",
                    f"tau{j}_residual_nm",
                ]
            )

        fieldnames.append(
            "residual_norm_nm"
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(num_samples):
            row: dict[str, object] = {
                "sample": i,
                "simulation_time_s": float(
                    time_s[i]
                ),
                "writing_phase": phases[i],
            }

            for j in range(7):
                index = j + 1

                row[
                    f"q{index}_rad"
                ] = float(q[i, j])

                row[
                    f"dq{index}_rad_s"
                ] = float(dq[i, j])

                row[
                    f"ddq{index}_estimated_rad_s2"
                ] = float(ddq[i, j])

                row[
                    f"tau{index}_nominal_nm"
                ] = float(
                    tau_nominal_all[i, j]
                )

                row[
                    f"tau{index}_perturbed_nm"
                ] = float(
                    tau_perturbed_all[i, j]
                )

                row[
                    f"tau{index}_residual_nm"
                ] = float(
                    residuals[i, j]
                )

            row["residual_norm_nm"] = float(
                np.linalg.norm(
                    residuals[i]
                )
            )

            writer.writerow(row)

    all_summary = summarize(
        residuals
    )

    write_mask = np.asarray(
        [
            phase.upper() == "WRITE"
            for phase in phases
        ],
        dtype=bool,
    )

    summary: dict[str, object] = {
        "description": (
            "Controlled dynamics-model mismatch "
            "evaluated on the validated full "
            "1x Franka writing trajectory."
        ),
        "source_csv": str(INPUT_CSV),
        "perturbed_joint": PERTURBED_JOINT,
        "inertia_scale": INERTIA_SCALE,
        "mass_change_percent": (
            INERTIA_SCALE - 1.0
        ) * 100.0,
        "acceleration_estimation": (
            "numpy.gradient of logged dq "
            "with simulation timestamps"
        ),
        "overall": all_summary,
    }

    if np.any(write_mask):
        summary["write_phase"] = summarize(
            residuals[write_mask]
        )

    OUTPUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Overall residual RMS by joint:")

    for name, value in (
        all_summary[
            "joint_residual_rms_nm"
        ]
    ).items():
        print(
            f"  {name}: {value:.6f} N*m"
        )

    print()
    print(
        "Overall RMS residual norm: "
        f"{all_summary['rms_residual_norm_nm']:.6f} N*m"
    )

    if "write_phase" in summary:
        write_summary = summary[
            "write_phase"
        ]

        print(
            "WRITE samples:             "
            f"{write_summary['samples']}"
        )

        print(
            "WRITE RMS residual norm:   "
            f"{write_summary['rms_residual_norm_nm']:.6f} N*m"
        )

        print(
            "WRITE max residual norm:   "
            f"{write_summary['max_residual_norm_nm']:.6f} N*m"
        )

    print()
    print(f"CSV:     {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
