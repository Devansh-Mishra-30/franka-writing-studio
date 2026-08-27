"""Phase 5 controlled model-mismatch experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from franka_mechanics import FrankaMechanics


OUTPUT_DIR = Path("artifacts/phase5_model_mismatch")
OUTPUT_CSV = OUTPUT_DIR / "model_mismatch_residuals.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"

SEED = 505
NUM_STATES = 5000
PERTURBED_JOINT = "fer_joint4"
INERTIA_SCALE = 1.15


def build_perturbed_model() -> FrankaMechanics:
    mechanics = FrankaMechanics()

    joint_id = mechanics.model.getJointId(PERTURBED_JOINT)
    inertia = mechanics.model.inertias[joint_id]

    nominal_mass = float(inertia.mass)
    nominal_com = inertia.lever.copy()
    nominal_rotational_inertia = inertia.inertia.copy()

    mechanics.model.inertias[joint_id] = pin.Inertia(
        nominal_mass * INERTIA_SCALE,
        nominal_com,
        nominal_rotational_inertia * INERTIA_SCALE,
    )

    # Model changed, so regenerate Pinocchio data.
    mechanics.data = mechanics.model.createData()

    return mechanics


def generate_state(
    rng: np.random.Generator,
    mechanics: FrankaMechanics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_lower = mechanics.joint_position_lower_limits
    q_upper = mechanics.joint_position_upper_limits

    center = 0.5 * (q_lower + q_upper)
    half_range = 0.5 * (q_upper - q_lower)

    q = center + 0.55 * half_range * rng.uniform(
        -1.0,
        1.0,
        size=7,
    )

    dq = rng.uniform(
        -0.6,
        0.6,
        size=7,
    )

    ddq = rng.uniform(
        -1.0,
        1.0,
        size=7,
    )

    return q, dq, ddq


def main() -> None:
    nominal = FrankaMechanics()
    perturbed = build_perturbed_model()

    rng = np.random.default_rng(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    residual_matrix: list[np.ndarray] = []

    for state_id in range(NUM_STATES):
        q, dq, ddq = generate_state(rng, nominal)

        tau_nominal = nominal.get_tau(q, dq, ddq)
        tau_perturbed = perturbed.get_tau(q, dq, ddq)

        residual = tau_perturbed - tau_nominal
        residual_matrix.append(residual)

        row: dict[str, float | int] = {
            "state_id": state_id,
        }

        for j in range(7):
            row[f"q{j+1}_rad"] = float(q[j])
            row[f"dq{j+1}_rad_s"] = float(dq[j])
            row[f"ddq{j+1}_rad_s2"] = float(ddq[j])

            row[f"tau{j+1}_nominal_nm"] = float(tau_nominal[j])
            row[f"tau{j+1}_perturbed_nm"] = float(tau_perturbed[j])
            row[f"tau{j+1}_residual_nm"] = float(residual[j])

        row["residual_norm_nm"] = float(
            np.linalg.norm(residual)
        )

        rows.append(row)

    residuals = np.vstack(residual_matrix)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    joint_rms = np.sqrt(
        np.mean(residuals**2, axis=0)
    )

    joint_max_abs = np.max(
        np.abs(residuals),
        axis=0,
    )

    residual_norms = np.linalg.norm(
        residuals,
        axis=1,
    )

    summary = {
        "description": "Controlled Franka model-mismatch experiment",
        "states": NUM_STATES,
        "seed": SEED,
        "perturbed_joint": PERTURBED_JOINT,
        "inertia_scale": INERTIA_SCALE,
        "mass_change_percent": (INERTIA_SCALE - 1.0) * 100.0,
        "joint_residual_rms_nm": {
            f"joint_{j+1}": float(joint_rms[j])
            for j in range(7)
        },
        "joint_residual_max_abs_nm": {
            f"joint_{j+1}": float(joint_max_abs[j])
            for j in range(7)
        },
        "mean_residual_norm_nm": float(
            np.mean(residual_norms)
        ),
        "rms_residual_norm_nm": float(
            np.sqrt(np.mean(residual_norms**2))
        ),
        "max_residual_norm_nm": float(
            np.max(residual_norms)
        ),
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Phase 5 Model Mismatch")
    print("======================")
    print(f"States:             {NUM_STATES}")
    print(f"Perturbed joint:    {PERTURBED_JOINT}")
    print(f"Inertia scale:      {INERTIA_SCALE:.3f}")
    print()
    print("Residual torque RMS by joint:")

    for j, value in enumerate(joint_rms, start=1):
        print(f"  J{j}: {value:.6f} N*m")

    print()
    print(
        "RMS residual norm:  "
        f"{summary['rms_residual_norm_nm']:.6f} N*m"
    )
    print(
        "Max residual norm:  "
        f"{summary['max_residual_norm_nm']:.6f} N*m"
    )
    print()
    print(f"CSV:     {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
