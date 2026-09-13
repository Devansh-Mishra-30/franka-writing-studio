"""Export deterministic Pinocchio reference data for MATLAB verification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from franka_mechanics import FrankaMechanics


OUTPUT_DIR = Path("artifacts/phase4_matlab_verification")
OUTPUT_CSV = OUTPUT_DIR / "pinocchio_reference.csv"
OUTPUT_META = OUTPUT_DIR / "metadata.json"

SEED = 441
NUM_STATES = 10

# Validated configuration used in the writing experiments.
HOME_Q = np.array(
    [
        0.23081834,
        -0.69476370,
        -0.12918779,
        -2.02518342,
        -0.08491880,
        1.33535219,
        0.11158610,
    ],
    dtype=float,
)


def flatten(prefix: str, array: np.ndarray) -> dict[str, float]:
    """Flatten arrays into deterministic scalar CSV columns."""
    arr = np.asarray(array, dtype=float)

    if arr.ndim == 1:
        return {
            f"{prefix}{i + 1}": float(value)
            for i, value in enumerate(arr)
        }

    if arr.ndim == 2:
        return {
            f"{prefix}{r + 1}_{c + 1}": float(arr[r, c])
            for r in range(arr.shape[0])
            for c in range(arr.shape[1])
        }

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def make_states(
    mechanics: FrankaMechanics,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate deterministic, joint-limit-safe verification states."""
    rng = np.random.default_rng(SEED)

    q_lower = mechanics.joint_position_lower_limits
    q_upper = mechanics.joint_position_upper_limits

    center = 0.5 * (q_lower + q_upper)
    half_range = 0.5 * (q_upper - q_lower)

    states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (
            HOME_Q.copy(),
            np.zeros(7),
            np.zeros(7),
        )
    ]

    for _ in range(NUM_STATES - 1):
        # Stay comfortably away from joint limits.
        q = center + 0.55 * half_range * rng.uniform(-1.0, 1.0, size=7)

        dq = rng.uniform(
            low=-0.6,
            high=0.6,
            size=7,
        )

        ddq = rng.uniform(
            low=-1.0,
            high=1.0,
            size=7,
        )

        states.append((q, dq, ddq))

    return states


def main() -> None:
    mechanics = FrankaMechanics()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []

    for state_id, (q, dq, ddq) in enumerate(make_states(mechanics)):
        # Forward kinematics for the exact task frame used by the project.
        pin.forwardKinematics(
            mechanics.model,
            mechanics.data,
            q,
        )
        pin.updateFramePlacements(
            mechanics.model,
            mechanics.data,
        )

        placement = mechanics.data.oMf[mechanics.tool_frame_id]

        tool_position = np.asarray(
            placement.translation,
            dtype=float,
        ).reshape(3)

        tool_rotation = np.asarray(
            placement.rotation,
            dtype=float,
        ).reshape(3, 3)

        jacobian = mechanics.get_jacobian(q)

        mass_matrix = mechanics.get_M(q)
        coriolis_matrix = mechanics.get_C(q, dq)
        gravity = mechanics.get_G(q)
        tau_rnea = mechanics.get_tau(q, dq, ddq)

        velocity_product = coriolis_matrix @ dq

        tau_decomposed = (
            mass_matrix @ ddq
            + velocity_product
            + gravity
        )

        reconstruction_error = float(
            np.linalg.norm(tau_rnea - tau_decomposed)
        )

        row: dict[str, float | int] = {
            "state_id": state_id,
            "pinocchio_reconstruction_error_nm": reconstruction_error,
        }

        row.update(flatten("q", q))
        row.update(flatten("dq", dq))
        row.update(flatten("ddq", ddq))

        row.update(flatten("tool_position_", tool_position))
        row.update(flatten("tool_rotation_", tool_rotation))

        row.update(flatten("jacobian_", jacobian))
        row.update(flatten("mass_matrix_", mass_matrix))

        row.update(flatten("velocity_product_", velocity_product))
        row.update(flatten("gravity_", gravity))
        row.update(flatten("tau_rnea_", tau_rnea))

        rows.append(row)

    if not rows:
        raise RuntimeError("No verification states generated.")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "description": (
            "Pinocchio reference data for independent MATLAB "
            "Robotics System Toolbox verification."
        ),
        "urdf": str(mechanics.urdf_path),
        "tool_frame": mechanics.tool_frame_name,
        "number_of_states": len(rows),
        "random_seed": SEED,
        "jacobian_reference_frame": "LOCAL_WORLD_ALIGNED",
        "pinocchio_jacobian_order": [
            "linear_x",
            "linear_y",
            "linear_z",
            "angular_x",
            "angular_y",
            "angular_z",
        ],
        "notes": [
            "Raw Coriolis matrices are not compared across libraries.",
            "Verification compares C(q,dq)*dq via velocity-product torque.",
            "State 0 uses the validated writing configuration with dq=ddq=0.",
        ],
    }

    OUTPUT_META.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    max_reconstruction_error = max(
        float(row["pinocchio_reconstruction_error_nm"])
        for row in rows
    )

    print(f"Exported {len(rows)} verification states")
    print(f"CSV:      {OUTPUT_CSV}")
    print(f"Metadata: {OUTPUT_META}")
    print(
        "Max Pinocchio internal reconstruction error: "
        f"{max_reconstruction_error:.3e} N*m"
    )


if __name__ == "__main__":
    main()
