"""Phase 6: physics + learned residual dynamics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


TRAIN_CSV = Path(
    "artifacts/phase5_model_mismatch/model_mismatch_residuals.csv"
)

TEST_CSV = Path(
    "artifacts/phase5_model_mismatch/writing_trajectory/writing_residuals.csv"
)

OUTPUT_DIR = Path("artifacts/phase6_residual_learning")
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"
OUTPUT_MODEL = OUTPUT_DIR / "ridge_model.npz"

RIDGE_ALPHA = 1.0e-3


def load_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            q = np.array(
                [float(row[f"q{j}_rad"]) for j in range(1, 8)]
            )

            dq = np.array(
                [float(row[f"dq{j}_rad_s"]) for j in range(1, 8)]
            )

            ddq_key = (
                "ddq1_rad_s2"
                if "ddq1_rad_s2" in row
                else "ddq1_estimated_rad_s2"
            )

            if ddq_key == "ddq1_rad_s2":
                ddq = np.array(
                    [
                        float(row[f"ddq{j}_rad_s2"])
                        for j in range(1, 8)
                    ]
                )
            else:
                ddq = np.array(
                    [
                        float(row[f"ddq{j}_estimated_rad_s2"])
                        for j in range(1, 8)
                    ]
                )

            residual = np.array(
                [
                    float(row[f"tau{j}_residual_nm"])
                    for j in range(1, 8)
                ]
            )

            x_rows.append(
                np.concatenate([q, dq, ddq])
            )
            y_rows.append(residual)

    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
    )


def nonlinear_features(x: np.ndarray) -> np.ndarray:
    """Feature map for configuration-dependent robot dynamics."""
    q = x[:, 0:7]
    dq = x[:, 7:14]
    ddq = x[:, 14:21]

    features = [
        np.ones((len(x), 1)),
        q,
        np.sin(q),
        np.cos(q),
        dq,
        ddq,
        dq**2,
        dq * ddq,
    ]

    # Pairwise velocity products approximate Coriolis structure.
    pairwise = []

    for i in range(7):
        for j in range(i, 7):
            pairwise.append(
                (dq[:, i] * dq[:, j])[:, None]
            )

    features.extend(pairwise)

    # Configuration-dependent acceleration effects.
    for j in range(7):
        features.append(
            np.sin(q) * ddq[:, j : j + 1]
        )
        features.append(
            np.cos(q) * ddq[:, j : j + 1]
        )

    return np.hstack(features)


def rms(values: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.square(values)))
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_raw, y_train = load_dataset(TRAIN_CSV)
    x_test_raw, y_test = load_dataset(TEST_CSV)

    x_train = nonlinear_features(x_train_raw)
    x_test = nonlinear_features(x_test_raw)

    # Standardize every feature except the constant bias column.
    mean = np.mean(x_train[:, 1:], axis=0)
    std = np.std(x_train[:, 1:], axis=0)

    std[std < 1.0e-12] = 1.0

    x_train[:, 1:] = (
        x_train[:, 1:] - mean
    ) / std

    x_test[:, 1:] = (
        x_test[:, 1:] - mean
    ) / std

    # Closed-form multivariate ridge regression.
    identity = np.eye(x_train.shape[1])
    identity[0, 0] = 0.0  # Do not regularize intercept.

    weights = np.linalg.solve(
        x_train.T @ x_train
        + RIDGE_ALPHA * identity,
        x_train.T @ y_train,
    )

    y_pred_train = x_train @ weights
    y_pred_test = x_test @ weights

    train_error = y_train - y_pred_train
    test_error = y_test - y_pred_test

    # Physics-only error is exactly the unmodeled residual.
    physics_only_error = y_test

    physics_rms_joint = np.sqrt(
        np.mean(physics_only_error**2, axis=0)
    )

    hybrid_rms_joint = np.sqrt(
        np.mean(test_error**2, axis=0)
    )

    physics_norm = np.linalg.norm(
        physics_only_error,
        axis=1,
    )

    hybrid_norm = np.linalg.norm(
        test_error,
        axis=1,
    )

    physics_rms_norm = rms(physics_norm)
    hybrid_rms_norm = rms(hybrid_norm)

    improvement_percent = (
        100.0
        * (physics_rms_norm - hybrid_rms_norm)
        / physics_rms_norm
    )

    summary = {
        "model": "NumPy nonlinear-feature ridge regression",
        "ridge_alpha": RIDGE_ALPHA,
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "features": int(x_train.shape[1]),
        "training_residual_rmse_nm": rms(train_error),
        "physics_only_rms_residual_norm_nm": physics_rms_norm,
        "hybrid_rms_residual_norm_nm": hybrid_rms_norm,
        "improvement_percent": improvement_percent,
        "physics_only_joint_rms_nm": {
            f"joint_{j+1}": float(physics_rms_joint[j])
            for j in range(7)
        },
        "hybrid_joint_rms_nm": {
            f"joint_{j+1}": float(hybrid_rms_joint[j])
            for j in range(7)
        },
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    np.savez(
        OUTPUT_MODEL,
        weights=weights,
        feature_mean=mean,
        feature_std=std,
        ridge_alpha=RIDGE_ALPHA,
    )

    print("Phase 6 Residual Learning")
    print("=========================")
    print(f"Training samples:       {len(x_train)}")
    print(f"Writing test samples:   {len(x_test)}")
    print(f"Feature count:          {x_train.shape[1]}")
    print()
    print(
        f"Physics-only RMS norm:  "
        f"{physics_rms_norm:.6f} N*m"
    )
    print(
        f"Hybrid RMS norm:        "
        f"{hybrid_rms_norm:.6f} N*m"
    )
    print(
        f"Improvement:            "
        f"{improvement_percent:.2f}%"
    )
    print()
    print("Per-joint physics -> hybrid RMS:")

    for j in range(7):
        print(
            f"  J{j+1}: "
            f"{physics_rms_joint[j]:.6f} -> "
            f"{hybrid_rms_joint[j]:.6f} N*m"
        )

    print()
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(f"Model:   {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
