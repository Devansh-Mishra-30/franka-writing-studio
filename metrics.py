"""Metrics for robot tracking and simulation performance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _validated_vector(
    values: Sequence[float],
    *,
    name: str,
    nonnegative: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values"
        )

    if nonnegative and np.any(array < 0.0):
        raise ValueError(
            f"{name} must contain nonnegative values"
        )

    return array


def summarize_values(
    values: Sequence[float],
    *,
    name: str,
) -> dict[str, float | int]:
    """Return descriptive statistics for scalar measurements."""

    array = _validated_vector(values, name=name)

    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def build_experiment_summary(
    *,
    position_errors_m: Sequence[float],
    joint_errors_rad: Sequence[float],
    loop_durations_s: Sequence[float],
    ik_durations_s: Sequence[float],
    timestep_s: float,
    wall_duration_s: float,
) -> dict[str, Any]:
    """Build the machine-readable summary for one run."""

    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be greater than zero")

    if wall_duration_s <= 0.0:
        raise ValueError(
            "wall_duration_s must be greater than zero"
        )

    steps = len(position_errors_m)

    sequence_lengths = {
        steps,
        len(joint_errors_rad),
        len(loop_durations_s),
        len(ik_durations_s),
    }

    if len(sequence_lengths) != 1:
        raise ValueError(
            "all per-step metric sequences must have equal length"
        )

    if steps == 0:
        raise ValueError(
            "experiment must contain at least one step"
        )

    simulated_duration_s = steps * timestep_s

    return {
        "execution": {
            "steps": steps,
            "timestep_s": float(timestep_s),
            "simulated_duration_s": float(
                simulated_duration_s
            ),
            "wall_duration_s": float(wall_duration_s),
            "real_time_factor": float(
                simulated_duration_s / wall_duration_s
            ),
        },
        "cartesian_position_error_m": summarize_values(
            position_errors_m,
            name="position_errors_m",
        ),
        "joint_position_error_norm_rad": summarize_values(
            joint_errors_rad,
            name="joint_errors_rad",
        ),
        "loop_duration_s": summarize_values(
            loop_durations_s,
            name="loop_durations_s",
        ),
        "ik_duration_s": summarize_values(
            ik_durations_s,
            name="ik_durations_s",
        ),
    }
