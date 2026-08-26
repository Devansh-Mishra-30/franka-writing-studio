"""Smooth Cartesian motion profiles for writing execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianSetpoint:
    """Position, velocity, and acceleration at one instant."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray


def minimum_jerk_segment(
    start_m: np.ndarray,
    end_m: np.ndarray,
    time_s: float,
    duration_s: float,
) -> CartesianSetpoint:
    """Sample a quintic minimum-jerk point-to-point motion.

    The trajectory has zero velocity and acceleration at both ends.
    """

    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)

    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(
            "start_m and end_m must both have shape (3,)"
        )

    if not np.all(np.isfinite(start)):
        raise ValueError("start_m contains invalid values")

    if not np.all(np.isfinite(end)):
        raise ValueError("end_m contains invalid values")

    if not np.isfinite(time_s):
        raise ValueError("time_s must be finite")

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(
            "duration_s must be finite and greater than zero"
        )

    t = float(np.clip(time_s, 0.0, duration_s))
    tau = t / duration_s

    # Quintic time scaling:
    #
    # s(0) = 0, s(1) = 1
    # ds/dt = 0 at both boundaries
    # d2s/dt2 = 0 at both boundaries
    s = (
        10.0 * tau**3
        - 15.0 * tau**4
        + 6.0 * tau**5
    )

    ds_dt = (
        30.0 * tau**2
        - 60.0 * tau**3
        + 30.0 * tau**4
    ) / duration_s

    d2s_dt2 = (
        60.0 * tau
        - 180.0 * tau**2
        + 120.0 * tau**3
    ) / (duration_s**2)

    displacement = end - start

    return CartesianSetpoint(
        position_m=start + s * displacement,
        velocity_m_s=ds_dt * displacement,
        acceleration_m_s2=d2s_dt2 * displacement,
    )
