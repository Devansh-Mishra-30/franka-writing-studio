"""Normal-direction force control for pen-on-paper writing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact_dynamics import (
    DEFAULT_PEN_CONTACT_PARAMETERS,
    PenContactParameters,
)


@dataclass(frozen=True)
class NormalForceControlResult:
    """Result of one normal-force controller update."""

    force_error_n: float
    commanded_world_z_velocity_m_s: float
    saturated: bool


def compute_normal_force_velocity(
    *,
    measured_normal_force_n: float,
    parameters: PenContactParameters = (
        DEFAULT_PEN_CONTACT_PARAMETERS
    ),
) -> NormalForceControlResult:
    """Convert force error into a bounded world-Z velocity.

    World +Z points away from the writing surface.

    Therefore:
      force too low  -> negative Z velocity -> push downward
      force too high -> positive Z velocity -> unload upward
    """

    measured_force = float(
        measured_normal_force_n
    )

    if not np.isfinite(measured_force):
        raise ValueError(
            "measured_normal_force_n must be finite"
        )

    if measured_force < 0.0:
        raise ValueError(
            "measured_normal_force_n must be nonnegative"
        )

    force_error_n = (
        parameters.desired_normal_force_n
        - measured_force
    )

    unsaturated_velocity = (
        -parameters.force_velocity_gain_m_s_n
        * force_error_n
    )

    velocity_limit = (
        parameters
        .maximum_force_correction_velocity_m_s
    )

    commanded_velocity = float(
        np.clip(
            unsaturated_velocity,
            -velocity_limit,
            velocity_limit,
        )
    )

    saturated = not np.isclose(
        commanded_velocity,
        unsaturated_velocity,
        atol=1e-15,
        rtol=0.0,
    )

    return NormalForceControlResult(
        force_error_n=float(
            force_error_n
        ),
        commanded_world_z_velocity_m_s=(
            commanded_velocity
        ),
        saturated=bool(
            saturated
        ),
    )
