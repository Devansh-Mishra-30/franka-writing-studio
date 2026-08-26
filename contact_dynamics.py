"""Contact-model parameters and utilities for pen-on-paper writing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PenContactParameters:
    """Physical and control parameters for pen/paper contact."""

    # Writing-surface top plane.
    surface_height_m: float = 0.525

    # Desired normal writing force.
    desired_normal_force_n: float = 1.0

    # Initial open-loop compression used to establish contact.
    preload_depth_m: float = 0.0005

    # Kelvin-Voigt normal contact model:
    #
    # F_n = k * delta + c * delta_dot
    contact_stiffness_n_m: float = 2000.0
    contact_damping_n_s_m: float = 10.0

    # Coulomb friction coefficient used at the pen/paper interface.
    lateral_friction: float = 0.10

    # Force-control safety limits.
    maximum_normal_force_n: float = 5.0

    # Later used by the normal-force/admittance controller.
    force_velocity_gain_m_s_n: float = 0.002

    maximum_force_correction_velocity_m_s: float = 0.005

    @property
    def pybullet_body_contact_stiffness_n_m(
        self,
    ) -> float:
        """Per-body stiffness for symmetric pen/paper compliance.

        Two identical compliant contacts acting in series give:

            k_eff = k_body / 2

        Therefore each Bullet body receives twice the desired
        system-level effective stiffness.
        """

        return (
            2.0
            * self.contact_stiffness_n_m
        )

    @property
    def pybullet_body_contact_damping_n_s_m(
        self,
    ) -> float:
        """Per-body damping for symmetric pen/paper compliance."""

        return (
            2.0
            * self.contact_damping_n_s_m
        )


DEFAULT_PEN_CONTACT_PARAMETERS = (
    PenContactParameters()
)


def kelvin_voigt_normal_force(
    *,
    penetration_m: float,
    penetration_velocity_m_s: float,
    parameters: PenContactParameters = (
        DEFAULT_PEN_CONTACT_PARAMETERS
    ),
) -> float:
    """Predict unilateral normal force with a Kelvin-Voigt model."""

    delta = max(
        0.0,
        float(penetration_m),
    )

    if delta <= 0.0:
        return 0.0

    force_n = (
        parameters.contact_stiffness_n_m
        * delta
        + parameters.contact_damping_n_s_m
        * float(penetration_velocity_m_s)
    )

    return max(
        0.0,
        float(force_n),
    )


def apply_write_preload(
    positions_m: np.ndarray,
    *,
    nominal_write_height_m: float,
    parameters: PenContactParameters = (
        DEFAULT_PEN_CONTACT_PARAMETERS
    ),
    atol: float = 1e-9,
) -> np.ndarray:
    """Lower nominal WRITE waypoints by the configured preload depth."""

    positions = np.asarray(
        positions_m,
        dtype=float,
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions_m must have shape (N, 3)"
        )

    if parameters.preload_depth_m < 0.0:
        raise ValueError(
            "preload_depth_m must be nonnegative"
        )

    result = positions.copy()

    write_mask = np.isclose(
        result[:, 2],
        nominal_write_height_m,
        atol=atol,
        rtol=0.0,
    )

    result[
        write_mask,
        2,
    ] -= parameters.preload_depth_m

    return result
