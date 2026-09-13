"""Differential inverse kinematics for Cartesian tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class DifferentialIKResult:
    """Result of one XYZ resolved-rate IK calculation."""

    joint_velocity_rad_s: np.ndarray
    commanded_cartesian_velocity_m_s: np.ndarray
    position_error_m: np.ndarray


@dataclass(frozen=True)
class PoseDifferentialIKResult:
    """Result of one 6D pose resolved-rate IK calculation."""

    joint_velocity_rad_s: np.ndarray

    commanded_linear_velocity_m_s: np.ndarray
    commanded_angular_velocity_rad_s: np.ndarray

    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray


def _validate_rotation_matrix(
    rotation: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a 3x3 SO(3) rotation matrix."""

    R = np.asarray(rotation, dtype=float)

    if R.shape != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3)"
        )

    if not np.all(np.isfinite(R)):
        raise ValueError(
            f"{name} contains invalid values"
        )

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-6,
    ):
        raise ValueError(
            f"{name} is not orthonormal"
        )

    if not np.isclose(
        np.linalg.det(R),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{name} must have determinant +1"
        )

    return R


def solve_position_differential_ik(
    *,
    jacobian: np.ndarray,
    current_position_m: np.ndarray,
    desired_position_m: np.ndarray,
    desired_velocity_m_s: np.ndarray,
    position_gain_s_inv: float = 4.0,
    damping: float = 0.02,
) -> DifferentialIKResult:
    """Compute damped-least-squares joint velocity for XYZ tracking."""

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    current = np.asarray(
        current_position_m,
        dtype=float,
    )

    desired = np.asarray(
        desired_position_m,
        dtype=float,
    )

    desired_velocity = np.asarray(
        desired_velocity_m_s,
        dtype=float,
    )

    if J.shape != (6, 7):
        raise ValueError(
            f"jacobian must have shape (6, 7), got {J.shape}"
        )

    if current.shape != (3,):
        raise ValueError(
            "current_position_m must have shape (3,)"
        )

    if desired.shape != (3,):
        raise ValueError(
            "desired_position_m must have shape (3,)"
        )

    if desired_velocity.shape != (3,):
        raise ValueError(
            "desired_velocity_m_s must have shape (3,)"
        )

    if position_gain_s_inv < 0.0:
        raise ValueError(
            "position_gain_s_inv must be nonnegative"
        )

    if damping <= 0.0:
        raise ValueError(
            "damping must be positive"
        )

    # Pinocchio LOCAL_WORLD_ALIGNED:
    # first 3 rows are linear velocity.
    Jv = J[:3, :]

    position_error = (
        desired - current
    )

    commanded_velocity = (
        desired_velocity
        + position_gain_s_inv
        * position_error
    )

    regularized = (
        Jv @ Jv.T
        + (damping**2)
        * np.eye(3)
    )

    joint_velocity = (
        Jv.T
        @ np.linalg.solve(
            regularized,
            commanded_velocity,
        )
    )

    return DifferentialIKResult(
        joint_velocity_rad_s=joint_velocity,
        commanded_cartesian_velocity_m_s=(
            commanded_velocity
        ),
        position_error_m=position_error,
    )


def solve_pose_differential_ik(
    *,
    jacobian: np.ndarray,
    current_position_m: np.ndarray,
    desired_position_m: np.ndarray,
    desired_linear_velocity_m_s: np.ndarray,
    current_rotation: np.ndarray,
    desired_rotation: np.ndarray,
    desired_angular_velocity_rad_s: np.ndarray | None = None,
    position_gain_s_inv: float = 4.0,
    orientation_gain_s_inv: float = 4.0,
    damping: float = 0.02,
) -> PoseDifferentialIKResult:
    """Compute damped-least-squares joint velocity for 6D pose tracking."""

    J = np.asarray(
        jacobian,
        dtype=float,
    )

    current_position = np.asarray(
        current_position_m,
        dtype=float,
    )

    desired_position = np.asarray(
        desired_position_m,
        dtype=float,
    )

    desired_linear_velocity = np.asarray(
        desired_linear_velocity_m_s,
        dtype=float,
    )

    current_R = _validate_rotation_matrix(
        current_rotation,
        name="current_rotation",
    )

    desired_R = _validate_rotation_matrix(
        desired_rotation,
        name="desired_rotation",
    )

    if desired_angular_velocity_rad_s is None:
        desired_angular_velocity = np.zeros(
            3,
            dtype=float,
        )
    else:
        desired_angular_velocity = np.asarray(
            desired_angular_velocity_rad_s,
            dtype=float,
        )

    if J.shape != (6, 7):
        raise ValueError(
            f"jacobian must have shape (6, 7), got {J.shape}"
        )

    if current_position.shape != (3,):
        raise ValueError(
            "current_position_m must have shape (3,)"
        )

    if desired_position.shape != (3,):
        raise ValueError(
            "desired_position_m must have shape (3,)"
        )

    if desired_linear_velocity.shape != (3,):
        raise ValueError(
            "desired_linear_velocity_m_s must have shape (3,)"
        )

    if desired_angular_velocity.shape != (3,):
        raise ValueError(
            "desired_angular_velocity_rad_s must have shape (3,)"
        )

    if position_gain_s_inv < 0.0:
        raise ValueError(
            "position_gain_s_inv must be nonnegative"
        )

    if orientation_gain_s_inv < 0.0:
        raise ValueError(
            "orientation_gain_s_inv must be nonnegative"
        )

    if damping <= 0.0:
        raise ValueError(
            "damping must be positive"
        )

    position_error = (
        desired_position
        - current_position
    )

    # World-frame orientation error:
    #
    # R_error maps the current orientation
    # toward the desired orientation.
    R_error = (
        desired_R
        @ current_R.T
    )

    orientation_error = np.asarray(
        pin.log3(R_error),
        dtype=float,
    ).reshape(3)

    commanded_linear_velocity = (
        desired_linear_velocity
        + position_gain_s_inv
        * position_error
    )

    commanded_angular_velocity = (
        desired_angular_velocity
        + orientation_gain_s_inv
        * orientation_error
    )

    commanded_twist = np.concatenate(
        (
            commanded_linear_velocity,
            commanded_angular_velocity,
        )
    )

    regularized = (
        J @ J.T
        + (damping**2)
        * np.eye(6)
    )

    joint_velocity = (
        J.T
        @ np.linalg.solve(
            regularized,
            commanded_twist,
        )
    )

    return PoseDifferentialIKResult(
        joint_velocity_rad_s=joint_velocity,
        commanded_linear_velocity_m_s=(
            commanded_linear_velocity
        ),
        commanded_angular_velocity_rad_s=(
            commanded_angular_velocity
        ),
        position_error_m=position_error,
        orientation_error_rad=(
            orientation_error
        ),
    )
