"""Kinematics and dynamics for the official Franka FER model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation


DEFAULT_URDF_PATH = Path(
    "urdfs/official_franka/fer.urdf"
)

DEFAULT_TOOL_FRAME_NAME = "fer_link8"


class FrankaMechanics:
    """Pinocchio model of the official seven-axis Franka FER."""

    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF_PATH,
        tool_frame_name: str = DEFAULT_TOOL_FRAME_NAME,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.tool_frame_name = tool_frame_name

        if not self.urdf_path.is_file():
            raise FileNotFoundError(
                f"Franka URDF does not exist: {self.urdf_path}"
            )

        self.model = pin.buildModelFromUrdf(
            str(self.urdf_path)
        )
        self.data = self.model.createData()

        if self.model.nq != 7:
            raise RuntimeError(
                "Expected seven configuration coordinates, "
                f"found nq={self.model.nq}"
            )

        if self.model.nv != 7:
            raise RuntimeError(
                "Expected seven velocity coordinates, "
                f"found nv={self.model.nv}"
            )

        self.tool_frame_id = self.model.getFrameId(
            self.tool_frame_name
        )

        if self.tool_frame_id >= len(self.model.frames):
            raise RuntimeError(
                f"Frame {self.tool_frame_name!r} "
                "does not exist in the Franka model"
            )

        self.last_ik_diagnostics: dict[str, Any] = {}

    @property
    def joint_position_lower_limits(self) -> np.ndarray:
        return np.asarray(
            self.model.lowerPositionLimit,
            dtype=float,
        ).copy()

    @property
    def joint_position_upper_limits(self) -> np.ndarray:
        return np.asarray(
            self.model.upperPositionLimit,
            dtype=float,
        ).copy()

    @property
    def joint_velocity_limits(self) -> np.ndarray:
        return np.asarray(
            self.model.velocityLimit,
            dtype=float,
        ).copy()

    @property
    def joint_effort_limits(self) -> np.ndarray:
        return np.asarray(
            self.model.effortLimit,
            dtype=float,
        ).copy()

    def _validate_q(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(q, dtype=float)

        if vector.shape != (self.model.nq,):
            raise ValueError(
                f"q must have shape ({self.model.nq},), "
                f"received {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "q contains NaN or infinite values"
            )

        return vector

    def _validate_dq(
        self,
        dq: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(dq, dtype=float)

        if vector.shape != (self.model.nv,):
            raise ValueError(
                f"dq must have shape ({self.model.nv},), "
                f"received {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "dq contains NaN or infinite values"
            )

        return vector

    def _validate_ddq(
        self,
        ddq: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(
            ddq,
            dtype=float,
        )

        if vector.shape != (self.model.nv,):
            raise ValueError(
                f"ddq must have shape ({self.model.nv},), "
                f"received {vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "ddq contains NaN or infinite values"
            )

        return vector

    def solve_fk(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return tool pose as [x, y, z, roll, pitch, yaw]."""

        q = self._validate_q(q)

        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )
        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        transform = self.data.oMf[
            self.tool_frame_id
        ]

        position = np.asarray(
            transform.translation,
            dtype=float,
        ).reshape(3)

        rpy = Rotation.from_matrix(
            np.asarray(transform.rotation)
        ).as_euler(
            "xyz",
            degrees=False,
        )

        return np.concatenate((position, rpy))

    def solve_ik(
        self,
        q: np.ndarray,
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        """Solve legacy iterative IK for the named tool frame.

        This retains the original damped least-squares algorithm while
        replacing the guessed final joint with the explicit fer_link8
        frame. A more robust IK interface will replace this later.
        """

        q_iterate = self._validate_q(q).copy()

        target_values = np.array(
            [x, y, z],
            dtype=float,
        )

        if not np.all(np.isfinite(target_values)):
            raise ValueError(
                "IK target contains invalid values"
            )

        desired_transform = pin.SE3(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=float,
            ),
            target_values,
        )

        tolerance = 1e-4
        max_iterations = 10000
        integration_step = 1e-1
        damping = 1e-12

        success = False
        final_error_norm = float("inf")
        iterations = 0

        for iteration in range(max_iterations + 1):
            pin.forwardKinematics(
                self.model,
                self.data,
                q_iterate,
            )
            pin.updateFramePlacements(
                self.model,
                self.data,
            )

            current_transform = self.data.oMf[
                self.tool_frame_id
            ]

            desired_to_current = (
                desired_transform.actInv(
                    current_transform
                )
            )

            error = pin.log6(
                desired_to_current
            ).vector

            final_error_norm = float(norm(error))
            iterations = iteration

            if final_error_norm < tolerance:
                success = True
                break

            if iteration >= max_iterations:
                break

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_iterate,
                self.tool_frame_id,
                pin.ReferenceFrame.LOCAL,
            )

            velocity = -jacobian.T.dot(
                solve(
                    jacobian.dot(jacobian.T)
                    + damping * np.eye(6),
                    error,
                )
            )

            q_iterate = pin.integrate(
                self.model,
                q_iterate,
                velocity * integration_step,
            )

        self.last_ik_diagnostics = {
            "success": success,
            "iterations": iterations,
            "final_error_norm": final_error_norm,
            "tolerance": tolerance,
            "maximum_iterations": max_iterations,
            "tool_frame": self.tool_frame_name,
        }

        if not success:
            print(
                "[WARNING] IK did not converge: "
                f"iterations={iterations}, "
                f"error={final_error_norm:.6e}"
            )

        return np.asarray(
            q_iterate,
            dtype=float,
        ).reshape(self.model.nq)

    def get_tool_position(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return tool position in the world frame."""

        q = self._validate_q(q)

        pin.forwardKinematics(
            self.model,
            self.data,
            q,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        return (
            self.data.oMf[
                self.tool_frame_id
            ]
            .translation.copy()
        )

    def get_tool_rotation(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return tool orientation in the world frame."""

        q_array = np.asarray(
            q,
            dtype=float,
        )

        if q_array.shape != (7,):
            raise ValueError(
                "q must have shape (7,)"
            )

        if not np.all(np.isfinite(q_array)):
            raise ValueError(
                "q contains invalid values"
            )

        pin.forwardKinematics(
            self.model,
            self.data,
            q_array,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        return (
            self.data.oMf[
                self.tool_frame_id
            ]
            .rotation.copy()
        )

    def get_jacobian(
        self,
        q: np.ndarray,
        *,
        reference_frame: pin.ReferenceFrame = (
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ),
    ) -> np.ndarray:
        """Return the 6x7 Jacobian of the named tool frame."""

        q = self._validate_q(q)

        return np.asarray(
            pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.tool_frame_id,
                reference_frame,
            ),
            dtype=float,
        )

    def get_Jacobian(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible Jacobian method."""

        return self.get_jacobian(q)

    def solve_fk_velocity(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        q = self._validate_q(q)
        dq = self._validate_dq(dq)

        return self.get_jacobian(q) @ dq

    def get_M(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return the symmetric joint-space inertia matrix."""

        q = self._validate_q(q)

        mass_matrix = np.asarray(
            pin.crba(
                self.model,
                self.data,
                q,
            ),
            dtype=float,
        )

        return 0.5 * (
            mass_matrix + mass_matrix.T
        )

    def get_C(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """Return the Coriolis matrix."""

        q = self._validate_q(q)
        dq = self._validate_dq(dq)

        return np.asarray(
            pin.computeCoriolisMatrix(
                self.model,
                self.data,
                q,
                dq,
            ),
            dtype=float,
        )

    def get_G(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Return generalized gravity torques."""

        q = self._validate_q(q)

        return np.asarray(
            pin.computeGeneralizedGravity(
                self.model,
                self.data,
                q,
            ),
            dtype=float,
        )


    def get_tau(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Return inverse-dynamics joint torques using RNEA.

        Computes the generalized torque required for the supplied
        configuration, velocity, and acceleration.
        """

        q = self._validate_q(q)
        dq = self._validate_dq(dq)
        ddq = self._validate_ddq(ddq)

        return np.asarray(
            pin.rnea(
                self.model,
                self.data,
                q,
                dq,
                ddq,
            ),
            dtype=float,
        ).reshape(self.model.nv)

    def get_tau_decomposed(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct inverse dynamics as M(q)ddq + C(q,dq)dq + g(q)."""

        q = self._validate_q(q)
        dq = self._validate_dq(dq)
        ddq = self._validate_ddq(ddq)

        mass_matrix = self.get_M(q)
        coriolis_matrix = self.get_C(
            q,
            dq,
        )
        gravity_torque = self.get_G(q)

        return (
            mass_matrix @ ddq
            + coriolis_matrix @ dq
            + gravity_torque
        )
