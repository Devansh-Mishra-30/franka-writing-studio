"""PyBullet adapter for the official Franka FER and Franka Hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from simulation.interfaces import (
    RobotState,
    VelocityCommand,
)


OFFICIAL_FRANKA_HAND_URDF = Path(
    "urdfs/official_franka/fer_franka_hand_pybullet.urdf"
)

ARM_JOINT_NAMES = tuple(
    f"fer_joint{index}"
    for index in range(1, 8)
)

FINGER_JOINT_NAMES = (
    "fer_finger_joint1",
    "fer_finger_joint2",
)

TOOL_FRAME_NAME = "fer_link8"


@dataclass(frozen=True)
class PyBulletSettings:
    mode: str
    timestep_s: float
    output_dir: Path
    record_video: bool = False


class PyBulletAdapter:
    """Simulator adapter using explicit official names."""

    def __init__(
        self,
        settings: PyBulletSettings,
    ) -> None:
        self.settings = settings

        self.client_id: int | None = None
        self.robot_id: int | None = None
        self.table_id: int | None = None
        self.plane_id: int | None = None
        self.video_log_id: int | None = None

        self.joint_indices: tuple[int, ...] = ()
        self.finger_joint_indices: tuple[int, ...] = ()
        self.arm_effort_limits: tuple[float, ...] = ()
        self.arm_velocity_limits: tuple[float, ...] = ()

        self.tool_link_index: int | None = None
        self.tool_frame_name = TOOL_FRAME_NAME

    @property
    def joint_count(self) -> int:
        return len(self.joint_indices)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return ARM_JOINT_NAMES

    def configure(self) -> None:
        if self.client_id is not None:
            raise RuntimeError(
                "PyBullet adapter is already configured"
            )

        if not OFFICIAL_FRANKA_HAND_URDF.is_file():
            raise FileNotFoundError(
                "Official Franka Hand URDF is missing: "
                f"{OFFICIAL_FRANKA_HAND_URDF}"
            )

        if self.settings.mode == "direct":
            connection_mode = p.DIRECT
        elif self.settings.mode == "gui":
            connection_mode = p.GUI
        else:
            raise ValueError(
                f"Unsupported PyBullet mode: "
                f"{self.settings.mode!r}"
            )

        client_id = p.connect(connection_mode)

        if client_id < 0:
            raise RuntimeError(
                "Failed to connect to PyBullet"
            )

        self.client_id = client_id

        try:
            p.resetSimulation(
                physicsClientId=client_id
            )

            p.setAdditionalSearchPath(
                pybullet_data.getDataPath(),
                physicsClientId=client_id,
            )

            p.setRealTimeSimulation(
                0,
                physicsClientId=client_id,
            )

            p.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=client_id,
            )

            p.setGravity(
                0.0,
                0.0,
                -9.81,
                physicsClientId=client_id,
            )

            p.setTimeStep(
                self.settings.timestep_s,
                physicsClientId=client_id,
            )

            orientation = p.getQuaternionFromEuler(
                [0.0, 0.0, 0.0]
            )

            self.plane_id = p.loadURDF(
                "plane.urdf",
                physicsClientId=client_id,
            )

            self.robot_id = p.loadURDF(
                str(OFFICIAL_FRANKA_HAND_URDF),
                [0.0, 0.0, 0.0],
                orientation,
                useFixedBase=True,
                flags=0,
                physicsClientId=client_id,
            )

            self.table_id = p.loadURDF(
                "urdfs/writing_surface_pybullet.urdf",
                [0.45, 0.0, 0.0],
                orientation,
                useFixedBase=True,
                physicsClientId=client_id,
            )

            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=-1,
                lateralFriction=0.1,
                physicsClientId=client_id,
            )

            self._discover_named_structure()

            if self.settings.mode == "gui":
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.0,
                    cameraYaw=-20.0,
                    cameraPitch=-40.0,
                    cameraTargetPosition=[
                        0.0,
                        0.0,
                        0.6,
                    ],
                    physicsClientId=client_id,
                )

            if self.settings.record_video:
                self.settings.output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                self.video_log_id = p.startStateLogging(
                    p.STATE_LOGGING_VIDEO_MP4,
                    str(
                        self.settings.output_dir
                        / "pybullet_recording.mp4"
                    ),
                    physicsClientId=client_id,
                )

        except Exception:
            self.shutdown()
            raise

    def _discover_named_structure(self) -> None:
        client_id, robot_id = self._require_robot()

        joint_index_by_name: dict[str, int] = {}
        link_index_by_name: dict[str, int] = {}
        joint_type_by_name: dict[str, int] = {}
        effort_limit_by_name: dict[str, float] = {}
        velocity_limit_by_name: dict[str, float] = {}

        total_joints = p.getNumJoints(
            robot_id,
            physicsClientId=client_id,
        )

        for joint_index in range(total_joints):
            info = p.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )

            joint_name = info[1].decode("utf-8")
            child_link_name = info[12].decode("utf-8")

            joint_index_by_name[joint_name] = (
                joint_index
            )
            link_index_by_name[child_link_name] = (
                joint_index
            )
            joint_type_by_name[joint_name] = info[2]
            effort_limit_by_name[joint_name] = float(
                info[10]
            )
            velocity_limit_by_name[joint_name] = float(
                info[11]
            )

        missing_joints = [
            name
            for name in (
                *ARM_JOINT_NAMES,
                *FINGER_JOINT_NAMES,
            )
            if name not in joint_index_by_name
        ]

        if missing_joints:
            raise RuntimeError(
                "Official Franka joints are missing: "
                f"{missing_joints}"
            )

        if TOOL_FRAME_NAME not in link_index_by_name:
            raise RuntimeError(
                f"Official tool frame "
                f"{TOOL_FRAME_NAME!r} is missing"
            )

        for name in ARM_JOINT_NAMES:
            if (
                joint_type_by_name[name]
                != p.JOINT_REVOLUTE
            ):
                raise RuntimeError(
                    f"{name} is not revolute"
                )

        for name in FINGER_JOINT_NAMES:
            if (
                joint_type_by_name[name]
                != p.JOINT_PRISMATIC
            ):
                raise RuntimeError(
                    f"{name} is not prismatic"
                )

        self.joint_indices = tuple(
            joint_index_by_name[name]
            for name in ARM_JOINT_NAMES
        )

        self.finger_joint_indices = tuple(
            joint_index_by_name[name]
            for name in FINGER_JOINT_NAMES
        )

        self.arm_effort_limits = tuple(
            effort_limit_by_name[name]
            for name in ARM_JOINT_NAMES
        )

        self.arm_velocity_limits = tuple(
            velocity_limit_by_name[name]
            for name in ARM_JOINT_NAMES
        )

        self.tool_link_index = (
            link_index_by_name[TOOL_FRAME_NAME]
        )

    def reset(
        self,
        joint_positions_rad: np.ndarray,
    ) -> None:
        client_id, robot_id = self._require_robot()

        joint_positions = np.asarray(
            joint_positions_rad,
            dtype=float,
        )

        if joint_positions.shape != (
            self.joint_count,
        ):
            raise ValueError(
                "joint_positions_rad has incorrect shape"
            )

        if not np.all(np.isfinite(joint_positions)):
            raise ValueError(
                "joint_positions_rad contains "
                "invalid values"
            )

        for joint_index, position in zip(
            self.joint_indices,
            joint_positions,
        ):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                targetValue=float(position),
                targetVelocity=0.0,
                physicsClientId=client_id,
            )

        finger_position_m = 0.02

        for finger_index in self.finger_joint_indices:
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=finger_index,
                targetValue=finger_position_m,
                targetVelocity=0.0,
                physicsClientId=client_id,
            )

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=list(
                self.finger_joint_indices
            ),
            controlMode=p.POSITION_CONTROL,
            targetPositions=[
                finger_position_m,
                finger_position_m,
            ],
            forces=[20.0, 20.0],
            physicsClientId=client_id,
        )

    def has_robot_table_collision(
        self,
    ) -> bool:
        """Return whether robot geometry contacts the writing surface."""

        client_id, robot_id = self._require_robot()

        if self.table_id is None:
            raise RuntimeError(
                "Writing surface has not been loaded"
            )

        p.performCollisionDetection(
            physicsClientId=client_id
        )

        contacts = p.getContactPoints(
            bodyA=robot_id,
            bodyB=self.table_id,
            physicsClientId=client_id,
        )

        return bool(contacts)

    def read_state(
        self,
        simulation_time_s: float,
    ) -> RobotState:
        client_id, robot_id = self._require_robot()

        if self.tool_link_index is None:
            raise RuntimeError(
                "Tool frame has not been resolved"
            )

        joint_states = p.getJointStates(
            robot_id,
            list(self.joint_indices),
            physicsClientId=client_id,
        )

        joint_positions = np.array(
            [state[0] for state in joint_states],
            dtype=float,
        )

        joint_velocities = np.array(
            [state[1] for state in joint_states],
            dtype=float,
        )

        link_state = p.getLinkState(
            robot_id,
            self.tool_link_index,
            computeLinkVelocity=1,
            computeForwardKinematics=1,
            physicsClientId=client_id,
        )

        tool_position = np.asarray(
            link_state[4],
            dtype=float,
        )

        tool_quaternion = link_state[5]

        tool_rotation = np.asarray(
            p.getMatrixFromQuaternion(
                tool_quaternion
            ),
            dtype=float,
        ).reshape(3, 3)

        return RobotState(
            simulation_time_s=float(
                simulation_time_s
            ),
            joint_positions_rad=joint_positions,
            joint_velocities_rad_s=joint_velocities,
            tool_position_m=tool_position,
            tool_rotation_matrix=tool_rotation,
        )

    def apply_velocity_command(
        self,
        command: VelocityCommand,
    ) -> None:
        client_id, robot_id = self._require_robot()

        velocities = np.asarray(
            command.joint_velocities_rad_s,
            dtype=float,
        )

        if velocities.shape != (
            self.joint_count,
        ):
            raise ValueError(
                "velocity command has incorrect shape"
            )

        if not np.all(np.isfinite(velocities)):
            raise ValueError(
                "velocity command contains "
                "invalid values"
            )

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=list(self.joint_indices),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocities.tolist(),
            forces=list(self.arm_effort_limits),
            physicsClientId=client_id,
        )

    def step(self) -> None:
        if self.client_id is None:
            raise RuntimeError(
                "PyBullet adapter is not configured"
            )

        p.stepSimulation(
            physicsClientId=self.client_id
        )

    def shutdown(self) -> None:
        if self.client_id is None:
            return

        client_id = self.client_id

        if self.video_log_id is not None:
            try:
                p.stopStateLogging(
                    self.video_log_id,
                    physicsClientId=client_id,
                )
            finally:
                self.video_log_id = None

        if p.isConnected(
            physicsClientId=client_id
        ):
            p.disconnect(
                physicsClientId=client_id
            )

        self.client_id = None
        self.robot_id = None
        self.table_id = None
        self.plane_id = None
        self.video_log_id = None

        self.joint_indices = ()
        self.finger_joint_indices = ()
        self.arm_effort_limits = ()
        self.arm_velocity_limits = ()

        self.tool_link_index = None

    def _require_robot(
        self,
    ) -> tuple[int, int]:
        if self.client_id is None:
            raise RuntimeError(
                "PyBullet adapter is not configured"
            )

        if self.robot_id is None:
            raise RuntimeError(
                "Robot has not been loaded"
            )

        return self.client_id, self.robot_id
