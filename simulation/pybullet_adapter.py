"""PyBullet adapter for the official Franka FER and Franka Hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from contact_dynamics import (
    DEFAULT_PEN_CONTACT_PARAMETERS,
)
from pen_tip_kinematics import (
    PEN_TIP_OFFSET_TOOL_M,
)
from simulation.interfaces import (
    PenContactState,
    RobotState,
    VelocityCommand,
)
from workcells import BoxGeometry, WRITING_STUDIO


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

PHYSICAL_PEN_URDF = Path(
    "urdfs/pen_tool_pybullet.urdf"
)

PEN_RADIUS_M = 0.004
PEN_LENGTH_M = float(
    np.linalg.norm(PEN_TIP_OFFSET_TOOL_M)
)

# Panda finger displacement is measured per finger.
# A 4 mm displacement on each side gives an
# approximately 8 mm opening for the 8 mm pen.
PEN_GRIPPER_JOINT_POSITION_M = (
    PEN_RADIUS_M
)


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
        self.desk_id: int | None = None
        self.writing_surface_id: int | None = None
        self.plane_id: int | None = None
        self.video_log_id: int | None = None
        self.pen_id: int | None = None
        self.pen_constraint_id: int | None = None

        self.joint_indices: tuple[int, ...] = ()
        self.finger_joint_indices: tuple[int, ...] = ()
        self.arm_effort_limits: tuple[float, ...] = ()
        self.arm_velocity_limits: tuple[float, ...] = ()

        self.tool_link_index: int | None = None
        self.tool_frame_name = TOOL_FRAME_NAME

    @property
    def table_id(self) -> int | None:
        """Compatibility alias for the notebook contact surface."""
        return self.writing_surface_id

    @property
    def joint_count(self) -> int:
        return len(self.joint_indices)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return ARM_JOINT_NAMES

    def _create_box_body(
        self,
        geometry: BoxGeometry,
        rgba: tuple[float, float, float, float],
    ) -> int:
        """Create a fixed box body from workcell configuration."""

        if self.client_id is None:
            raise RuntimeError(
                "PyBullet client is unavailable"
            )

        half_extents = [
            0.5 * dimension
            for dimension in geometry.size_m
        ]

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )

        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.client_id,
        )

        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=geometry.center_m,
            physicsClientId=self.client_id,
        )

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

        if not PHYSICAL_PEN_URDF.is_file():
            raise FileNotFoundError(
                "Physical pen URDF is missing: "
                f"{PHYSICAL_PEN_URDF}"
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

            self.desk_id = self._create_box_body(
                WRITING_STUDIO.desk,
                rgba=(0.45, 0.25, 0.10, 1.0),
            )

            self.writing_surface_id = self._create_box_body(
                WRITING_STUDIO.notebook,
                rgba=(0.92, 0.92, 0.88, 1.0),
            )

            contact_parameters = (
                DEFAULT_PEN_CONTACT_PARAMETERS
            )

            p.changeDynamics(
                bodyUniqueId=self.writing_surface_id,
                linkIndex=-1,
                lateralFriction=(
                    contact_parameters.lateral_friction
                ),
                restitution=0.0,
                contactStiffness=(
                    contact_parameters
                    .pybullet_body_contact_stiffness_n_m
                ),
                contactDamping=(
                    contact_parameters
                    .pybullet_body_contact_damping_n_s_m
                ),
                physicsClientId=client_id,
            )

            self._discover_named_structure()
            self._create_physical_pen()

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

        finger_position_m = (
            PEN_GRIPPER_JOINT_POSITION_M
        )

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

        self._sync_physical_pen_pose()

    def has_robot_table_collision(
        self,
    ) -> bool:
        """Return whether robot geometry contacts the writing surface."""

        client_id, robot_id = self._require_robot()

        if self.desk_id is None:
            raise RuntimeError(
                "Writing desk has not been loaded"
            )

        if self.writing_surface_id is None:
            raise RuntimeError(
                "Writing surface has not been loaded"
            )

        p.performCollisionDetection(
            physicsClientId=client_id
        )

        for body_id in (
            self.desk_id,
            self.writing_surface_id,
        ):
            contacts = p.getContactPoints(
                bodyA=robot_id,
                bodyB=body_id,
                physicsClientId=client_id,
            )

            if contacts:
                return True

        return False

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
        self.desk_id = None
        self.writing_surface_id = None
        self.plane_id = None
        self.video_log_id = None
        self.pen_id = None
        self.pen_constraint_id = None

        self.joint_indices = ()
        self.finger_joint_indices = ()
        self.arm_effort_limits = ()
        self.arm_velocity_limits = ()

        self.tool_link_index = None

    def _create_physical_pen(self) -> None:
        """Load and rigidly attach the physical pen."""

        client_id, robot_id = self._require_robot()

        if self.pen_id is not None:
            raise RuntimeError(
                "Physical pen has already been created"
            )

        if self.tool_link_index is None:
            raise RuntimeError(
                "Tool frame has not been resolved"
            )

        link_state = p.getLinkState(
            robot_id,
            self.tool_link_index,
            computeForwardKinematics=1,
            physicsClientId=client_id,
        )

        tool_position = link_state[4]
        tool_orientation = link_state[5]

        self.pen_id = p.loadURDF(
            str(PHYSICAL_PEN_URDF),
            basePosition=tool_position,
            baseOrientation=tool_orientation,
            useFixedBase=False,
            physicsClientId=client_id,
        )

        contact_parameters = (
            DEFAULT_PEN_CONTACT_PARAMETERS
        )

        p.changeDynamics(
            self.pen_id,
            -1,
            lateralFriction=(
                contact_parameters.lateral_friction
            ),
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            contactStiffness=(
                contact_parameters
                .pybullet_body_contact_stiffness_n_m
            ),
            contactDamping=(
                contact_parameters
                .pybullet_body_contact_damping_n_s_m
            ),
            physicsClientId=client_id,
        )

        # --------------------------------------------------
        # PyBullet constraint frames are expressed relative
        # to the center-of-mass frames.
        #
        # We want:
        #
        #     pen link origin == fer_link8 origin
        #
        # so convert each URDF link origin into its
        # corresponding COM-frame coordinates.
        # --------------------------------------------------

        parent_dynamics = p.getDynamicsInfo(
            robot_id,
            self.tool_link_index,
            physicsClientId=client_id,
        )

        child_dynamics = p.getDynamicsInfo(
            self.pen_id,
            -1,
            physicsClientId=client_id,
        )

        parent_inertial_position = (
            parent_dynamics[3]
        )
        parent_inertial_orientation = (
            parent_dynamics[4]
        )

        child_inertial_position = (
            child_dynamics[3]
        )
        child_inertial_orientation = (
            child_dynamics[4]
        )

        (
            parent_frame_position,
            parent_frame_orientation,
        ) = p.invertTransform(
            parent_inertial_position,
            parent_inertial_orientation,
        )

        (
            child_frame_position,
            child_frame_orientation,
        ) = p.invertTransform(
            child_inertial_position,
            child_inertial_orientation,
        )

        self.pen_constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.tool_link_index,
            childBodyUniqueId=self.pen_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=(
                parent_frame_position
            ),
            childFramePosition=(
                child_frame_position
            ),
            parentFrameOrientation=(
                parent_frame_orientation
            ),
            childFrameOrientation=(
                child_frame_orientation
            ),
            physicsClientId=client_id,
        )

        p.changeConstraint(
            self.pen_constraint_id,
            maxForce=1000.0,
            physicsClientId=client_id,
        )

        # The pen visually passes between the fingers.
        # Because it is modeled as a rigidly mounted tool,
        # disable pen/robot collision pairs.
        #
        # Pen/table collision remains ENABLED.
        total_robot_joints = p.getNumJoints(
            robot_id,
            physicsClientId=client_id,
        )

        for robot_link_index in range(
            -1,
            total_robot_joints,
        ):
            p.setCollisionFilterPair(
                robot_id,
                self.pen_id,
                robot_link_index,
                -1,
                enableCollision=0,
                physicsClientId=client_id,
            )

        self._sync_physical_pen_pose()

    def _sync_physical_pen_pose(self) -> None:
        """Align the pen link origin exactly with fer_link8.

        This is used only after explicit robot resets.
        During normal simulation the fixed constraint owns
        the pen motion.
        """

        if self.pen_id is None:
            return

        client_id, robot_id = self._require_robot()

        if self.tool_link_index is None:
            raise RuntimeError(
                "Tool frame has not been resolved"
            )

        link_state = p.getLinkState(
            robot_id,
            self.tool_link_index,
            computeForwardKinematics=1,
            physicsClientId=client_id,
        )

        tool_position = link_state[4]
        tool_orientation = link_state[5]

        dynamics = p.getDynamicsInfo(
            self.pen_id,
            -1,
            physicsClientId=client_id,
        )

        local_inertial_position = dynamics[3]
        local_inertial_orientation = dynamics[4]

        # Desired world COM pose:
        #
        # T_WI = T_WL * T_LI
        #
        # where L is the pen link frame and I is its
        # inertial/center-of-mass frame.
        (
            com_position,
            com_orientation,
        ) = p.multiplyTransforms(
            tool_position,
            tool_orientation,
            local_inertial_position,
            local_inertial_orientation,
        )

        p.resetBasePositionAndOrientation(
            self.pen_id,
            com_position,
            com_orientation,
            physicsClientId=client_id,
        )

    def _read_physical_pen_link_pose(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physical pen link pose in world coordinates."""

        if self.pen_id is None:
            raise RuntimeError(
                "Physical pen has not been created"
            )

        if self.client_id is None:
            raise RuntimeError(
                "PyBullet adapter is not configured"
            )

        (
            com_position,
            com_orientation,
        ) = p.getBasePositionAndOrientation(
            self.pen_id,
            physicsClientId=self.client_id,
        )

        dynamics = p.getDynamicsInfo(
            self.pen_id,
            -1,
            physicsClientId=self.client_id,
        )

        local_inertial_position = dynamics[3]
        local_inertial_orientation = dynamics[4]

        (
            inertial_to_link_position,
            inertial_to_link_orientation,
        ) = p.invertTransform(
            local_inertial_position,
            local_inertial_orientation,
        )

        (
            link_position,
            link_orientation,
        ) = p.multiplyTransforms(
            com_position,
            com_orientation,
            inertial_to_link_position,
            inertial_to_link_orientation,
        )

        rotation = np.asarray(
            p.getMatrixFromQuaternion(
                link_orientation
            ),
            dtype=float,
        ).reshape(3, 3)

        return (
            np.asarray(
                link_position,
                dtype=float,
            ),
            rotation,
        )

    def read_pen_tip_position(
        self,
    ) -> np.ndarray:
        """Return physical pen-tip position in world coordinates."""

        (
            pen_position,
            pen_rotation,
        ) = self._read_physical_pen_link_pose()

        return (
            pen_position
            + pen_rotation
            @ PEN_TIP_OFFSET_TOOL_M
        )

    def get_pen_table_contacts(
        self,
    ) -> tuple:
        """Return current physical pen/table contacts."""

        if self.pen_id is None:
            raise RuntimeError(
                "Physical pen has not been created"
            )

        if self.writing_surface_id is None:
            raise RuntimeError(
                "Writing surface has not been loaded"
            )

        if self.client_id is None:
            raise RuntimeError(
                "PyBullet adapter is not configured"
            )

        contacts = p.getContactPoints(
            bodyA=self.pen_id,
            bodyB=self.writing_surface_id,
            physicsClientId=self.client_id,
        )

        return tuple(contacts)

    def read_pen_contact_state(
        self,
    ) -> PenContactState:
        """Return physical pen/writing-surface contact state."""

        contacts = (
            self.get_pen_table_contacts()
        )

        if not contacts:
            return PenContactState(
                active=False,
                contact_count=0,
                minimum_distance_m=None,
                normal_force_n=0.0,
            )

        minimum_distance_m = min(
            float(contact[8])
            for contact in contacts
        )

        normal_force_n = sum(
            max(
                0.0,
                float(contact[9]),
            )
            for contact in contacts
        )

        return PenContactState(
            active=True,
            contact_count=len(contacts),
            minimum_distance_m=float(
                minimum_distance_m
            ),
            normal_force_n=float(
                normal_force_n
            ),
        )

    def has_pen_table_contact(
        self,
    ) -> bool:
        """Return whether the physical pen contacts the table."""

        return bool(
            self.get_pen_table_contacts()
        )

    def read_pen_normal_force_n(
        self,
    ) -> float:
        """Return total normal pen/table contact force."""

        contacts = (
            self.get_pen_table_contacts()
        )

        return float(
            sum(
                max(
                    0.0,
                    float(contact[9]),
                )
                for contact in contacts
            )
        )

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
