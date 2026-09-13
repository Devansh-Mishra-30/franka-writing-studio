"""Validate the official Franka FER models in Pinocchio and PyBullet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pybullet as p


ARM_URDF = Path("urdfs/official_franka/fer.urdf")
HAND_URDF = Path(
    "urdfs/official_franka/fer_franka_hand.urdf"
)

INITIAL_Q = np.array(
    [
        0.0,
        -0.785,
        0.0,
        -2.355,
        0.0,
        1.57,
        0.785,
    ],
    dtype=float,
)


def validate_files() -> None:
    for path in (ARM_URDF, HAND_URDF):
        if not path.is_file():
            raise FileNotFoundError(
                f"Official Franka model is missing: {path}"
            )

    print("PASS: official URDF files exist")


def validate_pinocchio() -> None:
    model = pin.buildModelFromUrdf(str(ARM_URDF))
    data = model.createData()

    print()
    print("=== PINOCCHIO ARM MODEL ===")
    print("Model name:", model.name)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("njoints:", model.njoints)

    if model.nq != 7:
        raise RuntimeError(
            f"Expected Pinocchio nq=7, found {model.nq}"
        )

    if model.nv != 7:
        raise RuntimeError(
            f"Expected Pinocchio nv=7, found {model.nv}"
        )

    pin.forwardKinematics(
        model,
        data,
        INITIAL_Q,
    )
    pin.updateFramePlacements(
        model,
        data,
    )

    print()
    print("Pinocchio joints:")

    for index, name in enumerate(model.names):
        print(f"{index:2d}: {name}")

    print()
    print("Pinocchio frames:")

    for index, frame in enumerate(model.frames):
        print(f"{index:2d}: {frame.name}")

    print()
    print("PASS: official arm-only FER model loaded")


def validate_pybullet() -> None:
    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Could not connect to PyBullet DIRECT mode"
        )

    try:
        robot_id = p.loadURDF(
            str(HAND_URDF),
            useFixedBase=True,
            physicsClientId=client_id,
        )

        total_joints = p.getNumJoints(
            robot_id,
            physicsClientId=client_id,
        )

        revolute_joints: list[
            tuple[int, str, str]
        ] = []

        prismatic_joints: list[
            tuple[int, str, str]
        ] = []

        fixed_joints: list[
            tuple[int, str, str]
        ] = []

        print()
        print("=== PYBULLET ARM + HAND MODEL ===")
        print("Total joints:", total_joints)
        print()

        for joint_index in range(total_joints):
            info = p.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=client_id,
            )

            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower_limit = float(info[8])
            upper_limit = float(info[9])
            effort_limit = float(info[10])
            velocity_limit = float(info[11])
            child_link = info[12].decode("utf-8")

            record = (
                joint_index,
                joint_name,
                child_link,
            )

            if joint_type == p.JOINT_REVOLUTE:
                type_name = "REVOLUTE"
                revolute_joints.append(record)
            elif joint_type == p.JOINT_PRISMATIC:
                type_name = "PRISMATIC"
                prismatic_joints.append(record)
            elif joint_type == p.JOINT_FIXED:
                type_name = "FIXED"
                fixed_joints.append(record)
            else:
                type_name = f"OTHER({joint_type})"

            print(
                f"{joint_index:2d} | "
                f"{joint_name:32s} | "
                f"{type_name:10s} | "
                f"child={child_link:28s} | "
                f"limits=[{lower_limit:.4f}, "
                f"{upper_limit:.4f}] | "
                f"effort={effort_limit:.2f} | "
                f"velocity={velocity_limit:.4f}"
            )

        print()
        print(
            "Revolute arm joints:",
            len(revolute_joints),
        )
        print(
            "Prismatic finger joints:",
            len(prismatic_joints),
        )
        print(
            "Fixed joints:",
            len(fixed_joints),
        )

        if len(revolute_joints) != 7:
            raise RuntimeError(
                "Expected seven revolute arm joints, "
                f"found {len(revolute_joints)}"
            )

        if len(prismatic_joints) != 2:
            raise RuntimeError(
                "Expected two prismatic finger joints, "
                f"found {len(prismatic_joints)}"
            )

        print()
        print("Arm joints:")

        for record in revolute_joints:
            print(record)

        print()
        print("Finger joints:")

        for record in prismatic_joints:
            print(record)

        print()
        print(
            "PASS: official FER + Franka Hand "
            "model loaded"
        )

    finally:
        p.disconnect(
            physicsClientId=client_id
        )


def main() -> None:
    validate_files()
    validate_pinocchio()
    validate_pybullet()

    print()
    print(
        "PASS: both official Franka models "
        "validated successfully"
    )


if __name__ == "__main__":
    main()
