"""Create PyBullet-compatible runtime URDF derivatives."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROBOT_URDF = Path(
    "urdfs/official_franka/"
    "fer_franka_hand_pybullet.urdf"
)

SURFACE_SOURCE = Path(
    "urdfs/writing_surface.urdf"
)

SURFACE_RUNTIME = Path(
    "urdfs/writing_surface_pybullet.urdf"
)

EPSILON_MASS_KG = "1e-9"
EPSILON_INERTIA_KG_M2 = "1e-12"


def add_epsilon_inertial(
    link: ET.Element,
) -> None:
    """Add negligible inertial data to a frame-only link."""

    inertial = ET.Element("inertial")

    ET.SubElement(
        inertial,
        "origin",
        {
            "xyz": "0 0 0",
            "rpy": "0 0 0",
        },
    )

    ET.SubElement(
        inertial,
        "mass",
        {"value": EPSILON_MASS_KG},
    )

    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": EPSILON_INERTIA_KG_M2,
            "ixy": "0",
            "ixz": "0",
            "iyy": EPSILON_INERTIA_KG_M2,
            "iyz": "0",
            "izz": EPSILON_INERTIA_KG_M2,
        },
    )

    link.insert(0, inertial)


def sanitize(
    path: Path,
    *,
    remove_accelerometers: bool,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"URDF does not exist: {path}"
        )

    backup = path.with_name(
        f"{path.stem}_unsanitized{path.suffix}"
    )

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Created backup: {backup}")

    tree = ET.parse(path)
    root = tree.getroot()

    removed_link_names: set[str] = set()

    if remove_accelerometers:
        for link in root.findall("link"):
            name = link.get("name", "")

            if "_accelerometer_" in name:
                removed_link_names.add(name)

        for joint in list(root.findall("joint")):
            child = joint.find("child")

            if (
                child is not None
                and child.get("link")
                in removed_link_names
            ):
                root.remove(joint)

        for link in list(root.findall("link")):
            if link.get("name") in removed_link_names:
                root.remove(link)

    parent_joint_type: dict[str, str] = {}

    for joint in root.findall("joint"):
        child = joint.find("child")

        if child is not None:
            parent_joint_type[
                child.get("link", "")
            ] = joint.get("type", "")

    added_inertials: list[str] = []

    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue

        name = link.get("name", "")
        parent_type = parent_joint_type.get(name)

        if parent_type not in (None, "fixed"):
            raise RuntimeError(
                "Refusing to add numerical inertial "
                f"to movable link {name!r}"
            )

        add_epsilon_inertial(link)
        added_inertials.append(name)

    ET.indent(tree, space="  ")

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print()
    print(f"Sanitized: {path}")
    print(
        "Removed accelerometer links:",
        len(removed_link_names),
    )
    print(
        "Added frame inertials:",
        len(added_inertials),
    )

    for name in added_inertials:
        print(f"  inertial: {name}")


def main() -> None:
    if not SURFACE_RUNTIME.exists():
        shutil.copy2(
            SURFACE_SOURCE,
            SURFACE_RUNTIME,
        )
        print(
            "Created PyBullet writing-surface derivative"
        )

    sanitize(
        ROBOT_URDF,
        remove_accelerometers=True,
    )

    sanitize(
        SURFACE_RUNTIME,
        remove_accelerometers=False,
    )

    print()
    print("PASS: PyBullet URDF derivatives sanitized")


if __name__ == "__main__":
    main()
