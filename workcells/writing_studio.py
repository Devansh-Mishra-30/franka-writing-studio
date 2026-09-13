"""Interactive Franka writing workcell definition."""

from __future__ import annotations

from dataclasses import dataclass

from simulation.scene import BoxGeometry
from workcells.base import WorkcellSpec


@dataclass(frozen=True)
class WritingStudioWorkcell(WorkcellSpec):
    """Desk and notebook setup for interactive writing."""

    desk: BoxGeometry = BoxGeometry(
        name="writing_desk",
        size_m=(0.70, 0.70, 0.040),
        center_m=(0.55, 0.0, 0.489),
    )

    notebook: BoxGeometry = BoxGeometry(
        name="notebook",
        size_m=(0.42, 0.30, 0.016),
        center_m=(0.45, 0.0, 0.517),
    )

    pen_holder_position_m: tuple[float, float, float] = (
        0.28,
        -0.24,
        0.535,
    )


WRITING_STUDIO = WritingStudioWorkcell(
    name="writing_studio",
    robot_name="franka",
    tool_name="pen",
    objects=(
        "writing_desk",
        "notebook",
        "pen_holder",
    ),
)
