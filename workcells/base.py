"""Shared workcell configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkcellSpec:
    """Description of a simulated robot workcell."""

    name: str
    robot_name: str = "franka"
    tool_name: str | None = None
    objects: tuple[str, ...] = field(default_factory=tuple)
