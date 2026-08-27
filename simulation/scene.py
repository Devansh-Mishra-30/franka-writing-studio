"""Generic geometry definitions shared by simulated workcells."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoxGeometry:
    """Axis-aligned box geometry in world coordinates."""

    name: str
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]

    @property
    def top_height_m(self) -> float:
        return (
            self.center_m[2]
            + 0.5 * self.size_m[2]
        )
