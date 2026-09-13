"""Shared tool definitions for Franka workcells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToolSpec:
    """Physical and geometric description of an attached tool."""

    name: str
    tcp_offset_m: np.ndarray
    mass_kg: float
    urdf_path: str | None = None

    def __post_init__(self) -> None:
        tcp = np.asarray(self.tcp_offset_m, dtype=float)

        if tcp.shape != (3,):
            raise ValueError("tcp_offset_m must have shape (3,)")

        if not np.all(np.isfinite(tcp)):
            raise ValueError("tcp_offset_m must contain finite values")

        if self.mass_kg < 0.0:
            raise ValueError("mass_kg cannot be negative")

        object.__setattr__(
            self,
            "tcp_offset_m",
            tcp.copy(),
        )
