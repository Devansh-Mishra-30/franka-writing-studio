"""Backward-compatible import for the official Franka model."""

from franka_mechanics import FrankaMechanics


PandaMechanics = FrankaMechanics

__all__ = [
    "FrankaMechanics",
    "PandaMechanics",
]
