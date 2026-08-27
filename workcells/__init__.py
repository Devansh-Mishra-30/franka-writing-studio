"""Configurable robot workcell definitions."""

from .base import WorkcellSpec
from .writing_studio import (
    BoxGeometry,
    WritingStudioWorkcell,
    WRITING_STUDIO,
)

__all__ = [
    "WorkcellSpec",
    "BoxGeometry",
    "WritingStudioWorkcell",
    "WRITING_STUDIO",
]
