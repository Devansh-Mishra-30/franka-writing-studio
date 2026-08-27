"""Interchangeable end-effector and process-tool definitions."""

from .base import ToolSpec
from .pen import PEN_TOOL

__all__ = [
    "ToolSpec",
    "PEN_TOOL",
]
