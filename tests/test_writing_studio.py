"""Tests for the interactive writing workcell."""

import pytest

from workcells import WRITING_STUDIO


def test_writing_studio_identity() -> None:
    assert WRITING_STUDIO.name == "writing_studio"
    assert WRITING_STUDIO.robot_name == "franka"
    assert WRITING_STUDIO.tool_name == "pen"


def test_notebook_preserves_validated_contact_height() -> None:
    assert WRITING_STUDIO.notebook.top_height_m == pytest.approx(
        0.525
    )


def test_notebook_sits_on_desk() -> None:
    desk_top = WRITING_STUDIO.desk.top_height_m

    notebook_bottom = (
        WRITING_STUDIO.notebook.center_m[2]
        - 0.5 * WRITING_STUDIO.notebook.size_m[2]
    )

    assert notebook_bottom == pytest.approx(desk_top)
