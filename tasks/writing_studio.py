"""Writing Studio task lifecycle."""

from __future__ import annotations

from experiment_config import ExperimentConfig
from experiments.writing_experiment import (
    INITIAL_JOINT_POSITIONS_RAD,
    WritingExperiment,
    WritingPlanningResult,
)
from pen_tip_kinematics import pen_tip_position
from tasks.base import TaskStatus


class WritingStudioTask:
    """Backend task coordinating the Writing Studio workflow."""

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        self._experiment = WritingExperiment(config)
        self._status = TaskStatus.IDLE
        self._planning_result: WritingPlanningResult | None = None

    @property
    def name(self) -> str:
        return "writing_studio"

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def planning_result(
        self,
    ) -> WritingPlanningResult | None:
        return self._planning_result

    def plan(self) -> WritingPlanningResult:
        """Create the writing trajectory without starting simulation."""

        self._status = TaskStatus.PLANNING

        try:
            mechanics = self._experiment.mechanics

            tool_position_m = mechanics.get_tool_position(
                INITIAL_JOINT_POSITIONS_RAD
            )

            tool_rotation = mechanics.get_tool_rotation(
                INITIAL_JOINT_POSITIONS_RAD
            )

            initial_pen_tip_position_m = pen_tip_position(
                tool_position_m=tool_position_m,
                tool_rotation=tool_rotation,
            )

            self._planning_result = (
                self._experiment.build_plan(
                    initial_pen_tip_position_m
                )
            )

        except Exception:
            self._status = TaskStatus.FAILED
            raise

        self._status = TaskStatus.READY
        return self._planning_result

    def reset(self) -> None:
        """Return the task to its initial lifecycle state."""

        self._planning_result = None
        self._status = TaskStatus.IDLE
