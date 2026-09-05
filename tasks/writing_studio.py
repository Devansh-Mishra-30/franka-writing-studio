"""Writing Studio task lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiment_config import ExperimentConfig
from experiments.writing_experiment import (
    INITIAL_JOINT_POSITIONS_RAD,
    ExperimentResult,
    ExperimentStopped,
    WritingExperiment,
    WritingPlanningResult,
)
from pen_tip_kinematics import (
    PEN_TIP_OFFSET_TOOL_M,
    pen_tip_position,
)
from simulation.pybullet_adapter import (
    PyBulletAdapter,
    PyBulletSettings,
)
from tasks.base import TaskStatus
from workcells import WRITING_STUDIO


@dataclass(frozen=True)
class WritingValidationReport:
    """Pre-execution validation results for the Writing Studio."""

    trajectory_finite: bool
    within_notebook_bounds: bool
    sampled_reachability: bool
    sampled_collision_free: bool
    checked_waypoints: int
    failed_waypoint_index: int | None
    collision_waypoint_index: int | None
    max_position_error_m: float
    max_orientation_error_rad: float

    @property
    def success(self) -> bool:
        return (
            self.trajectory_finite
            and self.within_notebook_bounds
            and self.sampled_reachability
            and self.sampled_collision_free
        )


class WritingStudioTask:
    """Backend task coordinating the Writing Studio workflow."""

    def __init__(
        self,
        config: ExperimentConfig,
        telemetry_callback: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        telemetry_period_s: float = 0.05,
    ) -> None:
        self._experiment = WritingExperiment(
            config,
            telemetry_callback=telemetry_callback,
            telemetry_period_s=telemetry_period_s,
        )
        self._status = TaskStatus.IDLE
        self._planning_result: WritingPlanningResult | None = None
        self._validation_report: WritingValidationReport | None = None

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

    @property
    def validation_report(
        self,
    ) -> WritingValidationReport | None:
        return self._validation_report

    def plan(self) -> WritingPlanningResult:
        """Create the writing trajectory without starting simulation."""

        self._status = TaskStatus.PLANNING
        self._validation_report = None

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

    def validate(
        self,
        *,
        maximum_ik_samples: int = 24,
    ) -> WritingValidationReport:
        """Validate geometry and sampled model-based reachability."""

        if self._planning_result is None:
            raise RuntimeError(
                "Task must be planned before validation"
            )

        if maximum_ik_samples <= 0:
            raise ValueError(
                "maximum_ik_samples must be positive"
            )

        planning = self._planning_result
        points = np.asarray(
            planning.pen_tip_waypoints_m,
            dtype=float,
        )

        trajectory_finite = bool(
            points.ndim == 2
            and points.shape[1] == 3
            and len(points) > 0
            and np.all(np.isfinite(points))
        )

        notebook = WRITING_STUDIO.notebook

        half_x = 0.5 * notebook.size_m[0]
        half_y = 0.5 * notebook.size_m[1]

        center_x = notebook.center_m[0]
        center_y = notebook.center_m[1]

        geometry_tolerance_m = 1e-9

        within_notebook_bounds = bool(
            trajectory_finite
            and np.all(
                points[:, 0]
                >= center_x - half_x - geometry_tolerance_m
            )
            and np.all(
                points[:, 0]
                <= center_x + half_x + geometry_tolerance_m
            )
            and np.all(
                points[:, 1]
                >= center_y - half_y - geometry_tolerance_m
            )
            and np.all(
                points[:, 1]
                <= center_y + half_y + geometry_tolerance_m
            )
        )

        sampled_reachability = trajectory_finite
        sampled_collision_free = trajectory_finite
        failed_waypoint_index: int | None = None
        collision_waypoint_index: int | None = None
        max_position_error_m = 0.0
        max_orientation_error_rad = 0.0
        checked_waypoints = 0

        if trajectory_finite:
            mechanics = self._experiment.mechanics

            desired_rotation = mechanics.get_tool_rotation(
                INITIAL_JOINT_POSITIONS_RAD
            )

            pen_offset_world_m = (
                desired_rotation
                @ PEN_TIP_OFFSET_TOOL_M
            )

            sample_count = min(
                maximum_ik_samples,
                len(points),
            )

            sample_indices = np.unique(
                np.linspace(
                    0,
                    len(points) - 1,
                    sample_count,
                    dtype=int,
                )
            )

            simulator = PyBulletAdapter(
                PyBulletSettings(
                    mode="direct",
                    timestep_s=0.001,
                    output_dir=Path(
                        "/tmp/writing_studio_validation"
                    ),
                    desk_geometry=WRITING_STUDIO.desk,
                    writing_surface_geometry=(
                        WRITING_STUDIO.notebook
                    ),
                )
            )

            q_seed = INITIAL_JOINT_POSITIONS_RAD.copy()

            try:
                simulator.configure()

                for waypoint_index in sample_indices:
                    target_pen_tip_m = points[
                        waypoint_index
                    ]

                    target_tool_position_m = (
                        target_pen_tip_m
                        - pen_offset_world_m
                    )

                    ik_result = mechanics.solve_pose_ik(
                        q_seed,
                        target_tool_position_m,
                        desired_rotation,
                    )

                    checked_waypoints += 1

                    max_position_error_m = max(
                        max_position_error_m,
                        ik_result.position_error_m,
                    )

                    max_orientation_error_rad = max(
                        max_orientation_error_rad,
                        ik_result.orientation_error_rad,
                    )

                    if (
                        not ik_result.success
                        or not ik_result.within_joint_limits
                    ):
                        sampled_reachability = False
                        failed_waypoint_index = int(
                            waypoint_index
                        )
                        break

                    q_seed = ik_result.q

                    simulator.reset(q_seed)

                    if simulator.has_robot_table_collision():
                        sampled_collision_free = False
                        collision_waypoint_index = int(
                            waypoint_index
                        )
                        break

            finally:
                simulator.shutdown()

        self._validation_report = WritingValidationReport(
            trajectory_finite=trajectory_finite,
            within_notebook_bounds=within_notebook_bounds,
            sampled_reachability=sampled_reachability,
            sampled_collision_free=sampled_collision_free,
            checked_waypoints=checked_waypoints,
            failed_waypoint_index=failed_waypoint_index,
            collision_waypoint_index=collision_waypoint_index,
            max_position_error_m=max_position_error_m,
            max_orientation_error_rad=(
                max_orientation_error_rad
            ),
        )

        if not self._validation_report.success:
            self._status = TaskStatus.FAILED
        else:
            self._status = TaskStatus.READY

        return self._validation_report

    def run(self) -> ExperimentResult:
        """Execute a successfully planned and validated task."""

        if self._planning_result is None:
            raise RuntimeError(
                "Task must be planned before running"
            )

        if self._validation_report is None:
            raise RuntimeError(
                "Task must be validated before running"
            )

        if not self._validation_report.success:
            self._status = TaskStatus.FAILED
            raise RuntimeError(
                "Task validation failed; execution is blocked"
            )

        self._status = TaskStatus.RUNNING

        try:
            result = self._experiment.run()

        except ExperimentStopped:
            self._status = TaskStatus.STOPPED
            raise

        except Exception:
            self._status = TaskStatus.FAILED
            raise

        self._status = TaskStatus.COMPLETE
        return result

    def request_stop(self) -> None:
        """Request cooperative interruption of active execution."""

        self._experiment.request_stop()

    def reset(self) -> None:
        """Return the task to its initial lifecycle state."""

        self._planning_result = None
        self._validation_report = None

        self._experiment.clear_stop_request()

        self._status = TaskStatus.IDLE
