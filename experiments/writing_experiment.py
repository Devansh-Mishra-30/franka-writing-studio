"""Finite, measurable writing-robot experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from differential_ik import (
    solve_pose_differential_ik,
)
from experiment_config import ExperimentConfig
from logging_utils import (
    prepare_output_directory,
    write_csv,
    write_json,
)
from metrics import (
    build_experiment_summary,
    summarize_values,
)
from franka_mechanics import FrankaMechanics
from pen_tip_kinematics import (
    PEN_TIP_OFFSET_TOOL_M,
    legacy_positions_to_pen_tip,
    pen_tip_jacobian,
    pen_tip_position,
)
from simulation.interfaces import VelocityCommand
from simulation.pybullet_adapter import (
    PyBulletAdapter,
    PyBulletSettings,
)
from trajectory import SvgTrajectory
from writing_plan import TimedWritingPlan


INITIAL_JOINT_POSITIONS_RAD = np.array(
    [
        0.23081834,
        -0.69476370,
        -0.12918779,
        -2.02518342,
        -0.08491880,
        1.33535219,
        0.11158610,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ExperimentResult:
    """Files and summary produced by one experiment."""

    samples_path: Path
    summary_path: Path
    summary: dict[str, Any]


class WritingExperiment:
    """Coordinate trajectory, model, control, and simulation."""

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        self.config = config.validate()

        self.mechanics = FrankaMechanics()

        self.trajectory = SvgTrajectory(
            self.config.svg_file
        )

    def run(self) -> ExperimentResult:
        """Run one finite writing experiment."""

        output_dir = prepare_output_directory(
            self.config.output_dir
        )

        samples_path = output_dir / "samples.csv"
        summary_path = output_dir / "summary.json"
        failure_path = output_dir / "failure.json"

        settings = PyBulletSettings(
            mode=self.config.mode,
            timestep_s=self.config.timestep_s,
            output_dir=output_dir,
            record_video=self.config.record_video,
        )

        simulator = PyBulletAdapter(settings)

        rows: list[dict[str, Any]] = []

        simulator_position_errors_m: list[float] = []
        model_position_errors_m: list[float] = []

        model_simulator_disagreements_m: list[
            float
        ] = []

        orientation_errors_rad: list[float] = []

        loop_durations_s: list[float] = []
        ik_durations_s: list[float] = []

        np.random.seed(self.config.seed)

        setup_start = time.perf_counter()
        run_start: float | None = None

        try:
            simulator.configure()

            simulator.reset(
                INITIAL_JOINT_POSITIONS_RAD
            )

            if simulator.has_robot_table_collision():
                raise RuntimeError(
                    "Initial robot configuration "
                    "collides with writing surface"
                )

            initial_state = simulator.read_state(
                0.0
            )

            initial_tool_position_m = np.asarray(
                initial_state.tool_position_m,
                dtype=float,
            )

            initial_tool_rotation = np.asarray(
                initial_state.tool_rotation_matrix,
                dtype=float,
            )

            initial_pen_tip_position_m = (
                pen_tip_position(
                    tool_position_m=(
                        initial_tool_position_m
                    ),
                    tool_rotation=(
                        initial_tool_rotation
                    ),
                )
            )

            desired_rotation = (
                self.mechanics.get_tool_rotation(
                    INITIAL_JOINT_POSITIONS_RAD
                )
            )

            pen_tip_waypoints_m = (
                legacy_positions_to_pen_tip(
                    self.trajectory.positions_m
                )
            )

            write_height_m = float(
                np.min(
                    pen_tip_waypoints_m[:, 2]
                )
            )

            writing_plan = TimedWritingPlan(
                initial_position_m=(
                    initial_pen_tip_position_m
                ),
                svg_positions_m=(
                    pen_tip_waypoints_m
                ),
                write_height_m=write_height_m,
            )

            setup_duration_s = (
                time.perf_counter()
                - setup_start
            )

            run_start = time.perf_counter()

            for step_index in range(
                self.config.num_steps
            ):
                cycle_start = (
                    time.perf_counter()
                )

                simulation_time_s = (
                    step_index
                    * self.config.timestep_s
                )

                state = simulator.read_state(
                    simulation_time_s
                )

                q = np.asarray(
                    state.joint_positions_rad,
                    dtype=float,
                )

                dq = np.asarray(
                    state.joint_velocities_rad_s,
                    dtype=float,
                )

                writing_setpoint = (
                    writing_plan.sample(
                        simulation_time_s
                    )
                )

                desired_position_m = np.asarray(
                    writing_setpoint.position_m,
                    dtype=float,
                )

                desired_velocity_m_s = np.asarray(
                    writing_setpoint.velocity_m_s,
                    dtype=float,
                )

                # ------------------------------
                # PyBullet tool + pen-tip state
                # ------------------------------

                simulator_tool_position_m = (
                    np.asarray(
                        state.tool_position_m,
                        dtype=float,
                    )
                )

                simulator_tool_rotation = (
                    np.asarray(
                        state.tool_rotation_matrix,
                        dtype=float,
                    )
                )

                simulator_pen_tip_position_m = (
                    pen_tip_position(
                        tool_position_m=(
                            simulator_tool_position_m
                        ),
                        tool_rotation=(
                            simulator_tool_rotation
                        ),
                    )
                )

                # ------------------------------
                # Pinocchio tool + pen-tip state
                # ------------------------------

                model_pose = np.asarray(
                    self.mechanics.solve_fk(q),
                    dtype=float,
                )

                model_tool_position_m = (
                    model_pose[:3]
                )

                model_tool_rotation = (
                    self.mechanics.get_tool_rotation(
                        q
                    )
                )

                model_pen_tip_position_m = (
                    pen_tip_position(
                        tool_position_m=(
                            model_tool_position_m
                        ),
                        tool_rotation=(
                            model_tool_rotation
                        ),
                    )
                )

                # ------------------------------
                # Pen-tip Jacobian
                # ------------------------------

                tool_jacobian = (
                    self.mechanics.get_jacobian(
                        q
                    )
                )

                tip_jacobian = (
                    pen_tip_jacobian(
                        tool_jacobian=(
                            tool_jacobian
                        ),
                        tool_rotation=(
                            model_tool_rotation
                        ),
                    )
                )

                # ------------------------------
                # 6D resolved-rate IK
                # ------------------------------

                ik_start = time.perf_counter()

                differential_ik_result = (
                    solve_pose_differential_ik(
                        jacobian=tip_jacobian,
                        current_position_m=(
                            simulator_pen_tip_position_m
                        ),
                        desired_position_m=(
                            desired_position_m
                        ),
                        desired_linear_velocity_m_s=(
                            desired_velocity_m_s
                        ),
                        current_rotation=(
                            simulator_tool_rotation
                        ),
                        desired_rotation=(
                            desired_rotation
                        ),
                        desired_angular_velocity_rad_s=(
                            np.zeros(3)
                        ),
                        position_gain_s_inv=4.0,
                        orientation_gain_s_inv=4.0,
                        damping=0.02,
                    )
                )

                velocity_command = np.asarray(
                    differential_ik_result
                    .joint_velocity_rad_s,
                    dtype=float,
                )

                velocity_limits = np.asarray(
                    self.mechanics
                    .joint_velocity_limits,
                    dtype=float,
                )

                velocity_command = np.clip(
                    velocity_command,
                    -velocity_limits,
                    velocity_limits,
                )

                ik_duration_s = (
                    time.perf_counter()
                    - ik_start
                )

                # ------------------------------
                # Metrics
                # ------------------------------

                simulator_error_m = float(
                    np.linalg.norm(
                        simulator_pen_tip_position_m
                        - desired_position_m
                    )
                )

                model_error_m = float(
                    np.linalg.norm(
                        model_pen_tip_position_m
                        - desired_position_m
                    )
                )

                model_simulator_disagreement_m = (
                    float(
                        np.linalg.norm(
                            model_pen_tip_position_m
                            - simulator_pen_tip_position_m
                        )
                    )
                )

                orientation_error_rad = float(
                    np.linalg.norm(
                        differential_ik_result
                        .orientation_error_rad
                    )
                )

                simulator.apply_velocity_command(
                    VelocityCommand(
                        joint_velocities_rad_s=(
                            velocity_command
                        )
                    )
                )

                simulator.step()

                loop_duration_s = (
                    time.perf_counter()
                    - cycle_start
                )

                sleep_duration_s = 0.0

                if self.config.realtime:
                    sleep_duration_s = max(
                        0.0,
                        self.config.timestep_s
                        - loop_duration_s,
                    )

                    if sleep_duration_s > 0.0:
                        time.sleep(
                            sleep_duration_s
                        )

                # ------------------------------
                # Logging
                # ------------------------------

                row: dict[str, Any] = {
                    "step": step_index,
                    "simulation_time_s": (
                        simulation_time_s
                    ),
                    "writing_phase": (
                        writing_setpoint.phase.name
                    ),
                    "writing_segment_index": (
                        writing_setpoint.segment_index
                    ),

                    "desired_pen_x_m": float(
                        desired_position_m[0]
                    ),
                    "desired_pen_y_m": float(
                        desired_position_m[1]
                    ),
                    "desired_pen_z_m": float(
                        desired_position_m[2]
                    ),

                    "desired_pen_vx_m_s": float(
                        desired_velocity_m_s[0]
                    ),
                    "desired_pen_vy_m_s": float(
                        desired_velocity_m_s[1]
                    ),
                    "desired_pen_vz_m_s": float(
                        desired_velocity_m_s[2]
                    ),

                    "simulator_pen_x_m": float(
                        simulator_pen_tip_position_m[0]
                    ),
                    "simulator_pen_y_m": float(
                        simulator_pen_tip_position_m[1]
                    ),
                    "simulator_pen_z_m": float(
                        simulator_pen_tip_position_m[2]
                    ),

                    "model_pen_x_m": float(
                        model_pen_tip_position_m[0]
                    ),
                    "model_pen_y_m": float(
                        model_pen_tip_position_m[1]
                    ),
                    "model_pen_z_m": float(
                        model_pen_tip_position_m[2]
                    ),

                    "tool_x_m": float(
                        simulator_tool_position_m[0]
                    ),
                    "tool_y_m": float(
                        simulator_tool_position_m[1]
                    ),
                    "tool_z_m": float(
                        simulator_tool_position_m[2]
                    ),

                    "simulator_position_error_m": (
                        simulator_error_m
                    ),
                    "model_position_error_m": (
                        model_error_m
                    ),
                    "model_simulator_disagreement_m": (
                        model_simulator_disagreement_m
                    ),

                    "orientation_error_rad": (
                        orientation_error_rad
                    ),

                    "ik_duration_s": (
                        ik_duration_s
                    ),
                    "loop_duration_s": (
                        loop_duration_s
                    ),
                    "sleep_duration_s": (
                        sleep_duration_s
                    ),
                }

                for joint_index in range(
                    simulator.joint_count
                ):
                    joint_number = (
                        joint_index + 1
                    )

                    row[
                        f"q{joint_number}_rad"
                    ] = float(
                        q[joint_index]
                    )

                    row[
                        f"dq{joint_number}_rad_s"
                    ] = float(
                        dq[joint_index]
                    )

                    row[
                        f"dq{joint_number}_command_rad_s"
                    ] = float(
                        velocity_command[
                            joint_index
                        ]
                    )

                rows.append(row)

                simulator_position_errors_m.append(
                    simulator_error_m
                )

                model_position_errors_m.append(
                    model_error_m
                )

                model_simulator_disagreements_m.append(
                    model_simulator_disagreement_m
                )

                orientation_errors_rad.append(
                    orientation_error_rad
                )

                loop_durations_s.append(
                    loop_duration_s
                )

                ik_durations_s.append(
                    ik_duration_s
                )

            wall_duration_s = (
                time.perf_counter()
                - run_start
            )

            summary = build_experiment_summary(
                position_errors_m=(
                    simulator_position_errors_m
                ),
                joint_errors_rad=None,
                loop_durations_s=(
                    loop_durations_s
                ),
                ik_durations_s=(
                    ik_durations_s
                ),
                timestep_s=(
                    self.config.timestep_s
                ),
                wall_duration_s=(
                    wall_duration_s
                ),
            )

            deadline_miss_count = sum(
                duration
                > self.config.timestep_s
                for duration in loop_durations_s
            )

            summary["status"] = "completed"

            summary["config"] = (
                self.config.to_dict()
            )

            summary["trajectory"] = (
                self.trajectory.metadata()
            )

            summary["writing_plan"] = {
                "type": (
                    "minimum_jerk_pen_tip_plan"
                ),
                "number_of_segments": len(
                    writing_plan.segments
                ),
                "total_duration_s": float(
                    writing_plan.total_duration_s
                ),
                "write_height_m": (
                    write_height_m
                ),
                "initial_pen_tip_position_m": (
                    initial_pen_tip_position_m
                    .tolist()
                ),
            }

            summary["pen_tip"] = {
                "offset_tool_m": (
                    PEN_TIP_OFFSET_TOOL_M
                    .tolist()
                ),
                "controlled_frame": (
                    "virtual_pen_tip"
                ),
                "parent_frame": (
                    "fer_link8"
                ),
            }

            summary["controller"] = {
                "type": (
                    "pose_differential_ik_pen_tip"
                ),
                "position_gain_s_inv": 4.0,
                "orientation_gain_s_inv": 4.0,
                "damping": 0.02,
                "orientation_reference": (
                    "collision_free_home_orientation"
                ),
                "joint_velocity_limits": (
                    "official_urdf"
                ),
            }

            summary[
                "orientation_error_norm_rad"
            ] = summarize_values(
                orientation_errors_rad,
                name="orientation_errors_rad",
            )

            summary["execution"][
                "setup_duration_s"
            ] = float(
                setup_duration_s
            )

            summary["execution"][
                "deadline_miss_count"
            ] = int(
                deadline_miss_count
            )

            summary["execution"][
                "deadline_miss_rate"
            ] = float(
                deadline_miss_count
                / self.config.num_steps
            )

            summary[
                "pinocchio_pen_tip_position_error_m"
            ] = summarize_values(
                model_position_errors_m,
                name=(
                    "model_position_errors_m"
                ),
            )

            summary[
                "pinocchio_pybullet_pen_tip_disagreement_m"
            ] = summarize_values(
                model_simulator_disagreements_m,
                name=(
                    "model_simulator_"
                    "disagreements_m"
                ),
            )

            summary["warnings"] = [
                (
                    "The pen tip is currently a "
                    "virtual fixed-offset task frame; "
                    "explicit pen collision geometry "
                    "has not yet been added."
                ),
                (
                    "Writing-force and surface-contact "
                    "control are not yet enabled."
                ),
            ]

            write_csv(
                samples_path,
                rows,
            )

            write_json(
                summary_path,
                summary,
            )

            if failure_path.exists():
                failure_path.unlink()

            return ExperimentResult(
                samples_path=samples_path,
                summary_path=summary_path,
                summary=summary,
            )

        except Exception as error:
            elapsed_s = (
                time.perf_counter()
                - setup_start
            )

            write_json(
                failure_path,
                {
                    "status": "failed",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(
                        error
                    ),
                    "elapsed_s": float(
                        elapsed_s
                    ),
                    "config": (
                        self.config.to_dict()
                    ),
                },
            )

            raise

        finally:
            simulator.shutdown()
