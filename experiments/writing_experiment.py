"""Finite, measurable writing-robot experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from contact_dynamics import (
    DEFAULT_PEN_CONTACT_PARAMETERS,
    apply_write_preload,
    fit_xy_path_to_surface,
)
from differential_ik import (
    solve_pose_differential_ik,
)
from force_control import (
    compute_normal_force_velocity,
)
from experiment_config import ExperimentConfig
from logging_utils import (
    prepare_output_directory,
    write_csv,
    write_json,
)
from metrics import (
    build_experiment_summary,
    summarize_signed_values,
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
from simulation.live_debug_visualizer import (
    LiveDebugSample,
    LiveWritingDebugVisualizer,
)
from simulation.pybullet_adapter import (
    PyBulletAdapter,
    PyBulletSettings,
)
from trajectory import SvgTrajectory
from writing_plan import TimedWritingPlan
from workcells import WRITING_STUDIO


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
class WritingPlanningResult:
    """Simulator-independent output of writing trajectory planning."""

    contact_parameters: Any
    nominal_pen_tip_waypoints_m: np.ndarray
    pen_tip_waypoints_m: np.ndarray
    nominal_write_height_m: float
    write_height_m: float
    writing_plan: TimedWritingPlan


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
        telemetry_callback: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        telemetry_period_s: float = 0.05,
    ) -> None:
        self.config = config.validate()

        if telemetry_period_s <= 0.0:
            raise ValueError(
                "telemetry_period_s must be positive"
            )

        self.telemetry_callback = telemetry_callback
        self.telemetry_period_s = telemetry_period_s

        self.mechanics = FrankaMechanics()

        self.trajectory = SvgTrajectory(
            self.config.svg_file
        )

    def build_plan(
        self,
        initial_pen_tip_position_m: np.ndarray,
    ) -> WritingPlanningResult:
        """Build the complete writing plan without running simulation."""

        initial_position = np.asarray(
            initial_pen_tip_position_m,
            dtype=float,
        )

        if initial_position.shape != (3,):
            raise ValueError(
                "initial_pen_tip_position_m must have shape (3,)"
            )

        if not np.all(np.isfinite(initial_position)):
            raise ValueError(
                "initial_pen_tip_position_m contains invalid values"
            )

        contact_parameters = replace(
            DEFAULT_PEN_CONTACT_PARAMETERS,
            surface_height_m=(
                WRITING_STUDIO.notebook.top_height_m
            ),
            surface_center_x_m=(
                WRITING_STUDIO.notebook.center_m[0]
            ),
            surface_center_y_m=(
                WRITING_STUDIO.notebook.center_m[1]
            ),
        )

        nominal_pen_tip_waypoints_m = (
            fit_xy_path_to_surface(
                legacy_positions_to_pen_tip(
                    self.trajectory.positions_m
                ),
                surface_size_m=(
                    WRITING_STUDIO.notebook.size_m[:2]
                ),
                margin_m=0.02,
                parameters=contact_parameters,
            )
        )

        nominal_write_height_m = float(
            np.min(
                nominal_pen_tip_waypoints_m[:, 2]
            )
        )

        pen_tip_waypoints_m = apply_write_preload(
            nominal_pen_tip_waypoints_m,
            nominal_write_height_m=(
                nominal_write_height_m
            ),
            parameters=contact_parameters,
        )

        write_height_m = float(
            np.min(
                pen_tip_waypoints_m[:, 2]
            )
        )

        writing_plan = TimedWritingPlan(
            initial_position_m=initial_position,
            svg_positions_m=pen_tip_waypoints_m,
            write_height_m=write_height_m,
            speed_scale=self.config.speed_scale,
        )

        return WritingPlanningResult(
            contact_parameters=contact_parameters,
            nominal_pen_tip_waypoints_m=(
                nominal_pen_tip_waypoints_m
            ),
            pen_tip_waypoints_m=pen_tip_waypoints_m,
            nominal_write_height_m=(
                nominal_write_height_m
            ),
            write_height_m=write_height_m,
            writing_plan=writing_plan,
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
            desk_geometry=WRITING_STUDIO.desk,
            writing_surface_geometry=WRITING_STUDIO.notebook,
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

        pen_normal_forces_n: list[float] = []
        pen_contact_distances_m: list[float] = []
        physical_tip_disagreements_m: list[float] = []

        xy_tracking_errors_m: list[float] = []
        write_force_errors_n: list[float] = []
        force_correction_velocities_m_s: list[float] = []

        force_control_saturation_count = 0

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

            planning = self.build_plan(
                initial_pen_tip_position_m
            )

            contact_parameters = (
                planning.contact_parameters
            )
            nominal_pen_tip_waypoints_m = (
                planning.nominal_pen_tip_waypoints_m
            )
            pen_tip_waypoints_m = (
                planning.pen_tip_waypoints_m
            )
            nominal_write_height_m = (
                planning.nominal_write_height_m
            )
            write_height_m = (
                planning.write_height_m
            )
            writing_plan = planning.writing_plan

            if simulator.client_id is None:
                raise RuntimeError(
                    "PyBullet client is unavailable "
                    "after simulator configuration"
                )

            live_visualizer = (
                LiveWritingDebugVisualizer(
                    client_id=simulator.client_id,
                    enabled=(
                        self.config.mode == "gui"
                    ),
                    surface_height_m=(
                        contact_parameters
                        .surface_height_m
                    ),
                    # 40 Hz visualization while
                    # simulation/control remains
                    # independently timestep-driven.
                    update_period_s=0.05,
                )
            )

            live_visualizer.draw_reference_path(
                pen_tip_waypoints_m,
                total_duration_s=(
                    writing_plan.total_duration_s
                ),
                desired_force_n=(
                    contact_parameters
                    .desired_normal_force_n
                ),
            )

            if self.config.full_plan:
                run_num_steps = (
                    int(
                        np.ceil(
                            writing_plan.total_duration_s
                            / self.config.timestep_s
                        )
                    )
                    + 1
                )
            else:
                run_num_steps = (
                    self.config.num_steps
                )

            setup_duration_s = (
                time.perf_counter()
                - setup_start
            )

            run_start = time.perf_counter()

            next_telemetry_time_s = 0.0

            # Used only for dynamics telemetry.
            # This does not modify the controller.
            previous_dq: np.ndarray | None = None

            for step_index in range(
                run_num_steps
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

                pen_contact_state = (
                    simulator.read_pen_contact_state()
                )

                physical_pen_tip_position_m = (
                    simulator.read_pen_tip_position()
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

                physical_tip_disagreement_m = float(
                    np.linalg.norm(
                        physical_pen_tip_position_m
                        - simulator_pen_tip_position_m
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
                # Hybrid Cartesian control
                #
                # APPROACH / LOWER / LIFT / TRANSFER:
                #   XYZ position + orientation control.
                #
                # WRITE:
                #   XY position control
                #   Z normal-force control
                #   orientation control
                # ------------------------------

                controller_position_m = (
                    desired_position_m.copy()
                )

                controller_velocity_m_s = (
                    desired_velocity_m_s.copy()
                )

                force_control_active = (
                    writing_setpoint.phase.name
                    == "WRITE"
                )

                force_control_result = None

                if force_control_active:
                    measured_force_n = float(
                        pen_contact_state.normal_force_n
                    )

                    if (
                        measured_force_n
                        > contact_parameters
                        .maximum_normal_force_n
                    ):
                        raise RuntimeError(
                            "Pen normal force exceeded "
                            "safety limit: "
                            f"{measured_force_n:.6f} N > "
                            f"{contact_parameters.maximum_normal_force_n:.6f} N"
                        )

                    force_control_result = (
                        compute_normal_force_velocity(
                            measured_normal_force_n=(
                                measured_force_n
                            ),
                            parameters=(
                                contact_parameters
                            ),
                        )
                    )

                    # Remove Z position feedback.
                    #
                    # Setting desired Z to current Z makes:
                    #
                    #     e_z = z_d - z = 0
                    #
                    # so Z motion comes from force control.
                    controller_position_m[2] = (
                        simulator_pen_tip_position_m[2]
                    )

                    controller_velocity_m_s[2] = (
                        force_control_result
                        .commanded_world_z_velocity_m_s
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
                            controller_position_m
                        ),
                        desired_linear_velocity_m_s=(
                            controller_velocity_m_s
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

                xy_tracking_error_m = float(
                    np.linalg.norm(
                        simulator_pen_tip_position_m[:2]
                        - desired_position_m[:2]
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

                force_correction_velocity_m_s = (
                    float(
                        force_control_result
                        .commanded_world_z_velocity_m_s
                    )
                    if force_control_result
                    is not None
                    else 0.0
                )

                live_visualizer.update(
                    LiveDebugSample(
                        simulation_time_s=(
                            simulation_time_s
                        ),
                        phase=(
                            writing_setpoint
                            .phase.name
                        ),
                        desired_position_m=(
                            desired_position_m
                        ),
                        actual_position_m=(
                            physical_pen_tip_position_m
                        ),
                        contact_active=(
                            pen_contact_state.active
                        ),
                        normal_force_n=float(
                            pen_contact_state
                            .normal_force_n
                        ),
                        desired_force_n=float(
                            contact_parameters
                            .desired_normal_force_n
                        ),
                        xy_tracking_error_m=(
                            xy_tracking_error_m
                        ),
                        force_correction_velocity_m_s=(
                            force_correction_velocity_m_s
                        ),
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

                # -----------------------------
                # Rigid-body dynamics telemetry.
                # -----------------------------
                if previous_dq is None:
                    dynamics_ddq_estimated_rad_s2 = (
                        np.zeros(
                            self.model.nv
                            if hasattr(self, "model")
                            else self.mechanics.model.nv,
                            dtype=float,
                        )
                    )
                else:
                    dynamics_ddq_estimated_rad_s2 = (
                        dq - previous_dq
                    ) / self.config.timestep_s

                previous_dq = dq.copy()

                dynamics_mass_matrix = (
                    self.mechanics.get_M(q)
                )

                dynamics_coriolis_matrix = (
                    self.mechanics.get_C(
                        q,
                        dq,
                    )
                )

                dynamics_gravity_nm = (
                    self.mechanics.get_G(q)
                )

                dynamics_inertia_nm = (
                    dynamics_mass_matrix
                    @ dynamics_ddq_estimated_rad_s2
                )

                dynamics_coriolis_nm = (
                    dynamics_coriolis_matrix
                    @ dq
                )

                dynamics_tau_model_nm = (
                    dynamics_inertia_nm
                    + dynamics_coriolis_nm
                    + dynamics_gravity_nm
                )

                dynamics_tau_rnea_nm = (
                    self.mechanics.get_tau(
                        q,
                        dq,
                        dynamics_ddq_estimated_rad_s2,
                    )
                )

                dynamics_reconstruction_error_nm = (
                    dynamics_tau_rnea_nm
                    - dynamics_tau_model_nm
                )

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

                    "physical_pen_x_m": float(
                        physical_pen_tip_position_m[0]
                    ),
                    "physical_pen_y_m": float(
                        physical_pen_tip_position_m[1]
                    ),
                    "physical_pen_z_m": float(
                        physical_pen_tip_position_m[2]
                    ),

                    "physical_tip_disagreement_m": (
                        physical_tip_disagreement_m
                    ),

                    "pen_contact_active": int(
                        pen_contact_state.active
                    ),
                    "pen_contact_count": int(
                        pen_contact_state.contact_count
                    ),
                    "pen_contact_distance_m": (
                        pen_contact_state.minimum_distance_m
                    ),
                    "pen_normal_force_n": float(
                        pen_contact_state.normal_force_n
                    ),

                    "desired_pen_normal_force_n": float(
                        contact_parameters.desired_normal_force_n
                    ),

                    "force_control_active": int(
                        force_control_active
                    ),

                    "force_error_n": (
                        float(
                            force_control_result.force_error_n
                        )
                        if force_control_result is not None
                        else None
                    ),

                    "force_correction_velocity_m_s": (
                        force_correction_velocity_m_s
                    ),

                    "force_control_saturated": int(
                        force_control_result.saturated
                        if force_control_result is not None
                        else False
                    ),

                    "controller_pen_z_target_m": float(
                        controller_position_m[2]
                    ),

                    "controller_pen_vz_m_s": float(
                        controller_velocity_m_s[2]
                    ),

                    "xy_tracking_error_m": (
                        xy_tracking_error_m
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

                for joint_index in range(
                    simulator.joint_count
                ):
                    joint_number = (
                        joint_index + 1
                    )

                    row[
                        f"ddq{joint_number}_estimated_rad_s2"
                    ] = float(
                        dynamics_ddq_estimated_rad_s2[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_inertia_model_nm"
                    ] = float(
                        dynamics_inertia_nm[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_coriolis_model_nm"
                    ] = float(
                        dynamics_coriolis_nm[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_gravity_model_nm"
                    ] = float(
                        dynamics_gravity_nm[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_inverse_dynamics_model_nm"
                    ] = float(
                        dynamics_tau_model_nm[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_rnea_model_nm"
                    ] = float(
                        dynamics_tau_rnea_nm[
                            joint_index
                        ]
                    )

                    row[
                        f"tau{joint_number}_reconstruction_error_nm"
                    ] = float(
                        dynamics_reconstruction_error_nm[
                            joint_index
                        ]
                    )

                row[
                    "dynamics_reconstruction_error_norm_nm"
                ] = float(
                    np.linalg.norm(
                        dynamics_reconstruction_error_nm
                    )
                )

                rows.append(row)

                if (
                    self.telemetry_callback is not None
                    and simulation_time_s
                    + 1.0e-12
                    >= next_telemetry_time_s
                ):
                    self.telemetry_callback(
                        dict(row)
                    )

                    while (
                        next_telemetry_time_s
                        <= simulation_time_s
                    ):
                        next_telemetry_time_s += (
                            self.telemetry_period_s
                        )

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

                pen_normal_forces_n.append(
                    float(
                        pen_contact_state.normal_force_n
                    )
                )

                physical_tip_disagreements_m.append(
                    physical_tip_disagreement_m
                )

                xy_tracking_errors_m.append(
                    xy_tracking_error_m
                )

                if force_control_result is not None:
                    write_force_errors_n.append(
                        float(
                            force_control_result.force_error_n
                        )
                    )

                    force_correction_velocities_m_s.append(
                        float(
                            force_control_result
                            .commanded_world_z_velocity_m_s
                        )
                    )

                    if force_control_result.saturated:
                        force_control_saturation_count += 1

                if (
                    pen_contact_state.minimum_distance_m
                    is not None
                ):
                    pen_contact_distances_m.append(
                        float(
                            pen_contact_state.minimum_distance_m
                        )
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
                "speed_scale": float(
                    self.config.speed_scale
                ),
                "full_plan": bool(
                    self.config.full_plan
                ),
                "execution_num_steps": int(
                    run_num_steps
                ),
                "nominal_surface_height_m": (
                    nominal_write_height_m
                ),
                "commanded_write_height_m": (
                    write_height_m
                ),
                "preload_depth_m": (
                    contact_parameters.preload_depth_m
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

            summary["contact_model"] = {
                "type": "kelvin_voigt_unilateral",
                "surface_height_m": (
                    contact_parameters.surface_height_m
                ),
                "desired_normal_force_n": (
                    contact_parameters.desired_normal_force_n
                ),
                "preload_depth_m": (
                    contact_parameters.preload_depth_m
                ),
                "contact_stiffness_n_m": (
                    contact_parameters.contact_stiffness_n_m
                ),
                "contact_damping_n_s_m": (
                    contact_parameters.contact_damping_n_s_m
                ),
                "lateral_friction": (
                    contact_parameters.lateral_friction
                ),
                "maximum_normal_force_n": (
                    contact_parameters.maximum_normal_force_n
                ),
            }

            summary["controller"] = {
                "type": (
                    "hybrid_xy_position_z_force_pose_control"
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
                "write_control": {
                    "xy": "cartesian_position",
                    "z": "normal_force_admittance",
                    "orientation": "pose_hold",
                    "desired_normal_force_n": (
                        contact_parameters
                        .desired_normal_force_n
                    ),
                    "force_velocity_gain_m_s_n": (
                        contact_parameters
                        .force_velocity_gain_m_s_n
                    ),
                    "maximum_force_correction_velocity_m_s": (
                        contact_parameters
                        .maximum_force_correction_velocity_m_s
                    ),
                    "maximum_normal_force_n": (
                        contact_parameters
                        .maximum_normal_force_n
                    ),
                },
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

            contact_rows = [
                row
                for row in rows
                if int(row["pen_contact_active"]) == 1
            ]

            write_rows = [
                row
                for row in rows
                if row["writing_phase"] == "WRITE"
            ]

            write_contact_rows = [
                row
                for row in write_rows
                if int(row["pen_contact_active"]) == 1
            ]

            summary["hybrid_control"] = {
                "xy_tracking_error_m": (
                    summarize_values(
                        xy_tracking_errors_m,
                        name="xy_tracking_errors_m",
                    )
                ),
                "write_force_error_n": (
                    summarize_signed_values(
                        write_force_errors_n,
                        name="write_force_errors_n",
                    )
                    if write_force_errors_n
                    else None
                ),
                "force_correction_velocity_m_s": (
                    summarize_signed_values(
                        force_correction_velocities_m_s,
                        name=(
                            "force_correction_velocities_m_s"
                        ),
                    )
                    if force_correction_velocities_m_s
                    else None
                ),
                "force_control_saturation_count": int(
                    force_control_saturation_count
                ),
            }

            summary["physical_pen"] = {
                "tip_tracking_disagreement_m": (
                    summarize_values(
                        physical_tip_disagreements_m,
                        name=(
                            "physical_tip_disagreements_m"
                        ),
                    )
                ),
                "contact": {
                    "active_sample_count": len(
                        contact_rows
                    ),
                    "overall_contact_rate": (
                        len(contact_rows) / len(rows)
                        if rows
                        else 0.0
                    ),
                    "write_contact_sample_count": len(
                        write_contact_rows
                    ),
                    "write_contact_rate": (
                        len(write_contact_rows)
                        / len(write_rows)
                        if write_rows
                        else 0.0
                    ),
                    "normal_force_n": (
                        summarize_values(
                            pen_normal_forces_n,
                            name="pen_normal_forces_n",
                        )
                    ),
                    "minimum_contact_distance_m": (
                        min(pen_contact_distances_m)
                        if pen_contact_distances_m
                        else None
                    ),
                },
            }

            summary["warnings"] = [
                (
                    "The physical pen is rigidly attached "
                    "to fer_link8 and contact force is "
                    "measured, but force regulation is "
                    "not yet enabled."
                ),
                (
                    "Writing currently uses Cartesian "
                    "pose control; impedance/contact-force "
                    "control is the next controller stage."
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
