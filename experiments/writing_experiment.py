"""Finite, measurable writing-robot experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from controller import PD_velocity
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
from simulation.interfaces import VelocityCommand
from simulation.pybullet_adapter import (
    PyBulletAdapter,
    PyBulletSettings,
)
from trajectory import SvgTrajectory


INITIAL_JOINT_POSITIONS_RAD = np.array(
    [
        0.0,
        -0.785,
        0.0,
        -2.355,
        0.0,
        1.57,
        0.785,
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
        model_simulator_disagreements_m: list[float] = []
        joint_errors_rad: list[float] = []
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

            setup_duration_s = (
                time.perf_counter() - setup_start
            )

            run_start = time.perf_counter()

            for step_index in range(
                self.config.num_steps
            ):
                cycle_start = time.perf_counter()

                simulation_time_s = (
                    step_index
                    * self.config.timestep_s
                )

                state = simulator.read_state(
                    simulation_time_s
                )

                q = state.joint_positions_rad
                dq = state.joint_velocities_rad_s

                desired_pose = self.trajectory.sample(
                    simulation_time_s
                )

                model_pose = np.asarray(
                    self.mechanics.solve_fk(q),
                    dtype=float,
                )

                ik_start = time.perf_counter()

                q_desired = np.asarray(
                    self.mechanics.solve_ik(
                        q=q.copy(),
                        x=float(desired_pose[0]),
                        y=float(desired_pose[1]),
                        z=float(desired_pose[2]),
                    ),
                    dtype=float,
                )

                ik_duration_s = (
                    time.perf_counter() - ik_start
                )

                velocity_command = np.asarray(
                    PD_velocity(
                        q,
                        dq,
                        q_desired,
                    ),
                    dtype=float,
                )

                desired_position_m = desired_pose[:3]
                model_position_m = model_pose[:3]
                simulator_position_m = (
                    state.tool_position_m
                )

                simulator_error_m = float(
                    np.linalg.norm(
                        simulator_position_m
                        - desired_position_m
                    )
                )

                model_error_m = float(
                    np.linalg.norm(
                        model_position_m
                        - desired_position_m
                    )
                )

                model_simulator_disagreement_m = float(
                    np.linalg.norm(
                        model_position_m
                        - simulator_position_m
                    )
                )

                joint_error_rad = float(
                    np.linalg.norm(
                        q_desired - q
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
                        time.sleep(sleep_duration_s)

                row: dict[str, Any] = {
                    "step": step_index,
                    "simulation_time_s": (
                        simulation_time_s
                    ),
                    "trajectory_index": (
                        self.trajectory.index_at(
                            simulation_time_s
                        )
                    ),
                    "desired_x_m": float(
                        desired_position_m[0]
                    ),
                    "desired_y_m": float(
                        desired_position_m[1]
                    ),
                    "desired_z_m": float(
                        desired_position_m[2]
                    ),
                    "simulator_x_m": float(
                        simulator_position_m[0]
                    ),
                    "simulator_y_m": float(
                        simulator_position_m[1]
                    ),
                    "simulator_z_m": float(
                        simulator_position_m[2]
                    ),
                    "model_x_m": float(
                        model_position_m[0]
                    ),
                    "model_y_m": float(
                        model_position_m[1]
                    ),
                    "model_z_m": float(
                        model_position_m[2]
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
                    "joint_error_norm_rad": (
                        joint_error_rad
                    ),
                    "ik_duration_s": ik_duration_s,
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
                    joint_number = joint_index + 1

                    row[
                        f"q{joint_number}_rad"
                    ] = float(q[joint_index])

                    row[
                        f"dq{joint_number}_rad_s"
                    ] = float(dq[joint_index])

                    row[
                        f"q{joint_number}_desired_rad"
                    ] = float(
                        q_desired[joint_index]
                    )

                    row[
                        f"dq{joint_number}_command_rad_s"
                    ] = float(
                        velocity_command[joint_index]
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
                joint_errors_rad.append(
                    joint_error_rad
                )
                loop_durations_s.append(
                    loop_duration_s
                )
                ik_durations_s.append(
                    ik_duration_s
                )

            wall_duration_s = (
                time.perf_counter() - run_start
            )

            summary = build_experiment_summary(
                position_errors_m=(
                    simulator_position_errors_m
                ),
                joint_errors_rad=joint_errors_rad,
                loop_durations_s=loop_durations_s,
                ik_durations_s=ik_durations_s,
                timestep_s=self.config.timestep_s,
                wall_duration_s=wall_duration_s,
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

            summary["execution"][
                "setup_duration_s"
            ] = float(setup_duration_s)

            summary["execution"][
                "deadline_miss_count"
            ] = int(deadline_miss_count)

            summary["execution"][
                "deadline_miss_rate"
            ] = float(
                deadline_miss_count
                / self.config.num_steps
            )

            summary[
                "franka_pinocchio_position_error_m"
            ] = summarize_values(
                model_position_errors_m,
                name="model_position_errors_m",
            )

            summary[
                "pinocchio_pybullet_disagreement_m"
            ] = summarize_values(
                model_simulator_disagreements_m,
                name=(
                    "model_simulator_disagreements_m"
                ),
            )

            summary["warnings"] = [
                (
                    "The measured tool frame is fer_link8, "
                    "not yet the physical pen tip."
                ),
                (
                    "The legacy IK solver does not yet "
                    "return convergence diagnostics."
                ),
                (
                    "The existing velocity controller "
                    "and gains are intentionally preserved."
                ),
                (
                    "The SVG trajectory remains "
                    "piecewise constant at 0.1-second "
                    "waypoint intervals."
                ),
            ]

            write_csv(samples_path, rows)
            write_json(summary_path, summary)

            if failure_path.exists():
                failure_path.unlink()

            return ExperimentResult(
                samples_path=samples_path,
                summary_path=summary_path,
                summary=summary,
            )

        except Exception as error:
            elapsed_s = (
                time.perf_counter() - setup_start
            )

            write_json(
                failure_path,
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "elapsed_s": float(elapsed_s),
                    "config": self.config.to_dict(),
                },
            )

            raise

        finally:
            simulator.shutdown()
