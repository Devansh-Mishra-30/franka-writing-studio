"""Generate publication-quality validation plots for Franka Writing Studio."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Global plotting standard
# ============================================================

FIGSIZE = (10.0, 5.6)
DPI = 300

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 16,
        "lines.linewidth": 1.3,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "savefig.dpi": DPI,
    }
)


def load_samples(path: Path) -> dict[str, np.ndarray]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise RuntimeError(
            f"No samples found in {path}"
        )

    data: dict[str, np.ndarray] = {}

    for column in rows[0].keys():
        if column == "writing_phase":
            data[column] = np.asarray(
                [row[column] for row in rows],
                dtype=object,
            )
            continue

        try:
            data[column] = np.asarray(
                [
                    float(row[column])
                    for row in rows
                ],
                dtype=float,
            )
        except ValueError:
            data[column] = np.asarray(
                [row[column] for row in rows],
                dtype=object,
            )

    return data


def create_figure(
    *,
    title: str,
    subtitle: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Create one consistently formatted validation figure."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    # Explicit spacing is intentional. It prevents legends,
    # subtitles, metric summaries, and axes from overlapping.
    fig.subplots_adjust(
        left=0.105,
        right=0.975,
        bottom=0.175,
        top=0.78,
    )

    fig.suptitle(
        title,
        x=0.5,
        y=0.965,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.905,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.grid(
        True,
        which="major",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=4,
        width=0.8,
    )

    return fig, ax


def add_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    columns: int = 2,
) -> None:
    """Place the legend in reserved whitespace above the axes."""

    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=columns,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.0,
    )


def add_footer(
    fig: plt.Figure,
    text: str,
) -> None:
    """Add an evidence/interpretation note outside the data region."""

    fig.text(
        0.5,
        0.035,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )


def save(
    fig: plt.Figure,
    path: Path,
) -> None:
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    plt.close(fig)

    print(f"wrote: {path}")


def nice_upper_limit(
    value: float,
    *,
    minimum: float,
    step: float,
    margin: float = 1.15,
) -> float:
    """Return a stable readable upper plotting bound."""

    target = max(
        minimum,
        value * margin,
    )

    return (
        math.ceil(
            target / step
        )
        * step
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate engineering validation plots "
            "from a completed Writing Studio artifact."
        )
    )

    parser.add_argument(
        "artifact",
        type=Path,
        help=(
            "Artifact directory containing "
            "samples.csv and summary.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "docs/media/results"
        ),
    )

    args = parser.parse_args()

    artifact = args.artifact
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_path = (
        artifact / "samples.csv"
    )

    summary_path = (
        artifact / "summary.json"
    )

    if not samples_path.is_file():
        raise FileNotFoundError(
            samples_path
        )

    if not summary_path.is_file():
        raise FileNotFoundError(
            summary_path
        )

    data = load_samples(
        samples_path
    )

    summary = json.loads(
        summary_path.read_text()
    )

    if summary.get("status") != "completed":
        raise RuntimeError(
            "Result plots require a completed artifact"
        )

    time_s = data[
        "simulation_time_s"
    ]

    phase = data[
        "writing_phase"
    ]

    write = (
        phase == "WRITE"
    )

    if not np.any(write):
        raise RuntimeError(
            "Artifact contains no WRITE samples"
        )

    write_time = time_s[write]

    # Convert Cartesian quantities to millimetres
    # for engineering readability.
    desired_x_mm = (
        1000.0
        * data["desired_pen_x_m"][write]
    )

    desired_y_mm = (
        1000.0
        * data["desired_pen_y_m"][write]
    )

    actual_x_mm = (
        1000.0
        * data["physical_pen_x_m"][write]
    )

    actual_y_mm = (
        1000.0
        * data["physical_pen_y_m"][write]
    )

    xy_error_mm = (
        1000.0
        * data[
            "xy_tracking_error_m"
        ][write]
    )

    force_n = data[
        "pen_normal_force_n"
    ][write]

    target_force_n = data[
        "desired_pen_normal_force_n"
    ][write]

    contact = data[
        "pen_contact_active"
    ][write]

    saturation = data[
        "force_control_saturated"
    ][write]

    write_samples = int(
        np.sum(write)
    )

    write_contact_rate = float(
        np.mean(
            contact > 0.5
        )
    )

    xy_rmse_mm = float(
        np.sqrt(
            np.mean(
                xy_error_mm**2
            )
        )
    )

    xy_max_mm = float(
        np.max(
            xy_error_mm
        )
    )

    mean_force_n = float(
        np.mean(
            force_n
        )
    )

    target_mean_force_n = float(
        np.mean(
            target_force_n
        )
    )

    force_error_n = (
        force_n
        - target_force_n
    )

    force_rms_error_n = float(
        np.sqrt(
            np.mean(
                force_error_n**2
            )
        )
    )

    saturation_count = int(
        np.sum(
            saturation > 0.5
        )
    )

    # ============================================================
    # 1. Desired vs actual WRITE trajectory
    # ============================================================

    # The SVG trajectory is mapped into the robot/world frame with
    # the notebook's long writing direction aligned with -World Y.
    # For human-readable validation we apply the same rigid
    # coordinate transform to desired and actual trajectories:
    #
    #   surface horizontal = -World Y
    #   surface vertical   =  World X
    #
    # This changes only visualization coordinates; tracking metrics
    # remain computed in the original robot/world frame.

    surface_x_desired_mm = -desired_y_mm
    surface_y_desired_mm = desired_x_mm

    surface_x_actual_mm = -actual_y_mm
    surface_y_actual_mm = actual_x_mm

    # Center the displayed coordinates on the desired path.
    surface_x_origin_mm = 0.5 * (
        np.min(surface_x_desired_mm)
        + np.max(surface_x_desired_mm)
    )

    surface_y_origin_mm = 0.5 * (
        np.min(surface_y_desired_mm)
        + np.max(surface_y_desired_mm)
    )

    surface_x_desired_mm -= surface_x_origin_mm
    surface_y_desired_mm -= surface_y_origin_mm

    surface_x_actual_mm -= surface_x_origin_mm
    surface_y_actual_mm -= surface_y_origin_mm

    fig, ax = create_figure(
        title="Pen-Tip Trajectory Tracking",
        subtitle=(
            "Desired versus simulated physical pen-tip trajectory "
            "shown in writing-surface coordinates"
        ),
    )

    ax.plot(
        surface_x_desired_mm,
        surface_y_desired_mm,
        linestyle="--",
        linewidth=2.2,
        label="Desired trajectory",
        zorder=3,
    )

    ax.plot(
        surface_x_actual_mm,
        surface_y_actual_mm,
        linewidth=1.15,
        label="Simulated physical pen tip",
        zorder=2,
    )

    ax.set_xlabel(
        "Writing-surface horizontal position [mm]"
    )

    ax.set_ylabel(
        "Writing-surface vertical position [mm]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    x_min = min(
        np.min(surface_x_desired_mm),
        np.min(surface_x_actual_mm),
    )

    x_max = max(
        np.max(surface_x_desired_mm),
        np.max(surface_x_actual_mm),
    )

    y_min = min(
        np.min(surface_y_desired_mm),
        np.min(surface_y_actual_mm),
    )

    y_max = max(
        np.max(surface_y_desired_mm),
        np.max(surface_y_actual_mm),
    )

    x_pad = max(
        5.0,
        0.035 * (x_max - x_min),
    )

    y_pad = max(
        5.0,
        0.06 * (y_max - y_min),
    )

    ax.set_xlim(
        x_min - x_pad,
        x_max + x_pad,
    )

    ax.set_ylim(
        y_min - y_pad,
        y_max + y_pad,
    )

    add_legend(
        fig,
        ax,
        columns=2,
    )

    add_footer(
        fig,
        (
            f"WRITE samples: {write_samples:,}   |   "
            f"XY RMSE: {xy_rmse_mm:.4f} mm   |   "
            f"Maximum XY error: {xy_max_mm:.4f} mm   |   "
            f"Contact: {100.0 * write_contact_rate:.2f}%"
        ),
    )

    save(
        fig,
        output_dir
        / "writing_path_tracking.png",
    )

    # ============================================================
    # 2. Cartesian tracking error
    # ============================================================

    fig, ax = create_figure(
        title=(
            "WRITE-Phase Cartesian Tracking Error"
        ),
        subtitle=(
            "Planar XY pen-tip tracking error over "
            "the complete canonical writing trajectory"
        ),
    )

    ax.plot(
        write_time,
        xy_error_mm,
        linewidth=1.0,
        label="Instantaneous XY error",
    )

    ax.axhline(
        xy_rmse_mm,
        linestyle="--",
        linewidth=1.6,
        label=(
            f"RMSE = {xy_rmse_mm:.4f} mm"
        ),
    )

    ax.set_xlabel(
        "Simulation time [s]"
    )

    ax.set_ylabel(
        "XY tracking error [mm]"
    )

    ax.set_xlim(
        float(write_time[0]),
        float(write_time[-1]),
    )

    error_upper_mm = (
        nice_upper_limit(
            xy_max_mm,
            minimum=0.20,
            step=0.05,
        )
    )

    ax.set_ylim(
        0.0,
        error_upper_mm,
    )

    add_legend(
        fig,
        ax,
        columns=2,
    )

    add_footer(
        fig,
        (
            f"RMSE: {xy_rmse_mm:.4f} mm   |   "
            f"Maximum: {xy_max_mm:.4f} mm   |   "
            f"WRITE samples: {write_samples:,}   |   "
            "Computed from simulated physical pen-tip position"
        ),
    )

    save(
        fig,
        output_dir
        / "xy_tracking_error.png",
    )

    # ============================================================
    # 3. Normal-force regulation
    # ============================================================

    fig, ax = create_figure(
        title=(
            "Normal-Force Regulation During Writing"
        ),
        subtitle=(
            "Velocity-level Z admittance during WRITE "
            "with a 1.0 N normal-force design target"
        ),
    )

    ax.plot(
        write_time,
        force_n,
        linewidth=1.0,
        label="Simulated normal force",
    )

    ax.plot(
        write_time,
        target_force_n,
        linestyle="--",
        linewidth=1.7,
        label="Target normal force",
    )

    ax.set_xlabel(
        "Simulation time [s]"
    )

    ax.set_ylabel(
        "Normal force [N]"
    )

    ax.set_xlim(
        float(write_time[0]),
        float(write_time[-1]),
    )

    force_min = float(
        np.min(force_n)
    )

    force_max = float(
        np.max(force_n)
    )

    lower_force_limit = min(
        0.70,
        math.floor(
            (
                force_min
                - 0.05
            )
            / 0.05
        )
        * 0.05,
    )

    upper_force_limit = max(
        1.30,
        math.ceil(
            (
                force_max
                + 0.05
            )
            / 0.05
        )
        * 0.05,
    )

    ax.set_ylim(
        lower_force_limit,
        upper_force_limit,
    )

    add_legend(
        fig,
        ax,
        columns=2,
    )

    add_footer(
        fig,
        (
            f"Mean force: {mean_force_n:.4f} N   |   "
            f"Target: {target_mean_force_n:.4f} N   |   "
            f"RMS force error: {force_rms_error_n:.4f} N   |   "
            f"Controller saturation events: {saturation_count}"
        ),
    )

    save(
        fig,
        output_dir
        / "writing_force_control.png",
    )

    # ============================================================
    # 4. Motion-phase timeline
    # ============================================================

    phase_order = [
        "APPROACH",
        "LOWER",
        "WRITE",
        "LIFT",
        "TRANSFER",
        "RETREAT",
    ]

    phase_to_value = {
        name: index
        for index, name
        in enumerate(
            phase_order
        )
    }

    phase_values = np.asarray(
        [
            phase_to_value.get(
                str(value),
                -1,
            )
            for value in phase
        ],
        dtype=float,
    )

    fig, ax = create_figure(
        title=(
            "Writing Task Execution Timeline"
        ),
        subtitle=(
            "Minimum-jerk Cartesian motion sequence "
            "from approach through final retreat"
        ),
    )

    ax.step(
        time_s,
        phase_values,
        where="post",
        linewidth=1.25,
    )

    ax.set_xlabel(
        "Simulation time [s]"
    )

    ax.set_ylabel(
        "Motion phase"
    )

    ax.set_xlim(
        float(time_s[0]),
        float(time_s[-1]),
    )

    ax.set_ylim(
        -0.4,
        len(phase_order) - 0.6,
    )

    ax.set_yticks(
        range(
            len(phase_order)
        ),
        phase_order,
    )

    ax.grid(
        True,
        axis="x",
    )

    ax.grid(
        False,
        axis="y",
    )

    total_duration_s = float(
        time_s[-1]
        - time_s[0]
    )

    add_footer(
        fig,
        (
            f"Complete application execution: "
            f"{total_duration_s:.2f} s simulated time   |   "
            f"{len(phase_order)} defined motion phases   |   "
            "WRITE uses XY Cartesian tracking with Z normal-force admittance"
        ),
    )

    save(
        fig,
        output_dir
        / "execution_phases.png",
    )

    # ============================================================
    # Console validation summary
    # ============================================================

    print()
    print(
        "=== CANONICAL RESULT SUMMARY ==="
    )

    print(
        f"artifact: {artifact}"
    )

    print(
        f"status: {summary.get('status')}"
    )

    print(
        f"WRITE samples: {write_samples}"
    )

    print(
        "WRITE contact: "
        f"{100.0 * write_contact_rate:.2f}%"
    )

    print(
        "WRITE XY RMSE: "
        f"{xy_rmse_mm:.4f} mm"
    )

    print(
        "WRITE XY max: "
        f"{xy_max_mm:.4f} mm"
    )

    print(
        "mean normal force: "
        f"{mean_force_n:.4f} N"
    )

    print(
        "force RMS error: "
        f"{force_rms_error_n:.4f} N"
    )

    print(
        "force saturation count: "
        f"{saturation_count}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
