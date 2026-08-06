"""Command-line entry point for writing-robot experiments."""

from __future__ import annotations

from experiment_config import parse_config
from experiments.writing_experiment import (
    WritingExperiment,
)


def main() -> int:
    config = parse_config()
    result = WritingExperiment(config).run()

    execution = result.summary["execution"]
    position_error = result.summary[
        "cartesian_position_error_m"
    ]

    print()
    print("[COMPLETED] Writing-robot experiment")
    print(f"Samples: {result.samples_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Steps: {execution['steps']}")
    print(
        "Cartesian RMSE: "
        f"{position_error['rmse']:.6f} m"
    )
    print(
        "Cartesian maximum error: "
        f"{position_error['maximum']:.6f} m"
    )
    print(
        "Real-time factor: "
        f"{execution['real_time_factor']:.3f}"
    )
    print(
        "Deadline misses: "
        f"{execution['deadline_miss_count']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
