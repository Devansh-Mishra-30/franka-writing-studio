"""Validated configuration for writing-robot experiments."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from config import CANONICAL_SVG, RUNTIME


VALID_MODES = ("direct", "gui")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the CLI and simulation runner."""

    mode: str = "direct"
    duration_s: float = 2.0
    timestep_s: float = RUNTIME.timestep_s
    svg_file: Path = CANONICAL_SVG
    output_dir: Path = Path(
        "artifacts/baseline"
    )
    realtime: bool = False
    record_video: bool = False
    seed: int = 0

    speed_scale: float = 1.0
    full_plan: bool = False

    def validate(self) -> "ExperimentConfig":
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}; "
                f"received {self.mode!r}"
            )

        if (
            not math.isfinite(self.duration_s)
            or self.duration_s <= 0.0
        ):
            raise ValueError(
                "duration_s must be finite and positive"
            )

        if (
            not math.isfinite(self.timestep_s)
            or self.timestep_s <= 0.0
        ):
            raise ValueError(
                "timestep_s must be finite and positive"
            )

        if self.duration_s < self.timestep_s:
            raise ValueError(
                "duration_s must be at least one timestep"
            )

        if (
            not math.isfinite(self.speed_scale)
            or self.speed_scale <= 0.0
        ):
            raise ValueError(
                "speed_scale must be finite and positive"
            )

        if not self.svg_file.is_file():
            raise FileNotFoundError(
                "SVG trajectory file does not exist: "
                f"{self.svg_file}"
            )

        if self.record_video and self.mode != "gui":
            raise ValueError(
                "--record-video requires --mode gui"
            )

        return self

    @property
    def num_steps(self) -> int:
        return max(
            1,
            int(
                round(
                    self.duration_s
                    / self.timestep_s
                )
            ),
        )

    @property
    def simulated_duration_s(self) -> float:
        return (
            self.num_steps
            * self.timestep_s
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)

        payload["svg_file"] = str(
            self.svg_file
        )
        payload["output_dir"] = str(
            self.output_dir
        )

        # duration_s controls fixed-duration runs only.
        # Full-plan runs derive their actual execution duration
        # from WritingPlan.total_duration_s.
        payload["requested_duration_s"] = payload.pop(
            "duration_s"
        )
        payload["requested_num_steps"] = self.num_steps
        payload["requested_simulated_duration_s"] = (
            self.simulated_duration_s
        )
        payload["execution_duration_source"] = (
            "writing_plan"
            if self.full_plan
            else "requested_duration"
        )

        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a finite Franka Panda "
            "writing-robot experiment."
        )
    )

    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="direct",
        help="PyBullet connection mode.",
    )

    parser.add_argument(
        "--duration",
        dest="duration_s",
        type=float,
        default=2.0,
        help=(
            "Requested simulation duration in seconds. "
            "With --full-plan the entire writing plan "
            "is executed instead."
        ),
    )

    parser.add_argument(
        "--timestep",
        dest="timestep_s",
        type=float,
        default=RUNTIME.timestep_s,
        help="Physics timestep in seconds.",
    )

    parser.add_argument(
        "--svg",
        dest="svg_file",
        type=Path,
        default=CANONICAL_SVG,
        help="SVG trajectory file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/baseline"
        ),
        help="Directory for experiment evidence.",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle simulation toward real time.",
    )

    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record the PyBullet GUI.",
    )

    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply all writing-plan motion "
            "speeds by this factor."
        ),
    )

    parser.add_argument(
        "--full-plan",
        action="store_true",
        help=(
            "Execute the complete writing plan "
            "instead of stopping at --duration."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for repeatable experiments.",
    )

    return parser


def parse_config(
    argv: Sequence[str] | None = None,
) -> ExperimentConfig:
    args = build_parser().parse_args(argv)

    return ExperimentConfig(
        mode=args.mode,
        duration_s=args.duration_s,
        timestep_s=args.timestep_s,
        svg_file=args.svg_file,
        output_dir=args.output_dir,
        realtime=args.realtime,
        record_video=args.record_video,
        seed=args.seed,
        speed_scale=args.speed_scale,
        full_plan=args.full_plan,
    ).validate()


if __name__ == "__main__":
    print(parse_config().to_dict())
