"""Validated configuration for writing-robot experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


VALID_MODES = ("direct", "gui")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the CLI and simulation runner."""

    mode: str = "direct"
    duration_s: float = 2.0
    timestep_s: float = 0.001
    svg_file: Path = Path("svg/portfolio_writing1 (8).svg")
    output_dir: Path = Path("artifacts/baseline")
    realtime: bool = False
    record_video: bool = False
    seed: int = 0

    def validate(self) -> "ExperimentConfig":
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}; received {self.mode!r}"
            )

        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be greater than zero")

        if self.timestep_s <= 0.0:
            raise ValueError("timestep_s must be greater than zero")

        if self.duration_s < self.timestep_s:
            raise ValueError(
                "duration_s must be at least one timestep"
            )

        if not self.svg_file.is_file():
            raise FileNotFoundError(
                f"SVG trajectory file does not exist: {self.svg_file}"
            )

        if self.record_video and self.mode != "gui":
            raise ValueError(
                "--record-video requires --mode gui"
            )

        return self

    @property
    def num_steps(self) -> int:
        """Deterministic number of simulation iterations."""

        return max(
            1,
            int(round(self.duration_s / self.timestep_s)),
        )

    @property
    def simulated_duration_s(self) -> float:
        return self.num_steps * self.timestep_s

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["svg_file"] = str(self.svg_file)
        payload["output_dir"] = str(self.output_dir)
        payload["num_steps"] = self.num_steps
        payload["simulated_duration_s"] = (
            self.simulated_duration_s
        )
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a finite Franka Panda writing-robot experiment."
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
        help="Requested simulation duration in seconds.",
    )
    parser.add_argument(
        "--timestep",
        dest="timestep_s",
        type=float,
        default=0.001,
        help="Physics timestep in seconds.",
    )
    parser.add_argument(
        "--svg",
        dest="svg_file",
        type=Path,
        default=Path("svg/portfolio_writing1 (8).svg"),
        help="SVG trajectory file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline"),
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
    ).validate()


if __name__ == "__main__":
    print(parse_config().to_dict())
