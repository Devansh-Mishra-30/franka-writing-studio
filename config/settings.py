"""Central application settings for the Franka Writing Studio."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    timestep_s: float = 0.001
    telemetry_period_s: float = 0.05
    visualization_update_period_s: float = 0.05


@dataclass(frozen=True)
class PlanningSettings:
    notebook_margin_m: float = 0.02


@dataclass(frozen=True)
class ValidationSettings:
    maximum_ik_samples: int = 24


@dataclass(frozen=True)
class ControllerSettings:
    position_gain_s_inv: float = 4.0
    orientation_gain_s_inv: float = 4.0
    differential_ik_damping: float = 0.02


RUNTIME = RuntimeSettings()
PLANNING = PlanningSettings()
VALIDATION = ValidationSettings()
CONTROLLER = ControllerSettings()


CANONICAL_SVG = Path("svg/franka_portfolio.svg")
