# System Architecture

## Scope

Franka Writing Studio is a simulation-only robotic writing application. It separates task-level commands, planning, validation, control, simulation, telemetry, and user-interface responsibilities.

## Data Flow

```text
SVG input
   |
   v
Cartesian source waypoints
   |
   v
Notebook scaling and placement
   |
   v
Minimum-jerk writing plan
   |
   v
Bounds, reachability, and collision preflight
   |
   v
RobotInterface lifecycle
   |
   v
Hybrid Cartesian command
   |
   v
Damped least-squares differential IK
   |
   v
Bounded joint-velocity command
   |
   v
PyBullet simulation and telemetry
```

## Components

| Component | Responsibility |
|---|---|
| `config/` | Release configuration and canonical inputs |
| `robot/` | Commands, application state, faults, status, and public interface |
| `tasks/` | Writing-task planning, validation, execution, and cancellation |
| `workcells/` | Notebook, table, tools, workspace, and collision preflight |
| `experiments/` | Experiment lifecycle, telemetry, metrics, and artifacts |
| `simulation/` | PyBullet-specific robot and scene implementation |
| `ui/` | PySide6 command center and worker thread |
| `verification/` | Pinocchio/MATLAB and model-mismatch workflows |
| `learning/` | Offline residual model |
| `tests/` | Numerical, unit, component, lifecycle, and integration tests |

## Dependency Direction

Application and task logic depend on robot and simulation abstractions. PyBullet-specific behavior is contained inside the simulation layer. The GUI issues commands through the application interface and receives status and telemetry without owning control logic.

## Planning and Execution

SVG processing produces source waypoints. `TimedWritingPlan` adds approach, lowering, writing, transfer, lifting, and return segments using minimum-jerk interpolation.

Before execution, the application checks notebook bounds, sampled model reachability, and sampled robot-table collision. The controller then tracks Cartesian XY position, regulates Z normal force through velocity admittance, and holds orientation.

## Kinematics, Dynamics, and Simulation

Pinocchio supplies kinematics and rigid-body dynamics. PyBullet supplies simulation, contact response, and measured state. MATLAB Robotics System Toolbox is used as an independent software implementation for cross-tool comparison.

The active writing controller sends bounded joint-velocity commands. The dynamics calculations support verification and offline model-mismatch experiments; they are not used as an active torque controller.

## Application Boundary

`RobotInterface` is the public application boundary. It owns the high-level command contract, lifecycle state, fault reporting, task progress, telemetry, cooperative cancellation, and reset behavior.

This design allows a future ROS 2 action server or PLC communication adapter to translate external messages into the same application commands without moving control logic into the adapter.

## Deployment Boundary

Version 1.0 does not contain ROS 2, PLC, physical I/O, perception, calibration, or safety-certified functionality. Collision preflight is sampled rather than continuous, and all reported performance is from simulation.
