# Robot Task Interface

## Purpose

`RobotInterface` is the public application boundary for Franka Writing Studio. It converts high-level commands into controlled task transitions while maintaining authoritative state, progress, telemetry, and fault information.

The interface is independent of any ROS 2 or PLC transport. Future adapters should translate external messages into this same command contract rather than duplicating task or control logic.

## Commands

| Command | Intended precondition | Result |
|---|---|---|
| `HOME` | `INIT` or an allowed recoverable state | Homes the simulated robot and enters `READY` |
| `START_WRITING` | `READY` | Plans, validates, and executes the writing task |
| `STOP` | Active execution | Requests cooperative cancellation |
| `RESET` | `FAULT`, `STOPPED`, or `COMPLETE` | Clears task state and returns the application to a recoverable state |

Invalid commands are rejected rather than silently changing application state.

## Application States

The nominal lifecycle is:

```text
INIT → HOMING → READY → PLANNING → VALIDATING → EXECUTING → COMPLETE
                                              └───────────→ STOPPED
                                              └───────────→ FAULT
```

### State Responsibilities

| State | Meaning |
|---|---|
| `INIT` | Interface exists but the robot is not ready for a writing command |
| `HOMING` | Robot is moving to the defined collision-free home configuration |
| `READY` | Robot can accept `START_WRITING` |
| `PLANNING` | SVG waypoints are being converted into a timed writing plan |
| `VALIDATING` | Workspace, reachability, and collision preflight checks are running |
| `EXECUTING` | The validated plan is being executed |
| `COMPLETE` | Writing completed successfully |
| `STOPPED` | Cooperative cancellation was acknowledged |
| `FAULT` | Execution or validation failed and diagnostic state was recorded |

## Nominal Writing Cycle

1. Construct `RobotInterface`.
2. Confirm initial state `INIT`.
3. Issue `HOME`.
4. Confirm transition through `HOMING` to `READY`.
5. Issue `START_WRITING`.
6. Build the writing plan.
7. Validate workcell bounds, sampled reachability, and sampled collision state.
8. Execute while updating progress and telemetry.
9. Confirm `COMPLETE` and a completed experiment summary.
10. Return the simulated robot home and restore `READY`.

## Status Contract

The authoritative status can report:

- Current application state
- Active command or task
- Progress
- Current writing phase
- Contact state
- Normal force
- Cartesian tracking error
- Fault code
- Diagnostic message
- Last transition information

The GUI consumes this state but does not own the task lifecycle or control algorithm.

## Faults

Defined operational failures include:

- Command rejection
- Workspace violation
- Inverse-kinematics failure
- Controller or execution failure

A fault records diagnostic information and blocks normal execution until the required reset or recovery sequence is completed.

## STOP Semantics

`STOP` uses cooperative cancellation:

1. The interface receives the stop request.
2. The active writing task delegates cancellation to the running experiment.
3. Execution observes the cancellation request at a defined control-loop boundary.
4. The task reports stopped status.
5. The application enters `STOPPED`.

This is an operational software stop. It is not a safety-rated emergency stop, Safe Torque Off function, or certified protective stop.

## RESET Semantics

`RESET` clears task-local plan and execution state and returns the application to its defined recoverable state. Reset does not conceal an active failure; the caller must observe the resulting state before issuing another command.

## Test Coverage

Automated tests verify:

- Initial state
- Homing and transition to `READY`
- Rejection of writing outside `READY`
- Successful writing cycle
- Workspace failure and entry into `FAULT`
- Reset recovery from `FAULT`
- Cooperative STOP acknowledgement
- Telemetry propagation
- Reset after completion
- Legal and illegal state transitions
- Task planning, validation, execution, failure, stop, and reset behavior

## Future ROS 2 Adapter

A future ROS 2 layer should provide:

- `ExecuteWriting` action
- HOME and RESET services
- Robot-status topic
- Action feedback for phase and progress
- Action cancellation mapped to `STOP`
- Integration tests that preserve application semantics

## Future PLC Adapter

A future PLC handshake should preserve:

- Unique command ID
- Acknowledgement ID
- `Ready`, `Busy`, `Done`, and `Fault`
- Heartbeat/watchdog
- Duplicate-command protection
- Communication-loss behavior
- Explicit reset and recovery
- Separation between operational STOP and certified safety functions

The PLC should sequence the cell; the robotics application should retain responsibility for trajectory planning, kinematics, control, simulation, and detailed diagnostic status.
