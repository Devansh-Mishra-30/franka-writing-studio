# Verification and Validation

## Purpose

This document defines the evidence supporting Franka Writing Studio v1.0. It separates software verification, simulation acceptance, cross-software agreement, and offline learning experiments from claims that would require physical hardware.

All writing-performance results reported here are simulation results.

## Release Configuration

The canonical v1 experiment uses:

- `svg/franka_portfolio.svg`
- A 1 ms simulation timestep
- The complete writing plan
- The pinned Franka description submodule
- Official-model joint limits
- A 1.0 N desired normal force
- Clean Git provenance

Canonical experiment commit:

```text
a12ad83d0ca8784b6059cd63f59b7ac2044c00ba
```

The canonical artifact was written to:

```text
artifacts/release/v1_0_0
```

## Automated Test Suite

The release suite contains 129 automated tests covering:

- SVG waypoint-source and trajectory behavior
- Trajectory sampling and timing
- Minimum-jerk motion profiles
- Differential inverse kinematics
- Tool orientation and pen-tip kinematics
- Contact and force-control behavior
- Pinocchio mechanics and dynamics
- PyBullet adapter behavior
- Planning and workcell validation
- Robot commands and legal state transitions
- Fault entry and RESET recovery
- Cooperative STOP behavior
- Experiment configuration
- Logging, metrics, and provenance

Run the suite with:

```bash
./wr test
```

GitHub Actions checks out the pinned Franka submodule, builds the Docker environment, and runs the same pytest suite for pushes and pull requests.

## Repository and Environment Gates

Run:

```bash
./wr doctor
./wr audit
```

`doctor` checks required tools, Docker access, required source/model files, the pinned Franka submodule, the development image, repository whitespace integrity, and GUI display availability.

`audit` reports repository hygiene, generated files, root-owned files, SVG inputs, and tracked-file count.

## End-to-End Acceptance

Run the canonical acceptance cycle with:

```bash
./wr e2e \
  --svg svg/franka_portfolio.svg \
  --output-dir artifacts/release/v1_0_0
```

The acceptance runner checks:

1. Initial application state is `INIT`.
2. HOME reaches `READY`.
3. Task execution reaches `COMPLETE`.
4. The experiment summary reports `completed`.
5. The complete writing plan was executed.
6. WRITE samples exist.
7. Contact and tracking metrics satisfy acceptance limits.
8. Force control does not saturate.
9. The robot returns home and the application returns to `READY`.

## Canonical Writing Results

| Acceptance metric | Result |
|---|---:|
| Task state | `COMPLETE` |
| Experiment status | `completed` |
| WRITE samples | 23,525 |
| WRITE contact rate | 100.00% |
| WRITE XY RMSE | 0.0848 mm |
| WRITE maximum XY error | 0.1556 mm |
| Mean normal force | 1.0375 N |
| Target normal force | 1.0000 N |
| Force RMS error | 0.0957 N |
| Force saturation count | 0 |
| Simulated steps | 56,560 |
| Wall-clock duration | 22.67 s |
| Real-time factor | 2.49x |
| Final robot state | `READY` |

## Result Figures

The canonical artifact generates:

- `docs/media/results/writing_path_tracking.png`
- `docs/media/results/xy_tracking_error.png`
- `docs/media/results/writing_force_control.png`
- `docs/media/results/execution_phases.png`

Regenerate them with:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "$PWD:/workspace" \
  --workdir /workspace \
  writing-robot:dev \
  python3 scripts/generate_results.py \
  artifacts/release/v1_0_0 \
  --output-dir docs/media/results
```

## Kinematic and Dynamic Verification

The Python verification stage exports 10 seeded configurations and associated Pinocchio results. Pinocchio internal inverse-dynamics reconstruction produced a maximum error of approximately `1.466e-14 N·m`.

MATLAB Robotics System Toolbox independently loads the same pinned URDF and recomputes the requested quantities.

Run:

```bash
./wr phase4
```

## Pinocchio-MATLAB Agreement

| Quantity | Maximum difference |
|---|---:|
| FK position | 1.357971e-16 m |
| FK rotation | 2.107342e-08 rad |
| Jacobian | 1.364945e-15 Frobenius norm |
| Mass matrix | 1.247479e-15 Frobenius norm |
| Velocity-product torque | 8.198225e-15 N·m |
| Gravity torque | 9.243609e-15 N·m |
| Inverse dynamics | 9.655752e-15 N·m |

This demonstrates cross-library implementation and convention agreement. Because both implementations use the same URDF, the comparison does not independently establish physical-model accuracy.

## Controlled Model Mismatch

The Phase 5 experiment increases the modeled mass and rotational inertia associated with joint 4 by 15%.

Run:

```bash
./wr phase5
```

Randomized dataset:

- 5,000 seeded states
- Residual torque RMS norm: 0.701464 N·m
- Maximum residual norm: 1.335360 N·m

Writing-trajectory evaluation:

- 126,960 evaluation samples
- 54,862 WRITE samples
- Overall residual RMS norm: 0.294519 N·m
- WRITE residual RMS norm: 0.303318 N·m
- WRITE maximum residual norm: 0.701433 N·m

## Residual Learning

Phase 6 trains a multivariate ridge model using 176 nonlinear dynamics-inspired features and the 5,000-state randomized dataset. It evaluates the model on the separate writing trajectory.

Run:

```bash
./wr phase6
```

| Metric | Result |
|---|---:|
| Training samples | 5,000 |
| Writing evaluation samples | 126,960 |
| Feature count | 176 |
| Physics-only residual RMS norm | 0.294519 N·m |
| Hybrid residual RMS norm | 0.073457 N·m |
| Reduction | 75.06% |

Per-joint RMS residuals:

| Joint | Physics only | Physics plus residual model |
|---|---:|---:|
| J1 | 0.006858 N·m | 0.001717 N·m |
| J2 | 0.231826 N·m | 0.049451 N·m |
| J3 | 0.058158 N·m | 0.053228 N·m |
| J4 | 0.171956 N·m | 0.010693 N·m |
| J5 | 0.000000 N·m | 0.000000 N·m |
| J6 | 0.000000 N·m | 0.000000 N·m |
| J7 | 0.000000 N·m | 0.000000 N·m |

This is an offline synthetic mismatch-correction experiment. It is not physical system identification, deep learning, online adaptation, or a deployed controller.

## Provenance

Experiment summaries record:

- Exact Git commit
- Dirty or clean working-tree state
- UTC creation timestamp
- Python version
- Dependency versions
- Runtime platform
- URDF path
- URDF SHA-256
- Experiment configuration
- Controller and contact-model metadata

## Claim Boundaries

The evidence supports claims about:

- Deterministic simulation execution
- Cartesian path tracking in PyBullet
- Simulated normal-force regulation
- Application lifecycle and fault behavior
- Software-level reproducibility
- Cross-library implementation agreement
- Offline synthetic residual learning

It does not support claims of:

- Physical Franka deployment
- Real-world submillimeter accuracy
- Physical force-control performance
- Safety certification
- Continuous collision-free planning
- ROS 2 or PLC integration
- Physical system identification
- Online adaptive control
- Arbitrary SVG compatibility

## Current Limitations

- Simulation only
- Physical pen rigidly attached to `fer_link8`
- Velocity-level normal-force admittance rather than torque impedance
- Sampled collision preflight rather than continuous collision checking
- No camera, calibration, or perception
- No physical I/O, ROS 2, or PLC adapter
- No safety PLC, guarding, certified E-stop chain, STO, SS1, or SLS validation
