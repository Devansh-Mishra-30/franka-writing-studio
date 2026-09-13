# Official Franka Model Validation

## Configuration

- Pinocchio model: official FER arm model
- PyBullet model: sanitized official FER arm and hand model
- Shared validation frame: `fer_link8`
- Simulation timestep: 0.001 s
- Smoke-test duration: 0.005 s
- Samples: 5

## Kinematic Cross-Validation

| Metric | Value |
|---|---:|
| Mean disagreement | 1.089610295490e-08 m |
| Maximum disagreement | 1.551357423155e-08 m |
| Mean disagreement | 10.896 nm |
| Maximum disagreement | 15.514 nm |

## Result

**PASS**

Pinocchio and PyBullet agree on the named `fer_link8` position
to within 15.514 nanometers.

This validates model topology, joint ordering, named-frame
resolution, initial joint-state mapping, and forward-kinematics
consistency.

## Scope Limitations

This smoke test does not validate trajectory tracking, controller
convergence, inverse dynamics, contact behavior, pen-tip geometry,
or real-time performance.

## Automated Regression Result

- Test suite: Python `unittest`
- Tests executed: 16
- Tests passed: 16
- Missing-inertia warnings: 0
- Bad-inertia warnings: 0
- Missing named-frame errors: 0
