# Changelog

All notable release-level changes to Franka Writing Studio are documented here.

## [v1.1.0] - 2026-11-06

### Added

- Interactive freehand drawing input in the PySide6 command center.
- Input-agnostic `DrawingDocument` and ordered multi-stroke representation.
- Backward-compatible SVG adapter feeding the same common drawing pipeline.
- Mouse input with best-effort touchscreen/stylus compatibility.
- Undo Last Stroke and Clear controls.
- Canvas-aware notebook placement that preserves natural whitespace.
- Freehand drawing and dashboard end-to-end acceptance runners.
- Persistent PyBullet scene after successful GUI completion.
- Isometric, Top, Notebook, and Robot camera presets.
- Dashboard orbit, tilt, zoom, and pan camera controls.
- Engineering-telemetry MP4 recording at 20 FPS.
- Dual PyBullet and telemetry recording in the same timestamped run directory.

### Changed

- SVG and freehand input now converge before planning instead of maintaining separate execution paths.
- The live dashboard now exposes the workflow `INPUT -> PLAN -> VALIDATE -> EXECUTE -> COMPLETE`.
- Simulator lifetime is explicitly managed across COMPLETE, RESET, STOP, new runs, and application close.
- Completed PyBullet video recording is finalized even when the final scene remains open.
- Automated release gate expanded from 129 to 181 tests.

### Verified

- 181 automated tests passing.
- Canonical SVG E2E: PASS.
- Three independent freehand drawing E2Es: PASS.
- Dashboard DRAW E2E: PASS.
- `./wr doctor`: PASS.
- `./wr audit`: COMPLETE.
- Canonical SVG WRITE contact: 100.00%.
- Canonical SVG WRITE XY RMSE: 0.0685 mm.
- Canonical SVG mean WRITE force: 0.9908 N for a 1.0000 N target.
- Canonical SVG force RMS error: 0.0692 N.
- Canonical SVG force-control saturation count: 0.

### Limitations

- Simulation only; not deployed on a physical Franka.
- No ROS 2 or PLC adapter in this release.
- No perception or physical work-object calibration.
- No safety-rated control or certified safety chain.
- Collision preflight remains sampled rather than continuous.
- The pen remains rigidly attached to the simulated end effector.

## [v1.0.0]

Initial validated Franka Writing Studio platform release with SVG-driven writing, hybrid Cartesian position/force control, PyBullet execution, Pinocchio mechanics, MATLAB verification, model-mismatch experiments, residual learning, reproducible Docker workflows, application lifecycle management, and automated acceptance testing.
