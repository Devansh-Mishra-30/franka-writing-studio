"""Offscreen dashboard smoke for DRAW -> COMPLETE and media capture."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from ui.live_command_center import LiveCommandCenter


def draw_stroke(
    window: LiveCommandCenter,
    points: list[tuple[float, float]],
) -> None:
    window.drawing_pad._begin_stroke(
        QPointF(*points[0])
    )

    for point in points[1:-1]:
        window.drawing_pad._extend_stroke(
            QPointF(*point)
        )

    window.drawing_pad._end_stroke(
        QPointF(*points[-1])
    )


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = LiveCommandCenter()
    window.show()
    app.processEvents()

    width = float(window.drawing_pad.width())
    height = float(window.drawing_pad.height())
    left = 0.30 * width
    right = 0.70 * width
    top = 0.25 * height
    bottom = 0.75 * height
    middle = 0.50 * height

    draw_stroke(
        window,
        [(left, top), (left, bottom)],
    )
    draw_stroke(
        window,
        [(right, top), (right, bottom)],
    )
    draw_stroke(
        window,
        [(left, middle), (right, middle)],
    )

    if window.drawing_pad.stroke_count != 3:
        raise RuntimeError(
            "Dashboard did not preserve the three input strokes"
        )

    window.show_robot_checkbox.setChecked(False)
    window.full_plan_checkbox.setChecked(True)
    window._run_live()

    deadline = time.monotonic() + 180.0

    while window.worker_thread is not None:
        app.processEvents()

        if time.monotonic() > deadline:
            raise TimeoutError(
                "Dashboard drawing run did not finish in time"
            )

        time.sleep(0.005)

    app.processEvents()

    status = window.status_label.text()
    workflow = window.workflow_label.text()

    if "COMPLETE" not in status:
        raise RuntimeError(
            f"Dashboard did not reach COMPLETE: {status}"
        )

    if "[COMPLETE]" not in workflow:
        raise RuntimeError(
            f"Workflow indicator did not reach COMPLETE: {workflow}"
        )

    output_dir = (
        Path("artifacts/dashboard_smoke")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    control_path = output_dir / "dashboard_complete.png"
    telemetry_path = output_dir / "telemetry_complete.png"

    if not window.grab().save(str(control_path)):
        raise RuntimeError(
            "Failed to save dashboard screenshot"
        )

    if not window.telemetry_window.grab().save(
        str(telemetry_path)
    ):
        raise RuntimeError(
            "Failed to save telemetry screenshot"
        )

    print("DASHBOARD DRAW E2E: PASS")
    print(f"Status: {status}")
    print(f"Workflow: {workflow}")
    print(f"Dashboard: {control_path}")
    print(f"Telemetry: {telemetry_path}")

    window.telemetry_window.close()
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
