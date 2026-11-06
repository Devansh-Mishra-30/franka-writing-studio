"""Reusable PySide6 drawing canvas for interactive stroke input.

This widget captures mouse, touchscreen, and stylus input as
multi-stroke 2D drawings.  It emits drawing data only — it does
not perform any robot control.

Behavior
--------
- Pointer-down starts a new stroke.
- Pointer movement appends points to the active stroke.
- Pointer-up ends the stroke.
- Zero-length strokes (no movement) are discarded.
- Distance-based point filtering prevents every raw event
  from becoming a trajectory waypoint.
- Multiple strokes are accumulated in order.
- Undo removes the last stroke.
- Clear removes all strokes.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import (
    QEvent,
    QPointF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from drawing import (
    DrawingDocument,
    Stroke,
    filter_stroke_points,
)


# Minimum squared pixel distance between consecutive raw points.
# Points closer than this are skipped during capture.
_MIN_CAPTURE_DISTANCE_PX = 3.0

# Minimum pixel distance for post-capture stroke filtering.
_MIN_FILTER_DISTANCE_PX = 2.0

# A stroke must traverse at least this total pixel distance
# to be considered valid (rejects accidental taps/clicks).
_MIN_STROKE_LENGTH_PX = 5.0


class DrawingPadWidget(QWidget):
    """Interactive drawing canvas producing DrawingDocument data.

    Signals
    -------
    drawing_changed
        Emitted whenever a stroke is added, undone, or cleared.
        The int payload is the current stroke count.
    """

    drawing_changed = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setMinimumSize(200, 200)

        self.setAttribute(
            Qt.WidgetAttribute.WA_AcceptTouchEvents,
            True,
        )

        # Accumulated completed strokes.
        self._strokes: list[list[QPointF]] = []

        # Active stroke being drawn (None when idle).
        self._active_stroke: list[QPointF] | None = (
            None
        )

        # Visual styling.
        self._bg_color = QColor(255, 255, 255)
        self._stroke_color = QColor(30, 30, 30)
        self._active_color = QColor(0, 120, 215)
        self._stroke_width = 2.5

        self.setStyleSheet(
            "border: 1px solid #ccc; "
            "border-radius: 4px;"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    @property
    def stroke_count(self) -> int:
        return len(self._strokes)

    @property
    def has_drawing(self) -> bool:
        return len(self._strokes) > 0

    def get_drawing(self) -> DrawingDocument | None:
        """Return the current drawing as a DrawingDocument.

        Returns None if no valid strokes exist.
        Coordinates are in widget pixel space.
        """

        valid_strokes: list[Stroke] = []

        for raw_points in self._strokes:
            pts = np.array(
                [
                    [p.x(), p.y()]
                    for p in raw_points
                ],
                dtype=float,
            )

            filtered = filter_stroke_points(
                pts,
                min_distance=_MIN_FILTER_DISTANCE_PX,
            )

            if filtered.shape[0] >= 2:
                valid_strokes.append(
                    Stroke(points=filtered)
                )

        if not valid_strokes:
            return None

        return DrawingDocument(
            strokes=tuple(valid_strokes),
            source="touchpad",
            canvas_size=(
                float(self.width()),
                float(self.height()),
            ),
        )

    def undo_last_stroke(self) -> None:
        """Remove the last completed stroke."""

        if self._strokes:
            self._strokes.pop()
            self.update()

            self.drawing_changed.emit(
                len(self._strokes)
            )

    def clear_drawing(self) -> None:
        """Remove all strokes."""

        self._strokes.clear()
        self._active_stroke = None
        self.update()

        self.drawing_changed.emit(0)

    # ----------------------------------------------------------
    # Mouse events
    # ----------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._begin_stroke(
                event.position()
            )

            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._active_stroke is not None:
            self._extend_stroke(
                event.position()
            )

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._active_stroke is not None
        ):
            self._end_stroke(
                event.position()
            )
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ----------------------------------------------------------
    # Touch events (stylus / touchscreen)
    # ----------------------------------------------------------

    def event(self, event) -> bool:
        event_type = event.type()

        if event_type in (
            QEvent.Type.TouchBegin,
            QEvent.Type.TouchUpdate,
            QEvent.Type.TouchEnd,
        ):
            points = event.points()

            if not points:
                return True

            position = points[0].position()

            if event_type == QEvent.Type.TouchBegin:
                self._begin_stroke(position)
            elif event_type == QEvent.Type.TouchUpdate:
                self._extend_stroke(position)
            elif self._active_stroke is not None:
                self._end_stroke(position)

            event.accept()
            return True

        return super().event(event)

    def tabletEvent(self, event) -> None:
        """Handle tablet/stylus events for pressure-aware devices."""

        event_type = event.type()

        if event_type == event.Type.TabletPress:
            self._begin_stroke(
                event.position()
            )
            event.accept()

        elif event_type == event.Type.TabletMove:
            if self._active_stroke is not None:
                self._extend_stroke(
                    event.position()
                )
            event.accept()

        elif event_type == event.Type.TabletRelease:
            if self._active_stroke is not None:
                self._end_stroke(
                    event.position()
                )
            event.accept()

        else:
            super().tabletEvent(event)

    # ----------------------------------------------------------
    # Stroke management
    # ----------------------------------------------------------

    def _bounded_position(
        self,
        pos: QPointF,
    ) -> QPointF:
        return QPointF(
            min(
                max(pos.x(), 0.0),
                float(self.width()),
            ),
            min(
                max(pos.y(), 0.0),
                float(self.height()),
            ),
        )

    def _begin_stroke(
        self,
        pos: QPointF,
    ) -> None:
        bounded = self._bounded_position(pos)
        self._active_stroke = [
            bounded
        ]

        self.update()

    def _extend_stroke(
        self,
        pos: QPointF,
    ) -> None:
        if self._active_stroke is None:
            return

        bounded = self._bounded_position(pos)
        last = self._active_stroke[-1]

        dx = bounded.x() - last.x()
        dy = bounded.y() - last.y()
        dist_sq = dx * dx + dy * dy

        if dist_sq < (
            _MIN_CAPTURE_DISTANCE_PX
            * _MIN_CAPTURE_DISTANCE_PX
        ):
            return

        self._active_stroke.append(
            bounded
        )

        self.update()

    def _end_stroke(
        self,
        final_position: QPointF | None = None,
    ) -> None:
        if (
            final_position is not None
            and self._active_stroke is not None
        ):
            self._extend_stroke(
                final_position
            )

        stroke = self._active_stroke
        self._active_stroke = None

        if stroke is None or len(stroke) < 2:
            self.update()
            return

        # Compute total stroke length.
        total_length = 0.0

        for i in range(1, len(stroke)):
            dx = stroke[i].x() - stroke[i - 1].x()
            dy = stroke[i].y() - stroke[i - 1].y()
            total_length += (
                dx * dx + dy * dy
            ) ** 0.5

        if total_length < _MIN_STROKE_LENGTH_PX:
            self.update()
            return

        self._strokes.append(stroke)
        self.update()

        self.drawing_changed.emit(
            len(self._strokes)
        )

    # ----------------------------------------------------------
    # Painting
    # ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        # Background.
        painter.fillRect(
            self.rect(),
            self._bg_color,
        )

        # Completed strokes.
        stroke_pen = QPen(
            self._stroke_color,
            self._stroke_width,
        )

        stroke_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )

        stroke_pen.setJoinStyle(
            Qt.PenJoinStyle.RoundJoin
        )

        painter.setPen(stroke_pen)

        for stroke_points in self._strokes:
            if len(stroke_points) < 2:
                continue

            path = QPainterPath(
                stroke_points[0]
            )

            for pt in stroke_points[1:]:
                path.lineTo(pt)

            painter.drawPath(path)

        # Active stroke (in progress).
        if (
            self._active_stroke is not None
            and len(self._active_stroke) >= 2
        ):
            active_pen = QPen(
                self._active_color,
                self._stroke_width,
            )

            active_pen.setCapStyle(
                Qt.PenCapStyle.RoundCap
            )

            active_pen.setJoinStyle(
                Qt.PenJoinStyle.RoundJoin
            )

            painter.setPen(active_pen)

            path = QPainterPath(
                self._active_stroke[0]
            )

            for pt in self._active_stroke[1:]:
                path.lineTo(pt)

            painter.drawPath(path)

        painter.end()

