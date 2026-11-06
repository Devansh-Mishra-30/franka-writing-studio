"""Behavior tests for the reusable drawing pad."""

import os
import unittest

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from ui.drawing_pad import DrawingPadWidget


class DrawingPadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pad = DrawingPadWidget()
        self.pad.resize(320, 240)

    def draw_stroke(self, points):
        self.pad._begin_stroke(QPointF(*points[0]))

        for point in points[1:-1]:
            self.pad._extend_stroke(QPointF(*point))

        self.pad._end_stroke(QPointF(*points[-1]))

    def test_multiple_strokes_preserve_order(self):
        first = [(10.0, 10.0), (30.0, 20.0)]
        second = [(100.0, 80.0), (120.0, 90.0)]

        self.draw_stroke(first)
        self.draw_stroke(second)

        drawing = self.pad.get_drawing()

        self.assertIsNotNone(drawing)
        self.assertEqual(drawing.stroke_count, 2)
        self.assertEqual(drawing.source, "touchpad")
        self.assertEqual(drawing.canvas_size, (320.0, 240.0))

        self.assertEqual(
            tuple(drawing.strokes[0].points[0]),
            first[0],
        )
        self.assertEqual(
            tuple(drawing.strokes[1].points[0]),
            second[0],
        )

    def test_tap_and_short_stroke_are_ignored(self):
        self.pad._begin_stroke(QPointF(10.0, 10.0))
        self.pad._end_stroke(QPointF(10.0, 10.0))

        self.pad._begin_stroke(QPointF(20.0, 20.0))
        self.pad._end_stroke(QPointF(22.0, 20.0))

        self.assertFalse(self.pad.has_drawing)
        self.assertIsNone(self.pad.get_drawing())

    def test_undo_and_clear(self):
        self.draw_stroke([(0.0, 0.0), (10.0, 0.0)])
        self.draw_stroke([(20.0, 0.0), (30.0, 0.0)])

        self.pad.undo_last_stroke()
        self.assertEqual(self.pad.stroke_count, 1)

        self.pad.clear_drawing()
        self.assertEqual(self.pad.stroke_count, 0)
        self.assertIsNone(self.pad.get_drawing())


if __name__ == "__main__":
    unittest.main()
