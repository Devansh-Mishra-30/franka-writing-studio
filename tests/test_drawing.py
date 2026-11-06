"""Tests for the common drawing representation."""

import unittest

import numpy as np

from drawing import (
    DrawingDocument,
    Stroke,
    drawing_to_waypoints,
    filter_stroke_points,
)


class StrokeTests(unittest.TestCase):
    """Tests for the Stroke data class."""

    def test_valid_stroke(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )

        stroke = Stroke(points=pts)

        self.assertEqual(
            stroke.points.shape,
            (2, 2),
        )

    def test_single_point_is_rejected(self):
        pts = np.array([[0.0, 0.0]])

        with self.assertRaises(ValueError):
            Stroke(points=pts)

    def test_empty_is_rejected(self):
        pts = np.zeros((0, 2))

        with self.assertRaises(ValueError):
            Stroke(points=pts)

    def test_wrong_shape_is_rejected(self):
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        with self.assertRaises(ValueError):
            Stroke(points=pts)

    def test_nan_is_rejected(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [float("nan"), 1.0],
            ]
        )

        with self.assertRaises(ValueError):
            Stroke(points=pts)

    def test_inf_is_rejected(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [float("inf"), 1.0],
            ]
        )

        with self.assertRaises(ValueError):
            Stroke(points=pts)

    def test_points_are_immutable_copy(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )

        stroke = Stroke(points=pts)
        pts[0, 0] = 99.0

        self.assertEqual(stroke.points[0, 0], 0.0)

        with self.assertRaises(ValueError):
            stroke.points[0, 0] = 42.0

    def test_points_are_float_array(self):
        pts = np.array(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=int,
        )

        stroke = Stroke(points=pts)

        self.assertEqual(
            stroke.points.dtype,
            float,
        )


class DrawingDocumentTests(unittest.TestCase):
    """Tests for the DrawingDocument data class."""

    def _make_stroke(
        self,
        n: int = 5,
    ) -> Stroke:
        return Stroke(
            points=np.column_stack(
                [
                    np.linspace(0.0, 10.0, n),
                    np.linspace(0.0, 10.0, n),
                ]
            )
        )

    def test_valid_document(self):
        doc = DrawingDocument(
            strokes=(self._make_stroke(),),
            source="test",
        )

        self.assertEqual(
            doc.stroke_count,
            1,
        )

        self.assertEqual(
            doc.source,
            "test",
        )

    def test_multiple_strokes(self):
        doc = DrawingDocument(
            strokes=(
                self._make_stroke(3),
                self._make_stroke(4),
                self._make_stroke(5),
            ),
            source="test",
        )

        self.assertEqual(
            doc.stroke_count,
            3,
        )

        self.assertEqual(
            doc.total_points,
            12,
        )

    def test_empty_strokes_rejected(self):
        with self.assertRaises(ValueError):
            DrawingDocument(
                strokes=(),
                source="test",
            )

    def test_empty_source_rejected(self):
        with self.assertRaises(ValueError):
            DrawingDocument(
                strokes=(self._make_stroke(),),
                source="",
            )

    def test_list_is_coerced_to_tuple(self):
        doc = DrawingDocument(
            strokes=[self._make_stroke()],
            source="test",
        )

        self.assertIsInstance(
            doc.strokes,
            tuple,
        )

    def test_bounding_box(self):
        stroke = Stroke(
            points=np.array(
                [
                    [1.0, 2.0],
                    [5.0, 8.0],
                    [3.0, 4.0],
                ]
            )
        )

        doc = DrawingDocument(
            strokes=(stroke,),
            source="test",
        )

        bb_min, bb_max = doc.bounding_box

        np.testing.assert_allclose(
            bb_min,
            [1.0, 2.0],
        )

        np.testing.assert_allclose(
            bb_max,
            [5.0, 8.0],
        )

    def test_invalid_canvas_size_rejected(self):
        with self.assertRaises(ValueError):
            DrawingDocument(
                strokes=(self._make_stroke(),),
                source="touchpad",
                canvas_size=(0.0, 300.0),
            )

    def test_non_stroke_member_rejected(self):
        with self.assertRaises(TypeError):
            DrawingDocument(
                strokes=(object(),),
                source="test",
            )

    def test_metadata(self):
        doc = DrawingDocument(
            strokes=(
                self._make_stroke(5),
                self._make_stroke(3),
            ),
            source="touchpad",
            canvas_size=(400, 300),
        )

        meta = doc.metadata()

        self.assertEqual(
            meta["type"],
            "drawing_document",
        )

        self.assertEqual(
            meta["source"],
            "touchpad",
        )

        self.assertEqual(
            meta["stroke_count"],
            2,
        )

        self.assertEqual(
            meta["total_points"],
            8,
        )


class DrawingToWaypointsTests(unittest.TestCase):
    """Tests for the drawing-to-waypoints conversion."""

    def _make_stroke(
        self,
        points: list[list[float]],
    ) -> Stroke:
        return Stroke(
            points=np.array(
                points,
                dtype=float,
            )
        )

    def _make_single_stroke_doc(self):
        return DrawingDocument(
            strokes=(
                self._make_stroke(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                    ]
                ),
            ),
            source="test",
        )

    def test_single_stroke_waypoint_structure(self):
        doc = self._make_single_stroke_doc()

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
            lift_clearance_m=0.05,
        )

        # Expected: lift, lower, 3 write points, final lift
        # = 2 + 3 + 1 = 6 waypoints
        # But first write point is part of the lower,
        # so: lift(0,0), lower(0,0), write(1,0), write(2,0), lift(2,0)
        # = 5 waypoints
        self.assertEqual(
            waypoints.shape[1],
            3,
        )

        self.assertTrue(
            np.all(np.isfinite(waypoints))
        )

    def test_pen_up_down_encoding(self):
        doc = self._make_single_stroke_doc()

        surface_h = 0.525
        lift_h = 0.575

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=surface_h,
            lift_clearance_m=0.05,
        )

        # First waypoint: pen up.
        self.assertAlmostEqual(
            waypoints[0, 2],
            lift_h,
        )

        # Second waypoint: pen down (lower).
        self.assertAlmostEqual(
            waypoints[1, 2],
            surface_h,
        )

        # Last waypoint: pen up.
        self.assertAlmostEqual(
            waypoints[-1, 2],
            lift_h,
        )

    def test_multi_stroke_has_pen_lifts(self):
        doc = DrawingDocument(
            strokes=(
                self._make_stroke(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                    ]
                ),
                self._make_stroke(
                    [
                        [5.0, 5.0],
                        [6.0, 5.0],
                    ]
                ),
            ),
            source="test",
        )

        surface_h = 0.525
        lift_h = 0.575

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=surface_h,
            lift_clearance_m=0.05,
        )

        # Count pen-up waypoints.
        pen_up_count = int(
            np.sum(
                np.isclose(
                    waypoints[:, 2],
                    lift_h,
                )
            )
        )

        # At least: initial lift, inter-stroke lift+transfer,
        # final lift.
        self.assertGreaterEqual(
            pen_up_count,
            3,
        )

    def test_multi_stroke_preserves_order(self):
        doc = DrawingDocument(
            strokes=(
                self._make_stroke(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                    ]
                ),
                self._make_stroke(
                    [
                        [10.0, 10.0],
                        [11.0, 10.0],
                    ]
                ),
            ),
            source="test",
        )

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        write_xy = waypoints[
            np.isclose(
                waypoints[:, 2],
                0.525,
            ),
            :2,
        ]

        # Display coordinates map to the established notebook-plane
        # convention, but stroke and point order remain unchanged.
        np.testing.assert_allclose(
            write_xy,
            [
                [0.0, 0.0],
                [0.0, -1.0],
                [-10.0, -10.0],
                [-10.0, -11.0],
            ],
        )

    def test_aspect_ratio_preserved_by_fit(self):
        """Drawing coords feed through fit_xy_path_to_surface."""

        from contact_dynamics import fit_xy_path_to_surface

        # 2:1 aspect ratio drawing.
        doc = DrawingDocument(
            strokes=(
                self._make_stroke(
                    [
                        [0.0, 0.0],
                        [200.0, 0.0],
                        [200.0, 100.0],
                        [0.0, 100.0],
                    ]
                ),
            ),
            source="test",
        )

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        fitted = fit_xy_path_to_surface(
            waypoints,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
        )

        # Check XY range preserves aspect ratio.
        xy_min = np.min(
            fitted[:, :2],
            axis=0,
        )

        xy_max = np.max(
            fitted[:, :2],
            axis=0,
        )

        x_span = xy_max[0] - xy_min[0]
        y_span = xy_max[1] - xy_min[1]

        # Original was 2:1 in SVG coords.
        # After scale_coords_to_arena the axes get swapped,
        # but fit_xy_path_to_surface preserves aspect ratio.
        if x_span > 0 and y_span > 0:
            ratio = max(
                x_span / y_span,
                y_span / x_span,
            )

            self.assertAlmostEqual(
                ratio,
                2.0,
                places=4,
            )

    def test_negative_lift_clearance_rejected(self):
        doc = self._make_single_stroke_doc()

        with self.assertRaises(ValueError):
            drawing_to_waypoints(
                doc,
                surface_height_m=0.525,
                lift_clearance_m=-0.01,
            )

    def test_zero_lift_clearance_rejected(self):
        doc = self._make_single_stroke_doc()

        with self.assertRaises(ValueError):
            drawing_to_waypoints(
                doc,
                surface_height_m=0.525,
                lift_clearance_m=0.0,
            )

    def test_deterministic(self):
        doc = self._make_single_stroke_doc()

        w1 = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        w2 = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        np.testing.assert_array_equal(
            w1,
            w2,
        )


class FilterStrokePointsTests(unittest.TestCase):
    """Tests for distance-based point filtering."""

    def test_removes_close_points(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
            ]
        )

        filtered = filter_stroke_points(
            pts,
            min_distance=1.0,
        )

        # Points at 0.1 and 0.2 are too close.
        self.assertLess(
            filtered.shape[0],
            pts.shape[0],
        )

    def test_always_keeps_first_and_last(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [0.01, 0.0],
                [0.02, 0.0],
                [100.0, 0.0],
            ]
        )

        filtered = filter_stroke_points(
            pts,
            min_distance=50.0,
        )

        np.testing.assert_allclose(
            filtered[0],
            pts[0],
        )

        np.testing.assert_allclose(
            filtered[-1],
            pts[-1],
        )

    def test_two_point_input_unchanged(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [0.001, 0.0],
            ]
        )

        filtered = filter_stroke_points(
            pts,
            min_distance=100.0,
        )

        self.assertEqual(
            filtered.shape[0],
            2,
        )

    def test_wrong_shape_rejected(self):
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        with self.assertRaises(ValueError):
            filter_stroke_points(pts)

    def test_non_finite_points_rejected(self):
        with self.assertRaises(ValueError):
            filter_stroke_points(
                np.array(
                    [
                        [0.0, 0.0],
                        [np.nan, 1.0],
                    ]
                )
            )

    def test_non_finite_distance_rejected(self):
        with self.assertRaises(ValueError):
            filter_stroke_points(
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 1.0],
                    ]
                ),
                min_distance=np.nan,
            )

    def test_zero_distance_keeps_all_unique(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

        filtered = filter_stroke_points(
            pts,
            min_distance=0.0,
        )

        self.assertEqual(
            filtered.shape[0],
            pts.shape[0],
        )


class CanvasPlacementTests(unittest.TestCase):
    def test_canvas_bounds_preserve_whitespace(self):
        from contact_dynamics import fit_xy_path_to_surface

        document = DrawingDocument(
            strokes=(
                Stroke(
                    np.array(
                        [
                            [40.0, 40.0],
                            [60.0, 60.0],
                        ]
                    )
                ),
            ),
            source="touchpad",
            canvas_size=(100.0, 100.0),
        )

        waypoints = drawing_to_waypoints(
            document,
            surface_height_m=0.525,
        )
        fitted = fit_xy_path_to_surface(
            waypoints,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
            source_bounds_xy=(
                np.array([-100.0, -100.0]),
                np.array([0.0, 0.0]),
            ),
        )

        span = np.ptp(
            fitted[:, :2],
            axis=0,
        )
        self.assertLessEqual(
            float(np.max(span)),
            0.2 * 0.26 + 1e-12,
        )

    def test_source_bounds_must_contain_path(self):
        from contact_dynamics import fit_xy_path_to_surface

        with self.assertRaises(ValueError):
            fit_xy_path_to_surface(
                np.array(
                    [
                        [0.0, 0.0, 0.525],
                        [10.0, 10.0, 0.525],
                    ]
                ),
                surface_size_m=(0.42, 0.30),
                source_bounds_xy=(
                    np.array([0.0, 0.0]),
                    np.array([5.0, 5.0]),
                ),
            )


class NotebookBoundsTests(unittest.TestCase):
    """Test that drawing waypoints stay within notebook after fitting."""

    def test_waypoints_within_notebook(self):
        from contact_dynamics import (
            fit_xy_path_to_surface,
        )
        from workcells import WRITING_STUDIO

        # Large drawing that needs scaling down.
        stroke = Stroke(
            points=np.array(
                [
                    [0.0, 0.0],
                    [1000.0, 0.0],
                    [1000.0, 800.0],
                    [0.0, 800.0],
                    [0.0, 0.0],
                ]
            )
        )

        doc = DrawingDocument(
            strokes=(stroke,),
            source="test",
        )

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        fitted = fit_xy_path_to_surface(
            waypoints,
            surface_size_m=(
                WRITING_STUDIO.notebook.size_m[:2]
            ),
            margin_m=0.02,
        )

        notebook = WRITING_STUDIO.notebook
        half_x = 0.5 * notebook.size_m[0]
        half_y = 0.5 * notebook.size_m[1]
        center_x = notebook.center_m[0]
        center_y = notebook.center_m[1]

        tolerance = 1e-9

        self.assertTrue(
            np.all(
                fitted[:, 0]
                >= center_x - half_x - tolerance
            ),
            "X below notebook lower bound",
        )

        self.assertTrue(
            np.all(
                fitted[:, 0]
                <= center_x + half_x + tolerance
            ),
            "X above notebook upper bound",
        )

        self.assertTrue(
            np.all(
                fitted[:, 1]
                >= center_y - half_y - tolerance
            ),
            "Y below notebook lower bound",
        )

        self.assertTrue(
            np.all(
                fitted[:, 1]
                <= center_y + half_y + tolerance
            ),
            "Y above notebook upper bound",
        )


if __name__ == "__main__":
    unittest.main()

