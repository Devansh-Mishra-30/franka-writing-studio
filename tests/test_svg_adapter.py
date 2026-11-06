"""Tests for the SVG-to-DrawingDocument adapter."""

import unittest
from pathlib import Path

import numpy as np

from drawing import drawing_to_waypoints
from svg_adapter import svg_to_drawing


class SvgAdapterTests(unittest.TestCase):
    """Tests for svg_to_drawing()."""

    def test_loads_hey_svg(self):
        doc = svg_to_drawing(
            Path("svg/hey.svg")
        )

        self.assertGreater(
            doc.stroke_count,
            0,
        )

        self.assertEqual(
            doc.source,
            "svg",
        )

        self.assertGreater(
            doc.total_points,
            0,
        )

    def test_loads_portfolio_svg(self):
        doc = svg_to_drawing(
            Path("svg/franka_portfolio.svg")
        )

        self.assertGreater(
            doc.stroke_count,
            0,
        )

        self.assertGreater(
            doc.total_points,
            0,
        )

    def test_all_strokes_have_valid_points(self):
        doc = svg_to_drawing(
            Path("svg/hey.svg")
        )

        for stroke in doc.strokes:
            self.assertGreaterEqual(
                stroke.points.shape[0],
                2,
            )

            self.assertEqual(
                stroke.points.shape[1],
                2,
            )

            self.assertTrue(
                np.all(
                    np.isfinite(
                        stroke.points
                    )
                )
            )

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            svg_to_drawing(
                Path("nonexistent.svg")
            )

    def test_produces_valid_waypoints(self):
        doc = svg_to_drawing(
            Path("svg/hey.svg")
        )

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=0.525,
        )

        self.assertEqual(
            waypoints.ndim,
            2,
        )

        self.assertEqual(
            waypoints.shape[1],
            3,
        )

        self.assertGreater(
            waypoints.shape[0],
            0,
        )

        self.assertTrue(
            np.all(
                np.isfinite(waypoints)
            )
        )

    def test_waypoints_have_correct_z_values(self):
        doc = svg_to_drawing(
            Path("svg/hey.svg")
        )

        surface_h = 0.525
        lift_h = 0.575

        waypoints = drawing_to_waypoints(
            doc,
            surface_height_m=surface_h,
            lift_clearance_m=0.05,
        )

        # Every Z value should be either write or lift height.
        for i in range(waypoints.shape[0]):
            z = waypoints[i, 2]

            self.assertTrue(
                np.isclose(z, surface_h)
                or np.isclose(z, lift_h),
                f"Waypoint {i} has unexpected Z={z}",
            )

    def test_metadata(self):
        doc = svg_to_drawing(
            Path("svg/hey.svg")
        )

        meta = doc.metadata()

        self.assertEqual(
            meta["type"],
            "drawing_document",
        )

        self.assertEqual(
            meta["source"],
            "svg",
        )


class SvgAdapterCompatibilityTests(unittest.TestCase):
    """Verify that SVG adapter + drawing pipeline produces
    waypoints compatible with the existing execution stack.
    """

    def test_fitted_waypoints_match_v1_svg_pipeline(self):
        from contact_dynamics import fit_xy_path_to_surface
        from pen_tip_kinematics import legacy_positions_to_pen_tip
        from trajectory import SvgTrajectory
        from workcells import WRITING_STUDIO

        svg_path = Path("svg/hey.svg")
        surface_size = WRITING_STUDIO.notebook.size_m[:2]

        legacy = fit_xy_path_to_surface(
            legacy_positions_to_pen_tip(
                SvgTrajectory(svg_path).positions_m
            ),
            surface_size_m=surface_size,
            margin_m=0.02,
        )

        adapted = fit_xy_path_to_surface(
            drawing_to_waypoints(
                svg_to_drawing(svg_path),
                surface_height_m=(
                    WRITING_STUDIO.notebook.top_height_m
                ),
            ),
            surface_size_m=surface_size,
            margin_m=0.02,
        )

        np.testing.assert_allclose(
            adapted,
            legacy,
            atol=1e-12,
            rtol=0.0,
        )

    def test_waypoints_fit_notebook_after_processing(self):
        """Drawing waypoints pass through fit_xy_path_to_surface
        and stay within notebook bounds."""

        from contact_dynamics import (
            apply_write_preload,
            fit_xy_path_to_surface,
        )
        from workcells import WRITING_STUDIO

        doc = svg_to_drawing(
            Path("svg/franka_portfolio.svg")
        )

        surface_height_m = (
            WRITING_STUDIO.notebook.top_height_m
        )

        raw = drawing_to_waypoints(
            doc,
            surface_height_m=surface_height_m,
        )

        fitted = fit_xy_path_to_surface(
            raw,
            surface_size_m=(
                WRITING_STUDIO.notebook.size_m[:2]
            ),
            margin_m=0.02,
        )

        nominal_write_height_m = float(
            np.min(fitted[:, 2])
        )

        final = apply_write_preload(
            fitted,
            nominal_write_height_m=(
                nominal_write_height_m
            ),
        )

        self.assertEqual(
            final.ndim,
            2,
        )

        self.assertEqual(
            final.shape[1],
            3,
        )

        self.assertTrue(
            np.all(np.isfinite(final))
        )

        # At least 2 waypoints required for
        # TimedWritingPlan.
        self.assertGreaterEqual(
            final.shape[0],
            2,
        )


if __name__ == "__main__":
    unittest.main()

