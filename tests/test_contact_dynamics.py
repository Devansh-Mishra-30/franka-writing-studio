import unittest

import numpy as np

from contact_dynamics import (
    DEFAULT_PEN_CONTACT_PARAMETERS,
    apply_write_preload,
    fit_xy_path_to_surface,
    kelvin_voigt_normal_force,
)


class ContactDynamicsTests(unittest.TestCase):
    def test_no_penetration_produces_zero_force(self):
        force = kelvin_voigt_normal_force(
            penetration_m=0.0,
            penetration_velocity_m_s=0.0,
        )

        self.assertEqual(
            force,
            0.0,
        )

    def test_half_mm_preload_predicts_one_newton_static_force(self):
        force = kelvin_voigt_normal_force(
            penetration_m=0.0005,
            penetration_velocity_m_s=0.0,
        )

        self.assertAlmostEqual(
            force,
            1.0,
            places=12,
        )

    def test_damping_contributes_during_compression(self):
        static_force = kelvin_voigt_normal_force(
            penetration_m=0.0005,
            penetration_velocity_m_s=0.0,
        )

        dynamic_force = kelvin_voigt_normal_force(
            penetration_m=0.0005,
            penetration_velocity_m_s=0.002,
        )

        self.assertGreater(
            dynamic_force,
            static_force,
        )

    def test_symmetric_body_parameters_recover_effective_model(self):
        parameters = (
            DEFAULT_PEN_CONTACT_PARAMETERS
        )

        self.assertAlmostEqual(
            parameters
            .pybullet_body_contact_stiffness_n_m,
            4000.0,
        )

        self.assertAlmostEqual(
            parameters
            .pybullet_body_contact_damping_n_s_m,
            20.0,
        )

    def test_fit_xy_path_preserves_aspect_ratio_and_margin(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.525],
                [0.4, 0.0, 0.525],
                [0.4, 0.2, 0.525],
                [0.0, 0.2, 0.525],
            ],
            dtype=float,
        )

        result = fit_xy_path_to_surface(
            positions,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
        )

        span = np.ptp(
            result[:, :2],
            axis=0,
        )

        # Available notebook area:
        # X = 0.42 - 2(0.02) = 0.38 m
        # Y = 0.30 - 2(0.02) = 0.26 m
        #
        # Original aspect ratio is 2:1, so X is limiting.
        self.assertAlmostEqual(
            span[0],
            0.38,
            places=12,
        )

        self.assertAlmostEqual(
            span[1],
            0.19,
            places=12,
        )

        np.testing.assert_allclose(
            np.mean(
                [
                    np.min(result[:, :2], axis=0),
                    np.max(result[:, :2], axis=0),
                ],
                axis=0,
            ),
            [0.45, 0.0],
            atol=1e-12,
        )

    def test_fit_xy_path_does_not_change_height(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.575],
                [0.4, 0.4, 0.525],
            ],
            dtype=float,
        )

        result = fit_xy_path_to_surface(
            positions,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
        )

        np.testing.assert_allclose(
            result[:, 2],
            positions[:, 2],
            atol=0.0,
        )

    def test_fit_xy_path_preserves_aspect_ratio_and_margin(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.525],
                [0.4, 0.0, 0.525],
                [0.4, 0.2, 0.525],
                [0.0, 0.2, 0.525],
            ],
            dtype=float,
        )

        result = fit_xy_path_to_surface(
            positions,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
        )

        span = np.ptp(
            result[:, :2],
            axis=0,
        )

        # Available notebook area:
        # X = 0.42 - 2(0.02) = 0.38 m
        # Y = 0.30 - 2(0.02) = 0.26 m
        #
        # Original aspect ratio is 2:1, so X is limiting.
        self.assertAlmostEqual(
            span[0],
            0.38,
            places=12,
        )

        self.assertAlmostEqual(
            span[1],
            0.19,
            places=12,
        )

        np.testing.assert_allclose(
            np.mean(
                [
                    np.min(result[:, :2], axis=0),
                    np.max(result[:, :2], axis=0),
                ],
                axis=0,
            ),
            [0.45, 0.0],
            atol=1e-12,
        )

    def test_fit_xy_path_does_not_change_height(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.575],
                [0.4, 0.4, 0.525],
            ],
            dtype=float,
        )

        result = fit_xy_path_to_surface(
            positions,
            surface_size_m=(0.42, 0.30),
            margin_m=0.02,
        )

        np.testing.assert_allclose(
            result[:, 2],
            positions[:, 2],
            atol=0.0,
        )

    def test_preload_only_changes_write_height(self):
        positions = np.array(
            [
                [0.2, 0.1, 0.575],
                [0.2, 0.1, 0.525],
                [0.3, 0.1, 0.525],
                [0.3, 0.1, 0.575],
            ],
            dtype=float,
        )

        result = apply_write_preload(
            positions,
            nominal_write_height_m=0.525,
        )

        preload = (
            DEFAULT_PEN_CONTACT_PARAMETERS
            .preload_depth_m
        )

        np.testing.assert_allclose(
            result[:, 2],
            [
                0.575,
                0.525 - preload,
                0.525 - preload,
                0.575,
            ],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
