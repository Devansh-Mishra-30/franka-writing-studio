"""Validation tests for Franka rigid-body dynamics."""

from __future__ import annotations

import unittest

import numpy as np

from franka_mechanics import FrankaMechanics


class TestFrankaDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mechanics = FrankaMechanics()

        # Previously validated collision-free Franka configuration.
        cls.q = np.array(
            [
                0.23081834,
                -0.69476370,
                -0.12918779,
                -2.02518342,
                -0.08491880,
                1.33535219,
                0.11158610,
            ],
            dtype=float,
        )

        cls.dq = np.array(
            [
                0.15,
                -0.10,
                0.08,
                -0.12,
                0.05,
                0.09,
                -0.04,
            ],
            dtype=float,
        )

        cls.ddq = np.array(
            [
                0.30,
                -0.20,
                0.15,
                0.10,
                -0.12,
                0.18,
                -0.08,
            ],
            dtype=float,
        )

    def test_mass_matrix_shape_and_finiteness(self) -> None:
        mass = self.mechanics.get_M(
            self.q
        )

        self.assertEqual(
            mass.shape,
            (7, 7),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    mass
                )
            )
        )

    def test_mass_matrix_is_symmetric(self) -> None:
        mass = self.mechanics.get_M(
            self.q
        )

        np.testing.assert_allclose(
            mass,
            mass.T,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_mass_matrix_is_positive_definite(self) -> None:
        mass = self.mechanics.get_M(
            self.q
        )

        eigenvalues = np.linalg.eigvalsh(
            mass
        )

        self.assertTrue(
            np.all(
                eigenvalues > 0.0
            ),
            msg=(
                "Mass-matrix eigenvalues must "
                f"be positive; got {eigenvalues}"
            ),
        )

    def test_coriolis_matrix_shape(self) -> None:
        coriolis = self.mechanics.get_C(
            self.q,
            self.dq,
        )

        self.assertEqual(
            coriolis.shape,
            (7, 7),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    coriolis
                )
            )
        )

    def test_gravity_vector_shape(self) -> None:
        gravity = self.mechanics.get_G(
            self.q
        )

        self.assertEqual(
            gravity.shape,
            (7,),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    gravity
                )
            )
        )

    def test_static_rnea_equals_gravity(self) -> None:
        zeros = np.zeros(
            7,
            dtype=float,
        )

        tau = self.mechanics.get_tau(
            self.q,
            zeros,
            zeros,
        )

        gravity = self.mechanics.get_G(
            self.q
        )

        np.testing.assert_allclose(
            tau,
            gravity,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_rnea_matches_dynamic_decomposition(self) -> None:
        tau_rnea = self.mechanics.get_tau(
            self.q,
            self.dq,
            self.ddq,
        )

        tau_decomposed = (
            self.mechanics.get_tau_decomposed(
                self.q,
                self.dq,
                self.ddq,
            )
        )

        np.testing.assert_allclose(
            tau_rnea,
            tau_decomposed,
            rtol=1.0e-9,
            atol=1.0e-9,
        )

    def test_dynamics_outputs_have_correct_shape(self) -> None:
        tau = self.mechanics.get_tau(
            self.q,
            self.dq,
            self.ddq,
        )

        decomposed = (
            self.mechanics.get_tau_decomposed(
                self.q,
                self.dq,
                self.ddq,
            )
        )

        self.assertEqual(
            tau.shape,
            (7,),
        )

        self.assertEqual(
            decomposed.shape,
            (7,),
        )


if __name__ == "__main__":
    unittest.main()
