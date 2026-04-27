"""Unit-test scaffold for `rebus_synthesis.identification`.

This file uses Python's built-in `unittest` framework so it can run without
pytest. Solver-dependent smoke tests are skipped automatically when cvxpy is
not installed.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rebus_synthesis import identification as rid


class TestSyntheticScaffold(unittest.TestCase):
    def test_synthetic_data_shapes_and_keys(self) -> None:
        data, truth = rid.make_synthetic_rebus_data(T=48, nx=3, seed=11)
        required = {
            "X_t",
            "U_t",
            "X_tp1",
            "alpha_t",
            "alpha_tp1",
            "omega_t",
            "chi_t",
            "e_t",
            "e_tp1",
            "alpha_t_for_e",
            "nu_t",
            "strain_delta_t",
            "alpha_t_for_b",
            "nu_t_for_b",
            "alpha_bar",
            "e_bar",
            "omega_bar",
            "barV",
        }
        self.assertTrue(required.issubset(data.keys()))
        self.assertEqual(data["X_t"].shape, (3, 48))
        self.assertEqual(data["U_t"].shape, (3, 48))
        self.assertEqual(data["X_tp1"].shape, (3, 48))
        self.assertGreater(float(data["barV"]), 0.0)
        self.assertTrue(np.all(np.isfinite(truth.A)))

    def test_one_sided_bounds_helper(self) -> None:
        samples = [
            {
                "phi_alpha": 0.75,
                "phi_e": 0.65,
                "psi_e": 0.15,
                "kappa": 0.18,
                "c_alpha": 0.30,
                "c_nu": 0.50,
            },
            {
                "phi_alpha": 0.80,
                "phi_e": 0.70,
                "psi_e": 0.20,
                "kappa": 0.22,
                "c_alpha": 0.45,
                "c_nu": 0.55,
            },
        ]
        out = rid.one_sided_bounds(samples, delta_alpha=0.1, delta_e=0.1, delta_b=0.1)
        self.assertGreaterEqual(out["phi_alpha_ub"], 0.75)
        self.assertGreaterEqual(out["phi_e_ub"], 0.65)
        self.assertGreaterEqual(out["kappa_ub"], out["psi_e_ub"])
        self.assertGreater(out["c_alpha_ub"], 0.0)
        self.assertGreater(out["c_nu_ub"], 0.0)
        self.assertLess(out["lambda_lb"], 1.0)
        self.assertLess(out["mu_lb"], 1.0)

    def test_gain_synthesis_returns_positive_scalars(self) -> None:
        bounds = {
            "a_lb": 0.12,
            "lambda_lb": 0.20,
            "mu_lb": 0.25,
            "kappa_ub": 0.30,
            "b1_ub": 0.18,
            "b2_ub": 0.16,
            "b3_ub": 0.21,
        }
        gains = rid.synthesize_supervisor_gains(bounds, eta=0.25)
        self.assertGreater(gains.p, 0.0)
        self.assertGreater(gains.q, 0.0)
        self.assertGreater(gains.Gamma, bounds["b3_ub"])
        self.assertGreater(gains.delta_safe, 0.0)

    def test_missing_keys_raise(self) -> None:
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=3)
        broken = dict(data)
        broken.pop("U_t")
        with self.assertRaises(KeyError):
            rid.identify_rebus_bounds(broken, B=4, block_len=8)

    def test_bootstrap_pipeline_misaligned_nu_t_raises(self) -> None:
        # nu_t one sample shorter than e_t would otherwise hit a low-level
        # np.take IndexError mid-loop in bootstrap_take. The pre-loop
        # validation should turn that into a clear ValueError. Calling
        # bootstrap_pipeline directly isolates the new check from any
        # upstream errors raised by nominal fits in identify_rebus_bounds.
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=4)
        broken = dict(data)
        broken["nu_t"] = np.asarray(data["nu_t"], dtype=float)[:-1]
        with self.assertRaises(ValueError) as ctx:
            rid.bootstrap_pipeline(broken, B=4, block_len=8)
        msg = str(ctx.exception)
        self.assertIn("nu_t", msg)
        self.assertIn("e_t", msg)

    def test_bootstrap_pipeline_misaligned_alpha_t_for_e_raises(self) -> None:
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=4)
        broken = dict(data)
        broken["alpha_t_for_e"] = np.asarray(data["alpha_t_for_e"], dtype=float)[:-1]
        with self.assertRaises(ValueError) as ctx:
            rid.bootstrap_pipeline(broken, B=4, block_len=8)
        self.assertIn("alpha_t_for_e", str(ctx.exception))

    def test_bootstrap_pipeline_misaligned_e_tp1_raises(self) -> None:
        # Excess-gate fit also reuses idx_e on e_tp1; same IndexError risk.
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=4)
        broken = dict(data)
        broken["e_tp1"] = np.asarray(data["e_tp1"], dtype=float)[:-1]
        with self.assertRaises(ValueError) as ctx:
            rid.bootstrap_pipeline(broken, B=4, block_len=8)
        self.assertIn("e_tp1", str(ctx.exception))

    def test_bootstrap_pipeline_misaligned_omega_t_raises(self) -> None:
        # Alpha-gate branch: idx_a is drawn from len(alpha_t) and reused on
        # alpha_tp1 / omega_t / chi_t. omega_t shorter must surface as ValueError.
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=4)
        broken = dict(data)
        broken["omega_t"] = np.asarray(data["omega_t"], dtype=float)[:-1]
        with self.assertRaises(ValueError) as ctx:
            rid.bootstrap_pipeline(broken, B=4, block_len=8)
        msg = str(ctx.exception)
        self.assertIn("omega_t", msg)
        self.assertIn("alpha_t", msg)

    def test_bootstrap_pipeline_misaligned_nu_t_for_b_raises(self) -> None:
        # Budget branch: idx_b is drawn from len(strain_delta_t) and reused on
        # alpha_t_for_b / nu_t_for_b. nu_t_for_b shorter must surface as ValueError.
        data, _ = rid.make_synthetic_rebus_data(T=32, nx=2, seed=4)
        broken = dict(data)
        broken["nu_t_for_b"] = np.asarray(data["nu_t_for_b"], dtype=float)[:-1]
        with self.assertRaises(ValueError) as ctx:
            rid.bootstrap_pipeline(broken, B=4, block_len=8)
        msg = str(ctx.exception)
        self.assertIn("nu_t_for_b", msg)
        self.assertIn("strain_delta_t", msg)

    def test_smoke_test_contract_without_solver(self) -> None:
        result = rid.synthetic_smoke_test(T=24, nx=2, B=4, block_len=8)
        self.assertIn(result["status"], {"passed", "skipped_no_cvxpy"})
        self.assertTrue(result["checks"]["required_keys_present"])

    def test_estimate_homeostatic_center_median_does_not_require_cvxpy(self) -> None:
        # The median branch is pure numpy; simulating cvxpy-absent must not
        # regress to ImportError. Save/restore rid.cp around the call rather
        # than deleting it, so other tests stay unaffected.
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(3, 50))
        saved_cp = rid.cp
        try:
            rid.cp = None
            center = rid.estimate_homeostatic_center(Y, method="median")
        finally:
            rid.cp = saved_cp
        self.assertEqual(center.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(center)))
        # Sanity: equal to numpy's coordinatewise median (keepdims).
        self.assertTrue(np.allclose(center, np.median(Y, axis=1, keepdims=True)))

    def test_estimate_homeostatic_center_huber_still_requires_cvxpy(self) -> None:
        rng = np.random.default_rng(1)
        Y = rng.normal(size=(2, 20))
        saved_cp = rid.cp
        try:
            rid.cp = None
            with self.assertRaises(ImportError):
                rid.estimate_homeostatic_center(Y, method="huber")
        finally:
            rid.cp = saved_cp

    @unittest.skipIf(rid.cp is None, "cvxpy not installed")
    def test_bisection_returns_a_lb_zero_when_marginally_stable(self) -> None:
        # A = I: contraction LMI A^T P A - (1-a)P <= 0 reduces to a*P <= 0,
        # only feasible at a=0. Bisection only tests strict midpoints, so
        # without the a_lo pre-check best_P would stay None and the function
        # would raise even though a_lb=0 is the correct answer.
        result = rid.bisection_a_lower_bound(
            [np.eye(2)], a_lo=0.0, a_hi=0.5, tol=1e-3, max_iter=10
        )
        self.assertAlmostEqual(result["a_lb"], 0.0, places=3)
        self.assertEqual(result["P"].shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(result["P"])))
        # P must be PSD with trace ≈ 1 per the feasibility constraints.
        eigvals = np.linalg.eigvalsh(result["P"])
        self.assertTrue(np.all(eigvals > -1e-6))
        self.assertAlmostEqual(float(np.trace(result["P"])), 1.0, places=3)

    @unittest.skipIf(rid.cp is None, "cvxpy not installed")
    def test_bisection_raises_when_a_lo_infeasible(self) -> None:
        # A = 2*I: spectral radius 2, no a in [0, 0.999] makes the LMI
        # feasible, so the late-RuntimeError compatibility path must still
        # fire — the new pre-check should not mask genuine infeasibility.
        with self.assertRaises(RuntimeError):
            rid.bisection_a_lower_bound(
                [2.0 * np.eye(2)], a_lo=0.0, a_hi=0.5, tol=1e-3, max_iter=8
            )

    @unittest.skipIf(rid.cp is None, "cvxpy not installed")
    def test_end_to_end_smoke(self) -> None:
        result = rid.run_demo(T=40, nx=2, seed=5, B=6, block_len=8, eta=0.25, solver="CLARABEL")
        bounds = result["bounds"]
        gains = result["gains"]
        self.assertTrue(np.all(np.isfinite(bounds.P)))
        self.assertGreaterEqual(bounds.a_lb, 0.0)
        self.assertGreater(bounds.lambda_lb, 0.0)
        self.assertGreater(bounds.mu_lb, 0.0)
        self.assertGreaterEqual(bounds.kappa_ub, 0.0)
        self.assertGreater(gains.p, 0.0)
        self.assertGreater(gains.q, 0.0)
        self.assertTrue(math.isfinite(gains.delta_safe))
        self.assertTrue(math.isfinite(gains.Gamma))


if __name__ == "__main__":
    unittest.main()
