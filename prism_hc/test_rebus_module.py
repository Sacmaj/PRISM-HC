"""Unit tests for REBUS scalar-state updates and precision modulation."""

from __future__ import annotations

import math
import unittest

import torch

from prism_hc import rebus


class PrecisionModulationTests(unittest.TestCase):
    def test_precision_finite_and_positive(self) -> None:
        pi_bar = torch.ones(8)
        for R_val in (-5.0, -1.0, 0.0, 1.0, 5.0):
            for u_d_val in (-5.0, -1.0, 0.0, 1.0, 5.0):
                R = torch.tensor([[R_val]])
                u_d = torch.tensor([[u_d_val]])
                pi = rebus.precision_modulation(
                    pi_bar, R, u_d, delta_l=0.3, kappa_l=0.5,
                    log_clamp=(-6.0, 6.0),
                )
                self.assertTrue(torch.all(torch.isfinite(pi)))
                self.assertTrue(torch.all(pi > 0.0))

    def test_log_clamp_respected(self) -> None:
        pi_bar = torch.ones(4)
        # Push extreme inputs that would otherwise overflow.
        R = torch.tensor([[-100.0]])
        u_d = torch.tensor([[100.0]])
        pi = rebus.precision_modulation(
            pi_bar, R, u_d, delta_l=1.0, kappa_l=1.0, log_clamp=(-6.0, 6.0)
        )
        self.assertLessEqual(float(pi.max().item()), math.exp(6.0) + 1e-3)
        # Symmetric extreme
        R = torch.tensor([[100.0]])
        u_d = torch.tensor([[-100.0]])
        pi = rebus.precision_modulation(
            pi_bar, R, u_d, delta_l=1.0, kappa_l=1.0, log_clamp=(-6.0, 6.0)
        )
        self.assertGreaterEqual(float(pi.min().item()), math.exp(-6.0) - 1e-6)

    def test_precision_increases_with_u_d(self) -> None:
        pi_bar = torch.ones(4)
        R = torch.zeros(1, 1)
        pi_lo = rebus.precision_modulation(pi_bar, R, torch.tensor([[0.0]]), 0.3, 0.5)
        pi_hi = rebus.precision_modulation(pi_bar, R, torch.tensor([[1.0]]), 0.3, 0.5)
        self.assertTrue(torch.all(pi_hi >= pi_lo))


class UpdateRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = dict(
            lam_R=0.08, R0=0.0, beta_h=0.1,
            gamma_s=0.2, gamma_d=0.15, gamma_eps=0.1, dt=1.0,
        )

    def test_R_increases_with_safety_signal(self) -> None:
        R = torch.zeros(1)
        h = torch.zeros(1)
        eps = torch.zeros(1)
        u_d = torch.zeros(1)
        R_low = rebus.update_R(R, h, u_s=torch.tensor([-2.0]), u_d=u_d,
                               eps_norm=eps, **self.kwargs)
        R_high = rebus.update_R(R, h, u_s=torch.tensor([2.0]), u_d=u_d,
                                eps_norm=eps, **self.kwargs)
        self.assertGreater(float(R_high.item()), float(R_low.item()))

    def test_R_stays_in_bounds(self) -> None:
        R = torch.zeros(1)
        h = torch.zeros(1)
        u_s = torch.tensor([10.0])
        u_d = torch.tensor([10.0])
        eps = torch.tensor([10.0])
        for _ in range(200):
            R = rebus.update_R(R, h, u_s, u_d, eps_norm=eps, **self.kwargs)
            self.assertGreaterEqual(float(R.item()), 0.0)
            self.assertLessEqual(float(R.item()), 1.0)

    def test_h_grows_when_R_above_R0(self) -> None:
        h0 = torch.zeros(1)
        R = torch.tensor([0.5])
        h_next = rebus.update_h(h0, R, R0=0.0, lam_h=0.03, eta_h=0.05, dt=1.0)
        self.assertGreater(float(h_next.item()), 0.0)

    def test_h_decays_when_R_at_R0(self) -> None:
        h0 = torch.tensor([0.5])
        R = torch.tensor([0.0])
        h_next = rebus.update_h(h0, R, R0=0.0, lam_h=0.03, eta_h=0.05, dt=1.0)
        self.assertLess(float(h_next.item()), float(h0.item()))


if __name__ == "__main__":
    unittest.main()
