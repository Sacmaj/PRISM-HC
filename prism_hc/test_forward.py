"""End-to-end forward + plasticity smoke tests."""

from __future__ import annotations

import unittest

import torch

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


class ForwardSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.cfg = PrismConfig()
        self.model = PrismHCLite(self.cfg)
        self.state = self.model.init_state(batch=1)
        self.belief = self.model.init_belief(batch=1)

    def _step(self, t: int) -> None:
        x = 0.3 * torch.sin(torch.tensor([t / 3.0])).expand(1, self.cfg.d_in)
        x = x + 0.05 * torch.randn(1, self.cfg.d_in)
        _y, self.state, self.belief, _rec = self.model.forward(
            x, self.state, self.belief
        )

    def test_forward_no_nan(self) -> None:
        for t in range(20):
            self._step(t)
        # Probe state for finiteness
        for l, R in self.state.R_l.items():
            self.assertTrue(torch.all(torch.isfinite(R)), f"R_l[{l}] not finite")
        for name, t in (
            ("E", self.state.E), ("S", self.state.S),
            ("rho", self.state.rho), ("h", self.state.h),
        ):
            self.assertTrue(torch.all(torch.isfinite(t)), f"{name} not finite")
        for l, mu in self.belief.mu_l.items():
            self.assertTrue(torch.all(torch.isfinite(mu)), f"mu_l[{l}] not finite")
        self.assertTrue(torch.isfinite(self.belief.free_energy))

    def test_bounds_maintained(self) -> None:
        for t in range(40):
            self._step(t)
        S = float(self.state.S.item())
        E = float(self.state.E.item())
        rho = float(self.state.rho.item())
        self.assertGreaterEqual(S, 0.0)
        self.assertLessEqual(S, 1.0)
        self.assertGreaterEqual(E, 0.0)
        self.assertLessEqual(E, 1.0)
        self.assertGreaterEqual(rho, 0.0)
        self.assertLessEqual(rho, self.cfg.rho_max + 1e-6)
        cbf = (
            S
            - self.cfg.cbf_a * (E ** self.cfg.cbf_p)
            - self.cfg.cbf_delta
        )
        self.assertGreaterEqual(cbf, -1e-6)

    def test_plasticity_step_blocked_when_unsafe(self) -> None:
        # Brand-new state has S=0, dwell=0 -> commit must be refused.
        tele = TelemetryRecorder()
        grads = {
            n: torch.randn_like(p) * 0.01
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.model.plasticity_step(self.state, grads, tele)
        self.assertEqual(len(tele.commits), 1)
        self.assertFalse(tele.commits[0].committed)

    def test_plasticity_step_succeeds_when_open(self) -> None:
        # Force-open the gate by hand and confirm a commit lands.
        self.state.S = torch.tensor([0.95])
        self.state.E = torch.tensor([0.10])
        self.state.rho = torch.tensor([0.10])
        self.state.dwell_counter = torch.tensor([10], dtype=torch.long)
        self.state.P = torch.tensor([1.0])
        # Snapshot a parameter
        params_before = {
            n: p.detach().clone() for n, p in self.model.named_parameters()
        }
        tele = TelemetryRecorder()
        grads = {
            n: torch.ones_like(p)
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.model.plasticity_step(self.state, grads, tele)
        self.assertEqual(len(tele.commits), 1)
        self.assertTrue(tele.commits[0].committed)
        # At least one weight must have changed
        any_changed = False
        for n, p in self.model.named_parameters():
            if not torch.equal(p, params_before[n]):
                any_changed = True
                break
        self.assertTrue(any_changed)
        # Priming was drained
        self.assertLess(float(self.state.P.item()), 1.0)
        self.assertEqual(self.state.commits, 1)

    def test_telemetry_cbf_uses_robust_delta_eff(self) -> None:
        """Telemetry's cbf field must be computed with delta_eff (the same
        cbf_delta + cbf_robust_gamma the gate uses), or telemetry can report
        a positive margin while can_commit() rejects with cbf_violated and
        consumers of r.cbf miss real barrier violations.

        Pick state values where the baseline delta margin is positive but
        the robust delta_eff margin is negative, so the test fails loudly
        if telemetry ever drifts back to using only cbf_delta.
        """
        cfg = PrismConfig(cbf_robust_gamma=0.20)
        model = PrismHCLite(cfg)
        state = model.init_state(batch=1)
        belief = model.init_belief(batch=1)
        # Force a state where: baseline cbf = 0.30 - 0.5*0.36 - 0.05 = +0.07
        # robust cbf      = 0.30 - 0.5*0.36 - (0.05 + 0.20) = -0.13
        state.S = torch.tensor([0.30])
        state.E = torch.tensor([0.60])
        x = torch.zeros(1, cfg.d_in)
        _y, state, _belief, rec = model.forward(x, state, belief)
        # State.E may have been clamped during forward; recompute the
        # expected reading using the post-forward E/S the recorder saw.
        S_post = float(state.S.item())
        E_post = float(state.E.item())
        expected_robust = (
            S_post - cfg.cbf_a * (E_post ** cfg.cbf_p)
            - (cfg.cbf_delta + cfg.cbf_robust_gamma)
        )
        expected_baseline_only = (
            S_post - cfg.cbf_a * (E_post ** cfg.cbf_p) - cfg.cbf_delta
        )
        self.assertAlmostEqual(rec.cbf, expected_robust, places=5)
        # Defensive: the two formulas must actually differ here, otherwise
        # the test isn't exercising what it claims.
        self.assertNotAlmostEqual(expected_robust, expected_baseline_only, places=2)

    def test_reset_episode_zeros_fast_state_only(self) -> None:
        """reset_episode must zero fast state but leave U and W_rand bit-identical.

        The slow/fast boundary is the load-bearing safety claim of the design;
        previously asserted only by docs/comments.
        """
        U_snapshot = self.model.anchors.U.clone()
        W_snapshot = self.model.reservoir.W_rand.clone()
        # Drive fast state to non-zero values via a few forward steps.
        for t in range(5):
            self._step(t)
        # Force-write every fast field to a non-zero value so reset has work to do.
        self.state.E.fill_(0.5)
        self.state.S.fill_(0.8)
        self.state.rho.fill_(0.3)
        self.state.chi.fill_(0.4)
        self.state.h.fill_(0.2)
        self.state.dwell_counter.fill_(7)
        self.state.P.fill_(0.5)
        for l in self.state.R_l:
            self.state.R_l[l].fill_(0.6)
        # Reset.
        self.model.reset_episode(self.state)
        # Fast state zeroed (P resets to 1.0 by convention).
        for name, t in (
            ("E", self.state.E), ("S", self.state.S),
            ("rho", self.state.rho), ("chi", self.state.chi),
            ("h", self.state.h),
        ):
            self.assertTrue(torch.equal(t, torch.zeros_like(t)),
                            f"{name} not zeroed by reset_episode")
        self.assertTrue(torch.equal(self.state.P, torch.ones_like(self.state.P)),
                        "P did not reset to 1.0")
        self.assertTrue(torch.equal(
            self.state.dwell_counter, torch.zeros_like(self.state.dwell_counter)
        ), "dwell_counter not zeroed")
        for l in self.state.R_l:
            self.assertTrue(torch.equal(
                self.state.R_l[l], torch.zeros_like(self.state.R_l[l])
            ), f"R_l[{l}] not zeroed")
        # Slow state bit-identical (the load-bearing claim).
        self.assertTrue(torch.equal(self.model.anchors.U, U_snapshot),
                        "anchors.U mutated by reset_episode")
        self.assertTrue(torch.equal(self.model.reservoir.W_rand, W_snapshot),
                        "reservoir.W_rand mutated by reset_episode")


if __name__ == "__main__":
    unittest.main()
