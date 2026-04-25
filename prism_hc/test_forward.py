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


if __name__ == "__main__":
    unittest.main()
