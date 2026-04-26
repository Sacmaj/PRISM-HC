"""End-to-end forward + plasticity smoke tests."""

from __future__ import annotations

import contextlib
import io
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


class StateFactoryDeviceTests(unittest.TestCase):
    """init_state / init_belief must produce tensors on the model's device."""

    def test_init_state_uses_module_device(self) -> None:
        cfg = PrismConfig()
        model = PrismHCLite(cfg)
        ref = next(model.parameters())
        state = model.init_state(batch=2)
        for l, R in state.R_l.items():
            self.assertEqual(R.device, ref.device, f"R_l[{l}] off device")
            self.assertEqual(R.dtype, ref.dtype, f"R_l[{l}] wrong dtype")
        for name, tensor in (
            ("E", state.E), ("S", state.S), ("rho", state.rho),
            ("chi", state.chi), ("alpha", state.alpha), ("h", state.h),
            ("P", state.P),
        ):
            self.assertEqual(tensor.device, ref.device, f"{name} off device")
            self.assertEqual(tensor.dtype, ref.dtype, f"{name} wrong dtype")
        # dwell_counter intentionally stays integer; only device must match.
        self.assertEqual(state.dwell_counter.device, ref.device)
        self.assertEqual(state.dwell_counter.dtype, torch.long)

    def test_init_belief_uses_module_device(self) -> None:
        cfg = PrismConfig()
        model = PrismHCLite(cfg)
        ref = next(model.parameters())
        belief = model.init_belief(batch=2)
        for l in range(cfg.L):
            for name, tensor in (
                ("mu_l", belief.mu_l[l]),
                ("epsilon_l", belief.epsilon_l[l]),
                ("pi_l", belief.pi_l[l]),
            ):
                self.assertEqual(tensor.device, ref.device, f"{name}[{l}] off device")
                self.assertEqual(tensor.dtype, ref.dtype, f"{name}[{l}] wrong dtype")


class NonDefaultDepthTests(unittest.TestCase):
    """L != 2 must work end-to-end now that per-layer arrays auto-broadcast."""

    def test_forward_with_L3(self) -> None:
        torch.manual_seed(0)
        cfg = PrismConfig(L=3)  # default delta_l=0.30, kappa_l=0.50 must broadcast
        model = PrismHCLite(cfg)
        state = model.init_state(batch=1)
        belief = model.init_belief(batch=1)
        for t in range(15):
            x = 0.05 * torch.sin(torch.tensor([t / 3.0])).expand(1, cfg.d_in)
            x = x + 0.02 * torch.randn(1, cfg.d_in)
            _y, state, belief, _rec = model.forward(x, state, belief)
        self.assertEqual(len(state.R_l), 3)
        for l in range(3):
            self.assertTrue(torch.all(torch.isfinite(state.R_l[l])))
            self.assertTrue(torch.all(torch.isfinite(belief.mu_l[l])))


class DemoSmokeTests(unittest.TestCase):
    """The demo is the headline smoke run; importing and calling main() locks
    in (a) that no torch API used in the demo is unsupported on torch>=2.0
    (notably the previous `torch.randn_like(generator=)` regression) and
    (b) that the t=30 commit still lands.
    """

    def test_demo_runs_to_completion_and_commits(self) -> None:
        from prism_hc import demo  # local import keeps test isolation

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = demo.main()
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        # The demo prints a final OK and exactly one commit attempt.
        self.assertIn("OK", out)
        self.assertIn("commits_attempted=1", out)
        self.assertIn("committed=True", out)


if __name__ == "__main__":
    unittest.main()
