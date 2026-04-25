"""Unit tests for LATCH lexicographic gate and joint CBF clamp."""

from __future__ import annotations

import unittest

import torch

from prism_hc.latch import LATCHPlasticityController
from prism_hc.state import ControllerState, SafetyState


def _fresh_state(batch: int = 1) -> ControllerState:
    z = lambda: torch.zeros(batch)
    return ControllerState(
        R_l={0: z(), 1: z()},
        alpha=z(), h=z(), E=z(), S=z(), rho=z(), chi=z(),
        P=torch.ones(batch),
        dwell_counter=torch.zeros(batch, dtype=torch.long),
        commits=0,
    )


def _safety(drift: float) -> SafetyState:
    return SafetyState(
        anchor_coords={},
        drift=torch.tensor(drift),
        canary_margin=float(max(0.0, 1.0 - drift)),
    )


class LATCHGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = LATCHPlasticityController(
            lam_E=0.10, lam_S=0.15, lam_rho=0.05,
            S_min=0.7, dwell_min=3, rho_max=0.9,
            cbf_a=0.5, cbf_p=2.0, cbf_delta=0.05,
        )

    def test_gate_blocks_E_when_S_low(self) -> None:
        """High surprise + low safety must keep E pinned near zero."""
        state = _fresh_state()
        # drift=0.8 -> instant_safety=0.2, well below S_min=0.7
        safety = _safety(drift=0.8)
        surprise = torch.tensor([0.6])
        for _ in range(20):
            state = self.ctrl.step(state, safety, surprise, dt=1.0)
        self.assertLess(float(state.E.item()), 0.05)
        self.assertEqual(int(state.dwell_counter.item()), 0)

    def test_gate_opens_after_dwell(self) -> None:
        """With safety high, dwell counter accumulates and E rises."""
        state = _fresh_state()
        safety = _safety(drift=0.0)  # instant_safety = 1.0
        surprise = torch.tensor([0.5])
        # Run long enough for S to climb past S_min and dwell to satisfy.
        for _ in range(30):
            state = self.ctrl.step(state, safety, surprise, dt=1.0)
        self.assertGreaterEqual(int(state.dwell_counter.item()), self.ctrl.dwell_min)
        self.assertGreater(float(state.E.item()), 0.05)

    def test_dwell_resets_when_S_drops(self) -> None:
        """Once S falls below S_min, dwell counter must reset to 0."""
        state = _fresh_state()
        safety_safe = _safety(drift=0.0)
        for _ in range(20):
            state = self.ctrl.step(state, safety_safe, torch.tensor([0.3]), dt=1.0)
        self.assertGreater(int(state.dwell_counter.item()), 0)
        # Now spike drift
        safety_unsafe = _safety(drift=0.95)
        for _ in range(15):
            state = self.ctrl.step(state, safety_unsafe, torch.tensor([0.0]), dt=1.0)
        self.assertEqual(int(state.dwell_counter.item()), 0)

    def test_cbf_clamp_binds(self) -> None:
        """Joint CBF clamps E so that h(E,S) = S - a*E^p - delta >= 0."""
        state = _fresh_state()
        safety = _safety(drift=0.5)  # instant_safety ~ 0.5
        # First let S settle near 0.5
        for _ in range(40):
            state = self.ctrl.step(state, safety, torch.tensor([0.0]), dt=1.0)
        S = float(state.S.item())
        # Now hit it with maximal surprise
        for _ in range(20):
            state = self.ctrl.step(state, safety, torch.tensor([1.0]), dt=1.0)
        E = float(state.E.item())
        cbf = (
            float(state.S.item())
            - self.ctrl.cbf_a * (E ** self.ctrl.cbf_p)
            - self.ctrl.cbf_delta
        )
        self.assertGreaterEqual(cbf, -1e-6, f"CBF violated: h={cbf} S={S} E={E}")

    def test_cbf_clamp_general_p(self) -> None:
        """Clamp must respect h(E,S) = S - a*E^p - delta >= 0 for any p, not
        just p=2. The earlier sqrt-based clamp silently violated the barrier
        for p < 2 (e.g. p=1.5). We pick a regime where the clamp is the
        binding constraint: cbf_a is large so max_safe_E < 1, and S_min=0
        lets the gate open immediately.
        """
        for p in (1.5, 3.0, 4.0):
            ctrl = LATCHPlasticityController(
                lam_E=0.10, lam_S=0.15, lam_rho=0.05,
                S_min=0.0, dwell_min=0, rho_max=0.9,
                cbf_a=2.0, cbf_p=p, cbf_delta=0.05,
            )
            state = _fresh_state()
            safety = _safety(drift=0.5)  # instant_safety ~ 0.5, S settles near 0.5
            # Drive maximal surprise; gate is open immediately (S_min=0, dwell_min=0).
            for _ in range(60):
                state = ctrl.step(state, safety, torch.tensor([1.0]), dt=1.0)
            E = float(state.E.item())
            S = float(state.S.item())
            cbf = S - ctrl.cbf_a * (E ** ctrl.cbf_p) - ctrl.cbf_delta
            self.assertGreaterEqual(
                cbf, -1e-6,
                f"CBF violated at p={p}: h={cbf:.4f} S={S:.4f} E={E:.4f}",
            )

    def test_can_commit_requires_all_conditions(self) -> None:
        """Each gating condition independently blocks commits."""
        state = _fresh_state()
        # Force fully-open conditions
        state.S = torch.tensor([0.95])
        state.E = torch.tensor([0.10])
        state.rho = torch.tensor([0.10])
        state.dwell_counter = torch.tensor([10], dtype=torch.long)
        state.P = torch.tensor([1.0])
        self.assertTrue(self.ctrl.can_commit(state, P_min=0.6))

        # Refractory saturated
        state.rho = torch.tensor([0.95])
        self.assertFalse(self.ctrl.can_commit(state, P_min=0.6))
        state.rho = torch.tensor([0.10])

        # Priming below threshold
        state.P = torch.tensor([0.4])
        self.assertFalse(self.ctrl.can_commit(state, P_min=0.6))
        state.P = torch.tensor([1.0])

        # Dwell too short
        state.dwell_counter = torch.tensor([1], dtype=torch.long)
        self.assertFalse(self.ctrl.can_commit(state, P_min=0.6))
        state.dwell_counter = torch.tensor([10], dtype=torch.long)

        # CBF violated by raising E past the clamp
        state.S = torch.tensor([0.10])
        state.E = torch.tensor([0.80])
        self.assertFalse(self.ctrl.can_commit(state, P_min=0.6))


if __name__ == "__main__":
    unittest.main()
