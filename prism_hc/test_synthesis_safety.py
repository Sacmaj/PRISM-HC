"""Empirical safety tests for the rebus_synthesis -> prism_hc runtime contract.

These tests run the full REBUS synthesis pipeline (cvxpy required) and assert
that the synthesized gains, when wired into PrismConfig via from_rebus_synthesis,
keep the runtime in its joint-CBF safe set under benign disturbance and that
the synthesized Gamma is observable as a runtime decision flip.

The class is gated on cvxpy availability and skips cleanly when cvxpy is
missing (matches the slim CI `test` job; the real path runs in `pipeline`).
"""

from __future__ import annotations

import dataclasses
import math
import unittest

import torch

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder
from rebus_synthesis import cp, run_demo


@unittest.skipIf(cp is None, "cvxpy not installed; skipping synth->runtime safety tests")
class SynthesisSafetyTests(unittest.TestCase):
    """End-to-end: real LMI synthesis -> PrismConfig -> runtime forward steps."""

    @classmethod
    def setUpClass(cls) -> None:
        # Mirror synthesis_demo.py:71 exactly. Clarabel is deterministic given
        # seed, so all three tests see the same gains across runs.
        result = run_demo(T=40, nx=2, seed=5, B=6, block_len=8, eta=0.25, solver="CLARABEL")
        cls.gains = result["gains"]
        cls.cfg = PrismConfig.from_rebus_synthesis(cls.gains)

    def setUp(self) -> None:
        torch.manual_seed(self.cfg.seed)
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        self.model = PrismHCLite(self.cfg)
        self.state = self.model.init_state(batch=1)
        self.belief = self.model.init_belief(batch=1)

    # ------------------------------------------------------------------
    # Test 1: wire-up surface (no forward pass)
    # ------------------------------------------------------------------
    def test_gamma_propagation_into_runtime_config(self) -> None:
        """SupervisorGains -> PrismConfig: Gamma routed and heuristic applied."""
        # Gamma propagation (mirrors PR #6's wire-up).
        self.assertEqual(self.cfg.cbf_robust_gamma, self.gains.Gamma)
        # Synthesizer must produce a non-trivial bound; a zero Gamma would
        # silently disable the joint-CBF tightening this PR is testing.
        self.assertGreater(self.cfg.cbf_robust_gamma, 0.0)
        # Heuristic composition (mirrors PR #8's default_gamma_map regression
        # test, but anchored to real synthesis output instead of _FakeGains).
        self.assertAlmostEqual(self.cfg.gamma_s, 0.5 / self.gains.p, places=12)
        self.assertAlmostEqual(self.cfg.gamma_d, 0.5 / self.gains.q, places=12)
        self.assertAlmostEqual(self.cfg.gamma_eps, 0.5 * self.gains.delta_safe, places=12)

    # ------------------------------------------------------------------
    # Test 2: empirical safe-set entry + invariance under benign input
    # ------------------------------------------------------------------
    def test_safe_set_entry_and_invariance_under_benign_input(self) -> None:
        """Gamma-shrunk safe set is reachable and invariant under benign drive.

        init_state sets S=0, so the runtime starts OUTSIDE the safe set
        (margin = -delta_eff < 0). Under benign input, S converges toward 1
        with time constant 1/lam_S, and the safe set is entered when
        S >= delta_eff = cbf_delta + Gamma. After entry, LATCH's E-clamp
        keeps the post-clamp margin >= 0 every step.

        Earlier this test held a conditional skip for `delta_eff >= 1`,
        because SCS at the canonical scenario emitted `Gamma ~= 125448` and
        rendered the safe set empty. The Clarabel switch in this PR converges
        to a finite Gamma in the runtime [0,1] range, so the skip is gone and
        the entry+invariance assertions run unconditionally.
        """
        cfg = self.cfg
        delta_eff = cfg.cbf_delta + cfg.cbf_robust_gamma

        tele = TelemetryRecorder()
        N = 30
        for _t in range(N):
            x = 0.05 * torch.randn(1, cfg.d_in, generator=self.gen)
            _y, self.state, self.belief, rec = self.model.forward(
                x, self.state, self.belief
            )
            tele.append_step(rec)

        self.assertEqual(len(tele.steps), N)

        # Finiteness across the whole trajectory.
        for t, rec in enumerate(tele.steps):
            for name, v in (
                ("R", rec.R), ("E", rec.E), ("S", rec.S),
                ("rho", rec.rho), ("h", rec.h), ("F", rec.F),
                ("cbf", rec.cbf), ("drift", rec.drift),
            ):
                self.assertTrue(
                    math.isfinite(v),
                    f"{name} not finite at step {t}: {v!r}",
                )

        # Find first step where the (Gamma-shrunk) safe set is entered.
        entry_step = next(
            (t for t, rec in enumerate(tele.steps) if rec.cbf >= -1e-6),
            None,
        )
        self.assertIsNotNone(
            entry_step,
            f"safe set never entered in {N} steps; "
            f"final cbf={tele.steps[-1].cbf:.4f}, "
            f"delta_eff={delta_eff:.4f}",
        )
        # Under lam_S=0.15 and benign input, S(t) ~= 1 - exp(-0.15*t). Even at
        # delta_eff up to 0.55 (Gamma ~= 0.5), entry happens by t=6. Allowing
        # 20 leaves substantial slack while still catching gross drift.
        self.assertLessEqual(
            entry_step,
            20,
            f"safe-set entry too slow: t={entry_step} (delta_eff={delta_eff:.4f})",
        )

        # Invariance: once entered, the post-clamp margin stays nonnegative.
        for t in range(entry_step, N):
            self.assertGreaterEqual(
                tele.steps[t].cbf,
                -1e-6,
                f"safe set exited at step {t} after entry at step {entry_step}: "
                f"cbf={tele.steps[t].cbf:.6f}",
            )

    # ------------------------------------------------------------------
    # Test 3: behavioral observability of synthesized Gamma
    # ------------------------------------------------------------------
    def test_synthesized_gamma_shifts_post_step_margin_by_exactly_gamma(self) -> None:
        """Synthesized Gamma shifts the post-forward CBF margin by exactly -Gamma.

        Mirrors test_forward.test_telemetry_cbf_uses_robust_delta_eff but
        anchored to a *solver-produced* Gamma rather than a hand-set 0.20.

        At fresh init (S=0) with x=zeros, the joint-CBF E-clamp leaves E_post
        identical across the synth and Gamma=0 configs: the LATCH gate is
        closed (dwell=0), so gated_target=0, so E_post = exp_euler(0, 0, ...)
        = 0 in both cases (the clamp can only reduce E). LATCH's S update
        doesn't depend on cbf_robust_gamma, so S_post is also identical.
        The post-step margin therefore satisfies
            rec_ng.cbf - rec.cbf == gains.Gamma
        exactly (modulo float precision). This holds for ANY gains.Gamma > 0
        — small or large — so the assertion needs no sign-flip threshold.
        Catches regressions where Gamma silently drops out of the wire-up
        (rec_ng.cbf - rec.cbf would equal 0) regardless of Gamma magnitude.
        """
        x = torch.zeros(1, self.cfg.d_in)

        # Synthesized config (cbf_robust_gamma = gains.Gamma).
        _y, _state, _belief, rec = self.model.forward(x, self.state, self.belief)

        # Negative-control twin: same cfg with cbf_robust_gamma zeroed.
        # Reseeding torch before constructing the twin model gives bit-
        # identical linear-layer weights, so the only behavioral difference
        # between the two forward passes is the Gamma robustification.
        cfg_no_gamma = dataclasses.replace(self.cfg, cbf_robust_gamma=0.0)
        torch.manual_seed(cfg_no_gamma.seed)
        model_ng = PrismHCLite(cfg_no_gamma)
        state_ng = model_ng.init_state(batch=1)
        belief_ng = model_ng.init_belief(batch=1)
        _y_ng, _state_ng, _belief_ng, rec_ng = model_ng.forward(
            x, state_ng, belief_ng
        )

        margin_shift = rec_ng.cbf - rec.cbf
        # Tolerance scales with Gamma so the same assertion works whether
        # Gamma is tiny (e.g., 0.0027) or large (e.g., 1e5). The CBF margin
        # is computed in float32 (one torch ULP ~= 1.2e-7 relative), so the
        # relative term must accommodate at least a few float32 ULPs.
        # Absolute 1e-5 covers small-Gamma cases where the relative term
        # underflows below float32 precision.
        tol = max(1e-5, 1e-6 * abs(self.gains.Gamma))
        self.assertAlmostEqual(
            margin_shift,
            self.gains.Gamma,
            delta=tol,
            msg=(
                f"Gamma not observable at runtime: rec_ng.cbf - rec.cbf "
                f"= {margin_shift:.6g}, expected gains.Gamma "
                f"= {self.gains.Gamma:.6g} (delta_eff difference). "
                f"This signals a regression in cbf_robust_gamma propagation "
                f"or in the LATCH joint-CBF margin computation."
            ),
        )


if __name__ == "__main__":
    unittest.main()
