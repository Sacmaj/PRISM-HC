"""Tests for PrismConfig normalization and synthesis composition."""

from __future__ import annotations

import unittest

from prism_hc.config import PrismConfig, default_gamma_map


class _FakeGains:
    p = 1.0
    q = 1.0
    delta_safe = 0.5
    Gamma = 0.4


class _GainsNoGamma:
    p = 1.0
    q = 1.0
    delta_safe = 0.5


class PrismConfigNormalizationTests(unittest.TestCase):
    def test_default_l2_backward_compatible(self) -> None:
        c = PrismConfig()
        self.assertEqual(c.delta_l, (0.30, 0.30))
        self.assertEqual(c.kappa_l, (0.50, 0.50))

    def test_scalar_default_expands_for_larger_L(self) -> None:
        c = PrismConfig(L=3)
        self.assertEqual(c.delta_l, (0.30, 0.30, 0.30))
        self.assertEqual(c.kappa_l, (0.50, 0.50, 0.50))

    def test_explicit_full_tuple_passes_through(self) -> None:
        c = PrismConfig(L=3, delta_l=(0.1, 0.2, 0.3))
        self.assertEqual(c.delta_l, (0.1, 0.2, 0.3))

    def test_one_tuple_broadcasts(self) -> None:
        c = PrismConfig(L=4, delta_l=(0.7,))
        self.assertEqual(c.delta_l, (0.7, 0.7, 0.7, 0.7))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            PrismConfig(L=3, delta_l=(0.1, 0.2))

    def test_from_rebus_synthesis_supports_L_override(self) -> None:
        # Regression: building a synthesized config with L>2 must not trip the
        # __post_init__ length check on default delta_l/kappa_l.
        cfg = PrismConfig.from_rebus_synthesis(_FakeGains(), L=3)
        self.assertEqual(cfg.L, 3)
        self.assertEqual(cfg.delta_l, (0.30, 0.30, 0.30))
        self.assertEqual(cfg.kappa_l, (0.50, 0.50, 0.50))

    def test_from_rebus_synthesis_respects_explicit_per_layer(self) -> None:
        cfg = PrismConfig.from_rebus_synthesis(
            _FakeGains(), L=3, delta_l=(0.1, 0.2, 0.3)
        )
        self.assertEqual(cfg.delta_l, (0.1, 0.2, 0.3))

    def test_from_rebus_synthesis_propagates_gamma(self) -> None:
        """SupervisorGains.Gamma must flow into PrismConfig.cbf_robust_gamma so
        LATCH actually sees the disturbance-gain bound."""
        cfg = PrismConfig.from_rebus_synthesis(_FakeGains())
        self.assertEqual(cfg.cbf_robust_gamma, 0.4)

    def test_from_rebus_synthesis_gamma_absent_defaults_zero(self) -> None:
        """Duck-typed gains without a Gamma attribute must yield the field
        default (0.0) — no AttributeError, no surprise."""
        cfg = PrismConfig.from_rebus_synthesis(_GainsNoGamma())
        self.assertEqual(cfg.cbf_robust_gamma, 0.0)

    def test_from_rebus_synthesis_gamma_override_wins(self) -> None:
        """Explicit cbf_robust_gamma in **overrides must override the
        gains-derived value (operator escape hatch)."""
        cfg = PrismConfig.from_rebus_synthesis(
            _FakeGains(), cbf_robust_gamma=0.05
        )
        self.assertEqual(cfg.cbf_robust_gamma, 0.05)

    def test_negative_cbf_robust_gamma_rejected_direct(self) -> None:
        """Direct construction with negative robust gamma must raise. Negative
        values would shrink delta_eff below baseline, expanding the CBF safe
        set and inverting the intended robustness tightening."""
        with self.assertRaises(ValueError) as ctx:
            PrismConfig(cbf_robust_gamma=-0.10)
        self.assertIn("cbf_robust_gamma", str(ctx.exception))

    def test_negative_gamma_rejected_via_synthesis(self) -> None:
        """A SupervisorGains-shaped object whose Gamma went negative
        (upstream computes Gamma = b3_ub + p*eta; negative eta can flip the
        sign) must surface as a ValueError, not silently expand the safe
        set."""
        class _NegativeGammaGains:
            p = 1.0
            q = 1.0
            delta_safe = 0.5
            Gamma = -0.20
        with self.assertRaises(ValueError):
            PrismConfig.from_rebus_synthesis(_NegativeGammaGains())

    def test_zero_cbf_robust_gamma_accepted(self) -> None:
        """The default 0.0 must remain valid (non-strict inequality)."""
        cfg = PrismConfig(cbf_robust_gamma=0.0)
        self.assertEqual(cfg.cbf_robust_gamma, 0.0)


class DefaultGammaMapTests(unittest.TestCase):
    def test_formula_pinned(self) -> None:
        class G:
            p, q, delta_safe = 2.0, 4.0, 0.6
        self.assertEqual(default_gamma_map(G()), (0.25, 0.125, 0.30))

    def test_default_used_by_from_rebus_synthesis(self) -> None:
        gains = _FakeGains()
        cfg = PrismConfig.from_rebus_synthesis(gains)
        gs, gd, ge = default_gamma_map(gains)
        self.assertEqual(cfg.gamma_s, gs)
        self.assertEqual(cfg.gamma_d, gd)
        self.assertEqual(cfg.gamma_eps, ge)

    def test_explicit_default_passed_as_gamma_map(self) -> None:
        gains = _FakeGains()
        a = PrismConfig.from_rebus_synthesis(gains)
        b = PrismConfig.from_rebus_synthesis(gains, gamma_map=default_gamma_map)
        self.assertEqual(
            (a.gamma_s, a.gamma_d, a.gamma_eps),
            (b.gamma_s, b.gamma_d, b.gamma_eps),
        )


if __name__ == "__main__":
    unittest.main()
