"""Tests for PrismConfig normalization and synthesis composition."""

from __future__ import annotations

import unittest

from prism_hc.config import PrismConfig


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


if __name__ == "__main__":
    unittest.main()
