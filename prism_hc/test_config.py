"""PrismConfig per-layer validation: scalar/broadcast/exact-length accepted,
mismatched lengths rejected loudly so a non-default `L` doesn't silently
crash mid-forward with `IndexError`.
"""

from __future__ import annotations

import unittest

from prism_hc.config import PrismConfig


class PerLayerNormalizationTests(unittest.TestCase):
    def test_default_scalar_broadcasts_to_L(self) -> None:
        cfg = PrismConfig(L=2)
        self.assertEqual(cfg.delta_l, (0.30, 0.30))
        self.assertEqual(cfg.kappa_l, (0.50, 0.50))
        cfg3 = PrismConfig(L=3)
        self.assertEqual(cfg3.delta_l, (0.30, 0.30, 0.30))
        self.assertEqual(cfg3.kappa_l, (0.50, 0.50, 0.50))

    def test_explicit_scalar_broadcasts(self) -> None:
        cfg = PrismConfig(L=4, delta_l=0.1, kappa_l=0.2)
        self.assertEqual(cfg.delta_l, (0.1, 0.1, 0.1, 0.1))
        self.assertEqual(cfg.kappa_l, (0.2, 0.2, 0.2, 0.2))

    def test_length_one_sequence_broadcasts(self) -> None:
        cfg = PrismConfig(L=3, delta_l=(0.7,), kappa_l=[0.9])
        self.assertEqual(cfg.delta_l, (0.7, 0.7, 0.7))
        self.assertEqual(cfg.kappa_l, (0.9, 0.9, 0.9))

    def test_length_L_sequence_kept_as_is(self) -> None:
        cfg = PrismConfig(L=3, delta_l=(0.1, 0.2, 0.3), kappa_l=[0.4, 0.5, 0.6])
        self.assertEqual(cfg.delta_l, (0.1, 0.2, 0.3))
        self.assertEqual(cfg.kappa_l, (0.4, 0.5, 0.6))

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PrismConfig(L=3, delta_l=(0.1, 0.2))
        self.assertIn("delta_l", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            PrismConfig(L=2, kappa_l=(0.1, 0.2, 0.3))
        self.assertIn("kappa_l", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
