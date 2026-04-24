"""Unit tests for null-space projection used by the plasticity adapter."""

from __future__ import annotations

import unittest

import torch

from prism_hc.adapter import NullSpaceAdapter


class NullSpaceProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        H, k = 16, 4
        raw = torch.randn(H, k)
        self.U, _ = torch.linalg.qr(raw)
        self.H = H
        self.k = k

    def test_projection_orthogonal_to_U(self) -> None:
        g = torch.randn(8, self.H)
        g_safe = NullSpaceAdapter.project_update(g, self.U)
        residual = g_safe @ self.U
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-5))

    def test_projection_idempotent(self) -> None:
        g = torch.randn(8, self.H)
        g1 = NullSpaceAdapter.project_update(g, self.U)
        g2 = NullSpaceAdapter.project_update(g1, self.U)
        self.assertTrue(torch.allclose(g1, g2, atol=1e-6))

    def test_projection_preserves_null_components(self) -> None:
        # A vector with zero anchor-component must pass through unchanged.
        anchor_part = self.U @ torch.randn(self.k)
        full = torch.randn(self.H)
        null_part = full - self.U @ (self.U.T @ full)
        out = NullSpaceAdapter.project_update(null_part, self.U)
        self.assertTrue(torch.allclose(out, null_part, atol=1e-6))
        # And mixing in an anchor part must be removed.
        mixed = null_part + anchor_part
        out2 = NullSpaceAdapter.project_update(mixed, self.U)
        self.assertTrue(torch.allclose(out2, null_part, atol=1e-5))

    def test_skips_dim_mismatch(self) -> None:
        # Bias-style 1-D tensor with last-dim != H must pass through unchanged.
        g = torch.randn(7)
        out = NullSpaceAdapter.project_update(g, self.U)
        self.assertTrue(torch.equal(out, g))


if __name__ == "__main__":
    unittest.main()
