"""Compatibility wrapper for the built-in synthetic scaffold in `rebus_identification`.

This module re-exports the synthetic-data generator so older imports continue to
work after folding the helper directly into the main module.
"""

from rebus_identification import SyntheticTruth, make_synthetic_rebus_data

__all__ = ["SyntheticTruth", "make_synthetic_rebus_data"]
