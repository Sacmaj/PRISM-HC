"""Compatibility wrapper for the built-in demo in `rebus_identification`.

Example
-------
    python rebus_identification_demo.py --T 64 --nx 3 --B 20 --block-len 16
"""

from rebus_identification import main, run_demo

__all__ = ["run_demo", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
