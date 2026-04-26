"""Compatibility wrapper for the built-in demo in `.identification`.

Example
-------
    python -m rebus_synthesis.demo --T 64 --nx 3 --B 20 --block-len 16
"""

from .identification import main, run_demo

__all__ = ["run_demo", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
