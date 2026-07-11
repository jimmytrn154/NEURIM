#!/usr/bin/env python3
"""Compatibility entrypoint for FAA-driven optimizer sessions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.session.real_runner import RealEEGOptimizerRunner, main

__all__ = ["RealEEGOptimizerRunner", "main"]


if __name__ == "__main__":
    main()
