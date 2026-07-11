#!/usr/bin/env python3
"""Compatibility entrypoint for scripted optimizer sessions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.session.mock_runner import MockOptimizerRunner, main

__all__ = ["MockOptimizerRunner", "main"]


if __name__ == "__main__":
    main()
