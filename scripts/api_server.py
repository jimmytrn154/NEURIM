#!/usr/bin/env python3
"""Compatibility entrypoint for the local NEURIM API server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.server.api import EEGConnectionManager, SessionManager, StartSessionRequest, app, create_app
from src.server.api.cli import main

__all__ = ["EEGConnectionManager", "SessionManager", "StartSessionRequest", "app", "create_app", "main"]


if __name__ == "__main__":
    main()
