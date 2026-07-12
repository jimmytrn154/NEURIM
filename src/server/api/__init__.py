"""Local frontend API bridge."""

from .app import app, create_app
from .eeg import EEGConnectionManager
from .manager import SessionManager
from .models import StartSessionRequest

__all__ = ["EEGConnectionManager", "SessionManager", "StartSessionRequest", "app", "create_app"]
