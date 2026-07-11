"""Live-frame and session snapshot storage."""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class FrameStore:
    def __init__(self, directory: Path = DEFAULT_PROCESSED_DIR) -> None:
        self.directory = directory
        self._start_saved = False

    def save_live(self, png_bytes: bytes, capture_start: bool = False) -> Path:
        live_path = self.save(png_bytes, "live_frame.png")
        if capture_start and not self._start_saved:
            self.save(png_bytes, "session_start.png")
            self._start_saved = True
        return live_path

    def save_end(self, png_bytes: bytes) -> Path:
        return self.save(png_bytes, "session_end.png")

    def save(self, png_bytes: bytes, name: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / name
        destination.write_bytes(png_bytes)
        return destination
