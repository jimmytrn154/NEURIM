"""OpenCV stimulus presentation for real EEG calibration sessions.

The pairwise recorder needs a subject-facing interface: show the target to
memorize, then image A, morph A -> B, hold B - while EEG is recorded during the
stable A and B windows. Critically, the window must keep repainting *during* EEG
capture (otherwise it freezes / never renders on macOS), so `paint()` is called
inside the capture loop.

Mock/offline recording does not use this at all (no human in the loop); it is
only constructed when --present is passed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class AbortPresentation(Exception):
    """Raised when the operator presses q/ESC to stop the session early."""


class StimulusPresenter:
    def __init__(self, window_name: str = "NEURIM calibration", size: int = 768,
                 fullscreen: bool = False):
        import cv2

        self.cv2 = cv2
        self.window_name = window_name
        self.size = size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window_name, size, size)

    # -- image helpers -----------------------------------------------------
    def load(self, path: Path) -> np.ndarray:
        img = self.cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read stimulus image: {path}")
        return self.cv2.resize(img, (self.size, self.size))

    def _blank(self) -> np.ndarray:
        return np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def _with_caption(self, image: np.ndarray, caption: str | None) -> np.ndarray:
        if not caption:
            return image
        out = image.copy()
        band = out[: 52].copy()
        overlay = band.copy()
        overlay[:] = (0, 0, 0)
        self.cv2.addWeighted(overlay, 0.55, band, 0.45, 0.0, band)
        out[: 52] = band
        self.cv2.putText(out, caption, (16, 34), self.cv2.FONT_HERSHEY_SIMPLEX,
                         0.7, (255, 255, 255), 2, self.cv2.LINE_AA)
        return out

    # -- key handling ------------------------------------------------------
    def _pump(self, frame: np.ndarray, wait_ms: int = 1) -> None:
        self.cv2.imshow(self.window_name, frame)
        key = self.cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q")):
            raise AbortPresentation

    def paint(self, image: np.ndarray, caption: str | None = None) -> None:
        """One repaint - call this inside the EEG capture loop so the stable
        image stays on screen (and the quit key is honored) during recording."""
        self._pump(self._with_caption(image, caption))

    # -- timed phases ------------------------------------------------------
    def _hold_frame(self, frame: np.ndarray, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._pump(frame, wait_ms=15)

    def message(self, lines: list[str], seconds: float | None = None) -> None:
        """Show centered text. If seconds is None, wait for the space bar."""
        frame = self._blank()
        y = self.size // 2 - (len(lines) - 1) * 22
        for line in lines:
            (w, _), _ = self.cv2.getTextSize(line, self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            self.cv2.putText(frame, line, ((self.size - w) // 2, y),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, self.cv2.LINE_AA)
            y += 44
        if seconds is not None:
            self._hold_frame(frame, seconds)
            return
        while True:
            self.cv2.imshow(self.window_name, frame)
            key = self.cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                raise AbortPresentation
            if key == ord(" "):
                return

    def fixation(self, seconds: float) -> None:
        frame = self._blank()
        c = self.size // 2
        self.cv2.line(frame, (c - 18, c), (c + 18, c), (200, 200, 200), 2)
        self.cv2.line(frame, (c, c - 18), (c, c + 18), (200, 200, 200), 2)
        self._hold_frame(frame, seconds)

    def show_target(self, image: np.ndarray, seconds: float) -> None:
        self._hold_frame(self._with_caption(image, "Memorize this target"), seconds)

    def crossfade(self, a: np.ndarray, b: np.ndarray, seconds: float, fps: int = 30) -> None:
        """Morph A -> B by alpha blend (the calibration analogue of the live latent
        morph). Not an EEG-scored phase."""
        n = max(2, int(seconds * fps))
        for i in range(n):
            alpha = i / (n - 1)
            frame = self.cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)
            self._pump(self._with_caption(frame, "..."), wait_ms=max(1, int(1000 / fps)))

    def close(self) -> None:
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
