"""Headless tests for the calibration presentation interface.

A stub `cv2` records calls so the presentation sequence and the EEG-capture
interleave can be verified without a real display.
"""

import sys
import types

import numpy as np
import pytest

ROOT_OK = True


class _FakeCV2:
    WINDOW_NORMAL = 0
    WND_PROP_FULLSCREEN = 1
    WINDOW_FULLSCREEN = 1
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, key_script=None):
        self.imshow_calls = 0
        self.addweighted_calls = 0
        self._keys = list(key_script or [])

    def namedWindow(self, *a, **k): pass
    def resizeWindow(self, *a, **k): pass
    def setWindowProperty(self, *a, **k): pass
    def destroyAllWindows(self, *a, **k): pass
    def imread(self, *a, **k): return np.zeros((640, 640, 3), dtype=np.uint8)
    def resize(self, img, size): return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    def putText(self, *a, **k): pass
    def line(self, *a, **k): pass
    def getTextSize(self, *a, **k): return ((100, 20), 0)

    def addWeighted(self, a, wa, b, wb, g, dst=None):
        self.addweighted_calls += 1
        return (a.astype(float) * wa + b.astype(float) * wb).astype(np.uint8)

    def imshow(self, *a, **k):
        self.imshow_calls += 1

    def waitKey(self, ms):
        return self._keys.pop(0) if self._keys else -1


@pytest.fixture
def fake_cv2(monkeypatch):
    def _install(key_script=None):
        fake = _FakeCV2(key_script)
        monkeypatch.setitem(sys.modules, "cv2", fake)
        return fake
    return _install


def _presenter(fake):
    from src.signal_service.stimulus_presenter import StimulusPresenter
    return StimulusPresenter(size=128)


def test_presentation_phases_paint(fake_cv2):
    fake = fake_cv2()
    p = _presenter(fake)
    a = np.zeros((128, 128, 3), np.uint8)
    b = np.full((128, 128, 3), 255, np.uint8)
    p.show_target(a, 0.02)
    p.fixation(0.02)
    p.crossfade(a, b, 0.05, fps=30)
    assert fake.imshow_calls > 0
    assert fake.addweighted_calls > 0  # the morph actually blended frames


def test_quit_key_raises_abort(fake_cv2):
    from src.signal_service.stimulus_presenter import AbortPresentation

    fake = fake_cv2(key_script=[ord("q")])
    p = _presenter(fake)
    with pytest.raises(AbortPresentation):
        p.paint(np.zeros((128, 128, 3), np.uint8), "A")


def test_capture_paints_during_recording(fake_cv2):
    """capture_window must repaint the stimulus while pulling EEG samples."""
    fake = fake_cv2()
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import record_reward_trials as rec
    from src.signal_service.mock_preference import MockPreferenceEEGSource
    from src.signal_service.learned_reward import EEGFeatureExtractor

    ch = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    src = MockPreferenceEEGSource(ch, 128, signal_gain=0.5)
    src.connect()
    stream = src.stream()
    ex = EEGFeatureExtractor(128, ch, window_s=1.0)
    p = _presenter(fake)
    image = np.zeros((128, 128, 3), np.uint8)
    before = fake.imshow_calls
    vec, names, artifact = rec.capture_window(stream, ex, None, 128, 1.0,
                                              mock_src=src, preference=0.5,
                                              presenter=p, image=image, caption="A")
    assert fake.imshow_calls > before          # painted during capture
    assert vec.shape[0] == len(names) > 0       # features still produced
