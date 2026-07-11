from datetime import datetime

from src.server.api.eeg import EEGConnectionManager


class FakeSource:
    def __init__(self, samples=None, fail=False):
        self.samples = samples or [(0.0, {"F3": 1.0, "F4": 1.0}), (31.0, {"F3": 1.0, "F4": 1.0})]
        self.fail = fail
        self.connected = False
        self.closed = False

    def connect(self):
        if self.fail:
            raise RuntimeError("no headset")
        self.connected = True

    def stream(self):
        yield from self.samples

    def close(self):
        self.closed = True


def test_connect_failure_records_retry_state():
    source = FakeSource(fail=True)
    manager = EEGConnectionManager(
        source_factory=lambda: source,
        calibrator=lambda _computer, _stream, _seconds: None,
        retry_interval_s=60.0,
    )

    manager._connect_and_calibrate()
    status = manager.status()

    assert status["state"] == "error"
    assert status["last_error"] == "no headset"
    assert status["next_retry_at"] is not None


def test_successful_connect_runs_30_second_calibration():
    calls = []
    source = FakeSource()

    def calibrator(_computer, stream, seconds):
        calls.append(seconds)
        list(stream)

    manager = EEGConnectionManager(source_factory=lambda: source, calibrator=calibrator)

    manager._connect_and_calibrate()
    status = manager.status()

    assert calls == [30.0]
    assert status["state"] == "ready"
    assert status["connected"] is True
    assert status["calibrated"] is True
    assert datetime.fromisoformat(status["last_connected_at"])
    assert datetime.fromisoformat(status["last_calibrated_at"])


def test_retry_now_marks_retry_due():
    manager = EEGConnectionManager(
        source_factory=lambda: FakeSource(fail=True),
        calibrator=lambda _computer, _stream, _seconds: None,
    )

    status = manager.retry_now()

    assert status["state"] == "disconnected"
    assert status["next_retry_at"] is not None
