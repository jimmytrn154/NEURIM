"""EMOTIV Cortex API client for the EPOC X headset.

Requires EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET in the environment (.env).
"""


class EmotivClient:
    """Connection/auth flow against the EMOTIV Cortex API.

    TODO: requestAccess -> authorize -> createSession -> subscribe("eeg")
    """

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        pass

    def connect(self) -> None:
        raise NotImplementedError

    def stream_eeg(self):
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockEmotivClient:
    """Simulated EEG data source with the same interface as EmotivClient,
    for development without hardware.

    TODO: generate synthetic multi-channel EEG samples.
    """

    def __init__(self, channels: list[str], sample_rate_hz: int = 128):
        pass

    def connect(self) -> None:
        pass

    def stream_eeg(self):
        raise NotImplementedError

    def close(self) -> None:
        pass
