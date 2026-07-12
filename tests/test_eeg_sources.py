import json

from src.signal_service.eeg_sources import EmotivCortexSource


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)

    def recv(self):
        return next(self.messages)


def test_emotiv_stream_uses_cortex_cols_and_message_time():
    source = EmotivCortexSource()
    source._ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "eeg": [
                        14,
                        0,
                        4161.41,
                        4212.051,
                        4135.0,
                        4161.538,
                        4195.0,
                        4184.103,
                        4182.0,
                        4201.0,
                        4177.0,
                        4160.0,
                        4149.0,
                        4130.0,
                        4122.0,
                        4111.0,
                        0,
                        0,
                        [],
                    ],
                    "sid": "session",
                    "time": 1559902873.8976,
                }
            )
        ]
    )
    source._eeg_cols = [
        "COUNTER",
        "INTERPOLATED",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "RAW_CQ",
        "MARKER_HARDWARE",
        "MARKERS",
    ]

    t, sample = next(source.stream())

    assert t == 1559902873.8976
    assert sample["F3"] == 4135.0
    assert sample["F4"] == 4130.0


def test_emotiv_extracts_eeg_cols_from_subscribe_result():
    result = {
        "success": [
            {
                "streamName": "eeg",
                "cols": ["COUNTER", "INTERPOLATED", "AF3", "F3", "F4", "MARKERS"],
            }
        ],
        "failure": [],
    }

    assert EmotivCortexSource._extract_eeg_cols(result) == [
        "COUNTER",
        "INTERPOLATED",
        "AF3",
        "F3",
        "F4",
        "MARKERS",
    ]

def test_emotiv_formats_unpublished_app_error_with_owner_hint():
    message = EmotivCortexSource._format_api_error(
        "authorize",
        {
            "code": -32142,
            "message": "Unpublished application can only be accessed by the Owner.",
        },
    )

    assert "Cortex API error on authorize" in message
    assert "logged in as the account that created this EMOTIV_CLIENT_ID" in message
    assert "publish/share the app through EMOTIV" in message


def test_emotiv_formats_unknown_api_error_with_raw_payload():
    error = {"code": -39999, "message": "unexpected"}

    assert (
        EmotivCortexSource._format_api_error("authorize", error)
        == f"Cortex API error on authorize: {error}"
    )
