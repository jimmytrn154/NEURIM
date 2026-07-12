"""EEG data sources. All of them yield (timestamp, {channel_name: value}) samples.

The rest of the Signal service (faa.py, service.py) doesn't care which of
these is plugged in - that's the point of the abstraction.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterator, Protocol

import numpy as np


class EEGSource(Protocol):
    def connect(self) -> None: ...
    def stream(self) -> Iterator[tuple[float, dict[str, Any]]]: ...
    def close(self) -> None: ...


class MockEEGSource:
    """Synthetic 14-channel EEG for development without hardware.

    Alpha-band power across the frontal mirror pairs is modulated by `bias(t)`
    (defaults to slow random drift) so FAARewardComputer has something
    non-trivial to decode.
    Pass a custom `bias_fn(t) -> float in [-1, 1]` to script a known ground
    truth for tests (e.g. "reward should rise for the first 10s").
    """

    def __init__(
        self,
        channels: list[str],
        sample_rate_hz: int = 128,
        bias_fn=None,
        seed: int = 0,
    ):
        self.channels = channels
        self.fs = sample_rate_hz
        self._bias_fn = bias_fn or (lambda t: 0.0)
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._dt = 1.0 / sample_rate_hz

    def connect(self) -> None:
        self._t = 0.0

    def stream(self) -> Iterator[tuple[float, dict[str, float]]]:
        while True:
            self._t += self._dt
            bias = float(np.clip(self._bias_fn(self._t), -1.0, 1.0))
            sample = {}
            for ch in self.channels:
                # Base alpha oscillation (10 Hz) plus 1/f-ish noise.
                alpha = np.sin(2 * np.pi * 10.0 * self._t)
                noise = self._rng.normal(0, 0.3)
                gain = 1.0
                if ch in {"F8", "AF4", "F4", "FC6"}:
                    gain = 1.0 + 0.6 * bias
                elif ch in {"F7", "AF3", "F3", "FC5"}:
                    gain = 1.0 - 0.6 * bias
                sample[ch] = alpha * gain + noise
            yield self._t, sample

    def close(self) -> None:
        pass


class EmotivCortexSource:
    """EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).

    Flow: requestAccess (poll until a human clicks Accept in EMOTIV Launcher)
    -> authorize -> queryHeadsets -> controlDevice (connect if needed) ->
    createSession(headset) -> subscribe("eeg"). No official PyPI SDK exists;
    this talks to the Cortex websocket directly via `websocket-client`.
    Requires EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET, EMOTIV Launcher running
    locally (it hosts the Cortex service this connects to), and the headset
    already paired to it.
    """

    CORTEX_URL = "wss://localhost:6868"
    ACCESS_POLL_INTERVAL_S = 2.0
    ACCESS_POLL_TIMEOUT_S = 60.0
    DEVICE_CONNECT_TIMEOUT_S = 20.0

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        headset_id: str | None = None,
        connect_timeout_s: float = 20.0,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.headset_id = headset_id or os.environ.get("EMOTIV_HEADSET_ID")
        self.connect_timeout_s = connect_timeout_s
        self._ws = None
        self._cortex_token: str | None = None
        self._session_id: str | None = None
        self._req_id = 0
        self._io_lock = threading.RLock()
        # Column labels for the eeg data array, captured from the subscribe
        # response - Cortex sends [COUNTER, INTERPOLATED, <channels>, RAW_CQ,
        # MARKER_HARDWARE, MARKERS], so channels must be mapped by label, not
        # by naive position.
        self._eeg_cols: list[str] | None = None

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _call(self, method: str, params: dict | None = None) -> dict:
        import json

        assert self._ws is not None
        req_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        with self._io_lock:
            self._ws.send(json.dumps(payload))
            while True:
                response = json.loads(self._ws.recv())
                if response.get("id") != req_id:
                    continue
                if "error" in response:
                    raise RuntimeError(self._format_api_error(method, response["error"]))
                return response["result"]

    @staticmethod
    def _format_api_error(method: str, error: dict) -> str:
        code = error.get("code")
        message = error.get("message", "")
        detail = f"Cortex API error on {method}: {error}"
        if code == -32142:
            return (
                f"{detail}\n"
                "Unpublished Cortex apps can only be authorized by the EMOTIV ID "
                "that owns the Cortex app credentials. Open EMOTIV Launcher and "
                "confirm it is logged in as the account that created this "
                "EMOTIV_CLIENT_ID, or publish/share the app through EMOTIV before "
                "using these credentials from another account."
            )
        if code == -32102:
            return (
                f"{detail}\n"
                "Open EMOTIV Launcher and approve this application in the pending "
                "access request."
            )
        if code == -32021:
            return (
                f"{detail}\n"
                "Check EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET; Cortex rejected "
                "the client credentials."
            )
        if code == -32033:
            return (
                f"{detail}\n"
                "Log in to EMOTIV Launcher before authorizing the Cortex app."
            )
        if message:
            return detail
        return f"Cortex API error on {method}: code={code}"

    def _query_headsets(self) -> list[dict]:
        params = {"id": self.headset_id} if self.headset_id else None
        return self._call("queryHeadsets", params)

    def _wait_for_connected_headset(self, headset_id: str) -> dict:
        deadline = time.monotonic() + self.connect_timeout_s
        last_status = "unknown"
        while time.monotonic() < deadline:
            matches = self._call("queryHeadsets", {"id": headset_id})
            if matches:
                headset = matches[0]
                last_status = headset.get("status", "unknown")
                if last_status == "connected":
                    return headset
            time.sleep(1.0)
        raise RuntimeError(
            f"Timed out waiting for Cortex headset {headset_id!r} to connect "
            f"(last status: {last_status})"
        )

    def _select_headset(self) -> str:
        self._call("controlDevice", {"command": "refresh"})

        deadline = time.monotonic() + self.connect_timeout_s
        headsets: list[dict] = []
        while time.monotonic() < deadline:
            headsets = self._query_headsets()
            if headsets:
                break
            time.sleep(1.0)

        if not headsets:
            target = f" matching {self.headset_id!r}" if self.headset_id else ""
            raise RuntimeError(f"No EMOTIV headset{target} found by Cortex")

        connected = [h for h in headsets if h.get("status") == "connected"]
        headset = connected[0] if connected else headsets[0]
        headset_id = headset["id"]

        if headset.get("status") != "connected":
            self._call("controlDevice", {"command": "connect", "headset": headset_id})
            headset = self._wait_for_connected_headset(headset_id)

        if headset.get("connectedBy") == "usb cable":
            raise RuntimeError(
                f"Cortex reports headset {headset_id!r} is connected by USB cable; "
                "createSession requires dongle or Bluetooth."
            )

        self.headset_id = headset_id
        return headset_id

    @staticmethod
    def _extract_eeg_cols(subscription_result: dict) -> list[str] | None:
        """Return Cortex EEG column labels from a subscribe response."""
        for item in subscription_result.get("success", []):
            if item.get("streamName") == "eeg" and item.get("cols"):
                return list(item["cols"])
        return None

    @staticmethod
    def _is_eeg_sensor_col(col: Any, channels: list[str] | None = None) -> bool:
        if not isinstance(col, str):
            return False
        if channels is not None:
            return col in channels
        return col not in {
            "COUNTER",
            "INTERPOLATED",
            "RAW_CQ",
            "MARKER_HARDWARE",
            "MARKERS",
        }

    def _wait_for_access(self) -> None:
        deadline = time.monotonic() + self.ACCESS_POLL_TIMEOUT_S
        printed_prompt = False
        while True:
            result = self._call(
                "requestAccess", {"clientId": self.client_id, "clientSecret": self.client_secret}
            )
            if result.get("accessGranted"):
                return
            if not printed_prompt:
                print(
                    "[emotiv] waiting for approval - open EMOTIV Launcher and click "
                    "'Accept' on the access request popup"
                )
                printed_prompt = True
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Cortex access not granted after {self.ACCESS_POLL_TIMEOUT_S:.0f}s - "
                    "check EMOTIV Launcher for a pending approval request"
                )
            time.sleep(self.ACCESS_POLL_INTERVAL_S)

    def connect(self) -> None:
        import ssl

        import websocket

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET are required to connect to Cortex"
            )
        # Cortex serves a self-signed cert on localhost; there's no real MITM
        # risk to guard against on a loopback connection to your own machine.
        self._ws = websocket.create_connection(
            self.CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE}
        )
        self._wait_for_access()
        auth = self._call(
            "authorize", {"clientId": self.client_id, "clientSecret": self.client_secret, "debit": 10}
        )
        self._cortex_token = auth["cortexToken"]
        headset_id = self._select_headset()
        session = self._call(
            "createSession",
            {"cortexToken": self._cortex_token, "headset": headset_id, "status": "active"},
        )
        self._session_id = session["id"]
        subscription = self._call(
            "subscribe",
            {
                "cortexToken": self._cortex_token,
                "session": self._session_id,
                "streams": ["eeg"],
            },
        )
        self._eeg_cols = self._extract_eeg_cols(subscription)

    def stream(self, channels: list[str] | None = None) -> Iterator[tuple[float, dict[str, Any]]]:
        import json

        assert self._ws is not None, "call connect() first"
        while True:
            with self._io_lock:
                msg = json.loads(self._ws.recv())
            if "eeg" not in msg:
                continue
            values = msg["eeg"]
            if self._eeg_cols:
                sample = {
                    col: value
                    for col, value in zip(self._eeg_cols, values)
                    if self._is_eeg_sensor_col(col, channels) and isinstance(value, (int, float))
                }
            else:
                if channels is None:
                    raise RuntimeError(
                        "Cortex EEG column labels are unavailable; pass channels to stream()"
                    )
                sample = dict(zip(channels, values[2:]))
            t = msg.get("time", time.time())
            yield t, sample

    def is_headset_connected(self) -> bool:
        if self._ws is None or not self.headset_id:
            return False
        matches = self._call("queryHeadsets", {"id": self.headset_id})
        return bool(matches and matches[0].get("status") == "connected")

    def close(self) -> None:
        # Unpublished Cortex apps are limited to one active session at a
        # time - if this isn't told to close cleanly, Cortex keeps it "active"
        # server-side and the *next* connect attempt gets rejected (confusingly,
        # with the same "unpublished application" error as an owner mismatch).
        if self._ws is not None and self._cortex_token and self._session_id:
            try:
                self._call(
                    "updateSession",
                    {"cortexToken": self._cortex_token, "session": self._session_id, "status": "close"},
                )
            except Exception:
                pass  # best-effort - the socket may already be half-dead
        if self._ws is not None:
            self._ws.close()
            self._ws = None
        self._cortex_token = None
        self._session_id = None
        self._eeg_cols = None


class BrainFlowLSLSource:
    """Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports
    pylsl so the rest of the codebase works without it installed.
    """

    def __init__(self, channels: list[str], stream_name: str = "obci_eeg"):
        self.channels = channels
        self.stream_name = stream_name
        self._inlet = None

    def connect(self) -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("name", self.stream_name, timeout=5.0)
        if not streams:
            raise RuntimeError(f"No LSL stream named '{self.stream_name}' found")
        self._inlet = StreamInlet(streams[0])

    def stream(self) -> Iterator[tuple[float, dict[str, float]]]:
        assert self._inlet is not None, "call connect() first"
        while True:
            sample, timestamp = self._inlet.pull_sample()
            yield timestamp, dict(zip(self.channels, sample))

    def close(self) -> None:
        self._inlet = None


def wall_clock_pace(sample_iter, fs: float):
    """Wrap a sample iterator to yield in real time (for mock sources that
    would otherwise produce samples faster than realtime)."""
    period = 1.0 / fs
    next_tick = time.monotonic()
    for item in sample_iter:
        yield item
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
