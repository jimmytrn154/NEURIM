#!/usr/bin/env python3
"""
Real-time EMOTIV EPOC X Attention + Relaxation monitor.

Prerequisites
-------------
1. Install and open EMOTIV Launcher.
2. Log in and connect the EPOC X by USB receiver or Bluetooth.
3. Create a Cortex application in your EMOTIV account dashboard.
4. Install the only Python dependency:

       python3 -m pip install websocket-client

5. Set your Cortex application credentials:

       export EMOTIV_CLIENT_ID="your_client_id"
       export EMOTIV_CLIENT_SECRET="your_client_secret"

6. Run:

       python3 emotiv_attention_relaxation.py

Optional examples
-----------------
Save every metric sample to CSV:

       python3 emotiv_attention_relaxation.py --csv metrics.csv

Emit newline-delimited JSON for another program:

       python3 emotiv_attention_relaxation.py --json

Select a specific headset:

       python3 emotiv_attention_relaxation.py --headset-id EPOCPLUS-XXXXXXXX

Notes
-----
- Values are EMOTIV proprietary performance metrics in [0, 1], not literal
  percentages of a person's mental state.
- With a Performance Metrics ("pm") license scope and an activated session,
  metrics normally arrive at 2 Hz. Otherwise they may arrive at 0.1 Hz.
- Press Ctrl+C to stop and close the Cortex session cleanly.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import ssl
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

try:
    import websocket
except ImportError:
    print(
        "Missing dependency: websocket-client\n"
        "Install it with:\n"
        "  python3 -m pip install websocket-client",
        file=sys.stderr,
    )
    raise SystemExit(2)

CORTEX_URL = "wss://localhost:6868"


class CortexError(RuntimeError):
    """Raised when Cortex returns a JSON-RPC error."""

    def __init__(self, method: str, error: Dict[str, Any]):
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message", "Unknown Cortex error")
        super().__init__(f"{method} failed [{self.code}]: {self.message}")


class CortexClient:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.ws = None
        self.next_id = 1
        self.pending: Deque[Dict[str, Any]] = deque()

    def connect(self) -> None:
        # Prevent system proxy variables from intercepting localhost traffic.
        no_proxy = os.environ.get("NO_PROXY", "")
        entries = [x.strip() for x in no_proxy.split(",") if x.strip()]
        for host in ("localhost", "127.0.0.1"):
            if host not in entries:
                entries.append(host)
        os.environ["NO_PROXY"] = ",".join(entries)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

        try:
            self.ws = websocket.create_connection(
                CORTEX_URL,
                timeout=10,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                http_proxy_host=None,
                http_proxy_port=None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Cortex at wss://localhost:6868.\n"
                "Open EMOTIV Launcher, log in, and confirm Cortex is running.\n"
                f"Underlying error: {exc}"
            ) from exc

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _recv_json(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("Cortex WebSocket is not connected.")

        if timeout is not None:
            self.ws.settimeout(timeout)

        raw = self.ws.recv()
        if not raw:
            raise RuntimeError("Cortex closed the WebSocket connection.")

        msg = json.loads(raw)
        if self.debug:
            print(f"\n<- {json.dumps(msg, ensure_ascii=False)}", file=sys.stderr)
        return msg

    def rpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Any:
        if self.ws is None:
            raise RuntimeError("Cortex WebSocket is not connected.")

        request_id = self.next_id
        self.next_id += 1

        request: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        if self.debug:
            safe_request = json.loads(json.dumps(request))
            safe_params = safe_request.get("params", {})
            for secret_key in ("clientSecret", "cortexToken"):
                if secret_key in safe_params:
                    safe_params[secret_key] = "***"
            print(f"\n-> {json.dumps(safe_request)}", file=sys.stderr)

        self.ws.send(json.dumps(request))
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for Cortex method {method}.")

            msg = self._recv_json(timeout=remaining)
            if msg.get("id") != request_id:
                self.pending.append(msg)
                continue

            if "error" in msg:
                raise CortexError(method, msg["error"])
            return msg.get("result")

    def receive(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        if self.pending:
            return self.pending.popleft()

        try:
            return self._recv_json(timeout=timeout)
        except websocket.WebSocketTimeoutException:
            return None


@dataclass
class TimedMedian:
    seconds: float
    values: Deque[Tuple[float, float]]

    @classmethod
    def create(cls, seconds: float) -> "TimedMedian":
        return cls(seconds=seconds, values=deque())

    def add(self, timestamp: float, value: float) -> float:
        self.values.append((timestamp, value))
        cutoff = timestamp - self.seconds
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()
        return float(statistics.median(v for _, v in self.values))


def first_present(data: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def number_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def active_or_default(
    data: Dict[str, Any],
    names: Sequence[str],
    metric_value: Optional[float],
) -> bool:
    value = first_present(data, names)
    if value is None:
        # Older Cortex responses may omit isActive fields.
        return metric_value is not None
    return bool(value)


def bar(value: Optional[float], width: int = 20) -> str:
    if value is None:
        return "[" + ("-" * width) + "]"
    value = max(0.0, min(1.0, value))
    filled = round(value * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def level(value: Optional[float]) -> str:
    if value is None:
        return "N/A "
    if value >= 0.65:
        return "HIGH"
    if value >= 0.40:
        return "MED "
    return "LOW "


def format_metric(value: Optional[float]) -> str:
    return " N/A" if value is None else f"{value:4.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream real-time EMOTIV attention, relaxation, and EEG quality."
    )
    parser.add_argument(
        "--client-id",
        default=os.getenv("EMOTIV_CLIENT_ID"),
        help="Cortex client ID; defaults to EMOTIV_CLIENT_ID.",
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("EMOTIV_CLIENT_SECRET"),
        help="Cortex client secret; defaults to EMOTIV_CLIENT_SECRET.",
    )
    parser.add_argument(
        "--headset-id",
        default=None,
        help="Specific headset ID. By default, use the first available headset.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3.0,
        help="Time window for median smoothing. Default: 3 seconds.",
    )
    parser.add_argument(
        "--quality-min",
        type=float,
        default=40.0,
        help="Minimum overall EEG quality (0-100) for valid=True. Default: 40.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print newline-delimited JSON instead of the terminal gauge.",
    )
    parser.add_argument(
        "--basic-session",
        action="store_true",
        help=(
            "Do not activate the Cortex session. This avoids licensed activation, "
            "but performance metrics may update only once every 10 seconds."
        ),
    )
    parser.add_argument(
        "--access-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for Launcher application approval. Default: 120.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive.")
    if not 0 <= args.quality_min <= 100:
        parser.error("--quality-min must be between 0 and 100.")
    return args


def get_credentials(args: argparse.Namespace) -> Tuple[str, str]:
    client_id = args.client_id or input("Cortex client ID: ").strip()
    client_secret = args.client_secret or getpass.getpass(
        "Cortex client secret: "
    ).strip()

    if not client_id or not client_secret:
        raise RuntimeError("A Cortex client ID and client secret are required.")
    return client_id, client_secret


def ensure_access(
    client: CortexClient,
    client_id: str,
    client_secret: str,
    timeout: float,
) -> None:
    params = {"clientId": client_id, "clientSecret": client_secret}
    result = client.rpc("requestAccess", params)

    if result.get("accessGranted"):
        return

    print(
        "\nApprove this application in EMOTIV Launcher.\n"
        "The script will detect approval automatically.",
        file=sys.stderr,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        result = client.rpc("requestAccess", params)
        if result.get("accessGranted"):
            print("Application access approved.", file=sys.stderr)
            return

    raise TimeoutError(
        "Application approval was not detected. Approve it in EMOTIV Launcher "
        "and run the script again."
    )


def query_headsets(
    client: CortexClient,
    headset_id: Optional[str] = None,
) -> list[Dict[str, Any]]:
    params = {"id": headset_id} if headset_id else None
    result = client.rpc("queryHeadsets", params)
    return list(result or [])


def wait_for_headset(
    client: CortexClient,
    headset_id: Optional[str],
    timeout: float = 30.0,
) -> Dict[str, Any]:
    headsets = query_headsets(client, headset_id)

    if not headsets:
        print("Scanning for an EMOTIV headset...", file=sys.stderr)
        try:
            client.rpc("controlDevice", {"command": "refresh"}, timeout=10)
        except CortexError:
            # Some Launcher states may already be scanning. Continue polling.
            pass

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(1)
            headsets = query_headsets(client, headset_id)
            if headsets:
                break

    if not headsets:
        requested = f" {headset_id}" if headset_id else ""
        raise RuntimeError(
            f"No EMOTIV headset{requested} was found. Turn on the EPOC X, "
            "connect its USB receiver or Bluetooth, and check EMOTIV Launcher."
        )

    connected = next(
        (headset for headset in headsets if headset.get("status") == "connected"),
        None,
    )
    if connected is not None:
        return connected

    headset = headsets[0]
    selected_id = headset.get("id")
    if not selected_id:
        raise RuntimeError("Cortex returned a headset without an ID.")

    if headset.get("connectedBy") == "usb cable":
        raise RuntimeError(
            "The headset is connected by USB cable. Cortex sessions require "
            "a wireless connection through the USB receiver or Bluetooth."
        )

    print(f"Connecting to {selected_id}...", file=sys.stderr)
    client.rpc(
        "controlDevice",
        {"command": "connect", "headset": selected_id},
        timeout=15,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(1)
        current = query_headsets(client, selected_id)
        if current and current[0].get("status") == "connected":
            return current[0]

    raise TimeoutError(f"Headset {selected_id} did not reach connected status.")


def open_csv(path: Optional[Path]):
    if path is None:
        return None, None

    path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "timestamp",
        "attention_raw",
        "attention_smoothed",
        "relaxation_raw",
        "relaxation_smoothed",
        "attention_active",
        "relaxation_active",
        "eeg_quality_overall",
        "sample_rate_quality",
        "valid",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return csv_file, writer


def main() -> int:
    args = parse_args()
    client_id, client_secret = get_credentials(args)

    client = CortexClient(debug=args.debug)
    token: Optional[str] = None
    session_id: Optional[str] = None
    csv_file = None

    try:
        print("Connecting to EMOTIV Cortex...", file=sys.stderr)
        client.connect()

        ensure_access(
            client,
            client_id,
            client_secret,
            args.access_timeout,
        )

        auth_result = client.rpc(
            "authorize",
            {
                "clientId": client_id,
                "clientSecret": client_secret,
            },
        )
        token = auth_result["cortexToken"]

        headset = wait_for_headset(client, args.headset_id)
        headset_id = headset["id"]
        print(
            f"Using {headset_id} "
            f"({headset.get('connectedBy', 'unknown connection')}).",
            file=sys.stderr,
        )

        requested_status = "open" if args.basic_session else "active"
        actual_status = requested_status

        try:
            session = client.rpc(
                "createSession",
                {
                    "cortexToken": token,
                    "headset": headset_id,
                    "status": requested_status,
                },
            )
        except CortexError as active_error:
            if requested_status == "open":
                raise

            print(
                "\nCould not activate a licensed session:\n"
                f"  {active_error}\n"
                "Falling back to a basic open session. Attention/relaxation "
                "may update only once every 10 seconds.\n",
                file=sys.stderr,
            )
            actual_status = "open"
            session = client.rpc(
                "createSession",
                {
                    "cortexToken": token,
                    "headset": headset_id,
                    "status": "open",
                },
            )

        session_id = session["id"]
        print(
            f"Cortex session created ({actual_status}).",
            file=sys.stderr,
        )

        subscription = client.rpc(
            "subscribe",
            {
                "cortexToken": token,
                "session": session_id,
                "streams": ["met", "eq"],
            },
        )

        columns: Dict[str, list[str]] = {}
        for success in subscription.get("success", []):
            columns[success["streamName"]] = list(success["cols"])

        failures = subscription.get("failure", [])
        for failure in failures:
            print(
                f"Stream {failure.get('streamName')} unavailable: "
                f"{failure.get('message')}",
                file=sys.stderr,
            )

        if "met" not in columns:
            raise RuntimeError(
                "The performance-metrics stream ('met') is unavailable. "
                "Check your Cortex application/license and EPOC X connection."
            )

        print(f"MET columns: {', '.join(columns['met'])}", file=sys.stderr)
        if "eq" in columns:
            print("EEG quality stream enabled.", file=sys.stderr)
        else:
            print(
                "EEG quality stream unavailable; values will not be quality-gated.",
                file=sys.stderr,
            )

        csv_file, csv_writer = open_csv(args.csv)
        if args.csv:
            print(f"Writing CSV to {args.csv}", file=sys.stderr)

        attention_filter = TimedMedian.create(args.window_seconds)
        relaxation_filter = TimedMedian.create(args.window_seconds)

        attention_smooth: Optional[float] = None
        relaxation_smooth: Optional[float] = None
        eeg_quality: Optional[float] = None
        sample_rate_quality: Optional[float] = None
        previous_met_time: Optional[float] = None
        warned_low_rate = False

        if not args.json:
            print(
                "\nStreaming. Values are metric scores from 0 to 1. Ctrl+C stops.\n",
                file=sys.stderr,
            )

        while True:
            msg = client.receive(timeout=2.0)
            if msg is None:
                continue

            if "warning" in msg and args.debug:
                print(f"\nCortex warning: {msg['warning']}", file=sys.stderr)

            if "eq" in msg and "eq" in columns:
                eq_values = dict(zip(columns["eq"], msg["eq"]))
                eeg_quality = number_or_none(
                    first_present(eq_values, ("overall", "OVERALL"))
                )
                sample_rate_quality = number_or_none(
                    first_present(
                        eq_values,
                        ("sampleRateQuality", "sample_rate_quality"),
                    )
                )

            if "met" not in msg:
                continue

            timestamp = number_or_none(msg.get("time")) or time.time()
            met_values = dict(zip(columns["met"], msg["met"]))

            attention_raw = number_or_none(
                first_present(
                    met_values,
                    ("attention", "foc", "focus"),
                )
            )
            relaxation_raw = number_or_none(
                first_present(
                    met_values,
                    ("rel", "relaxation"),
                )
            )

            attention_active = active_or_default(
                met_values,
                (
                    "attention.isActive",
                    "foc.isActive",
                    "focus.isActive",
                ),
                attention_raw,
            )
            relaxation_active = active_or_default(
                met_values,
                (
                    "rel.isActive",
                    "relaxation.isActive",
                ),
                relaxation_raw,
            )

            if attention_active and attention_raw is not None:
                attention_smooth = attention_filter.add(
                    timestamp,
                    attention_raw,
                )
            else:
                attention_smooth = None

            if relaxation_active and relaxation_raw is not None:
                relaxation_smooth = relaxation_filter.add(
                    timestamp,
                    relaxation_raw,
                )
            else:
                relaxation_smooth = None

            quality_ok = (
                eeg_quality is None or eeg_quality >= args.quality_min
            )
            sample_rate_ok = (
                sample_rate_quality is None or sample_rate_quality >= 0
            )
            valid = (
                quality_ok
                and sample_rate_ok
                and attention_active
                and relaxation_active
                and attention_smooth is not None
                and relaxation_smooth is not None
            )

            row = {
                "timestamp": timestamp,
                "attention_raw": attention_raw,
                "attention_smoothed": attention_smooth,
                "relaxation_raw": relaxation_raw,
                "relaxation_smoothed": relaxation_smooth,
                "attention_active": attention_active,
                "relaxation_active": relaxation_active,
                "eeg_quality_overall": eeg_quality,
                "sample_rate_quality": sample_rate_quality,
                "valid": valid,
            }

            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_file.flush()

            if args.json:
                print(json.dumps(row, separators=(",", ":")), flush=True)
            else:
                quality_text = (
                    " N/A" if eeg_quality is None else f"{eeg_quality:5.1f}"
                )
                validity = "VALID" if valid else "REJECT"
                line = (
                    f"\rATT {bar(attention_smooth)} "
                    f"{format_metric(attention_smooth)} {level(attention_smooth)}  "
                    f"REL {bar(relaxation_smooth)} "
                    f"{format_metric(relaxation_smooth)} {level(relaxation_smooth)}  "
                    f"EEG-Q {quality_text}/100  {validity}"
                )
                print(line, end="", flush=True)

            if previous_met_time is not None:
                interval = timestamp - previous_met_time
                if interval > 5 and not warned_low_rate:
                    print(
                        "\n\nWarning: performance metrics are arriving slowly "
                        f"(last interval: {interval:.1f}s). An activated session "
                        "with the 'pm' license scope is required for approximately "
                        "2 Hz performance metrics.",
                        file=sys.stderr,
                    )
                    warned_low_rate = True
            previous_met_time = timestamp

    except KeyboardInterrupt:
        if not args.json:
            print()
        print("Stopping...", file=sys.stderr)
        return 0
    except (CortexError, RuntimeError, TimeoutError, KeyError) as exc:
        if not args.json:
            print()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if csv_file is not None:
            csv_file.close()

        if token and session_id:
            try:
                client.rpc(
                    "updateSession",
                    {
                        "cortexToken": token,
                        "session": session_id,
                        "status": "close",
                    },
                    timeout=5,
                )
            except Exception:
                pass

        client.close()


if __name__ == "__main__":
    raise SystemExit(main())