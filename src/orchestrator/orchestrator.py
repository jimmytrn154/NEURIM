"""Wires Signal -> Optimizer -> Generator together and drives the timing.

Two flavors, same message protocol (src.common.messages):

  LocalOrchestrator - everything in one process/asyncio loop. What
      scripts/run_fake_loop.py and the convergence tests use; also fine for a
      single-machine demo with real EEG + image generation.

  WebSocketOrchestrator - a hub server so the Signal service (possibly on a
      different machine, e.g. a laptop next to the headset) and the frontend
      display can be separate processes. Optimizer + Generator stay
      co-located with the hub since they're cheap and don't need isolation.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from src.common.config import Config
from src.common.messages import FrameMessage, LatentMessage, RewardMessage
from src.generator.service import GeneratorService, Interpolator
from src.optimizer.service import OptimizerService
from src.signal_service.service import SignalService

if TYPE_CHECKING:
    from src.signal_service.service import FAARewardSource


class LocalOrchestrator:
    def __init__(
        self,
        config: Config,
        signal_service: SignalService,
        generator: GeneratorService,
        optimizer: OptimizerService | None = None,
        on_frame: Callable[[FrameMessage], None] | None = None,
        on_step: Callable[[LatentMessage], None] | None = None,
    ):
        self.config = config
        self.signal_service = signal_service
        self.generator = generator
        self.optimizer = optimizer or OptimizerService(config)
        self.interpolator = Interpolator()
        self.interpolator.set_target(self.optimizer.pending_candidate())
        self.on_frame = on_frame or (lambda msg: None)
        self.on_step = on_step or (lambda msg: None)
        self._stop_event: asyncio.Event | None = None
        self._step_started_at = time.monotonic()
        self._last_step_index = 0
        self._last_state: str = "calibrate"
        self._last_reward_estimate: float = 0.0

    async def calibrate(self) -> None:
        """Real FAA needs 30s of rest to fit the baseline; fake reward sources
        (keyboard/scripted) skip straight to EXPLORE.
        """
        from src.signal_service.service import FAARewardSource  # noqa: PLC0415 (lazy: avoids scipy at startup)

        source = self.signal_service.reward_source
        if isinstance(source, FAARewardSource):
            from src.signal_service.baseline import calibrate_baseline

            calibrate_baseline(
                source.computer,
                source.eeg_source.stream(),
                duration_s=self.config.faa.baseline_duration_s,
            )
        self.optimizer.notify_calibrated()

    async def _reward_loop(self) -> None:
        async for msg in self.signal_service.stream():
            result = self.optimizer.observe_reward(msg.r)
            if result is not None:
                self.interpolator.set_target(np.array(result.z, dtype=float))
                self._last_step_index = result.step_index
                self._last_state = result.state
                self._last_reward_estimate = result.reward_estimate
                self._step_started_at = time.monotonic()
                self.on_step(result)
                if self.optimizer.state_machine.should_stop():
                    assert self._stop_event is not None
                    self._stop_event.set()
                    return

    async def _render_loop(self) -> None:
        frame_interval = 1.0 / self.config.generator.target_fps
        step_interval = self.config.loop.optimizer_step_interval_s
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._step_started_at
            alpha = min(1.0, elapsed / step_interval) if step_interval > 0 else 1.0
            z = self.interpolator.sample(alpha)
            frame = self.generator.render(
                z,
                self._last_step_index,
                state=self._last_state,
                reward_estimate=self._last_reward_estimate,
            )
            self.on_frame(frame)
            await asyncio.sleep(frame_interval)

    async def run(self, max_wall_seconds: float | None = None) -> None:
        self._stop_event = asyncio.Event()
        self._step_started_at = time.monotonic()
        tasks = [asyncio.create_task(self._reward_loop()), asyncio.create_task(self._render_loop())]
        try:
            if max_wall_seconds:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=max_wall_seconds)
            else:
                await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            pass
        finally:
            for t in tasks:
                t.cancel()


class WebSocketOrchestrator:
    """Hub server: Signal service clients push RewardMessages; display clients
    receive a continuous FrameMessage broadcast. Optimizer/Generator run
    in-process on the hub.
    """

    def __init__(self, config: Config, generator: GeneratorService, host: str = "0.0.0.0", port: int = 8765):
        self.config = config
        self.generator = generator
        self.host = host
        self.port = port
        self.optimizer = OptimizerService(config)
        self.interpolator = Interpolator()
        self.interpolator.set_target(self.optimizer.pending_candidate())
        self._display_clients: set = set()
        self._step_started_at = time.monotonic()
        self._last_step_index = 0
        self._last_state: str = "calibrate"
        self._last_reward_estimate: float = 0.0

    async def _handle_signal_client(self, websocket) -> None:
        async for raw in websocket:
            msg = RewardMessage.from_json(raw)
            result = self.optimizer.observe_reward(msg.r)
            if result is not None:
                self.interpolator.set_target(np.array(result.z, dtype=float))
                self._last_step_index = result.step_index
                self._last_state = result.state
                self._last_reward_estimate = result.reward_estimate
                self._step_started_at = time.monotonic()

    async def _handle_display_client(self, websocket) -> None:
        self._display_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self._display_clients.discard(websocket)

    async def _router(self, websocket) -> None:
        hello_raw = await websocket.recv()
        role = json.loads(hello_raw).get("role")
        if role == "signal":
            await self._handle_signal_client(websocket)
        else:
            await self._handle_display_client(websocket)

    async def _broadcast(self, payload: str) -> None:
        stale = []
        for ws in self._display_clients:
            try:
                await ws.send(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._display_clients.discard(ws)

    async def _render_loop(self) -> None:
        frame_interval = 1.0 / self.config.generator.target_fps
        step_interval = self.config.loop.optimizer_step_interval_s
        while True:
            elapsed = time.monotonic() - self._step_started_at
            alpha = min(1.0, elapsed / step_interval) if step_interval > 0 else 1.0
            z = self.interpolator.sample(alpha)
            frame = self.generator.render(
                z,
                self._last_step_index,
                state=self._last_state,
                reward_estimate=self._last_reward_estimate,
            )
            await self._broadcast(frame.to_json())
            await asyncio.sleep(frame_interval)

    async def serve_forever(self) -> None:
        import websockets

        async with websockets.serve(self._router, self.host, self.port):
            await self._render_loop()
