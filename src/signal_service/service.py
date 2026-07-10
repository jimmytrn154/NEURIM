"""The Signal service: whatever the source, emit r(t) a few times a second.

Downstream (Optimizer) never learns whether r(t) came from a brain, a
keyboard, or a script - see fake_reward.py and eeg_sources.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, AsyncIterator, Callable

from src.common.config import Config
from src.common.messages import RewardMessage
from src.signal_service.eeg_sources import EEGSource
from src.signal_service.fake_reward import RewardSource

if TYPE_CHECKING:
    from src.signal_service.faa import FAARewardComputer


class FAARewardSource(RewardSource):
    """Wraps an EEGSource + FAARewardComputer behind the RewardSource interface."""

    def __init__(self, eeg_source: EEGSource, computer: FAARewardComputer):
        self.eeg_source = eeg_source
        self.computer = computer
        self._stream = None

    def _pump_until_ready(self) -> None:
        if self._stream is None:
            self._stream = self.eeg_source.stream()
        while not self.computer.ready():
            _t, sample = next(self._stream)
            self.computer.push_sample(sample)

    def read_reward(self) -> RewardMessage | None:
        self._pump_until_ready()
        raw = self.computer.raw_value()
        r = self.computer.reward()
        if r is None:
            return None
        return RewardMessage(
            r=r,
            raw_faa=raw,
            source="eeg",
            eeg_features=self.computer.eeg_features(reward=r, raw=raw),
        )


class SignalService:
    """Runs a RewardSource on a fixed cadence and hands RewardMessages to `on_reward`."""

    def __init__(
        self,
        reward_source: RewardSource,
        update_interval_s: float,
        on_reward: Callable[[RewardMessage], None] | None = None,
    ):
        self.reward_source = reward_source
        self.update_interval_s = update_interval_s
        self.on_reward = on_reward or (lambda msg: None)

    def run_forever(self) -> None:
        while True:
            start = time.monotonic()
            msg = self.reward_source.read_reward()
            if msg is not None:
                self.on_reward(msg)
            elapsed = time.monotonic() - start
            sleep_for = self.update_interval_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    async def stream(self) -> AsyncIterator[RewardMessage]:
        """Async generator form, for the websocket-serving entry point."""
        while True:
            start = time.monotonic()
            msg = self.reward_source.read_reward()
            if msg is not None:
                yield msg
            elapsed = time.monotonic() - start
            sleep_for = self.update_interval_s - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


def build_faa_service(config: Config, eeg_source: EEGSource) -> SignalService:
    # Lazy import: scipy (pulled in by faa.py) is expensive and broken on
    # Python 3.14 pre-release. Only load it when EEG is actually in use.
    from src.signal_service.faa import FAARewardComputer  # noqa: PLC0415

    computer = FAARewardComputer(
        fs=config.eeg.sample_rate_hz,
        channel_left=config.faa.channel_left,
        channel_right=config.faa.channel_right,
        band=config.faa.band_hz,
        window_s=config.faa.window_s,
        clip=config.faa.clip,
        channels=config.eeg.channels,
        channel_pairs=config.faa.channel_pairs,
        pair_weights=config.faa.pair_weights,
    )
    source = FAARewardSource(eeg_source, computer)
    return SignalService(source, update_interval_s=config.faa.update_interval_s)
