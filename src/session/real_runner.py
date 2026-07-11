"""FAA-driven optimizer session."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.common.config import Config, emotiv_credentials
from src.signal_service.baseline import calibrate_baseline
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource
from src.signal_service.service import FAARewardSource, build_faa_service

from .diffusion_client import DiffusionClient
from .frame_store import FrameStore
from .optimizer_loop import OptimizerRenderLoop


def reward_cue(reward: float) -> str:
    if reward > 0.15:
        return "lean-in"
    if reward < -0.15:
        return "pull-away"
    return "neutral"


class RealEEGOptimizerRunner:
    def __init__(self, args) -> None:
        self.args = args
        self.config = Config.load()
        if args.algorithm:
            self.config.optimizer.algorithm = args.algorithm
        self.baseline_seconds = (
            self.config.faa.baseline_duration_s if args.baseline is None else args.baseline
        )
        if args.mock:
            self.eeg_source = MockEEGSource(
                self.config.eeg.channels, self.config.eeg.sample_rate_hz
            )
        else:
            client_id, client_secret = emotiv_credentials()
            self.eeg_source = EmotivCortexSource(client_id, client_secret)

    def run(self) -> None:
        source_label = "mock" if self.args.mock else "EPOC X via Cortex"
        print(f"[real-eeg-opt] connecting ({source_label}) ...")
        self.eeg_source.connect()
        try:
            reward_source = self._build_reward_source()
            self._calibrate(reward_source)
            client = None if self.args.dry_run else DiffusionClient(
                self.args.server_url, self.args.timeout
            )
            loop = OptimizerRenderLoop(
                self.config,
                frames_per_step=self.args.frames_per_step,
                client=client,
                frame_store=FrameStore(),
                capture_snapshots=True,
            )
            self._run_loop(loop, reward_source)
        finally:
            self.eeg_source.close()

    def _build_reward_source(self) -> FAARewardSource:
        signal_service = build_faa_service(self.config, self.eeg_source)
        return signal_service.reward_source  # type: ignore[return-value]

    def _calibrate(self, reward_source: FAARewardSource) -> None:
        if self.baseline_seconds <= 0:
            print("[real-eeg-opt] --baseline 0: no calibration")
            return
        print(f"[real-eeg-opt] hold still and rest for {self.baseline_seconds:.0f}s ...")
        baseline = calibrate_baseline(
            reward_source.computer,
            self.eeg_source.stream(),
            duration_s=self.baseline_seconds,
        )
        print(
            f"[real-eeg-opt] baseline fitted: mean={baseline.mean:+.4f} "
            f"std={baseline.std:.4f} n={baseline.n}"
        )

    def _run_loop(self, loop: OptimizerRenderLoop, reward_source: FAARewardSource) -> None:
        optimizer = loop.optimizer
        print(
            f"[real-eeg-opt] algorithm={self.config.optimizer.algorithm} "
            f"{'DRY-RUN' if self.args.dry_run else self.args.server_url}"
        )
        try:
            while True:
                message = reward_source.read_reward()
                if message is None:
                    continue
                result = optimizer.observe_reward(message.r)
                if result is None:
                    continue
                loop.render_candidate(np.asarray(result.z, dtype=float))
                raw = message.raw_faa if message.raw_faa is not None else float("nan")
                print(
                    f"{result.step_index:>4} {result.state:>9} "
                    f"{result.reward_estimate:>+7.2f} {raw:>+7.2f} "
                    f"{reward_cue(result.reward_estimate):<10} {loop.frame_count}"
                )
                if optimizer.state_machine.should_stop():
                    break
        except KeyboardInterrupt:
            print("\n[real-eeg-opt] interrupted")
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"[real-eeg-opt] server error: {exc}\n"
                f"          is the diffusion render server listening at {self.args.server_url}?"
            )

        loop.save_final_frame()
        print(
            f"[real-eeg-opt] final state={optimizer.state_machine.state} "
            f"steps={optimizer.state_machine.step_index} frames emitted={loop.frame_count}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an FAA-driven optimizer session.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--server-url", default="http://localhost:8766")
    parser.add_argument(
        "--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None
    )
    parser.add_argument("--frames-per-step", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    RealEEGOptimizerRunner(build_parser().parse_args(argv)).run()
