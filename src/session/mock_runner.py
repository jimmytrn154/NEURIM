"""Scripted-reward optimizer session."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.common.config import Config
from src.signal_service.fake_reward import ScriptedRewardSource

from .diffusion_client import DiffusionClient
from .frame_store import FrameStore
from .optimizer_loop import OptimizerRenderLoop


class MockOptimizerRunner:
    def __init__(self, args) -> None:
        self.args = args
        self.config = Config.load()
        if args.algorithm:
            self.config.optimizer.algorithm = args.algorithm
        self.target = np.random.default_rng(args.seed).uniform(
            -0.8, 0.8, size=self.config.optimizer.search_dims
        )
        client = None if args.dry_run else DiffusionClient(args.server_url, args.timeout)
        self.loop = OptimizerRenderLoop(
            self.config,
            frames_per_step=args.frames_per_step,
            client=client,
            frame_store=FrameStore(),
        )
        self.reward_source = ScriptedRewardSource(
            target=self.target,
            get_current_z=self.loop.optimizer.pending_candidate,
            noise_std=args.noise,
            seed=args.seed,
        )

    def run(self) -> None:
        optimizer = self.loop.optimizer
        start_distance = float(np.linalg.norm(optimizer.current_z() - self.target))
        max_ticks = self.config.state_machine.max_steps * self.config.optimizer.reward_window_steps + 50
        print(
            f"[mock-opt] target=seed{self.args.seed} ({self.config.optimizer.search_dims}-D) "
            f"algorithm={self.config.optimizer.algorithm} noise={self.args.noise} "
            f"{'DRY-RUN' if self.args.dry_run else self.args.server_url}"
        )
        try:
            for _ in range(max_ticks):
                result = optimizer.observe_reward(self.reward_source.read_reward().r)
                if result is None:
                    continue
                self.loop.render_candidate(np.asarray(result.z, dtype=float))
                distance = float(np.linalg.norm(optimizer.current_z() - self.target))
                print(
                    f"{result.step_index:>4} {result.state:>9} {result.reward_estimate:>+7.2f} "
                    f"{distance:>6.2f} {self.loop.frame_count}"
                )
                if optimizer.state_machine.should_stop():
                    break
        except KeyboardInterrupt:
            print("\n[mock-opt] interrupted")
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"[mock-opt] server error: {exc}\n"
                f"          is the diffusion render server listening at {self.args.server_url}?"
            )

        final_distance = float(np.linalg.norm(optimizer.current_z() - self.target))
        closer = 1.0 - final_distance / start_distance if start_distance > 0 else 0.0
        print(
            f"[mock-opt] final state={optimizer.state_machine.state} "
            f"started {start_distance:.2f} -> ended {final_distance:.2f} "
            f"({closer:.0%} closer) frames emitted={self.loop.frame_count}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a scripted-reward optimizer session.")
    parser.add_argument("--server-url", default="http://localhost:8766")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None
    )
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--frames-per-step", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    MockOptimizerRunner(build_parser().parse_args(argv)).run()
