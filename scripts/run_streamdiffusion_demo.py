#!/usr/bin/env python3
"""Full architecture, single process, on the GPU box: real EEG -> FAA reward ->
Optimizer -> latent-walk generator -> saved frames. No HTTP split, no tunnel
for rendering - only the EEG connection crosses machines.

This is the "everything in one place" alternative to the split architecture
(run_demo.py --backend remote_diffusion talking to run_streamdiffusion_server.py
over HTTP). Reuses load_pipeline()/_fit_projector()/LatentWalkRenderServer
from run_streamdiffusion_server.py directly - same pattern test_streamdiffusion.py
uses - rather than duplicating the plain-diffusers setup. See that file's
docstring for why this no longer uses StreamDiffusion (fixed seeded latent +
fresh render every frame, no img2img continuity to tune).

Run on the GPU server (plain torch-env is enough now - no StreamDiffusion-
specific env/pins needed):

    python scripts/run_streamdiffusion_demo.py --mock
    python scripts/run_streamdiffusion_demo.py

REQUIREMENTS for the real-EEG (non --mock) case, since EEG now runs in this
same process/environment instead of on your local machine:
  - This environment needs the base EEG/FAA deps too:
    pip install -r requirements.txt   (websocket-client, scipy, etc.)
  - EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET must be set in THIS environment
    (the server's), not your local machine's.
  - The tunnel direction flips back from the split architecture: run a
    REVERSE tunnel from your local machine so this server's "localhost:6868"
    reaches your local Cortex/Launcher:
        ssh -R 6868:localhost:6868 <your-ssh-alias>
    (the split architecture's LocalForward 8766 is NOT needed for this script -
    there's no HTTP call to a remote generator here.)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.common.messages import FrameMessage
from src.orchestrator.orchestrator import LocalOrchestrator
from src.signal_service.eeg_sources import EmotivCortexSource, MockEEGSource
from src.signal_service.service import build_faa_service

# Imported from the sibling script (same directory), same pattern
# test_streamdiffusion.py uses - avoids duplicating the plain-diffusers
# setup / z-injection / projector-fit logic.
from run_streamdiffusion_server import LatentWalkRenderServer, _fit_projector, load_pipeline

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


class LatentWalkGeneratorAdapter:
    """Adapts LatentWalkRenderServer (built for an HTTP response: z in,
    PNG bytes out) to the generator interface LocalOrchestrator expects:
    .render(z, step_index, state, reward_estimate) -> FrameMessage. No HTTP,
    no serialization round-trip beyond the PNG encode render_png() already does.
    """

    def __init__(self, render_server: LatentWalkRenderServer):
        self._render_server = render_server

    def render(
        self,
        z,
        step_index: int,
        as_pyramid: bool = False,
        state: str = "explore",
        reward_estimate: float = 0.0,
    ) -> FrameMessage:
        if as_pyramid:
            raise NotImplementedError(
                "pyramid/mirrored-quadrant mode isn't implemented for the in-process "
                "latent-walk adapter - LocalOrchestrator doesn't request it by "
                "default, so seeing this means something explicitly asked for it."
            )
        png_bytes = self._render_server.render_png({"z": list(map(float, z))})
        return FrameMessage(
            frame_b64=base64.b64encode(png_bytes).decode("ascii"),
            z=list(map(float, z)),
            step_index=step_index,
            format="png",
            state=state,
            reward_estimate=reward_estimate,
        )


def _save_frame(frame_msg, name: str = "live_frame.png") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "wb") as f:
        f.write(base64.b64decode(frame_msg.frame_b64))


class _SessionSnapshot:
    """Same start/end capture as run_demo.py, for the optional offline
    DiffMorpher showcase - see scripts/run_diffmorpher_showcase.py.
    """

    def __init__(self):
        self._start_saved = False

    def on_frame(self, frame_msg) -> None:
        _save_frame(frame_msg)
        if not self._start_saved:
            _save_frame(frame_msg, "session_start.png")
            self._start_saved = True

    def save_end(self, frame_msg) -> None:
        _save_frame(frame_msg, "session_end.png")


async def run(config: Config, eeg_source, generator: LatentWalkGeneratorAdapter) -> None:
    signal_service = build_faa_service(config, eeg_source)
    snapshot = _SessionSnapshot()
    orchestrator = LocalOrchestrator(config, signal_service, generator, on_frame=snapshot.on_frame)
    print("[streamdiffusion-demo] calibrating baseline...")
    await orchestrator.calibrate()
    print("[streamdiffusion-demo] running - writing frames to", OUT_DIR / "live_frame.png")
    await orchestrator.run()
    final_state = orchestrator.optimizer.state_machine.state
    final_step = orchestrator.optimizer.state_machine.step_index
    print(f"[streamdiffusion-demo] state={final_state} steps={final_step}")

    final_frame = generator.render(
        orchestrator.optimizer.current_z(),
        final_step,
        state=final_state,
        reward_estimate=orchestrator._last_reward_estimate,
    )
    snapshot.save_end(final_frame)
    print(f"[streamdiffusion-demo] saved {OUT_DIR / 'session_start.png'} and {OUT_DIR / 'session_end.png'}")
    if final_state == "settle":
        print("[streamdiffusion-demo] session settled - run scripts/run_diffmorpher_showcase.py "
              "(in DiffMorpher's own venv) for a polished closing morph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="use synthetic EEG instead of real hardware")
    parser.add_argument("--model-id", default="stabilityai/sd-turbo")
    parser.add_argument("--steps", type=int, default=1,
                         help="denoising steps - 1-4 for sd-turbo, 20-50 for a plain SD checkpoint")
    parser.add_argument("--guidance-scale", type=float, default=0.0,
                         help="0.0 (default, no CFG) for turbo/LCM. ~7-8 for a plain SD checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.generator.remote_diffusion_seed")
    parser.add_argument("--algorithm", choices=["hill_climb", "es_1p1", "gp_bo", "latent_turbo"], default=None)
    args = parser.parse_args()

    config = Config.load()
    if args.algorithm:
        config.optimizer.algorithm = args.algorithm
    seed = args.seed if args.seed is not None else config.generator.remote_diffusion_seed
    frame_size = config.generator.frame_size

    print("[streamdiffusion-demo] loading pipeline (this takes a while - model download/load)...")
    pipe, fixed_latent, device, dtype = load_pipeline(args.model_id, seed, frame_size)
    projector, embed_shape = _fit_projector(pipe, config.generator.anchor_prompts, config.optimizer.search_dims, device)
    render_server = LatentWalkRenderServer(
        pipe, projector, embed_shape, frame_size, config.optimizer.search_dims,
        fixed_latent, device, dtype, args.steps, args.guidance_scale,
    )
    generator = LatentWalkGeneratorAdapter(render_server)

    if args.mock:
        eeg_source = MockEEGSource(config.eeg.channels, config.eeg.sample_rate_hz)
    else:
        client_id, client_secret = emotiv_credentials()
        eeg_source = EmotivCortexSource(client_id, client_secret)
    eeg_source.connect()

    try:
        asyncio.run(run(config, eeg_source, generator))
    finally:
        eeg_source.close()


if __name__ == "__main__":
    main()
