#!/usr/bin/env python3
"""Compatibility entrypoint for the manifest-driven diffusion server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.server.diffusion import *  # noqa: F403
from src.server.diffusion.cli import main
from src.server.diffusion.http import ThreadingHTTPServer
from src.server.diffusion.renderer import (
    AnchorMorphRenderServer,
    encode_anchor_prompts,
    load_pipeline,
    make_anchor_latents,
    make_cpu_generator,
)


if __name__ == "__main__":
    main()
