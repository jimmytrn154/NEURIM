# NEURIM

An online, human-in-the-loop optimizer that searches a generative model's
latent space using a single scalar reward decoded from EEG.

Not image reconstruction, not mind-reading - a search loop where the
objective function is a number computed from the user's brain, and the thing
being searched is "which image." Everything else is plumbing around that.

## Architecture: four services

```
Signal ──r(t)──▶ Optimizer ──z──▶ Generator ──frame──▶ (display)
                      ▲                                    │
                      └──────────── Orchestrator ──────────┘
                           (state machine + timing)
```

| Service       |  Input | Output | Code |
|---|---|---|---|
| **Signal**  | raw 14-channel EEG | `r(t)`, a scalar in `[-1, 1]` | `src/signal_service/` |
| **Optimizer** | reward | a stream of latents `z` | `src/optimizer/` |
| **Generator** | latent `z` | rendered frames | `src/generator/` |
| **Orchestrator** | - | wires the above over websockets, drives timing | `src/orchestrator/` |

Nothing downstream of the Signal service knows or cares that `r(t)` came from
`RewardMessage` shape from a keyboard or a scripted target instead.

### The reward: frontal alpha asymmetry

Every `faa.update_interval_s` (~250ms), over a `faa.window_s` (3.0s) sliding
window: compute alpha-band (8-13Hz) power over the frontal mirror pairs
F7/F8, AF3/AF4, F3/F4, and FC5/FC6. Each pair contributes
`ln(P_right) - ln(P_left)`. The live composite is a fixed-weight mean:
F3/F4 = 1.00, F7/F8 = 0.75, AF3/AF4 = 0.50, and FC5/FC6 = 0.50.
At session start, 30s of rest establishes a per-subject baseline mean/std;
every reading is z-scored against it and clipped to `[-1, 1]`. That normalized
value is `r(t)` - implemented in `src/signal_service/faa.py`.

### The optimizer: hill-climbing on a noisy scalar

`f(z)` = the user's approval - unknown, noisy, one reading every 1-2s. Two
rules:

1. **Reduce the search space to 8-16 dims.** `src/optimizer/projection.py`
   projects a bank of CLIP prompt embeddings via PCA, or interpolates across
   a handful of anchor prompts - the optimizer never touches the raw latent.
2. **Use a noise-robust, sample-efficient optimizer.** The default,
   `src/optimizer/hill_climb.py`, is a momentum hill-climb: propose a step,
   average the reward over the whole window the candidate is shown, and only
   accept/reverse once the difference clears `optimizer.noise_threshold` -
   otherwise EEG jitter yanks the search around. `src/optimizer/evolution.py`
   has the upgrade path: a (1+1)-ES with the 1/5 success rule, and a GP-BO
   bandit whose posterior uncertainty gives backtracking for free.

### The state machine

`CALIBRATE → EXPLORE → REFINE → SETTLE`, with `RECOVER` as an escape hatch -
implemented in `src/optimizer/state_machine.py`:

- **EXPLORE**: large steps, wide search.
- **REFINE**: once the *recent average* reward climbs past
  `refine_entry_reward`, steps shrink toward `step_size_refine_min`. This is
  level-gated, not slope-gated - right as the search converges the reward
  trend naturally flattens out from noise, so gating on slope alone can
  strand the search in EXPLORE forever with an oversized step (see the
  tuning notes below).
- **SETTLE**: reward stays above `settle_reward_threshold` and motion drops
  below `settle_motion_threshold` for `settle_patience_steps` in a row → lock.
- **RECOVER**: reward negative for `recover_negative_streak` steps in a row →
  revert to the last high-reward checkpoint, widen the search, blacklist the
  region just left.

### Generation

`src/generator/procedural.py` is a CPU-only, GPU-free fallback: it maps `z`
deterministically onto a star/polygon's hue, spike count, rotation and scale,
so a human watching the morph can confirm the search is converging on
*something* - this is what the fake-reward loop uses, and what production
falls back to if real-time text-to-3D blows the latency budget.
`src/generator/openai_image.py` wraps the OpenAI Image API. Set
`OPENAI_API_KEY` in the environment or `.env`, then use
`scripts/run_demo.py --backend openai`.
`src/generator/diffusion_pipeline.py` wraps SDXL-Turbo/LCM via `diffusers`
(lazy-imported; needs a CUDA GPU - see `requirements-diffusion.txt`).
`src/generator/remote_diffusion.py` sends post-FAA optimizer state to an
external diffusion server and uses the returned PNG as the live frame.
`src/generator/to_3d.py` wraps TripoSR for image-to-3D, with a
`ProceduralPseudo3D` fallback (rotate the 2D sprite to fake a viewing angle),
and composes four mirrored quadrants for a tabletop hologram pyramid.

## Build order (non-negotiable)

1. **Fake-reward loop.** `scripts/run_fake_loop.py --mode scripted` drives
   the whole Signal→Optimizer→Generator→Orchestrator loop with a hidden
   target latent standing in for "what the person wants" - no EEG, no GPU.
   If the loop doesn't converge here, no EEG signal will save it.
   `--mode keyboard` is the interactive version (up/down arrows), for
   watching a person confirm convergence by eye.
2. **Real FAA.** `scripts/run_calibration.py` fits the per-subject baseline;
   `scripts/run_demo.py` (no `--mock`) swaps in `EmotivCortexSource` or
   `BrainFlowLSLSource`.
3. **Generation quality.** `scripts/run_demo.py --backend openai`,
   `scripts/run_demo.py --backend diffusion`, or
   `scripts/run_demo.py --backend remote_diffusion --remote-url http://GPU_HOST:8766`.
4. **The pyramid and 3D.** `src/generator/to_3d.py`.
5. **Polish.**

## Tuning notes: hill-climb convergence is dimension-sensitive

`tests/test_optimizer.py` proves the loop makes real progress toward a
hidden target. In synthetic worst-case testing (a hidden target drawn
uniformly at random across *all* 8 dimensions simultaneously, no informative
structure to exploit), plain momentum hill-climb settles cleanly something
like a third to a half of the time within a 100-step budget, and gets
meaningfully closer (40-65% reduction in distance) most of the rest of the
time. At 12+ dims the settle rate drops further - random-direction search
degrades with dimensionality, which is exactly why the spec frames hill-climb
as "the dumbest thing that works" and recommends (1+1)-ES or GP-BO as
upgrades (`optimizer.algorithm: "es_1p1" | "gp_bo"` in `config.yaml`).

A real person's approach/avoidance signal is presumably far less adversarial
than a uniformly-random hidden target - there's usually a broad, informative
gradient toward "better," not an exact point to hit - so this is a
conservative benchmark, not an expected demo outcome. Even so: **keep the
manual confirm as a backstop**, per the risk register below. `search_dims`
defaults to 8 (the favorable end of the spec's 8-16 range) for this reason.

## Milestone: real-time morph via StreamDiffusion (SD-Turbo)

`scripts/run_diffusion_server.py` (raw `diffusers`, SDXL-Turbo) only reaches
~5 real renders/sec, so the client (`src/generator/remote_diffusion.py`)
crossfades between keyframes to fake `target_fps`. `scripts/run_streamdiffusion_server.py`
is an alternative backend built on
[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), which is
designed for genuinely continuous generation (a rolling image buffer across
calls, ~30-100fps on an RTX 4090) - the crossfade becomes unnecessary once
render itself is fast enough. Same `POST /render` wire contract, so the client
and `GeneratorService` need no changes - only `remote_diffusion_url` and
`remote_diffusion_keyframe_interval_s` (config.yaml) change.

Two real constraints, not implementation details to paper over:
- StreamDiffusion's public API only supports SD1.x-family models (no
  `pooled_prompt_embeds`/`add_text_embeds`) - it cannot load
  `stabilityai/sdxl-turbo`. This backend uses `stabilityai/sd-turbo` instead,
  a different checkpoint with a different visual style than the SDXL path.
- There is no official API to feed a continuous embedding instead of a text
  prompt. The server encodes each `anchor_prompts` entry itself (mirroring
  what `update_prompt()` does internally) to fit a `PCAProjector`, then on
  every request overwrites the wrapper's internal `stream.prompt_embeds`
  tensor directly with `projector.to_embedding(z)`. This is an undocumented
  attribute, not a supported integration point, and could break on a future
  StreamDiffusion release.

### `t_index_list` tuning (found empirically - see `scripts/test_streamdiffusion.py`)

`t_index_list` selects indices from the scheduler's noise-timestep schedule;
each index is one denoising step the UNet actually runs. Index 0 = maximum
noise (the starting point for generating from scratch); higher indices = less
noise (the right range for lightly restyling an *existing* image). One value
has to serve two different jobs here, and getting it wrong looks like two
different bugs:

- **`[32,45]`** (StreamDiffusion's documented img2img tutorial value) applied
  to the from-scratch bootstrap frame: garbage. Telling the model "this
  random noise tensor is already 32-45 steps denoised" when it isn't produces
  a colored-mosaic non-image, not a puppy.
- **`[0]`** alone: fixed content (real, coherent puppies matching the anchor
  prompts) but broke continuity between frames. At index 0 the img2img step
  treats the *previous frame* as pure noise too, so each call effectively
  regenerates from scratch - every frame is a different, unrelated puppy
  (different pose/background), just biased toward the right fur color.
- **`[0,16]`** (current default): both fixed. Same pose/composition held
  across an entire z-sweep while the conditioned attribute (fur color) drifts
  smoothly frame to frame - a real morph, not a slideshow. `[0,32]` also
  works about as well; `[0,16]` was picked for slightly more expressive color
  range in side-by-side comparison, not a large margin.

If you change the anchor prompts, the model, or the search dims, re-run
`scripts/test_streamdiffusion.py --streamdiffusion-repo <path>` (writes a
`t2img` sanity image plus a z-sweep to `data/processed/streamdiffusion_test/`)
before assuming `[0,16]` still holds - this was tuned empirically against one
specific anchor-prompt bank, not derived from a documented default.

## Risk register

1. **FAA is too noisy/slow to converge in a loud demo hall.** Mitigated by
   building the entire loop against a fake reward first (see Build order,
   step 1) and proving convergence before EEG ever touches it.
2. **FAA reflects lots of things besides "I like this"** - movement,
   arousal, room noise. Mitigated by: keep the subject still, baseline
   per-person (`scripts/run_calibration.py`), treat the signal as a
   probabilistic nudge, keep a manual confirm as a backstop.
3. **Integrating all four services on the last night.** Mitigated by the
   build order above - fake-signal end-to-end first, then swap components in
   one at a time.

## Layout

```
NEURIM/
├── config/config.yaml          # FAA, optimizer, state machine, generator config
├── requirements.txt             # base deps (no GPU needed)
├── requirements-diffusion.txt   # + torch/diffusers/TripoSR for local diffusion
├── src/
│   ├── common/                  # config loading, websocket message schemas
│   ├── signal_service/          # FAA reward, EEG sources, fake reward sources
│   ├── optimizer/                # projection, hill-climb/ES/GP-BO, state machine
│   ├── generator/                # procedural fallback, diffusion, OpenAI image API, image-to-3D
│   └── orchestrator/             # wires the above, local or over websockets
├── scripts/
│   ├── run_fake_loop.py          # build-order step 1 (see above)
│   ├── run_calibration.py        # per-subject FAA baseline
│   ├── run_diffusion_server.py    # remote diffusion HTTP server (SDXL-Turbo)
│   ├── run_streamdiffusion_server.py  # remote diffusion HTTP server (SD-Turbo, StreamDiffusion, real-time)
│   ├── run_streamdiffusion_demo.py    # full loop in ONE process on the GPU box (no HTTP split)
│   ├── test_streamdiffusion.py    # diagnostic: t_index_list tuning, see Milestone section above
│   └── run_demo.py               # the real thing
├── data/
│   ├── calibration/               # per-subject baselines (gitignored)
│   └── processed/                  # live frame output (gitignored)
└── tests/
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # base - no GPU needed
OPENAI_API_KEY=... python scripts/run_demo.py --backend openai
# pip install -r requirements-diffusion.txt   # only for --backend diffusion
python scripts/run_diffusion_server.py --host 0.0.0.0 --port 8766
python scripts/run_demo.py --backend remote_diffusion --remote-url http://GPU_HOST:8766

pytest tests/                             # unit tests + convergence proof
python scripts/run_fake_loop.py --mode scripted   # end-to-end, no hardware
```
