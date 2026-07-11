# Graph Report - NEURIM  (2026-07-11)

## Corpus Check
- 77 files · ~33,881 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 773 nodes · 1469 edges · 52 communities (44 shown, 8 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e4ea3b65`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- NoiseAwareLatentTuRBO
- config.py
- neurim-dashboard.tsx
- RewardMessage
- dependencies
- test_faa.py
- devDependencies
- compilerOptions
- DiffusionGenerator
- Any
- service.py
- EmotivCortexSource
- LatentMorpher
- service.py
- run_stablediffusion.py
- test_generator.py
- PCAProjector
- test_api_server.py
- components.json
- FAARewardComputer
- orchestrator.py
- NEURIM
- Config
- OptimizerService
- FAARewardSource
- run_real_eeg_optimizer.py
- OpenAIImageGenerator
- RemoteDiffusionClient
- MockEEGSource
- WebSocketOrchestrator
- test_faa_stream.py
- route.ts
- TripoSRConverter
- BrainFlowLSLSource
- route.ts
- .current_z
- .render
- layout.tsx
- AGENTS.md
- eslint.config.mjs
- next.config.ts
- next-env.d.ts
- postcss.config.mjs

## God Nodes (most connected - your core abstractions)
1. `Config` - 37 edges
2. `FAARewardComputer` - 33 edges
3. `EmotivCortexSource` - 28 edges
4. `OptimizerService` - 26 edges
5. `RewardMessage` - 23 edges
6. `Interpolator` - 23 edges
7. `GeneratorService` - 22 edges
8. `WebSocketOrchestrator` - 22 edges
9. `DiffusionGenerator` - 21 edges
10. `NoiseAwareLatentTuRBO` - 21 edges

## Surprising Connections (you probably didn't know these)
- `_SessionSnapshot` --uses--> `GeneratorService`  [INFERRED]
  scripts/run_demo.py → src/generator/service.py
- `_SessionSnapshot` --uses--> `LocalOrchestrator`  [INFERRED]
  scripts/run_demo.py → src/orchestrator/orchestrator.py
- `_SessionSnapshot` --uses--> `WebSocketOrchestrator`  [INFERRED]
  scripts/run_demo.py → src/orchestrator/orchestrator.py
- `_SessionSnapshot` --uses--> `EmotivCortexSource`  [INFERRED]
  scripts/run_demo.py → src/signal_service/eeg_sources.py
- `_SessionSnapshot` --uses--> `MockEEGSource`  [INFERRED]
  scripts/run_demo.py → src/signal_service/eeg_sources.py

## Import Cycles
- None detected.

## Communities (52 total, 8 thin omitted)

### Community 0 - "NoiseAwareLatentTuRBO"
Cohesion: 0.06
Nodes (36): _erf(), NoiseAwareLatentTuRBO, ndarray, Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc, math.erf on a scalar (avoids importing scipy just for the normal CDF)., effective_sample_size() (+28 more)

### Community 1 - "config.py"
Cohesion: 0.06
Nodes (40): Path, EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, OptimizerConfig, PresentationConfig, Loads config/config.yaml into plain dataclasses.  Secrets (EMOTIV_CLIENT_ID/SECR (+32 more)

### Community 2 - "neurim-dashboard.tsx"
Cohesion: 0.07
Nodes (36): channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), BrainActivity3D, EEGFeatures, FaaRewardBar(), formatMs() (+28 more)

### Community 3 - "RewardMessage"
Cohesion: 0.09
Nodes (21): BreedTargetRewardSource, main(), _post_anchors(), _post_render(), ndarray, Scripted reward for the breed-weight server: reward a selected breed., _softmax_weights(), Signal service -> Orchestrator. One scalar reward reading. (+13 more)

### Community 4 - "dependencies"
Cohesion: 0.06
Nodes (31): class-variance-authority, clsx, dependencies, class-variance-authority, clsx, lucide-react, next, openai (+23 more)

### Community 5 - "test_faa.py"
Cohesion: 0.10
Nodes (20): band_power(), PairFAAMetrics, ndarray, Mean/std of raw FAA collected during the rest period, for z-scoring., Pair-level alpha power and raw FAA values for diagnostics., Compact EEG visualization payload for the frontend.          Uses alpha-band pow, Baseline-normalized r(t), or None if the buffer isn't full yet., Welch PSD power in `band` (Hz) for a single-channel 1D signal. (+12 more)

### Community 6 - "devDependencies"
Cohesion: 0.07
Nodes (29): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+21 more)

### Community 7 - "compilerOptions"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 8 - "DiffusionGenerator"
Cohesion: 0.13
Nodes (13): DiffusionRenderServer, _fit_projector(), main(), make_handler(), main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100- (+5 more)

### Community 9 - "Any"
Cohesion: 0.19
Nodes (10): BaseModel, Popen, Any, session_logs(), session_status(), SessionManager, start_session(), StartSessionRequest (+2 more)

### Community 10 - "service.py"
Cohesion: 0.12
Nodes (9): GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, _build_algorithm(), The Optimizer service: reward in, latent stream out. ~150 lines including the st (+1 more)

### Community 11 - "EmotivCortexSource"
Cohesion: 0.17
Nodes (8): EmotivCortexSource, Return Cortex EEG column labels from a subscribe response., EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result(), test_emotiv_formats_unknown_api_error_with_raw_payload(), test_emotiv_formats_unpublished_app_error_with_owner_hint(), test_emotiv_stream_uses_cortex_cols_and_message_time()

### Community 12 - "LatentMorpher"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "service.py"
Cohesion: 0.15
Nodes (11): _encode_jpeg(), _encode_png(), GeneratorService, FrameMessage, Image, ndarray, The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB (+3 more)

### Community 14 - "run_stablediffusion.py"
Cohesion: 0.20
Nodes (15): blend_noise_latents(), blend_prompt_embeds(), BreedMorphRenderServer, encode_breed_prompts(), load_pipeline(), main(), make_breed_latents(), make_handler() (+7 more)

### Community 15 - "test_generator.py"
Cohesion: 0.21
Nodes (11): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, ProceduralPseudo3D, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., _FakeHTTPResponse, _FakeHTTPSession, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z() (+3 more)

### Community 16 - "PCAProjector"
Cohesion: 0.15
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 17 - "test_api_server.py"
Cohesion: 0.18
Nodes (10): _client(), FakeProcess, test_duplicate_start_returns_conflict(), test_health(), test_logs_clamps_line_count(), test_rejects_invalid_server_url(), test_start_mock_session_builds_command(), test_start_real_session_omits_mock() (+2 more)

### Community 18 - "components.json"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 19 - "FAARewardComputer"
Cohesion: 0.22
Nodes (9): main(), calibrate_baseline(), Per-session baseline calibration: 30s of rest before anything else runs., Consume samples from `sample_iter` for `duration_s`, fitting the baseline., FAARewardComputer, Any, Frontal alpha asymmetry: the entire "reward" signal.  FAA = ln(alpha_power(right, Sliding-window weighted FAA -> baseline z-score -> clip to [-1, 1] = r(t). (+1 more)

### Community 20 - "orchestrator.py"
Cohesion: 0.18
Nodes (7): ControlMessage, FrameMessage, Wire format for the websocket messages passed between services.  Signal -> Orche, Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., LocalOrchestrator, Wires Signal -> Optimizer -> Generator together and drives the timing.  Two flav

### Community 21 - "NEURIM"
Cohesion: 0.14
Nodes (13): Architecture: four services, Build order (non-negotiable), Generation, Layout, Milestone: real-time morph via StreamDiffusion (SD-Turbo), NEURIM, Risk register, Setup (+5 more)

### Community 22 - "Config"
Cohesion: 0.29
Nodes (9): main(), Saves the session's first and final frame to data/processed/, for the     OFFLIN, run_local(), run_served(), _save_frame(), _SessionSnapshot, Config, build_faa_service() (+1 more)

### Community 23 - "OptimizerService"
Cohesion: 0.21
Nodes (6): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., OptimizerService, Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif, FrameMessage

### Community 24 - "FAARewardSource"
Cohesion: 0.18
Nodes (6): Protocol, Real FAA needs 30s of rest to fit the baseline; fake reward sources         (key, EEGSource, Any, FAARewardSource, Wraps an EEGSource + FAARewardComputer behind the RewardSource interface.

### Community 25 - "run_real_eeg_optimizer.py"
Cohesion: 0.26
Nodes (9): _cue(), main(), _post_anchors(), _post_render(), ndarray, Same start/end capture as run_demo.py's _SessionSnapshot, for the     optional o, _save_frame(), _SessionSnapshot (+1 more)

### Community 26 - "OpenAIImageGenerator"
Cohesion: 0.18
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 27 - "RemoteDiffusionClient"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 28 - "MockEEGSource"
Cohesion: 0.23
Nodes (7): MockEEGSource, Synthetic 14-channel EEG for development without hardware.      Alpha-band power, _build_source(), Regression tests for FAARewardSource (KNOWN_ISSUES #1).  The bug: once the FAA w, test_raw_faa_varies_across_reads_after_warmup(), test_reward_is_not_constant_across_reads(), test_samples_per_read_is_floored_at_one()

### Community 30 - "test_faa_stream.py"
Cohesion: 0.36
Nodes (8): _cue_label(), main(), _no_reward_reason(), _pair_label(), _pair_value(), EEG data sources. All of them yield (timestamp, {channel_name: value}) samples., Wrap a sample iterator to yield in real time (for mock sources that     would ot, wall_clock_pace()

### Community 31 - "route.ts"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 32 - "TripoSRConverter"
Cohesion: 0.33
Nodes (3): Image, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, TripoSRConverter

### Community 34 - "route.ts"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 35 - ".current_z"
Cohesion: 0.40
Nodes (3): ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict.

### Community 36 - ".render"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

## Knowledge Gaps
- **93 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `metadata`, `$schema`, `style` (+88 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `NoiseAwareLatentTuRBO`, `config.py`, `RewardMessage`, `DiffusionGenerator`, `service.py`, `service.py`, `run_stablediffusion.py`, `FAARewardComputer`, `orchestrator.py`, `OptimizerService`, `FAARewardSource`, `run_real_eeg_optimizer.py`, `WebSocketOrchestrator`, `test_faa_stream.py`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `OptimizerService` connect `OptimizerService` to `NoiseAwareLatentTuRBO`, `config.py`, `RewardMessage`, `.current_z`, `service.py`, `orchestrator.py`, `Config`, `run_real_eeg_optimizer.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `FAARewardComputer` connect `FAARewardComputer` to `RewardMessage`, `test_faa.py`, `Config`, `FAARewardSource`, `MockEEGSource`, `test_faa_stream.py`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `Config` (e.g. with `_SessionSnapshot` and `DiffusionRenderServer`) actually correct?**
  _`Config` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FAARewardComputer` (e.g. with `FAARewardSource` and `SignalService`) actually correct?**
  _`FAARewardComputer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `EmotivCortexSource` (e.g. with `_SessionSnapshot` and `_SessionSnapshot`) actually correct?**
  _`EmotivCortexSource` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `OptimizerService` (e.g. with `Config` and `LatentMessage`) actually correct?**
  _`OptimizerService` has 7 INFERRED edges - model-reasoned connections that need verification._