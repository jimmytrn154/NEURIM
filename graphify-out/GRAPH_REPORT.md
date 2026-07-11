# Graph Report - NEURIM  (2026-07-11)

## Corpus Check
- 86 files · ~38,883 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 901 nodes · 1661 edges · 81 communities (72 shown, 9 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `96fb022e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 29
- orchestrator.py
- Community 35
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- WebSocketOrchestrator
- Community 43
- Community 46
- NEURIM
- fake_reward.py
- AGENTS.md
- run_stablediffusion.py
- test_generator.py
- RemoteDiffusionClient
- ProceduralPseudo3D
- PCAProjector
- OptimizerService

## God Nodes (most connected - your core abstractions)
1. `Config` - 40 edges
2. `FAARewardComputer` - 33 edges
3. `EmotivCortexSource` - 28 edges
4. `OptimizerService` - 26 edges
5. `RewardMessage` - 23 edges
6. `Interpolator` - 23 edges
7. `GeneratorService` - 22 edges
8. `NoiseAwareLatentTuRBO` - 22 edges
9. `WebSocketOrchestrator` - 22 edges
10. `DiffusionGenerator` - 21 edges

## Surprising Connections (you probably didn't know these)
- `_SessionSnapshot` --uses--> `WebSocketOrchestrator`  [INFERRED]
  scripts/run_demo.py → src/orchestrator/orchestrator.py
- `_SessionSnapshot` --uses--> `EmotivCortexSource`  [INFERRED]
  scripts/run_demo.py → src/signal_service/eeg_sources.py
- `_SessionSnapshot` --uses--> `MockEEGSource`  [INFERRED]
  scripts/run_demo.py → src/signal_service/eeg_sources.py
- `DiffusionRenderServer` --uses--> `Config`  [INFERRED]
  scripts/run_diffusion_server.py → src/common/config.py
- `DiffusionRenderServer` --uses--> `PCAProjector`  [INFERRED]
  scripts/run_diffusion_server.py → src/optimizer/projection.py

## Import Cycles
- None detected.

## Communities (81 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.20
Nodes (11): main(), Saves the session's first and final frame to data/processed/, for the     OFFLIN, run_local(), run_served(), _save_frame(), _SessionSnapshot, Config, GeneratorService (+3 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (9): GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, _build_algorithm(), The Optimizer service: reward in, latent stream out. ~150 lines including the st (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (53): main(), _cue(), main(), _post_anchors(), _post_render(), ndarray, Same start/end capture as run_demo.py's _SessionSnapshot, for the     optional o, _save_frame() (+45 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (13): PresentationSchedule, Presentation schedule and scoring gate.  The reward-delay / credit-assignment fi, Feed a reward reading at `elapsed` seconds since this candidate was         pres, Interpolation fraction for the morph: reaches 1.0 by the end of the         tran, Return a list of human-readable warnings (empty if the schedule is sound)., Collects only the scoring-interval reward readings for one candidate,     then e, Start a fresh candidate (call when a new latent is presented)., ScoringGate (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (40): main(), load_prompt_session_manifest(), Path, Helpers for manifest-backed anchor sessions used by the generalized server., _clean_string_list(), curate_prompt_manifest(), _extract_response_text(), format_manifest_summary() (+32 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (15): AnchorMorphRenderServer, blend_noise_latents(), blend_prompt_embeds(), encode_anchor_prompts(), load_pipeline(), main(), make_anchor_latents(), make_cpu_generator() (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (11): BrainFlowLSLSource, EmotivCortexSource, Any, Return Cortex EEG column labels from a subscribe response., Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports     pyl, EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result() (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (12): BaseModel, Popen, _default_port(), main(), Any, session_logs(), session_status(), SessionManager (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (40): EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, OptimizerConfig, PresentationConfig, Path, Loads config/config.yaml into plain dataclasses.  Secrets (EMOTIV_CLIENT_ID/SECR (+32 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (41): BrainActivity3D(), channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), BackendSession, BrainActivity3D, EEGFeatures (+33 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (13): DiffusionRenderServer, _fit_projector(), main(), make_handler(), main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100- (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (10): _client(), FakeProcess, test_duplicate_start_returns_conflict(), test_health(), test_logs_clamps_line_count(), test_rejects_invalid_server_url(), test_start_mock_session_builds_command(), test_start_real_session_omits_mock() (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.36
Nodes (8): backendError(), BackendSession, cleanUrl(), POST(), requestBoolean(), requestNumber(), requestString(), SessionIntentRequest

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (19): Architecture: four services, Build order (non-negotiable), Current pipeline, File roles in the generalized path, Generalized manifest-driven flow, Generation, Layout, Legacy dog-specific path (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: retrieve context, Source Nodes

### Community 18 - "Community 18"
Cohesion: 0.06
Nodes (31): class-variance-authority, clsx, dependencies, class-variance-authority, clsx, lucide-react, next, openai (+23 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.07
Nodes (29): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+21 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (17): Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, effective_sample_size(), Observation, ndarray, Observation model for the optimizer: a presentation window is NOT reduced to a s, One presentation window's reward estimate and its uncertainty., ESS accounting for autocorrelation of overlapping-window FAA samples.      Uses, Turn a window of FAA reward samples into an Observation.      artifact_fraction (+9 more)

### Community 24 - "Community 24"
Cohesion: 0.18
Nodes (11): BreedTargetRewardSource, main(), _post_anchors(), _post_render(), ndarray, Scripted reward for the breed-weight server: reward a selected breed., _softmax_weights(), Interpolator (+3 more)

### Community 29 - "Community 29"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 35 - "Community 35"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 37 - "Community 37"
Cohesion: 0.23
Nodes (6): _erf(), NoiseAwareLatentTuRBO, ndarray, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc, math.erf on a scalar (avoids importing scipy just for the normal CDF).

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (6): ControlMessage, FrameMessage, Wire format for the websocket messages passed between services.  Signal -> Orche, Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Wires Signal -> Optimizer -> Generator together and drives the timing.  Two flav

### Community 42 - "WebSocketOrchestrator"
Cohesion: 0.25
Nodes (6): _encode_jpeg(), _encode_png(), FrameMessage, Image, ndarray, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 50 - "NEURIM"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 61 - "AGENTS.md"
Cohesion: 0.11
Nodes (16): Protocol, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, EEGSource, KeyboardRewardSource, Fake reward sources with the exact same interface FAA reward has: a scalar in [-, Common interface: FAARewardComputer-backed or fake, doesn't matter., Up/down arrow keys nudge reward; it decays toward 0 between presses.      Uses ` (+8 more)

### Community 63 - "run_stablediffusion.py"
Cohesion: 0.20
Nodes (15): blend_noise_latents(), blend_prompt_embeds(), BreedMorphRenderServer, encode_breed_prompts(), load_pipeline(), main(), make_breed_latents(), make_handler() (+7 more)

### Community 86 - "test_generator.py"
Cohesion: 0.31
Nodes (8): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, _FakeHTTPSession, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z(), test_procedural_renderer_produces_image_of_requested_size(), test_pseudo_3d_squashes_at_90_degrees(), test_remote_diffusion_sends_optimizer_state_and_caches_step()

### Community 89 - "RemoteDiffusionClient"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 93 - "ProceduralPseudo3D"
Cohesion: 0.18
Nodes (9): The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, mirrored_quadrants(), ProceduralPseudo3D, Image, Image -> pseudo-3D pyramid quadrants.  Real-time text-to-3D (TripoSR) is the par, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., Compose 4 copies of `image`, each facing outward from center, for a     tabletop (+1 more)

### Community 111 - "PCAProjector"
Cohesion: 0.13
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 116 - "OptimizerService"
Cohesion: 0.17
Nodes (8): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., OptimizerService, ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict., Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif

## Knowledge Gaps
- **118 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `SessionIntentRequest`, `BackendSession`, `metadata` (+113 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 38`, `Community 40`, `Community 10`, `Community 13`, `AGENTS.md`, `OptimizerService`, `Community 24`, `ProceduralPseudo3D`, `run_stablediffusion.py`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Why does `OptimizerService` connect `OptimizerService` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 37`, `Community 38`, `Community 10`, `Community 24`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `FAARewardComputer` connect `Community 2` to `AGENTS.md`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `Config` (e.g. with `_SessionSnapshot` and `DiffusionRenderServer`) actually correct?**
  _`Config` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FAARewardComputer` (e.g. with `FAARewardSource` and `SignalService`) actually correct?**
  _`FAARewardComputer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `EmotivCortexSource` (e.g. with `_SessionSnapshot` and `_SessionSnapshot`) actually correct?**
  _`EmotivCortexSource` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `OptimizerService` (e.g. with `Config` and `LatentMessage`) actually correct?**
  _`OptimizerService` has 7 INFERRED edges - model-reasoned connections that need verification._