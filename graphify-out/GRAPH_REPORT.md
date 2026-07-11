# Graph Report - NEURIM  (2026-07-11)

## Corpus Check
- 86 files · ~37,667 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 863 nodes · 1641 edges · 54 communities (47 shown, 7 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6f5eb276`
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
- Community 9
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
- Community 26
- Community 27
- Community 28
- Community 29
- Interpolator
- Community 31
- Community 32
- Config
- orchestrator.py
- Community 35
- Community 37
- Community 38
- Community 39
- Community 40
- WebSocketOrchestrator
- NEURIM
- _FakeHTTPResponse
- fake_reward.py
- AGENTS.md

## God Nodes (most connected - your core abstractions)
1. `Config` - 41 edges
2. `FAARewardComputer` - 33 edges
3. `EmotivCortexSource` - 28 edges
4. `OptimizerService` - 26 edges
5. `RewardMessage` - 24 edges
6. `Interpolator` - 24 edges
7. `GeneratorService` - 22 edges
8. `WebSocketOrchestrator` - 22 edges
9. `DiffusionGenerator` - 21 edges
10. `NoiseAwareLatentTuRBO` - 21 edges

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

## Communities (54 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (30): band_power(), FAARewardComputer, PairFAAMetrics, Any, ndarray, Frontal alpha asymmetry: the entire "reward" signal.  FAA = ln(alpha_power(right, Mean/std of raw FAA collected during the rest period, for z-scoring., Sliding-window weighted FAA -> baseline z-score -> clip to [-1, 1] = r(t). (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (36): _erf(), NoiseAwareLatentTuRBO, ndarray, Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc, math.erf on a scalar (avoids importing scipy just for the normal CDF)., effective_sample_size() (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (12): Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, MomentumHillClimb, ndarray, The dumbest thing that works: momentum hill-climbing on a noisy scalar.  Propose, The candidate z to show next; doesn't mutate state until update()., Tell the optimizer what happened after showing `candidate` for a         full wi, _build_algorithm(), The Optimizer service: reward in, latent stream out. ~150 lines including the st (+4 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (41): BrainActivity3D(), channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), BackendSession, BrainActivity3D, EEGFeatures (+33 more)

### Community 4 - "Community 4"
Cohesion: 0.16
Nodes (12): BaseModel, Popen, _default_port(), main(), Any, session_logs(), session_status(), SessionManager (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (36): main(), load_prompt_session_manifest(), manifest_metadata(), Any, Path, Helpers for manifest-backed anchor sessions used by the generalized server., _clean_string_list(), curate_prompt_manifest() (+28 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (31): class-variance-authority, clsx, dependencies, class-variance-authority, clsx, lucide-react, next, openai (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (29): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (13): DiffusionRenderServer, _fit_projector(), main(), make_handler(), main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100- (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (10): _client(), FakeProcess, test_duplicate_start_returns_conflict(), test_health(), test_logs_clamps_line_count(), test_rejects_invalid_server_url(), test_start_mock_session_builds_command(), test_start_real_session_omits_mock() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (13): Protocol, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, EEGSource, Common interface: FAARewardComputer-backed or fake, doesn't matter., RewardSource, build_faa_service(), FAARewardSource (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (43): main(), _cue(), main(), _post_anchors(), _post_render(), ndarray, Same start/end capture as run_demo.py's _SessionSnapshot, for the     optional o, _save_frame() (+35 more)

### Community 15 - "Community 15"
Cohesion: 0.25
Nodes (10): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, mirrored_quadrants(), Image -> pseudo-3D pyramid quadrants.  Real-time text-to-3D (TripoSR) is the par, Compose 4 copies of `image`, each facing outward from center, for a     tabletop, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z() (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.20
Nodes (15): blend_noise_latents(), blend_prompt_embeds(), BreedMorphRenderServer, encode_breed_prompts(), load_pipeline(), main(), make_breed_latents(), make_handler() (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (6): GPBanditOptimizer, OnePlusOneES, ndarray, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, test_one_plus_one_es_adapts_sigma_on_success_streak()

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (17): OptimizerConfig, StateMachineConfig, CALIBRATE -> EXPLORE -> REFINE -> SETTLE, with a RECOVER escape hatch.  CALIBRAT, test_optimizer_converges_toward_hidden_target(), test_observe_observation_takes_one_step(), _sm(), test_explore_moves_to_refine_on_climbing_trend(), test_min_steps_before_settle_prevents_immediate_lock() (+9 more)

### Community 21 - "Community 21"
Cohesion: 0.23
Nodes (4): Interpolate step size: wide in EXPLORE, shrinking in REFINE as the         rewar, Call once per optimizer step with the accepted/estimated reward and         the, StateMachine, State

### Community 22 - "Community 22"
Cohesion: 0.16
Nodes (9): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., OptimizerService, ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict., Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (13): AnchorMorphRenderServer, blend_noise_latents(), blend_prompt_embeds(), encode_anchor_prompts(), load_pipeline(), main(), make_anchor_latents(), make_cpu_generator() (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 25 - "Community 25"
Cohesion: 0.33
Nodes (3): Image, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, TripoSRConverter

### Community 26 - "Community 26"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: retrieve context, Source Nodes

### Community 27 - "Community 27"
Cohesion: 0.25
Nodes (6): _encode_jpeg(), _encode_png(), FrameMessage, Image, ndarray, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB

### Community 28 - "Community 28"
Cohesion: 0.36
Nodes (8): backendError(), BackendSession, cleanUrl(), POST(), requestBoolean(), requestNumber(), requestString(), SessionIntentRequest

### Community 29 - "Community 29"
Cohesion: 0.16
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 30 - "Interpolator"
Cohesion: 0.15
Nodes (14): BreedTargetRewardSource, main(), _post_anchors(), _post_render(), ndarray, Mirror the real EEG optimizer's live/start/end frame capture behavior., Scripted reward for the breed-weight server: reward a selected breed., _save_frame() (+6 more)

### Community 31 - "Community 31"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 32 - "Community 32"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 33 - "Config"
Cohesion: 0.20
Nodes (11): main(), Saves the session's first and final frame to data/processed/, for the     OFFLIN, run_local(), run_served(), _save_frame(), _SessionSnapshot, Config, GeneratorService (+3 more)

### Community 34 - "orchestrator.py"
Cohesion: 0.22
Nodes (6): ControlMessage, FrameMessage, Wire format for the websocket messages passed between services.  Signal -> Orche, Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Wires Signal -> Optimizer -> Generator together and drives the timing.  Two flav

### Community 50 - "NEURIM"
Cohesion: 0.14
Nodes (13): Architecture: four services, Build order (non-negotiable), Generation, Layout, Milestone: real-time morph via StreamDiffusion (SD-Turbo), NEURIM, Risk register, Setup (+5 more)

### Community 51 - "_FakeHTTPResponse"
Cohesion: 0.25
Nodes (5): ProceduralPseudo3D, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., _FakeHTTPResponse, _FakeHTTPSession, test_remote_diffusion_sends_optimizer_state_and_caches_step()

### Community 52 - "fake_reward.py"
Cohesion: 0.25
Nodes (3): KeyboardRewardSource, Fake reward sources with the exact same interface FAA reward has: a scalar in [-, Up/down arrow keys nudge reward; it decays toward 0 between presses.      Uses `

## Knowledge Gaps
- **113 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `SessionIntentRequest`, `BackendSession`, `metadata` (+108 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `Community 1`, `Community 2`, `orchestrator.py`, `Community 5`, `Community 9`, `WebSocketOrchestrator`, `Community 13`, `Community 14`, `Community 15`, `Community 16`, `Community 20`, `Community 21`, `Community 22`, `Community 23`, `Community 29`, `Interpolator`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `OptimizerService` connect `Community 22` to `Config`, `Community 2`, `Community 1`, `orchestrator.py`, `Community 14`, `Community 17`, `Community 20`, `Community 21`, `Interpolator`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `Interpolator` connect `Interpolator` to `Config`, `orchestrator.py`, `Community 9`, `Community 10`, `WebSocketOrchestrator`, `Community 14`, `Community 15`, `Community 19`, `Community 27`, `Community 29`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `Config` (e.g. with `_SessionSnapshot` and `DiffusionRenderServer`) actually correct?**
  _`Config` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FAARewardComputer` (e.g. with `FAARewardSource` and `SignalService`) actually correct?**
  _`FAARewardComputer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `EmotivCortexSource` (e.g. with `_SessionSnapshot` and `_SessionSnapshot`) actually correct?**
  _`EmotivCortexSource` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `OptimizerService` (e.g. with `Config` and `LatentMessage`) actually correct?**
  _`OptimizerService` has 7 INFERRED edges - model-reasoned connections that need verification._