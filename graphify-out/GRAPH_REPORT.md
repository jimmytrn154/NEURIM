# Graph Report - NEURIM  (2026-07-12)

## Corpus Check
- 126 files · ~48,363 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1206 nodes · 2365 edges · 97 communities (80 shown, 17 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 167 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f5a41685`
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
- Community 24
- Community 25
- Community 29
- orchestrator.py
- Community 35
- run_poodle_turbo_morph.py
- EEGFeatureExtractor
- Community 39
- record_reward_trials.py
- Community 41
- WebSocketOrchestrator
- Community 43
- NoiseAwareLatentTuRBO
- StimulusPresenter
- Community 46
- NEURIM
- fake_reward.py
- AGENTS.md
- EEG Preference-Reward Redesign
- LearnedPreferenceReward
- _FakeCV2
- test_stimulus_presenter.py
- Config
- WebSocketOrchestrator
- test_preference_closed_loop.py
- test_generator.py
- package.json
- RemoteDiffusionClient
- LocalOrchestrator
- ProceduralPseudo3D
- @radix-ui/react-tooltip
- react
- react-dom
- @react-three/drei
- tailwind-merge
- three
- ws
- PCAProjector
- OptimizerService

## God Nodes (most connected - your core abstractions)
1. `Config` - 42 edges
2. `FAARewardComputer` - 38 edges
3. `EmotivCortexSource` - 34 edges
4. `NoiseAwareLatentTuRBO` - 29 edges
5. `Observation` - 28 edges
6. `OptimizerService` - 24 edges
7. `EEGFeatureExtractor` - 22 edges
8. `MockPreferenceEEGSource` - 21 edges
9. `cn()` - 20 edges
10. `RewardMessage` - 20 edges

## Surprising Connections (you probably didn't know these)
- `_Writer` --uses--> `Config`  [INFERRED]
  scripts/record_reward_trials.py → src/common/config.py
- `_Writer` --uses--> `EmotivCortexSource`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/eeg_sources.py
- `_Writer` --uses--> `EEGFeatureExtractor`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/learned_reward.py
- `_Writer` --uses--> `FeatureBaseline`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/learned_reward.py
- `_Writer` --uses--> `MockPreferenceEEGSource`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/mock_preference.py

## Import Cycles
- None detected.

## Communities (97 total, 17 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (41): EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, OptimizerConfig, PreprocessingConfig, PresentationConfig, Path (+33 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (7): GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, test_one_plus_one_es_adapts_sigma_on_success_streak()

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (45): main(), _cue_label(), main(), _no_reward_reason(), _pair_label(), _pair_value(), emotiv_credentials(), calibrate_baseline() (+37 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (39): Real EMOTIV/FAA reward adapter for the same Observation interface., RealFAAReward, _erf(), NoiseAwareLatentTuRBO, ndarray, Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (43): load_prompt_session_manifest(), Path, Helpers for manifest-backed anchor sessions used by the generalized server., _clean_string_list(), curate_prompt_manifest(), _extract_response_text(), format_manifest_summary(), _load_default_client() (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.18
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (23): manifest_metadata(), Any, build_parser(), create_renderer(), main(), ArgumentParser, CLI composition root for the manifest-driven diffusion server., DiffusionServer (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (11): BrainFlowLSLSource, EmotivCortexSource, Any, Return Cortex EEG column labels from a subscribe response., Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports     pyl, EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result() (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (13): PreprocessedSource, Preprocessor, Any, Streaming EEG signal conditioning: the stage that was missing entirely.  Raw EPO, Wrap an EEGSource so downstream consumers get conditioned samples.      Same (ti, Stateful causal conditioning for streaming multi-channel EEG., Reference + filter one raw sample. `quality` optionally maps channel ->, _alpha() (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (8): Namespace, _build_preference_reward(), build_reward_source(), LearnedEEGReward, Real EEG reward scored by a trained sklearn reward model., Pairwise-preference EEG reward (real headset or --mock-eeg)., build_preprocessor(), Construct a Preprocessor from a Config, or None if disabled.

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (41): BrainActivity3D(), channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), BackendSession, BrainActivity3D, EEGFeatures (+33 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (9): main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100-, Reconstruct (prompt_embeds, pooled_prompt_embeds) from a flat vector         pro, img2img pass against the previous frame, for smoother morphing         between o, A fixed-seed torch.Generator so nearby embeddings render to nearby         image, Straight text -> image, bypassing the embedding/projector path. (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (29): BaseModel, FastAPI, Popen, ProcessFactory, create_app(), FastAPI application factory for the local frontend bridge., main(), CLI for the local frontend API bridge. (+21 more)

### Community 15 - "Community 15"
Cohesion: 0.36
Nodes (8): backendError(), BackendSession, cleanUrl(), POST(), requestBoolean(), requestNumber(), requestString(), SessionIntentRequest

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (23): 1. Why the redesign, 2. Architecture, 3. Files added / changed, 4.0 Generate A/B candidate images (per target), 4.1 Record calibration trials (mock, offline), 4.2 Train + validate (the scientific gate), 4.3 Run the closed loop (mock EEG, headless-friendly reward core), 4.4 Tests (+15 more)

### Community 17 - "Community 17"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: retrieve context, Source Nodes

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (17): class-variance-authority, clsx, dependencies, class-variance-authority, clsx, lucide-react, next, openai (+9 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (21): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+13 more)

### Community 24 - "Community 24"
Cohesion: 0.07
Nodes (28): DiffusionClient, Any, ndarray, HTTP client for a private manifest-driven diffusion server., FrameStore, Path, Atomic live-frame and session snapshot storage., Local optimizer session clients and persistence. (+20 more)

### Community 29 - "Community 29"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 35 - "Community 35"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 37 - "run_poodle_turbo_morph.py"
Cohesion: 0.17
Nodes (28): device, dtype, add_hud(), blend_noise_latents(), blend_prompt_embeds(), cosine_ease(), encode_breed_prompts(), interpolate_z() (+20 more)

### Community 38 - "EEGFeatureExtractor"
Cohesion: 0.05
Nodes (48): augment_antisymmetric(), augment_jitter(), build_ensemble(), faa_feature_mask(), leave_session_out(), _load_one(), load_pairwise(), main() (+40 more)

### Community 40 - "record_reward_trials.py"
Cohesion: 0.14
Nodes (26): _as_feature_tensor(), _build_presenter(), build_trials(), capture_window(), _clip_embed(), embed_images(), fit_session_baseline(), _images_in() (+18 more)

### Community 42 - "WebSocketOrchestrator"
Cohesion: 0.15
Nodes (11): _encode_jpeg(), _encode_png(), GeneratorService, FrameMessage, Image, ndarray, The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB (+3 more)

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 44 - "NoiseAwareLatentTuRBO"
Cohesion: 0.60
Nodes (5): encode(), generate_for_target(), load_targets(), main(), Path

### Community 45 - "StimulusPresenter"
Cohesion: 0.07
Nodes (18): Exception, AbortPresentation, ndarray, Path, OpenCV stimulus presentation for real EEG calibration sessions.  The pairwise re, Morph A -> B by alpha blend (the calibration analogue of the live latent, Raised when the operator presses q/ESC to stop the session early., One repaint - call this inside the EEG capture loop so the stable         image (+10 more)

### Community 50 - "NEURIM"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 61 - "AGENTS.md"
Cohesion: 0.13
Nodes (16): Protocol, Wire format for the websocket messages passed between services.  Signal -> Orche, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, Wires Signal -> Optimizer -> Generator together and drives the timing.  Two flav, EEGSource, Fake reward sources with the exact same interface FAA reward has: a scalar in [-, Common interface: FAARewardComputer-backed or fake, doesn't matter. (+8 more)

### Community 79 - "_FakeCV2"
Cohesion: 0.40
Nodes (3): ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict.

### Community 80 - "test_stimulus_presenter.py"
Cohesion: 0.50
Nodes (4): ndarray, Fraction of samples in a cleaned window flagged as blink or EMG.          Blink:, Median/MAD z-score; robust to the very spikes we want to flag., _robust_z()

### Community 81 - "Config"
Cohesion: 0.23
Nodes (8): Config, Interpolator, -- Morphing Process (animation)     Linear interpolation between the last two ac, _build_algorithm(), OptimizerService, The Optimizer service: reward in, latent stream out. ~150 lines including the st, FrameMessage, Shared optimizer-to-diffusion rendering loop primitives.

### Community 84 - "WebSocketOrchestrator"
Cohesion: 0.21
Nodes (4): ControlMessage, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Hub server: Signal service clients push RewardMessages; display clients     rece, WebSocketOrchestrator

### Community 85 - "test_preference_closed_loop.py"
Cohesion: 0.15
Nodes (12): MockPreferenceEEGSource, Synthetic EEG carrying a controllable, decodable *preference* signal.  MockEEGSo, 14-channel synthetic EEG with a tunable-SNR preference signal.      SNR is gover, _make_reward(), _quiet(), Headless tests for the pairwise preference reward + optimizer core.  Exercises t, A near-target candidate must out-reward a far-from-target one (deterministic)., _run_loop() (+4 more)

### Community 86 - "test_generator.py"
Cohesion: 0.21
Nodes (11): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, ProceduralPseudo3D, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., _FakeHTTPResponse, _FakeHTTPSession, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z() (+3 more)

### Community 88 - "package.json"
Cohesion: 0.22
Nodes (8): name, private, scripts, build, dev, lint, start, version

### Community 89 - "RemoteDiffusionClient"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 91 - "LocalOrchestrator"
Cohesion: 0.22
Nodes (4): FrameMessage, Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, LocalOrchestrator, Real FAA needs 30s of rest to fit the baseline; fake reward sources         (key

### Community 93 - "ProceduralPseudo3D"
Cohesion: 0.33
Nodes (3): Image, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, TripoSRConverter

### Community 111 - "PCAProjector"
Cohesion: 0.15
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 116 - "OptimizerService"
Cohesion: 0.28
Nodes (4): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif

## Knowledge Gaps
- **122 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `SessionIntentRequest`, `BackendSession`, `metadata` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `run_poodle_turbo_morph.py`, `Community 6`, `record_reward_trials.py`, `Community 10`, `WebSocketOrchestrator`, `LearnedPreferenceReward`, `WebSocketOrchestrator`, `Community 24`, `LocalOrchestrator`, `AGENTS.md`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Why does `Interpolator` connect `Config` to `Community 5`, `WebSocketOrchestrator`, `Community 13`, `PCAProjector`, `WebSocketOrchestrator`, `test_generator.py`, `Community 24`, `RemoteDiffusionClient`, `LocalOrchestrator`, `AGENTS.md`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `EmotivCortexSource` connect `Community 7` to `Community 2`, `Community 3`, `run_poodle_turbo_morph.py`, `record_reward_trials.py`, `Community 10`, `LearnedPreferenceReward`, `Community 24`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Config` (e.g. with `_Writer` and `FakeFAAReward`) actually correct?**
  _`Config` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FAARewardComputer` (e.g. with `FakeFAAReward` and `LearnedEEGReward`) actually correct?**
  _`FAARewardComputer` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `EmotivCortexSource` (e.g. with `_Writer` and `FakeFAAReward`) actually correct?**
  _`EmotivCortexSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `NoiseAwareLatentTuRBO` (e.g. with `FakeFAAReward` and `LearnedEEGReward`) actually correct?**
  _`NoiseAwareLatentTuRBO` has 6 INFERRED edges - model-reasoned connections that need verification._