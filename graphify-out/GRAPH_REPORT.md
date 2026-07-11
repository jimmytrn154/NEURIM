# Graph Report - NEURIM  (2026-07-12)

## Corpus Check
- 147 files · ~51,671 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1301 nodes · 2652 edges · 112 communities (92 shown, 20 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 191 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bc214fd1`
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
- test_api_server.py
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
- Community 48
- Community 49
- NEURIM
- _FakeHTTPResponse
- fake_reward.py
- AGENTS.md
- EEG Preference-Reward Redesign
- LearnedPreferenceReward
- _FakeCV2
- test_stimulus_presenter.py
- Config
- neurim-app.tsx
- ProcessLogStore
- WebSocketOrchestrator
- test_preference_closed_loop.py
- test_generator.py
- MockPreferenceEEGSource
- package.json
- RemoteDiffusionClient
- mock_runner.py
- LocalOrchestrator
- FakeEEGManager
- ProceduralPseudo3D
- @radix-ui/react-tooltip
- react
- react-dom
- @react-three/drei
- tailwind-merge
- three
- ws
- .process_sample
- alert.tsx
- button.tsx
- tabs.tsx
- .__init__
- route.ts
- route.ts
- clsx
- @radix-ui/react-tabs
- @react-three/fiber
- PCAProjector

## God Nodes (most connected - your core abstractions)
1. `Config` - 49 edges
2. `FAARewardComputer` - 38 edges
3. `EmotivCortexSource` - 37 edges
4. `SessionManager` - 33 edges
5. `cn()` - 31 edges
6. `NoiseAwareLatentTuRBO` - 29 edges
7. `EEGConnectionManager` - 29 edges
8. `Observation` - 28 edges
9. `PromptCurationManifest` - 26 edges
10. `OptimizerService` - 24 edges

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

## Communities (112 total, 20 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (22): EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, OptimizerConfig, PreprocessingConfig, PresentationConfig, Path (+14 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (7): GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, test_one_plus_one_es_adapts_sigma_on_success_streak()

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (46): main(), Real EMOTIV/FAA reward adapter for the same Observation interface., RealFAAReward, _cue_label(), main(), _no_reward_reason(), _pair_label(), _pair_value() (+38 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (39): LearnedEEGReward, Real EEG reward scored by a trained sklearn reward model., _erf(), NoiseAwareLatentTuRBO, ndarray, Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (43): load_prompt_session_manifest(), Path, Helpers for manifest-backed anchor sessions used by the generalized server., _clean_string_list(), curate_prompt_manifest(), _extract_response_text(), format_manifest_summary(), _load_default_client() (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (23): manifest_metadata(), Any, build_parser(), create_renderer(), main(), ArgumentParser, CLI composition root for the manifest-driven diffusion server., DiffusionServer (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (11): BrainFlowLSLSource, EmotivCortexSource, Any, Return Cortex EEG column labels from a subscribe response., Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports     pyl, EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result() (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.19
Nodes (11): PreprocessedSource, Preprocessor, Streaming EEG signal conditioning: the stage that was missing entirely.  Raw EPO, Wrap an EEGSource so downstream consumers get conditioned samples.      Same (ti, Stateful causal conditioning for streaming multi-channel EEG., _alpha(), ndarray, _stream_window() (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (8): EEGConnectionManager, Any, Owns the real EEG source while the API process is alive., _utc_now(), FakeSource, test_connect_failure_records_retry_state(), test_retry_now_marks_retry_due(), test_successful_connect_runs_30_second_calibration()

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (11): ApproachMeter(), RewardReadout(), Badge(), BadgeProps, badgeVariants, Input, Progress(), Separator() (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (9): main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100-, Reconstruct (prompt_embeds, pooled_prompt_embeds) from a flat vector         pro, img2img pass against the previous frame, for smoother morphing         between o, A fixed-seed torch.Generator so nearby embeddings render to nearby         image, Straight text -> image, bypassing the embedding/projector path. (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.21
Nodes (6): BaseModel, Event, Any, SessionManager, _utc_now(), StartSessionRequest

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
Cohesion: 0.11
Nodes (19): clsx, dependencies, clsx, lucide-react, next, next-themes, openai, @radix-ui/react-slot (+11 more)

### Community 19 - "Community 19"
Cohesion: 0.29
Nodes (5): hanken, metadata, newsreader, plexMono, ThemeProvider()

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (21): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+13 more)

### Community 23 - "test_api_server.py"
Cohesion: 0.17
Nodes (14): _client(), FakeCurationService, FakeDiffusionClient, _manifest(), test_duplicate_start_returns_conflict(), test_eeg_status_and_retry(), test_health(), test_logs_clamps_line_count() (+6 more)

### Community 24 - "Community 24"
Cohesion: 0.08
Nodes (24): DiffusionClient, Any, ndarray, HTTP client for a private manifest-driven diffusion server., FrameStore, Live-frame and session snapshot storage., Local optimizer session clients and persistence., OptimizerRenderLoop (+16 more)

### Community 29 - "Community 29"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 35 - "Community 35"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 37 - "run_poodle_turbo_morph.py"
Cohesion: 0.10
Nodes (41): device, dtype, Namespace, encode(), generate_for_target(), load_targets(), main(), Path (+33 more)

### Community 38 - "EEGFeatureExtractor"
Cohesion: 0.05
Nodes (48): augment_antisymmetric(), augment_jitter(), build_ensemble(), faa_feature_mask(), leave_session_out(), _load_one(), load_pairwise(), main() (+40 more)

### Community 40 - "record_reward_trials.py"
Cohesion: 0.14
Nodes (26): _as_feature_tensor(), _build_presenter(), build_trials(), capture_window(), _clip_embed(), embed_images(), fit_session_baseline(), _images_in() (+18 more)

### Community 42 - "WebSocketOrchestrator"
Cohesion: 0.25
Nodes (6): _encode_jpeg(), _encode_png(), FrameMessage, Image, ndarray, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 44 - "NoiseAwareLatentTuRBO"
Cohesion: 0.19
Nodes (10): datetime, FastAPI, FastAPI application factory for the local frontend bridge., _iso(), API-owned EMOTIV connection and calibration lifecycle., Local frontend API bridge., Optimizer session lifecycle management., Request models for the local frontend API. (+2 more)

### Community 45 - "StimulusPresenter"
Cohesion: 0.07
Nodes (18): Exception, AbortPresentation, ndarray, Path, OpenCV stimulus presentation for real EEG calibration sessions.  The pairwise re, Morph A -> B by alpha blend (the calibration analogue of the live latent, Raised when the operator presses q/ESC to stop the session early., One repaint - call this inside the EEG capture loop so the stable         image (+10 more)

### Community 48 - "Community 48"
Cohesion: 0.18
Nodes (12): BrainActivity3D(), channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), PhaseChips(), BrainActivity3D, BackendSession (+4 more)

### Community 49 - "Community 49"
Cohesion: 0.23
Nodes (13): FrameStream, useFrameStream(), NeurimSession, SessionPhase, epocPositions, makeMockImageSrc(), makeMockSession(), MockSession (+5 more)

### Community 50 - "NEURIM"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 51 - "_FakeHTTPResponse"
Cohesion: 0.22
Nodes (7): MomentumHillClimb, ndarray, The candidate z to show next; doesn't mutate state until update()., Tell the optimizer what happened after showing `candidate` for a         full wi, test_hill_climb_accepts_clear_improvement(), test_hill_climb_rejects_within_noise_band(), test_hill_climb_reverses_on_clear_drop()

### Community 61 - "AGENTS.md"
Cohesion: 0.10
Nodes (15): Protocol, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, EEGSource, KeyboardRewardSource, Fake reward sources with the exact same interface FAA reward has: a scalar in [-, Common interface: FAARewardComputer-backed or fake, doesn't matter., Up/down arrow keys nudge reward; it decays toward 0 between presses.      Uses ` (+7 more)

### Community 63 - "EEG Preference-Reward Redesign"
Cohesion: 0.29
Nodes (12): _sm(), test_explore_moves_to_refine_on_climbing_trend(), test_min_steps_before_settle_prevents_immediate_lock(), test_recover_after_negative_streak(), test_recover_ignores_small_negatives_above_margin(), test_recover_returns_to_explore_next_step(), test_settle_after_sustained_high_reward_low_motion(), test_settle_fires_at_modest_plateau_below_old_threshold() (+4 more)

### Community 78 - "LearnedPreferenceReward"
Cohesion: 0.24
Nodes (7): HeroCandidate(), ProcessingState(), PromptBubble(), SignalRail(), SteerInput(), TopBar(), stateBadge()

### Community 79 - "_FakeCV2"
Cohesion: 0.40
Nodes (3): ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict.

### Community 80 - "test_stimulus_presenter.py"
Cohesion: 0.50
Nodes (4): ndarray, Fraction of samples in a cleaned window flagged as blink or EMG.          Blink:, Median/MAD z-score; robust to the very spikes we want to flag., _robust_z()

### Community 81 - "Config"
Cohesion: 0.10
Nodes (19): Config, FrameMessage, LatentMessage, Wire format for the websocket messages passed between services.  Signal -> Orche, Optimizer service -> Orchestrator. Next point in the low-dim search space., Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, GeneratorService, Interpolator (+11 more)

### Community 82 - "neurim-app.tsx"
Cohesion: 0.25
Nodes (6): Landing(), NeurimApp(), SessionView(), ThemeToggle(), useSession(), examplePrompts

### Community 83 - "ProcessLogStore"
Cohesion: 0.20
Nodes (4): DiffusionClientFactory, ProcessLogStore, Path, _slug()

### Community 84 - "WebSocketOrchestrator"
Cohesion: 0.21
Nodes (4): ControlMessage, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Hub server: Signal service clients push RewardMessages; display clients     rece, WebSocketOrchestrator

### Community 85 - "test_preference_closed_loop.py"
Cohesion: 0.33
Nodes (9): _make_reward(), _quiet(), Headless tests for the pairwise preference reward + optimizer core.  Exercises t, A near-target candidate must out-reward a far-from-target one (deterministic)., _run_loop(), _softmax(), _target_pref(), test_reward_gradient_points_toward_target() (+1 more)

### Community 86 - "test_generator.py"
Cohesion: 0.33
Nodes (8): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, mirrored_quadrants(), Compose 4 copies of `image`, each facing outward from center, for a     tabletop, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z(), test_procedural_renderer_produces_image_of_requested_size(), test_pseudo_3d_squashes_at_90_degrees()

### Community 87 - "MockPreferenceEEGSource"
Cohesion: 0.22
Nodes (3): MockPreferenceEEGSource, Synthetic EEG carrying a controllable, decodable *preference* signal.  MockEEGSo, 14-channel synthetic EEG with a tunable-SNR preference signal.      SNR is gover

### Community 88 - "package.json"
Cohesion: 0.22
Nodes (8): name, private, scripts, build, dev, lint, start, version

### Community 89 - "RemoteDiffusionClient"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 90 - "mock_runner.py"
Cohesion: 0.43
Nodes (5): build_parser(), main(), MockOptimizerRunner, ArgumentParser, Scripted-reward optimizer session.

### Community 91 - "LocalOrchestrator"
Cohesion: 0.48
Nodes (4): create_app(), main(), CLI for the local frontend API bridge., ApiSettings

### Community 93 - "ProceduralPseudo3D"
Cohesion: 0.22
Nodes (6): ProceduralPseudo3D, Image, Image -> pseudo-3D pyramid quadrants.  Real-time text-to-3D (TripoSR) is the par, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., TripoSRConverter

### Community 95 - "react"
Cohesion: 0.29
Nodes (3): _FakeHTTPResponse, _FakeHTTPSession, test_remote_diffusion_sends_optimizer_state_and_caches_step()

### Community 96 - "react-dom"
Cohesion: 0.33
Nodes (5): Card, CardContent, CardDescription, CardHeader, CardTitle

### Community 102 - "alert.tsx"
Cohesion: 0.50
Nodes (3): Alert, AlertDescription, AlertTitle

### Community 103 - "button.tsx"
Cohesion: 0.50
Nodes (3): Button, ButtonProps, buttonVariants

### Community 104 - "tabs.tsx"
Cohesion: 0.50
Nodes (3): TabsContent, TabsList, TabsTrigger

### Community 111 - "PCAProjector"
Cohesion: 0.14
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

## Knowledge Gaps
- **126 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `SessionIntentRequest`, `BackendSession`, `hanken` (+121 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `run_poodle_turbo_morph.py`, `Community 5`, `Community 6`, `record_reward_trials.py`, `.__init__`, `Community 10`, `NoiseAwareLatentTuRBO`, `Community 14`, `ProcessLogStore`, `WebSocketOrchestrator`, `Community 24`, `mock_runner.py`, `AGENTS.md`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Why does `FAARewardComputer` connect `Community 2` to `AGENTS.md`, `Community 24`, `Community 3`, `run_poodle_turbo_morph.py`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `EmotivCortexSource` connect `Community 7` to `Community 2`, `Community 3`, `run_poodle_turbo_morph.py`, `record_reward_trials.py`, `Community 10`, `NoiseAwareLatentTuRBO`, `Community 24`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `Config` (e.g. with `_Writer` and `FakeFAAReward`) actually correct?**
  _`Config` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FAARewardComputer` (e.g. with `FakeFAAReward` and `LearnedEEGReward`) actually correct?**
  _`FAARewardComputer` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `EmotivCortexSource` (e.g. with `_Writer` and `FakeFAAReward`) actually correct?**
  _`EmotivCortexSource` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `SessionManager` (e.g. with `Config` and `PromptCurationManifest`) actually correct?**
  _`SessionManager` has 12 INFERRED edges - model-reasoned connections that need verification._