# EEG Preference-Reward Redesign

This document describes the redesign of the NEURIM closed-loop reward signal, from
raw **Frontal Alpha Asymmetry (FAA)** to a subject-calibrated **pairwise
preference model**. It covers the motivation, the architecture, every file that
was added or changed, how to run each stage, and the validation results.

> TL;DR — Instead of reading an absolute "goodness" score off the scalp (which a
> 14-channel consumer headset cannot do reliably), we learn
> `P(image B is preferred over image A | EEG)` from a **within-trial A-vs-B
> contrast**. The contrast cancels the session/electrode drift that makes
> single-window FAA unusable. FAA becomes one feature among ~71.

---

## 1. Why the redesign

The old reward was: raw EEG → Welch band power → `ln(alpha_right) − ln(alpha_left)`
→ z-score against a 30 s rest baseline → clip to `[-1, 1]`
(`src/signal_service/faa.py`). It was too noisy and unstable for the optimizer to
converge, for two root reasons:

1. **Wrong target.** An absolute single-window score conflates the preference we
   want with novelty, image complexity, blink rate, and arousal. The tractable,
   drift-robust signal is a **pairwise contrast** `features(B) − features(A)`.
2. **No signal conditioning.** The original pipeline fed *raw* EEG straight into
   band power — no bandpass, notch, re-reference, or blink/EMG rejection anywhere.
   Frontal channels (AF3/AF4/F3/F4/F7/F8) are dominated by eye and forehead
   artifacts, and blink rate is itself a state confound.

**What already existed and was reused (not rebuilt):**

- `NoiseAwareLatentTuRBO` (`src/optimizer/latent_turbo.py`) — a trust-region
  Bayesian optimizer that already consumes `Observation(reward_mean,
  reward_variance, …)`. Uncertainty-aware optimization was already done.
- Low-dimensional search — the optimizer works on a 7-dim breed-weight vector
  (or 8-dim PCA), never the raw ~16k UNet latent.
- `window_statistics` (`src/optimizer/observation.py`) — mean, variance-of-mean
  via effective sample size, and an artifact fraction.

---

## 2. Architecture

```
EmotivCortexSource / MockPreferenceEEGSource  (raw µV)
  → Preprocessor: bandpass 1–40 Hz + mains notch + CAR + blink/EMG reject   [Phase 1]
  → EEGFeatureExtractor: baseline-normalized band power + ERP + FAA features  [Phase 2]
  → per-window feature vectors  f(A), f(B)
  → contrast = f(B) − f(A)   (+ antisymmetric augmentation in training)
  → shrinkage-LDA, probability-calibrated, bagged ensemble                    [Phase 5]
  → p(B ≻ A) with uncertainty
  → reward_mean = 2p − 1 ;  reward_variance from ensemble std + artifacts      [Phase 6]
  → Observation → NoiseAwareLatentTuRBO.observe(...)                           [existing]
```

**Design principles**

- **Contrast, not absolutes** — the A-vs-B difference cancels per-session
  baseline/impedance drift, the main reason single-window EEG fails across days.
- **Fixed per-session anchor** — each candidate B is scored against a *fixed*
  reference window A captured once (`set_anchor`), not the rolling previous
  candidate. A rolling A makes the reward a function of `(z_current, z_previous)`,
  which is non-Markovian and breaks the GP surrogate. (This corrects the original
  plan's "previous display = A" idea — see §6.)
- **Label by embedding distance, not introspection** — ground truth is which of
  A/B is closer to a shown target in CLIP (or pixel) embedding space.
- **Mock-first** — a synthetic EEG source injects a tunable-SNR preference signal
  so the whole pipeline is testable offline, with a `signal_gain = 0` negative
  control.

---

## 3. Files added / changed

### New modules (`src/signal_service/`)

| File | Purpose |
|---|---|
| `preprocessing.py` | Streaming `Preprocessor` (causal bandpass + mains notch + common-average reference, per-channel filter state) and `window_artifact_fraction` (blink/EMG). `PreprocessedSource` wraps any EEG source and yields cleaned samples in the same `(t, {channel: value})` shape. `build_preprocessor(config)`. |
| `mock_preference.py` | `MockPreferenceEEGSource` — 14-channel synthetic EEG whose parietal theta power (and a weaker frontal alpha asymmetry) rises monotonically with the current image's preference. SNR = `signal_gain / noise_std`. `signal_gain = 0` is the negative control. |
| `preference_reward.py` | `LearnedPreferenceReward` — the optimizer adapter. Scores each candidate against a fixed anchor, averages multiple sliding sub-windows per candidate, and returns an `Observation` with variance from ensemble disagreement + artifacts + session reliability. |

### Changed modules

| File | Change |
|---|---|
| `src/signal_service/learned_reward.py` | `EEGFeatureExtractor` now: baseline-normalizes band powers to a within-session rest baseline (`FeatureBaseline`), adds ERP-window features (P300 parietal ~300–500 ms, FRN frontocentral ~250–350 ms), adds aggregate frontal-midline theta, keeps FAA asymmetries, and drops the old `mean`/`delta` features. Added `contrast_features(B, A)` (strictly antisymmetric). Added `PreferenceEnsemble` (bagged + calibrated, returns mean + uncertainty) with `save/load_preference_ensemble`. |
| `src/common/config.py` + `config/config.yaml` | New `preprocessing:` section (`PreprocessingConfig`). |
| `scripts/record_reward_trials.py` | **Rewritten** from single-image "focus/relax" to pairwise A-vs-B trials with embedding auto-labels, two EEG windows per trial, session/subject/target ids, balanced sides, catch trials, a per-session rest baseline sidecar, and multi-session mock synthesis. |
| `scripts/train_reward_model.py` | **Rewritten**: contrast input + antisymmetric augmentation, shrinkage-LDA default (logreg/svm/mlp options), calibrated bagged ensemble, and **leave-one-session-out** AUC reported against **FAA-alone** and **chance**. |
| `scripts/run_poodle_turbo_morph.py` | Added `--reward-source preference` and `--mock-eeg`; wired preprocessing into the FAA/learned paths too. |

### Tests

| File | Covers |
|---|---|
| `tests/test_preprocessing.py` | Bandpass removes DC + line noise; a blink raises the artifact fraction; `PreprocessedSource` is a drop-in. |
| `tests/test_preference_closed_loop.py` | Reward gradient points toward the target (deterministic); with a mock signal the optimizer steers the search above chance (`@pytest.mark.slow`). |

---

## 4. How to run

Use the project virtualenv (`.venv/bin/python`); the base interpreter lacks
scipy/sklearn.

### 4.0 Generate A/B candidate images (per target)

Calibration needs candidates that form a **near→far gradient around each target**,
not random unrelated images — otherwise "which is closer" is arbitrary and the EEG
signal is weak. Generate them with the **same SD-Turbo pipeline the live loop
uses** (no calibration-vs-inference distribution gap), building the gradient by
interpolating in prompt-embedding space between the target and a distractor:

```bash
.venv/bin/python scripts/generate_candidates.py --targets data/targets.json \
  --out-dir data/candidates_ai --levels 5 --per-level 6 --model stabilityai/sd-turbo
```

`--targets` is a JSON manifest (see `data/targets.example.json`): one entry per
target with a `name` (which **must match the real target photo's filename stem**)
and a `prompt`; `distractors` is optional (defaults to the other targets'
prompts). Output is one subfolder per target:

```
data/targets_real/poodle.jpg          # the real photo the subject memorizes
data/candidates_ai/poodle/L0_0.png …  # generated candidates, near→far
data/candidates_ai/beagle/…
```

The recorder auto-detects the per-target subfolders and draws each trial's A/B
**only from that target's own set**, so trials stay coherent. (A flat
`--candidate-dir` with no subfolders still works — every target then shares one
pool.) First run downloads the SD-Turbo weights; a GPU/MPS machine is strongly
recommended.

### 4.1 Record calibration trials (mock, offline)

Self-consistent: the injected EEG signal is driven by the same embedding
closeness that produces the labels.

```bash
.venv/bin/python scripts/record_reward_trials.py --mock --sessions 3 --trials 120 \
  --signal-gain 0.6 --stimuli-dir scripts/data/stimuli \
  --out scripts/data/reward_training/pref_trials.csv
```

Real headset (one session) — **real-photo target, AI-generated A/B candidates**,
plus `--present` for the subject-facing interface:

```bash
.venv/bin/python scripts/record_reward_trials.py --subject S01 --session-id S01_day1 \
  --present --fullscreen --embed clip \
  --target-dir data/targets_real --candidate-dir data/candidates_ai \
  --out scripts/data/reward_training/S01_day1.csv
```

The **target** (what the subject memorizes) is drawn from `--target-dir` (real
photographs); **images A and B** are drawn from `--candidate-dir` (AI-generated
images). Both pools are embedded together so distances are comparable, and the
label is which candidate is closer to the target. Catch trials use an obvious
easy pair from the candidate pool (closest vs furthest), side-balanced. The
single-pool `--stimuli-dir` form still exists for mock/offline runs.

**The presentation interface** (`--present`, `src/signal_service/stimulus_presenter.py`)
runs the trial as your design specifies:

1. **Target** shown to memorize (`--target-seconds`), then a **fixation** cross
   (`--fixation-seconds`).
2. **Image A** held while its EEG window is recorded (the window is repainted
   during capture, so it never freezes).
3. **A → B crossfade morph** (`--transition-seconds`) — the calibration analogue
   of the live latent morph; not scored.
4. **Image B** held while its EEG window is recorded.
5. **Inter-trial** fixation rest (`--iti-seconds`).

Labels still come from embedding distance (no button press). Press **q/ESC** to
stop early — trials collected so far are saved. Mock recording does not use the
interface (no human in the loop); pass `--present` with `--mock` only to preview
the UI.

Output CSV columns: `session_id, subject_id, trial, target_path, a_path, b_path,
is_catch, label, pref_a, pref_b, dist_a, dist_b, artifact_a, artifact_b` plus
`fa_<feature>` and `fb_<feature>` (the A- and B-window feature vectors).

### 4.2 Train + validate (the scientific gate)

```bash
.venv/bin/python scripts/train_reward_model.py \
  --data scripts/data/reward_training/pref_trials.csv \
  --model lda --out models/preference_model.joblib
```

`--data` accepts **one or more CSVs and/or a directory** — so you can record many
short (~5 min, `--trials ~28`) blocks as separate files with distinct
`--session-id`s and train on the whole folder at once:

```bash
.venv/bin/python scripts/train_reward_model.py \
  --data scripts/data/reward_training/ --model lda --out models/preference_model.joblib
```

The recorder **does not append** — each run overwrites its `--out`, so give every
block its own filename. Each block re-fits its own rest baseline (good — it
absorbs drift and headset re-seating), and `session_id` keeps blocks distinct for
leave-one-session-out.

`--features` selects which extractor columns the model trains on; the full 71-col
schema is still what the live loop feeds, and a `feature_mask` stored in the
ensemble applies the subset at inference. Default is `bandpower` (the 42 band-power
columns only), which generalised best out-of-session on real S01 data — LDA
~0.61 vs ~0.55 on the full vector. The per-channel `std`, mirror-pair asymmetry,
and single-value ERP columns mostly added cross-session noise. Options:
`all | bandpower | no-erp | power-asym`. Prefer the deterministic `lda`/`logreg`
over `mlp`, whose per-seed AUC swings ~±0.05 — never report a single-seed MLP run.

Prints leave-one-session-out AUC, FAA-alone AUC, and chance. **If the model does
not beat FAA-alone out-of-session, that is the finding — do not wire it into the
optimizer.** A JSON metrics sidecar is written next to the model.

### 4.3 Run the closed loop (mock EEG, headless-friendly reward core)

```bash
.venv/bin/python scripts/run_poodle_turbo_morph.py --reward-source preference \
  --mock-eeg --mock-signal-gain 0.6 --reward-model models/preference_model.joblib
```

Real headset: drop `--mock-eeg` (uses `EmotivCortexSource`). The full script needs
torch + SD-Turbo + OpenCV.

### 4.4 Tests

```bash
.venv/bin/python -m pytest            # everything
.venv/bin/python -m pytest -m "not slow"   # skip the closed-loop run
```

---

## 5. Validation results (mock)

| Condition | Leave-session-out AUC | FAA-alone AUC | Verdict |
|---|---|---|---|
| Signal (`--signal-gain 0.6`) | **0.95** | 0.60 | beats FAA + chance |
| Low SNR (`0.15`) | 0.86 | 0.57 | beats FAA + chance |
| Negative control (`0.0`) | **0.46 ≈ chance** | 0.50 | correctly finds nothing |

The harness recovers signal monotonically with SNR and returns ~chance when there
is none — it does not hallucinate signal. In the closed loop the preference reward
reliably steers the search above chance with low variance.

---

## 6. Honest limitations (read before over-claiming)

- **This is not image reconstruction from the brain.** Scalp EEG cannot decode
  content similarity to an imagined image. The reward is an affective/target
  relevance proxy; the mental-image link is indirect.
- **Closed-loop convergence is modest.** At realistic single-window SNR the
  classifier reward saturates, so the optimizer concentrates the search *above
  chance* but does not sharply pin the exact target. The strong result is the
  offline, leave-session-out decodability; the honest contribution is
  methods/systems (calibrated, uncertainty-aware, drift-cancelling pairwise reward
  feeding preferential Bayesian optimization), not decoding accuracy.
- **Deviation from the original plan:** the reward uses a *fixed* per-session
  anchor, not the rolling previous candidate. Rolling-previous froze the
  optimizer (`best_z` never updated) because it violates the GP's `reward = f(z)`
  assumption.
- **Real-headset and full-GUI paths** are wired and compile-checked but were
  validated only through the headless reward+optimizer core.
- **Window-length coupling:** the live feature window is hardcoded to 2.0 s in
  `_build_preference_reward` to match the recorder's `--window-seconds` default.
  Keep them equal, or the model's feature schema will mismatch the live features.

---

## 7. Suggested next steps

- Collect real multi-session data per subject (~300–600 clean pairwise trials
  over ≥3 days) and re-run the leave-session-out gate on real EEG.
- Wire per-session **reliability** (from catch-trial accuracy) into the reward's
  variance (`LearnedPreferenceReward(reliability=…)`).
- Consider EEGNet as a comparator only once ≥1–2k clean trials exist; a featurized
  linear model is expected to win at this data scale.
