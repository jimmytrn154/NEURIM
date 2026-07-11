#!/usr/bin/env python3
"""Record pairwise A-vs-B EEG preference trials for a learned reward model.

Each trial shows a (hidden) target, then image A, then image B. The label is
which of A/B is closer to the target in embedding space - not the subject's
introspection. Two EEG windows are captured per trial (A-settle, B-settle),
turned into baseline-normalized feature vectors, and written with full metadata
so scripts/train_reward_model.py can learn P(B preferred over A) and validate
leave-one-session-out.

Mock mode is self-consistent: the injected EEG preference signal is driven by the
SAME embedding closeness that produces the labels, so the full pipeline can be
validated offline at a known SNR (raise --signal-gain to add signal, set it to 0
for the negative control).

The target is a real photograph (what the subject holds in mind); images A and B
are AI-generated candidates. Pass real targets via --target-dir and generated
candidates via --candidate-dir. For mock/offline pipeline tests a single
--stimuli-dir serves as both pools.

Examples:
    # Offline: synthesize 3 mock sessions with a decodable signal (single pool)
    python scripts/record_reward_trials.py --mock --sessions 3 \
        --stimuli-dir data/stimuli \
        --out data/reward_training/pref_trials.csv

    # Real headset: real-photo targets, AI-generated A/B candidates
    python scripts/record_reward_trials.py --present --subject S01 --session-id S01_day1 \
        --target-dir data/targets_real --candidate-dir data/candidates_ai --embed clip \
        --out data/reward_training/S01_day1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Config, emotiv_credentials
from src.signal_service.eeg_sources import EmotivCortexSource
from src.signal_service.mock_preference import MockPreferenceEEGSource
from src.signal_service.preprocessing import PreprocessedSource, build_preprocessor
from src.signal_service.learned_reward import EEGFeatureExtractor, FeatureBaseline

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Stimuli + embeddings
# ---------------------------------------------------------------------------
def _images_in(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _pixel_embed(paths: list[Path]) -> np.ndarray:
    """Deterministic, network-free image embedding for offline pipeline testing."""
    from PIL import Image

    vecs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((24, 24))
        v = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        v -= v.mean()
        n = np.linalg.norm(v)
        vecs.append(v / n if n > 1e-8 else v)
    return np.asarray(vecs, dtype=np.float32)


def _as_feature_tensor(out):
    """Extract an embedding tensor from get_image_features across transformers versions.

    Some versions return a plain tensor; others wrap it in a ModelOutput
    (BaseModelOutputWithPooling / CLIP output) that has no `.norm`.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(out, attr, None)
        if value is not None:
            return value.mean(dim=1) if attr == "last_hidden_state" else value
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"Cannot extract image embedding tensor from {type(out).__name__}")


def _clip_embed(paths: list[Path]) -> np.ndarray:
    """CLIP image embeddings (openai/clip-vit-base-patch32). Requires a download."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name)
    proc = CLIPProcessor.from_pretrained(name)
    model.eval()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    with torch.no_grad():
        inputs = proc(images=imgs, return_tensors="pt")
        feats = _as_feature_tensor(model.get_image_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats.cpu().numpy().astype(np.float32)


def embed_images(paths: list[Path], method: str) -> np.ndarray:
    if method == "clip":
        return _clip_embed(paths)
    if method == "pixel":
        return _pixel_embed(paths)
    # auto: prefer CLIP if it imports and loads, else pixel
    try:
        return _clip_embed(paths)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[record] CLIP embed unavailable ({exc}); using pixel embeddings")
        return _pixel_embed(paths)


def preference_by_target(cand_embeds: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """Closeness of every candidate to a target embedding, min-max scaled to [-1, 1]."""
    sims = cand_embeds @ target_vec
    lo, hi = float(sims.min()), float(sims.max())
    if hi - lo < 1e-8:
        return np.zeros_like(sims)
    return (2.0 * (sims - lo) / (hi - lo) - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------
def _sample_pair_with_margin(prefs, cand_idx, min_margin, rng, max_tries=30):
    """Pick two candidates whose closeness-to-target differs by >= min_margin.

    Rejection sampling: keeps trials from being near-ties (where the label is a
    coin flip and the subject can't tell A from B), while the accepted pairs still
    span a range of gaps >= min_margin, giving graded difficulty for free. Falls
    back to the widest gap found if the pool is too clustered to meet the margin.
    """
    best = None
    best_gap = -1.0
    for _ in range(max_tries):
        a, b = rng.sample(cand_idx, 2)
        gap = abs(float(prefs[a]) - float(prefs[b]))
        if gap >= min_margin:
            return a, b
        if gap > best_gap:
            best_gap, best = gap, (a, b)
    return best


def build_trials(target_embeds, cand_embeds, target_cands, n_trials, catch_frac,
                 rng, min_margin, prefs_cache):
    """Yield (target_idx, a_idx, b_idx, is_catch).

    a_idx/b_idx are drawn from `target_cands[target]` - the candidate indices
    allowed for that target (its own generated set in per-target mode, or the
    whole pool otherwise). Normal trials draw an A/B pair whose closeness differs
    by at least `min_margin` (0 = unconstrained); catch trials use placeholders
    overridden later with an obvious easy pair. Populates `prefs_cache`
    (target_idx -> per-candidate closeness) so the recording loop reuses it."""
    n_targets = len(target_embeds)

    def prefs_for(t):
        p = prefs_cache.get(t)
        if p is None:
            p = preference_by_target(cand_embeds, target_embeds[t])
            prefs_cache[t] = p
        return p

    trials = []
    n_catch = int(round(n_trials * catch_frac))
    for k in range(n_trials):
        target = rng.randrange(n_targets)
        is_catch = k < n_catch
        if is_catch:
            a = b = -1  # overridden in the loop with the closest/furthest pair
        else:
            a, b = _sample_pair_with_margin(prefs_for(target), target_cands[target],
                                            min_margin, rng)
        trials.append((target, a, b, is_catch))
    rng.shuffle(trials)
    return trials


# ---------------------------------------------------------------------------
# EEG capture
# ---------------------------------------------------------------------------
def capture_window(stream, extractor, preproc, fs, window_s, mock_src=None, preference=None,
                   presenter=None, image=None, caption=None):
    if mock_src is not None and preference is not None:
        mock_src.set_preference(preference)
    extractor.clear()  # epoch starts at settle onset -> ERP features are stimulus-locked
    win: dict[str, list[float]] = {}
    need = int(fs * window_s)
    count = 0
    while count < need or not extractor.ready():
        _t, sample = next(stream)
        extractor.push_sample(sample)
        for ch, v in sample.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                win.setdefault(ch, []).append(float(v))
        count += 1
        # Keep the stimulus on screen (and the quit key live) during EEG capture.
        if presenter is not None and image is not None:
            presenter.paint(image, caption)
    result = extractor.vector()
    if result is None:
        raise RuntimeError("EEG feature extractor did not become ready")
    vec, names = result
    winarr = {ch: np.asarray(v, dtype=float) for ch, v in win.items()}
    artifact = preproc.window_artifact_fraction(winarr) if preproc is not None else 0.0
    return vec, names, artifact


def fit_session_baseline(stream, extractor, fs, window_s, n_windows, mock_src=None):
    """Capture rest windows and fit a per-session FeatureBaseline."""
    if mock_src is not None:
        mock_src.set_preference(0.0)
    vectors, names = [], None
    for _ in range(n_windows):
        extractor.clear()
        need = int(fs * window_s)
        count = 0
        while count < need or not extractor.ready():
            _t, sample = next(stream)
            extractor.push_sample(sample)
            count += 1
        vec, names = extractor.vector()
        vectors.append(vec)
    baseline = FeatureBaseline()
    baseline.fit(vectors, names)
    return baseline


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
def record_session(args, config, session_id, seed, pools, prefs_cache, writer_state):
    target_paths, target_embeds, cand_paths, cand_embeds, target_cands = pools
    fs = config.eeg.sample_rate_hz
    window_s = args.window_seconds
    preproc = build_preprocessor(config)

    mock_src = None
    if args.mock:
        base = MockPreferenceEEGSource(
            config.eeg.channels, fs, signal_gain=args.signal_gain,
            noise_std=args.noise_std, seed=seed,
        )
        mock_src = base
        source = PreprocessedSource(base, preproc) if preproc else base
    else:
        client_id, client_secret = emotiv_credentials()
        base = EmotivCortexSource(client_id, client_secret, headset_id=args.headset_id)
        source = PreprocessedSource(base, preproc) if preproc else base

    presenter = _build_presenter(args)

    source.connect()
    stream = source.stream()
    try:
        # Feature extractor without baseline for the rest fit, then attach baseline.
        extractor = EEGFeatureExtractor(fs, config.eeg.channels, window_s=window_s,
                                        pairs=config.faa.channel_pairs)
        if presenter is not None:
            presenter.message([
                f"Session {session_id}", "",
                "Keep still and relax for the baseline.", "",
                "Press SPACE to begin.",
            ])
            presenter.message(["Fitting rest baseline...", "Hold still."], seconds=0.5)
        print(f"[record] session={session_id} fitting rest baseline...")
        baseline = fit_session_baseline(stream, extractor, fs, window_s,
                                        args.baseline_windows, mock_src)
        extractor.baseline = baseline

        rng = random.Random(seed)
        trials = build_trials(target_embeds, cand_embeds, target_cands, args.trials,
                              args.catch_frac, rng, args.min_margin, prefs_cache)

        for i, (target, a, b, is_catch) in enumerate(trials, start=1):
            prefs = prefs_cache.get(target)
            if prefs is None:
                prefs = preference_by_target(cand_embeds, target_embeds[target])
                prefs_cache[target] = prefs
            if is_catch:
                # Obvious easy pair among this target's candidates (closest vs
                # furthest), side randomized so catch labels stay balanced.
                allowed = set(target_cands[target])
                order = [c for c in np.argsort(prefs) if c in allowed]
                far, near = int(order[0]), int(order[-1])
                a, b = (far, near) if rng.random() < 0.5 else (near, far)
            pref_a, pref_b = float(prefs[a]), float(prefs[b])
            label = int(pref_b > pref_a)  # 1 = B closer to target
            dist_a, dist_b = 1.0 - pref_a, 1.0 - pref_b

            img_a = img_b = None
            if presenter is not None:
                # target -> memorize, fixation, then A (scored), morph A->B, B (scored)
                presenter.show_target(presenter.load(target_paths[target]), args.target_seconds)
                presenter.fixation(args.fixation_seconds)
                img_a, img_b = presenter.load(cand_paths[a]), presenter.load(cand_paths[b])

            cap = f"A  (trial {i}/{len(trials)})"
            fa, names, art_a = capture_window(stream, extractor, preproc, fs, window_s,
                                              mock_src, pref_a, presenter, img_a, cap)
            if presenter is not None:
                presenter.crossfade(img_a, img_b, args.transition_seconds)
            cap = f"B  (trial {i}/{len(trials)})"
            fb, _, art_b = capture_window(stream, extractor, preproc, fs, window_s,
                                          mock_src, pref_b, presenter, img_b, cap)
            if presenter is not None and args.iti_seconds > 0:
                presenter.fixation(args.iti_seconds)

            row = {
                "session_id": session_id,
                "subject_id": args.subject,
                "trial": i,
                "target_path": str(target_paths[target]),
                "a_path": str(cand_paths[a]),
                "b_path": str(cand_paths[b]),
                "is_catch": int(is_catch),
                "label": label,
                "pref_a": pref_a, "pref_b": pref_b,
                "dist_a": dist_a, "dist_b": dist_b,
                "artifact_a": float(art_a), "artifact_b": float(art_b),
            }
            row.update({f"fa_{n}": float(v) for n, v in zip(names, fa)})
            row.update({f"fb_{n}": float(v) for n, v in zip(names, fb)})

            writer_state.append(row, names)
            if args.rest_seconds > 0 and not args.mock:
                time.sleep(args.rest_seconds)
            print(f"[record] {session_id} trial {i:03d}/{len(trials)} "
                  f"label={label} catch={int(is_catch)}")

        # Persist this session's baseline for the live loop.
        bpath = Path(args.out).with_name(f"{session_id}_baseline.json")
        bpath.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
    finally:
        source.close()
        if presenter is not None:
            presenter.close()


def _build_presenter(args):
    if not args.present:
        return None
    from src.signal_service.stimulus_presenter import StimulusPresenter

    return StimulusPresenter(size=args.present_size, fullscreen=args.fullscreen)


class _Writer:
    """Streams rows to CSV, writing the header once the feature schema is known."""

    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.feature_names: list[str] | None = None

    def append(self, row: dict, names: list[str]) -> None:
        if self.feature_names is None:
            self.feature_names = list(names)
        elif names != self.feature_names:
            raise RuntimeError("feature schema changed mid-recording")
        self.rows.append(row)

    def flush(self) -> None:
        assert self.feature_names is not None, "no rows recorded"
        meta = ["session_id", "subject_id", "trial", "target_path", "a_path", "b_path",
                "is_catch", "label", "pref_a", "pref_b", "dist_a", "dist_b",
                "artifact_a", "artifact_b"]
        feat_cols = [f"fa_{n}" for n in self.feature_names] + [f"fb_{n}" for n in self.feature_names]
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=meta + feat_cols)
            w.writeheader()
            w.writerows(self.rows)


def load_pools(args):
    """Return (target_paths, target_embeds, cand_paths, cand_embeds, target_cands).

    target_cands[t] is the list of candidate indices allowed for target t.

    - Two-pool per-target (--candidate-dir has subfolders named like the target
      stems): each target draws A/B only from its own generated candidate set -
      the coherent design that matches the live loop.
    - Two-pool flat (--candidate-dir is flat images): every target may use the
      whole candidate pool.
    - Single pool (--stimuli-dir): one pool serves as both; the target is excluded
      from its own candidates.
    """
    if args.target_dir and args.candidate_dir:
        tpaths = _images_in(args.target_dir)
        if not tpaths:
            sys.exit(f"No target images in {args.target_dir}")
        subdirs = [d for d in sorted(args.candidate_dir.iterdir()) if d.is_dir()] \
            if args.candidate_dir.exists() else []

        if subdirs:  # per-target: match subfolder name to target file stem
            by_name = {d.name: d for d in subdirs}
            cpaths: list[Path] = []
            target_cands: list[list[int]] = []
            for tp in tpaths:
                d = by_name.get(tp.stem)
                if d is None:
                    sys.exit(f"No candidate subfolder '{tp.stem}' in {args.candidate_dir} "
                             f"for target {tp.name}")
                imgs = _images_in(d)
                if len(imgs) < 2:
                    sys.exit(f"Need >=2 candidates in {d}, found {len(imgs)}")
                start = len(cpaths)
                cpaths.extend(imgs)
                target_cands.append(list(range(start, len(cpaths))))
            print(f"[record] embedding {len(tpaths)} targets + {len(cpaths)} candidates "
                  f"({len(subdirs)} per-target sets, {args.embed})")
        else:  # flat candidate pool shared by all targets
            cpaths = _images_in(args.candidate_dir)
            if len(cpaths) < 2:
                sys.exit(f"Need >=2 candidate images in {args.candidate_dir}, found {len(cpaths)}")
            target_cands = [list(range(len(cpaths)))] * len(tpaths)
            print(f"[record] embedding {len(tpaths)} targets + {len(cpaths)} candidates "
                  f"(flat pool, {args.embed})")

        both = embed_images(tpaths + cpaths, args.embed)
        return tpaths, both[: len(tpaths)], cpaths, both[len(tpaths):], target_cands

    if not args.stimuli_dir:
        sys.exit("Provide --stimuli-dir, or both --target-dir and --candidate-dir")
    paths = _images_in(args.stimuli_dir)
    if len(paths) < 3:
        sys.exit(f"Need >=3 images in {args.stimuli_dir}, found {len(paths)}")
    print(f"[record] embedding {len(paths)} stimuli ({args.embed})")
    embeds = embed_images(paths, args.embed)
    target_cands = [[i for i in range(len(paths)) if i != t] for t in range(len(paths))]
    return paths, embeds, paths, embeds, target_cands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Stimuli: either a single --stimuli-dir (target and candidates share one pool,
    # used for mock/offline), or separate pools -- --target-dir (real photos to
    # memorize) and --candidate-dir (AI-generated images shown as A/B).
    parser.add_argument("--stimuli-dir", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="real target images (memorize); pair with --candidate-dir")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="AI-generated images shown as A/B; pair with --target-dir")
    parser.add_argument("--out", type=Path,
                        default=Path("data/reward_training/pref_trials.csv"))
    parser.add_argument("--subject", default="S00")
    parser.add_argument("--session-id", default=None, help="single-session id (real hardware)")
    parser.add_argument("--sessions", type=int, default=1, help="synthesize N mock sessions")
    parser.add_argument("--trials", type=int, default=120, help="pairwise trials per session")
    parser.add_argument("--catch-frac", type=float, default=0.1)
    parser.add_argument("--min-margin", type=float, default=0.3,
                        help="min closeness gap between A and B per trial (0-2 scale; "
                             "0 = unconstrained). Avoids near-tie trials with noisy labels.")
    parser.add_argument("--embed", choices=["auto", "clip", "pixel"], default="pixel")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--baseline-windows", type=int, default=8)
    parser.add_argument("--rest-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--signal-gain", type=float, default=0.6,
                        help="mock preference SNR (0 = negative control)")
    parser.add_argument("--noise-std", type=float, default=0.3)
    # Subject-facing presentation (real sessions): target -> A -> morph -> B.
    parser.add_argument("--present", action="store_true",
                        help="show the OpenCV stimulus interface (target, A, morph, B)")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--present-size", type=int, default=768)
    parser.add_argument("--target-seconds", type=float, default=3,
                        help="how long the target is shown to memorize")
    parser.add_argument("--fixation-seconds", type=float, default=0.8)
    parser.add_argument("--transition-seconds", type=float, default=1.2,
                        help="A->B crossfade duration (not scored)")
    parser.add_argument("--iti-seconds", type=float, default=1.0,
                        help="inter-trial fixation rest")
    parser.add_argument("--headset-id", default=None)
    args = parser.parse_args()

    pools = load_pools(args)
    config = Config.load()
    prefs_cache: dict[int, np.ndarray] = {}

    from src.signal_service.stimulus_presenter import AbortPresentation

    writer = _Writer(args.out)
    n_sessions = args.sessions if args.mock else 1
    try:
        for s in range(n_sessions):
            session_id = args.session_id or f"{args.subject}_sess{s:02d}"
            # Vary seed (and, for mock, day-to-day drift) per session.
            record_session(args, config, session_id, args.seed + 1000 * s,
                           pools, prefs_cache, writer)
    except AbortPresentation:
        print("[record] aborted by operator; saving trials collected so far")

    if not writer.rows:
        sys.exit("[record] no trials recorded; nothing to save")
    writer.flush()
    n_catch = sum(r["is_catch"] for r in writer.rows)
    n_pos = sum(r["label"] for r in writer.rows)
    print(f"[record] wrote {len(writer.rows)} trials to {args.out} "
          f"(sessions={n_sessions}, catch={n_catch}, B-closer={n_pos})")


if __name__ == "__main__":
    main()
