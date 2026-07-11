#!/usr/bin/env python3
"""Train and validate a pairwise EEG preference model.

Input is the contrast features(B) - features(A) from a pairwise recording
(scripts/record_reward_trials.py); the target is P(B preferred over A). This
script is the project's scientific gate: it reports leave-one-session-out AUC and
compares it against two baselines - chance (0.5) and FAA-alone (the classic
alpha-asymmetry features on the same windows). If the full model does not beat
FAA-alone out-of-session, that is the finding; do not wire it into the optimizer.

The deployed artifact is a calibrated bagged PreferenceEnsemble (mean probability
+ uncertainty), saved via src.signal_service.learned_reward.

Example:
    python scripts/train_reward_model.py \
        --data scripts/data/reward_training/pref_trials.csv \
        --model lda --out models/preference_model.joblib
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_service.learned_reward import PreferenceEnsemble, save_preference_ensemble


def _resolve_csvs(inputs: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of trial CSVs."""
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.exists():
            files.append(p)
        else:
            raise RuntimeError(f"--data path not found: {p}")
    # Baseline sidecars written by the recorder are *_baseline.json, not CSVs, so
    # they are already excluded; skip anything without pairwise feature columns below.
    if not files:
        raise RuntimeError(f"No CSV files found in: {[str(p) for p in inputs]}")
    return files


def _load_one(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path} has no rows")
    fa_cols = [c for c in cols if c.startswith("fa_")]
    fb_cols = [c[3:] for c in cols if c.startswith("fb_")]
    names = [c[3:] for c in fa_cols]
    if not names:
        raise RuntimeError(f"{path} has no fa_/fb_ feature columns (is it a trials CSV?)")
    if names != fb_cols:
        raise RuntimeError(f"{path}: fa_/fb_ feature schemas differ")
    fa = np.asarray([[float(r[f"fa_{n}"]) for n in names] for r in rows], dtype=np.float32)
    fb = np.asarray([[float(r[f"fb_{n}"]) for n in names] for r in rows], dtype=np.float32)
    X = fb - fa  # the pairwise contrast
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    groups = np.asarray([r["session_id"] for r in rows])
    catch = np.asarray([int(r.get("is_catch", 0)) for r in rows], dtype=int)
    return X, y, groups, catch, names


def load_pairwise(inputs: list[Path]):
    """Load and concatenate one or more trial CSVs / directories.

    Every file must share the same feature schema; session_id keeps the blocks
    distinct for leave-one-session-out. This lets you record many short (~5 min)
    blocks as separate files and train on all of them at once.
    """
    files = _resolve_csvs(inputs)
    Xs, ys, gs, cs = [], [], [], []
    names0 = None
    for path in files:
        X, y, groups, catch, names = _load_one(path)
        if names0 is None:
            names0 = names
        elif names != names0:
            raise RuntimeError(f"{path}: feature schema differs from {files[0]}")
        Xs.append(X); ys.append(y); gs.append(groups); cs.append(catch)
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(cs), names0)


def make_estimator(model: str, seed: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if model == "lda":
        est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model == "logreg":
        est = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    elif model == "svm":
        est = SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale")
    else:
        est = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, early_stopping=True,
                            max_iter=1000, random_state=seed)
        
    return make_pipeline(StandardScaler(), est)


def _proba_pos(model, X):
    if hasattr(model, "predict_proba"):
        idx = list(model.classes_).index(1)
        return model.predict_proba(X)[:, idx]
    s = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))


def augment_antisymmetric(X, y, groups):
    """contrast(A,B) == -contrast(B,A): add the mirrored, label-flipped copy."""
    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([y, 1 - y])
    g_aug = np.concatenate([groups, groups])
    return X_aug, y_aug, g_aug


def augment_jitter(X, y, groups, copies: int, noise_std: float, seed: int):
    """Add small Gaussian jitter to training features.

    This is only for train folds / final ensemble, never held-out validation.
    `noise_std` is in standardized-ish feature units after baseline preprocessing;
    keep it small because EEG labels are noisy already.
    """
    if copies <= 0 or noise_std <= 0:
        return X, y, groups
    rng = np.random.default_rng(seed)
    Xs = [X]
    ys = [y]
    gs = [groups]
    scale = np.std(X, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    for _ in range(copies):
        Xs.append(X + rng.normal(0.0, noise_std, size=X.shape) * scale)
        ys.append(y.copy())
        gs.append(groups.copy())
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def training_augment(X, y, groups, jitter_copies: int, jitter_std: float, seed: int):
    X_aug, y_aug, g_aug = augment_antisymmetric(X, y, groups)
    return augment_jitter(X_aug, y_aug, g_aug, jitter_copies, jitter_std, seed)


def leave_session_out(model_name, X, y, groups, seed, jitter_copies=0, jitter_std=0.0):
    """Pooled out-of-fold predictions from leave-one-session-out CV.

    Training folds are antisymmetrically augmented; the held-out session is not.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    sessions = sorted(set(groups))
    oof = np.full(len(y), np.nan)
    per_session = {}
    for s in sessions:
        test = groups == s
        train = ~test
        Xtr, ytr, _gtr = training_augment(
            X[train], y[train], groups[train], jitter_copies, jitter_std, seed
        )
        model = make_estimator(model_name, seed)
        model.fit(Xtr, ytr)
        p = _proba_pos(model, X[test])
        oof[test] = p
        if len(np.unique(y[test])) == 2:
            per_session[s] = float(roc_auc_score(y[test], p))
    pooled_auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) == 2 else None
    pooled_acc = float(accuracy_score(y, (oof >= 0.5).astype(int)))
    return oof, pooled_auc, pooled_acc, per_session


def faa_feature_mask(names):
    """Columns corresponding to classic frontal alpha asymmetry."""
    return np.asarray([("alpha_asym" in n) for n in names], dtype=bool)


# Feature subsets the model can train on. Leave-one-session-out sweeps on real
# S01 data showed that pruning down to band power alone generalises best out of
# session (LDA ~0.61 vs ~0.55 on the full 71-column vector): per-channel `std`,
# the mirror-pair asymmetries, and the single-value ERP columns mostly add noise
# across sessions. The full extractor schema is unchanged; this only selects
# which of its columns the preference model sees.
FEATURE_SETS = ("all", "bandpower", "no-erp", "power-asym")


def select_feature_mask(names, kind):
    """Boolean mask over `names` selecting the columns for a feature subset."""
    def keep(n):
        if kind == "all":
            return True
        if kind == "bandpower":
            return "log_power" in n
        if kind == "no-erp":
            return not ("erp" in n or "frontal_midline" in n)
        if kind == "power-asym":
            return ("log_power" in n) or ("_asym" in n)
        raise ValueError(f"unknown feature set: {kind}")

    mask = np.asarray([keep(n) for n in names], dtype=bool)
    if not mask.any():
        raise RuntimeError(f"feature set '{kind}' selected 0 of {len(names)} columns")
    return mask


def build_ensemble(model_name, X, y, oof, n_members, seed, jitter_copies=0, jitter_std=0.0):
    """Bagged ensemble on all (augmented) data + a 1-D probability calibrator."""
    from sklearn.linear_model import LogisticRegression

    Xa, ya, _ = training_augment(
        X, y, np.zeros(len(y)), jitter_copies, jitter_std, seed
    )
    rng = np.random.default_rng(seed)
    models = []
    n = len(ya)
    for k in range(n_members):
        idx = rng.integers(0, n, n)  # bootstrap resample
        m = make_estimator(model_name, seed + k)
        m.fit(Xa[idx], ya[idx])
        models.append(m)
    # Calibrate ensemble-mean prob against the pooled leave-session-out predictions,
    # so calibration is measured out-of-session (not on the training fit).
    calibrator = None
    if len(np.unique(y)) == 2:
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(oof.reshape(-1, 1), y)
    return models, calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, nargs="+",
                        help="one or more trial CSVs and/or directories of CSVs")
    parser.add_argument("--out", type=Path, default=Path("models/preference_model.joblib"))
    parser.add_argument("--model", choices=["lda", "logreg", "svm", "mlp"], default="lda")
    parser.add_argument("--features", choices=FEATURE_SETS, default="bandpower",
                        help="which extractor columns the model trains on "
                             "(default: bandpower - the band-power columns only, "
                             "which generalised best out-of-session on S01)")
    parser.add_argument("--ensemble-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-catch", action="store_true",
                        help="exclude catch trials from validation metrics")
    parser.add_argument("--jitter-copies", type=int, default=0,
                        help="extra noisy copies per augmented training sample")
    parser.add_argument("--jitter-std", type=float, default=0.0,
                        help="Gaussian jitter size relative to each feature's train-fold std")
    args = parser.parse_args()

    X, y, groups, catch, names = load_pairwise(args.data)
    if len(set(groups)) < 2:
        sys.exit("Need >=2 sessions for leave-one-session-out validation")
    if args.drop_catch:
        keep = catch == 0
        X, y, groups = X[keep], y[keep], groups[keep]

    # Feature subset the model trains on. `names` stays the full extractor schema
    # (so the ensemble's saved feature_names still matches the live loop); the
    # mask selects the model's columns and is applied both here and at inference.
    sel_mask = select_feature_mask(names, args.features)
    Xsel = X[:, sel_mask]

    # Full model (on the selected subset)
    oof, auc, acc, per_session = leave_session_out(
        args.model, Xsel, y, groups, args.seed, args.jitter_copies, args.jitter_std
    )
    # FAA-alone baseline: fixed control on the classic asymmetry columns of the
    # full vector, regardless of --features, so the comparison stays honest.
    faa_mask = faa_feature_mask(names)
    _, faa_auc, faa_acc, faa_per = leave_session_out(
        args.model, X[:, faa_mask], y, groups, args.seed
    )

    models, calibrator = build_ensemble(
        args.model, Xsel, y, oof, args.ensemble_size, args.seed,
        args.jitter_copies, args.jitter_std
    )
    ensemble = PreferenceEnsemble(models=models, feature_names=names, model_type=args.model,
                                  calibrator=calibrator, feature_mask=sel_mask)

    metrics = {
        "model_type": args.model,
        "feature_set": args.features,
        "n_trials": int(len(y)),
        "n_sessions": int(len(set(groups))),
        "n_features": int(X.shape[1]),
        "n_model_features": int(sel_mask.sum()),
        "chance_auc": 0.5,
        "loso_auc": auc,
        "loso_accuracy": acc,
        "loso_auc_per_session": per_session,
        "faa_baseline_auc": faa_auc,
        "faa_baseline_accuracy": faa_acc,
        "faa_n_features": int(faa_mask.sum()),
        "beats_faa": (auc is not None and faa_auc is not None and auc > faa_auc),
        "beats_chance": (auc is not None and auc > 0.5),
        "ensemble_size": args.ensemble_size,
        "antisymmetric_augmentation": True,
        "jitter_copies": args.jitter_copies,
        "jitter_std": args.jitter_std,
    }
    save_preference_ensemble(args.out, ensemble, metrics)

    print(f"[train] model={args.model}  features={args.features} "
          f"({int(sel_mask.sum())}/{len(names)})  "
          f"sessions={metrics['n_sessions']}  trials={metrics['n_trials']}")
    print(f"[train] leave-session-out AUC = {auc:.3f}   (accuracy {acc:.3f})")
    print(f"[train] FAA-alone     AUC = {faa_auc:.3f}   ({faa_mask.sum()} features)")
    print(f"[train] chance        AUC = 0.500")
    verdict = "BEATS FAA" if metrics["beats_faa"] else "does NOT beat FAA"
    print(f"[train] verdict: {verdict};  per-session AUC = "
          + ", ".join(f"{s}:{v:.2f}" for s, v in per_session.items()))
    print(f"[train] saved calibrated ensemble -> {args.out}")


if __name__ == "__main__":
    main()
