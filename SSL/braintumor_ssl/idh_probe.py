"""Frozen linear-probe IDH evaluation on UCSF-PDGM — the bildiri's downstream measurement.

    UCSF MRI -> FROZEN encoder -> 256-d features -> StandardScaler -> linear SVM -> IDH

The encoder never moves: each checkpoint is turned into 256-d features once (precise-BN on the
target cohort, single deterministic view) and only the linear head is fitted. That is the
measurement this paper's claim needs — "does the label-free selection rule change how well the
representation transfers?" — whereas fine-tuning would change the encoder and answer a different
question (how good an initialization it is).

What is deliberately NOT here, and why
--------------------------------------
  * **No feature selection.** All 256 dimensions go to the SVM. Not an omission: on this cohort 95%
    of their variance lies in 2 principal directions, so the signal is in COMBINATIONS of columns,
    not in individual ones. Measured — a |r|>0.90 in-fold correlation filter keeps 2-3 of 256
    columns and drops AUC from ~0.73 to ~0.58; an elastic-net cap at 30 features gives 0.678 vs
    0.729. See `make_head`.
  * **No head sweep.** One classifier, identical for every arm, so the only thing that differs
    between arms is the selection rule that produced the encoder.
  * **No clinical arms by default.** They cannot separate selection rules (every imaging arm shares
    the confound), so they are opt-in via `--clinical_baseline` and belong in Limitations, not the
    main table.

What is here because the data demanded it
-----------------------------------------
  * **The threshold is never chosen on the test fold.** Sensitivity / specificity / F1 / accuracy
    are threshold-dependent, and picking the threshold where they look best on the same data you
    report is self-confirmation. Youden's J is maximized on each fold's TRAINING part and applied
    unchanged to its test part. The scaler and the classifier are fitted there too, so nothing about
    a held-out subject informs anything applied to it.
  * **Checkpoint differences are compared PAIRED.** Every checkpoint sees identical subjects and
    identical CV splits, so the difference between two of them is far better determined than either
    absolute AUC. A paired bootstrap over subjects gives the delta-AUC interval that actually
    decides whether one rule transfers better than another, instead of eyeballing overlapping CIs.
  * **An untrained-encoder floor.** The label-free metrics PREFER random initialization (RankMe
    86-105 untrained versus 6-25 trained), so `--random_baseline` is what turns "our encoders
    differ" into "SSL pretraining beat random weights".

Features are cached per (run, pretraining epoch) under `features/idh/`. This is deliberate: when
multiple policies resolve to the same epoch, they have exactly the same encoder weights and must
share one feature matrix. Extraction is the only expensive step, so re-scoring with different CV
settings afterwards is seconds of CPU.

Usage
-----
  # one run, every arm of the comparison ladder
  python -m braintumor_ssl.idh_probe --runs checkpoints/simsiam_r18_unified_seed42_full

  # every finished run + the untrained-encoder floor
  python -m braintumor_ssl.idh_probe --runs checkpoints/simsiam_r18_unified_seed*_full \
      --random_baseline --device 0

  # re-score from cache (no GPU), plus the clinical numbers for the Limitations section
  python -m braintumor_ssl.idh_probe --runs <dirs> --cache_only --clinical_baseline
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from braintumor_ssl.data import make_views, scan_subjects
from braintumor_ssl.labels import attach, load_ucsf_labels
from braintumor_ssl.models import SimSiam
from braintumor_ssl.tracking import Tracker
from braintumor_ssl.utils import recompute_bn_stats, resolve_device, set_seed, use_local_tmpdir

# The paper's comparison ladder, in table order (methodology §8.2). The ungated `bestNaive*`
# checkpoints are the literature-faithful application of each metric to epoch selection — no
# convergence gate — and are produced by `select_epoch.py --baselines --write`. They exist so the
# proposed rule is not measured only against the gated variants (which would hide why the gates
# matter) nor only against the ungated ones (which would be a straw man).
CHECKPOINTS = ["bestNaiveRankMe.pth", "bestNaiveLiDAR.pth", "bestNaiveA-ReQ.pth",
               "bestRankMe.pth", "bestLiDAR.pth", "bestA-ReQ.pth",
               "bestSPS.pth", "bestPareto.pth", "last.pth"]
CACHE_DIR = "features/idh"
# Changing the precise-BN pass changes the resulting feature representation. A versioned cache key
# prevents pre-2026-07-30 stochastic caches from silently entering the paper tables.
FEATURE_CACHE_VERSION = "detbn-v2"
COHORT = "UCSF-PDGM"
EXCLUDE_FILE = "splits/ucsf_excluded_subjects.txt"


def load_exclusions(path: str = EXCLUDE_FILE) -> dict[str, str]:
    """{subject_id: reason} for subjects that must be dropped from the cohort.

    One UCSF volume is genuinely corrupt (gzip fails deterministically, unlike the transient
    shared-storage reads data.py retries around). Dropping it in a listed, reasoned file rather than
    swallowing the read error keeps the cohort size auditable: the paper has to say n=500, not 501.
    """
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sid, _, reason = line.partition("  ")
                out[sid.strip()] = reason.strip()
    return out

# arm -> which columns of the design matrix it uses ("img" = the 256-d encoder features)
ARMS = {
    "age": ["age"],
    "grade": ["grade"],
    "age+sex": ["age", "sex"],
    "image": ["img"],
    "image+age": ["img", "age"],
    # the full 258-column design (256 encoder features + age + sex) the elastic-net arm sparsifies
    "image+age+sex": ["img", "age", "sex"],
}


def make_head(seed: int, class_weight: str | None = "balanced") -> LinearSVC:
    """The classifier fitted inside each fold's TRAINING part.

    Linear SVM, as the experimental framework specifies, and also the standard "linear probe" of the
    SSL literature. `LinearSVC` rather than `SVC(kernel="linear", probability=True)` because the
    latter runs an internal Platt calibration (a nested CV) per fit for a score that only has to be
    MONOTONE — AUC and the Youden threshold both depend on the ordering alone.

    NO feature selection, and that is a measured decision rather than an omission. This encoder's
    256 dimensions are not 256 semi-independent measurements: 95% of their variance on the target
    cohort lies in 2 principal directions (RankMe 9-25). The signal therefore lives in COMBINATIONS
    of the columns, not in individual ones, and pruning "redundant" features destroys the averaging
    that suppresses their noise. Measured on this cohort: a |r|>0.90 in-fold correlation filter
    keeps 2-3 of 256 columns and drops AUC from ~0.73 to ~0.58; an elastic-net cap at 30 features
    gives 0.678 versus 0.729 unrestricted. All 256 columns are used.

    Regularization is left at the library default on purpose: the study's claim is that ONLY the
    selection rule differs between arms, so the head must be identical across them, and an
    untuned-but-identical head serves that better than a tuned one whose chosen C could vary by arm.
    The primary protocol uses `class_weight="balanced"` throughout (103 mutant / 500).
    The optional SMOTE robustness analysis deliberately uses unweighted SVM instead: combining
    oversampling and inverse-frequency weights would over-emphasize the mutant class twice.
    """
    return LinearSVC(C=1.0, max_iter=20000, class_weight=class_weight, random_state=seed)


def positive_scores(clf, X: np.ndarray) -> np.ndarray:
    """Positive-class score. The linear SVM has no calibrated probability, but AUC and the Youden
    threshold depend only on the ORDER of the scores, so the decision margin is a valid score. It is
    not on a probability scale, which is why `score_arm` rank-normalizes before averaging across CV
    repeats — averaging raw margins from differently scaled fits would be meaningless."""
    return np.asarray(clf.decision_function(X), dtype=np.float64)


def _rank01(v: np.ndarray) -> np.ndarray:
    """Monotone map to [0, 1] by rank — makes scores from different fits commensurable."""
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


_EPOCH_CACHE: dict[str, int] = {}


def checkpoint_epoch(run_dir: str, ckpt: str) -> int:
    """Which PRETRAINING epoch this checkpoint holds.

    Recorded per row so the results CSV can answer "did the selection rules actually pick different
    epochs?" without re-opening the run. Deriving it from the filename only works for the periodic
    `ckpt_eNNN.pth`; for `bestRankMe.pth` / `bestPareto.pth` / `last.pth` the epoch lives inside the
    file, and defaulting those to -1 (as this did) silently erased exactly the distinction the
    paper's table has to check. Memoized: the read is ~270 MB.
    """
    if ckpt.startswith("ckpt_e"):
        return int(ckpt[6:-4])
    path = os.path.join(run_dir, ckpt)
    if path not in _EPOCH_CACHE:
        try:
            _EPOCH_CACHE[path] = int(torch.load(path, map_location="cpu",
                                                weights_only=False).get("epoch", -1))
        except (FileNotFoundError, KeyError, ValueError, RuntimeError):
            _EPOCH_CACHE[path] = -1
    return _EPOCH_CACHE[path]


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_cohort(ckpt_path: str, records: list[dict], device: str, bs: int, nw: int):
    """256-d frozen-encoder features after a precise-BN recompute on THIS cohort.

    The BN recompute is not polish: SimSiam's loss is L2-normalized so BN running stats are
    unconstrained, and this is also the site adaptation (AdaBN) for the BraTS -> UCSF shift. The
    dataset is rebuilt from the cfg stored inside the checkpoint, so UCSF goes through exactly the
    crop / normalization the encoder was trained with.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = SimSiam(backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
                    feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
                    proj_hidden_dim=cfg["proj_hidden_dim"],
                    pred_hidden_dim=cfg["pred_hidden_dim"]).to(device)
    model.load_state_dict(ck["model"])
    # This pass is part of the evaluated encoder, not a stochastic augmentation. In particular,
    # `drop_last=True` used to omit a different set of UCSF subjects on every policy invocation;
    # consequently byte-identical checkpoints could produce slightly different BN statistics and
    # downstream scores. Fixed order + full cohort make the representation reproducible.
    bn_ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=max(bs, 2), shuffle=False,
                       drop_last=False, num_workers=nw)
    recompute_bn_stats(model, bn_ld, device, max_batches=200)
    model.eval()
    ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=bs, shuffle=False,
                    num_workers=nw)
    feats, ids = [], []
    for batch in ld:
        feats.append(model.encode(batch["image"].to(device)).float().cpu())
        ids.extend(batch["id"])
    return torch.cat(feats).numpy().astype(np.float32), ids


@torch.no_grad()
def encode_random_init(cfg: dict, seed: int, records: list[dict], device: str, bs: int, nw: int):
    """Features from an UNTRAINED encoder of the same architecture — the control this study needs.

    Comparing the SSL encoders only to each other cannot support "we learned a good 3D MRI
    representation": that claim needs a floor. A randomly initialized network of the identical
    architecture, run through the identical preprocessing and the identical precise-BN adaptation,
    is that floor. It matters especially here because the label-free metrics PREFER random init
    (RankMe 86-102 untrained vs 6-20 trained), so the downstream comparison against random weights
    is the only thing that tells us the training did something useful.
    """
    set_seed(seed)
    model = SimSiam(backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
                    feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
                    proj_hidden_dim=cfg["proj_hidden_dim"],
                    pred_hidden_dim=cfg["pred_hidden_dim"]).to(device)
    bn_ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=max(bs, 2), shuffle=False,
                       drop_last=False, num_workers=nw)
    recompute_bn_stats(model, bn_ld, device, max_batches=200)
    model.eval()
    ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=bs, shuffle=False,
                    num_workers=nw)
    feats, ids = [], []
    for batch in ld:
        feats.append(model.encode(batch["image"].to(device)).float().cpu())
        ids.extend(batch["id"])
    return torch.cat(feats).numpy().astype(np.float32), ids


def cached_random_features(cfg: dict, seed: int, records: list[dict], device: str, bs: int, nw: int,
                           cache_only: bool):
    name = f"randominit_{cfg['backbone']}__seed{seed}"
    path = os.path.join(CACHE_DIR, f"{name}__{FEATURE_CACHE_VERSION}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            return z["X"], [str(i) for i in z["ids"]]
    if cache_only:
        return None, None
    X, ids = encode_random_init(cfg, seed, records, device, bs, nw)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, ids=np.array(ids))
    print(f"  [cache] wrote {path}  {X.shape}")
    return X, ids


def cached_features(run_dir: str, ckpt: str, records: list[dict], device: str, bs: int, nw: int,
                    cache_only: bool):
    """Load features canonicalized by resolved pretraining epoch.

    Selection files contain provenance metadata and therefore need not be byte-identical as files,
    but policies that name the same epoch have identical ``model`` tensors. Keying the cache by
    ``(run, epoch, cache-version)`` makes that identity explicit and prevents artificial policy
    differences from separate precise-BN passes.
    """
    run = os.path.basename(os.path.normpath(run_dir))
    epoch = checkpoint_epoch(run_dir, ckpt)
    if epoch < 0:
        return None, None
    path = os.path.join(CACHE_DIR, f"{run}__epoch{epoch:03d}__{FEATURE_CACHE_VERSION}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            return z["X"], [str(i) for i in z["ids"]]
    if cache_only:
        return None, None
    ckpt_path = os.path.join(run_dir, ckpt)
    if not os.path.exists(ckpt_path):
        return None, None
    X, ids = encode_cohort(ckpt_path, records, device, bs, nw)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, ids=np.array(ids))
    print(f"  [cache] wrote {path}  {X.shape}  (canonical epoch {epoch}; requested {ckpt})")
    return X, ids


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _design(arm: str, X_img: np.ndarray, cov: dict[str, np.ndarray]) -> np.ndarray:
    parts = [X_img if c == "img" else cov[c].reshape(-1, 1) for c in ARMS[arm]]
    return np.hstack(parts)


def _youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1, computed on TRAINING data only."""
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P, N = max(1, ys.sum()), max(1, len(ys) - ys.sum())
    j = tp / P - fp / N
    return float(p[order][int(np.argmax(j))])


def smote_resample(X: np.ndarray, y: np.ndarray, *, k_neighbors: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard binary SMOTE, restricted to an outer-CV training fold.

    This deliberately has no dependency on ``imbalanced-learn`` because the cluster environment
    does not ship that package.  It implements the original SMOTE interpolation rule in the
    *standardized training-feature space*: each synthetic minority example is placed uniformly on
    the segment from a randomly selected minority point to one of its k nearest minority neighbours.
    Callers must never pass validation/test samples here.
    """
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"SMOTE requires two classes; got labels={labels.tolist()}")
    minority = labels[int(np.argmin(counts))]
    n_min, n_max = int(counts.min()), int(counts.max())
    if n_min < 2:
        raise ValueError("SMOTE requires at least two minority training samples")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    n_new = n_max - n_min
    if n_new == 0:
        return X.copy(), y.copy()

    X_min = np.asarray(X[y == minority], dtype=np.float64)
    n_neighbors = min(int(k_neighbors) + 1, n_min)  # includes the query sample itself
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_min)
    neighbours = nn.kneighbors(X_min, return_distance=False)
    rng = np.random.default_rng(seed)
    source = rng.integers(0, n_min, size=n_new)
    synth = np.empty((n_new, X.shape[1]), dtype=X_min.dtype)
    for j, src in enumerate(source):
        candidates = neighbours[src][neighbours[src] != src]
        # With n_min >= 2, candidates is non-empty even if duplicate feature vectors occur.
        near = int(rng.choice(candidates))
        lam = float(rng.random())
        synth[j] = X_min[src] + lam * (X_min[near] - X_min[src])
    return np.vstack([X, synth]), np.concatenate([y, np.full(n_new, minority, dtype=y.dtype)])


def oof_predictions(X: np.ndarray, y: np.ndarray, folds: int, seed: int,
                    *, resampler: str = "none", smote_k_neighbors: int = 5):
    """Out-of-fold probabilities, the per-fold Youden threshold, and the per-fold sparsity.

    Scaling, optional SMOTE, the classifier and the threshold are all fitted inside the training
    part of every fold, so nothing about held-out subjects informs any of them.  SMOTE is applied
    only after training-fold scaling; it never sees test features and is never used upstream of the
    frozen 256-D representation.

    Returns (p, thr, n_nonzero). `n_nonzero` is the mean number of non-zero coefficients across
    folds; with the linear SVM it is always the full 256, and it is reported rather than assumed so
    that "no feature selection happened" is visible in the results file instead of being a claim.
    """
    p = np.zeros(len(y), dtype=np.float64)
    thr = np.zeros(len(y), dtype=np.float64)
    nz = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if resampler not in {"none", "smote"}:
        raise ValueError(f"unknown resampler {resampler!r}")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        X_tr = sc.transform(X[tr])
        y_tr = y[tr]
        class_weight: str | None = "balanced"
        if resampler == "smote":
            # A fold-specific seed makes the resampling deterministic while avoiding five copies
            # of the same interpolation draw across different outer folds/repeats.
            X_tr, y_tr = smote_resample(X_tr, y_tr, k_neighbors=smote_k_neighbors,
                                        seed=(seed + 1) * 10_000 + fold_idx)
            class_weight = None
        clf = make_head(seed, class_weight=class_weight)
        clf.fit(X_tr, y_tr)
        # The operating threshold is learned from the *original* training subjects, not SMOTE
        # pseudo-points. This keeps the decision threshold tied to the observed clinical prevalence.
        p_tr = positive_scores(clf, sc.transform(X[tr]))
        p[te] = positive_scores(clf, sc.transform(X[te]))
        thr[te] = _youden_threshold(y[tr], p_tr)
        nz.append(int(np.count_nonzero(clf.coef_)))
    return p, thr, float(np.mean(nz))


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict:
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(1, tp + fn)                       # = recall
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    return {"sens": sens, "spec": spec, "prec": prec, "npv": npv,
            "acc": (tp + tn) / max(1, len(y)),
            "f1": 2 * prec * sens / max(1e-9, prec + sens), "balanced_acc": (sens + spec) / 2,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI by resampling SUBJECTS (stratified, so every resample keeps both classes)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_delta_ci(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000, seed: int = 0,
                    alpha: float = 0.05):
    """CI for AUC(a) - AUC(b) with the SAME subjects resampled for both — the comparison that
    matters when two checkpoints are scored on one cohort. Overlapping marginal CIs say nothing
    about a paired difference."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(d)), float(lo), float(hi)


def score_arm(X: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int,
              *, resampler: str = "none", smote_k_neighbors: int = 5) -> dict:
    """Repeated stratified CV with every reported metric aggregated over repeats.

    AUC is computed from each repeat's pooled out-of-fold scores. Threshold-dependent metrics use
    the corresponding repeat's training-fold Youden thresholds, then are averaged across repeats;
    reporting only the first repeat would make F1/Sensitivity/Specificity less stable than AUC.
    The bootstrap CI uses repeat-averaged rank-normalized out-of-fold predictions.
    """
    ps, aucs, nzs, thresholded = [], [], [], []
    for r in range(repeats):
        p, thr, nz = oof_predictions(X, y, folds, seed + r, resampler=resampler,
                                     smote_k_neighbors=smote_k_neighbors)
        # Averaging across repeats needs commensurable scores, and SVM margins are not (every
        # repeat's fits carry their own scale), so they are rank-mapped first.
        ps.append(_rank01(p))
        aucs.append(roc_auc_score(y, p))
        nzs.append(nz)
        thresholded.append(threshold_metrics(y, p, thr))
    p_bar = np.mean(ps, axis=0)
    lo, hi = bootstrap_auc_ci(y, p_bar, seed=seed)
    tm = {key: float(np.mean([m[key] for m in thresholded])) for key in thresholded[0]}
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "ci_lo": lo, "ci_hi": hi, "head": "svm", "n_features": int(X.shape[1]),
            "n_selected": round(float(np.mean(nzs)), 1), "resampler": resampler,
            "smote_k_neighbors": int(smote_k_neighbors) if resampler == "smote" else None,
            **tm, "_p": p_bar}


# --------------------------------------------------------------------------- #
# experiment tracking
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    """MLflow metric keys allow only [A-Za-z0-9_-./ ] — 'bestA-ReQ.pth' would silently break."""
    return "".join(c if (c.isalnum() or c in "_-./ ") else "_" for c in name)


def _tracker(run: str, args, n: int, n_pos: int):
    """One tracking run per encoder, so the downstream numbers land next to that encoder's
    pretraining curves instead of only existing as a local CSV."""
    cfg = {"mlflow": not args.no_mlflow, "mlflow_experiment": args.mlflow_experiment}
    t = Tracker(cfg, run_name=f"idh_probe_{run}")
    t.log_params({"encoder_run": run, "cohort": COHORT, "n_subjects": n, "n_mutant": n_pos,
                  "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
                  "arms": ",".join(args.arms), "head": "linear_svm",
                  "probe": "frozen encoder + linear SVM, all 256 features"})
    return t


def _log_arm(tracker, ckpt: str, arm: str, s: dict, epoch: int) -> None:
    """Log one (checkpoint, arm) result — every reported metric, not just AUC.

    Two shapes, and the discriminator is the checkpoint's ROLE, not its epoch number:

      * periodic `ckpt_eNNN.pth` -> one metric name, step=<pretraining epoch>, so the tracker draws
        downstream performance as a CURVE over pretraining. That plot is what tests whether the
        label-free rules (which all select the earliest eligible epoch) picked an under-trained
        encoder.
      * named checkpoints (`bestRankMe`, `bestPareto`, `last`, `randominit`, ...) -> flat one-off
        metrics under their own name, because each is a distinct ARM of the comparison table.

    The branch used to key off `epoch >= 0`, which silently depended on `checkpoint_epoch` returning
    -1 for named checkpoints. Once that function was fixed to read the real epoch out of the file,
    every named checkpoint started going down the curve branch instead — so the arms vanished from
    the tracker as individual metrics and survived only as indistinguishable points on the sweep
    curve. Selecting on the filename cannot drift that way.
    """
    if tracker is None:
        return
    keys = ("auc", "auc_sd", "ci_lo", "ci_hi", "sens", "spec", "prec", "npv", "acc", "f1",
            "balanced_acc")
    if ckpt.startswith("ckpt_e"):
        tracker.log_metrics({f"sweep_{_safe(arm)}_{k}": s[k] for k in keys}, step=max(epoch, 0))
        return
    base = f"{_safe(ckpt.replace('.pth', ''))}_{_safe(arm)}"
    payload = {f"{base}_{k}": s[k] for k in keys}
    # which pretraining epoch this arm resolved to — the table's other half, and the thing that says
    # whether two rules actually selected different weights
    if epoch >= 0:
        payload[f"{base}_epoch"] = float(epoch)
    tracker.log_metrics(payload)


def main() -> None:
    use_local_tmpdir()
    ap = argparse.ArgumentParser(description="Frozen linear-probe IDH evaluation on UCSF-PDGM")
    ap.add_argument("--runs", nargs="+", required=True, help="encoder run dirs holding best*.pth")
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    ap.add_argument("--data_root", default="data_unified")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--resampler", choices=("none", "smote"), default="none",
                    help="downstream training-fold resampling; primary analysis uses none")
    ap.add_argument("--smote_k_neighbors", type=int, default=5,
                    help="minority-neighbour count for --resampler smote")
    ap.add_argument("--arms", nargs="+", default=["image"],
                    help="imaging arms to score. The paper reports `image` alone: that is the "
                         f"framework's design (features -> SVM -> IDH). Available: {sorted(ARMS)}")
    ap.add_argument("--clinical_baseline", action="store_true",
                    help="also score the clinical-only arms (age / grade / age+sex). Off by "
                         "default because they are not arms of the comparison — every imaging arm "
                         "shares the same confound, so they cannot separate selection rules. Kept "
                         "reachable because the Limitations section has to quote one number: in "
                         "this cohort IDH is nearly collinear with WHO grade and age ALONE reaches "
                         "AUC ~0.90, which is the context any imaging AUC must be read against.")
    ap.add_argument("--reference", default="bestRankMe.pth",
                    help="checkpoint the paired delta-AUC comparison is measured against")
    ap.add_argument("--out", default="results/idh_probe.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="GPU index; default auto")
    ap.add_argument("--cache_only", action="store_true", help="score from cached features only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random_baseline", action="store_true",
                    help="also score an UNTRAINED encoder of each backbone (the control that turns "
                         "'our encoders differ' into 'SSL pretraining beat random weights')")
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--no_mlflow", action="store_true", help="skip experiment tracking")
    ap.add_argument("--mlflow_experiment", default="simsiam-brats-idh")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.cache_only else resolve_device(args.device)

    records = [r for r in scan_subjects(args.data_root) if r.get("collection") == COHORT]
    excl = load_exclusions()
    if excl:
        before = len(records)
        records = [r for r in records if r["id"] not in excl]
        for sid, reason in sorted(excl.items()):
            print(f"[cohort] excluded {sid}: {reason}")
        print(f"[cohort] {before} -> {len(records)} subjects after exclusions")
    if not records:
        raise SystemExit(f"no {COHORT} subjects under {args.data_root}")
    records, labs = attach(records, load_ucsf_labels())
    y = np.array([l["idh"] for l in labs], dtype=int)
    cov = {k: np.array([float(l[k]) for l in labs]) for k in ("age", "sex", "grade")}
    print(f"[cohort] {len(records)} subjects, {int(y.sum())} mutant / {int((1 - y).sum())} wildtype")

    rows, preds, trackers = [], {}, {}
    for run_dir in args.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        print(f"\n=== {run} ===")
        tracker = trackers[run] = _tracker(run, args, len(y), int(y.sum()))
        cfg_src = next((os.path.join(run_dir, c) for c in args.checkpoints
                        if os.path.exists(os.path.join(run_dir, c))), None)
        meta = torch.load(cfg_src, map_location="cpu", weights_only=False).get("cfg", {}) if cfg_src else {}
        backbone, seed = meta.get("backbone", "?"), meta.get("seed", "?")

        ckpts = list(args.checkpoints)
        for ckpt in ckpts:
            X, ids = cached_features(run_dir, ckpt, records, device, args.batch_size,
                                     args.num_workers, args.cache_only)
            if X is None:
                print(f"  {ckpt:<16} — not available (missing checkpoint or no cache)")
                continue
            assert ids == [r["id"] for r in records], f"{ckpt}: cached ids do not match the cohort order"
            for arm in args.arms:
                s = score_arm(_design(arm, X, cov), y, args.folds, args.repeats, args.seed,
                              resampler=args.resampler, smote_k_neighbors=args.smote_k_neighbors)
                preds[(run, ckpt, arm)] = s.pop("_p")
                ep = checkpoint_epoch(run_dir, ckpt)
                rows.append({"run": run, "backbone": backbone, "seed": seed, "checkpoint": ckpt,
                             "pretrain_epoch": ep, "arm": arm, "n": len(y),
                             "n_pos": int(y.sum()), **s})
                sparse = f"  feats={s['n_selected']:.0f}/{s['n_features']}"
                print(f"  {ckpt:<16} {arm:<14} AUC={s['auc']:.3f}±{s['auc_sd']:.3f} "
                      f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]  sens={s['sens']:.2f} "
                      f"spec={s['spec']:.2f}{sparse}")
                _log_arm(tracker, ckpt, arm, s, ep)

    # --- untrained-encoder control, one per (backbone, seed) -----------------------
    if args.random_baseline:
        print("\n=== random-init control (untrained encoders, identical preprocessing) ===")
        cfgs = {}
        for run_dir in args.runs:                      # reuse a real run's cfg per backbone
            for c in args.checkpoints:
                p = os.path.join(run_dir, c)
                if os.path.exists(p):
                    m = torch.load(p, map_location="cpu", weights_only=False).get("cfg", {})
                    cfgs.setdefault(m.get("backbone", "?"), m)
                    break
        for bb, cfg in sorted(cfgs.items()):
            for rs in args.random_seeds:
                X, ids = cached_random_features(cfg, rs, records, device, args.batch_size,
                                                args.num_workers, args.cache_only)
                if X is None:
                    continue
                for arm in args.arms:
                    s = score_arm(_design(arm, X, cov), y, args.folds, args.repeats, args.seed,
                                  resampler=args.resampler, smote_k_neighbors=args.smote_k_neighbors)
                    preds[(f"randominit_{bb}", f"seed{rs}", arm)] = s.pop("_p")
                    rows.append({"run": f"randominit_{bb}_seed{rs}", "backbone": bb, "seed": rs,
                                 "checkpoint": "randominit", "pretrain_epoch": -1, "arm": arm,
                                 "n": len(y), "n_pos": int(y.sum()), **s})
                    print(f"  {bb:<13} seed{rs} {arm:<10} AUC={s['auc']:.3f}±{s['auc_sd']:.3f} "
                          f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")

    # clinical-only arms: identical for every run, so score them once. Opt-in (see
    # --clinical_baseline): context for the Limitations section, never a row of the main table.
    clin_auc = {}
    if args.clinical_baseline:
        print("\n=== clinical-only baselines (no imaging) ===")
    for arm in (("age", "grade", "age+sex") if args.clinical_baseline else ()):
        # SAME head as the imaging arms on purpose: a clinical baseline scored with a different
        # classifier is not the baseline for these numbers, it is a different experiment.
        s = score_arm(_design(arm, np.zeros((len(y), 0), dtype=np.float32), cov), y,
                      args.folds, args.repeats, args.seed,
                      resampler=args.resampler, smote_k_neighbors=args.smote_k_neighbors)
        preds[("-", "-", arm)] = s.pop("_p")
        rows.append({"run": "-", "backbone": "-", "seed": "-", "checkpoint": "-",
                     "pretrain_epoch": -1, "arm": arm, "n": len(y), "n_pos": int(y.sum()), **s})
        clin_auc[arm] = s["auc"]
        print(f"  {arm:<10} AUC={s['auc']:.3f}±{s['auc_sd']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")

    # paired comparisons: each checkpoint vs the reference, the matched random-init encoder,
    # and (when requested) image+age vs the clinical age baseline.
    print(f"\n=== paired delta-AUC (same subjects, same splits) ===")
    cmp_rows = []
    for (run, ckpt, arm), p in sorted(preds.items()):
        if arm != "image" or ckpt == "-":
            continue
        ref = preds.get((run, args.reference, "image"))
        if ref is not None and ckpt != args.reference and not ckpt.startswith("ckpt_e"):
            d, lo, hi = paired_delta_ci(y, p, ref, seed=args.seed)
            sig = "" if lo <= 0 <= hi else "  <- interval excludes 0"
            print(f"  {run} {ckpt:<16} vs {args.reference:<16} dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            cmp_rows.append({"run": run, "a": ckpt, "b": args.reference, "d_auc": d,
                             "ci_lo": lo, "ci_hi": hi})
        # ADDED VALUE over the clinical baseline, only when BOTH the image+age arm and the clinical
        # baseline were requested. Not part of the main table: the contrast answers a clinical
        # question (do the features add to age?), not the paper's question (which selection rule
        # transfers best?), and in this cohort age dominates it either way.
        p_aug = preds.get((run, ckpt, "image+age"))
        if p_aug is not None:
            d, lo, hi = paired_delta_ci(y, p_aug, preds[("-", "-", "age")], seed=args.seed)
            sig = "" if lo <= 0 <= hi else "  <- interval excludes 0"
            print(f"  {run} {ckpt:<16} image+age vs age   dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            cmp_rows.append({"run": run, "a": f"{ckpt}+age", "b": "age", "d_auc": d,
                             "ci_lo": lo, "ci_hi": hi})

    # Random initialization is a representation floor, not a checkpoint-selection policy, so its
    # comparison must be paired with the SSL encoder produced by the SAME initialization seed.
    # This is intentionally separate from the loop above: random-init rows use a synthetic run
    # name (``randominit_<backbone>_seed<seed>``), whereas pretrained rows use the run directory.
    if args.random_baseline:
        print("\n=== paired SSL gain over matched random-init encoder ===")
        for run_dir in args.runs:
            run = os.path.basename(os.path.normpath(run_dir))
            run_rows = [r for r in rows if r["run"] == run]
            if not run_rows:
                continue
            backbone, seed = run_rows[0]["backbone"], run_rows[0]["seed"]
            random_key = (f"randominit_{backbone}", f"seed{seed}", "image")
            random_scores = preds.get(random_key)
            if random_scores is None:
                print(f"  {run}: matched random-init seed{seed} unavailable")
                continue
            for ckpt in args.checkpoints:
                ssl_scores = preds.get((run, ckpt, "image"))
                if ssl_scores is None:
                    continue
                d, lo, hi = paired_delta_ci(y, ssl_scores, random_scores, seed=args.seed)
                sig = "" if lo <= 0 <= hi else "  <- interval excludes 0"
                print(f"  {run} {ckpt:<16} vs random-init(seed{seed}) dAUC={d:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]{sig}")
                cmp_rows.append({"run": run, "a": ckpt, "b": "randominit", "d_auc": d,
                                 "ci_lo": lo, "ci_hi": hi})

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")
    if cmp_rows:
        cpath = args.out.replace(".csv", "_paired.csv")
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"wrote {cpath} ({len(cmp_rows)} paired comparisons)")
    # artifacts + the clinical reference land on every run's tracking entry, then close them
    for run, tk in trackers.items():
        if tk is None:
            continue
        tk.log_metrics({f"clinical_{_safe(a)}_auc": v for a, v in clin_auc.items()})
        for f in (args.out, args.out.replace(".csv", "_paired.csv"),
                  os.path.join("checkpoints", run, "metric_agreement.png"),
                  os.path.join("checkpoints", run, "training_curves.png")):
            tk.log_artifact(f, artifact_path="downstream")
        tk.close()

    if args.clinical_baseline:
        print("\nRead the imaging arms NEXT TO the clinical baselines above — in this cohort IDH is "
              "nearly collinear with grade, so an imaging AUC alone does not establish clinical "
              "value. The comparison BETWEEN arms is unaffected: they share the confound.")


if __name__ == "__main__":
    main()
