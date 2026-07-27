"""Frozen linear-probe IDH evaluation on UCSF-PDGM — the bildiri's downstream measurement.

The encoder is FROZEN: each checkpoint is turned into 256-d features once (precise-BN on the target
cohort, single deterministic view), and only a logistic-regression head is fitted. That is the
measurement that speaks to this paper's claim — "does the SSL representation carry IDH information,
and does the label-free selection rule change how much?" — whereas fine-tuning would change the
encoder and answer a different question (how good an initialization it is).

Three things here exist because of what the UCSF metadata actually looks like, not as boilerplate:

  * **The clinical baseline is not optional.** In this cohort IDH is nearly collinear with WHO grade
    (374/402 grade-4 subjects are wildtype; 46/56 grade-2 are mutant). Age ALONE gives AUC ~0.90.
    An imaging AUC printed without that number beside it is uninterpretable, so every call scores
    the clinical-only arms and the image+age arm too. The honest question is not "what AUC do the
    features get" but "what do they add over age".

  * **The threshold is never chosen on the test fold.** Sensitivity / specificity / F1 / accuracy are
    threshold-dependent, and picking the threshold where they look best on the same data you report
    is self-confirmation. Youden's J is maximized on each fold's TRAINING part and applied unchanged
    to its test part.

  * **Checkpoint differences are compared PAIRED.** Every checkpoint sees identical subjects and
    identical CV splits, so the difference between two of them is far better determined than either
    absolute AUC. A paired bootstrap over subjects gives the delta-AUC interval that actually decides
    whether bestLiDAR transfers better than bestRankMe, instead of eyeballing two overlapping CIs.

Features are cached per (run, checkpoint) under `features/idh/`, because extraction is the only
expensive step — re-scoring with different CV settings is then seconds.

Usage
-----
  # one run, all four checkpoints
  python -m braintumor_ssl.idh_probe --runs checkpoints/simsiam_densenet121_unified_seed42

  # every finished run, aggregated into one CSV
  python -m braintumor_ssl.idh_probe --runs checkpoints/simsiam_*_unified_seed* --device 0

  # re-score from cache only (no GPU needed once features exist)
  python -m braintumor_ssl.idh_probe --runs <dirs> --cache_only --repeats 10
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from braintumor_ssl.data import make_views, scan_subjects
from braintumor_ssl.labels import attach, load_ucsf_labels
from braintumor_ssl.models import SimSiam
from braintumor_ssl.utils import recompute_bn_stats, resolve_device, set_seed, use_local_tmpdir

CHECKPOINTS = ["bestRankMe.pth", "bestLiDAR.pth", "bestA-ReQ.pth", "last.pth"]
CACHE_DIR = "features/idh"
COHORT = "UCSF-PDGM"

# arm -> which columns of the design matrix it uses ("img" = the 256-d encoder features)
ARMS = {
    "age": ["age"],
    "grade": ["grade"],
    "age+sex": ["age", "sex"],
    "image": ["img"],
    "image+age": ["img", "age"],
}


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
    bn_ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=max(bs, 2), shuffle=True,
                       drop_last=True, num_workers=nw)
    recompute_bn_stats(model, bn_ld, device, max_batches=200)
    model.eval()
    ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=bs, shuffle=False,
                    num_workers=nw)
    feats, ids = [], []
    for batch in ld:
        feats.append(model.encode(batch["image"].to(device)).float().cpu())
        ids.extend(batch["id"])
    return torch.cat(feats).numpy().astype(np.float32), ids


def cached_features(run_dir: str, ckpt: str, records: list[dict], device: str, bs: int, nw: int,
                    cache_only: bool):
    run = os.path.basename(os.path.normpath(run_dir))
    path = os.path.join(CACHE_DIR, f"{run}__{ckpt.replace('.pth', '')}.npz")
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
    print(f"  [cache] wrote {path}  {X.shape}")
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


def oof_predictions(X: np.ndarray, y: np.ndarray, folds: int, seed: int):
    """Out-of-fold probabilities + the per-fold training-set Youden threshold applied to each fold.

    Scaling and the threshold are fitted inside the training part of every fold, so nothing about
    the held-out subjects informs either.
    """
    p = np.zeros(len(y), dtype=np.float64)
    thr = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p_tr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        thr[te] = _youden_threshold(y[tr], p_tr)
    return p, thr


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


def score_arm(X: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int) -> dict:
    """Repeated stratified CV. AUC per repeat (pooled out-of-fold) -> mean ± SD captures split
    variability; the bootstrap CI on the repeat-averaged predictions captures subject sampling."""
    ps, aucs = [], []
    for r in range(repeats):
        p, thr = oof_predictions(X, y, folds, seed + r)
        ps.append(p)
        aucs.append(roc_auc_score(y, p))
        if r == 0:
            tm = threshold_metrics(y, p, thr)
    p_bar = np.mean(ps, axis=0)
    lo, hi = bootstrap_auc_ci(y, p_bar, seed=seed)
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "ci_lo": lo, "ci_hi": hi, **tm, "_p": p_bar}


# --------------------------------------------------------------------------- #
def main() -> None:
    use_local_tmpdir()
    ap = argparse.ArgumentParser(description="Frozen linear-probe IDH evaluation on UCSF-PDGM")
    ap.add_argument("--runs", nargs="+", required=True, help="encoder run dirs holding best*.pth")
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    ap.add_argument("--data_root", default="data_unified")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--reference", default="bestRankMe.pth",
                    help="checkpoint the paired delta-AUC comparison is measured against")
    ap.add_argument("--out", default="results/idh_probe.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="GPU index; default auto")
    ap.add_argument("--cache_only", action="store_true", help="score from cached features only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.cache_only else resolve_device(args.device)

    records = [r for r in scan_subjects(args.data_root) if r.get("collection") == COHORT]
    if not records:
        raise SystemExit(f"no {COHORT} subjects under {args.data_root}")
    records, labs = attach(records, load_ucsf_labels())
    y = np.array([l["idh"] for l in labs], dtype=int)
    cov = {k: np.array([float(l[k]) for l in labs]) for k in ("age", "sex", "grade")}
    print(f"[cohort] {len(records)} subjects, {int(y.sum())} mutant / {int((1 - y).sum())} wildtype")

    rows, preds = [], {}
    for run_dir in args.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        print(f"\n=== {run} ===")
        cfg_src = next((os.path.join(run_dir, c) for c in args.checkpoints
                        if os.path.exists(os.path.join(run_dir, c))), None)
        meta = torch.load(cfg_src, map_location="cpu", weights_only=False).get("cfg", {}) if cfg_src else {}
        backbone, seed = meta.get("backbone", "?"), meta.get("seed", "?")

        for ckpt in args.checkpoints:
            X, ids = cached_features(run_dir, ckpt, records, device, args.batch_size,
                                     args.num_workers, args.cache_only)
            if X is None:
                print(f"  {ckpt:<16} — not available (missing checkpoint or no cache)")
                continue
            assert ids == [r["id"] for r in records], f"{ckpt}: cached ids do not match the cohort order"
            for arm in ("image", "image+age"):
                s = score_arm(_design(arm, X, cov), y, args.folds, args.repeats, args.seed)
                preds[(run, ckpt, arm)] = s.pop("_p")
                rows.append({"run": run, "backbone": backbone, "seed": seed, "checkpoint": ckpt,
                             "arm": arm, "n": len(y), "n_pos": int(y.sum()), **s})
                print(f"  {ckpt:<16} {arm:<10} AUC={s['auc']:.3f}±{s['auc_sd']:.3f} "
                      f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]  sens={s['sens']:.2f} spec={s['spec']:.2f}")

    # clinical-only arms: identical for every run, so score them once
    print("\n=== clinical-only baselines (no imaging) ===")
    for arm in ("age", "grade", "age+sex"):
        s = score_arm(_design(arm, np.zeros((len(y), 0), dtype=np.float32), cov), y,
                      args.folds, args.repeats, args.seed)
        preds[("-", "-", arm)] = s.pop("_p")
        rows.append({"run": "-", "backbone": "-", "seed": "-", "checkpoint": "-", "arm": arm,
                     "n": len(y), "n_pos": int(y.sum()), **s})
        print(f"  {arm:<10} AUC={s['auc']:.3f}±{s['auc_sd']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")

    # paired comparisons: each checkpoint vs the reference, and image vs the age baseline
    print(f"\n=== paired delta-AUC (same subjects, same splits) ===")
    cmp_rows = []
    for (run, ckpt, arm), p in sorted(preds.items()):
        if arm != "image" or ckpt == "-":
            continue
        ref = preds.get((run, args.reference, "image"))
        if ref is not None and ckpt != args.reference:
            d, lo, hi = paired_delta_ci(y, p, ref, seed=args.seed)
            sig = "" if lo <= 0 <= hi else "  <- interval excludes 0"
            print(f"  {run} {ckpt:<16} vs {args.reference:<16} dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            cmp_rows.append({"run": run, "a": ckpt, "b": args.reference, "d_auc": d,
                             "ci_lo": lo, "ci_hi": hi})
        # ADDED VALUE over the clinical baseline. The contrast has to be (image+age) vs (age), not
        # (image) vs (age): the latter only asks whether features beat age on their own, which is
        # not the clinical question and which age usually wins outright in this cohort.
        p_aug = preds.get((run, ckpt, "image+age"))
        if p_aug is not None:
            d, lo, hi = paired_delta_ci(y, p_aug, preds[("-", "-", "age")], seed=args.seed)
            sig = "" if lo <= 0 <= hi else "  <- interval excludes 0"
            print(f"  {run} {ckpt:<16} image+age vs age   dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            cmp_rows.append({"run": run, "a": f"{ckpt}+age", "b": "age", "d_auc": d,
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
    print("\nReport the imaging arms NEXT TO the clinical baselines above — in this cohort IDH is "
          "nearly collinear with grade, so an imaging AUC alone does not establish added value.")


if __name__ == "__main__":
    main()
