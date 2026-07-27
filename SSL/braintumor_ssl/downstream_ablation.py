"""SUPERSEDED by `idh_probe.py` — kept for its internal-split scaffold only.

The UCSF-is-external rule this module enforces was scoped to the downstream PAPER. For the
bildiri the user decided (2026-07-27) that UCSF-PDGM is the labelled downstream cohort and is
used NOW, so the frozen-probe evaluation lives in `braintumor_ssl.idh_probe`, which scores the
four checkpoints on UCSF with repeated CV, in-fold thresholds, clinical baselines and paired
delta-AUC. UCSF stays out of SSL PRETRAINING either way — that is what keeps it a clean
transfer target and is unchanged.

Original docstring follows.

Downstream ablation scaffold: which selection rule (RankMe / LiDAR / alpha-ReQ) transfers best?

From ONE encoder run this loads the three per-metric best checkpoints (+ `last.pth` as the
final-plateau reference), extracts 256-d features on an INTERNAL downstream split, and — once an
IDH label table exists — compares their downstream performance with a linear probe and k-NN under
cross-validation. The checkpoint whose features give the best labelled score is the empirical answer
to "does the extra LiDAR / alpha-ReQ checkpoint beat plain RankMe downstream?".

HARD RULE (enforced below): UCSF-PDGM is EXTERNAL validation. It must never enter this comparison /
selection. It is admissible ONLY through `--external`, and only to REPORT a final number for an
already-chosen checkpoint — never to decide which metric/checkpoint is better. This keeps UCSF a
clean, ideally single-use external test for the downstream paper.

Until an IDH label CSV is supplied the script still runs end-to-end (loads all checkpoints, extracts
+ caches features) and clearly reports that the labelled comparison is deferred.

Usage
-----
  # internal comparison (no labels yet -> extracts features, defers scoring)
  python -m braintumor_ssl.downstream_ablation --run checkpoints/simsiam_densenet121_unified_seed42 \
      --splits_file splits/splits_downstream_internal.json --split val

  # once labels exist: id,label CSV drives the linear-probe / kNN comparison
  python -m braintumor_ssl.downstream_ablation --run <run> --splits_file <internal> --labels idh.csv

  # FINAL external eval of the ALREADY-CHOSEN checkpoint only (never for selection)
  python -m braintumor_ssl.downstream_ablation --run <run> --external splits/ucsf_external.json \
      --labels idh_ucsf.csv --final_checkpoint bestRankMe.pth
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from braintumor_ssl.data import load_splits, make_views, scan_subjects
from braintumor_ssl.models import SimSiam
from braintumor_ssl.utils import recompute_bn_stats, set_seed, use_local_tmpdir

CANDIDATES = ["bestRankMe.pth", "bestLiDAR.pth", "bestA-ReQ.pth", "last.pth"]
UCSF = "UCSF-PDGM"


def _has_ucsf(records: list[dict]) -> bool:
    return any(r.get("collection") == UCSF for r in records)


def load_records(splits_file: str | None, split: str, data_root: str | None) -> list[dict]:
    if splits_file:
        return load_splits(splits_file)[split]
    if data_root:
        return scan_subjects(data_root)
    raise SystemExit("provide --splits_file or --data_root")


@torch.no_grad()
def extract_features(ckpt_path: str, records: list[dict], device: str, bs: int, nw: int):
    """256-d encoder features (single deterministic eval view) after a precise-BN recompute on the
    target cohort — same protocol as extract_features.py / evaluate.py."""
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    model = SimSiam(backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
                    feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
                    proj_hidden_dim=cfg["proj_hidden_dim"], pred_hidden_dim=cfg["pred_hidden_dim"]).to(device)
    model.load_state_dict(ck["model"])
    bn_ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=max(bs, 2), shuffle=True,
                       drop_last=True, num_workers=nw)
    recompute_bn_stats(model, bn_ld, device, max_batches=200)
    model.eval()
    ld = DataLoader(make_views(records, cfg, mode="eval"), batch_size=bs, shuffle=False, num_workers=nw)
    feats, ids = [], []
    for batch in ld:
        feats.append(model.encode(batch["image"].to(device)).cpu())
        ids.extend(batch["id"])
    return torch.cat(feats).numpy(), ids


def linear_probe_scores(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Stratified 5-fold CV AUC for a logistic-regression linear probe and a k-NN classifier
    (the standard label-free-encoder transfer readouts). Returns mean±sd for each."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    k = min(5, int(np.bincount(y).min()))
    if k < 2:
        return {"probe_auc": float("nan"), "knn_auc": float("nan"), "note": "too few per-class samples"}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    probe, knn = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lp = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])
        probe.append(roc_auc_score(y[te], lp.predict_proba(Xte)[:, 1]))
        kn = KNeighborsClassifier(n_neighbors=min(5, len(tr))).fit(Xtr, y[tr])
        knn.append(roc_auc_score(y[te], kn.predict_proba(Xte)[:, 1]))
    return {"probe_auc": float(np.mean(probe)), "probe_sd": float(np.std(probe)),
            "knn_auc": float(np.mean(knn)), "knn_sd": float(np.std(knn))}


def load_labels(path: str | None, ids: list[str]) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    tab = pd.read_csv(path)
    id_col = "id" if "id" in tab.columns else tab.columns[0]
    lab_col = "label" if "label" in tab.columns else tab.columns[1]
    m = dict(zip(tab[id_col].astype(str), tab[lab_col]))
    y = [m.get(str(i)) for i in ids]
    if any(v is None for v in y):
        missing = sum(v is None for v in y)
        print(f"[labels][warn] {missing}/{len(ids)} subjects have no label in {path} — dropped from scoring")
    return np.array([(-1 if v is None else int(v)) for v in y])


def main() -> None:
    use_local_tmpdir()
    ap = argparse.ArgumentParser(description="Downstream ablation across the 3 selection checkpoints")
    ap.add_argument("--run", required=True, help="encoder run dir holding best*.pth")
    ap.add_argument("--splits_file", help="INTERNAL downstream split (for the selection comparison)")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--data_root")
    ap.add_argument("--labels", help="id,label CSV (IDH). If absent, scoring is deferred.")
    ap.add_argument("--checkpoints", nargs="+", default=CANDIDATES)
    ap.add_argument("--external", help="EXTERNAL split (e.g. UCSF) — FINAL report only, never selection")
    ap.add_argument("--final_checkpoint", help="with --external: the single, already-chosen checkpoint to report")
    ap.add_argument("--out", default="results/downstream_ablation.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------- FINAL external-only mode --------------------
    if args.external:
        if not args.final_checkpoint:
            raise SystemExit("--external requires --final_checkpoint: the checkpoint must already be chosen "
                             "on INTERNAL data; UCSF/external is report-only, never for selection.")
        recs = load_records(args.external, args.split, None)
        print(f"[external] FINAL external evaluation on {len(recs)} subjects "
              f"({'UCSF present' if _has_ucsf(recs) else 'no UCSF'}) — REPORT ONLY, not selection.")
        X, ids = extract_features(os.path.join(args.run, args.final_checkpoint), recs,
                                  device, args.batch_size, args.num_workers)
        y = load_labels(args.labels, ids)
        if y is None:
            print("[external] no labels -> features extracted; final AUC deferred until labels exist.")
            return
        mask = y >= 0
        sc = linear_probe_scores(X[mask], y[mask], args.seed)
        print(f"[external][FINAL] {args.final_checkpoint}: probe_auc={sc.get('probe_auc'):.4f} "
              f"knn_auc={sc.get('knn_auc'):.4f}  (single external report)")
        return

    # -------------------- INTERNAL selection comparison --------------------
    recs = load_records(args.splits_file, args.split, args.data_root)
    if _has_ucsf(recs):
        raise SystemExit("[guard] the internal comparison split contains UCSF-PDGM. UCSF is EXTERNAL "
                         "validation and must NOT be used to compare/select checkpoints. Remove UCSF "
                         "from this split (use --external for the final UCSF report instead).")
    print(f"[internal] downstream comparison on {len(recs)} non-UCSF subjects | device={device}")

    ckpts = [c for c in args.checkpoints if os.path.exists(os.path.join(args.run, c))]
    missing = [c for c in args.checkpoints if c not in ckpts]
    if missing:
        print(f"[internal][note] absent checkpoints skipped: {missing} "
              "(a metric may have had no eligible epoch).")

    rows, feat_cache = [], {}
    for c in ckpts:
        X, ids = extract_features(os.path.join(args.run, c), recs, device, args.batch_size, args.num_workers)
        feat_cache[c] = (X, ids)
        print(f"  extracted {X.shape} from {c}")

    y = load_labels(args.labels, ids) if ckpts else None
    if y is None:
        print("\n[internal] No IDH label table supplied yet -> features extracted and cached; the "
              "linear-probe / kNN comparison is DEFERRED until labels arrive. Wiring is in place: pass "
              "--labels id_label.csv to activate scoring. (Selection will use these INTERNAL scores only; "
              "UCSF stays external.)")
    else:
        print("\n[internal] linear-probe + kNN AUC per checkpoint (5-fold CV; higher = better transfer):")
        for c in ckpts:
            X, ids_c = feat_cache[c]
            mask = y >= 0
            sc = linear_probe_scores(X[mask], y[mask], args.seed)
            rows.append({"run": os.path.basename(os.path.normpath(args.run)), "checkpoint": c, **sc})
            print(f"  {c:<16} probe_auc={sc.get('probe_auc', float('nan')):.4f} "
                  f"knn_auc={sc.get('knn_auc', float('nan')):.4f}")
        if rows:
            best = max(rows, key=lambda r: (r.get("probe_auc") or 0))
            print(f"\n[internal] best INTERNAL downstream transfer: {best['checkpoint']} "
                  f"(probe_auc={best.get('probe_auc'):.4f}). This is the selection answer; confirm ONCE on "
                  "UCSF via --external.")
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            df = pd.DataFrame(rows)
            if os.path.exists(args.out):
                df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
            df.to_csv(args.out, index=False)
            print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
