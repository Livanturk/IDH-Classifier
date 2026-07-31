"""How reproducible is a checkpoint selection if the validation cohort had been slightly different?

`report_run.py` says WHICH epoch each metric picked. This measures whether a trailing-mean argmax
surrogate is sensitive to the validation cohort. It replays the eligibility gate and smoothed metric
curve on random subsets of the val subjects, using the per-epoch feature dumps
(`<out_dir>/val_features/ep####.npz`, written when `dump_val_features: true`), and reports the
distribution of selected epochs:

    P(selected epoch == the full-cohort pick)          <- the headline stability number
    the spread of picks across subsamples              <- how wide the "equally good" region is

Why this and not a confidence interval on the metric: the paper's claim is about a CHOICE, so the
quantity that has to be stable is the choice, not the metric value. A run where RankMe's peak moves
by 40 epochs when you drop 20% of the val subjects has not identified an optimal epoch, however
tight the error bar on RankMe itself looks.

It also re-runs the selection under a LOWER-DIMENSIONAL random orthogonal projection of the same
features. With n=200 val subjects and d=256 the covariance is rank-deficient; if the selected epochs
survive a projection to 128 or 64 dimensions (where n > d), the result is not an artifact of that
rank deficiency. A fixed-seed random orthogonal map is used rather than PCA on purpose — a PCA fitted
at one epoch would bake that epoch's basis into every other epoch's score.

The online gated policies additionally require a 1-SE improvement before replacing an incumbent
checkpoint. Recomputing that threshold in every subject subsample would require a nested delete-d
jackknife, which this utility does not currently do. Thus, whenever `select_se_mult > 0`, its output
must be called *argmax-surrogate stability*, not stability of the exact saved policy. SPS needs its
own exact replay because it uses a terminal-plateau rule rather than an argmax.

Everything here is post-hoc: no model, no GPU, no retraining. Add a metric or change a fit window
and re-run it over the same dumps.

Usage
-----
  python -m braintumor_ssl.selection_stability --run checkpoints/simsiam_densenet121_unified_seed42
  python -m braintumor_ssl.selection_stability --run <dir> --reps 200 --keep_frac 0.8 --project 128 64
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics

import numpy as np
import torch

from braintumor_ssl.utils import alpha_req_fit, lidar, rankme

METRIC_NAMES = {"rankme": "RankMe", "lidar": "LiDAR", "alpha_req": "alpha-ReQ"}


def load_dumps(run_dir: str) -> list[tuple[int, dict]]:
    """[(epoch, {name: (n, d) float32 array}), ...] sorted by epoch."""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "val_features", "ep*.npz"))):
        epoch = int(os.path.basename(p)[2:-4])
        with np.load(p) as z:
            out.append((epoch, {k: z[k] for k in z.files}))
    return out


def eligible_epochs(run_dir: str) -> set[int]:
    """Epochs that passed the run's own convergence + non-collapse gate (from metrics.csv).

    The stability replay has to use the SAME eligibility as training did — otherwise it would let a
    pre-convergence epoch win on a subsample and report spurious instability.
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return set()
    ok = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            conv = str(r.get("converged", "")).lower() in ("true", "1", "1.0")
            coll = str(r.get("collapsed", "")).lower() in ("true", "1", "1.0")
            if conv and not coll:
                ok.add(int(float(r["epoch"])))
    return ok


def _projector(d: int, k: int, seed: int) -> torch.Tensor:
    """(d, k) random orthonormal columns — a fixed, epoch-independent lower-dimensional view."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return q.float()


def score_epoch(feats: dict, metric: str, areq_kw: dict, idx=None, proj=None) -> float:
    """The selection score of one epoch on a subset of subjects, optionally projected.

    Same orientation as training: higher is better for all three, so alpha-ReQ is scored as
    -|alpha - 1| rather than alpha itself.
    """
    def prep(a):
        t = torch.from_numpy(a)
        if idx is not None:
            t = t[idx]
        return t @ proj if proj is not None else t

    if metric == "lidar":
        views = [prep(feats[k]) for k in sorted(feats) if k.startswith("view")]
        return lidar(*views) if len(views) >= 2 else math.nan
    ev = prep(feats["eval"])
    if metric == "rankme":
        return rankme(ev)
    return -abs(alpha_req_fit(ev, **areq_kw)["alpha"] - 1.0)


def select(curves: dict[int, float], smooth_w: int) -> int:
    """Argmax of a trailing-mean curve; it omits the online 1-SE replacement threshold."""
    eps = sorted(curves)
    best_e, best_v = -1, -math.inf
    for i, e in enumerate(eps):
        w = [curves[x] for x in eps[max(0, i - smooth_w + 1): i + 1]]
        w = [v for v in w if math.isfinite(v)]
        if not w:
            continue
        v = sum(w) / len(w)
        if v > best_v:
            best_e, best_v = e, v
    return best_e


def stability(run_dir: str, reps: int, keep_frac: float, smooth_w: int,
              areq_kw: dict, project: list[int], seed: int = 0, se_mult: float = 0.0,
              out_csv: str | None = None) -> list[dict[str, object]]:
    dumps = load_dumps(run_dir)
    if not dumps:
        raise SystemExit(f"no val_features/*.npz under {run_dir} — was dump_val_features enabled?")
    ok = eligible_epochs(run_dir)
    if ok:
        dumps = [(e, f) for e, f in dumps if e in ok]
    if len(dumps) < 2:
        raise SystemExit(f"only {len(dumps)} eligible epoch(s) dumped — nothing to compare.")

    n, d = dumps[0][1]["eval"].shape
    keep = max(3, int(round(n * keep_frac)))
    print("=" * 78)
    print(f"SELECTION STABILITY  {os.path.basename(os.path.normpath(run_dir))}")
    print(f"  {len(dumps)} eligible epochs, n_val={n}, d={d}, "
          f"{reps} subsamples of {keep} subjects, smoothing window {smooth_w}")
    if se_mult > 0:
        print(f"  WARNING: training used select_se_mult={se_mult:g}; results below are "
              "argmax-surrogate stability, not an exact replay of the saved policy.")
    print("=" * 78)

    spaces = [("full d=%d" % d, None)] + [(f"proj d={k}", _projector(d, k, seed + k))
                                          for k in project if 0 < k < d]
    rng = np.random.default_rng(seed)
    subsets = [torch.from_numpy(rng.permutation(n)[:keep].copy()) for _ in range(reps)]
    rows: list[dict[str, object]] = []

    for label, proj in spaces:
        print(f"\n[{label}]")
        print(f"    {'metric':<11}{'full-cohort':>12}{'P(same)':>9}{'median':>8}"
              f"{'IQR':>13}{'mean|shift|':>12}")
        for metric in METRIC_NAMES:
            full = {e: score_epoch(f, metric, areq_kw, proj=proj) for e, f in dumps}
            ref = select(full, smooth_w)
            picks = []
            for idx in subsets:
                cur = {e: score_epoch(f, metric, areq_kw, idx=idx, proj=proj) for e, f in dumps}
                p = select(cur, smooth_w)
                if p >= 0:
                    picks.append(p)
            if not picks:
                print(f"    {METRIC_NAMES[metric]:<11}{'—':>12}   (not selectable on any subsample)")
                continue
            same = sum(1 for p in picks if p == ref) / len(picks)
            q = statistics.quantiles(picks, n=4) if len(picks) >= 4 else [min(picks), 0, max(picks)]
            shift = statistics.fmean(abs(p - ref) for p in picks)
            rows.append({"run": os.path.basename(os.path.normpath(run_dir)), "space": label,
                         "metric": METRIC_NAMES[metric], "full_epoch": ref,
                         "reps": len(picks), "keep_frac": keep_frac, "n_val": n,
                         "dimension": d if proj is None else int(proj.shape[1]),
                         "p_same": same, "median_epoch": int(statistics.median(picks)),
                         "iqr_lo": float(q[0]), "iqr_hi": float(q[2]),
                         "mean_abs_shift": shift, "seed": seed})
            print(f"    {METRIC_NAMES[metric]:<11}{ref:>12}{same:>9.2f}"
                  f"{int(statistics.median(picks)):>8}{f'[{q[0]:.0f},{q[2]:.0f}]':>13}{shift:>12.1f}")

    print("\nRead: P(same) is the fraction of subsamples that re-pick the full-cohort epoch. Low P(same)")
    print("with a small mean|shift| means a flat plateau (any epoch in it is fine — report the")
    print("plateau, not the argmax). Low P(same) with a large shift means the pick is not identified")
    print("at all, and that metric should not be presented as a selection rule for this run.")
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing val_features/")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--keep_frac", type=float, default=0.8,
                    help="fraction of val subjects kept per subsample (without replacement)")
    ap.add_argument("--smooth_window", type=int, default=None,
                    help="default: read select_smooth_window from the run's checkpoint cfg")
    ap.add_argument("--project", type=int, nargs="*", default=[128, 64],
                    help="extra lower-dimensional random-orthogonal checks (empty to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch_threads", type=int, default=1,
                    help="CPU threads for small repeated decompositions (default: 1)")
    ap.add_argument("--out", default=None, help="optional CSV path for the stability summary")
    args = ap.parse_args()

    # mirror the run's own alpha fit window + smoothing so the replay matches how it really selected
    cfg = {}
    for f in ("bestRankMe.pth", "last.pth"):
        p = os.path.join(args.run, f)
        if os.path.exists(p):
            cfg = torch.load(p, map_location="cpu", weights_only=False).get("cfg", {})
            break
    areq_kw = {"k_min": int(cfg.get("areq_k_min", 1)),
               "k_max_frac": float(cfg.get("areq_k_max_frac", 1.0))}
    smooth_w = args.smooth_window if args.smooth_window is not None \
        else max(1, int(cfg.get("select_smooth_window", 1)))

    torch.set_num_threads(max(1, int(args.torch_threads)))
    stability(args.run, args.reps, args.keep_frac, smooth_w, areq_kw, args.project, args.seed,
              float(cfg.get("select_se_mult", 0.0) or 0.0), args.out)


if __name__ == "__main__":
    main()
