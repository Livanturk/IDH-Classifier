"""Figures for the SSL pretraining study (academic outputs).

CPU-only and artifact-based (no model inference), so it runs on the login node:

  * ``curves`` — overlay the training metrics (RankMe / loss / z_std / participation / alignment /
    uniformity vs epoch) of several runs, plus a final-value bar chart. Reads each run's
    ``metrics.csv`` (written by train_simsiam when ``val_every`` > 0). This is THE backbone-
    comparison figure. It also writes the metric-agreement figures: per run, each selection rule's
    score with the epoch it picked (``metric_agreement_<run>.png``), and across runs a Spearman
    heatmap of whether the three metrics order the epochs the same way
    (``metric_agreement_spearman.png``). Both are restricted to epochs that passed the convergence /
    collapse gate — correlating through the random-init transient would mostly measure that all
    three metrics fall away from initialization.
  * ``roi`` — qualitative montage of the adaptive whole-tumour ROIs (4 modalities, best slice) and
    the two SimSiam augmented views of one subject. Reads the dataset via ``make_views``, so it
    shows exactly what the encoder sees. Methods-section figure.

Representation-space figures (t-SNE embedding, singular-value spectrum) live in ``evaluate.py``,
which already extracts features from checkpoints.

Examples
--------
  python -m braintumor_ssl.visualize curves \
      --runs checkpoints/simsiam_r18_pretrain checkpoints/simsiam_r34_pretrain \
             checkpoints/simsiam_densenet121_pretrain \
      --labels resnet18 resnet34 densenet121 --out results/figures

  python -m braintumor_ssl.visualize roi --config configs/simsiam_r18_pretrain.yaml \
      --splits_file splits/splits_pretrain.json --split val --n 4 --out results/figures
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _plt():
    """Lazy, headless matplotlib (kept out of import time so the rest of the pkg is light)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# metric -> (nice label, "higher"/"lower"/"" is better)
_METRICS = [
    ("rankme", "RankMe (effective rank)", "higher"),
    ("val_loss", "val loss", "lower"),
    ("z_std", "z_std (collapse gate)", "higher"),
    ("participation_ratio", "participation ratio", "higher"),
    ("alignment", "alignment", "lower"),
    ("uniformity", "uniformity", "lower"),
]


# --------------------------------------------------------------------------- #
# Training-curve figures (from metrics.csv)
# --------------------------------------------------------------------------- #
def save_training_curves(history, path: str, healthy_z_std: float | None = None) -> None:
    """Single-run curves: loss (train+val), z_std (+healthy line), RankMe, participation,
    alignment, uniformity vs epoch. `history` is a list-of-dicts or a DataFrame."""
    plt = _plt()
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    if "epoch" not in df or df.empty:
        return
    ep = df["epoch"]
    # LiDAR / alpha-ReQ are selection metrics now, not extras — they get their own panels, and
    # alpha travels with its power-law fit quality because the exponent alone is not interpretable.
    panels = [
        ("loss", [c for c in ("train_loss", "val_loss") if c in df]),
        ("z_std", ["z_std"]),
        ("rankme", ["rankme"]),
        ("lidar", ["lidar"]),
        ("alpha_req", ["alpha_req"]),
        ("areq_r2 (power-law fit)", ["areq_r2"]),
        ("participation_ratio", ["participation_ratio"]),
        ("alignment", ["alignment"]),
        ("uniformity", ["uniformity"]),
    ]
    panels = [(t, cs) for t, cs in panels if any(c in df for c in cs)]
    nrow = 3 if len(panels) > 6 else 2
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 4 * nrow))
    for ax, (title, cols) in zip(axes.ravel(), panels):
        for c in cols:
            if c in df:
                ax.plot(ep, df[c], marker="o", ms=3, lw=1.4, label=c)
        if title == "z_std" and healthy_z_std is not None:
            ax.axhline(healthy_z_std, ls="--", c="gray", lw=1, label="healthy 1/√d")
        if title == "alpha_req":
            ax.axhline(1.0, ls="--", c="crimson", lw=1, label="alpha=1 target")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if len(cols) > 1 or title in ("z_std", "alpha_req"):
            ax.legend(fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("SimSiam SSL training", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# selection metric -> (label, score transform to "higher is better")
_SEL = [
    ("rankme", "RankMe", lambda v: v),
    ("lidar", "LiDAR", lambda v: v),
    ("alpha_req", "alpha-ReQ", lambda v: -np.abs(v - 1.0)),
]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def eligible_mask(df) -> np.ndarray:
    """Epochs that passed the run's convergence + non-collapse gate — the only ones any rule could
    have selected, and therefore the only ones worth normalizing or correlating over."""
    ok = np.ones(len(df), bool)
    for col, want in (("converged", True), ("collapsed", False)):
        if col in df:
            flag = df[col].astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
            ok &= (flag == want)
    return ok


def _minmax(v: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Scale to [0,1] using the min/max of the `ref` subset (values outside it may exceed [0,1])."""
    r = v if ref is None else v[ref]
    lo, hi = np.nanmin(r), np.nanmax(r)
    return np.full_like(v, 0.5) if not np.isfinite(hi - lo) or hi <= lo else (v - lo) / (hi - lo)


def save_metric_agreement(history, path: str, selected: dict | None = None,
                          title: str = "") -> None:
    """Do the three label-free metrics identify the same stage of training?

    Top: each metric's SELECTION SCORE (RankMe, LiDAR, -|alpha-1| — all "higher is better") min-max
    normalized onto one axis, with the epoch each rule actually picked marked, and the epochs that
    failed the convergence / collapse gate shaded out. This is the visual form of the question the
    per-run report answers in numbers: same epoch, or three different ones?

    Bottom: the three pairwise RANK scatters with Spearman rho. Ranks, not raw values, because rho
    is what is reported — plotting raw values would show a curve whose shape rho does not describe.
    Points are coloured by epoch so the direction of travel through training is visible.
    """
    plt = _plt()
    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    if "epoch" not in df or df.empty or not all(k in df for k, _, _ in _SEL):
        return
    ep = df["epoch"].to_numpy(float)
    scores = {k: fn(df[k].to_numpy(float)) for k, _, fn in _SEL}
    ok = eligible_mask(df)

    # Normalize and correlate over the ELIGIBLE epochs only. The random-init transient is enormous
    # (RankMe starts ~10x its converged value and crashes), so including it both squashes the region
    # where selection actually happens into a sliver of the axis AND inflates every rank correlation
    # with a shared "falling from random init" trend that says nothing about ordering the candidates.
    ref = ok if ok.sum() >= 3 else np.ones(len(df), bool)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 1, 1)
    for (k, lab, _), c in zip(_SEL, ("C0", "C1", "C2")):
        ax.plot(ep, _minmax(scores[k], ref), marker="o", ms=3, lw=1.5, color=c, label=f"{lab} score")
        if selected and k in selected and selected[k] is not None:
            ax.axvline(selected[k], color=c, ls="--", lw=1.6, alpha=0.8)
            ax.text(selected[k], 1.02, f"{lab}\ne{int(selected[k])}", color=c, fontsize=8,
                    ha="center", va="bottom")
    # shade the epochs no rule was allowed to select from
    if (~ok).any():
        for lo, hi in _runs_of(~ok, ep):
            ax.axvspan(lo, hi, color="gray", alpha=0.12, lw=0)
        ax.plot([], [], color="gray", alpha=0.35, lw=8, label="not eligible (gate)")
    ax.set_ylim(-0.08, 1.22)                 # scaled on eligible epochs; the init transient clips off
    ax.set_xlabel("epoch")
    ax.set_ylabel("selection score (min-max over eligible epochs)")
    ax.set_title("Which epoch does each label-free rule select?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    from scipy.stats import rankdata
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, (a, b) in enumerate(pairs):
        ka, la, _ = _SEL[a]
        kb, lb, _ = _SEL[b]
        axp = fig.add_subplot(2, 3, 4 + i)
        va, vb = scores[ka], scores[kb]
        m = np.isfinite(va) & np.isfinite(vb) & ref
        if m.sum() >= 3:
            sc = axp.scatter(rankdata(va[m]), rankdata(vb[m]), c=ep[m], cmap="viridis", s=22)
            rho = _spearman(va, vb)
            axp.set_title(f"{la} ~ {lb}   Spearman ρ={rho:+.2f}  (n={int(m.sum())} eligible)",
                          fontsize=10)
            if i == 2:
                fig.colorbar(sc, ax=axp, label="epoch")
        axp.set_xlabel(f"rank({la})")
        axp.set_ylabel(f"rank({lb})")
        axp.grid(alpha=0.3)

    fig.suptitle(title or "Label-free metric agreement", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _runs_of(mask: np.ndarray, x: np.ndarray):
    """Contiguous [x_start, x_end] spans where `mask` is True (for axvspan shading)."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = x[i]
        elif not m and start is not None:
            out.append((start, x[i]))
            start = None
    if start is not None:
        out.append((start, x[-1]))
    return out


def save_spearman_heatmap(run_dfs: dict, path: str) -> None:
    """One row per run, one column per metric pair: Spearman rho of the selection scores over epochs.

    The cross-run view of "are these metrics redundant or complementary?". A column that is dark red
    in every run means those two metrics carry the same ordering information and keeping both is
    reporting the same evidence twice; mixed signs across runs mean the relationship is not stable
    and neither metric can stand in for the other.
    """
    plt = _plt()
    labels = [f"{_SEL[a][1]}~{_SEL[b][1]}" for a, b in ((0, 1), (0, 2), (1, 2))]
    names, mat = [], []
    for name, df in run_dfs.items():
        if not all(k in df for k, _, _ in _SEL):
            continue
        s = {k: fn(df[k].to_numpy(float)) for k, _, fn in _SEL}
        ok = eligible_mask(df)                       # same restriction as the per-run figure
        if ok.sum() < 3:
            ok = np.ones(len(df), bool)
        s = {k: v[ok] for k, v in s.items()}
        names.append(name)
        mat.append([_spearman(s[_SEL[a][0]], s[_SEL[b][0]]) for a, b in ((0, 1), (0, 2), (1, 2))])
    if not mat:
        return
    M = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 4, 0.45 * len(names) + 2.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ over epochs")
    ax.set_title("Do the label-free metrics order the epochs the same way?", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_curves_comparison(run_dfs: dict, path: str) -> None:
    """Overlay each run's metric-vs-epoch curve (one line per run) in a 2x3 grid."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, label, better) in zip(axes.ravel(), _METRICS):
        for name, df in run_dfs.items():
            if key in df and "epoch" in df:
                ax.plot(df["epoch"], df[key], marker="o", ms=3, lw=1.4, label=name)
        ax.set_title(f"{label}" + (f"  (↑ better)" if better == "higher"
                                    else f"  (↓ better)" if better == "lower" else ""))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Backbone comparison — SSL training dynamics", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_final_bars(run_dfs: dict, path: str) -> None:
    """Grouped bar chart of each run's BEST value per metric (best RankMe row of each run)."""
    plt = _plt()
    names = list(run_dfs)
    # pick each run's best-RankMe row as its representative checkpoint
    best = {}
    for name, df in run_dfs.items():
        d = df.dropna(subset=["rankme"]) if "rankme" in df else df
        best[name] = d.loc[d["rankme"].idxmax()] if ("rankme" in d and len(d)) else (df.iloc[-1] if len(df) else None)
    keys = [k for k, _, _ in _METRICS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    x = np.arange(len(names))
    for ax, (key, label, better) in zip(axes.ravel(), _METRICS):
        vals = [float(best[n][key]) if (best[n] is not None and key in best[n]
                                        and pd.notna(best[n][key])) else np.nan for n in names]
        ax.bar(x, vals, color=[f"C{i}" for i in range(len(names))])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_title(f"{label}" + (f"  (↑)" if better == "higher"
                                   else f"  (↓)" if better == "lower" else ""))
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Backbone comparison — best-RankMe checkpoint metrics", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Qualitative ROI / augmentation figures (from the dataset)
# --------------------------------------------------------------------------- #
def _best_slice_and_bbox(vol: np.ndarray, margin: int = 6):
    """Pick the axial slice with the most non-zero voxels and a zoom bbox around the tumour."""
    nz = np.abs(vol).sum(0) > 0                      # (H, W, D) union over channels
    if not nz.any():
        z = vol.shape[-1] // 2
        return z, (slice(None), slice(None))
    z = int(nz.sum((0, 1)).argmax())
    sl = nz[:, :, z]
    ys, xs = np.where(sl)
    if len(ys) == 0:
        return z, (slice(None), slice(None))
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, sl.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, sl.shape[1])
    return z, (slice(y0, y1), slice(x0, x1))


def save_roi_montage(rois: dict, modalities, path: str) -> None:
    """Rows = subjects, cols = the 4 modalities; each cell is the best tumour slice (zoomed)."""
    plt = _plt()
    ids = list(rois)
    ncol = len(modalities)
    fig, axes = plt.subplots(len(ids), ncol, figsize=(3 * ncol, 3 * len(ids)), squeeze=False)
    for r, sid in enumerate(ids):
        vol = np.asarray(rois[sid])
        z, (sy, sx) = _best_slice_and_bbox(vol)
        for c, m in enumerate(modalities):
            ax = axes[r][c]
            ax.imshow(np.rot90(vol[c, sy, sx, z]), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(m, fontsize=11)
            if c == 0:
                ax.set_ylabel(sid, fontsize=8)
    fig.suptitle("Adaptive whole-tumour ROIs (best axial slice)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_aug_views(base, view1, view2, modalities, path: str, channel: int = 1) -> None:
    """Original ROI vs the two SimSiam augmented views (one modality) — shows the clinical aug."""
    plt = _plt()
    base, view1, view2 = (np.asarray(v) for v in (base, view1, view2))
    z, (sy, sx) = _best_slice_and_bbox(base)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (title, vol) in zip(axes, [("original", base), ("view 1", view1), ("view 2", view2)]):
        ax.imshow(np.rot90(vol[channel, sy, sx, z]), cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Clinical SSL augmentation — two views ({modalities[channel]})", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_run_metrics(run_dirs, labels):
    labels = labels or [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    out = {}
    for d, name in zip(run_dirs, labels):
        p = d if d.endswith(".csv") else os.path.join(d, "metrics.csv")
        if os.path.exists(p):
            out[name] = pd.read_csv(p)
        else:
            print(f"[warn] no metrics.csv for {name} ({p}) — skipped")
    return out


def _log_figs(args, paths) -> None:
    if not getattr(args, "mlflow", False):
        return
    from braintumor_ssl.tracking import Tracker
    tk = Tracker({"mlflow": True, "mlflow_experiment": args.mlflow_experiment},
                 run_name=f"visualize-{args.mode}")
    for p in paths:
        tk.log_artifact(p, artifact_path="figures")
    tk.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["curves", "roi"])
    ap.add_argument("--runs", nargs="+", help="curves: run dirs (each with metrics.csv) or csv paths")
    ap.add_argument("--labels", nargs="+", help="curves: legend labels (default: dir name)")
    ap.add_argument("--config", help="roi: training config (for modalities/roi/crop settings)")
    ap.add_argument("--splits_file")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=4, help="roi: #subjects in the montage")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--mlflow", action="store_true", help="also log the figures to MLflow/DagsHub")
    ap.add_argument("--mlflow_experiment", default="simsiam-brats")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "curves":
        if not args.runs:
            raise SystemExit("curves mode needs --runs <run_dir> [run_dir ...]")
        run_dfs = _load_run_metrics(args.runs, args.labels)
        if not run_dfs:
            raise SystemExit("no metrics.csv found in any --runs entry")
        c1 = os.path.join(args.out, "training_comparison.png")
        c2 = os.path.join(args.out, "final_metrics_bars.png")
        c3 = os.path.join(args.out, "metric_agreement_spearman.png")
        save_curves_comparison(run_dfs, c1)
        save_final_bars(run_dfs, c2)
        save_spearman_heatmap(run_dfs, c3)          # are the 3 label-free metrics redundant?
        figs = [c1, c2, c3]
        for name, df in run_dfs.items():            # per-run: which epoch did each rule pick?
            f = os.path.join(args.out, f"metric_agreement_{name}.png")
            save_metric_agreement(df, f, title=f"Label-free metric agreement — {name}")
            if os.path.exists(f):
                figs.append(f)
        _log_figs(args, figs)
        print(f"[done] wrote {len(figs)} figures to {args.out}")
        return

    # roi mode
    import yaml

    from braintumor_ssl.data import load_splits, make_views
    if not (args.config and args.splits_file):
        raise SystemExit("roi mode needs --config and --splits_file")
    cfg = yaml.safe_load(open(args.config))
    records = load_splits(args.splits_file)[args.split][: args.n]
    mods = list(cfg["modalities"])
    ds_eval = make_views(records, cfg, mode="eval")
    rois = {ds_eval[i]["id"]: ds_eval[i]["image"].numpy() for i in range(len(records))}
    montage = os.path.join(args.out, "roi_montage.png")
    aug = os.path.join(args.out, "augmentation_views.png")
    save_roi_montage(rois, mods, montage)

    ds_train = make_views(records[:1], cfg, mode="train")
    item = ds_train[0]
    save_aug_views(rois[records[0]["id"]], item["view1"].numpy(), item["view2"].numpy(), mods, aug)
    _log_figs(args, [montage, aug])
    print(f"[done] wrote roi_montage.png + augmentation_views.png to {args.out}")


if __name__ == "__main__":
    main()
