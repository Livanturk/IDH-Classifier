"""Compare SSL checkpoints with LABEL-FREE representation-quality metrics.

For each checkpoint it (1) re-estimates BatchNorm on the eval cohort, (2) extracts
256-d features for two augmented views per subject, (3) computes collapse / RankMe /
alignment / uniformity / participation-ratio, appends a row to a leaderboard CSV, and
optionally saves a 2-D (PCA->t-SNE) embedding plot coloured by source collection.

This is the "which encoder is better?" tool for the current (unlabeled) stage. When an
IDH label table becomes available, add a linear-probe/k-NN column as the decisive metric
(see README "How to compare configs").

Examples
--------
  # one config
  python -m braintumor_ssl.evaluate --checkpoints checkpoints/simsiam_r18_all/last.pth \
      --splits_file splits/splits.json --split val --embed tsne

  # compare the whole ablation ladder in one leaderboard
  python -m braintumor_ssl.evaluate \
      --checkpoints checkpoints/simsiam_r18_all/last.pth checkpoints/simsiam_r50_all/last.pth \
                    checkpoints/simsiam_r18_noupenn/last.pth checkpoints/simsiam_r50_noupenn/last.pth \
      --splits_file splits/splits.json --split val --n_subjects 63 --embed tsne
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
from braintumor_ssl.utils import (
    alignment,
    jackknife_ci,
    participation_ratio,
    rankme,
    recompute_bn_stats,
    representation_std,
    set_seed,
    uniformity,
    use_local_tmpdir,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="checkpoint files (or dirs containing last.pth)")
    ap.add_argument("--splits_file")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--data_root", help="alternative to --splits_file: scan a cohort directly")
    ap.add_argument("--n_subjects", type=int, default=0, help="cap #subjects (0=all); fixed set across configs")
    ap.add_argument("--out", default="results/leaderboard.csv")
    ap.add_argument("--embed", choices=["none", "pca", "tsne"], default="none")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_ci", action="store_true",
                    help="skip the jackknife 95%% CI on rankme/PR/uniformity (faster)")
    ap.add_argument("--mlflow", action="store_true", help="log leaderboard + figures to MLflow/DagsHub")
    ap.add_argument("--mlflow_experiment", default="simsiam-brats-eval")
    return ap.parse_args()


def resolve_ckpt(p: str) -> str:
    return os.path.join(p, "last.pth") if os.path.isdir(p) else p


@torch.no_grad()
def features_two_views(model, records, cfg, roi, device, bs, nw):
    """Return (H1, H2, ids, collections) — two augmented-view embeddings per subject."""
    ds = make_views(records, cfg, mode="train")
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    H1, H2, ids = [], [], []
    for batch in ld:
        H1.append(model.encode(batch["view1"].to(device)).cpu())
        H2.append(model.encode(batch["view2"].to(device)).cpu())
        ids.extend(batch["id"])
    id2col = {r["id"]: r.get("collection", "?") for r in records}
    cols = [id2col[i] for i in ids]
    return torch.cat(H1), torch.cat(H2), ids, cols


def save_embedding(H: np.ndarray, cols: list[str], path: str, method: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if method == "tsne":
        from sklearn.manifold import TSNE
        Z = PCA(n_components=min(50, H.shape[1], H.shape[0])).fit_transform(H)
        Z = TSNE(n_components=2, perplexity=min(30, max(5, len(H) // 4)),
                 init="pca", random_state=0).fit_transform(Z)
    else:
        Z = PCA(n_components=2).fit_transform(H)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(set(cols)):
        m = [i for i, cc in enumerate(cols) if cc == c]
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.7, label=c)
    ax.set_title(os.path.basename(path)); ax.legend(fontsize=6, markerscale=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_spectrum(spectra: dict, path: str) -> None:
    """Overlaid feature singular-value spectra (normalized, log-y). A flat / slowly-decaying
    spectrum = variance spread over many directions (high effective rank); a steep cliff = a
    few dominant directions (dimensional collapse). Visual companion to RankMe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, H in spectra.items():
        X = H.float()
        X = X - X.mean(0, keepdim=True)
        try:
            s = torch.linalg.svdvals(X)
        except Exception:
            s = torch.linalg.svdvals(X.cpu())
        s = (s / s.max()).cpu().numpy()
        ax.plot(range(1, len(s) + 1), s, lw=1.6, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel("normalized σ (log)")
    ax.set_title("Feature singular-value spectrum (flatter = higher rank)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_local_tmpdir()                       # avoid the NFS DataLoader-cleanup noise on the cluster
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.splits_file:
        records = load_splits(args.splits_file)[args.split]
    elif args.data_root:
        records = scan_subjects(args.data_root)
    else:
        raise SystemExit("provide --splits_file or --data_root")
    if args.n_subjects:
        records = records[: args.n_subjects]
    print(f"[eval] {len(records)} subjects | device={device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    spectra: dict[str, torch.Tensor] = {}
    for ckpt_arg in args.checkpoints:
        ckpt = resolve_ckpt(ckpt_arg)
        ck = torch.load(ckpt, map_location=device)
        cfg = ck["cfg"]
        roi = tuple(cfg["roi_size"])
        name = os.path.basename(os.path.dirname(ckpt)) or os.path.basename(ckpt)

        model = SimSiam(backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
                        feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
                        proj_hidden_dim=cfg["proj_hidden_dim"],
                        pred_hidden_dim=cfg["pred_hidden_dim"]).to(device)
        model.load_state_dict(ck["model"])

        bn_ds = make_views(records, cfg, mode="eval")
        bn_ld = DataLoader(bn_ds, batch_size=max(args.batch_size, 2), shuffle=True,
                           drop_last=True, num_workers=args.num_workers)
        recompute_bn_stats(model, bn_ld, device, max_batches=200)
        model.eval()

        H1, H2, ids, cols = features_two_views(model, records, cfg, roi, device,
                                               args.batch_size, args.num_workers)
        spectra[name] = H1
        row = {
            "config": name,
            "backbone": cfg["backbone"],
            "epoch": ck["epoch"],
            "roi": "x".join(map(str, roi)),
            "splits": os.path.basename(cfg.get("splits_file", "?")),
            "n": len(ids),
            "z_std": round(representation_std(H1), 4),          # collapse gate (want high)
            "rankme": round(rankme(H1), 2),                     # want HIGH
            "alignment": round(alignment(H1, H2), 4),           # want LOW (if not collapsed)
            "uniformity": round(uniformity(H1), 4),             # want NEGATIVE
            "particip_ratio": round(participation_ratio(H1), 2),  # want HIGH
        }
        # jackknife 95% CIs over subjects for the ranking/tie-break metrics (D2 uses rankme CI)
        if not args.no_ci:
            rl, rh = jackknife_ci(rankme, H1)
            pl, ph = jackknife_ci(participation_ratio, H1)
            ul, uh = jackknife_ci(uniformity, H1)
            row.update(rankme_ci_lo=round(rl, 2), rankme_ci_hi=round(rh, 2),
                       pr_ci_lo=round(pl, 2), pr_ci_hi=round(ph, 2),
                       unif_ci_lo=round(ul, 4), unif_ci_hi=round(uh, 4))
        rows.append(row)
        ci = f" [{row['rankme_ci_lo']:.2f},{row['rankme_ci_hi']:.2f}]" if not args.no_ci else ""
        print(f"  {name:24s} rankme={row['rankme']:6.2f}{ci} align={row['alignment']:.4f} "
              f"unif={row['uniformity']:+.3f} pr={row['particip_ratio']:6.2f} z_std={row['z_std']:.4f}")

        # persist per-config features + optional 2-D embedding
        feat_dir = os.path.join(os.path.dirname(args.out) or ".", "features")
        os.makedirs(feat_dir, exist_ok=True)
        pd.DataFrame({"id": ids, "collection": cols,
                      **{f"f{j:03d}": H1[:, j].numpy() for j in range(H1.shape[1])}}
                     ).to_csv(os.path.join(feat_dir, f"{name}.csv"), index=False)
        if args.embed != "none":
            save_embedding(H1.numpy(), cols,
                           os.path.join(os.path.dirname(args.out) or ".", f"{name}_{args.embed}.png"),
                           args.embed)

    df = pd.DataFrame(rows)
    # append to leaderboard (keep history), then show current comparison sorted by rankme
    if os.path.exists(args.out):
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    df.to_csv(args.out, index=False)
    if spectra:
        spec_path = os.path.join(os.path.dirname(args.out) or ".", "singular_value_spectrum.png")
        save_spectrum(spectra, spec_path)
        print(f"[plot] {spec_path}")
    print("\n=== leaderboard (higher rankme / participation, more-negative uniformity = better) ===")
    print(df.sort_values("rankme", ascending=False).to_string(index=False))

    if args.mlflow:
        from braintumor_ssl.tracking import Tracker

        result_dir = os.path.dirname(args.out) or "."
        metric_keys = ("z_std", "rankme", "alignment", "uniformity", "particip_ratio")
        ci_keys = ("rankme_ci_lo", "rankme_ci_hi", "pr_ci_lo", "pr_ci_hi", "unif_ci_lo", "unif_ci_hi")
        # One MLflow run per checkpoint makes DagsHub's run comparison view useful.
        for row in rows:
            tk = Tracker(
                {"mlflow": True, "mlflow_experiment": args.mlflow_experiment},
                run_name=f"evaluate-{row['config']}",
                tags={"stage": "evaluation", "backbone": row["backbone"]},
            )
            tk.log_params({k: v for k, v in row.items() if k not in metric_keys and k not in ci_keys})
            tk.log_metrics({k: row[k] for k in metric_keys}, step=int(row["epoch"]))
            # jackknife CIs are numeric — log as metrics (not params) so DagsHub shows them with rankme
            tk.log_metrics({k: row[k] for k in ci_keys if k in row}, step=int(row["epoch"]))
            tk.log_artifact(args.out, artifact_path="tables")
            tk.log_artifact(os.path.join(result_dir, f"{row['config']}_{args.embed}.png"),
                            artifact_path="figures")
            tk.log_artifact(os.path.join(result_dir, "singular_value_spectrum.png"),
                            artifact_path="figures")
            tk.close()

    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
