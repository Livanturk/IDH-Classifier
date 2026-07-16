"""Extract 256-d SimSiam features for downstream IDH classification.

Loads a pretrained checkpoint, runs the frozen encoder over every subject, and
writes one row per subject: id + f000..f255  ->  CSV (Hasta x 256 Features).

Examples
--------
  # features for the BraTS split used in pretraining
  python -m braintumor_ssl.extract_features --checkpoint checkpoints/simsiam_brats/last.pth \
      --splits_file splits/splits.json --split train --out features/brats_train.csv

  # features for a downstream cohort (e.g. UPenn) — point --data_root at it, use --all
  python -m braintumor_ssl.extract_features --checkpoint checkpoints/simsiam_brats/last.pth \
      --data_root path/to/UPenn --all --out features/upenn.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from braintumor_ssl.data import load_splits, make_views, scan_subjects
from braintumor_ssl.models import SimSiam
from braintumor_ssl.utils import recompute_bn_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    # subject source: either a splits file (+ which split) or a raw data_root scan
    ap.add_argument("--splits_file")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--data_root")
    ap.add_argument("--all", action="store_true", help="with --data_root, use every discovered subject")
    ap.add_argument("--roi_size", nargs=3, type=int, default=None,
                    help="defaults to the roi the checkpoint was trained with")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tta_crops", type=int, default=0,
                    help="if >0, average the encoder output over N random crops instead of a single resize")
    ap.add_argument("--recompute_bn", dest="recompute_bn", action="store_true", default=True,
                    help="precise-BN over this cohort before extraction (on by default; adapts BN to the cohort)")
    ap.add_argument("--no_recompute_bn", dest="recompute_bn", action="store_false",
                    help="use the checkpoint's stored BatchNorm running stats as-is")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck["cfg"]
    if args.roi_size:                       # optional override -> make_views reads cfg["roi_size"]
        cfg["roi_size"] = list(args.roi_size)
    roi = tuple(cfg["roi_size"])

    model = SimSiam(
        backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
        feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
        proj_hidden_dim=cfg["proj_hidden_dim"], pred_hidden_dim=cfg["pred_hidden_dim"],
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[model] loaded {args.checkpoint} (epoch {ck['epoch']}) feature_dim={model.feature_dim}")

    # subject list
    if args.splits_file:
        records = load_splits(args.splits_file)[args.split]
    elif args.data_root:
        records = scan_subjects(args.data_root)
    else:
        raise SystemExit("provide either --splits_file or --data_root")
    print(f"[data] {len(records)} subjects, roi={roi}, tta_crops={args.tta_crops}")

    mode = "train" if args.tta_crops > 0 else "eval"   # 'train' mode yields fresh random crops each pass
    n_passes = args.tta_crops if args.tta_crops > 0 else 1
    ds = make_views(records, cfg, mode=mode)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.recompute_bn:
        # deterministic eval-view loader to re-estimate BatchNorm stats on this cohort
        bn_ds = make_views(records, cfg, mode="eval")
        bn_ld = DataLoader(bn_ds, batch_size=max(args.batch_size, 2), shuffle=True,
                           drop_last=True, num_workers=args.num_workers)
        nb = recompute_bn_stats(model, bn_ld, device, max_batches=200)
        print(f"[bn] recomputed BatchNorm running stats over {nb} batches")
    model.eval()

    sums: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for _ in range(n_passes):                     # each pass = one fresh crop per subject (or the fixed resize)
            for batch in ld:
                x = (batch["image"] if mode == "eval" else batch["view1"]).to(device)
                feats = model.encode(x).cpu()
                for sid, vec in zip(batch["id"], feats):
                    sums[sid] = vec if sid not in sums else sums[sid] + vec

    rows = [{"id": sid, **{f"f{j:03d}": float(v) for j, v in enumerate(vec / n_passes)}}
            for sid, vec in sums.items()]
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}  shape={df.shape} (subjects x [id + {model.feature_dim} feats])")


if __name__ == "__main__":
    main()
