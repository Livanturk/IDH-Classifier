"""Adaptive WT-centered ROI size selection + coverage analysis.

Reproduces the team's Untitled17 notebook (stages 1-4) inside the repo: for every
subject it measures the whole-tumour (WT = seg>0) bounding box, then picks a
dataset-wide crop size from a percentile of the WT box dimensions (rounded up to a
multiple, min size, optionally cubic), and reports how many subjects that crop fully
covers. Use the printed "recommended roi_size" in your training config.

Examples
--------
  python -m braintumor_ssl.adaptive_crop --splits_file splits/splits.json --split train
  python -m braintumor_ssl.adaptive_crop --data_root BraTS2021/BraTS2021_TrainingSet \
      --percentile 95 --round 16 --min 64 --cube --out_dir results/adaptive_crop
"""
from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

from braintumor_ssl.data import load_splits, scan_subjects


def _round_up(value, multiple, minimum):
    return max(int(np.ceil(float(value) / multiple) * multiple), minimum)


def _crop_start(image_size, center, crop_size):
    """Clamp a crop of `crop_size` centred at `center` inside [0, image_size]."""
    start = int(center) - crop_size // 2
    if image_size < crop_size:
        return (image_size - crop_size) // 2       # allow negative -> padded
    return int(min(max(start, 0), image_size - crop_size))


def wt_geometry(seg: np.ndarray) -> dict | None:
    wt = seg > 0
    if not wt.any():
        return None
    coords = np.argwhere(wt)
    lo, hi = coords.min(0), coords.max(0)           # hi inclusive
    size = hi - lo + 1
    return {
        "shape": seg.shape,
        "wt_voxels": int(wt.sum()),
        "bbox_min": lo, "bbox_max_excl": hi + 1, "bbox_size": size,
        "bbox_center": np.round((lo + hi) / 2.0).astype(int),
        "wt_centroid": np.round(coords.mean(0)).astype(int),
    }


def scan_geometry(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        segp = os.path.join(r["dir"], f"{r['id']}_seg.nii.gz")
        if not os.path.exists(segp):
            rows.append({"patient_id": r["id"], "status": "missing_mask"})
            continue
        seg = np.asarray(nib.load(segp).dataobj)
        g = wt_geometry(seg)
        if g is None:
            rows.append({"patient_id": r["id"], "status": "empty_wt"})
            continue
        rows.append({
            "patient_id": r["id"], "status": "ok", "collection": r.get("collection", "?"),
            "wt_voxels": g["wt_voxels"],
            "shape_x": g["shape"][0], "shape_y": g["shape"][1], "shape_z": g["shape"][2],
            "bbox_w": int(g["bbox_size"][0]), "bbox_h": int(g["bbox_size"][1]), "bbox_d": int(g["bbox_size"][2]),
            "xmin": int(g["bbox_min"][0]), "ymin": int(g["bbox_min"][1]), "zmin": int(g["bbox_min"][2]),
            "xmax_excl": int(g["bbox_max_excl"][0]), "ymax_excl": int(g["bbox_max_excl"][1]),
            "zmax_excl": int(g["bbox_max_excl"][2]),
            "cx": int(g["wt_centroid"][0]), "cy": int(g["wt_centroid"][1]), "cz": int(g["wt_centroid"][2]),
            "bx": int(g["bbox_center"][0]), "by": int(g["bbox_center"][1]), "bz": int(g["bbox_center"][2]),
        })
    return pd.DataFrame(rows)


def adaptive_size(df, percentile, multiple, minimum, cube):
    ok = df[df["status"] == "ok"]
    ps = [np.percentile(ok[c].astype(float), percentile) for c in ("bbox_w", "bbox_h", "bbox_d")]
    rounded = [_round_up(p, multiple, minimum) for p in ps]
    if cube:
        rounded = [max(rounded)] * 3
    return tuple(rounded), ps


def coverage(df, crop_size, center_mode):
    """Fraction of subjects whose full WT bbox fits inside the (clamped) crop."""
    ok = df[df["status"] == "ok"]
    n_full = 0
    for _, r in ok.iterrows():
        shape = [r["shape_x"], r["shape_y"], r["shape_z"]]
        cen = ([r["cx"], r["cy"], r["cz"]] if center_mode == "wt_centroid" else [r["bx"], r["by"], r["bz"]])
        start = [_crop_start(shape[d], cen[d], crop_size[d]) for d in range(3)]
        end = [start[d] + crop_size[d] for d in range(3)]
        bmin = [r["xmin"], r["ymin"], r["zmin"]]
        bmax = [r["xmax_excl"], r["ymax_excl"], r["zmax_excl"]]
        if all(bmin[d] >= start[d] and bmax[d] <= end[d] for d in range(3)):
            n_full += 1
    n = len(ok)
    return 100.0 * n_full / max(n, 1), n_full, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_file")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--data_root")
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--round", type=int, default=16)
    ap.add_argument("--min", type=int, default=64)
    ap.add_argument("--cube", action="store_true", default=True)
    ap.add_argument("--no_cube", dest="cube", action="store_false")
    ap.add_argument("--center_mode", default="wt_centroid", choices=["wt_centroid", "bbox_center"])
    ap.add_argument("--out_dir", default="results/adaptive_crop")
    args = ap.parse_args()

    if args.splits_file:
        records = load_splits(args.splits_file)[args.split]
    elif args.data_root:
        records = scan_subjects(args.data_root)
    else:
        raise SystemExit("provide --splits_file or --data_root")

    os.makedirs(args.out_dir, exist_ok=True)
    df = scan_geometry(records)
    df.to_csv(os.path.join(args.out_dir, "01_wt_geometry.csv"), index=False)
    n_ok = int((df["status"] == "ok").sum())
    print(f"[geometry] {len(df)} subjects, {n_ok} with a valid WT mask")

    size, ps = adaptive_size(df, args.percentile, args.round, args.min, args.cube)
    cov_pct, n_full, n = coverage(df, size, args.center_mode)
    print(f"[adaptive] p{args.percentile:g} WT bbox (w,h,d)={tuple(round(p,1) for p in ps)} "
          f"-> round{args.round}/min{args.min}/{'cube' if args.cube else 'axiswise'} "
          f"=> roi_size={list(size)}")
    print(f"[coverage] roi={list(size)} centred on {args.center_mode} "
          f"fully covers WT in {n_full}/{n} ({cov_pct:.1f}%)")

    # candidate table
    rows = []
    for p in (90, 95, 99, 100):
        s, _ = adaptive_size(df, p, args.round, args.min, args.cube)
        c, nf, nn = coverage(df, s, args.center_mode)
        rows.append({"candidate": f"adaptive_p{p}", "roi": "x".join(map(str, s)),
                     "coverage_%": round(c, 1), "covered": nf, "n": nn})
    for c in (64, 80, 96, 112, 128, 144, 160):
        cov, nf, nn = coverage(df, (c, c, c), args.center_mode)
        rows.append({"candidate": f"fixed_{c}", "roi": f"{c}x{c}x{c}",
                     "coverage_%": round(cov, 1), "covered": nf, "n": nn})
    cand = pd.DataFrame(rows)
    cand.to_csv(os.path.join(args.out_dir, "02_candidate_coverage.csv"), index=False)
    print("\n=== candidate crop coverage (center_mode=%s) ===" % args.center_mode)
    print(cand.to_string(index=False))
    print(f"\n[done] reports in {args.out_dir}  |  set roi_size: {list(size)} in your config")


if __name__ == "__main__":
    main()
