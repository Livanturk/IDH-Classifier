"""Scan BraTS 2021 and write a subject-level train/val split JSON.

Examples
--------
  python -m braintumor_ssl.make_splits --data_root BraTS2021/BraTS2021_TrainingSet --out splits/splits.json
  python -m braintumor_ssl.make_splits --limit 12 --out splits/smoke.json          # tiny split for smoke test
  python -m braintumor_ssl.make_splits --exclude_collections UPENN-GBM             # hold out downstream cohort
"""
from __future__ import annotations

import argparse
import json
import os

from braintumor_ssl.data import build_splits, scan_subjects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="BraTS2021/BraTS2021_TrainingSet")
    ap.add_argument("--out", default="splits/splits.json")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_collections", nargs="*", default=None,
                    help="collections to drop entirely, e.g. UPENN-GBM to avoid downstream leakage")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N subjects (smoke test)")
    args = ap.parse_args()

    records = scan_subjects(args.data_root)
    print(f"discovered {len(records)} subjects with all 4 modalities under {args.data_root}")
    by_col: dict[str, int] = {}
    for r in records:
        by_col[r["collection"]] = by_col.get(r["collection"], 0) + 1
    for c, n in sorted(by_col.items()):
        print(f"  {c:28s} {n}")

    # Drop excluded collections BEFORE --limit so `--limit N --exclude_collections X` yields N
    # subjects that are all non-X (build_splits re-applies the same exclusion; it is idempotent).
    if args.exclude_collections:
        ex = set(args.exclude_collections)
        before = len(records)
        records = [r for r in records if r["collection"] not in ex]
        print(f"excluded {before - len(records)} subjects from collections {sorted(ex)}")

    if args.limit:
        records = records[: args.limit]
        print(f"limited to first {len(records)} subjects (smoke test)")

    split = build_splits(records, val_frac=args.val_frac, seed=args.seed,
                         exclude_collections=args.exclude_collections)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"wrote {args.out}: train={split['n_train']} val={split['n_val']} "
          f"(excluded={split['excluded_collections']})")


if __name__ == "__main__":
    main()
