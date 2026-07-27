"""Aggregate per-run checkpoint selections into a policy x configuration table (mean +- SD).

`report_run.py` writes ONE row per run. That is the wrong unit for a claim about a backbone: the
best seed of three is a cherry-pick, and the spread across seeds is usually larger than the gap
between configurations. This module groups those rows by (configuration, selection policy) and
reports the mean +- SD ACROSS SEEDS, so every number in the paper's table is a family of runs rather
than a single lucky one.

Two tables come out of it:

  A. policy x metric   — for each configuration and each of the three selection policies
                         (RankMe-selected / LiDAR-selected / alpha-ReQ-selected checkpoints), the
                         mean +- SD of all three metrics AT that checkpoint. Reading down a column
                         answers "does the LiDAR-selected checkpoint also score well on RankMe?".
  B. selected epoch    — mean +- SD of the epoch each policy picked, plus how far apart the three
                         policies land. A large SD means the policy is not reproducible across
                         seeds and should not be presented as a selection RULE.

Configuration key = (backbone, split, and the config-directory stem with any trailing _seedNN /
_sNN removed), so it generalizes over backbones, splits and seeds without a hard-coded list.

Usage
-----
  python -m braintumor_ssl.aggregate_runs                                   # read the default summary CSV
  python -m braintumor_ssl.aggregate_runs --summary results/checkpoint_selection_summary.csv
  python -m braintumor_ssl.aggregate_runs --out results/checkpoint_selection_by_policy.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics

DEFAULT_SUMMARY = "results/checkpoint_selection_summary.csv"
DEFAULT_OUT = "results/checkpoint_selection_by_policy.csv"

# selection policy -> (column prefix used by report_run, pretty name)
POLICIES = {"rankme": ("RankMe", "RankMe-selected"),
            "lidar": ("LiDAR", "LiDAR-selected"),
            "alpha_req": ("alphaReQ", "alpha-ReQ-selected")}
METRIC_KEYS = ("rankme", "lidar", "alpha_req")
METRIC_NAMES = {"rankme": "RankMe", "lidar": "LiDAR", "alpha_req": "alpha-ReQ"}

_SEED_SUFFIX = re.compile(r"_(?:seed|s)\d+$")


def config_key(row: dict) -> str:
    """Configuration identity with the seed stripped — runs that differ ONLY by seed group here."""
    stem = _SEED_SUFFIX.sub("", str(row.get("config", "")))
    return f"{stem} | {row.get('backbone', '?')} | {row.get('split', '?')}"


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def mean_sd(vals: list[float]) -> tuple[float, float, int]:
    """Mean, sample SD and n over the finite values (SD is nan for a single run — say so, don't 0)."""
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return math.nan, math.nan, 0
    sd = statistics.stdev(xs) if len(xs) > 1 else math.nan
    return statistics.fmean(xs), sd, len(xs)


def _short(key: str, width: int = 42) -> str:
    """Configuration key trimmed to fit the console table (the CSV keeps the full key)."""
    return key if len(key) <= width else "…" + key[-(width - 1):]


def _fmt(m: float, sd: float, n: int, prec: int = 3) -> str:
    if n == 0:
        return "—"
    if n == 1 or not math.isfinite(sd):
        return f"{m:.{prec}f} (n=1)"
    return f"{m:.{prec}f}±{sd:.{prec}f}"


def aggregate(rows: list[dict]) -> list[dict]:
    """One output row per (configuration, policy), averaged over the seeds in that group."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(config_key(r), []).append(r)

    out = []
    for key in sorted(groups):
        rs = groups[key]
        seeds = sorted({str(r.get("seed", "?")) for r in rs})
        for pol, (pre, pretty) in POLICIES.items():
            rec = {"configuration": key, "policy": pretty, "n_seeds": len(rs),
                   "seeds": "/".join(seeds)}
            e_m, e_sd, e_n = mean_sd([_f(r.get(f"{pre}_epoch")) for r in rs])
            rec["epoch_mean"], rec["epoch_sd"], rec["n_selected"] = e_m, e_sd, e_n
            for m in METRIC_KEYS:
                mm, ms, mn = mean_sd([_f(r.get(f"{pre}_at_{m}")) for r in rs])
                rec[f"{m}_mean"], rec[f"{m}_sd"], rec[f"{m}_n"] = mm, ms, mn
            v_m, v_sd, _ = mean_sd([_f(r.get(f"{pre}_val_loss")) for r in rs])
            rec["val_loss_mean"], rec["val_loss_sd"] = v_m, v_sd
            r2_m, r2_sd, _ = mean_sd([_f(r.get(f"{pre}_areq_r2")) for r in rs])
            rec["areq_r2_mean"], rec["areq_r2_sd"] = r2_m, r2_sd
            out.append(rec)
    return out


def print_tables(agg: list[dict]) -> None:
    configs = sorted({a["configuration"] for a in agg})

    print("=" * 100)
    print("A. Metrics AT each selected checkpoint, mean±SD over seeds")
    print("=" * 100)
    print(f"{'configuration':<44}{'policy':<22}{'n':>3}  "
          + "".join(f"{METRIC_NAMES[m]:>18}" for m in METRIC_KEYS))
    for c in configs:
        first = True
        for a in [x for x in agg if x["configuration"] == c]:
            label = _short(c) if first else ""
            first = False
            cells = "".join(f"{_fmt(a[f'{m}_mean'], a[f'{m}_sd'], a[f'{m}_n']):>18}"
                            for m in METRIC_KEYS)
            print(f"{label:<44}{a['policy']:<22}{a['n_selected']:>3}  {cells}")
        print()

    print("=" * 100)
    print("B. Which epoch each policy picked, mean±SD over seeds")
    print("=" * 100)
    print(f"{'configuration':<44}{'policy':<22}{'epoch':>16}{'n_seeds':>9}")
    for c in configs:
        rows = [x for x in agg if x["configuration"] == c]
        first = True
        for a in rows:
            print(f"{(_short(c) if first else ''):<44}{a['policy']:<22}"
                  f"{_fmt(a['epoch_mean'], a['epoch_sd'], a['n_selected'], 1):>16}"
                  f"{a['n_seeds']:>9}")
            first = False
        # spread BETWEEN policies (on the seed means) — small spread = the three rules agree
        es = [a["epoch_mean"] for a in rows if math.isfinite(a["epoch_mean"])]
        if len(es) >= 2:
            sds = [a["epoch_sd"] for a in rows if math.isfinite(a["epoch_sd"])]
            within = statistics.fmean(sds) if sds else math.nan
            spread = max(es) - min(es)
            note = ("the policies disagree by more than the seed noise — a real disagreement"
                    if math.isfinite(within) and spread > 2 * within else
                    "policy differences are within seed noise — do NOT rank the policies on this")
            print(f"{'':<44}{'-> between-policy spread':<22}{spread:>16.1f}   ({note})")
        print()


def write_csv(agg: list[dict], path: str) -> None:
    if not agg:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"wrote {path}  ({len(agg)} configuration x policy rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=DEFAULT_SUMMARY,
                    help="per-run CSV written by report_run.py --summary")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        raise SystemExit(f"{args.summary} not found — run report_run.py over the run dirs first.")
    with open(args.summary) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.summary} is empty.")

    agg = aggregate(rows)
    print_tables(agg)
    write_csv(agg, args.out)
    n_single = sum(1 for a in agg if a["n_seeds"] < 2) // max(1, len(POLICIES))
    if n_single:
        print(f"\nNOTE: {n_single} configuration(s) have a single seed — their SD is undefined and "
              "they cannot support a comparison claim. Run >=3 seeds before reporting.")


if __name__ == "__main__":
    main()
