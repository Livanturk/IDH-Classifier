"""Build the bildiri's result tables and figures from everything the runs left on disk.

One command that turns the per-run artifacts into the two tables the paper actually reports, and
tells you plainly which claims the numbers do and do not support:

  Table 1 (label-free, no labels used)   backbone x selection policy -> the epoch each rule picked
                                          and the three metrics at that checkpoint, mean +- SD over
                                          seeds. Comes from report_run's per-run summary.

  Table 2 (downstream, UCSF-PDGM IDH)    backbone x selection policy -> AUC, accuracy, F1,
                                          sensitivity, specificity, precision, balanced accuracy,
                                          mean +- SD over seeds, WITH the clinical-only baselines
                                          on the same rows scale. Comes from idh_probe.

Everything is aggregated over seeds, never over the best seed: with three seeds per configuration
the spread between seeds is routinely larger than the gap between configurations, so a single-seed
number is not evidence. Configurations with one seed are printed with "(n=1)" instead of a fake SD.

The figures (metric-agreement per run, Spearman heatmap, training-curve overlay) are produced by
`visualize.py curves`, which this also invokes so a single call refreshes the whole set.

Usage
-----
  python -m braintumor_ssl.paper_tables
  python -m braintumor_ssl.paper_tables --runs checkpoints/simsiam_*_unified_seed* --out results/paper
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import subprocess
import sys

# downstream metric -> (column header, decimals)
DOWNSTREAM = [("auc", "AUC", 3), ("acc", "Acc", 3), ("f1", "F1", 3), ("sens", "Sens", 3),
              ("spec", "Spec", 3), ("prec", "Prec", 3), ("balanced_acc", "BalAcc", 3)]
# The comparison ladder, in table order (methodology §8.2). EVERY checkpoint the probe can emit is
# listed: anything missing here falls through to its raw filename and to the end of the table, which
# is how `bestSPS.pth` and the ungated baselines ended up printed as filenames BELOW the rows they
# are supposed to precede.
#
# `last.pth` is deliberately NOT called "final/plateau" any more. It is the last training epoch —
# i.e. the result of making no selection at all — and naming it after the plateau invited exactly
# the confusion that a genuine plateau rule (SPS) now sits next to it in the same table.
POLICY_ORDER = [
    "randominit",
    "bestNaiveRankMe.pth", "bestNaiveLiDAR.pth", "bestNaiveA-ReQ.pth",
    "bestRankMe.pth", "bestLiDAR.pth", "bestA-ReQ.pth",
    "bestSPS.pth",
    "bestPareto.pth",
    "last.pth",
]
POLICY_NAME = {
    "randominit": "random-init (no pretraining)",
    "bestNaiveRankMe.pth": "ungated argmax — RankMe",
    "bestNaiveLiDAR.pth": "ungated argmax — LiDAR",
    "bestNaiveA-ReQ.pth": "ungated argmax — AlphaReQ",
    "bestRankMe.pth": "RankMe-selected",
    "bestLiDAR.pth": "LiDAR-selected",
    "bestA-ReQ.pth": "AlphaReQ-selected",
    "bestSPS.pth": "Proposed (SPS)",
    "bestPareto.pth": "Pareto (ablation)",
    "last.pth": "last epoch (no selection)",
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _agg(vals, prec=3):
    xs = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.{prec}f} (n=1)"
    return f"{statistics.fmean(xs):.{prec}f}±{statistics.stdev(xs):.{prec}f}"


def read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                rows.extend(list(csv.DictReader(f)))
    return rows


# --------------------------------------------------------------------------- #
def table_labelfree(summary_csv: str, out_csv: str) -> None:
    rows = read_rows([summary_csv])
    if not rows:
        print(f"[table 1] {summary_csv} missing/empty — run report_run.py over the finished runs first.")
        return
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("backbone", "?"),), []).append(r)

    print("\n" + "=" * 104)
    print("TABLE 1 — label-free selection (no labels used).  mean±SD over seeds")
    print("=" * 104)
    hdr = f"{'backbone':<14}{'policy':<20}{'epoch':>14}{'RankMe':>16}{'LiDAR':>16}{'alpha-ReQ':>16}{'seeds':>7}"
    print(hdr)
    out = []
    for (backbone,), rs in sorted(groups.items()):
        for pre, pretty in (("RankMe", "RankMe-selected"), ("LiDAR", "LiDAR-selected"),
                            ("alphaReQ", "alpha-ReQ-selected")):
            cells = {m: _agg([_f(r.get(f"{pre}_at_{m}")) for r in rs]) for m in
                     ("rankme", "lidar", "alpha_req")}
            ep = _agg([_f(r.get(f"{pre}_epoch")) for r in rs], 1)
            print(f"{backbone:<14}{pretty:<20}{ep:>14}{cells['rankme']:>16}{cells['lidar']:>16}"
                  f"{cells['alpha_req']:>16}{len(rs):>7}")
            out.append({"backbone": backbone, "policy": pretty, "epoch": ep,
                        "rankme": cells["rankme"], "lidar": cells["lidar"],
                        "alpha_req": cells["alpha_req"], "n_seeds": len(rs)})
        print()
    _write(out, out_csv)


def table_downstream(probe_csvs: list[str], out_csv: str) -> None:
    rows = read_rows(probe_csvs)
    if not rows:
        print("[table 2] no results/idh_probe*.csv yet — the downstream jobs have not run.")
        return

    clinical = [r for r in rows if r.get("checkpoint") in ("-", "", None)]
    imaging = [r for r in rows if r.get("checkpoint") not in ("-", "", None)]

    print("\n" + "=" * 116)
    print("TABLE 2 — downstream IDH (UCSF-PDGM, frozen linear probe).  mean±SD over seeds")
    print("=" * 116)
    print(f"{'backbone':<13}{'policy':<29}{'arm':<11}"
          + "".join(f"{h:>13}" for _, h, _ in DOWNSTREAM) + f"{'seeds':>7}")

    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in imaging:
        groups.setdefault((r.get("backbone", "?"), r.get("checkpoint", "?"), r.get("arm", "?")), []).append(r)
    for key in sorted(groups, key=lambda k: (k[0], POLICY_ORDER.index(k[1]) if k[1] in POLICY_ORDER
                                             else 99, k[2])):
        backbone, ckpt, arm = key
        rs = groups[key]
        cells = {m: _agg([_f(r.get(m)) for r in rs], d) for m, _, d in DOWNSTREAM}
        print(f"{backbone:<13}{POLICY_NAME.get(ckpt, ckpt):<29}{arm:<11}"
              + "".join(f"{cells[m]:>13}" for m, _, _ in DOWNSTREAM) + f"{len(rs):>7}")
        # `ladder` travels with the row so the rung survives re-sorting: opening the CSV in a
        # spreadsheet and sorting by any column would otherwise scramble the comparison order.
        out.append({"ladder": POLICY_ORDER.index(ckpt) if ckpt in POLICY_ORDER else 99,
                    "backbone": backbone, "policy": POLICY_NAME.get(ckpt, ckpt), "arm": arm,
                    **{m: cells[m] for m, _, _ in DOWNSTREAM}, "n_seeds": len(rs)})

    if clinical:
        print("-" * 116)
        seen = set()
        for r in clinical:
            arm = r.get("arm")
            if arm in seen:
                continue
            seen.add(arm)
            cells = {m: f"{_f(r.get(m)):.{d}f}" for m, _, d in DOWNSTREAM}
            print(f"{'— clinical —':<13}{'(no imaging)':<29}{arm:<11}"
                  + "".join(f"{cells[m]:>13}" for m, _, _ in DOWNSTREAM) + f"{'1':>7}")
            out.append({"ladder": 99, "backbone": "clinical", "policy": "(no imaging)", "arm": arm,
                        **{m: cells[m] for m, _, _ in DOWNSTREAM}, "n_seeds": 1})
        print("\nClinical rows are CONTEXT, not rungs of the ladder: in this cohort IDH is nearly")
        print("collinear with WHO grade and age alone reaches ~0.90. Every imaging row shares that")
        print("confound, so the comparison BETWEEN rungs is unaffected by it — only the absolute")
        print("level of the imaging arms has to be read against these numbers.")
    _write(out, out_csv)


def paired_summary(paths: list[str]) -> None:
    rows = read_rows(paths)
    if not rows:
        return
    print("\n" + "=" * 90)
    print("Paired delta-AUC (same subjects, same CV splits) — the comparison that decides ranking")
    print("=" * 90)
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r.get("a", "?"), r.get("b", "?")), []).append(_f(r.get("d_auc")))
    for (a, b), ds in sorted(groups.items()):
        xs = [d for d in ds if math.isfinite(d)]
        if not xs:
            continue
        m = statistics.fmean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else math.nan
        verdict = ("consistent" if len(xs) > 1 and abs(m) > 2 * (sd / math.sqrt(len(xs)))
                   else "not separable across runs")
        print(f"  {a:<24} vs {b:<18} dAUC={m:+.3f}"
              + (f"±{sd:.3f}" if math.isfinite(sd) else " (n=1)")
              + f"  over {len(xs)} run(s)   [{verdict}]")


def _write(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the paper's tables + figures from run artifacts")
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs (default: all *_unified_seed*)")
    ap.add_argument("--summary", default="results/checkpoint_selection_summary.csv")
    ap.add_argument("--probe_glob", default="results/idh_probe*.csv")
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    runs = args.runs or sorted(glob.glob("checkpoints/simsiam_*_unified_seed*"))
    runs = [r for r in runs if os.path.isdir(r)]
    os.makedirs(args.out, exist_ok=True)

    table_labelfree(args.summary, os.path.join(args.out, "table1_labelfree.csv"))
    probe = sorted(p for p in glob.glob(args.probe_glob) if not p.endswith("_paired.csv"))
    table_downstream(probe, os.path.join(args.out, "table2_downstream.csv"))
    paired_summary(sorted(glob.glob(args.probe_glob.replace(".csv", "_paired.csv"))))

    if not args.no_figures and runs:
        print("\n[figures] visualize curves ...")
        cmd = [sys.executable, "-m", "braintumor_ssl.visualize", "curves", "--runs", *runs,
               "--out", os.path.join(args.out, "figures")]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[warn] figure generation failed: {e}")


if __name__ == "__main__":
    main()
