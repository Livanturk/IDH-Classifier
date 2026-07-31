"""The paper's headline table: SSL selection rule x backbone -> downstream IDH AUC.

                     ResNet18   ResNet34   DenseNet121   Average   Best backbone
        RankMe          ...        ...         ...         ...         ...
        LiDAR           ...        ...         ...         ...         ...
        AlphaReQ        ...        ...         ...         ...         ...
        Proposed        ...        ...         ...         ...         ...

Built from `results/idh_probe.csv` (written by `idh_probe.py`), one cell = mean +- SD over the three
seeds of that (rule, backbone). Never the best seed: a single run is not the unit of a claim.

What this table can and cannot show
-----------------------------------
Rows are only distinguishable if the rules actually select DIFFERENT epochs. On the original 20-eval
runs they did not — all three best*.pth were byte-identical, so the RankMe / LiDAR / AlphaReQ rows
came out within 0.003 AUC of each other while a single cell's CV SD was +-0.006-0.011 (lesson L21).
This module therefore refuses to pretend: `--check` reports, per (rule, backbone) cell, whether the
selected epochs differ at all, and the printed table marks cells whose rules collapsed onto the same
checkpoint. A table of four identical rows is a finding about the metrics, not a comparison.

The `regret` view is the honest way to score the Proposed rule
--------------------------------------------------------------
"Proposed" picks one of the candidate checkpoints WITHOUT labels. Its AUC therefore equals whichever
rule it landed on, so putting it in the same column as the rules it chose among and calling it a
winner is circular. What answers "how good is our selection algorithm" is the REGRET:

    regret = AUC(oracle: best of the rules on labels) - AUC(the rule Proposed picked label-free)

averaged over the (backbone, seed) cells. Zero regret means the label-free rule matched the labelled
oracle; the oracle itself is unattainable in practice, which is the point of the comparison.

Usage
-----
    python -m braintumor_ssl.paper_auc_table --csv results/idh_probe.csv
    python -m braintumor_ssl.paper_auc_table --csv results/idh_probe.csv --arm image+age --regret
    python -m braintumor_ssl.paper_auc_table --csv results/idh_probe.csv --check --out results/table1.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

# checkpoint filename -> the row label used in the paper
ROWS = [
    ("bestRankMe.pth", "RankMe"),
    ("bestLiDAR.pth", "LiDAR"),
    ("bestA-ReQ.pth", "AlphaReQ"),
    ("bestPareto.pth", "Proposed"),
    ("last.pth", "last (plateau ref)"),
    # The floor, not a selection rule: same architecture, UNTRAINED, same precise-BN recompute and
    # the same probe. Without it "AUC 0.76" is unreadable — a random 3D conv stack with cohort-
    # adapted BN is already a strong feature extractor on this cohort, so the quantity the paper is
    # actually claiming is the GAP to this row, per backbone and paired within seed.
    ("randominit", "random-init (baseline)"),
]
COLS = [("resnet18", "ResNet18"), ("resnet34", "ResNet34"), ("densenet121", "DenseNet121")]
RULE_ROWS = ["RankMe", "LiDAR", "AlphaReQ"]        # what Proposed chooses among, for regret
BASELINE_ROW = "random-init (baseline)"


def load(csv_paths: list[str], arm: str) -> list[dict]:
    """idh_probe writes ONE csv per encoder run, so the table is assembled from several files.

    Deduplicated on (run, checkpoint, arm): a single invocation that scored several encoders plus
    `--random_baseline` writes its encoder rows into BOTH its own per-run file and the control's
    file, so a naive concatenation counts those runs twice and reports n=4 seeds for a 3-seed cell.
    Later files win, which makes a re-scored run replace its stale row rather than average with it.
    """
    rows = []
    for path in csv_paths:
        with open(path) as f:
            rows += [r for r in csv.DictReader(f) if r.get("arm") == arm]
    dedup = {(r.get("run"), r.get("checkpoint"), r.get("arm")): r for r in rows}
    rows = list(dedup.values())
    for r in rows:
        for k in ("auc", "auc_sd", "ci_lo", "ci_hi"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                r[k] = math.nan
    return rows


def cells(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """(row label, backbone) -> the per-seed records in that cell."""
    by_ckpt = {c: lab for c, lab in ROWS}
    out = defaultdict(list)
    for r in rows:
        lab = by_ckpt.get(r.get("checkpoint"))
        if lab and r.get("backbone") in dict(COLS):
            out[(lab, r["backbone"])].append(r)
    return out


def _agg(recs: list[dict]) -> tuple[float, float, int]:
    if not recs:
        return math.nan, math.nan, 0
    a = [r["auc"] for r in recs if math.isfinite(r["auc"])]
    if not a:
        return math.nan, math.nan, 0
    mean = sum(a) / len(a)
    sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)) ** 0.5 if len(a) > 1 else 0.0
    return mean, sd, len(a)


def _fmt(mean: float, sd: float, n: int) -> str:
    return "—" if not math.isfinite(mean) else f"{mean:.3f}±{sd:.3f} (n={n})"


def build(csv_paths: list[str], arm: str) -> tuple[list[dict], dict]:
    data = cells(load(csv_paths, arm))
    table = []
    for _, lab in ROWS:
        row = {"SSL Method": lab}
        means = []
        for key, pretty in COLS:
            mean, sd, n = _agg(data.get((lab, key), []))
            row[pretty] = _fmt(mean, sd, n)
            if math.isfinite(mean):
                means.append((mean, pretty))
        row["Average"] = f"{sum(m for m, _ in means) / len(means):.3f}" if means else "—"
        row["Best Backbone"] = max(means)[1] if means else "—"
        table.append(row)
    return table, data


def check_distinct(data: dict) -> list[str]:
    """Warn where the selection rules collapsed onto the same checkpoint (lesson L21).

    Two rules that selected the same epoch produce the same weights, hence the same AUC to the last
    decimal. Reporting those as separate rows would invent a comparison that was never made.
    """
    notes = []
    for key, pretty in COLS:
        per_seed = defaultdict(dict)
        for lab in RULE_ROWS + ["Proposed"]:
            for r in data.get((lab, key), []):
                per_seed[r.get("seed")][lab] = r.get("pretrain_epoch", "?")
        for seed, sel in sorted(per_seed.items()):
            epochs = set(sel.values())
            if len(sel) < 2:
                continue
            if epochs == {"-1"} or epochs == {-1}:
                # Older result CSVs stored -1 for every non-`ckpt_eNNN` checkpoint, so the epochs
                # cannot be compared; fall back to the observable consequence instead of guessing.
                notes.append(f"{pretty} seed{seed}: selected epoch not recorded (old CSV) — "
                             f"re-run idh_probe to populate pretrain_epoch")
            elif len(epochs) == 1:
                notes.append(f"{pretty} seed{seed}: {', '.join(sel)} ALL selected epoch "
                             f"{epochs.pop()} — identical weights, rows are not comparable")
    return notes


def regret(data: dict) -> list[dict]:
    """Per (backbone, seed): the labelled oracle over the rules vs what Proposed actually picked."""
    out = []
    for key, pretty in COLS:
        per_seed = defaultdict(dict)
        for lab in RULE_ROWS + ["Proposed"]:
            for r in data.get((lab, key), []):
                per_seed[r.get("seed")][lab] = r["auc"]
        for seed, aucs in sorted(per_seed.items()):
            rule_aucs = {k: v for k, v in aucs.items() if k in RULE_ROWS and math.isfinite(v)}
            prop = aucs.get("Proposed", math.nan)
            if not rule_aucs or not math.isfinite(prop):
                continue
            best_rule = max(rule_aucs, key=rule_aucs.get)
            out.append({"backbone": pretty, "seed": seed, "oracle_rule": best_rule,
                        "oracle_auc": round(rule_aucs[best_rule], 4),
                        "proposed_auc": round(prop, 4),
                        "regret": round(rule_aucs[best_rule] - prop, 4)})
    return out


def ssl_gain(data: dict, rule: str = "RankMe") -> list[dict]:
    """SSL gain over the untrained control, PAIRED WITHIN SEED.

    Paired because the initialization seed alone moves the random-init AUC a lot (measured spread
    0.735-0.802 on DenseNet121), which is larger than the effect being claimed. Differencing the two
    arms of the same seed removes that shared variation; comparing the two column means does not.
    """
    out = []
    for key, pretty in COLS:
        per_seed = defaultdict(dict)
        for lab in (rule, BASELINE_ROW):
            for r in data.get((lab, key), []):
                per_seed[r.get("seed")][lab] = r["auc"]
        gains = []
        for seed, a in sorted(per_seed.items()):
            if rule in a and BASELINE_ROW in a and all(math.isfinite(v) for v in a.values()):
                gains.append({"seed": seed, "ssl": a[rule], "random": a[BASELINE_ROW],
                              "gain": a[rule] - a[BASELINE_ROW]})
        if gains:
            g = [x["gain"] for x in gains]
            mean = sum(g) / len(g)
            sd = (sum((x - mean) ** 2 for x in g) / (len(g) - 1)) ** 0.5 if len(g) > 1 else 0.0
            out.append({"backbone": pretty, "n": len(g), "mean_gain": mean, "sd": sd,
                        "per_seed": gains})
    return out


def print_table(table: list[dict]) -> None:
    cols = ["SSL Method"] + [p for _, p in COLS] + ["Average", "Best Backbone"]
    w = {c: max(len(c), max((len(str(r[c])) for r in table), default=0)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in table:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # `idh_probe_*` rather than `idh_probe_simsiam_*`: the random-init control is written to
    # results/idh_probe_randominit.csv, and the narrower glob silently dropped the baseline row.
    ap.add_argument("--csv", nargs="+", default=["results/idh_probe_*.csv"],
                    help="idh_probe result CSVs; globs are expanded (per-run files are the norm)")
    ap.add_argument("--arm", default="image",
                    help="'image' = encoder features alone; 'image+age' = with the clinical covariate")
    ap.add_argument("--check", action="store_true", help="flag cells whose rules picked the same epoch")
    ap.add_argument("--regret", action="store_true", help="oracle-vs-Proposed regret per cell")
    ap.add_argument("--gain", action="store_true",
                    help="SSL gain over the random-init baseline, paired within seed")
    ap.add_argument("--gain_rule", default="RankMe", help="which rule's row to measure the gain of")
    ap.add_argument("--out", help="also write the table as CSV")
    args = ap.parse_args()

    paths = sorted({p for pat in args.csv for p in glob.glob(pat) if not p.endswith("_paired.csv")}
                   or {p for p in args.csv if os.path.exists(p)})
    if not paths:
        raise SystemExit(f"no result CSVs matched {args.csv} — run idh_probe.py first")
    print(f"[read] {len(paths)} result file(s)")
    table, data = build(paths, args.arm)
    print(f"\nDownstream IDH AUC — arm='{args.arm}', mean±SD over seeds\n")
    print_table(table)

    if args.check:
        notes = check_distinct(data)
        print("\n[check] selection-rule distinctness:")
        print("\n".join(f"  ! {n}" for n in notes) if notes
              else "  all rules selected distinct epochs — the rows are genuinely comparable")

    if args.regret:
        rg = regret(data)
        print("\n[regret] labelled oracle over {RankMe, LiDAR, AlphaReQ} vs label-free Proposed:")
        for r in rg:
            print(f"  {r['backbone']:<12} seed{r['seed']}  oracle={r['oracle_rule']:<9} "
                  f"{r['oracle_auc']:.3f}  proposed={r['proposed_auc']:.3f}  "
                  f"regret={r['regret']:+.3f}")
        if rg:
            m = sum(r["regret"] for r in rg) / len(rg)
            print(f"  mean regret over {len(rg)} cells = {m:+.4f} AUC")

    if args.gain:
        print(f"\n[gain] {args.gain_rule} vs {BASELINE_ROW}, paired within seed:")
        for g in ssl_gain(data, args.gain_rule):
            per = "  ".join(f"s{x['seed']}:{x['gain']:+.3f}" for x in g["per_seed"])
            print(f"  {g['backbone']:<12} n={g['n']}  mean={g['mean_gain']:+.3f} ± {g['sd']:.3f}"
                  f"   [{per}]")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
