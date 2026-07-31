"""Per-run checkpoint-selection report: compare bestRankMe / bestLiDAR / bestA-ReQ.

For each SSL run directory this reads `metrics.csv` (the per-epoch RankMe / LiDAR / alpha-ReQ /
loss / LR trajectory written by train_simsiam) and the `selection` provenance stored inside the
three per-metric best checkpoints, then answers:

  1. epoch / metric-value / train+val loss of each of the three best checkpoints
  2. whether all three landed on the SAME epoch (metrics fully agree)
  3. if >=2 share an epoch, WHICH metrics co-selected that checkpoint
  4. if they differ, the epoch gaps + how the three metric curves correlate over training
  5. convergence / collapse / plateau read jointly from RankMe + LiDAR + alpha-ReQ (agree vs conflict)
  6. one appended row per run to a filterable summary CSV (filter by config / backbone / seed)

Generalizes across architectures, configs and seeds: backbone / seed / config / split are read from
the cfg embedded in each checkpoint, so nothing here is specific to DenseNet121 or seed 42.

Usage
-----
  python -m braintumor_ssl.report_run --run checkpoints/simsiam_densenet121_unified_seed42
  python -m braintumor_ssl.report_run --run checkpoints/run_a checkpoints/run_b   # several runs
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import torch

from braintumor_ssl.select_epoch import (FALLBACK, load_rows as load_raw_rows,
                                         select as select_pareto, sps_pick)

# Pre-registered parameters of the Proposed policy (see select_epoch.py and lessons L22): one
# jackknife SE as the indifference band, and "on the plateau" = within 0.01 of the run's loss floor.
# PRE-REGISTERED on a priori grounds, not tuned: the rule's output is sensitive to both (L23).
# Fixed here as the single source of truth so every report uses the same rule; they are deliberately
# NOT config keys (training never reads them). `select_epoch.py --sweep` audits the sensitivity.
PLATEAU_TOL = 0.01
PARETO_EPS = 1.0

# metric key -> (best-checkpoint filename, pretty name, "want" direction text)
METRICS = {
    "rankme": ("bestRankMe.pth", "RankMe", "higher = richer effective rank"),
    "lidar": ("bestLiDAR.pth", "LiDAR", "higher = more augmentation-invariant rank"),
    "alpha_req": ("bestA-ReQ.pth", "alpha-ReQ", "closest to 1.0 = optimal-transfer spectrum"),
}


def load_metrics_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                r[k] = math.nan
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass  # keep strings (stop_reason, bool-ish) as-is
    return rows


def load_selection(run_dir: str, fname: str) -> dict | None:
    """Read the `selection` provenance dict a best*.pth stored at save time."""
    p = os.path.join(run_dir, fname)
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu")
    sel = dict(ck.get("selection", {}))
    sel.setdefault("epoch", ck.get("epoch"))
    sel["_cfg"] = ck.get("cfg", {})
    return sel


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else math.nan


def _rank(vals: list[float]) -> list[float]:
    """Fractional ranks (ties averaged) — the transform that turns Pearson into Spearman."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Preferred over Pearson here: nothing says RankMe and LiDAR should move
    together LINEARLY — the question is only whether they order the epochs the same way, and ranks
    are also far less sensitive to a single outlying epoch than raw values."""
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    return _pearson(_rank([x for x, _ in pairs]), _rank([y for _, y in pairs]))


def _col(rows: list[dict], key: str) -> list[float]:
    return [r.get(key, math.nan) for r in rows]


def _eligible(rows: list[dict]) -> list[dict]:
    """Rows that passed the convergence + non-collapse gate.

    Correlations are computed over these ONLY. Including the random-init epochs would mostly measure
    "all three metrics fall away from random initialization", which is a shared transient, not
    evidence that they order the candidate checkpoints the same way.
    """
    def flag(r, k):
        return str(r.get(k)).lower() in ("true", "1", "1.0")
    keep = [r for r in rows if flag(r, "converged") and not flag(r, "collapsed")]
    return keep if len(keep) >= 3 else rows


def report_run(run_dir: str) -> dict:
    """Print the 6-item comparison for one run and return a flat summary row for the CSV."""
    name = os.path.basename(os.path.normpath(run_dir))
    rows = load_metrics_csv(os.path.join(run_dir, "metrics.csv"))
    sels = {k: load_selection(run_dir, f) for k, (f, _, _) in METRICS.items()}

    cfg = next((s["_cfg"] for s in sels.values() if s and s.get("_cfg")), {})
    backbone = cfg.get("backbone", "?")
    seed = cfg.get("seed", "?")
    split = os.path.basename(cfg.get("splits_file", "?"))

    print("=" * 78)
    print(f"RUN  {name}   backbone={backbone}  seed={seed}  split={split}")
    print("=" * 78)

    # ---- (1) per-checkpoint table -------------------------------------------------
    print("\n[1] Selected checkpoints (converged, non-collapsed epochs only):")
    print(f"    {'checkpoint':<14}{'epoch':>6}{'value':>10}{'train_loss':>12}{'val_loss':>10}{'lr':>10}"
          f"{'n_val':>7}{'a_R2':>8}")
    present = {}
    for key, (fname, pretty, _) in METRICS.items():
        s = sels[key]
        if not s:
            print(f"    {pretty:<14}{'—':>6}   (not written — no eligible epoch)")
            continue
        present[key] = s
        # alpha's fit quality is shown for EVERY checkpoint: it says whether the alpha reported in
        # this run's spectrum is interpretable at all, not just whether alpha-ReQ picked this epoch.
        fit = s.get("areq_fit") or {}
        print(f"    {pretty:<14}{int(s['epoch']):>6}{s.get('metric_value', float('nan')):>10.4f}"
              f"{s.get('train_loss', float('nan')):>12.4f}{s.get('val_loss', float('nan')):>10.4f}"
              f"{s.get('lr', float('nan')):>10.5f}{s.get('n_val', 0):>7}"
              f"{fit.get('r2', float('nan')):>8.3f}")

    # ---- (2)/(3) epoch agreement --------------------------------------------------
    epochs = {k: int(s["epoch"]) for k, s in present.items()}
    print("\n[2/3] Epoch agreement:")
    if len(epochs) >= 2 and len(set(epochs.values())) == 1 and len(epochs) == len(METRICS):
        e = next(iter(epochs.values()))
        print(f"    ALL THREE metrics selected the SAME epoch ({e}) — full agreement.")
    else:
        # group metrics by shared epoch
        by_epoch: dict[int, list[str]] = {}
        for k, e in epochs.items():
            by_epoch.setdefault(e, []).append(METRICS[k][1])
        shared = {e: ms for e, ms in by_epoch.items() if len(ms) >= 2}
        if shared:
            for e, ms in shared.items():
                print(f"    epoch {e}: co-selected by {', '.join(ms)} (these metrics agree).")
            singles = {e: ms for e, ms in by_epoch.items() if len(ms) == 1}
            for e, ms in singles.items():
                print(f"    epoch {e}: only {ms[0]}.")
        else:
            print("    All selected epochs differ — no two metrics co-selected a checkpoint.")

    # ---- (4) epoch gaps + curve correlation --------------------------------------
    if len(set(epochs.values())) > 1:
        lo, hi = min(epochs.values()), max(epochs.values())
        print(f"\n[4] Epoch spread: {hi - lo} epochs (from {lo} to {hi}).")
        if rows:
            elig = _eligible(rows)
            rk, li, ar = _col(elig, "rankme"), _col(elig, "lidar"), _col(elig, "alpha_req")
            print(f"    curve rank-correlation over {len(elig)} eligible epochs (Spearman):  "
                  f"RankMe~LiDAR={_spearman(rk, li):+.2f}  "
                  f"RankMe~alpha-ReQ={_spearman(rk, ar):+.2f}  "
                  f"LiDAR~alpha-ReQ={_spearman(li, ar):+.2f}")
            print("    (high +corr => metrics track together; near 0 / negative => they read the "
                  "spectrum differently, so their best epochs legitimately diverge.)")
    else:
        print("\n[4] Epochs coincide — no spread to summarize.")

    # ---- (5) convergence / collapse / plateau, read jointly -----------------------
    print("\n[5] Convergence / collapse / plateau (joint read):")
    if rows:
        last = rows[-1]
        ever_collapsed = any(str(r.get("collapsed")).lower() in ("true", "1", "1.0") for r in rows)
        n_converged = sum(1 for r in rows if str(r.get("converged")).lower() in ("true", "1", "1.0"))
        elig = _eligible(rows)
        rk, li, ar = _col(elig, "rankme"), _col(elig, "lidar"), _col(elig, "alpha_req")
        trend = _spearman(list(range(len(rk))), rk)
        print(f"    epochs validated={len(rows)}  converged={n_converged}  ever_collapsed={ever_collapsed}")
        print(f"    end plateau counters: RankMe+{int(last.get('no_improve_rankme', 0) or 0)} "
              f"LiDAR+{int(last.get('no_improve_lidar', 0) or 0)} "
              f"alpha-ReQ+{int(last.get('no_improve_areq', 0) or 0)}")
        stopped = str(last.get("stopped")).lower() in ("true", "1", "1.0")
        if stopped:
            print(f"    early-stopped: {last.get('stop_reason', '')}")
        # convergent-validity read: do RankMe and LiDAR agree on the direction of change?
        agree = _spearman(rk, li)
        msg = ("RankMe & LiDAR give a CONSISTENT signal" if agree > 0.3 else
               "RankMe & LiDAR DISAGREE — treat the run as ambiguous (defer to labelled probe)"
               if agree < -0.3 else "RankMe & LiDAR are weakly related")
        print(f"    RankMe trend over epochs={trend:+.2f}; {msg} (corr={agree:+.2f}).")
    else:
        print("    metrics.csv missing/empty — cannot assess convergence.")

    # ---- (5b) Stable Plateau Selection — the "Proposed" policy ----------------------
    # Plateau is read off the METRICS' settled values, so the rule needs neither a convergence gate
    # (untrained epochs are far from those values and drop out on their own) nor per-metric
    # direction knowledge (the test is a distance). See select_epoch.sps_pick.
    print("\n[5b] Stable Plateau Selection (Proposed):")
    sps = None
    if rows:
        sps = sps_pick(load_raw_rows(run_dir))
        print(f"    plateau evals={sps.get('n_plateau')}/{sps.get('n_evals')}  "
              f"tol={sps['tol']:.0%} consec={sps['consec']}")
        print(f"    -> {'epoch ' + str(sps['epoch']) if sps['epoch'] is not None else FALLBACK}"
              f"   ({sps['decision']})")
    else:
        print("    metrics.csv missing/empty.")

    # ---- (5c) plateau-gated Pareto front (ablation) ---------------------------------
    # The front SIZE is the headline diagnostic, not a by-product: it counts how many plateau
    # epochs the three metrics cannot rank against each other. A front of 1 means they agree and
    # the pick is theirs; a wide front means they carry no independent signal at this resolution
    # and the honest output is a deferral. See select_epoch.py.
    print("\n[5c] Plateau-gated Pareto selection (ablation):")
    pareto = None
    if rows:
        r2_min = float(cfg.get("areq_r2_min", 0.0) or 0.0)
        pareto = select_pareto(load_raw_rows(run_dir), PLATEAU_TOL, PARETO_EPS, r2_min)
        print(f"    pool={pareto['n_pool']}/{pareto['n_converged']} converged epochs on the plateau "
              f"(val_loss <= {pareto['plateau_val_loss']}+{PLATEAU_TOL})")
        print(f"    front size={pareto['front_size']}  epochs={pareto['front_epochs']}  "
              f"(eps={PARETO_EPS} SE)  -> {pareto['decision']}"
              + (f" epoch {pareto['epoch']}" if pareto["epoch"] is not None else f" ({FALLBACK})"))
    else:
        print("    metrics.csv missing/empty — cannot compute the front.")

    # ---- (6) flat summary row -----------------------------------------------------
    summary = {"config": name, "backbone": backbone, "seed": seed, "split": split,
               "n_val_epochs": len(rows),
               "final_epoch": int(rows[-1]["epoch"]) if rows else "",
               "stop_reason": (rows[-1].get("stop_reason", "") if rows else ""),
               "pareto_front_size": pareto["front_size"] if pareto else "",
               "pareto_front_epochs": " ".join(str(e) for e in pareto["front_epochs"]) if pareto else "",
               "pareto_pool": pareto["n_pool"] if pareto else "",
               "pareto_decision": pareto["decision"] if pareto else "",
               "pareto_epoch": (pareto["epoch"] if pareto and pareto["epoch"] is not None else ""),
               "sps_epoch": (sps["epoch"] if sps and sps["epoch"] is not None else ""),
               "sps_decision": sps["decision"] if sps else "",
               "sps_n_plateau": sps.get("n_plateau", "") if sps else ""}
    for key, (_, pretty, _) in METRICS.items():
        s = present.get(key)
        pre = pretty.replace("-", "").replace(" ", "")
        summary[f"{pre}_epoch"] = int(s["epoch"]) if s else ""
        summary[f"{pre}_value"] = s.get("metric_value", "") if s else ""
        summary[f"{pre}_train_loss"] = s.get("train_loss", "") if s else ""
        summary[f"{pre}_val_loss"] = s.get("val_loss", "") if s else ""
        summary[f"{pre}_lr"] = s.get("lr", "") if s else ""
        # ALL THREE metrics at this checkpoint, not just the one that selected it — this is what
        # makes the "policy x metric" table possible (does the LiDAR-selected checkpoint also have
        # a high RankMe?), and what aggregate_runs.py averages over seeds.
        at = dict(s.get("other_metrics", {})) if s else {}
        if s:
            at[key] = s.get("metric_value", math.nan)
        for m in METRICS:
            summary[f"{pre}_at_{m}"] = at.get(m, "")
        summary[f"{pre}_areq_r2"] = (s.get("areq_fit") or {}).get("r2", "") if s else ""
        summary[f"{pre}_metric_se"] = (s.get("rule") or {}).get("metric_se", "") if s else ""
    return summary


def append_summary(summary_path: str, row: dict) -> None:
    """Append/replace this run's row in the filterable cross-run summary CSV (keyed by config)."""
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    existing: list[dict] = []
    fields = list(row.keys())
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = list(csv.DictReader(f))
        for r in existing:
            for k in row:
                r.setdefault(k, "")
        fields = list(dict.fromkeys(list(existing[0].keys()) + fields)) if existing else fields
    existing = [r for r in existing if r.get("config") != row["config"]]  # replace prior row for this run
    existing.append(row)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)


def _write_agreement_figure(run_dir: str) -> None:
    """The picture behind report items [2]-[4]: the three selection-score curves with the epoch each
    rule picked, plus the rank scatters the printed Spearman rho is computed from. Plotting must
    never break a report, so a failure here is a warning."""
    try:
        import pandas as pd

        from braintumor_ssl.visualize import save_metric_agreement
        rows = load_metrics_csv(os.path.join(run_dir, "metrics.csv"))
        if not rows:
            return
        sels = {k: load_selection(run_dir, f) for k, (f, _, _) in METRICS.items()}
        selected = {k: (int(s["epoch"]) if s and s.get("epoch") is not None else None)
                    for k, s in sels.items()}
        name = os.path.basename(os.path.normpath(run_dir))
        out = os.path.join(run_dir, "metric_agreement.png")
        save_metric_agreement(pd.DataFrame(rows), out, selected=selected,
                              title=f"Label-free metric agreement — {name}")
        print(f"    figure -> {out}")
    except Exception as e:                       # never let a plot failure kill the report
        print(f"    [warn] metric_agreement.png not written: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-run RankMe/LiDAR/alpha-ReQ checkpoint-selection report")
    ap.add_argument("--run", nargs="+", required=True, help="run directory/-ies (each holds metrics.csv + best*.pth)")
    ap.add_argument("--summary", default="results/checkpoint_selection_summary.csv",
                    help="filterable cross-run summary CSV to append to")
    ap.add_argument("--no_figure", action="store_true",
                    help="skip <run>/metric_agreement.png (the visual form of items 2-4)")
    args = ap.parse_args()

    for run_dir in args.run:
        summary = report_run(run_dir)
        append_summary(args.summary, summary)
        print(f"\n[6] appended summary row -> {args.summary}")
        if not args.no_figure:
            _write_agreement_figure(run_dir)
        print()


if __name__ == "__main__":
    main()
