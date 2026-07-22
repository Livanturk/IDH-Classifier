"""D2 — deterministic across-run (backbone) model selection from a label-free leaderboard.

Implements Section 3, D2 of MODEL_SELECTION.md: a lexicographic *gated* procedure (not a
weighted sum) so the winner is defensible in the paper.

    1. Eligibility gate : finite metrics ∧ z_std ≥ z_std_min ∧ alignment ≤ align_max
                          (a run that collapsed or never converged is disqualified)
    2. Representative   : per config, the eligible checkpoint with the highest RankMe
                          (mirrors D1; leaderboards with several epochs collapse to one row/run)
    3. Primary key      : RankMe ↓  [Garrido2023]
    4. Uncertainty rule : two runs tie if their RankMe are within `delta` — or, when the
                          leaderboard carries CI columns (rankme_ci_lo/rankme_ci_hi), if those
                          intervals overlap (preferred; produced by evaluate.py once A0.6 lands)
    5. Tie-break        : participation_ratio ↓ then uniformity ↑ (more negative better); the two
                          must agree, else "no label-free separation" → defer to D3 (supervised)

Reads `results/leaderboard.csv` (as written by evaluate.py). val_loss is not in the leaderboard,
so convergence is gated on alignment + z_std only; this is the eval-protocol counterpart of the
training-time gate and is documented as such.

Usage:
    python -m braintumor_ssl.select_model --leaderboard results/leaderboard.csv
    python -m braintumor_ssl.select_model --select latest        # compare last.pth rows instead
"""
from __future__ import annotations
import argparse
import csv
import math

# gate defaults (see MODEL_SELECTION.md §4). NOTE: align_max here is the EVAL-protocol threshold,
# deliberately looser than the training-time gate (0.15). evaluate.py measures alignment on
# precise-BN-adapted features with freshly-sampled augmentation views, so the same checkpoint reads
# systematically higher than during training (e.g. densenet e19: 0.107 train vs 0.151 eval). The
# gate must only separate converged (align ≈ 0.05–0.15) from random-init (align ≈ 0.5); we set the
# threshold at the midpoint of that gap so a genuinely-converged model is never disqualified by
# boundary noise while an untrained checkpoint is still cleanly rejected.
Z_STD_MIN = 0.01
ALIGN_MAX = 0.30
DELTA_RANKME = 0.5          # RankMe margin below which two runs are "tied" (used if no CI columns)
_NUM = ("epoch", "z_std", "rankme", "alignment", "uniformity", "particip_ratio",
        "rankme_ci_lo", "rankme_ci_hi")


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in _NUM:
            if k in r and r[k] not in (None, ""):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = math.nan
    return rows


def eligible(r: dict, z_std_min: float, align_max: float) -> tuple[bool, str]:
    vals = [r.get(k) for k in ("z_std", "rankme", "alignment", "uniformity", "particip_ratio")]
    if any(v is None or (isinstance(v, float) and not math.isfinite(v)) for v in vals):
        return False, "non_finite"
    if r["z_std"] < z_std_min:
        return False, "collapsed(z_std)"
    if r["alignment"] > align_max:
        return False, "not_converged(align)"
    return True, "ok"


def representative(rows: list[dict], policy: str, z_std_min: float, align_max: float) -> dict | None:
    """Pick the row that represents this run: best-converged (default), latest, or a fixed epoch."""
    if policy == "latest":
        return max(rows, key=lambda r: r["epoch"])
    if policy.startswith("epoch:"):
        want = float(policy.split(":", 1)[1])
        cand = [r for r in rows if r["epoch"] == want]
        return cand[0] if cand else None
    # default: highest RankMe among the eligible (converged, non-collapsed) checkpoints
    elig = [r for r in rows if eligible(r, z_std_min, align_max)[0]]
    return max(elig, key=lambda r: r["rankme"]) if elig else None


def _tied(a: dict, b: dict, delta: float) -> bool:
    if all(k in a and k in b and math.isfinite(a[k]) and math.isfinite(b[k])
           for k in ("rankme_ci_lo", "rankme_ci_hi")):
        return a["rankme_ci_lo"] <= b["rankme_ci_hi"] and b["rankme_ci_lo"] <= a["rankme_ci_hi"]
    return abs(a["rankme"] - b["rankme"]) < delta


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def build_entries(reps: dict, group_by: str) -> list[dict]:
    """One comparison entry per unit (config, or backbone aggregated over seeds).

    group_by='backbone' averages a backbone's per-seed representatives and reports the ACROSS-SEED
    95% CI (mean ± 1.96·sd/√k) — the paper's headline uncertainty. With a single seed it falls back
    to that run's jackknife CI. group_by='config' keeps one entry per run (per-seed jackknife CI)."""
    if group_by == "config":
        out = []
        for r in reps.values():
            out.append({"label": r["config"], "backbone": r["backbone"], "n": 1,
                        "rankme": r["rankme"], "particip_ratio": r["particip_ratio"],
                        "uniformity": r["uniformity"],
                        "rankme_ci_lo": r.get("rankme_ci_lo", math.nan),
                        "rankme_ci_hi": r.get("rankme_ci_hi", math.nan),
                        "detail": f"e{int(r['epoch'])}"})
        return out
    groups: dict[str, list[dict]] = {}
    for r in reps.values():
        groups.setdefault(r["backbone"], []).append(r)
    out = []
    for bb, members in groups.items():
        rk = [m["rankme"] for m in members]
        mean = _mean(rk)
        if len(rk) >= 2:
            sd = (sum((x - mean) ** 2 for x in rk) / (len(rk) - 1)) ** 0.5
            half = 1.959963984540054 * sd / (len(rk) ** 0.5)
            lo, hi = mean - half, mean + half
        else:                                    # single seed -> use its jackknife CI
            lo = members[0].get("rankme_ci_lo", math.nan)
            hi = members[0].get("rankme_ci_hi", math.nan)
        out.append({"label": bb, "backbone": bb, "n": len(members),
                    "rankme": mean, "rankme_ci_lo": lo, "rankme_ci_hi": hi,
                    "particip_ratio": _mean([m["particip_ratio"] for m in members]),
                    "uniformity": _mean([m["uniformity"] for m in members]),
                    "detail": f"{len(members)} seed(s)"})
    return out


def rank_entries(entries: list[dict], delta: float) -> dict:
    ranked = sorted(entries, key=lambda e: e["rankme"], reverse=True)
    v: dict = {"ranked": ranked}
    if not ranked:
        return {**v, "winner": None, "reason": "all runs disqualified", "tie": []}
    top = ranked[0]
    if len(ranked) == 1:
        return {**v, "winner": top, "tie": [top], "reason": "only qualified unit (others disqualified)"}
    tie = [e for e in ranked if _tied(top, e, delta)]
    if len(tie) == 1:
        return {**v, "winner": top, "tie": tie,
                "reason": f"clear RankMe winner (margin {top['rankme'] - ranked[1]['rankme']:+.2f})"}
    by_pr = max(tie, key=lambda e: e["particip_ratio"])
    by_unif = min(tie, key=lambda e: e["uniformity"])
    if by_pr["label"] == by_unif["label"]:
        return {**v, "winner": by_pr, "tie": tie,
                "reason": f"RankMe tie ({len(tie)}); PR+uniformity agree"}
    return {**v, "winner": None, "tie": tie,
            "reason": "RankMe tie and PR/uniformity disagree -> no label-free separation; defer to D3"}


def select(rows: list[dict], z_std_min: float, align_max: float, delta: float,
           policy: str, group_by: str = "backbone") -> dict:
    reps, disq = {}, {}
    for c in sorted({r["config"] for r in rows}):
        rep = representative([r for r in rows if r["config"] == c], policy, z_std_min, align_max)
        if rep is None:
            disq[c] = "no eligible (converged, non-collapsed) checkpoint"
        else:
            reps[c] = rep
    entries = build_entries(reps, group_by)
    return {**rank_entries(entries, delta), "disqualified": disq, "reps": reps, "group_by": group_by}


def save_comparison_bar(entries: list[dict], winner_label, path: str, group_by: str) -> None:
    """RankMe bar chart with CI error bars (across-seed CI for backbone grouping, jackknife CI for
    config). The winner bar is highlighted. This is the paper's headline comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ent = sorted(entries, key=lambda e: e["rankme"], reverse=True)
    labels = [e["label"] for e in ent]
    vals = [e["rankme"] for e in ent]
    lo = [e["rankme"] - e["rankme_ci_lo"] if math.isfinite(e.get("rankme_ci_lo", math.nan)) else 0.0 for e in ent]
    hi = [e["rankme_ci_hi"] - e["rankme"] if math.isfinite(e.get("rankme_ci_hi", math.nan)) else 0.0 for e in ent]
    colors = ["#2c7fb8" if e["label"] == winner_label else "#a6bddb" for e in ent]

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(ent)), 4.5))
    ax.bar(range(len(ent)), vals, yerr=[lo, hi], capsize=6, color=colors, edgecolor="black", linewidth=0.6)
    top_pad = 0.02 * (max(vals) if vals else 1.0)
    for i, e in enumerate(ent):
        ytxt = vals[i] + hi[i] + top_pad                 # above the CI cap so it never overlaps
        ax.text(i, ytxt, f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(ent)))
    ax.set_xticklabels([f"{lab}\n({e['detail']})" for lab, e in zip(labels, ent)], fontsize=9)
    ax.set_ylabel("RankMe (effective rank)")
    ci_kind = "across-seed 95% CI" if group_by == "backbone" else "jackknife 95% CI"
    ax.set_title(f"Backbone comparison — RankMe ± {ci_kind}\n(higher = richer representation; winner highlighted)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt(e: dict) -> str:
    ci = ""
    if math.isfinite(e.get("rankme_ci_lo", math.nan)):
        ci = f" [{e['rankme_ci_lo']:.2f},{e['rankme_ci_hi']:.2f}]"
    return (f"{e['label']:<34} {e['detail']:<10} "
            f"RankMe={e['rankme']:6.3f}{ci}  PR={e['particip_ratio']:.3f}  unif={e['uniformity']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="D2 across-run model selection (MODEL_SELECTION.md §3)")
    ap.add_argument("--leaderboard", default="results/leaderboard.csv")
    ap.add_argument("--z_std_min", type=float, default=Z_STD_MIN)
    ap.add_argument("--align_max", type=float, default=ALIGN_MAX)
    ap.add_argument("--delta", type=float, default=DELTA_RANKME,
                    help="RankMe tie margin (ignored when CI columns / multiple seeds give a CI)")
    ap.add_argument("--select", default="best-converged",
                    choices=["best-converged", "latest"], help="or 'epoch:N'")
    ap.add_argument("--select_epoch", type=int, help="shortcut for --select epoch:N")
    ap.add_argument("--group_by", default="backbone", choices=["backbone", "config"],
                    help="'backbone' aggregates seeds (across-seed CI); 'config' keeps per-run")
    ap.add_argument("--fig", default="results/backbone_comparison.png",
                    help="write the RankMe±CI comparison bar chart here")
    ap.add_argument("--mlflow", action="store_true", help="push the table + figure + verdict to MLflow/DagsHub")
    ap.add_argument("--mlflow_experiment", default="simsiam-brats-selection")
    args = ap.parse_args()

    policy = f"epoch:{args.select_epoch}" if args.select_epoch is not None else args.select
    rows = load_rows(args.leaderboard)
    v = select(rows, args.z_std_min, args.align_max, args.delta, policy, args.group_by)

    print(f"[D2] leaderboard={args.leaderboard}  group_by={args.group_by}  policy={policy}  "
          f"gates: z_std>={args.z_std_min}, align<={args.align_max}\n")
    print(f"Ranked {args.group_by}s (by RankMe):")
    for e in v["ranked"]:
        print("  " + _fmt(e))
    for c, why in v["disqualified"].items():
        print(f"  {c:<34} DISQUALIFIED — {why}")
    winner_label = v["winner"]["label"] if v["winner"] is not None else None
    print()
    print(f"WINNER: {winner_label} — {v['reason']}" if winner_label else f"NO WINNER — {v['reason']}")

    if v["ranked"]:
        import os
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        save_comparison_bar(v["ranked"], winner_label, args.fig, args.group_by)
        print(f"[plot] {args.fig}")

    if args.mlflow:
        from braintumor_ssl.tracking import Tracker
        tk = Tracker({"mlflow": True, "mlflow_experiment": args.mlflow_experiment},
                     run_name="D2-selection", tags={"stage": "selection", "group_by": args.group_by})
        tk.log_params({"group_by": args.group_by, "policy": policy, "z_std_min": args.z_std_min,
                       "align_max": args.align_max, "winner": winner_label or "none",
                       "reason": v["reason"]})
        for e in v["ranked"]:
            tk.log_metrics({f"rankme[{e['label']}]": e["rankme"],
                            f"rankme_ci_lo[{e['label']}]": e["rankme_ci_lo"],
                            f"rankme_ci_hi[{e['label']}]": e["rankme_ci_hi"]})
        tk.log_artifact(args.fig, artifact_path="figures")
        tk.log_artifact(args.leaderboard, artifact_path="tables")
        tk.close()
        print("[mlflow] pushed selection summary (table + figure + verdict)")


if __name__ == "__main__":
    main()
