"""D1 — post-hoc WITHIN-run epoch selection: plateau gate + SE-aware Pareto over the three
label-free metrics. Writes `bestPareto.pth` (the paper's "Proposed" policy).

Why this exists, and why it is post-hoc
---------------------------------------
The online rule in `train_simsiam.py` saves one checkpoint per metric (`bestRankMe` / `bestLiDAR` /
`bestA-ReQ`) among epochs that clear a CONVERGENCE gate (`val_loss <= best_val_loss_max` and
`alignment <= best_align_max`). Two things go wrong with that pool, both measured (lessons L21/L22):

  1. RankMe and -|alpha-1| decrease monotonically after convergence, so "argmax among eligible"
     always resolves to the EARLIEST eligible epoch — which sits on the steep transient, not the
     plateau. All three metrics therefore fire on the same eval and the three best*.pth files come
     out byte-identical (verified across all 9 unified runs).
  2. `select_model.py` already documented the same failure one level up (across runs): the
     max-RankMe-among-converged representative "samples a steep transient at a seed-dependent
     convergence epoch" with CV up to ~24%, and the fix there was to use the plateau checkpoint.
     That lesson was never applied to the within-run epoch choice. This module applies it.

The plateau gate CANNOT be evaluated online: during training the running-minimum val_loss equals the
current val_loss, so the first eligible epoch trivially passes any "close to the plateau" test. It is
well defined only once the whole curve exists. Running post-hoc off `metrics.csv` is also cheaper —
the gate, the tolerance and the Pareto epsilon can all be re-tuned without retraining. Training's
only obligation is to have SAVED a checkpoint at every validated epoch (it does).

The rule
--------
  1. Candidate pool  : converged AND not collapsed AND areq_r2 >= areq_r2_min
                       AND val_loss <= min(val_loss) + `plateau_tol`      <- the new gate
  2. Pareto front    : non-dominated epochs under v(e) = (RankMe, LiDAR, -|alpha-1|), all
                       "higher is better". Domination is SE-aware: an epoch must be better by more
                       than `pareto_eps` x the larger jackknife SE on at least one axis and not
                       worse by that margin on any. eps=0 recovers the strict Pareto front.
  3. Decision        : |front| == 1 -> "selected": the three metrics agree, take it.
                       |front| >  1 -> "deferred": the metrics carry no label-free separation among
                       those epochs. The FLAG is kept and reported, but the checkpoint is the
                       front's MEDIAN epoch rather than `last.pth` (see below).
                       empty pool    -> "fallback": there is no candidate at all, use `last.pth`.

Why the median and not `last.pth` (changed 2026-07-28, methodology §8.4)
-----------------------------------------------------------------------
The first version fell back to `last.pth` whenever the front was wider than one epoch. That reads
well as "refusing to pick", but it breaks the experiment it feeds: `last.pth` is a SEPARATE arm of
the paper's comparison ladder, so a Proposed arm that lands on it in some seeds is partly a copy of
the arm it is being compared against, and the two become impossible to tell apart. Every front
member is already converged, on the plateau and non-dominated — they are all valid candidates
between which no label-free preference exists, not a signal to abandon the pool. The median epoch is
deterministic, still label-free, and far less sensitive to `pareto_eps` than the exact front size is.
The "I cannot separate these" information is not lost: `decision` stays `deferred` and the front size
is printed in every report. The rule produces an artifact AND declares its own uncertainty.

The front SIZE is itself a reported result: it measures how much independent evidence the three
metrics actually carry. On the current runs the front is 1-3 epochs out of a 13-14 epoch pool, and
rho(RankMe, alpha-ReQ) ~ 0.92-0.93, i.e. alpha-ReQ is close to a monotone transform of RankMe here.

Usage
-----
    python -m braintumor_ssl.select_epoch --run checkpoints/simsiam_r18_unified_seed42
    python -m braintumor_ssl.select_epoch --run checkpoints/*_seed4? --summary results/pareto.csv
    python -m braintumor_ssl.select_epoch --run <run> --write        # materialize bestPareto.pth
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil

import torch

# metric key -> (metrics.csv column, jackknife-SE column, pretty name)
AXES = [
    ("rankme", "rankme", "rankme_se_jk", "RankMe"),
    ("lidar", "lidar", "lidar_se_jk", "LiDAR"),
    ("alpha_req", "alpha_req", "areq_se_jk", "alpha-ReQ"),
]
OUT_NAME = "bestPareto.pth"
FALLBACK = "last.pth"

# metric key -> filename for the UNGATED baseline rule (see `naive_picks`)
NAIVE_FILES = {"rankme": "bestNaiveRankMe.pth", "lidar": "bestNaiveLiDAR.pth",
               "alpha_req": "bestNaiveA-ReQ.pth"}


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _flag(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("true", "1", "1.0")


def load_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def score_vector(row: dict) -> dict[str, float]:
    """All three axes as 'higher is better'. alpha-ReQ becomes -|alpha - 1| (target alpha = 1)."""
    return {"rankme": _f(row, "rankme"), "lidar": _f(row, "lidar"),
            "alpha_req": -abs(_f(row, "alpha_req") - 1.0)}


def se_vector(row: dict) -> dict[str, float]:
    return {key: _f(row, se_col) for key, _, se_col, _ in AXES}


def candidate_pool(rows: list[dict], plateau_tol: float, areq_r2_min: float) -> tuple[list[dict], float]:
    """Converged, non-collapsed, power-law-valid epochs that also sit ON the loss plateau.

    Returns (pool, plateau_val_loss). The plateau reference is the run's best val_loss over ALL
    evals — the deepest point the run reached — so `plateau_tol` reads as "how far above the final
    loss floor an epoch may sit and still count as converged-and-settled".
    """
    losses = [_f(r, "val_loss") for r in rows if math.isfinite(_f(r, "val_loss"))]
    if not losses:
        return [], math.nan
    plateau = min(losses)
    pool = [
        r for r in rows
        if _flag(r, "converged") and not _flag(r, "collapsed")
        and _f(r, "areq_r2") >= areq_r2_min
        and _f(r, "val_loss") <= plateau + plateau_tol
        and all(math.isfinite(v) for v in score_vector(r).values())
    ]
    return pool, plateau


def dominates(a: dict, b: dict, eps: float) -> bool:
    """True if `a` Pareto-dominates `b` allowing an eps-SE band of indifference on every axis."""
    va, vb, sa, sb = score_vector(a), score_vector(b), se_vector(a), se_vector(b)
    better = False
    for key, _, _, _ in AXES:
        se = max(sa[key], sb[key])
        tol = eps * (se if math.isfinite(se) else 0.0)
        if va[key] < vb[key] - tol:
            return False                      # meaningfully worse somewhere -> cannot dominate
        if va[key] > vb[key] + tol:
            better = True
    return better


def pareto_front(pool: list[dict], eps: float) -> list[dict]:
    return [c for c in pool if not any(dominates(o, c, eps) for o in pool if o is not c)]


def front_representative(front: list[dict]) -> dict:
    """The front's MEDIAN epoch — the deterministic tie-break when the metrics cannot separate.

    `median_low` rather than a mean so the result is always an epoch that was actually evaluated
    (and therefore has a checkpoint on disk). See the module docstring for why this replaced the
    old fall-back to `last.pth`.
    """
    ordered = sorted(front, key=lambda r: _f(r, "epoch"))
    return ordered[(len(ordered) - 1) // 2]


def select(rows: list[dict], plateau_tol: float, pareto_eps: float,
           areq_r2_min: float) -> dict:
    """Apply the full rule and return a decision record (no file I/O).

    `decision` is one of:
      selected  — |front| == 1, the three metrics agree on a single epoch
      deferred  — |front| > 1, no label-free separation; the checkpoint is the front's median epoch
                  and the flag is what tells the reader the metrics did not actually decide
      fallback  — the candidate pool is empty, so there is nothing to choose between: use last.pth
    """
    pool, plateau = candidate_pool(rows, plateau_tol, areq_r2_min)
    front = pareto_front(pool, pareto_eps)
    n_conv = sum(1 for r in rows if _flag(r, "converged") and not _flag(r, "collapsed"))
    rec = {
        "policy": "pareto", "policy_name": "plateau-gated SE-aware Pareto",
        "n_evals": len(rows), "n_converged": n_conv, "n_pool": len(pool),
        "front_size": len(front), "front_epochs": [int(_f(r, "epoch")) for r in front],
        "plateau_val_loss": round(plateau, 5) if math.isfinite(plateau) else None,
        "plateau_tol": plateau_tol, "pareto_eps": pareto_eps, "areq_r2_min": areq_r2_min,
    }
    if not front:
        # Nothing cleared the gates. This is an absence of candidates, not a tie — there is no
        # epoch to take the median of, so last.pth is the only defensible artifact.
        rec.update(epoch=None, decision="fallback", source=FALLBACK,
                   reason=(f"no epoch passed the plateau gate (converged={n_conv}, pool=0); "
                           f"falling back to {FALLBACK}"))
        return rec

    unique = len(front) == 1
    row = front[0] if unique else front_representative(front)
    sv = score_vector(row)
    rec.update(
        epoch=int(_f(row, "epoch")), decision="selected" if unique else "deferred", source=None,
        metrics={k: round(_f(row, col), 5) for k, col, _, _ in AXES},
        metric_se={k: round(v, 5) for k, v in se_vector(row).items()},
        val_loss=round(_f(row, "val_loss"), 5), train_loss=round(_f(row, "train_loss"), 5),
        areq_r2=round(_f(row, "areq_r2"), 5), alpha_score=round(sv["alpha_req"], 5),
        reason=(f"unique Pareto-optimal epoch among {len(pool)} plateau-gated candidates "
                f"(val_loss <= {plateau:+.4f}+{plateau_tol}); all three metrics agree at "
                f"eps={pareto_eps} SE" if unique else
                f"no label-free separation: {len(front)} mutually non-dominated epochs "
                f"{rec['front_epochs']} at eps={pareto_eps} SE among {len(pool)} plateau-gated "
                f"candidates; taking the front's median epoch ({int(_f(row, 'epoch'))}) as the "
                f"deterministic representative"),
    )
    return rec


SPS_FILE = "bestSPS.pth"
SPS_TOL = 0.05          # PRE-REGISTERED, see sps_pick
SPS_CONSEC = 3
SPS_TAIL = 10


def sps_pick(rows: list[dict], tol: float = SPS_TOL, consec: int = SPS_CONSEC,
             tail: int = SPS_TAIL) -> dict:
    """Stable Plateau Selection — the simple, deterministic alternative to the Pareto rule.

    Four steps, no gate, no per-metric direction knowledge required:

      1. reference  r_m = median of metric m over the LAST `tail` evaluations
      2. plateau(e) = |m(e) - r_m| <= tol * |r_m|  for ALL THREE metrics
      3. stable(e)  = plateau holds for `consec` consecutive evaluations starting at e
      4. select     = the FIRST such e

    Why the reference is the END of training and not the metric's MAXIMUM
    --------------------------------------------------------------------
    The originally proposed form of this rule set the threshold at 0.995 x max(metric). On these
    runs that selects nothing at all — measured, every seed, every threshold in {99%, 99.5%, 99.9%}.
    The reason is the shape of the curves: RankMe and LiDAR are HIGHEST at initialization (RankMe
    ~85 at epoch 0 versus ~6.4 at epoch 99, because an untrained network scatters features across
    every dimension) while alpha-ReQ peaks near epoch 70. The three "within 0.5% of my own maximum"
    regions therefore sit at OPPOSITE ends of training and their intersection is empty.

    Anchoring to the settled value instead fixes this and buys two further properties:

      * **Direction-agnostic.** The test is a distance, so the rule never needs to know whether
        higher or lower is better for a given metric. The original form had to, and got alpha-ReQ
        backwards: its target is alpha ~ 1, so max(alpha) = 2.28 here is the WORST value, not the best.
      * **No convergence gate needed.** Verified: identical picks with and without the
        converged/non-collapsed filter at every tolerance tried. The untrained epochs are excluded
        automatically because their metrics are nowhere near the settled values — the plateau test
        subsumes the gate rather than needing it bolted on.

    `tol` is PRE-REGISTERED at 0.05 on a label-free criterion: the SMALLEST tolerance at which the
    rule reaches a decision for every run (measured: 1/5 runs decide at 2%, 5/5 at 5%). It was NOT
    chosen by looking at downstream AUC. The sensitivity across {2, 5, 10, 15, 20}% is reported.
    """
    ep = [int(_f(r, "epoch")) for r in rows]
    ref, ok = {}, [True] * len(rows)
    for key, col, _, _ in AXES:
        vals = [_f(r, col) for r in rows]
        finite = [v for v in vals if math.isfinite(v)]
        if len(finite) < tail:
            return {"policy": "sps", "epoch": None, "decision": "fallback", "source": FALLBACK,
                    "tol": tol, "consec": consec, "reason": "too few evaluations to define a plateau"}
        r_m = sorted(finite[-tail:])[tail // 2]
        ref[key] = r_m
        for i, v in enumerate(vals):
            ok[i] = ok[i] and math.isfinite(v) and abs(v - r_m) <= tol * abs(r_m)

    rec = {"policy": "sps", "policy_name": f"Stable Plateau Selection (tol={tol}, consec={consec})",
           "tol": tol, "consec": consec, "tail": tail,
           "reference": {k: round(v, 5) for k, v in ref.items()},
           "n_plateau": sum(ok), "n_evals": len(rows)}
    for i in range(len(ok) - consec + 1):
        if all(ok[i:i + consec]):
            row = rows[i]
            rec.update(epoch=ep[i], decision="selected",
                       metrics={k: round(_f(row, col), 5) for k, col, _, _ in AXES},
                       val_loss=round(_f(row, "val_loss"), 5),
                       reason=(f"first epoch where all three metrics sit within {tol:.0%} of their "
                               f"settled value for {consec} consecutive evaluations"))
            return rec
    rec.update(epoch=None, decision="fallback", source=FALLBACK,
               reason=f"no run of {consec} consecutive plateau evaluations at tol={tol}")
    return rec


def naive_picks(rows: list[dict]) -> dict:
    """The UNGATED baseline: argmax of each metric over every evaluated epoch.

    This is the direct transposition of RankMe / LiDAR / alpha-ReQ to epoch selection, with no
    convergence gate, no collapse gate and no plateau gate — i.e. what you get if you take "higher
    effective rank is better" at face value. It exists as a comparison arm, not as a recommendation:
    all three metrics are HIGHEST at (or near) random initialization, because an untrained encoder
    scatters features across every dimension, so this rule reliably selects a barely-trained epoch
    (lesson L1; measured here: RankMe 86-102 untrained versus 6-20 trained).

    Reporting it matters for the paper's argument. Comparing the proposed rule only against the
    GATED variants would hide why the gates are needed at all; comparing it only against this one
    would be a straw man. Both arms are therefore reported, and the gap between them is precisely
    the value contributed by the convergence + collapse gates.

    Returns {metric_key: record}; `epoch` is the picked epoch and `score` its "higher is better"
    value (for alpha-ReQ that is -|alpha - 1|).
    """
    out: dict[str, dict] = {}
    for key, col, se_col, pretty in AXES:
        cand = [r for r in rows if math.isfinite(score_vector(r)[key])]
        if not cand:
            continue
        best = max(cand, key=lambda r: score_vector(r)[key])
        out[key] = {
            "policy": f"naive_{key}", "policy_name": f"ungated argmax {pretty}",
            "metric": key, "metric_name": pretty,
            "epoch": int(_f(best, "epoch")),
            "metric_value": round(_f(best, col), 5),
            "score": round(score_vector(best)[key], 5),
            "metric_se": round(_f(best, se_col), 5),
            "val_loss": round(_f(best, "val_loss"), 5),
            "areq_r2": round(_f(best, "areq_r2"), 5),
            "converged": _flag(best, "converged"), "collapsed": _flag(best, "collapsed"),
            "n_evals": len(rows),
            "reason": (f"argmax {pretty} over all {len(rows)} evaluated epochs with NO convergence "
                       f"or plateau gate (baseline arm; see naive_picks docstring)"),
        }
    return out


def print_naive(name: str, picks: dict) -> None:
    print("=" * 78)
    print(f"{name}   [ungated baseline rules]")
    if not picks:
        print("  (no finite metric values)")
        return
    for key, _, _, pretty in AXES:
        p = picks.get(key)
        if not p:
            continue
        # `converged` is the diagnostic that carries the argument: a False here means this rule
        # selected an epoch the training loop itself had not yet judged to have learned anything.
        flag = "" if p["converged"] else "   <- NOT converged at this epoch"
        print(f"  {pretty:<10} epoch {p['epoch']:>3}  value={p['metric_value']:>9.4f}  "
              f"val_loss={p['val_loss']:+.4f}{flag}")


def run_cfg(run_dir: str) -> dict:
    """Read the resolved cfg out of any checkpoint in the run (for areq_r2_min / provenance)."""
    for name in (FALLBACK, "bestRankMe.pth", "best.pth"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False).get("cfg", {}) or {}
            except Exception:
                continue
    return {}


def copy_epoch(run_dir: str, epoch: int | None, dst_name: str, rec: dict) -> str | None:
    """Copy one epoch's checkpoint to `dst_name` and stamp the decision record inside it.

    `epoch is None` means "no candidate existed" and resolves to last.pth. The stamped `selection`
    block is what lets a downstream table trace a result back to the rule that produced it without
    re-reading the run directory.
    """
    if epoch is not None:
        src = os.path.join(run_dir, f"ckpt_e{epoch:03d}.pth")
        if not os.path.exists(src):
            print(f"  [!] chosen epoch {epoch} has no ckpt_e{epoch:03d}.pth — this run predates "
                  f"per-eval checkpointing; re-run training or pick from the saved grid.")
            return None
    else:
        src = os.path.join(run_dir, FALLBACK)
        if not os.path.exists(src):
            print(f"  [!] fallback {FALLBACK} missing")
            return None
    dst = os.path.join(run_dir, dst_name)
    shutil.copyfile(src, dst)
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    ck["selection"] = rec
    torch.save(ck, dst)
    return dst


def materialize(run_dir: str, rec: dict) -> str | None:
    """Write bestPareto.pth for the Proposed rule.

    Both `selected` and `deferred` name a concrete epoch (the latter via the front's median), so
    both resolve to that epoch's checkpoint. Only `fallback` — an empty candidate pool — uses
    last.pth.
    """
    return copy_epoch(run_dir, rec["epoch"], OUT_NAME, rec)


def print_report(name: str, rec: dict) -> None:
    print("=" * 78)
    print(f"{name}   [{rec['policy_name']}]")
    print(f"  evals={rec['n_evals']}  converged={rec['n_converged']}  "
          f"plateau-gated pool={rec['n_pool']}  (plateau val_loss={rec['plateau_val_loss']}, "
          f"tol={rec['plateau_tol']})")
    print(f"  Pareto front size = {rec['front_size']}   epochs={rec['front_epochs']}   "
          f"(eps={rec['pareto_eps']} SE)")
    if rec["epoch"] is not None:
        m = rec["metrics"]
        tag = "SELECTED" if rec["decision"] == "selected" else "DEFERRED -> front median"
        print(f"  -> {tag} epoch {rec['epoch']}  rankme={m['rankme']} lidar={m['lidar']} "
              f"alpha={m['alpha_req']} (areq_r2={rec['areq_r2']}) val_loss={rec['val_loss']}")
    else:
        print(f"  -> FALLBACK to {rec['source']}")
    print(f"  {rec['reason']}")


def summary_row(name: str, rec: dict) -> dict:
    row = {"config": name, "policy": rec["policy"], "decision": rec["decision"],
           "epoch": rec["epoch"] if rec["epoch"] is not None else "",
           "front_size": rec["front_size"],
           "front_epochs": " ".join(str(e) for e in rec["front_epochs"]),
           "n_evals": rec["n_evals"], "n_converged": rec["n_converged"], "n_pool": rec["n_pool"],
           "plateau_val_loss": rec["plateau_val_loss"], "plateau_tol": rec["plateau_tol"],
           "pareto_eps": rec["pareto_eps"]}
    for key, _, _, _ in AXES:
        row[f"at_{key}"] = rec.get("metrics", {}).get(key, "")
    return row


def write_summary(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[summary] wrote {path}  ({len(rows)} runs)")


def sweep(run_dirs: list[str], tols: list[float], epss: list[float],
          areq_r2_min: float) -> list[dict]:
    """Robustness of the rule to its two free parameters — a REQUIRED report, not an optional extra.

    `plateau_tol` and `pareto_eps` are researcher degrees of freedom, and the methodology doc's
    multiple-comparison section says such knobs must be pre-registered rather than tuned. Printing
    the whole grid is what makes that check auditable: if the CHOSEN EPOCH moves a lot across the
    grid, the rule is reading noise, and no parameter pair that happens to maximize the number of
    unique fronts can fix that.

    Two things to read off the grid, and the second matters more now that a wide front still yields
    a checkpoint (the front median): `unique` counts how often the metrics agreed outright, while the
    epoch column shows how far the actual artifact moves. A rule whose epoch is stable across the
    grid is robust even where its front size is not.
    """
    names = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    cached = {d: load_rows(d) for d in run_dirs}
    out = []
    header = f"{'eps / tol':>13s} " + " ".join(f"{n.replace('simsiam_', '')[:15]:>15s}" for n in names)
    print(header + "   unique")
    for tol in tols:
        for eps in epss:
            cells, n_sel = [], 0
            for d in run_dirs:
                rec = select(cached[d], tol, eps, areq_r2_min)
                cells.append(f"{rec['front_size']}:{rec['epoch'] if rec['epoch'] is not None else '-'}")
                n_sel += rec["decision"] == "selected"
                out.append({"plateau_tol": tol, "pareto_eps": eps,
                            "config": os.path.basename(os.path.normpath(d)),
                            "front_size": rec["front_size"], "epoch": rec["epoch"] or "",
                            "decision": rec["decision"]})
            print(f"{eps:>5.1f} / {tol:<5.3f} " + " ".join(f"{c:>15s}" for c in cells)
                  + f"   {n_sel}/{len(run_dirs)}")
    print("\ncell = <front size>:<chosen epoch, '-' = empty pool>;  "
          "'unique' = runs whose front held exactly one epoch")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs="+", required=True, help="one or more run directories")
    ap.add_argument("--plateau_tol", type=float, default=0.01,
                    help="an epoch is 'on the plateau' if val_loss <= min(val_loss)+tol (default 0.01)")
    ap.add_argument("--pareto_eps", type=float, default=1.0,
                    help="domination must exceed eps x jackknife SE on an axis (0 = strict Pareto)")
    ap.add_argument("--areq_r2_min", type=float, default=None,
                    help="override the run's power-law R^2 floor (default: read from the run cfg)")
    ap.add_argument("--write", action="store_true", help=f"materialize {OUT_NAME}")
    ap.add_argument("--summary", help="also write a cross-run CSV here")
    ap.add_argument("--sweep", action="store_true",
                    help="report the (plateau_tol x pareto_eps) sensitivity grid instead of selecting")
    ap.add_argument("--baselines", action="store_true",
                    help="report (and with --write, materialize) the UNGATED argmax baseline arms "
                         "bestNaive{RankMe,LiDAR,A-ReQ}.pth instead of the Proposed rule")
    ap.add_argument("--sps", action="store_true",
                    help=f"apply Stable Plateau Selection instead -> {SPS_FILE}")
    ap.add_argument("--sps_tol", type=float, default=SPS_TOL,
                    help=f"SPS plateau tolerance (pre-registered {SPS_TOL}; see sps_pick)")
    ap.add_argument("--sps_sweep", action="store_true",
                    help="report SPS across tolerances instead of selecting (sensitivity audit)")
    args = ap.parse_args()

    if args.sps or args.sps_sweep:
        rows_out = []
        for run_dir in args.run:
            name = os.path.basename(os.path.normpath(run_dir))
            rows = load_rows(run_dir)
            if not rows:
                print(f"[skip] {name}: no metrics.csv")
                continue
            if args.sps_sweep:
                cells = []
                for t in (0.02, 0.05, 0.10, 0.15, 0.20):
                    r = sps_pick(rows, t)
                    cells.append(f"{t:.0%}:{r['epoch'] if r['epoch'] is not None else '-'}")
                    rows_out.append({"config": name, "tol": t, "epoch": r["epoch"] or "",
                                     "decision": r["decision"], "n_plateau": r.get("n_plateau", "")})
                print(f"{name:<38} " + "  ".join(f"{c:>9}" for c in cells))
                continue
            rec = sps_pick(rows, args.sps_tol)
            print("=" * 78)
            print(f"{name}   [{rec.get('policy_name', 'SPS')}]")
            print(f"  plateau evals={rec.get('n_plateau')}/{rec.get('n_evals')}   "
                  f"reference={rec.get('reference')}")
            print(f"  -> {'SELECTED epoch ' + str(rec['epoch']) if rec['epoch'] is not None else 'FALLBACK ' + FALLBACK}")
            print(f"  {rec['reason']}")
            if args.write:
                dst = copy_epoch(run_dir, rec["epoch"], SPS_FILE, rec)
                if dst:
                    print(f"  [write] {dst}")
            rows_out.append({"config": name, **{k: v for k, v in rec.items()
                                                if not isinstance(v, dict)}})
        if args.summary:
            write_summary(args.summary, rows_out)
        return

    if args.baselines:
        rows_out = []
        for run_dir in args.run:
            name = os.path.basename(os.path.normpath(run_dir))
            rows = load_rows(run_dir)
            if not rows:
                print(f"[skip] {name}: no metrics.csv")
                continue
            picks = naive_picks(rows)
            print_naive(name, picks)
            for key, rec in picks.items():
                if args.write:
                    dst = copy_epoch(run_dir, rec["epoch"], NAIVE_FILES[key], rec)
                    if dst:
                        print(f"  [write] {dst}")
                rows_out.append({"config": name, **rec})
            print()
        if args.summary:
            write_summary(args.summary, rows_out)
        return

    if args.sweep:
        r2 = args.areq_r2_min if args.areq_r2_min is not None else 0.80
        grid = sweep(args.run, [0.005, 0.01, 0.02], [0.0, 0.5, 1.0, 2.0], r2)
        if args.summary:
            write_summary(args.summary, grid)
        return

    rows_out = []
    for run_dir in args.run:
        name = os.path.basename(os.path.normpath(run_dir))
        rows = load_rows(run_dir)
        if not rows:
            print(f"[skip] {name}: no metrics.csv")
            continue
        r2_min = (args.areq_r2_min if args.areq_r2_min is not None
                  else float(run_cfg(run_dir).get("areq_r2_min", 0.0) or 0.0))
        rec = select(rows, args.plateau_tol, args.pareto_eps, r2_min)
        print_report(name, rec)
        if args.write:
            dst = materialize(run_dir, rec)
            if dst:
                print(f"  [write] {dst}")
        rows_out.append(summary_row(name, rec))
    if args.summary:
        write_summary(args.summary, rows_out)


if __name__ == "__main__":
    main()
