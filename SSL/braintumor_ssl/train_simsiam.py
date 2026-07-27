"""SimSiam SSL pretraining loop for BraTS 2021.

Run (full):
  python -m braintumor_ssl.train_simsiam --config configs/simsiam_brats.yaml

Smoke test (tiny, CPU-friendly, 1 epoch):
  python -m braintumor_ssl.train_simsiam --config configs/simsiam_brats.yaml \
      --splits_file splits/smoke.json --roi_size 32 32 32 --batch_size 2 \
      --epochs 1 --num_workers 0 --out_dir checkpoints/smoke --smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from braintumor_ssl.data import load_splits, make_views
from braintumor_ssl.models import SimSiam, simsiam_loss, vc_regularizer, whitening_mse_loss
from braintumor_ssl.tracking import Tracker
from braintumor_ssl.utils import (
    AverageMeter,
    alignment,
    alpha_req_fit,
    cosine_lr,
    delete_d_jackknife_se,
    lidar,
    participation_ratio,
    rankme,
    recompute_bn_stats,
    representation_std,
    resolve_device,
    save_checkpoint,
    scaled_lr,
    set_seed,
    uniformity,
    use_local_tmpdir,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", help="GPU index (0/1/2/...), 'cuda', or 'cpu'; default = auto")
    # every config key is overridable on the CLI
    ap.add_argument("--data_root")
    ap.add_argument("--splits_file")
    ap.add_argument("--modalities", nargs="+")
    ap.add_argument("--roi_size", nargs=3, type=int)
    ap.add_argument("--crop_mode", choices=["brain", "tumor", "tumor_margin"])
    ap.add_argument("--tumor_margin", type=int)
    ap.add_argument("--center_mode", choices=["wt_centroid", "bbox_center"])
    ap.add_argument("--mask_out_non_tumor", action="store_true", default=None)
    ap.add_argument("--aug_preset", choices=["standard", "gentle", "clinical", "auto"])
    ap.add_argument("--normalize_after_crop", action="store_true", default=None)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--backbone")
    ap.add_argument("--feature_dim", type=int)
    ap.add_argument("--proj_dim", type=int)
    ap.add_argument("--proj_hidden_dim", type=int)
    ap.add_argument("--pred_hidden_dim", type=int)
    ap.add_argument("--optimizer")
    ap.add_argument("--base_lr", type=float)
    ap.add_argument("--weight_decay", type=float)
    ap.add_argument("--momentum", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--warmup_epochs", type=int)
    ap.add_argument("--grad_clip", type=float)
    ap.add_argument("--fix_pred_lr", action="store_true", default=None)
    ap.add_argument("--amp", action="store_true", default=None)
    ap.add_argument("--out_dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save_every", type=int)
    ap.add_argument("--log_every", type=int)
    # label-free validation monitoring (0 = off; saves best.pth by RankMe when > 0)
    ap.add_argument("--val_every", type=int)
    ap.add_argument("--min_delta_rankme", type=float)
    # best.pth convergence gate: only rank an epoch by RankMe once it has genuinely learned
    ap.add_argument("--best_val_loss_max", type=float,
                    help="epoch eligible for best.pth only if val_loss <= this (default -0.8)")
    ap.add_argument("--best_align_max", type=float,
                    help="and alignment <= this (default 0.15); blocks the high-RankMe random-init epoch")
    # optional VICReg-style variance/covariance regularizer (0 = off; counters dimensional collapse)
    ap.add_argument("--vicreg_var", type=float, help="variance-term weight (0=off)")
    ap.add_argument("--vicreg_cov", type=float, help="covariance-term weight (0=off)")
    ap.add_argument("--vicreg_gamma", type=float, help="target per-dim std for the variance hinge (default 1.0)")
    ap.add_argument("--aug_strength", type=float, help="scale clinical intensity-aug magnitude (default 1.0)")
    ap.add_argument("--wmse_weight", type=float, help="W-MSE whitening-loss weight (0=off; needs proj_dim<batch)")
    ap.add_argument("--wmse_eps", type=float, help="shrinkage for the whitening covariance (default 1e-3)")
    ap.add_argument("--early_stop", action="store_true", default=None)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--min_epochs", type=int)
    ap.add_argument("--collapse_stop", action="store_true", default=None)
    ap.add_argument("--collapse_patience", type=int)
    # experiment tracking (MLflow -> DagsHub); URI + credentials come from the environment
    ap.add_argument("--mlflow", action="store_true", default=None)
    ap.add_argument("--mlflow_experiment")
    ap.add_argument("--mlflow_system_metrics", action="store_true", default=None,
                    help="also log CPU/RAM/GPU utilization (requires psutil; pynvml for GPU)")
    ap.add_argument("--no_mlflow", action="store_true", help="force-disable tracking (timing/debug runs)")
    ap.add_argument("--resume")
    ap.add_argument("--cache", action="store_true", default=None)
    ap.add_argument("--smoke", action="store_true", help="cap iterations/epoch for a fast dry run")
    return ap.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if k in ("config", "smoke", "no_mlflow"):
            continue
        if v is not None:
            cfg[k] = v
    cfg["smoke"] = args.smoke
    if args.no_mlflow:                 # explicit off wins over the config's mlflow: true
        cfg["mlflow"] = False
    return cfg


def build_optimizer(model: SimSiam, cfg: dict, lr: float):
    if cfg.get("fix_pred_lr"):
        # predictor kept at the (un-decayed) base lr; rest follows the cosine schedule
        params = [
            {"params": model.backbone.parameters(), "fix_lr": False},
            {"params": model.encoder_head.parameters(), "fix_lr": False},
            {"params": model.projector.parameters(), "fix_lr": False},
            {"params": model.predictor.parameters(), "fix_lr": True},
        ]
    else:
        params = [{"params": model.parameters(), "fix_lr": False}]

    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")


# --------------------------------------------------------------------------- #
# Label-free validation monitoring (optional; mirrors the reference evaluate_ssl)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model: SimSiam, val_records: list, cfg: dict, device: str, use_amp: bool) -> dict:
    """Label-free validation on the split's val subjects.

    Representation health (z_std, RankMe, participation ratio, uniformity) is measured on the
    deterministic single (eval) view AFTER a precise-BN pass over the val cohort — the same
    reason evaluate.py / extract_features.py do it (SimSiam's L2-normalized loss leaves BN
    running stats unconstrained, so raw eval-mode feature scale/rank can be misleading). The BN
    stats are snapshotted and restored so the training trajectory is untouched. Val loss and
    alignment then use the model's own (restored) BN on two augmented views. No labels, no grads.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    was_training = model.training
    # Validation is no longer a cheap side-show: 200 val subjects x (precise-BN pass + eval view +
    # lidar_views augmented views) is ~2400 volume-loads per epoch and it runs EVERY epoch, so with
    # the old hard cap of 2 workers it cost more wall time than the training epoch itself. It gets
    # its own worker count because nothing else is running while it does.
    nw = int(cfg.get("val_num_workers", 0) or min(cfg["num_workers"], 2))
    bs = cfg["batch_size"]
    max_bn = 2 if cfg["smoke"] else 200

    eval_ld = DataLoader(make_views(val_records, cfg, mode="eval"), batch_size=max(bs, 2),
                         shuffle=False, num_workers=nw, pin_memory=(device == "cuda"))

    # --- representation metrics on precise-BN-adapted features, then restore BN ---
    bn_mods = [m for m in model.modules() if isinstance(m, _BatchNorm) and m.running_mean is not None]
    snapshot = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                for m in bn_mods]
    recompute_bn_stats(model, eval_ld, device, max_batches=max_bn)  # adapts encoder-path BN; leaves model.eval()
    feats = []
    for it, batch in enumerate(eval_ld):
        if cfg["smoke"] and it >= 2:
            break
        with torch.amp.autocast("cuda", enabled=use_amp):
            h = model.encode(batch["image"].to(device, non_blocking=True))
        feats.append(h.float().cpu())
    feats = torch.cat(feats, dim=0)
    for m, (rm, rv, nb) in zip(bn_mods, snapshot):     # put training's BN state back
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb)

    # --- val loss + alignment + LiDAR's surrogate classes, with the model's own BN (eval mode) ---
    # LiDAR treats each subject as a class and its augmented views as that class's samples, so its
    # within-subject scatter has (A-1) degrees of freedom per subject. At the A=2 minimum that is 1
    # dof for a 256x256 matrix and the ridge does the work; A>2 is what makes the estimate real. The
    # extra views cost one crop+augment each, NOT another volume read (data.BraTSViews.n_views).
    n_views = max(2, int(cfg.get("lidar_views", 2)))
    model.eval()
    two_ld = DataLoader(make_views(val_records, cfg, mode="train", n_views=n_views), batch_size=bs,
                        shuffle=False, num_workers=nw, pin_memory=(device == "cuda"))
    losses = []
    hs: list[list[torch.Tensor]] = [[] for _ in range(n_views)]
    for it, batch in enumerate(two_ld):
        if cfg["smoke"] and it >= 2:
            break
        x1 = batch["view1"].to(device, non_blocking=True)
        x2 = batch["view2"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x1, x2)                 # views 1-2 also give the val loss + alignment
            loss = simsiam_loss(out)
            hs[0].append(out["h1"].float().cpu())   # 256-d encoder features, view 1 (paired w/ h2)
            hs[1].append(out["h2"].float().cpu())   # 256-d encoder features, view 2
            for v in range(2, n_views):             # extra LiDAR-only views: encoder path only
                hs[v].append(model.encode(batch[f"view{v + 1}"].to(device, non_blocking=True))
                             .float().cpu())
        losses.append(float(loss.item()))
    views = [torch.cat(h) for h in hs]
    h1, h2 = views[0], views[1]

    if was_training:
        model.train()
    # The three checkpoint-selection metrics are all computed here every epoch:
    #   rankme    — effective rank of the single (eval) view features (want HIGH)
    #   lidar     — augmentation-invariant rank from the two paired views (want HIGH)
    #   alpha_req — power-law decay exponent of the eval-view spectrum (want CLOSE TO 1)
    # alpha-ReQ's fit window is a config knob, not a default: with N < d the smallest sample
    # eigenvalues are biased down and steepen the slope, so only the leading fraction of the
    # spectrum is fitted (calibrated on synthetic power-law spectra; see the config comment).
    areq_kw = {"k_min": int(cfg.get("areq_k_min", 1)),
               "k_max_frac": float(cfg.get("areq_k_max_frac", 1.0))}
    fit = alpha_req_fit(feats, **areq_kw)

    # Sampling uncertainty of each metric OVER SUBJECTS, by delete-d jackknife. This is what makes
    # "is epoch B better than epoch A?" answerable: at n=200 RankMe's SE is ~0.5, i.e. TEN TIMES the
    # fixed min_delta_rankme default, so a fixed threshold selects noise. ~2 s/epoch for all three.
    reps = int(cfg.get("select_se_reps", 0) or 0)
    se_kw = {"drop_frac": float(cfg.get("select_se_drop_frac", 0.1)), "reps": reps,
             "seed": int(cfg.get("seed", 42))}
    nan = float("nan")
    se_jk = {"rankme": nan, "lidar": nan, "alpha_req": nan}
    if reps >= 2:
        se_jk["rankme"] = delete_d_jackknife_se(rankme, feats, **se_kw)
        se_jk["lidar"] = delete_d_jackknife_se(lidar, *views, **se_kw)
        se_jk["alpha_req"] = delete_d_jackknife_se(
            lambda Z: alpha_req_fit(Z, **areq_kw)["alpha"], feats, **se_kw)
    return {
        "val_loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "z_std": representation_std(feats),
        "rankme": rankme(feats),
        "lidar": lidar(*views),
        "lidar_views": len(views),
        "alpha_req": fit["alpha"],
        "areq_r2": fit["r2"],
        "areq_se": fit["se"],
        "areq_k_used": fit["k_used"],
        "se_jk": se_jk,                      # per-metric sampling SE over subjects (nan if disabled)
        "participation_ratio": participation_ratio(feats),
        "alignment": alignment(h1, h2),
        "uniformity": uniformity(feats),
        "n_val": int(feats.shape[0]),
        # raw features, for the per-epoch dump — every spectral quantity above can then be
        # recomputed post-hoc (different fit window, projection, resampling) without retraining
        "_features": {"eval": feats, **{f"view{i + 1}": v for i, v in enumerate(views)}},
    }


def dump_val_features(cfg: dict, epoch: int, feats: dict) -> None:
    """Write this epoch's val features to <out_dir>/val_features/ep####.npz (opt-in).

    ~0.6 MB per validation at 200 val subjects x 256-d x 3 matrices (eval view + the two augmented
    views LiDAR pairs). Cheap insurance with a large payoff: the fit window, the RankMe/LiDAR
    estimators, resampling-based selection stability and any lower-dimensional projection check can
    all be recomputed from these AFTER the run, instead of having to be decided correctly before it.
    """
    if not cfg.get("dump_val_features", False):
        return
    d = os.path.join(cfg["out_dir"], "val_features")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"ep{epoch:04d}.npz"),
                        **{k: v.numpy().astype(np.float32) for k, v in feats.items()})


def collapse_thresholds(n_val: int, cfg: dict) -> dict:
    """Full thresholds for a real val set; relaxed ones for a tiny (smoke) val where the
    label-free metrics are not yet meaningful."""
    if n_val < 10:
        return {"z_std": 0.003, "rankme": 1.01, "pr": 1.01}
    return {"z_std": cfg.get("z_std_min", 0.01),
            "rankme": cfg.get("rankme_min", 5.0),
            "pr": cfg.get("pr_min", 5.0)}


def is_collapsed(m: dict, th: dict) -> tuple[bool, str]:
    """Collapse gate: non-finite metrics or a near-zero representation std."""
    if any(v is None or not math.isfinite(v)
           for v in (m["z_std"], m["rankme"], m["participation_ratio"], m["uniformity"])):
        return True, "non_finite"
    if m["z_std"] < th["z_std"]:
        return True, "z_std_low"
    return False, "ok"


def is_converged(m: dict, cfg: dict) -> bool:
    """Best-checkpoint eligibility gate: only rank an epoch by RankMe once the encoder has
    actually learned view-invariance. Without this, RankMe is *highest at random init* — the
    untrained encoder scatters features across all dims, inflating effective rank — so a pure
    max-RankMe rule saves an early, essentially-untrained epoch as best.pth. Convergence is read
    from the SSL loss itself: val_loss near -1 (negative-cosine floor) and low alignment (the two
    views actually map together) both signal genuine learning, not random spread."""
    if cfg.get("smoke"):
        return True                          # smoke exercises the save paths; 2 epochs never "converge"
    return (m["val_loss"] <= cfg.get("best_val_loss_max", -0.8)
            and m["alignment"] <= cfg.get("best_align_max", 0.15))


# --------------------------------------------------------------------------- #
# Multi-metric checkpoint selection (RankMe / LiDAR / alpha-ReQ)
# --------------------------------------------------------------------------- #
# One best checkpoint per metric. Eligibility (converged + non-collapsed + finite) is SHARED so a
# random-init / collapsed epoch can never win any of them (lesson L13). Improvement is expressed as a
# single "score to MAXIMIZE" so the three read identically: RankMe/LiDAR use the value itself (higher
# better); alpha-ReQ uses -|alpha - 1| because alpha=1 is the optimal-transfer target (Agrawal 2022),
# so the epoch closest to 1 scores highest. `last.pth` (final plateau) stays the backbone-comparison
# basis; these are the "which selection rule wins downstream?" options the D3 ablation will judge.
def metric_specs(cfg: dict) -> list[dict]:
    return [
        {"key": "rankme", "name": "RankMe", "file": "bestRankMe.pth",
         "score": lambda m: m["rankme"], "delta": float(cfg.get("min_delta_rankme", 0.0) or 0.0),
         "goal": "max RankMe"},
        {"key": "lidar", "name": "LiDAR", "file": "bestLiDAR.pth",
         "score": lambda m: m["lidar"], "delta": float(cfg.get("min_delta_lidar", 0.0) or 0.0),
         "goal": "max LiDAR (augmentation-invariant rank)"},
        # alpha-ReQ additionally needs its power-law fit to actually HOLD: the OLS slope is defined
        # for any spectrum, so without an R^2 floor a badly non-power-law epoch can post an alpha
        # near 1 by accident and win. Epochs below the floor score -inf, i.e. are never selected.
        {"key": "alpha_req", "name": "alpha-ReQ", "file": "bestA-ReQ.pth",
         "score": lambda m: (-abs(m["alpha_req"] - 1.0)
                             if (m.get("areq_r2") is None or not math.isfinite(m.get("areq_r2", float("nan")))
                                 or m["areq_r2"] >= float(cfg.get("areq_r2_min", 0.0)))
                             else float("-inf")),
         "delta": float(cfg.get("min_delta_areq", 0.0) or 0.0), "goal": "alpha-ReQ closest to 1.0"},
    ]


def selection_record(spec: dict, epoch: int, vm: dict, train_loss: float, lr: float,
                     smoothed: float = float("nan"), threshold: float = float("nan"),
                     se: float = float("nan"), window: int = 1) -> dict:
    """Provenance stored inside each best*.pth: which metric picked it, its value, the OTHER two
    metrics at that epoch, the losses, the LR, the decision rule that fired, and a readable reason."""
    return {
        "metric": spec["key"], "metric_name": spec["name"],
        "metric_value": round(float(vm[spec["key"]]), 5), "goal": spec["goal"], "epoch": int(epoch),
        "other_metrics": {k: round(float(vm[k]), 5) for k in ("rankme", "lidar", "alpha_req")
                          if k != spec["key"]},
        "train_loss": round(float(train_loss), 5), "val_loss": round(float(vm["val_loss"]), 5),
        "lr": float(lr),
        # alpha's fit diagnostics travel with EVERY checkpoint, not just the alpha-selected one:
        # an alpha reported without its R^2 and its fit window is not interpretable.
        "areq_fit": {"r2": round(float(vm["areq_r2"]), 5), "se": round(float(vm["areq_se"]), 5),
                     "k_used": int(vm["areq_k_used"])},
        "n_val": int(vm["n_val"]),
        # what actually decided this save: the smoothed score, the sampling SE it had to clear, and
        # how many evals the trailing window held (a window of 1 means smoothing was off/warming up)
        "rule": {"smoothed_score": round(float(smoothed), 5), "threshold": round(float(threshold), 5),
                 "metric_se": round(float(se), 5), "window": int(window)},
        "reason": (f"{spec['goal']} among converged, non-collapsed epochs; smoothed over "
                   f"{window} eval(s), beat the previous best by more than {threshold:.4f} "
                   f"(selected at epoch {epoch})"),
    }


def write_metrics_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    use_local_tmpdir()                       # avoid the NFS DataLoader-cleanup noise on the cluster
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device"))
    dev_label = f"cuda:{torch.cuda.current_device()}" if device == "cuda" else device
    roi = tuple(cfg["roi_size"])
    print(f"[cfg] device={dev_label} backbone={cfg['backbone']} roi={roi} "
          f"crop_mode={cfg.get('crop_mode','brain')} batch={cfg['batch_size']} feature_dim={cfg['feature_dim']}")

    # ---- data ----
    split = load_splits(cfg["splits_file"])
    train_ds = make_views(split["train"], cfg, mode="train", cache=bool(cfg.get("cache")))
    train_ld = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
    )
    print(f"[data] train subjects={len(train_ds)} iters/epoch={len(train_ld)}")
    val_records = split.get("val", []) if cfg.get("val_every", 0) else []
    if val_records:
        print(f"[data] val subjects={len(val_records)} (label-free monitor every {cfg['val_every']} ep)")

    # ---- model ----
    model = SimSiam(
        backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
        feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
        proj_hidden_dim=cfg["proj_hidden_dim"], pred_hidden_dim=cfg["pred_hidden_dim"],
    ).to(device)
    print(f"[model] backbone_dim={model.backbone_dim} -> feature_dim={model.feature_dim} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    init_lr = scaled_lr(cfg["base_lr"], cfg["batch_size"])
    optimizer = build_optimizer(model, cfg, init_lr)
    use_amp = bool(cfg["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if cfg.get("resume"):
        ck = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        print(f"[resume] from {cfg['resume']} at epoch {start_epoch}")

    total_steps = cfg["epochs"] * len(train_ld)
    warmup_steps = cfg["warmup_epochs"] * len(train_ld)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # multi-metric best-checkpoint / early-stop monitor state (used only when val_every > 0).
    # One tracker per metric: best score-so-far, the epoch that achieved it, and a plateau counter.
    specs = metric_specs(cfg)
    best = {s["key"]: {"score": -float("inf"), "epoch": -1, "no_improve": 0, "value": float("nan"),
                       "hist": []}                       # trailing scores for the smoothing window
            for s in specs}
    smooth_w = max(1, int(cfg.get("select_smooth_window", 1)))
    se_mult = float(cfg.get("select_se_mult", 0.0) or 0.0)
    collapse_run = 0
    metrics_history: list[dict] = []

    # optional experiment tracking (MLflow -> DagsHub); no-op unless cfg['mlflow'] is true
    tracker = Tracker(cfg, run_name=os.path.basename(os.path.normpath(cfg["out_dir"])),
                      tags={"backbone": cfg["backbone"], "crop_mode": cfg.get("crop_mode", "brain")})
    tracker.log_params(cfg)
    tracker.log_dict(cfg, "config/resolved_config.json")
    tracker.log_artifact(args.config, artifact_path="config")

    # optional VICReg-style variance/covariance regularizer on the projector embeddings
    # (off by default; counters dimensional collapse — see models.vc_regularizer / MODEL_SELECTION)
    w_var, w_cov = float(cfg.get("vicreg_var", 0.0) or 0.0), float(cfg.get("vicreg_cov", 0.0) or 0.0)
    vic_on = w_var > 0.0 or w_cov > 0.0
    if vic_on:
        print(f"[reg] VICReg VC-reg ON: var_w={w_var} cov_w={w_cov} gamma={cfg.get('vicreg_gamma', 1.0)}")
    w_wmse = float(cfg.get("wmse_weight", 0.0) or 0.0)
    wmse_on = w_wmse > 0.0
    if wmse_on:
        if cfg["proj_dim"] >= cfg["batch_size"]:
            print(f"[reg][warn] W-MSE needs proj_dim < batch_size for full-rank whitening "
                  f"(proj_dim={cfg['proj_dim']}, batch={cfg['batch_size']}); whitening will be rank-deficient")
        print(f"[reg] W-MSE whitening ON: weight={w_wmse} eps={cfg.get('wmse_eps', 1e-3)}")

    # ---- train ----
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        loss_m, std_m = AverageMeter(), AverageMeter()
        var_m, cov_m, wmse_m = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        for it, batch in enumerate(train_ld):
            if cfg["smoke"] and it >= 3:
                break
            step = epoch * len(train_ld) + it
            lr = cosine_lr(init_lr, step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = init_lr if pg.get("fix_lr") else lr

            x1 = batch["view1"].to(device, non_blocking=True)
            x2 = batch["view2"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x1, x2)
                loss = simsiam_loss(out)
                if vic_on:
                    var_t, cov_t = vc_regularizer(out, float(cfg.get("vicreg_gamma", 1.0)))
                    loss = loss + w_var * var_t + w_cov * cov_t
                    var_m.update(float(var_t), x1.size(0))
                    cov_m.update(float(cov_t), x1.size(0))
                if wmse_on:
                    wmse_t = whitening_mse_loss(out, float(cfg.get("wmse_eps", 1e-3)))
                    loss = loss + w_wmse * wmse_t
                    wmse_m.update(float(wmse_t), x1.size(0))
            scaler.scale(loss).backward()
            if cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            loss_m.update(loss.item(), x1.size(0))
            std_m.update(representation_std(out["z1"]), x1.size(0))
            if it % cfg["log_every"] == 0:
                print(f"  e{epoch:03d} it{it:04d}/{len(train_ld)} lr={lr:.4f} "
                      f"loss={loss_m.avg:+.4f} z_std={std_m.avg:.4f}")

        expected = 1.0 / (cfg["feature_dim"] ** 0.5)
        vic_str = f" var={var_m.avg:.4f} cov={cov_m.avg:.4f}" if vic_on else ""
        vic_str += f" wmse={wmse_m.avg:+.4f}" if wmse_on else ""
        print(f"[epoch {epoch:03d}] loss={loss_m.avg:+.4f} z_std={std_m.avg:.4f}{vic_str} "
              f"(healthy~{expected:.4f}) time={time.time()-t0:.1f}s")
        train_metrics = {"train_loss": loss_m.avg, "train_z_std": std_m.avg,
                         "lr": optimizer.param_groups[0]["lr"]}
        if vic_on:
            train_metrics.update(train_vic_var=var_m.avg, train_vic_cov=cov_m.avg)
        if wmse_on:
            train_metrics.update(train_wmse=wmse_m.avg)
        tracker.log_metrics(train_metrics, step=epoch)

        is_last = epoch == cfg["epochs"] - 1
        if is_last or (epoch + 1) % cfg["save_every"] == 0:
            state = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "cfg": cfg}
            save_checkpoint(state, os.path.join(cfg["out_dir"], f"ckpt_e{epoch:03d}.pth"))
        # always refresh 'last.pth' for easy resume / extraction
        save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "cfg": cfg},
                        os.path.join(cfg["out_dir"], "last.pth"))

        # ---- optional label-free validation + best-checkpoint / early stop ----
        do_val = bool(val_records) and ((epoch + 1) % cfg["val_every"] == 0 or is_last)
        if do_val:
            vm = validate(model, val_records, cfg, device, use_amp)
            dump_val_features(cfg, epoch, vm.pop("_features"))   # pop: keep tensors out of the ckpt
            th = collapse_thresholds(vm["n_val"], cfg)
            collapsed, why = is_collapsed(vm, th)
            converged = is_converged(vm, cfg)
            print(f"[val {epoch:03d}] loss={vm['val_loss']:+.4f} z_std={vm['z_std']:.4f} "
                  f"rankme={vm['rankme']:.3f} lidar={vm['lidar']:.3f} areq={vm['alpha_req']:.3f} "
                  f"pr={vm['participation_ratio']:.2f} align={vm['alignment']:.4f} "
                  f"unif={vm['uniformity']:+.3f} converged={converged} collapsed={collapsed}"
                  + ("" if not collapsed else f" ({why})"))
            lr_now = optimizer.param_groups[0]["lr"]

            # --- per-metric best-checkpoint update (shared eligibility gate; L13) ---
            # Two guards against selecting a noise spike rather than a better representation:
            #   (a) the score is a trailing mean over the last `select_smooth_window` ELIGIBLE evals
            #       (causal, so it still works online; the saved weights sit at the window's end,
            #       i.e. inside the plateau rather than on its leading edge);
            #   (b) an improvement must clear `select_se_mult` x the metric's own sampling SE over
            #       subjects, not a hand-set constant. min_delta_* survives only as a floor.
            eligible = converged and (not collapsed)
            for s in specs:
                st = best[s["key"]]
                raw = vm[s["key"]]
                score_now = s["score"](vm)
                ok = eligible and math.isfinite(raw) and math.isfinite(score_now)
                if ok:
                    st["hist"].append(score_now)
                    del st["hist"][:-smooth_w]                 # keep only the trailing window
                score = sum(st["hist"]) / len(st["hist"]) if st["hist"] else float("-inf")
                se = vm["se_jk"].get(s["key"], float("nan"))
                thresh = max(s["delta"], se_mult * se if math.isfinite(se) else 0.0)
                improved = ok and score > st["score"] + thresh
                if improved:
                    st["score"], st["epoch"], st["no_improve"], st["value"] = score, epoch, 0, raw
                    sel = selection_record(s, epoch, vm, loss_m.avg, lr_now,
                                           smoothed=score, threshold=thresh, se=se, window=len(st["hist"]))
                    ckpt = {"epoch": epoch, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "cfg": cfg,
                            "val_metrics": vm, "selection": sel}
                    save_checkpoint(ckpt, os.path.join(cfg["out_dir"], s["file"]))
                    if s["key"] == "rankme":            # back-compat alias for evaluate/select_model tooling
                        save_checkpoint(ckpt, os.path.join(cfg["out_dir"], "best.pth"))
                    print(f"  -> new best {s['name']}={raw:.4f} (smoothed {score:.4f} over "
                          f"{len(st['hist'])} evals, cleared +{thresh:.4f}) @ epoch {epoch} "
                          f"(saved {s['file']})")
                elif st["epoch"] >= 0:
                    # only count a plateau once this metric has a converged baseline to plateau from —
                    # pre-convergence evals are "not learned yet", not "improvement stalled".
                    st["no_improve"] += 1

            # --- persist per-epoch record: 3 selection metrics + losses + LR + early-stop state ---
            patience = cfg.get("patience", 20)
            plateaued = {s["key"]: (best[s["key"]]["epoch"] >= 0 and best[s["key"]]["no_improve"] >= patience)
                         for s in specs}
            row = {
                "epoch": epoch, "lr": round(lr_now, 6),
                "train_loss": round(loss_m.avg, 5), "train_z_std": round(std_m.avg, 5),
                "val_loss": round(vm["val_loss"], 5), "z_std": round(vm["z_std"], 5),
                "rankme": round(vm["rankme"], 4), "lidar": round(vm["lidar"], 4),
                "lidar_views": vm["lidar_views"], "alpha_req": round(vm["alpha_req"], 4),
                "areq_r2": round(vm["areq_r2"], 4), "areq_se": round(vm["areq_se"], 5),
                "areq_k_used": vm["areq_k_used"],
                # sampling SE of each metric over subjects — the scale any epoch-to-epoch
                # difference has to beat before it means anything
                "rankme_se_jk": round(vm["se_jk"]["rankme"], 4),
                "lidar_se_jk": round(vm["se_jk"]["lidar"], 4),
                "areq_se_jk": round(vm["se_jk"]["alpha_req"], 5),
                "participation_ratio": round(vm["participation_ratio"], 4),
                "alignment": round(vm["alignment"], 5), "uniformity": round(vm["uniformity"], 5),
                "n_val": vm["n_val"], "converged": converged, "collapsed": collapsed,
                "no_improve_rankme": best["rankme"]["no_improve"],
                "no_improve_lidar": best["lidar"]["no_improve"],
                "no_improve_areq": best["alpha_req"]["no_improve"],
                "stopped": False, "stop_reason": "",
            }
            tracker.log_metrics({"val_loss": vm["val_loss"], "z_std": vm["z_std"], "rankme": vm["rankme"],
                                 "lidar": vm["lidar"], "alpha_req": vm["alpha_req"], "lr": lr_now,
                                 "areq_r2": vm["areq_r2"], "areq_se": vm["areq_se"],
                                 "participation_ratio": vm["participation_ratio"],
                                 "alignment": vm["alignment"], "uniformity": vm["uniformity"],
                                 "converged": float(converged), "collapsed": float(collapsed)}, step=epoch)

            # --- stop conditions (collapse OR all three metrics plateaued) ---
            stop_reason = ""
            collapse_run = collapse_run + 1 if collapsed else 0
            if cfg.get("collapse_stop") and collapse_run >= cfg.get("collapse_patience", 3):
                stop_reason = f"collapse detected {collapse_run}x in a row"
            elif cfg.get("early_stop") and (epoch + 1) >= cfg.get("min_epochs", 0) \
                    and all(plateaued.values()):
                # union rule: only stop once RankMe, LiDAR AND alpha-ReQ have each stalled for `patience`
                # evals, so no per-metric best checkpoint is cut off while still improving.
                stalls = ", ".join(f"{k}(+{best[k]['no_improve']})" for k in plateaued)
                stop_reason = f"all metrics plateaued [{stalls}]"

            if stop_reason:
                row["stopped"], row["stop_reason"] = True, stop_reason
            metrics_history.append(row)
            write_metrics_csv(os.path.join(cfg["out_dir"], "metrics.csv"), metrics_history)
            if stop_reason:
                print(f"[stop] {stop_reason} -> stopping")
                break

    any_best = any(best[s["key"]]["epoch"] >= 0 for s in specs)
    if any_best:
        print("[best] per-metric selected checkpoints (converged, non-collapsed):")
        for s in specs:
            st = best[s["key"]]
            if st["epoch"] >= 0:
                print(f"  {s['name']:>9s}: epoch {st['epoch']:>3d}  value={st['value']:.4f}  -> {s['file']}")
                # record each run's per-metric selection so DagsHub shows which epoch won each rule
                tracker.log_metrics({f"best_{s['key']}_epoch": float(st["epoch"]),
                                     f"best_{s['key']}_value": float(st["value"])})
            else:
                print(f"  {s['name']:>9s}: (no eligible epoch)")
    elif val_records:
        print("[best] NO epoch passed the convergence gate "
              f"(val_loss<={cfg.get('best_val_loss_max', -0.8)}, align<={cfg.get('best_align_max', 0.15)}); "
              "no best*.pth was written -> use last.pth for downstream, and check that training actually "
              "converged (val_loss should approach -1).")
    if metrics_history:
        try:
            from braintumor_ssl.visualize import save_training_curves
            save_training_curves(metrics_history, os.path.join(cfg["out_dir"], "training_curves.png"),
                                 healthy_z_std=1.0 / (cfg["feature_dim"] ** 0.5))
            print(f"[plot] wrote {os.path.join(cfg['out_dir'], 'training_curves.png')}")
        except Exception as e:                       # plotting must never fail a training run
            print(f"[warn] training_curves.png not written: {e}")
    tracker.log_artifact(os.path.join(cfg["out_dir"], "training_curves.png"), artifact_path="figures")
    tracker.log_artifact(os.path.join(cfg["out_dir"], "metrics.csv"), artifact_path="tables")
    tracker.close()
    print(f"[done] checkpoints in {cfg['out_dir']}")


if __name__ == "__main__":
    main()
