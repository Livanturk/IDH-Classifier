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

import torch
import yaml
from torch.utils.data import DataLoader

from braintumor_ssl.data import load_splits, make_views
from braintumor_ssl.models import SimSiam, simsiam_loss, vc_regularizer, whitening_mse_loss
from braintumor_ssl.tracking import Tracker
from braintumor_ssl.utils import (
    AverageMeter,
    alignment,
    cosine_lr,
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
    nw = min(cfg["num_workers"], 2)
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

    # --- val loss + alignment with the model's own BN (eval mode), two augmented views ---
    model.eval()
    two_ld = DataLoader(make_views(val_records, cfg, mode="train"), batch_size=bs,
                        shuffle=False, num_workers=nw, pin_memory=(device == "cuda"))
    losses, h1s, h2s = [], [], []
    for it, batch in enumerate(two_ld):
        if cfg["smoke"] and it >= 2:
            break
        x1 = batch["view1"].to(device, non_blocking=True)
        x2 = batch["view2"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x1, x2)
            loss = simsiam_loss(out)
        losses.append(float(loss.item()))
        h1s.append(out["h1"].float().cpu())
        h2s.append(out["h2"].float().cpu())
    h1, h2 = torch.cat(h1s), torch.cat(h2s)

    if was_training:
        model.train()
    return {
        "val_loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "z_std": representation_std(feats),
        "rankme": rankme(feats),
        "participation_ratio": participation_ratio(feats),
        "alignment": alignment(h1, h2),
        "uniformity": uniformity(feats),
        "n_val": int(feats.shape[0]),
    }


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
    return (m["val_loss"] <= cfg.get("best_val_loss_max", -0.8)
            and m["alignment"] <= cfg.get("best_align_max", 0.15))


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

    # best-checkpoint / early-stop monitor state (used only when val_every > 0)
    best_rankme, best_epoch, no_improve, collapse_run = -float("inf"), -1, 0, 0
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
            th = collapse_thresholds(vm["n_val"], cfg)
            collapsed, why = is_collapsed(vm, th)
            converged = is_converged(vm, cfg)
            print(f"[val {epoch:03d}] loss={vm['val_loss']:+.4f} z_std={vm['z_std']:.4f} "
                  f"rankme={vm['rankme']:.3f} pr={vm['participation_ratio']:.2f} "
                  f"align={vm['alignment']:.4f} unif={vm['uniformity']:+.3f} "
                  f"converged={converged} collapsed={collapsed}" + ("" if not collapsed else f" ({why})"))
            metrics_history.append({
                "epoch": epoch, "train_loss": round(loss_m.avg, 5), "train_z_std": round(std_m.avg, 5),
                "val_loss": round(vm["val_loss"], 5), "z_std": round(vm["z_std"], 5),
                "rankme": round(vm["rankme"], 4), "participation_ratio": round(vm["participation_ratio"], 4),
                "alignment": round(vm["alignment"], 5), "uniformity": round(vm["uniformity"], 5),
                "n_val": vm["n_val"], "converged": converged, "collapsed": collapsed,
            })
            write_metrics_csv(os.path.join(cfg["out_dir"], "metrics.csv"), metrics_history)
            tracker.log_metrics({"val_loss": vm["val_loss"], "z_std": vm["z_std"], "rankme": vm["rankme"],
                                 "participation_ratio": vm["participation_ratio"],
                                 "alignment": vm["alignment"], "uniformity": vm["uniformity"],
                                 "converged": float(converged), "collapsed": float(collapsed)}, step=epoch)

            # best.pth = highest RankMe *among converged, non-collapsed* epochs = the "max retained
            # rank" checkpoint. The convergence gate stops it latching onto the random-init epoch.
            # NOTE (L13): this rule = the earliest-converged epoch and its RankMe VALUE is volatile
            # (steep transient x seed-dependent convergence timing); it is NOT the backbone-comparison
            # basis. The comparison uses the final/plateau checkpoint (last.pth, always saved below);
            # best.pth is kept as the high-rank option whose downstream value D3 (labels) will judge.
            eligible = converged and (not collapsed) and math.isfinite(vm["rankme"])
            improved = eligible and vm["rankme"] > best_rankme + cfg.get("min_delta_rankme", 0.0)
            if improved:
                best_rankme, best_epoch, no_improve = vm["rankme"], epoch, 0
                save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(), "cfg": cfg, "val_metrics": vm},
                                os.path.join(cfg["out_dir"], "best.pth"))
                print(f"  -> new best RankMe={best_rankme:.3f} @ epoch {epoch} (converged; saved best.pth)")
            elif best_epoch >= 0:
                # only count a RankMe plateau once we have a converged baseline to plateau from —
                # pre-convergence evals are "not learned yet", not "improvement stalled".
                no_improve += 1

            collapse_run = collapse_run + 1 if collapsed else 0
            if cfg.get("collapse_stop") and collapse_run >= cfg.get("collapse_patience", 3):
                print(f"[stop] collapse detected {collapse_run}x in a row -> stopping")
                break
            if cfg.get("early_stop") and (epoch + 1) >= cfg.get("min_epochs", 0) \
                    and no_improve >= cfg.get("patience", 20):
                print(f"[stop] RankMe plateaued for {no_improve} evals -> early stop")
                break

    if best_epoch >= 0:
        print(f"[best] converged, max-RankMe={best_rankme:.3f} @ epoch {best_epoch} "
              f"-> {os.path.join(cfg['out_dir'], 'best.pth')}")
        # record the selection outcome so DagsHub shows which epoch each run's best.pth came from
        tracker.log_metrics({"best_epoch": float(best_epoch), "best_rankme": best_rankme})
    elif val_records:
        print("[best] NO epoch passed the convergence gate "
              f"(val_loss<={cfg.get('best_val_loss_max', -0.8)}, align<={cfg.get('best_align_max', 0.15)}); "
              "best.pth was NOT written -> use last.pth for downstream, and check that training actually "
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
