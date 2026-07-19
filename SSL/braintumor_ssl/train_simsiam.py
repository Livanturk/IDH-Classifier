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
from braintumor_ssl.models import SimSiam, simsiam_loss
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
    ap.add_argument("--aug_preset", choices=["standard", "gentle", "auto"])
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
    ap.add_argument("--early_stop", action="store_true", default=None)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--min_epochs", type=int)
    ap.add_argument("--collapse_stop", action="store_true", default=None)
    ap.add_argument("--collapse_patience", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--cache", action="store_true", default=None)
    ap.add_argument("--smoke", action="store_true", help="cap iterations/epoch for a fast dry run")
    return ap.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if k in ("config", "smoke"):
            continue
        if v is not None:
            cfg[k] = v
    cfg["smoke"] = args.smoke
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


def write_metrics_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cfg = load_config(parse_args())
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

    # ---- train ----
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        loss_m, std_m = AverageMeter(), AverageMeter()
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
        print(f"[epoch {epoch:03d}] loss={loss_m.avg:+.4f} z_std={std_m.avg:.4f} "
              f"(healthy~{expected:.4f}) time={time.time()-t0:.1f}s")

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
            print(f"[val {epoch:03d}] loss={vm['val_loss']:+.4f} z_std={vm['z_std']:.4f} "
                  f"rankme={vm['rankme']:.3f} pr={vm['participation_ratio']:.2f} "
                  f"align={vm['alignment']:.4f} unif={vm['uniformity']:+.3f} "
                  f"collapsed={collapsed}" + ("" if not collapsed else f" ({why})"))
            metrics_history.append({
                "epoch": epoch, "train_loss": round(loss_m.avg, 5), "train_z_std": round(std_m.avg, 5),
                "val_loss": round(vm["val_loss"], 5), "z_std": round(vm["z_std"], 5),
                "rankme": round(vm["rankme"], 4), "participation_ratio": round(vm["participation_ratio"], 4),
                "alignment": round(vm["alignment"], 5), "uniformity": round(vm["uniformity"], 5),
                "n_val": vm["n_val"], "collapsed": collapsed,
            })
            write_metrics_csv(os.path.join(cfg["out_dir"], "metrics.csv"), metrics_history)

            improved = (not collapsed) and math.isfinite(vm["rankme"]) \
                and vm["rankme"] > best_rankme + cfg.get("min_delta_rankme", 0.0)
            if improved:
                best_rankme, best_epoch, no_improve = vm["rankme"], epoch, 0
                save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(), "cfg": cfg, "val_metrics": vm},
                                os.path.join(cfg["out_dir"], "best.pth"))
                print(f"  -> new best RankMe={best_rankme:.3f} (saved best.pth)")
            else:
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
        print(f"[best] RankMe={best_rankme:.3f} @ epoch {best_epoch} -> {os.path.join(cfg['out_dir'], 'best.pth')}")
    if metrics_history:
        try:
            from braintumor_ssl.visualize import save_training_curves
            save_training_curves(metrics_history, os.path.join(cfg["out_dir"], "training_curves.png"),
                                 healthy_z_std=1.0 / (cfg["feature_dim"] ** 0.5))
            print(f"[plot] wrote {os.path.join(cfg['out_dir'], 'training_curves.png')}")
        except Exception as e:                       # plotting must never fail a training run
            print(f"[warn] training_curves.png not written: {e}")
    print(f"[done] checkpoints in {cfg['out_dir']}")


if __name__ == "__main__":
    main()
