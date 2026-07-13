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
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from braintumor_ssl.data import BraTSViews, load_splits
from braintumor_ssl.models import SimSiam, simsiam_loss
from braintumor_ssl.utils import (
    AverageMeter,
    cosine_lr,
    representation_std,
    save_checkpoint,
    scaled_lr,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    # every config key is overridable on the CLI
    ap.add_argument("--data_root")
    ap.add_argument("--splits_file")
    ap.add_argument("--modalities", nargs="+")
    ap.add_argument("--roi_size", nargs=3, type=int)
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


def main() -> None:
    cfg = load_config(parse_args())
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roi = tuple(cfg["roi_size"])
    print(f"[cfg] device={device} backbone={cfg['backbone']} roi={roi} "
          f"batch={cfg['batch_size']} feature_dim={cfg['feature_dim']}")

    # ---- data ----
    split = load_splits(cfg["splits_file"])
    train_ds = BraTSViews(split["train"], modalities=cfg["modalities"], roi_size=roi,
                          mode="train", cache=bool(cfg.get("cache")))
    train_ld = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
    )
    print(f"[data] train subjects={len(train_ds)} iters/epoch={len(train_ld)}")

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

    print(f"[done] checkpoints in {cfg['out_dir']}")


if __name__ == "__main__":
    main()
