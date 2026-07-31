"""Turn a pretraining checkpoint into a portable `<backbone>-<rule>-seed<N>-encoder.pth`.

This is the hand-off artifact between the two stages of the study:

    BraTS  -> SimSiam pretraining -> <rule> selects an epoch -> ENCODER.pth
    UCSF   -> frozen ENCODER.pth  -> head (logistic / elastic-net) -> IDH 0/1

A training checkpoint is not that artifact. It carries the optimizer state, the projector and the
predictor — ~270 MB of things the downstream stage must never use — and its filename (`bestLiDAR.pth`
inside `simsiam_r18_unified_seed42/`) encodes the provenance only by directory position. Exporting
gives a file that is self-describing (`resnet18-lidar-seed42-encoder.pth`), an order of magnitude
smaller, and structurally incapable of leaking the SSL head into the downstream model, because only
the `encode()` path — backbone + encoder_head -> the 256-d transferable feature h — is saved.

The exported file keeps the full resolved `cfg`. This is not optional metadata: `make_views` rebuilds
the input pipeline from it, so the encoder sees the same crop_mode / normalization / roi it was
trained under. An encoder without its cfg is not reproducible.

Selection provenance travels too. Each best*.pth stores WHY that epoch was chosen (the metric, its
value, the other two metrics, the Pareto front size for the Proposed rule); that block is copied into
the export so the downstream table can be traced back to a selection rule without re-reading the
run directory.

Usage
-----
    # every selection rule of one run
    python -m braintumor_ssl.export_encoder --run checkpoints/simsiam_r18_unified_seed42_dense

    # all runs, into the shared encoder shelf the downstream stage reads
    python -m braintumor_ssl.export_encoder --run checkpoints/*_dense --out_dir encoders

    # load one back
    from braintumor_ssl.export_encoder import load_encoder
    enc, cfg = load_encoder("encoders/resnet18-lidar-seed42-encoder.pth", device="cuda")
    h = enc(batch)          # (B, 256), eval mode, no grad needed
"""
from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn as nn

from braintumor_ssl.models import SimSiam

# source checkpoint -> the short rule name that goes in the filename and the paper's table row
RULES = {
    "bestRankMe.pth": "rankme",
    "bestLiDAR.pth": "lidar",
    "bestA-ReQ.pth": "areq",
    "bestPareto.pth": "proposed",     # the plateau-gated Pareto rule (select_epoch.py)
    "last.pth": "last",               # final-plateau reference, not a selection rule
}


class Encoder(nn.Module):
    """Just the transferable path: backbone -> encoder_head -> h (256-d). No projector/predictor."""

    def __init__(self, backbone: nn.Module, encoder_head: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.encoder_head = encoder_head
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_head(self.backbone(x))


def encoder_state_dict(model_state: dict) -> dict:
    """Keep only the `backbone.*` / `encoder_head.*` tensors of a SimSiam state dict."""
    return {k: v for k, v in model_state.items()
            if k.startswith("backbone.") or k.startswith("encoder_head.")}


def export(ckpt_path: str, out_dir: str, run_name: str) -> str | None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    fname = os.path.basename(ckpt_path)
    rule = RULES.get(fname)
    if rule is None:
        print(f"  [skip] {fname}: not a selection-rule checkpoint")
        return None
    backbone, seed = cfg.get("backbone", "unknown"), cfg.get("seed", "x")
    state = encoder_state_dict(ck["model"])
    payload = {
        "encoder": state,
        "cfg": cfg,
        "feature_dim": cfg.get("feature_dim", 256),
        "backbone": backbone,
        "rule": rule,
        "provenance": {
            "run": run_name, "source_checkpoint": fname, "epoch": ck.get("epoch"),
            # WHY this epoch: the metric that picked it, or the Pareto front that deferred.
            "selection": ck.get("selection", {}),
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{backbone}-{rule}-seed{seed}-encoder.pth")
    torch.save(payload, out)
    mb_in = os.path.getsize(ckpt_path) / 1e6
    mb_out = os.path.getsize(out) / 1e6
    print(f"  {fname:<16} e{ck.get('epoch'):>3}  ->  {os.path.basename(out):<42} "
          f"({mb_in:.0f} MB -> {mb_out:.0f} MB, {len(state)} tensors)")
    return out


def load_encoder(path: str, device: str = "cpu") -> tuple[Encoder, dict]:
    """Rebuild the frozen encoder + its cfg. Returns an eval-mode module with grads off.

    The cfg comes back with it because the caller needs it to build a matching input pipeline
    (`make_views(records, cfg, mode="eval")`) — feeding this encoder differently-preprocessed
    volumes than it was trained on silently degrades the features.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    full = SimSiam(
        backbone=cfg["backbone"], in_channels=len(cfg["modalities"]),
        feature_dim=cfg["feature_dim"], proj_dim=cfg["proj_dim"],
        proj_hidden_dim=cfg["proj_hidden_dim"], pred_hidden_dim=cfg["pred_hidden_dim"],
    )
    missing, unexpected = full.load_state_dict(payload["encoder"], strict=False)
    # Only the projector/predictor may be missing — anything else means a corrupt or mismatched file.
    stray = [k for k in missing if not (k.startswith("projector.") or k.startswith("predictor."))]
    if stray or unexpected:
        raise RuntimeError(f"{path}: unexpected keys (missing={stray[:3]}, unexpected={unexpected[:3]})")
    enc = Encoder(full.backbone, full.encoder_head, cfg["feature_dim"]).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", nargs="+", required=True, help="pretraining run directories")
    ap.add_argument("--out_dir", default="encoders", help="where the encoder shelf lives")
    ap.add_argument("--rules", nargs="+", default=list(RULES),
                    help=f"which checkpoints to export (default: all of {list(RULES)})")
    args = ap.parse_args()

    n = 0
    for run_dir in args.run:
        run_name = os.path.basename(os.path.normpath(run_dir))
        present = [r for r in args.rules if os.path.exists(os.path.join(run_dir, r))]
        if not present:
            print(f"[skip] {run_name}: none of {args.rules} present")
            continue
        print(f"[run] {run_name}")
        for fname in present:
            n += export(os.path.join(run_dir, fname), args.out_dir, run_name) is not None
    print(f"\n[done] {n} encoder(s) -> {args.out_dir}/")


if __name__ == "__main__":
    main()
