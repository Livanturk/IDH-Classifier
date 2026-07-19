"""Small, dependency-light training utilities (no framework)."""
from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec=None) -> str:
    """Resolve a device spec to 'cuda' or 'cpu', pinning the chosen GPU as current.

    spec: None -> auto (GPU 0 if available, else CPU); 'cpu'; a GPU index like 0/'1'/'2';
    or 'cuda'/'cuda:N'. Returns the device *type* ('cuda' or 'cpu') and calls
    torch.cuda.set_device(N) so every plain 'cuda' tensor lands on the requested GPU — this
    keeps all the existing `device == "cuda"` checks working while supporting `--device N`.
    """
    if not torch.cuda.is_available():
        if spec not in (None, "cpu"):
            print(f"[warn] --device {spec} requested but CUDA is unavailable; using CPU")
        return "cpu"
    if spec is None or spec == "cuda":
        return "cuda"                       # default GPU (cuda:0)
    if spec == "cpu":
        return "cpu"
    idx = int(str(spec).replace("cuda:", ""))
    if not 0 <= idx < torch.cuda.device_count():
        raise SystemExit(f"--device {spec}: GPU {idx} out of range (have {torch.cuda.device_count()})")
    torch.cuda.set_device(idx)
    return "cuda"


class AverageMeter:
    """Running average of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def scaled_lr(base_lr: float, batch_size: int) -> float:
    """SimSiam linear LR scaling rule: lr = base_lr * batch_size / 256."""
    return base_lr * batch_size / 256.0


def save_checkpoint(state: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def recompute_bn_stats(model, loader, device: str, max_batches: int = 100) -> int:
    """Precise-BN: reset and re-estimate BatchNorm running stats with a cumulative
    average over `loader` (encoder path only). Fixes eval-mode feature scale when a
    checkpoint is under-trained, and adapts BN to a new cohort for cross-site transfer.
    Returns the number of batches used.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bns:
        return 0
    saved = {}
    for bn in bns:
        bn.reset_running_stats()
        saved[bn] = bn.momentum
        bn.momentum = None            # cumulative moving average
    model.train()
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = (batch["image"] if "image" in batch else batch["view1"]).to(device)
        if x.size(0) < 2:            # BN needs >1 sample to estimate variance
            continue
        model.encode(x)
        n += 1
    for bn in bns:
        bn.momentum = saved[bn]
    model.eval()
    return n


def representation_std(z: torch.Tensor) -> float:
    """SimSiam collapse monitor: mean per-dim std of L2-normalized representations.

    For a d-dim unit-normalized embedding that does NOT collapse, this sits near
    1/sqrt(d); values near 0 indicate representational collapse.
    """
    z = torch.nn.functional.normalize(z.detach().float(), dim=1)
    return z.std(dim=0).mean().item()


# --------------------------------------------------------------------------- #
# Label-free representation-quality metrics (for comparing SSL checkpoints)
# --------------------------------------------------------------------------- #
def _svdvals(x: torch.Tensor) -> torch.Tensor:
    """Singular values, with a CPU retry if the (GPU) LAPACK/cuSOLVER driver fails to converge
    on a degenerate/ill-conditioned matrix (common when N << d or features nearly collapse)."""
    try:
        return torch.linalg.svdvals(x)
    except Exception:
        return torch.linalg.svdvals(x.cpu())


def rankme(Z: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023): effective rank via singular-value entropy.

    Range [1, min(N, d)]; HIGHER = richer, less-degenerate representation.
    Correlates with downstream transfer and needs no labels.
    """
    s = _svdvals(Z.float())
    p = s / (s.sum() + eps) + eps
    return float(torch.exp(-(p * p.log()).sum()))


def alignment(Z1: torch.Tensor, Z2: torch.Tensor) -> float:
    """Wang & Isola alignment: mean sq. distance between two views' unit embeddings.
    LOWER = more augmentation-invariant (only 'good' when uniformity is also healthy).
    """
    a = torch.nn.functional.normalize(Z1.float(), dim=1)
    b = torch.nn.functional.normalize(Z2.float(), dim=1)
    return float(((a - b) ** 2).sum(dim=1).mean())


def uniformity(Z: torch.Tensor, t: float = 2.0) -> float:
    """Wang & Isola uniformity: log E[exp(-t||u-v||^2)] on the unit sphere.
    MORE NEGATIVE = features spread out better (near 0 => collapse).
    """
    z = torch.nn.functional.normalize(Z.float(), dim=1)
    sq = torch.pdist(z).pow(2)
    return float(sq.mul(-t).exp().mean().log())


def participation_ratio(Z: torch.Tensor) -> float:
    """Effective dimensionality (sum(lambda)^2 / sum(lambda^2)) of the feature covariance.
    HIGHER = variance spread across more directions (max = d).

    Computed from the singular values of the centered features (lambda = s^2), not eigvalsh
    of the covariance: SVD is numerically robust to the rank-deficient N << d case (few val
    subjects, 256-d features), where eigvalsh can fail to converge. The value is identical for
    well-conditioned inputs (the (n-1) covariance scale cancels in the ratio)."""
    z = Z.float() - Z.float().mean(0, keepdim=True)
    ev = _svdvals(z).pow(2)
    return float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))
