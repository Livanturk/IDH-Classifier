"""3D ResNet backbone + SimSiam head (projector / predictor) and loss.

Design (faithful to Chen & He, "Exploring Simple Siamese Representation Learning", CVPR 2021,
adapted to 3D multi-modal MRI):

    x (B, 4, D, H, W)
        -> backbone (MONAI 3D ResNet, global-avg-pooled)      -> f  (backbone_dim)
        -> encoder linear + BN                                -> h  (feature_dim = 256)   [transferable feature]
        -> projector MLP (3-layer, BN)                        -> z  (proj_dim)
        -> predictor MLP (2-layer, bottleneck)                -> p  (proj_dim)

    loss = -0.5 * ( cos(p1, stopgrad(z2)) + cos(p2, stopgrad(z1)) )

Downstream feature extraction returns `h` (the encoder output, 256-d) — the transferable
representation used by the IDH classifier. `f`/`z` are exposed too for ablations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import DenseNet121, resnet18, resnet34, resnet50

_RESNETS = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
BACKBONES = (*_RESNETS, "densenet121")   # valid `backbone:` config values


def build_backbone(name: str, in_channels: int) -> tuple[nn.Module, int]:
    """Return a MONAI 3D backbone with its classification head removed, plus its output dim.

    ResNets expose the global-avg-pooled feature by setting `.fc = Identity`; DenseNet121 by
    replacing its final classifier linear (`class_layers.out`) with Identity. The encoder head
    then maps whatever the backbone width is (512 for r18/r34, 2048 for r50, 1024 for densenet121)
    to the shared 256-d transferable feature, so backbones are interchangeable via config.
    """
    if name in _RESNETS:
        net = _RESNETS[name](
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=1,          # replaced below; only affects the discarded fc
            feed_forward=True,
            shortcut_type="B",
            bias_downsample=False,
        )
        net.fc = nn.Identity()      # forward now returns the global-avg-pooled feature vector
    elif name == "densenet121":
        net = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1)
        net.class_layers.out = nn.Identity()   # drop the classifier -> pooled 1024-d feature
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    was_training = net.training
    net.eval()                      # eval-mode probe: BN uses running stats (safe at batch/spatial 1)
    with torch.no_grad():
        out_dim = net(torch.zeros(2, in_channels, 32, 32, 32)).shape[1]
    net.train(was_training)
    return net, out_dim


def _mlp(dims: list[int], last_bn: bool, relu_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        is_last = i == len(dims) - 2
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=not (is_last and last_bn)))
        if not is_last:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        else:
            if last_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1], affine=False))  # SimSiam: no affine on final proj BN
            if relu_last:
                layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 4,
        feature_dim: int = 256,
        proj_dim: int = 256,
        proj_hidden_dim: int = 512,
        pred_hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone, backbone_dim = build_backbone(backbone, in_channels)
        self.backbone_dim = backbone_dim
        self.feature_dim = feature_dim

        # Encoder head: backbone feature -> 256-d transferable representation h.
        self.encoder_head = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # Projection MLP (3-layer) : h -> z
        self.projector = _mlp([feature_dim, proj_hidden_dim, proj_hidden_dim, proj_dim], last_bn=True)
        # Prediction MLP (2-layer bottleneck) : z -> p
        self.predictor = _mlp([proj_dim, pred_hidden_dim, proj_dim], last_bn=False)

    # --- representation used downstream -------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Transferable 256-d feature h for a batch (used for IDH feature extraction)."""
        return self.encoder_head(self.backbone(x))

    # --- SSL forward --------------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h1, h2 = self.encode(x1), self.encode(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return {"p1": p1, "p2": p2, "z1": z1, "z2": z2, "h1": h1, "h2": h2}


def simsiam_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Symmetric negative cosine similarity with stop-gradient on the targets."""

    def d(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    return 0.5 * d(out["p1"], out["z2"]) + 0.5 * d(out["p2"], out["z1"])


def vc_regularizer(out: dict[str, torch.Tensor], gamma: float = 1.0,
                   eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance terms on the projector embeddings z1, z2 [Bardes2022].

    Optional add-on to `simsiam_loss` (which already supplies the invariance term via the
    predictor + stop-gradient). SimSiam has no explicit rank-preserving term, so it is prone to
    *dimensional collapse* [Jing2022]; these two terms counter it directly:
      - variance : hinge that keeps each embedding dim's std >= gamma across the batch, so no
                   dimension is allowed to shrink to a constant;
      - covariance: pushes off-diagonal feature covariances to zero, decorrelating the dimensions
                   so variance is spread over the full space rather than a few directions.
    Returns (var_term, cov_term) separately so the trainer can weight and log each. Computed in
    float32 for numerical stability under AMP. Not applied when its weights are 0 (default off)."""
    def _vc(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.float()
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + eps)
        var = torch.mean(F.relu(gamma - std))
        n, d = z.shape
        cov = (z.T @ z) / (n - 1)
        cov_off = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
        return var, cov_off / d

    v1, c1 = _vc(out["z1"])
    v2, c2 = _vc(out["z2"])
    return 0.5 * (v1 + v2), 0.5 * (c1 + c2)


def whitening_mse_loss(out: dict[str, torch.Tensor], eps: float = 1e-3) -> torch.Tensor:
    """W-MSE-style whitening loss on the projector embeddings z1, z2 [Ermolov2021].

    Whitens each view's batch (ZCA via Cholesky of the shrinkage-regularised covariance) so the
    embedding batch has ~identity covariance, then matches the two whitened views (negative cosine,
    same scale as simsiam_loss). Anti-collapse is IMPLICIT: a collapsed batch has a singular
    covariance and cannot be whitened, so the objective is driven away from collapse — a different
    mechanism from VICReg's soft penalty.

    IMPORTANT: full-rank batch whitening needs (batch size) > (embedding dim). With the 3D memory
    budget (batch 16) this forces a SMALL projector (proj_dim < batch) for the whitening to be
    meaningful — small projections are how W-MSE is designed to run, but our batch cap makes the
    projector unusually tight; this constraint is itself a finding for the 3D small-batch regime.
    Computed in float32 for Cholesky stability under AMP. Optional add-on; off by default."""
    def whiten(z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        z = z - z.mean(0, keepdim=True)
        n, d = z.shape
        cov = (z.T @ z) / (n - 1) + eps * torch.eye(d, device=z.device, dtype=z.dtype)
        L = torch.linalg.cholesky(cov)
        return torch.linalg.solve_triangular(L, z.T, upper=False).T   # z @ (L^{-1})^T

    z1w = F.normalize(whiten(out["z1"]), dim=1)
    z2w = F.normalize(whiten(out["z2"]), dim=1)
    return -0.5 * ((z1w * z2w).sum(1).mean() + (z2w * z1w).sum(1).mean())
