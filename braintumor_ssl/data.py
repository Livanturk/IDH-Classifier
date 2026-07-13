"""BraTS 2021 data: subject discovery, splits, and a two-view SSL dataset.

The four modalities (T1, T1ce, T2, FLAIR) are stacked as a 4-channel volume.
Segmentation is intentionally NOT used here — per BrainTumor.docx it belongs to the
downstream radiomics branch, and SSL pretraining is label/mask free. (Seg-guided
tumour cropping is available as an opt-in flag; off by default.)
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandCoarseDropout,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandRotate90,
    RandScaleIntensity,
    RandShiftIntensity,
    RandSpatialCrop,
    Resize,
    SpatialPad,
)

REQUIRED_SUFFIXES = ("t1", "t1ce", "t2", "flair")


# --------------------------------------------------------------------------- #
# Subject discovery + splits
# --------------------------------------------------------------------------- #
def scan_subjects(data_root: str) -> list[dict]:
    """Find every subject folder under data_root that has all 4 modalities.

    Returns records: {"id", "dir", "collection"}. Works with both the flat BraTS
    layout and the collection-grouped layout present here (…/<collection>/<id>/).
    """
    records = []
    for seg_like in sorted(glob.glob(os.path.join(data_root, "**", "*_t1.nii.gz"), recursive=True)):
        sub_dir = os.path.dirname(seg_like)
        sub_id = os.path.basename(sub_dir)
        if not all(os.path.exists(os.path.join(sub_dir, f"{sub_id}_{s}.nii.gz")) for s in REQUIRED_SUFFIXES):
            continue
        collection = os.path.basename(os.path.dirname(sub_dir))
        records.append({"id": sub_id, "dir": sub_dir, "collection": collection})
    # de-dup by id (a subject could match twice if globs overlap)
    uniq = {r["id"]: r for r in records}
    return sorted(uniq.values(), key=lambda r: r["id"])


def build_splits(
    records: list[dict],
    val_frac: float = 0.05,
    seed: int = 42,
    exclude_collections: Optional[list[str]] = None,
) -> dict:
    """Subject-level train/val split (val is only for collapse/loss monitoring)."""
    exclude = set(exclude_collections or [])
    kept = [r for r in records if r["collection"] not in exclude]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(kept))
    n_val = max(1, int(round(len(kept) * val_frac)))
    val_ids = set(idx[:n_val].tolist())
    train = [kept[i] for i in idx[n_val:]]
    val = [kept[i] for i in idx[:n_val]]
    return {
        "seed": seed,
        "val_frac": val_frac,
        "excluded_collections": sorted(exclude),
        "n_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "train": train,
        "val": val,
    }


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Volume loading + normalization
# --------------------------------------------------------------------------- #
def _load_volume(record: dict, modalities: tuple[str, ...]) -> np.ndarray:
    """Load modalities into a (C, H, W, D) float32 array, cropped to the brain bbox."""
    vols = []
    for m in modalities:
        p = os.path.join(record["dir"], f"{record['id']}_{m}.nii.gz")
        vols.append(np.asarray(nib.load(p).dataobj, dtype=np.float32))
    x = np.stack(vols, axis=0)  # (C, H, W, D)

    # brain bounding box = union of non-zero voxels across channels (skull-stripped)
    mask = x.sum(axis=0) > 0
    if mask.any():
        coords = np.array(np.nonzero(mask))
        lo = coords.min(axis=1)
        hi = coords.max(axis=1) + 1
        x = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return x


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #
def make_train_transforms(roi_size) -> Compose:
    """Two calls to this Compose on one volume yield two independent SSL views."""
    return Compose(
        [
            SpatialPad(spatial_size=roi_size),                       # guarantee >= roi
            RandSpatialCrop(roi_size=roi_size, random_size=False),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandAffine(
                prob=0.3,
                rotate_range=(0.26, 0.26, 0.26),   # ~15 deg
                scale_range=(0.1, 0.1, 0.1),
                mode="bilinear",
                padding_mode="zeros",
            ),
            RandGaussianNoise(prob=0.2, std=0.1),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensity(prob=0.5, factors=0.1),
            RandShiftIntensity(prob=0.5, offsets=0.1),
            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
            RandCoarseDropout(prob=0.2, holes=4, spatial_size=roi_size[0] // 8, fill_value=0.0),
        ]
    )


def make_eval_transforms(roi_size) -> Compose:
    """Deterministic whole-brain view for feature extraction (brain bbox -> roi_size)."""
    return Compose([Resize(spatial_size=roi_size, mode="trilinear")])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class BraTSViews(Dataset):
    """Returns (view1, view2) for SSL, or a single view for feature extraction."""

    def __init__(
        self,
        records: list[dict],
        modalities=("t1", "t1ce", "t2", "flair"),
        roi_size=(96, 96, 96),
        mode: str = "train",          # "train" -> two views ; "eval" -> one view
        cache: bool = False,
    ):
        assert mode in ("train", "eval")
        self.records = records
        self.modalities = tuple(modalities)
        self.mode = mode
        self.normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
        self.aug = make_train_transforms(roi_size)
        self.eval_tf = make_eval_transforms(roi_size)
        self._cache: dict[int, np.ndarray] = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _base(self, idx: int) -> torch.Tensor:
        if self._cache is not None and idx in self._cache:
            x = self._cache[idx]
        else:
            x = _load_volume(self.records[idx], self.modalities)
            if self._cache is not None:
                self._cache[idx] = x
        x = self.normalize(torch.as_tensor(x))          # per-channel z-score on brain voxels
        return x

    def __getitem__(self, idx: int):
        x = self._base(idx)
        if self.mode == "eval":
            return {"image": torch.as_tensor(self.eval_tf(x)).float(), "id": self.records[idx]["id"]}
        v1 = torch.as_tensor(self.aug(x)).float()
        v2 = torch.as_tensor(self.aug(x)).float()
        return {"view1": v1, "view2": v2, "id": self.records[idx]["id"]}
