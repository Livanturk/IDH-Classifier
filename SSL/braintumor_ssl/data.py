"""BraTS 2021 data: subject discovery, splits, and a two-view SSL dataset.

The four modalities (T1, T1ce, T2, FLAIR) are stacked as a 4-channel volume — this is
ALWAYS the model input; the segmentation is NEVER fed to the network.

`crop_mode` controls *where* the roi_size patch is taken from:
  * "brain"        : random patch from the brain bounding box (seg not used at all).
  * "tumor"        : roi_size box centred on the whole-tumour (WT) centroid from seg.
  * "tumor_margin" : WT bounding box + margin, resized to roi_size.
In the tumour modes, seg is used ONLY to compute a crop location/box (a few numbers);
its voxels never enter the tensor. Feature extraction re-uses the same crop_mode so the
encoder sees inputs consistent with pretraining.
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
    RandAdjustContrast,
    RandAffine,
    RandCoarseDropout,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandRotate90,
    RandScaleIntensity,
    RandShiftIntensity,
    Resize,
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
    for t1_path in sorted(glob.glob(os.path.join(data_root, "**", "*_t1.nii.gz"), recursive=True)):
        sub_dir = os.path.dirname(t1_path)
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


CROP_MODES = ("brain", "tumor", "tumor_margin")


# --------------------------------------------------------------------------- #
# Volume loading + normalization
# --------------------------------------------------------------------------- #
def _load_volume(record: dict, modalities: tuple[str, ...], load_seg: bool):
    """Load modalities into a (C, H, W, D) float32 array cropped to the brain bbox.

    If load_seg, also returns the segmentation cropped to the SAME frame (else None).
    """
    vols = []
    for m in modalities:
        p = os.path.join(record["dir"], f"{record['id']}_{m}.nii.gz")
        vols.append(np.asarray(nib.load(p).dataobj, dtype=np.float32))
    x = np.stack(vols, axis=0)  # (C, H, W, D)

    seg = None
    if load_seg:
        segp = os.path.join(record["dir"], f"{record['id']}_seg.nii.gz")
        if not os.path.exists(segp):
            raise FileNotFoundError(
                f"crop_mode requires segmentation but {segp} is missing. Use crop_mode=brain "
                f"or provide/auto-generate a mask for subject {record['id']}."
            )
        seg = np.asarray(nib.load(segp).dataobj)

    mask = x.sum(axis=0) > 0  # skull-stripped -> union of non-zero voxels = brain
    if mask.any():
        coords = np.array(np.nonzero(mask))
        lo = coords.min(axis=1)
        hi = coords.max(axis=1) + 1
        x = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if seg is not None:
            seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return x, seg


def normalize_nonzero_zscore(x: np.ndarray, clip_percentiles=(0.5, 99.5)) -> np.ndarray:
    """Per-channel nonzero z-score with optional foreground percentile clipping.

    Matches the team's `normalize_mri_nonzero_zscore`: background (==0) is excluded,
    the foreground is clipped to [p_lo, p_hi] percentiles (outlier robustness), z-scored,
    and the background is reset to 0.
    """
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        img = x[c].astype(np.float32)
        bg = img == 0
        fg = img[~bg]
        if fg.size < 10:
            out[c] = img
            continue
        if clip_percentiles is not None:
            lo, hi = np.percentile(fg, clip_percentiles)
            img = np.clip(img, lo, hi)
            fg = img[~bg]
        img = (img - fg.mean()) / (fg.std() + 1e-8)
        img[bg] = 0.0
        out[c] = img
    return out


def _wt_region(seg: np.ndarray):
    """Whole-tumour (seg>0 = labels 1|2|4) bbox, centroid and bbox-centre; None if empty."""
    wt = seg > 0
    if not wt.any():
        return None
    coords = np.argwhere(wt)
    lo, hi_incl = coords.min(0), coords.max(0)
    return {
        "lo": lo,
        "hi": hi_incl + 1,                                  # exclusive
        "centroid": coords.mean(0),
        "bbox_center": np.round((lo + hi_incl) / 2.0),
    }


# --------------------------------------------------------------------------- #
# Spatial cropping (mode-dependent) — network input is always intensity-only
# --------------------------------------------------------------------------- #
def _resize(x: torch.Tensor, roi) -> torch.Tensor:
    return Resize(spatial_size=roi, mode="trilinear")(x)


def _center_crop(x: torch.Tensor, center, roi) -> torch.Tensor:
    """roi-sized box centred at `center` (voxel coords), zero-padded past the edges."""
    out = x.new_zeros((x.shape[0], *roi))
    src = [slice(None)]
    dst = [slice(None)]
    for d in range(3):
        start = int(round(center[d])) - roi[d] // 2
        s0, s1 = max(start, 0), min(start + roi[d], x.shape[d + 1])
        d0 = s0 - start
        src.append(slice(s0, s1))
        dst.append(slice(d0, d0 + (s1 - s0)))
    out[tuple(dst)] = x[tuple(src)]
    return out


def _random_crop(x: torch.Tensor, roi, rng) -> torch.Tensor:
    centers = []
    for d in range(3):
        dim = x.shape[d + 1]
        if dim <= roi[d]:
            centers.append(dim // 2)                      # centred; _center_crop zero-pads
        else:
            lo, hi = roi[d] // 2, dim - (roi[d] - roi[d] // 2)
            centers.append(int(rng.integers(lo, hi + 1)))
    return _center_crop(x, centers, roi)


def _tumor_crop(x, wt_mask, center, roi, rng, training, mask_out) -> torch.Tensor:
    c = np.asarray(center, dtype=float)
    if training:                                          # jitter the centre for view diversity
        c = c + rng.integers(-(np.array(roi) // 4), np.array(roi) // 4 + 1)
    img = _center_crop(x, c, roi)
    if mask_out and wt_mask is not None:                  # zero out non-tumour voxels
        img = img * _center_crop(wt_mask, c, roi)
    return img


def _tumor_margin_crop(x, wt_mask, tumor, roi, margin, rng, training, mask_out) -> torch.Tensor:
    lo, hi = tumor["lo"].astype(int), tumor["hi"].astype(int)
    lo2, hi2 = [], []
    for d in range(3):
        m = int(rng.integers(int(margin * 0.6), int(margin * 1.6) + 1)) if training else margin
        lo2.append(max(0, lo[d] - m))
        hi2.append(min(x.shape[d + 1], hi[d] + m))
    sl = (slice(None), slice(lo2[0], hi2[0]), slice(lo2[1], hi2[1]), slice(lo2[2], hi2[2]))
    img = x[sl]
    if mask_out and wt_mask is not None:
        img = img * wt_mask[sl]
    return _resize(img, roi)


AUG_PRESETS = ("standard", "gentle", "auto")


def make_appearance_transforms(roi_size, preset: str = "standard") -> Compose:
    """Appearance/geometry augmentations applied AFTER the spatial crop (per view).

    preset:
      * "standard" — whole-brain / context ROIs: strong geometry (90-deg rot, cutout) is fine.
      * "gentle"   — WT-masked / small tumour ROIs: soft geometry (no 90-deg rot, no cutout,
                     ~5-deg affine, low-noise) but intensity augmentation kept strong, so the
                     SSL signal stays ("same tumour under different scanner/intensity/noise").
    """
    if preset == "gentle":
        return Compose(
            [
                RandFlip(prob=0.3, spatial_axis=0),          # L-R / A-P only (no z: superior-inferior flip is implausible)
                RandFlip(prob=0.3, spatial_axis=1),
                RandAffine(prob=0.2, rotate_range=(0.087, 0.087, 0.087),  # ~5 deg
                           scale_range=(0.05, 0.05, 0.05), mode="bilinear", padding_mode="zeros"),
                RandGaussianNoise(prob=0.3, std=0.03),
                RandGaussianSmooth(prob=0.15, sigma_x=(0.5, 0.8), sigma_y=(0.5, 0.8), sigma_z=(0.5, 0.8)),
                RandScaleIntensity(prob=0.5, factors=0.1),
                RandShiftIntensity(prob=0.5, offsets=0.1),
                RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
                # no RandRotate90 / RandCoarseDropout: they distort or erase a small masked tumour
            ]
        )
    return Compose(
        [
            RandFlip(prob=0.3, spatial_axis=0),          # L-R / A-P only (no z: superior-inferior flip is implausible)
            RandFlip(prob=0.3, spatial_axis=1),
            RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandAffine(prob=0.3, rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
                       mode="bilinear", padding_mode="zeros"),
            RandGaussianNoise(prob=0.2, std=0.1),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensity(prob=0.5, factors=0.1),
            RandShiftIntensity(prob=0.5, offsets=0.1),
            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
            RandCoarseDropout(prob=0.2, holes=4, spatial_size=roi_size[0] // 8, fill_value=0.0),
        ]
    )


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
        mode: str = "train",              # "train" -> two views ; "eval" -> one view
        crop_mode: str = "brain",         # brain | tumor | tumor_margin
        tumor_margin: int = 16,           # voxels, only for crop_mode=tumor_margin
        center_mode: str = "wt_centroid", # wt_centroid | bbox_center (tumor mode)
        mask_out_non_tumor: bool = False, # zero out non-tumour voxels inside the crop
        clip_percentiles=(0.5, 99.5),     # foreground intensity clip before z-score
        aug_preset: str = "auto",         # standard | gentle | auto (gentle if masked)
        normalize_after_crop: bool = False,  # normalize on the (masked) ROI, not the whole brain
        cache: bool = False,
    ):
        assert mode in ("train", "eval")
        assert crop_mode in CROP_MODES, f"crop_mode must be one of {CROP_MODES}"
        assert center_mode in ("wt_centroid", "bbox_center")
        assert aug_preset in AUG_PRESETS, f"aug_preset must be one of {AUG_PRESETS}"
        self.records = records
        self.modalities = tuple(modalities)
        self.mode = mode
        self.crop_mode = crop_mode
        self.roi = tuple(roi_size)
        self.margin = int(tumor_margin)
        self.center_mode = center_mode
        self.mask_out = bool(mask_out_non_tumor)
        self.clip = tuple(clip_percentiles) if clip_percentiles else None
        self.norm_after_crop = bool(normalize_after_crop)
        self.need_seg = crop_mode in ("tumor", "tumor_margin")
        # 'auto' -> gentle geometry for masked/tumour-only ROIs, standard otherwise
        preset = ("gentle" if self.mask_out else "standard") if aug_preset == "auto" else aug_preset
        self.aug_preset = preset
        self.appearance = make_appearance_transforms(roi_size, preset)
        self._cache: dict[int, tuple] = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _base(self, idx: int):
        if self._cache is not None and idx in self._cache:
            x, seg = self._cache[idx]
        else:
            x, seg = _load_volume(self.records[idx], self.modalities, self.need_seg)
            if self._cache is not None:
                self._cache[idx] = (x, seg)
        # normalize whole brain now, unless norm_after_crop (then normalize the ROI in _view)
        xt = torch.as_tensor(x if self.norm_after_crop else normalize_nonzero_zscore(x, self.clip))
        tumor = _wt_region(seg) if self.need_seg else None
        wt_mask = None
        if self.need_seg and self.mask_out and seg is not None:
            wt_mask = torch.as_tensor((seg > 0).astype(np.float32))[None]  # (1, H, W, D)
        return xt, tumor, wt_mask

    def _view(self, x, tumor, wt_mask, rng, training: bool) -> torch.Tensor:
        if self.crop_mode == "brain" or tumor is None:   # tumour-mode fallback if no mask/tumour
            v = _random_crop(x, self.roi, rng) if training else _resize(x, self.roi)
        elif self.crop_mode == "tumor":
            center = tumor["centroid"] if self.center_mode == "wt_centroid" else tumor["bbox_center"]
            v = _tumor_crop(x, wt_mask, center, self.roi, rng, training, self.mask_out)
        else:
            v = _tumor_margin_crop(x, wt_mask, tumor, self.roi, self.margin, rng, training, self.mask_out)
        if self.norm_after_crop:                          # z-score on the (masked) ROI, not the whole brain
            vt = v.as_tensor() if hasattr(v, "as_tensor") else v
            arr = np.asarray(vt.detach().cpu(), dtype=np.float32)
            v = torch.as_tensor(normalize_nonzero_zscore(arr, self.clip))
        return v

    def __getitem__(self, idx: int):
        x, tumor, wt_mask = self._base(idx)
        rng = np.random.default_rng()                    # fresh entropy per item/worker
        sid = self.records[idx]["id"]
        if self.mode == "eval":
            v = self._view(x, tumor, wt_mask, rng, training=False)
            return {"image": torch.as_tensor(v).float(), "id": sid}
        v1 = torch.as_tensor(self.appearance(self._view(x, tumor, wt_mask, rng, training=True))).float()
        v2 = torch.as_tensor(self.appearance(self._view(x, tumor, wt_mask, rng, training=True))).float()
        return {"view1": v1, "view2": v2, "id": sid}


def make_views(records: list[dict], cfg: dict, mode: str, cache: bool = False) -> BraTSViews:
    """Construct a BraTSViews from a config dict — keeps all crop options in sync across scripts."""
    return BraTSViews(
        records,
        modalities=cfg["modalities"],
        roi_size=tuple(cfg["roi_size"]),
        mode=mode,
        crop_mode=cfg.get("crop_mode", "brain"),
        tumor_margin=cfg.get("tumor_margin", 16),
        center_mode=cfg.get("center_mode", "wt_centroid"),
        mask_out_non_tumor=cfg.get("mask_out_non_tumor", False),
        clip_percentiles=cfg.get("clip_percentiles", (0.5, 99.5)),
        aug_preset=cfg.get("aug_preset", "auto"),
        normalize_after_crop=cfg.get("normalize_after_crop", False),
        cache=cache,
    )
