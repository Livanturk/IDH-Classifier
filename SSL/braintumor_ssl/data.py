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
import time
import zlib
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
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
NIFTI_READ_RETRIES = 3


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


def _stratified_val_ids(kept: list[dict], n_val: int, rng) -> set:
    """Pick n_val indices of `kept`, allocated across collections in proportion to their size.

    Largest-remainder apportionment: each collection gets floor(share), then the leftover seats go
    to the largest fractional remainders. The val cohort is where every label-free spectral metric
    is measured, so its site/scanner mix has to mirror the pretraining pool — a plain uniform draw
    would let the biggest collection dominate the eigenspectrum.
    """
    by_col: dict[str, list[int]] = {}
    for i, r in enumerate(kept):
        by_col.setdefault(r["collection"], []).append(i)

    share = {c: len(ix) * n_val / len(kept) for c, ix in by_col.items()}
    alloc = {c: min(int(s), len(by_col[c])) for c, s in share.items()}
    # leftover seats -> largest fractional remainder first (ties broken by name for determinism)
    order = sorted(share, key=lambda c: (-(share[c] - int(share[c])), c))
    left = n_val - sum(alloc.values())
    for c in order * 2:                                  # 2 passes always suffice
        if left <= 0:
            break
        if alloc[c] < len(by_col[c]):
            alloc[c] += 1
            left -= 1

    val_ids: set = set()
    for c in sorted(by_col):                             # sorted -> draw order independent of dict order
        ix = np.asarray(by_col[c])
        pick = rng.permutation(len(ix))[: alloc[c]]
        val_ids.update(ix[pick].tolist())
    return val_ids


def build_splits(
    records: list[dict],
    val_frac: float = 0.05,
    seed: int = 42,
    exclude_collections: Optional[list[str]] = None,
    val_n: Optional[int] = None,
    stratify: bool = False,
) -> dict:
    """Subject-level train/val split.

    `val_n` (absolute subject count) overrides `val_frac` when given; `stratify` draws the val
    cohort proportionally per collection instead of uniformly at random. Both exist because val is
    no longer just loss/collapse monitoring — RankMe / LiDAR / alpha-ReQ are estimated from its
    256-d feature covariance, and both the size and the composition of that cohort enter the
    estimate (see CHECKPOINT_SELECTION_METHODOLOGY.md).
    """
    exclude = set(exclude_collections or [])
    kept = [r for r in records if r["collection"] not in exclude]
    rng = np.random.default_rng(seed)
    n_val = int(val_n) if val_n else max(1, int(round(len(kept) * val_frac)))
    n_val = max(1, min(n_val, len(kept) - 1))

    if stratify:
        val_ids = _stratified_val_ids(kept, n_val, rng)
        val = [kept[i] for i in sorted(val_ids)]
        train = [kept[i] for i in range(len(kept)) if i not in val_ids]
    else:
        idx = rng.permutation(len(kept))
        train = [kept[i] for i in idx[n_val:]]
        val = [kept[i] for i in idx[:n_val]]

    def _by_col(rs):
        out: dict[str, int] = {}
        for r in rs:
            out[r["collection"]] = out.get(r["collection"], 0) + 1
        return dict(sorted(out.items()))

    return {
        "seed": seed,
        "val_frac": val_frac,
        "val_n": n_val,
        "stratified": bool(stratify),
        "excluded_collections": sorted(exclude),
        "n_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "train_by_collection": _by_col(train),
        "val_by_collection": _by_col(val),
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
def _read_nifti(path: str, dtype: np.dtype | None = None) -> np.ndarray:
    """Materialize a NIfTI volume, retrying transient reads from shared storage.

    The unified cohort consists of symlinks to gzip-compressed NIfTI files on shared storage.
    A busy multi-GPU launch can occasionally surface a transient gzip/zlib read error even though
    the file is valid on disk.  Retrying a fresh ``nib.load`` avoids failing an entire SSL run for
    that transient condition.  A persistent error remains fatal and names the exact source file.
    """
    last_error: Exception | None = None
    for attempt in range(1, NIFTI_READ_RETRIES + 1):
        try:
            img = nib.load(path, mmap=False, keep_file_open=False)
            return np.asarray(img.dataobj, dtype=dtype)
        except (OSError, EOFError, zlib.error) as exc:
            last_error = exc
            if attempt < NIFTI_READ_RETRIES:
                time.sleep(0.15 * attempt)

    raise RuntimeError(
        f"Failed to read gzipped NIfTI after {NIFTI_READ_RETRIES} attempts: {path}. "
        "The file may be corrupted or shared-storage I/O may be unstable; run "
        f"`gzip -t {path!r}` to distinguish them."
    ) from last_error


def _load_volume(record: dict, modalities: tuple[str, ...], load_seg: bool):
    """Load modalities into a (C, H, W, D) float32 array cropped to the brain bbox.

    If load_seg, also returns the segmentation cropped to the SAME frame (else None).
    """
    vols = []
    for m in modalities:
        p = os.path.join(record["dir"], f"{record['id']}_{m}.nii.gz")
        vols.append(_read_nifti(p, dtype=np.float32))
    x = np.stack(vols, axis=0)  # (C, H, W, D)

    seg = None
    if load_seg:
        segp = os.path.join(record["dir"], f"{record['id']}_seg.nii.gz")
        if not os.path.exists(segp):
            raise FileNotFoundError(
                f"crop_mode requires segmentation but {segp} is missing. Use crop_mode=brain "
                f"or provide/auto-generate a mask for subject {record['id']}."
            )
        seg = _read_nifti(segp)

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
    """roi-sized box centred at `center` (voxel coords), zero-padded past the edges.

    When the volume is at least roi-sized on an axis, the window is clamped to lie fully
    inside it, so a crop centre near the edge (e.g. a whole-tumour centroid close to the
    brain boundary) still yields a complete in-view crop instead of a half-empty padded one.
    This matches the coverage assumed by adaptive_crop.py's `_crop_start`. Axes smaller than
    roi are centred and zero-padded as before; brain-mode random crops already choose in-range
    centres, so their output is unchanged.
    """
    out = x.new_zeros((x.shape[0], *roi))
    src = [slice(None)]
    dst = [slice(None)]
    for d in range(3):
        dim = x.shape[d + 1]
        start = int(round(center[d])) - roi[d] // 2
        if dim >= roi[d]:
            start = min(max(start, 0), dim - roi[d])   # keep the whole window inside the volume
        s0, s1 = max(start, 0), min(start + roi[d], dim)
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


AUG_PRESETS = ("standard", "gentle", "clinical", "auto")


class ClinicalTransform:
    """Gentle, mask-preserving SSL augmentation for tumour ROIs (torch, applied per view).

    Faithful to the team's `ClinicalSSLTransform`: soft L-R / A-P flips plus channel-wise
    intensity scale/shift, light Gaussian noise and light blur — deliberately no 90-deg
    rotation, cutout, z-axis flip or large deformation, which would distort or erase a small
    masked tumour. Every op is confined to the originally-nonzero voxels and the background is
    re-zeroed afterwards, so a WT-masked ROI keeps its exact zero background. (The MONAI
    intensity presets do NOT: RandShiftIntensity / RandGaussianNoise would leak signal into
    the masked-out region.) Pair with crop_mode=tumor + mask_out_non_tumor + normalize_after_crop.

    Input/output: a (C, H, W, D) tensor. Spatial flips use axes 1, 2 (H, W); axis 3 (D,
    superior-inferior) is never flipped, consistent with the other presets.
    """

    def __init__(
        self,
        flip_prob: float = 0.25,
        scale_prob: float = 0.5,
        shift_prob: float = 0.5,
        noise_prob: float = 0.3,
        blur_prob: float = 0.15,
        noise_std: float = 0.03,
        scale_range: tuple[float, float] = (0.9, 1.1),
        shift_range: tuple[float, float] = (-0.1, 0.1),
        strength: float = 1.0,
    ):
        # `strength` scales the intensity-augmentation MAGNITUDES around their centres (noise std,
        # scale deviation, shift range); strength=1.0 leaves the reference recipe unchanged. It is
        # the config-driven "strong aug" knob for the Stage-C ablation (tasks/todo.md A0.4). Probs
        # and flips are left as-is so the ablation isolates distortion magnitude, not frequency.
        self.flip_prob = flip_prob
        self.scale_prob = scale_prob
        self.shift_prob = shift_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.noise_std = noise_std * strength
        self.scale_range = (1.0 - (1.0 - scale_range[0]) * strength,
                            1.0 + (scale_range[1] - 1.0) * strength)
        self.shift_range = (shift_range[0] * strength, shift_range[1] * strength)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).float().clone()
        if torch.rand(1).item() < self.flip_prob:
            x = torch.flip(x, dims=[1])                       # H (L-R / A-P)
        if torch.rand(1).item() < self.flip_prob:
            x = torch.flip(x, dims=[2])                       # W
        nonzero = x != 0                                      # (masked) background to preserve
        if torch.rand(1).item() < self.scale_prob:
            s = torch.empty(x.shape[0], 1, 1, 1).uniform_(*self.scale_range)
            x = torch.where(nonzero, x * s, x)               # channel-wise intensity scale
        if torch.rand(1).item() < self.shift_prob:
            b = torch.empty(x.shape[0], 1, 1, 1).uniform_(*self.shift_range)
            x = torch.where(nonzero, x + b, x)               # channel-wise intensity shift
        if torch.rand(1).item() < self.noise_prob:
            x = torch.where(nonzero, x + torch.randn_like(x) * self.noise_std, x)
        if torch.rand(1).item() < self.blur_prob:
            blurred = F.avg_pool3d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
            x = torch.where(nonzero, blurred, x)
        return torch.where(nonzero, x, torch.zeros_like(x))  # guarantee the background stays 0


def make_appearance_transforms(roi_size, preset: str = "standard", aug_strength: float = 1.0):
    """Appearance/geometry augmentations applied AFTER the spatial crop (per view).

    preset:
      * "standard" — whole-brain / context ROIs: strong geometry (90-deg rot, cutout) is fine.
      * "gentle"   — WT-masked / small tumour ROIs: soft geometry (no 90-deg rot, no cutout,
                     ~5-deg affine, low-noise) but intensity augmentation kept strong, so the
                     SSL signal stays ("same tumour under different scanner/intensity/noise").
      * "clinical" — like "gentle" but mask-preserving (see ClinicalTransform): keeps a
                     WT-masked ROI's zero background exact. This is the reference-recipe preset.
    """
    if preset == "clinical":
        return ClinicalTransform(strength=aug_strength)
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
        aug_strength: float = 1.0,        # scales clinical intensity-aug magnitude (Stage-C A0.4)
        normalize_after_crop: bool = False,  # normalize on the (masked) ROI, not the whole brain
        n_views: int = 2,                 # augmented views per subject in mode="train"
        cache: bool = False,
    ):
        assert mode in ("train", "eval")
        assert n_views >= 2, "mode='train' emits at least the two SSL views"
        assert crop_mode in CROP_MODES, f"crop_mode must be one of {CROP_MODES}"
        assert center_mode in ("wt_centroid", "bbox_center")
        assert aug_preset in AUG_PRESETS, f"aug_preset must be one of {AUG_PRESETS}"
        self.records = records
        self.modalities = tuple(modalities)
        self.mode = mode
        self.n_views = int(n_views)
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
        self.appearance = make_appearance_transforms(roi_size, preset, float(aug_strength))
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
        # n_views independent crop+augment passes over ONE volume read: the two SSL views are
        # view1/view2 (the training contract, n_views=2), and LiDAR's validation loader asks for
        # more surrogate-class samples per subject without paying a second NIfTI read.
        out = {f"view{i + 1}": torch.as_tensor(
                   self.appearance(self._view(x, tumor, wt_mask, rng, training=True))).float()
               for i in range(self.n_views)}
        out["id"] = sid
        return out


def make_views(records: list[dict], cfg: dict, mode: str, cache: bool = False,
               n_views: int = 2) -> BraTSViews:
    """Construct a BraTSViews from a config dict — keeps all crop options in sync across scripts.

    `n_views` stays at the 2-view training contract by default; only validation's LiDAR loader
    raises it (see train_simsiam.validate).
    """
    return BraTSViews(
        records,
        n_views=n_views,
        modalities=cfg["modalities"],
        roi_size=tuple(cfg["roi_size"]),
        mode=mode,
        crop_mode=cfg.get("crop_mode", "brain"),
        tumor_margin=cfg.get("tumor_margin", 16),
        center_mode=cfg.get("center_mode", "wt_centroid"),
        mask_out_non_tumor=cfg.get("mask_out_non_tumor", False),
        clip_percentiles=cfg.get("clip_percentiles", (0.5, 99.5)),
        aug_preset=cfg.get("aug_preset", "auto"),
        aug_strength=cfg.get("aug_strength", 1.0),
        normalize_after_crop=cfg.get("normalize_after_crop", False),
        cache=cache,
    )
