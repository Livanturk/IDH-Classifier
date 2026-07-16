# SimSiam SSL pretraining on BraTS 2021

Self-supervised pretraining of a **3D encoder** on BraTS 2021 brain MRI
(T1, T1ce, T2, FLAIR), to be transferred as a frozen backbone that emits a
**256-dim deep feature vector per subject** for the downstream IDH-mutation task.

This package covers **only** stages 1 (SSL pretraining) and the feature-extraction
handoff of the pipeline in `BrainTumor.docx`:

```
BraTS (T1,T1ce,T2,FLAIR)  ->  SimSiam SSL  ->  frozen encoder  ->  256-d deep features  -> [downstream IDH classifier]
```

The IDH classifier, radiomics (PyRadiomics), clinical fusion and external
validation are **out of scope here** by design.

---

## Why SimSiam (and not SimCLR / MAE)

`BrainTumor.docx` explicitly specifies **SimSiam** ("SimSiam SSL Pretraining",
and the novelty statement "the novelty of this work is not the use of SimSiam
itself"). It is also the right fit for 3D MRI:

* **No negative pairs / no large batch** — critical because 3D volumes are
  memory-heavy; contrastive methods (SimCLR/MoCo) need big batches or a memory bank.
* **No momentum encoder / no queue** — simpler, fewer moving parts.
* Collapse is prevented by the **predictor + stop-gradient** asymmetry.

## Architecture (`braintumor_ssl/models.py`)

```
x (B,4,D,H,W)
  backbone : MONAI 3D ResNet-18 (global-avg-pooled)     -> f  (512)
  encoder  : Linear+BN+ReLU                             -> h  (256)   <-- transferable feature
  projector: 3-layer MLP (BN)                           -> z  (256)
  predictor: 2-layer bottleneck MLP                     -> p  (256)
  loss = -0.5*( cos(p1, sg(z2)) + cos(p2, sg(z1)) )     (symmetric, stop-grad)
```

`feature_dim = 256` gives exactly the "Hasta × 256 Features" the docx asks for.
Downstream extraction uses `model.encode(x)` → `h` (256-d). Backbone is swappable
to `resnet34` / `resnet50` via config.

---

## Setup

Cluster conda already has torch + MONAI. The only extra dependency is a NIfTI
reader:

```bash
pip install --user nibabel        # only package that was missing
```

(`requirements.txt` lists everything for a fresh environment.)

---

## How to run

Run everything **from the repo root** (`/home/ibm`) so `python -m braintumor_ssl.*` resolves.

### 0. Smoke test first (CPU, ~1-2 min) — proves the pipeline end-to-end

```bash
bash scripts/smoke_test.sh
```

### 1. Build the subject-level split

```bash
python -m braintumor_ssl.make_splits \
    --data_root BraTS2021/BraTS2021_TrainingSet --out splits/splits.json
# -> 1251 subjects; train=1188 val=63 (val is for loss/collapse monitoring only)
```

Optional: `--exclude_collections UPENN-GBM` to hold the downstream cohort out of
pretraining (see "Data-leakage note" below).

### 2. Pretrain SimSiam (needs a **GPU** node)

This login node is CPU-only. Submit to a GPU node (e.g. SLURM):

```bash
python -m braintumor_ssl.train_simsiam --config configs/simsiam_brats.yaml
# override anything: --epochs 200 --batch_size 16 --backbone resnet50 --roi_size 112 112 112
```

Checkpoints land in `checkpoints/simsiam_brats/` (`last.pth` + periodic `ckpt_eNNN.pth`).
Watch the log: `loss` should drift toward **-1**, and `z_std` should stay well
**above 0** (near `1/sqrt(256)=0.0625`). `z_std -> 0` means representational collapse.

### 3. Extract 256-d features for the downstream cohort

```bash
python -m braintumor_ssl.extract_features \
    --checkpoint checkpoints/simsiam_brats/last.pth \
    --data_root path/to/UPenn --all --out features/upenn_deep.csv
# -> CSV: id, f000 .. f255   (one row per subject)
```

`--recompute_bn` (on by default) re-estimates BatchNorm stats on the target
cohort — this both fixes eval-mode feature scale and adapts BN across sites
(BraTS→UPenn→UCSF domain shift). Use `--tta_crops N` to average N random crops
per subject for a more robust vector.

---

## Parameter choices (defaults in `configs/simsiam_brats.yaml`)

| Item | Default | Rationale |
|---|---|---|
| SSL method | **SimSiam** | specified by docx; no negatives → small-batch friendly for 3D |
| Backbone | 3D ResNet-18 | good capacity/memory trade-off; resnet50 optional |
| Input | 4 ch [t1,t1ce,t2,flair], `96³` crop | brain-bbox crop keeps tumour context, fits GPU memory (pick size via `adaptive_crop.py`) |
| Normalization | per-channel nonzero z-score + `[0.5, 99.5]` foreground clip | intensities vary ~30× across modalities/subjects; clip tames outliers |
| Feature dim `h` | **256** | matches "Hasta × 256 Features" |
| Optimizer | **SGD**, momentum 0.9, wd 1e-4 | SimSiam paper; `adamw` also available |
| LR | `base_lr 0.05 × batch/256`, cosine + 10-epoch warmup | SimSiam linear scaling rule |
| Predictor LR | fixed (not decayed) | SimSiam appendix — improves stability |
| Loss | symmetric negative cosine similarity, stop-grad | SimSiam |
| Epochs / batch | 200 / 16 | starting point for ~1.2k subjects; tune to GPU memory |
| AMP | on (CUDA) | halves memory, faster |

### Augmentations (two views per subject, `braintumor_ssl/data.py`)
Two presets, selected by `aug_preset` (`standard` | `gentle` | `auto`):
- **standard** (whole-brain / context ROIs): in-plane flips (axes 0/1 @0.3; **no z /
  superior-inferior flip** — anatomically implausible), 90° rotations, ~15° affine,
  Gaussian noise/smooth, intensity scale/shift, gamma, coarse dropout.
- **gentle** (WT-masked / small tumour ROIs): in-plane flips + ~5° affine + low-noise + smooth, and the
  **intensity** augs kept strong (scale/shift/gamma) — but **no 90° rotation and no coarse
  dropout**, which would implausibly distort or erase a small masked tumour.
- **auto** (default): `gentle` when `mask_out_non_tumor=True`, else `standard`.

Rationale: SSL needs strong augmentation, but for a tumour-only ROI the *safe* strong augs are
intensity/appearance (they model scanner/protocol variation without cutting the tumour), while
aggressive geometry is risky — so we keep intensity strong and soften geometry only when masked.

---

## Ablation ladder (comparing configs)

3D SSL is expensive, so we use a disciplined ladder (single-axis changes from the
baseline), not a grid. Auto-generated under `configs/`:

| config | backbone | pretraining data | crop_mode |
|---|---|---|---|
| `simsiam_r18_all` (baseline) | ResNet-18 | all 1251 (train 1188) | brain |
| `simsiam_r50_all` | ResNet-50 | all 1251 | brain |
| `simsiam_r18_noupenn` | ResNet-18 | UPENN-GBM excluded (train 806) | brain |
| `simsiam_r50_noupenn` | ResNet-50 | UPENN-GBM excluded | brain |
| `simsiam_r18_all_tumor` | ResNet-18 | all 1251 | **tumor_margin** |

Axes: **backbone** (R18↔R50), **leakage** (UPenn in/out), **field-of-view** (`brain` vs
`tumor_margin`).

```bash
for c in simsiam_r18_all simsiam_r50_all simsiam_r18_noupenn simsiam_r50_noupenn simsiam_r18_all_tumor; do
    python -m braintumor_ssl.train_simsiam --config configs/$c.yaml   # -> checkpoints/$c/last.pth
done
```
(Run these as separate GPU/SLURM jobs. If ResNet-50 OOMs, lower `--batch_size`; the LR
auto-scales.)

### `crop_mode` — where the patch comes from

The network input is **always** the 4 intensity channels; `crop_mode` only sets *where* the
patch is taken. Segmentation, when used, produces only a crop location — it never enters the tensor.

| `crop_mode` | crop location | needs seg? |
|---|---|---|
| `brain` (default) | random patch in the brain bbox | no |
| `tumor` | roi box centred on the WT centroid (or `bbox_center`) | yes (locator only) |
| `tumor_margin` | WT bbox + `tumor_margin` voxels, resized to roi | yes (locator only) |

Extra knobs (config): `center_mode` (`wt_centroid` | `bbox_center`), and
`mask_out_non_tumor` (tumour modes only — zeroes every non-tumour voxel inside the crop, i.e.
the WT-masked ROI the team's notebook produces).

Feature extraction re-reads all of these from the checkpoint, so inputs stay consistent with
pretraining. Tumour modes need a mask per subject at extraction time too (present for BraTS/
UPenn/UCSF-PDGM; auto-generate, e.g. nnU-Net, for raw external cohorts).

### Choosing `roi_size` from the data (`adaptive_crop.py`)

Instead of hardcoding 96³, pick the crop size from the whole-tumour box distribution
(reproduces the team's Untitled17 notebook: WT geometry → percentile → round-up → coverage):

```bash
python -m braintumor_ssl.adaptive_crop --splits_file splits/splits.json --split train
# -> recommended roi_size (e.g. [112,112,112]) + a coverage table (% subjects whose WT
#    bbox fully fits each candidate crop). Put the chosen size in your config's roi_size.
```

**Masking caveat.** `mask_out_non_tumor=True` keeps only tumour voxels. It maximises focus but
discards peritumoral context (edema, T2-FLAIR mismatch — which correlate with IDH) and makes
deep features share radiomics' exact support, risking *more redundancy* (vs the docx's
complementarity goal). Recommended to run it as an **ablation** against a context-preserving
mode (`brain` / `tumor_margin` unmasked), not as the only setting.

## How to compare configs — which encoder is "better"?

Because pretraining loss is **not** comparable across configs (different augmentations/
architectures; low loss can even mean collapse), use the evaluation harness:

```bash
python -m braintumor_ssl.evaluate \
    --checkpoints checkpoints/simsiam_r18_all checkpoints/simsiam_r50_all \
                  checkpoints/simsiam_r18_noupenn checkpoints/simsiam_r50_noupenn \
    --splits_file splits/splits.json --split val --n_subjects 63 --embed tsne
```

It appends a `results/leaderboard.csv` row per checkpoint with **label-free** metrics
(evaluated with a fixed protocol: same subjects, same BN-recompute):

| metric | want | meaning |
|---|---|---|
| `z_std` | **high** (~0.06) | collapse gate — near 0 ⇒ discard the run |
| `rankme` | **high** | effective rank of embeddings; best single label-free predictor of transfer |
| `alignment` | **low** | two views of a subject map close together (invariance) |
| `uniformity` | **negative** | embeddings spread over the sphere |
| `particip_ratio`| **high** | variance spread across many dimensions |

Read them **together**: the healthy signature is *low alignment **and** very-negative
uniformity **and** high rankme/participation*. (Low alignment with near-zero uniformity =
collapse, not quality.) `--embed tsne` also saves a 2-D plot per config coloured by
collection for the UMAP/t-SNE analysis the docx asks for.

**The decisive metric (when labels arrive):** freeze the encoder, and on the IDH cohort
run a **linear probe (logistic regression) + k-NN with stratified k-fold CV** on the 256-d
features → AUC/balanced-accuracy. This is *not* the downstream fusion model (no radiomics/
clinical/attention) — just "are the deep features linearly IDH-separable?". No IDH label
table is in the repo yet; add it, then extend `evaluate.py` with a probe column. Until then,
rank configs by `rankme` (+ the others); it correlates well with downstream probe AUC.

## Should we use the segmentation masks?

**Not for SSL pretraining** — kept label/mask-free (the whole point of SSL, and the
docx routes `seg` into the **downstream PyRadiomics branch**, not the encoder).
Brain cropping uses only the intensity foreground (skull-stripped), no `seg` needed.
`seg`-guided tumour-centred cropping could be added later as an *optional* view
sampler, but it is intentionally off so the encoder learns whole-brain features.

## Data-leakage note (assumption)

`UPENN-GBM` (403 subjects) is present in this BraTS training set **and** is the
downstream IDH cohort in the docx. SSL is label-free, so including UPenn in
pretraining is standard and adds data. But if the final IDH *test* set reuses these
subjects, the encoder has seen their images. The docx already uses **UCSF** as the
external test set, which sidesteps this. If you want strict separation, pretrain with
`--exclude_collections UPENN-GBM`. **Default: include all** (more pretraining data).

## Environment note

Real training requires a **GPU** (this node has none). Everything is CPU-runnable for
the smoke test. Harmless `/tmp/pymp-*` "Device or resource busy" messages at process
exit are a DataLoader-worker cleanup race on cluster `/tmp` and do not affect results.
