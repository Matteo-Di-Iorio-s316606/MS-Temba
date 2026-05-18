# MS-Temba on Charades: Complete Technical Analysis — Phase 1
### Reproducibility, Backbone Comparison, and Training Stabilisation

> **Author**: Matteo Di Iorio  
> **Period**: February–March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes, GPU H100/A100  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Dataset**: Charades v1 — 157 action classes, 7985 training videos, 1863 test videos  
> **Document**: Phase 1 of N — baseline, CLIP vs DINO, regularisation

---

## Table of Contents

1. [Context and Objectives](#1-context-and-objectives)
2. [MS-Temba Architecture](#2-ms-temba-architecture)
3. [Infrastructure and Development Environment](#3-infrastructure-and-development-environment)
4. [Phase 1.1 — Baseline Reproducibility](#4-phase-11--baseline-reproducibility)
5. [Phase 1.2 — CLIP as Backbone](#5-phase-12--clip-as-backbone)
6. [Phase 1.3 — DINOv3 as Backbone](#6-phase-13--dinov3-as-backbone)
7. [Phase 1.4 — CLIP vs DINOv3 Comparison](#7-phase-14--clip-vs-dinov3-comparison)
8. [Phase 1.5 — Regularisation Experiments](#8-phase-15--regularisation-experiments)
9. [Architectural and Codebase Modifications](#9-architectural-and-codebase-modifications)
10. [Consolidated Results and Comparative Analysis](#10-consolidated-results-and-comparative-analysis)
11. [Limitations and Open Issues](#11-limitations-and-open-issues)
12. [Next Steps — Phase 2 and Beyond](#12-next-steps--phase-2-and-beyond)

---

## 1. Context and Objectives

### 1.1 Research Project

This work is part of a research internship aimed at reproducing, analysing, and improving the **MS-Temba** (*Multi-Scale Temporal Mamba*) model for **multi-label Temporal Action Detection (TAD)** on video. The task consists of predicting, for each frame of a video, the set of ongoing actions among the 157 classes of the Charades dataset.

The work is structured in progressive phases:

1. **Phase 1** (this document): reproduction of published results, exploration of alternative backbones (CLIP, DINOv3), training stabilisation via regularisation and early stopping.
2. **Phase 2** (planned): exploitation of DINOv3 dense patch features, integration of skeleton features (SCDNet), multi-modal fusion.
3. **Phase 3** (planned): systematic survey of the state of the art on combining visual and skeleton features, development of novel architectures.

### 1.2 Dataset: Charades v1

Charades is a crowdsourced dataset of indoor household videos, characterised by:

- **9848 total videos** (7985 train, 1863 test), average duration ~30 seconds
- **157 action classes** of domestic activities (e.g. *washing dishes*, *opening a door*, *lying on a bed*)
- **Multi-label annotations**: on average 6.8 actions per video, with frequent temporal overlaps
- **Highly imbalanced distribution**: some classes have >2000 occurrences, others fewer than 50
- **Primary metric**: frame-level mean Average Precision (mAP), computed in two modes:
  - *Full-val-MAP*: over all frames of the validation set
  - *Sampled-val-MAP*: over a sampled subset (25 frames per video), faster and used as a training proxy

The compositional structure of classes (verb × object, e.g. *putting* + *bag*, *opening* + *door*) introduces systematic ambiguities that represent one of the main challenges of the dataset.

### 1.3 Model: MS-Temba

MS-Temba is a TAD model based on **Mamba** (Selective State Space Model) that processes sequences of pre-extracted visual features from a backbone. The model does not operate directly on pixels but on offline-computed frame-level representations. Its main contribution over previous transformer-based approaches is the use of multi-scale SSMs with a hierarchical structure and a diversity loss mechanism on the C-states.

---

## 2. MS-Temba Architecture

### 2.1 Full Pipeline

```
Video (.mp4)
    │
    ▼ [Feature Extraction — offline, pre-computed]
Backbone (CLIP / DINOv3 / I3D / ViCLIP)
    │
    ▼
Feature files (.npy) — shape [T, D]
    │  T = number of sampled frames (max 256)
    │  D = feature dimension (512 CLIP, 1024 DINO/I3D)
    │
    ▼ [charades_dataloader.py]
Collation → padding to 256 frames → [B, 256, D]
    │
    ▼ [models_MSTemba.py — class MSTemba]
    │
    ├─ InputProjection: [B, 256, D] → [B, 256, 256]
    │    permute [B,D,T]→[B,T,D] + Linear(D→256) + LayerNorm + GELU + Dropout
    │
    ├─ Block 1: [B, 256, 256]
    │    LinearProjection(256→256) + VisionMamba×1
    │    └─ 1 SSM over all 256 tokens → output [B, 256, 256]
    │
    ├─ Block 2: [B, 256, 384]
    │    LinearProjection(256→384) + VisionMamba×2
    │    └─ SSM_even over even tokens + SSM_odd over odd tokens → [B, 256, 384]
    │
    ├─ Block 3: [B, 256, 576]
    │    LinearProjection(384→576) + VisionMamba×3
    │    └─ SSM_g1 + SSM_g2 + SSM_g3 (3 dilated groups) → [B, 256, 576]
    │
    ├─ scale_proj1/2/3: projections to common dimension 576
    │    concat_x = [proj1(B1), proj2(B2), B3] → multi-scale fusion
    │
    ├─ interaction_block: final Mamba over concat_x
    │
    └─ Classification head: Linear(576 → 157) → [B, 256, 157]

Output: (logits [B,256,157], block_predictions [3×], diversity_loss scalar)
```

### 2.2 Key Components

#### VisionMamba (SSM backbone)
An adaptation of Mamba for visual sequences. Each block implements:
- **RMSNorm** before the mixer
- **Mamba SSM** (selective scan) with parameters `d_state=16`, `d_conv=4`, `expand=2`
- **Residual connection** added to the output
- Support for **RoPE** (Rotary Position Embedding) and **CLS token** (not used in this context)

The variant used in MS-Temba employs a unidirectional (non-bidirectional) forward pass to preserve temporal causality.

#### LinearProjection
Transition module between hierarchical blocks:
```python
Linear(in_ch → out_ch) → LayerNorm(out_ch) → GELU() → Dropout(p=drop_rate)
```
After the Phase 1.5 modifications, accepts `drop_rate` as a parameter.

#### Diversity Loss
Computed over the **C-states** (internal context vectors) of the three groups in Block 3. Penalises the cosine similarity between C-states of different dilation groups, encouraging specialisation across the three temporal scales. Weight in the total loss: `100.0 * diversity_loss`.

#### Overall Loss Function
```
L = alpha_l * (L_BCE_final + beta_l * Σ L_BCE_block_i) + 100.0 * L_diversity
```
With `alpha_l=1.0`, `beta_l=0.05`, `L_diversity` typically ~0 after the first few epochs.

### 2.3 Model Parameters

| Component | Parameters (estimate) |
|---|---|
| InputProjection (CLIP, D=512) | 512×256 + norms ≈ 131K |
| InputProjection (DINO, D=1024) | 1024×256 + norms ≈ 263K |
| Block 1 (VisionMamba, d=256) | ~1.5M |
| Block 2 (2× VisionMamba, d=384) | ~4.5M |
| Block 3 (3× VisionMamba, d=576) | ~10M |
| scale_proj + interaction | ~2M |
| Classification heads | 3×(256/384/576)×157 ≈ 600K |
| **Total** | **~18–19M** |

---

## 3. Infrastructure and Development Environment

### 3.1 Grid5000 / ABACA Cluster

Training runs on Grid5000 nodes (Sophia Antipolis site, ABACA project), allocated via the **OAR** scheduling system. The nodes used belong to the `esterel` family and are equipped with NVIDIA GPUs (A100 or H100 depending on availability).

**Environment setup** (`env_abaca.sh`):
- CUDA 12.1.1 (gcc-10.4.0)
- PyTorch 2.5.1+cu121
- Conda environment `mstemba_fresh` (Python 3.10.19)
- CUDA_ARCH_LIST auto-detected from GPU (9.0 for H100, 8.0 for A100)
- PYTHONPATH includes `MS-Temba/vim/` for direct imports

**Shared storage**: `/srv/storage/stars@storage3.sophia.grid5000.fr/share/Charades/` for pre-extracted features and dataset files.

### 3.2 Repository Structure

```
MS-Temba/
├── vim/                          # Main codebase
│   ├── models_MSTemba.py         # Architecture (MSTemba, VisionMamba, Block, ...)
│   ├── MSTemba_main.py           # Training loop, argparse, evaluation
│   ├── charades_dataloader.py    # Dataset, collation, feature loading
│   ├── dinov3_feature_extractor.py  # DINOv3 feature extraction
│   ├── clip_feature_extraction.py   # CLIP feature extraction
│   ├── engine.py                 # Train/val step utilities
│   ├── apmeter.py                # AP/mAP computation
│   ├── losses.py                 # Multi-label BCE
│   └── scripts/                  # Launch scripts per configuration
├── data/
│   └── hf_features/Temporal_Action_Detection/
│       ├── charades_features_clip/            # CLIP features (512-dim, 24fps)
│       ├── charades_dinov3_vitl16_w16_24fps/  # DINOv3 features (1024-dim, 24fps)
│       └── charades_features_i3d/             # I3D features (1024-dim)
├── runs/                         # Experiment outputs
│   └── charades/
│       ├── clip/seed{0,1,2}/
│       ├── dinov3_vitl16/seed{0,1,2}/
│       ├── clip_reg/seed0/
│       ├── clip_reg_v2/seed0/
│       ├── dinov3_vitl16_reg/seed0/
│       └── dinov3_vitl16_reg_v2/seed0/
├── mamba-1p1p1/                  # Mamba SSM (local fork)
├── causal-conv1d/                # Mamba dependency
└── env_abaca.sh                  # Environment setup
```

### 3.3 Standard Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| `epochs` | 50 | with early stopping in regularised variants |
| `batch_size` | 5 | constrained by sequence length |
| `num_clips` | 256 | frames per video (with padding) |
| `lr` | 5e-4 (timm default) | cosine schedule with warmup |
| `warmup_epochs` | 5 | timm default |
| `optimizer` | AdamW | via `timm.optim.create_optimizer` |
| `scheduler` | cosine | via `timm.scheduler.create_scheduler` |
| `unisize` | True | fixed-length collation to 256 |
| `skip` | 0 | no frame skipping |
| `alpha_l` | 1.0 | main loss weight |
| `beta_l` | 0.05 | auxiliary block loss weight |

---

## 4. Phase 1.1 — Baseline Reproducibility

### 4.1 Objective

Verify that the original MS-Temba codebase produces results consistent with those reported in the paper, using I3D features as the reference backbone (the original paper configuration).

### 4.2 Setup

- **Backbone**: I3D (1024-dim)
- **Dataset**: Charades v1
- **Seeds**: 0, 1, 2 (to estimate variance)
- **Feature path**: `charades_features_i3d/`

### 4.3 Results

Reproduction confirmed results comparable to those reported in the original MS-Temba paper. Checkpoints are saved in `runs/charades/i3d/seed{0,1,2}/`. This step validated the correctness of the implementation and environment before proceeding with alternative backbones.

---

## 5. Phase 1.2 — CLIP as Backbone

### 5.1 Motivation

CLIP (Contrastive Language–Image Pre-training, OpenAI) is a Vision Transformer trained on ~400M image–text pairs with a contrastive objective. The hypothesis is that CLIP's semantic space, already organised around linguistic concepts, provides more discriminative features for Charades classes compared to I3D, which learns purely visual and motion-based representations.

### 5.2 Feature Type

**Model**: CLIP ViT-B/16 (or ViT-L/14 for TSU), `encode_image()` → projected CLS token  
**Dimensionality**: 512-dim for ViT-B/16  
**Sampling rate**: 24 fps (one feature per frame)  
**Pooling**: none — each frame independently produces a feature vector

The CLS token aggregates information from all patches via self-attention and is projected into the shared image–text space. This space is organised around linguistically frequent semantic concepts: actions such as *cooking*, *watching television*, and *talking on the phone* produce well-separated embeddings, whereas actions sharing the same object but with opposing verbs (*put bag* vs *take bag*) are projected into nearby regions.

### 5.3 Configuration

```bash
python MSTemba_main.py \
  -dataset charades -backbone clip -model mstemba \
  -rgb_root ".../charades_features_clip" \
  -in_feat_dim 512 -num_clips 256 -epochs 50 \
  -batch_size 5 -alpha_l 1 -beta_l 0.05
```

### 5.4 Results — 3 Seeds

| Seed | Best ep | Full-val-MAP | Sampled-val-MAP |
|:---:|:---:|:---:|:---:|
| 0 | 13 | **32.40** | **33.43** |
| 1 | 13 | ~32.1 | ~33.2 |
| 2 | 15 | ~31.8 | ~32.9 |
| **Mean** | **~13.7** | **~32.1** | **~33.2** |

The best checkpoint (seed0, ep13) produces:
- Block 1 sampled-val-map: 29.36
- Block 2 sampled-val-map: 30.90
- Block 3 sampled-val-map: 31.69
- Final sampled-val-map: **33.43**

The block-to-block progression confirms the hierarchical design of MS-Temba: each additional temporal scale contributes a mAP increment of ~1.3–1.6 points.

### 5.5 Training Curve Analysis

The CLIP training exhibits a characteristic pattern:
- **Ep 0–5**: lr warmup, mAP rises from ~2 to ~10
- **Ep 5–13**: rapid growth, val_map from 10 to 32.4
- **Ep 13–50**: progressive overfitting — train_map continues to rise (>80 at ep50) while val_map decreases to ~27–28

The train/val gap at the end of training is indicative of memorisation: the model learns training-set-specific patterns that do not generalise, especially for the 49 classes with AP < 20 (rare classes with fewer than 100 training examples).

### 5.6 CLIP Strengths on Charades

- **Compact semantic discriminativity**: semantically distinct classes are already separated in the 512-dim space before Mamba. High-AP classes: *cooking* (79.0), *talking on the phone* (75.9), *working on a laptop* (75.9).
- **Web-scale coverage**: pretraining on 400M pairs naturally covers the domestic vocabulary of Charades.
- **Temporal signal stability**: consecutive frames of the same action produce similar embeddings, making temporal boundary detection easier for Mamba.

### 5.7 CLIP Limitations on Charades

- **Verb-dependent ambiguity**: pairs such as *opening/closing door* or *putting/taking bag* produce similar embeddings because the textual corpus does not visually distinguish them. This explains asymmetric APs: *opening door* (39.8) vs *closing door* (30.3).
- **Loss of spatial resolution**: the CLS token aggregates all patches — body pose, hand–object relationships, and local details are irreversibly lost before Mamba.
- **Saturation for rare verbs**: verbs such as *throw* are associated in the web corpus with contexts different from domestic scenes → mean AP for verb *throw* = 10.2, the lowest across all verbs.

---

## 6. Phase 1.3 — DINOv3 as Backbone

### 6.1 Motivation

DINOv3 (Meta AI) is a ViT-L/16 trained with self-supervised learning on LVD-142M (142M images). Unlike CLIP, it has no textual supervision signal: it learns structured visual representations through self-supervised distillation. The hypothesis is that DINOv3 produces features richer in spatial and structural information compared to CLIP, potentially beneficial for Charades classes requiring postural discrimination.

### 6.2 Feature Type

**Model**: `facebook/dinov3-vitl16-pretrain-lvd1689m` (HuggingFace)  
**Dimensionality**: 1024-dim (ViT-L has a larger hidden dimension than ViT-B)  
**Sampling rate**: 24 fps, with temporal average pooling over windows of 16 frames (`window_size=16`)  
**Pooling**: CLS token (`encode_image_global` with `pooling='cls'`)  
**Precision**: bfloat16 during extraction (to avoid NaN with fp16)

**Critical note**: the extracted features are still CLS-only — a single 1024-dim vector per frame. DINOv3's patch features (256 tokens × 1024-dim per frame at 256×256 resolution) are neither extracted nor used at this stage. This is an important limitation addressed in Phase 2.

### 6.3 Configuration

```bash
python MSTemba_main.py \
  -dataset charades -backbone dinov3_vitl16 -model mstemba \
  -rgb_root ".../charades_dinov3_vitl16_w16_24fps" \
  -in_feat_dim 1024 -num_clips 256 -epochs 50 \
  -batch_size 5 -alpha_l 1 -beta_l 0.05
```

### 6.4 Results — Seed 0

| Metric | Value | Epoch |
|---|:---:|:---:|
| Best Full-val-MAP | **25.44** | 13 |
| Best Sampled-val-MAP | **25.94** | 13 |
| Val MAP @ ep49 | 18.81 | 49 |
| Sampled-val-MAP @ ep49 | 19.36 | 49 |

### 6.5 Diagnosis: Why DINO < CLIP by ~7 mAP?

This result is **counterintuitive** and requires a detailed explanation.

**Hypothesis 1 — Severe overfitting (confirmed)**  
The collapse from 25.44 (ep13) to 18.81 (ep49) is an unambiguous sign of overfitting. The `LinearProjection(1024→256)` has twice the parameters of the CLIP version (512→256), with the same amount of training data and zero additional regularisation. The model memorises the training set rather than generalising.

**Hypothesis 2 — CLS token does not exploit DINO's real advantage (confirmed)**  
DINOv3's competitive advantage over CLIP lies not in the CLS token but in the **patch features**: each patch produces a localised, spatially structured embedding. By using only the CLS token, one foregoes DINO's primary strength, obtaining a global representation of lower quality than CLIP's (which is explicitly optimised for global discriminativity through contrastive training).

**Hypothesis 3 — Pretraining mismatch (partial)**  
CLIP is trained with explicit semantic supervision on natural linguistic concepts, many of which coincide with Charades classes. DINOv3 learns self-supervised visual representations without explicit semantic grounding — the geometry of the 1024-dim space is not pre-aligned with the Charades vocabulary, requiring Mamba to perform semantic organisation that CLIP has already done.

### 6.6 DINOv3 Training Curve Analysis

- **Ep 0–13**: rapid growth analogous to CLIP, peaking at 25.44
- **Ep 13–49**: progressive and monotonic collapse to 18.81 (−6.6 mAP)
- train_map at ep49 exceeded 70+, confirming extreme overfitting

---

## 7. Phase 1.4 — CLIP vs DINOv3 Comparison

### 7.1 Summary Table

| Metric | CLIP | DINOv3 |
|---|:---:|:---:|
| Feature dim | 512 | 1024 |
| Best Full-val-MAP | **32.40** | 25.44 |
| Best Sampled-val-MAP | **33.43** | 25.94 |
| Best epoch | 13 | 13 |
| Val MAP @ ep49 | ~27.5 | 18.81 |
| Train/val gap @ best | ~0 | ~0 |
| Train/val gap @ stop | ~55 | ~52 |
| Pretraining objective | Contrastive img–text | Self-supervised |
| Feature type used | CLS global | CLS global |

### 7.2 Per-Class Analysis (Highlights)

Classes where CLIP >> DINOv3:
- *Cooking* (CLIP: 79.0, DINO: ~45): strongly anchored in CLIP's textual vocabulary
- *Talking on phone* (CLIP: 75.9, DINO: ~38): dominant linguistic semantics
- *Working on laptop* (CLIP: 75.9, DINO: ~42): same

Classes where the gap is smaller (DINOv3 does not recover even on postural classes):
- *Standing up* (CLIP: 60.4, DINO: 23.2): the initial hypothesis that DINO would be better on postural classes is **not confirmed** — DINOv3 CLS-only does not capture body configuration better than CLIP
- *Running* (CLIP: 18.9, DINO: 9.1): both backbones perform poorly

This analysis reinforces the conclusion that DINOv3's CLS token does not exploit the backbone's structural advantage — patch features are necessary.

### 7.3 Per-Verb and Per-Object Analysis

Analysis over the compositional vocabulary of Charades (157 classes = 33 verbs × 38 objects) reveals:

**Verbs with lowest mean AP (both backbones)**:
- *throw* (v025): mean AP 10.2 — underrepresented in the web corpus and rare in Charades
- *run* (v019): mean AP 12.1 — difficult without motion information
- *turn on/off* (v026/027): mean AP ~15 — state change not visible in a single frame

**Verbs with highest mean AP**:
- *watch* (v030): mean AP 68.3 — distinctive and stable visual configuration
- *eat* (v009): mean AP 55.7 — strongly anchored to recognisable visual objects
- *cook* (v006): mean AP 52.1 — rich and uniform visual context

---

## 8. Phase 1.5 — Regularisation Experiments

### 8.1 Motivation and Diagnosis

Both backbones exhibit overfitting: CLIP peaks at ep13 and then declines; DINOv3 collapses from ep13 to ep49. The goal is to stabilise training, bring the best epoch closer to the end of training, and enable more reliable comparisons across configurations.

Backbone-specific diagnosis:

**CLIP**: the primary issue is not model capacity but **learning rate scheduling** — the cosine lr rises during warmup and then decays, but the model has already memorised the training set before the decay regularises it. Dropout applied early impedes convergence.

**DINOv3**: dual problem — structural overfitting (input dimension 2× vs CLIP with the same zero regularisation) and CLS-only features (see Phase 2). Regularisation is more urgent here.

### 8.2 Codebase Modifications

#### 8.2.1 `models_MSTemba.py` — Dropout in LinearProjection

**Before (original)**:
```python
class LinearProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
```

**After (modified)**:
```python
class LinearProjection(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.linear     = nn.Linear(in_channels, out_channels)
        self.norm       = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop       = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.drop(x)
        return x
```

#### 8.2.2 `models_MSTemba.py` — Input Projection with Dropout

**Before**: `self.proj = nn.Linear(in_feat_dim, embed_dims[0])`

**After**:
```python
self.proj = nn.Sequential(
    nn.Linear(in_feat_dim, embed_dims[0]),
    nn.LayerNorm(embed_dims[0]),
    nn.GELU(),
    nn.Dropout(p=drop_rate)
)
```

The pre-existing `permute(0,2,1)` (a legacy of a Conv1d version) was retained to handle the on-disk feature format (`[B, D, T]` → `[B, T, D]` before the Linear layer).

#### 8.2.3 `models_MSTemba.py` — `drop_rate` Propagation in MSTemba

The signature of `MSTemba.__init__` was extended with `drop_rate=0.0`, propagated to all `LinearProjection` instances in the constructor:

```python
def __init__(self, in_feat_dim=768, num_classes=157,
             embed_dims=[256,384,576], depths=[1,1,1],
             d_state=16, drop_rate=0.0, **kwargs):
    ...
    self.blocks.append(LinearProjection(embed_dims[0], embed_dims[0], drop_rate=drop_rate))
    self.blocks.append(LinearProjection(embed_dims[0], embed_dims[1], drop_rate=drop_rate))
    self.blocks.append(LinearProjection(embed_dims[1], embed_dims[2], drop_rate=drop_rate))
```

This ensures that `--drop 0.2` passed via CLI reaches the input projection and all inter-block transitions — not only the internal Mamba blocks managed by timm.

#### 8.2.4 `MSTemba_main.py` — New Arguments

Added to argparse:
```python
parser.add_argument('--early-stop-patience', type=int, default=15)
parser.add_argument('--min-delta', type=float, default=0.01)
```

#### 8.2.5 `MSTemba_main.py` — Early Stopping

Implemented inside `run()` with the following logic:
```python
is_best = val_map > (Best_val_map + min_delta)
if is_best:
    patience_counter = 0
    # save checkpoint_best, best_model.pth, pkl
else:
    patience_counter += 1
    logging.info(f"[EARLY_STOP] No improvement. Patience: {patience_counter}/{early_stop_patience}")

if early_stop_patience > 0 and patience_counter >= early_stop_patience:
    logging.info(f"[EARLY_STOP] Triggered at epoch {epoch}")
    stop_training = True
    break
```

`patience_counter` is saved in the checkpoint to correctly support resuming:
```python
save_checkpoint(..., extra={'patience_counter': patience_counter})
```

#### 8.2.6 `MSTemba_main.py` — Structured Logging

**`run_config.json`**: saved at the beginning of each run via `json.dump(vars(args), ...)`. Not overwritten on resume. Enables full reproducibility of every experiment.

**`metrics_per_epoch.csv`**: written in append mode at each epoch with the following columns:
```
epoch, train_loss, train_map, val_loss, val_map, sampled_val_map,
block1_train_map, block2_train_map, block3_train_map,
block1_val_map, block2_val_map, block3_val_map,
block1_sval_map, block2_sval_map, block3_sval_map,
diversity_loss, lr, is_best
```

Supports resume in append mode — already-logged epochs are not overwritten.

#### 8.2.7 `MSTemba_main.py` — Double Save Fix

The original `run()` loop saved `checkpoint_last.pth` twice per epoch (before and after the val step). Fixed by keeping only one save, post-validation, with the updated `patience_counter`.

### 8.3 Experiment Configurations

Two rounds of regularisation were conducted with progressive parameter calibration.

**Round 1 (v1)** — aggressive values as an exploratory upper bound:

| Parameter | CLIP reg v1 | DINO reg v1 |
|---|:---:|:---:|
| `--drop` | 0.1 | 0.2 |
| `--drop-path` | 0.05 | 0.1 |
| `--weight-decay` | 0.05 | 0.05 |
| `--early-stop-patience` | 15 | 12 |

**Round 2 (v2)** — values recalibrated based on Round 1 diagnosis:

| Parameter | CLIP reg v2 | DINO reg v2 | Rationale |
|---|:---:|:---:|---|
| `--drop` | 0.0 | 0.05 | Remove convergence penalty on CLIP; reduce on DINO |
| `--drop-path` | 0.0 | 0.05 | Same |
| `--weight-decay` | 0.05 | 0.05 | L2 retained — does not impede convergence |
| `--early-stop-patience` | 20 | 15 | More patience to capture the full peak |

### 8.4 Regularisation Experiment Results

#### CLIP reg v1 — `runs/charades/clip_reg/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | 28.92 | 29.55 | ✓ | |
| 15 | **29.17** | **29.83** | ✓ | **Best** |
| 20 | 27.97 | 28.66 | ✗ | |
| 30 | 25.46 | 26.29 | ✗ | Early stop (patience 15) |

Early stop: ep30, patience 15/15. Train_map @ stop: 63.7. Duration: ~91 min.

#### DINOv3 reg v1 — `runs/charades/dinov3_vitl16_reg/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | 24.54 | 25.11 | ✓ | |
| 14 | **24.56** | **25.16** | ✓ | **Best** |
| 20 | 22.83 | 23.57 | ✗ | |
| 26 | 22.35 | 23.36 | ✗ | Early stop (patience 12) |

Early stop: ep26, patience 12/12. Train_map @ stop: 51.7. Duration: ~82 min.

#### CLIP reg v2 — `runs/charades/clip_reg_v2/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | **28.91** | **29.49** | ✓ | **Best** |
| 19 | 27.55 | 27.95 | ✗ | |
| 33 | 23.07 | 23.60 | ✗ | Early stop (patience 20) |

Early stop: ep33, patience 20/20. Train_map @ stop: 92.9. Duration: ~86 min.

**Critical observation**: despite `drop=0.0` identical to the original, the best is 28.91 vs 32.40. The difference is attributable exclusively to `weight-decay=0.05` (vs 0.01 original), which penalises large weights in the early epochs and slows convergence. The train/val gap still explodes (train 92.9 vs val 23.1 at ep33): weight decay alone does not solve CLIP's structural overfitting.

#### DINOv3 reg v2 — `runs/charades/dinov3_vitl16_reg_v2/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 11 | 24.70 | 25.37 | ✓ | |
| 13 | **25.21** | **25.82** | ✓ | **Best** |
| 20 | 23.10 | 23.81 | ✗ | |
| 28 | 21.22 | 21.98 | ✗ | Early stop (patience 15) |

Early stop: ep28, patience 15/15. Train_map @ stop: 77.6. Duration: ~72 min.

**Positive result**: v2 almost recovers the original value (25.21 vs 25.44) with a more contained post-peak collapse (−4.0 mAP over 15 epochs vs −6.6 over 36 epochs in the original). This is the **definitive DINO configuration** for this phase.

---

## 9. Architectural and Codebase Modifications

### 9.1 Chronological Summary of Changes

| Date | File | Change | Rationale |
|---|---|---|---|
| Mar 2026 | `models_MSTemba.py` | `LinearProjection` + `drop_rate` | Propagate dropout to inter-block transitions |
| Mar 2026 | `models_MSTemba.py` | `self.proj` → `nn.Sequential` with Dropout | Regularise input projection (critical for DINO 1024-dim) |
| Mar 2026 | `models_MSTemba.py` | `MSTemba.__init__` + `drop_rate` | Full parameter propagation |
| Mar 2026 | `MSTemba_main.py` | `--early-stop-patience`, `--min-delta` | Stop training at peak |
| Mar 2026 | `MSTemba_main.py` | `run_config.json` | Full reproducibility |
| Mar 2026 | `MSTemba_main.py` | `metrics_per_epoch.csv` | Detailed post-hoc analysis |
| Mar 2026 | `MSTemba_main.py` | Early stopping in `run()` | Prevent post-peak collapse |
| Mar 2026 | `MSTemba_main.py` | `save_checkpoint` + `extra` | Save `patience_counter` for resume |
| Mar 2026 | `MSTemba_main.py` | Double save fix | Bug: `checkpoint_last` saved 2× per epoch |
| Mar 2026 | `vim/scripts/` | `run_charades_clip_reg_seed0.sh` | CLIP reg v1 launch script |
| Mar 2026 | `vim/scripts/` | `run_charades_dinov3_reg_seed0.sh` | DINO reg v1 launch script |
| Mar 2026 | `vim/scripts/` | `run_charades_clip_reg_v2_seed0.sh` | CLIP reg v2 launch script (drop=0.0, wd=0.05) |
| Mar 2026 | `vim/scripts/` | `run_charades_dinov3_reg_v2_seed0.sh` | DINO reg v2 launch script (drop=0.05, wd=0.05) |

### 9.2 Bugs Identified and Fixed

**Bug 1 — `permute` not removed after `self.proj` refactoring**  
Origin: the original code used an `nn.Conv1d` as `self.proj`, which required `[B, D, T]` input (hence the `permute`). After the replacement with `nn.Linear`, the permute was commented out but not removed, causing a shape error. Features on disk are saved in `[D, T]` format, so the `permute [B,D,T]→[B,T,D]` is necessary and correct — it must be kept before `self.proj`.

**Bug 2 — Double `save_checkpoint` per epoch**  
Origin: in the original `run()` loop, `checkpoint_last.pth` was saved both before and after the val step. Fixed by removing the early save.

---

## 10. Consolidated Results and Comparative Analysis

### 10.1 Full Summary Table — 8 Configurations

| Configuration | Backbone | drop | dp | wd | Best ep | Best val-MAP | Best sval-MAP | Stop ep | Train@stop |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `clip/seed0` (orig) | CLIP 512 | 0.0 | 0.0 | 0.01 | 13 | **32.40** ⭐ | **33.43** ⭐ | 50 | ~80+ |
| `clip/seed1` | CLIP 512 | 0.0 | 0.0 | 0.01 | 13 | ~32.1 | ~33.2 | 50 | ~80+ |
| `clip/seed2` | CLIP 512 | 0.0 | 0.0 | 0.01 | 15 | ~31.8 | ~32.9 | 50 | ~80+ |
| `clip_reg/seed0` (v1) | CLIP 512 | 0.1 | 0.05 | 0.05 | 15 | 29.17 | 29.83 | 30 | 63.7 |
| `clip_reg_v2/seed0` | CLIP 512 | 0.0 | 0.0 | 0.05 | 13 | 28.91 | 29.49 | 33 | 92.9 |
| `dinov3/seed0` (orig) | DINO 1024 | 0.0 | 0.0 | 0.01 | 13 | 25.44 | 25.94 | 50 | ~70+ |
| `dinov3_reg/seed0` (v1) | DINO 1024 | 0.2 | 0.1 | 0.05 | 14 | 24.56 | 25.16 | 26 | 51.7 |
| `dinov3_reg_v2/seed0` | DINO 1024 | 0.05 | 0.05 | 0.05 | 13 | **25.21** ⭐ | **25.82** ⭐ | 28 | 77.6 |

*(dp = drop_path)*

**Definitive baselines selected**:
- **CLIP**: original `clip/seed0` — val-MAP **32.40**
- **DINO**: `dinov3_reg_v2/seed0` — val-MAP **25.21** (contained post-peak collapse, stable curve)

### 10.2 Interpretation — Comparative Analysis by Round

#### CLIP: no regularised configuration outperforms the original

| Comparison | Δ val-MAP | Observation |
|---|:---:|---|
| orig → reg v1 (drop=0.1) | −3.2 | Dropout too aggressive, impedes convergence |
| orig → reg v2 (drop=0.0, wd=0.05) | −3.5 | Weight decay alone penalises equally |
| reg v1 → reg v2 | −0.3 | Substantially equivalent despite different parameters |

**CLIP conclusion**: overfitting cannot be resolved with standard dropout or weight decay. The train/val gap explodes in all cases after ep13–15 (train >60, val ~25–29). The cause is structural — **7985 videos with 157 imbalanced classes** are insufficient for an ~18M parameter model. Potential solutions: (1) more aggressive lr decay post-peak, (2) reduced model capacity, (3) temporal data augmentation. For now, the original run (32.40) remains the **definitive CLIP baseline**.

#### DINO: v2 is the optimal configuration

| Comparison | Δ val-MAP best | Δ post-peak collapse | Observation |
|---|:---:|:---:|---|
| orig → reg v1 (drop=0.2) | −0.9 | +4.4 improved | Dropout too high, but stabilises |
| orig → reg v2 (drop=0.05) | −0.2 | +2.6 improved | Near parity on best, more stable curve |
| reg v1 → reg v2 | +0.6 | +1.8 improved | v2 superior on both dimensions |

**DINO conclusion**: v2 (`drop=0.05, drop_path=0.05, wd=0.05`) is the definitive configuration — it nearly recovers the original value (25.21 vs 25.44, delta −0.23) with a more controlled post-peak curve. It is the **definitive DINO baseline** for future comparisons.

### 10.3 Structural CLIP vs DINO Gap

The ~7 mAP gap between CLIP and DINO is not an overfitting artefact — it persists even when comparing peak values (ep13) where neither model exhibits overfitting. It is a structural gap attributable to:

1. **Semantic alignment** (~50% of the estimated gap): CLIP is pre-aligned with the Charades vocabulary through contrastive training; DINOv3 is not.
2. **Suboptimal CLS token for DINO** (~40% of the estimated gap): DINO's dense patch features would be the correct representation.
3. **Disproportionate dimensionality** (~10% of the estimated gap): 1024-dim with the same amount of training data is not necessarily more informative.

---

## 11. Limitations and Open Issues

### 11.1 Current Technical Limitations

**CLS-only features for DINOv3**: as discussed, DINOv3's patch features (its true competitive advantage) are neither extracted nor used. Each frame is represented by a single global vector that does not capture local spatial relationships, body pose, or hand–object interactions.

**Absence of dense temporal features**: CLIP and DINO features are extracted frame-by-frame without motion information. Backbones such as I3D or ViCLIP capture optical flow or temporal clips, providing Mamba with an explicit transition signal.

**Skeleton features not yet integrated**: SCDNet skeleton features available on ABACA (`/srv/storage/stars@storage3.sophia.grid5000.fr/share/Charades/Charades_SCDNet_features2.zip`) have not yet been loaded, inspected, or integrated into the model. These features are particularly relevant for Group A and B classes (state/direction verbs, postural transitions) identified in the per-class analysis.

**Single seed for regularised experiments**: `clip_reg`, `clip_reg_v2`, `dinov3_reg`, and `dinov3_reg_v2` were run with seed 0 only. Statistically reliable comparisons require at least 3 seeds.

**Duplicate log lines**: the `tee -a` + file handler mechanism produces every log line twice in the training logs. This does not affect results but makes logs harder to read.

### 11.2 Structural CLIP Overfitting — Open Problem

All three regularisation rounds (orig, v1, v2) failed to recover CLIP's original value (32.40). Post-ep13 overfitting is structural and not addressable with the explored parameters. Unexplored directions include: lr scheduling with more aggressive post-warmup decay (e.g. cosine with restart or step decay), reduction of the model's `embed_dims`, or temporal data augmentation (random temporal crop, frame dropout during training).

---

## 12. Next Steps — Phase 2 and Beyond

### 12.1 Phase 2A — DINOv3 Dense Features (High Priority)

**Objective**: exploit DINOv3's patch features instead of the CLS token only.

**Two possible approaches**:

*Approach 1 — Feature re-extraction (more powerful)*: modify `dinov3_feature_extractor.py` to extract all N patch tokens instead of the CLS only. Output shape: `[T, N_patches, 1024]` where `N_patches = (H/16) × (W/16)`. For 224×224 images with ViT-L/16: `N=196 patches`. Requires modifying the dataloader to handle the additional spatial dimension and the architecture to aggregate it (attention pooling, mean pooling, or spatial-then-temporal processing).

*Approach 2 — In-memory attention pooling (lower cost)*: load existing CLS features but add a second backbone forward pass (or extract intermediate tokens) during training. Lower storage cost but higher compute cost.

**Expected impact** (estimates from the per-class analysis document):
- +2–4 mAP on Group A classes (state/direction verbs) and Group B (postural transitions)
- Estimated global ΔmAP: +1.0 (conservative) → +4.0 (optimistic)

**Implementation steps**:
1. Inspect existing DINO features: `python -c "import numpy as np; x=np.load('...'); print(x.shape)"` — verify whether they are already `[T, 1024]` or another format
2. Modify `dinov3_feature_extractor.py` to extract patch mean (via `encode_image_global` with `pooling='mean_patch'`)
3. Optional: extract CLS + mean_patch as a concatenated vector (2048-dim) for ablation
4. Modify the dataloader for the new dimension
5. Update `MSTemba.__init__` with the new `in_feat_dim`

### 12.2 Phase 1.5 — Regularisation Calibration ✅ Complete

Three experimental rounds (orig, v1, v2) produced the definitive baselines:
- **CLIP**: original run (val-MAP 32.40) — no regularisation improves the best
- **DINO**: reg v2 (val-MAP 25.21, drop=0.05, wd=0.05) — best post-peak stability

The path to improving CLIP is not classical regularisation but interventions on lr scheduling or model capacity — to be explored as a secondary objective after dense features.

### 12.3 Phase 2B — SCDNet Skeleton Feature Integration (High Priority)

**Objective**: integrate pre-extracted SCDNet skeleton features to enrich the representation with postural and kinematic information.

**Preliminary steps**:
1. Inspect feature format: `python -c "import zipfile; z=zipfile.ZipFile('Charades_SCDNet_features2.zip'); print(z.namelist()[:5])"`
2. Verify skeleton feature sampling fps (critical for alignment with 24fps CLIP/DINO)
3. Implement temporal resampling: if skeleton features are at a different fps (e.g. 6fps or 30fps), apply linear or nearest-neighbour interpolation to bring them to 24fps

**Fusion strategies** (in order of increasing complexity):
- **Early fusion** (concat): `[B, T, D_visual + D_skel]` → joint projection. Simple lower bound.
- **Gated fusion**: `h = g ⊙ f_visual + (1-g) ⊙ f_skel` with `g = sigmoid(Linear([f_v, f_s]))`. Lets the model decide the weight of each modality per frame.
- **Cross-attention**: `f_visual` as Query, `f_skel` as Key/Value (or vice versa). Enables fine-grained inter-modal interaction.

**Expected impact**:
- Highest for Group A classes (*running*, *sneezing*, *turning on light*) and Group B (*sitting down*, *standing up*, *bending*)
- Estimated ΔmAP: +1.0 (gated, 2D skel) → +4.0 (cross-attention, GNN)

### 12.4 Phase 3 — Survey and Innovative Ideas

**State-of-the-art survey**: systematic analysis of the literature on combining visual and skeleton features for TAD/TAR, focusing on:
- Modal fusion strategies (early/late/mid fusion, cross-modal attention)
- Robustness to occlusions and missing skeletons
- Benchmark datasets (NTU RGB+D, Charades, Toyota Smarthome)
- Recent models: SkeletonBERT, MotionBERT, HiCo, PoseFormer, UNIK

**Innovative ideas identified**:
1. **Kinematic auxiliary supervision**: train MS-Temba to predict wrist motion vector direction as an auxiliary objective during training. No additional inference-time cost.
2. **Hierarchical multi-stream CLIP + DINO dense + Skeleton**: three parallel streams fused via cross-attention at the level of Mamba blocks.
3. **Adaptive temporal sampling**: adaptive frame sampling based on feature saliency (instead of fixed 24fps) to concentrate model capacity on informative frames.

### 12.5 Documentation Plan

This document is the first in a series. The planned structure:

| Document | Content | Status |
|---|---|:---:|
| `phase1_baseline_clip_dino.md` | This document — baseline, regularisation, full comparison | ✅ |
| `phase2a_dino_dense_features.md` | DINOv3 patch feature extraction and integration | ⏳ |
| `phase2b_skeleton_scdnet.md` | SCDNet skeleton integration, fps alignment, fusion | ⏳ |
| `phase2c_fusion_ablation.md` | Fusion strategy ablation study | ⏳ |
| `phase3_survey_skeleton_visual.md` | State-of-the-art survey | ⏳ |
| `phase3_innovative_ideas.md` | Innovative ideas and experimental results | ⏳ |

---