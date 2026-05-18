# MS-Temba on Charades: Comprehensive Experimental Analysis
### Multi-Modal Feature Integration for Temporal Action Detection

> **Author**: Matteo Di Iorio  
> **Period**: February–March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Dataset**: Charades v1 — 157 action classes, 7985 training videos, 1863 test videos  
> **Model**: MS-Temba (Multi-Scale Temporal Mamba), Pramanik et al. 2025, arXiv:2501.06138  
> **Document scope**: Full experimental record — baselines, dense visual features, skeleton features, fusion roadmap  
> **Last updated**: 27 March 2026

---

## Table of Contents

1. [Introduction and Research Objectives](#1-introduction-and-research-objectives)
2. [Experimental Framework](#2-experimental-framework)
3. [Phase 1 — Visual Feature Baselines](#3-phase-1--visual-feature-baselines)
4. [Phase 2A — DINOv3 Dense Feature Investigation](#4-phase-2a--dinov3-dense-feature-investigation)
5. [Phase 2B — Skeleton Feature Integration](#5-phase-2b--skeleton-feature-integration)
6. [Cross-Phase Comparative Analysis](#6-cross-phase-comparative-analysis)
7. [Synthesis and Key Findings](#7-synthesis-and-key-findings)
8. [Future Work — Fusion Architecture and Roadmap](#8-future-work--fusion-architecture-and-roadmap)
9. [References](#9-references)

---

## 1. Introduction and Research Objectives

### 1.1 Problem Statement

Temporal Action Detection (TAD) in untrimmed video requires simultaneously localising and classifying human actions across time. The Charades dataset poses a particularly challenging variant of this problem: it is a **multi-label** benchmark where multiple actions co-occur within a single video, often with overlapping temporal extents. The 157 action classes span a heterogeneous taxonomy that includes semantic categories (e.g., *cooking something*, *talking on a phone*), object-defined categories (e.g., *holding a vacuum*, *opening a refrigerator*), postural categories (e.g., *sitting down*, *walking*), and fine-grained manipulation categories (e.g., *putting a shoe somewhere*, *taking something from a box*).

This taxonomic diversity implies that **no single feature representation is optimal across all classes**. Visual features from large-scale pre-trained models such as CLIP excel at semantic and object-defined classes but struggle with actions discriminated by body kinematics. Conversely, skeleton-based representations capture postural dynamics but are blind to object identity and scene context. The central research question of this work is:

> *To what extent can multi-modal feature integration — combining visual semantics (CLIP, DINOv3) with kinematic representations (SCDNet skeleton) — improve temporal action detection on Charades within the MS-Temba architecture?*

### 1.2 Research Methodology

The investigation follows a **progressive ablation** methodology structured in four phases:

- **Phase 1 (Baselines)**: Establish definitive single-stream performance references for CLIP and DINOv3 CLS features, including regularisation ablations.
- **Phase 2A (Dense Visual Features)**: Investigate whether DINOv3 patch-level features provide complementary information beyond the CLS token.
- **Phase 2B (Skeleton Features)**: Extract, align, and evaluate SCDNet skeleton features as a standalone modality; analyse per-class complementarity with visual features.
- **Phase 2C/2D (Fusion)**: Design and evaluate multi-modal fusion architectures informed by the empirical findings of Phases 1–2B.

This document presents the complete experimental record for Phases 1, 2A, and 2B, together with a detailed analysis of the findings and the resulting fusion roadmap.

### 1.3 MS-Temba Architecture

MS-Temba (Multi-Scale Temporal Mamba) is a State Space Model (SSM) architecture for temporal action detection. The model processes pre-extracted video features through a multi-scale pipeline:

1. **Input Projection**: `Linear(in_feat_dim → 256)` followed by LayerNorm, GELU activation, and Dropout. This module transforms the backbone-specific feature dimension into the model's internal representation space.
2. **Block 1** (256-dim): Single SSM scan, capturing local temporal patterns.
3. **Block 2** (384-dim): Dual SSM with odd/even temporal interleaving, extending the receptive field.
4. **Block 3** (576-dim): Triple SSM with dilated scanning, capturing long-range temporal dependencies.
5. **Interaction Block**: Multi-scale feature aggregation across blocks.
6. **Classifier**: `Linear(576 → 157)` producing per-class logits.

The total parameter count is approximately 18M for the CLIP configuration (`in_feat_dim=768`). The architecture is agnostic to the input feature type — any pre-extracted `.npy` feature can be used by adjusting `in_feat_dim`.

**Training protocol**: BCE multi-label loss with diversity regularisation (weight=100.0) and auxiliary block-level losses (β=0.05). AdamW optimiser with learning rate 5×10⁻⁴, cosine scheduler (warmup=5 epochs, min_lr=1×10⁻⁵), EMA decay=0.99996. All videos are temporally padded to 256 windows.

---

## 2. Experimental Framework

### 2.1 Feature Extraction Summary

Five distinct feature sets were extracted and evaluated in this work, spanning three modalities:

| Feature set | Backbone | Dimension | Extraction rate | Files | Size | Status |
|-------------|----------|:---------:|:---------------:|------:|-----:|:------:|
| CLIP | ViT-B/16 (OpenAI) | 768 | ~1.5 fps (window=16) | 9848 | ~4.5 GB | ✅ |
| DINOv3 CLS | ViT-L/16 | 1024 | ~1.5 fps (window=16) | 9850 | ~8.2 GB | ✅ |
| DINOv3 mean_patch | ViT-L/16 | 1024 | ~1.5 fps (window=16) | 9848 | ~8.2 GB | ✅ |
| DINOv3 combined | ViT-L/16 | 2048 | ~1.5 fps (window=16) | 9848 | ~16.4 GB | ✅ |
| SCDNet skeleton | GNN on pose graph | 4096 | ~1.5 fps (aligned) | 9848 | ~18 GB | ✅ |

All visual features (CLIP, DINOv3 variants) are extracted at approximately 1.5 frames per second via temporal average pooling over non-overlapping windows of 16 video frames at 24fps. The skeleton features were originally extracted at ~24fps and subsequently aligned to the visual feature temporal resolution through identical window-16 average pooling (see Section 5.3 for details).

The source video frames reside in `data/charades_frames_24fps/`, with one subdirectory per video containing sequentially numbered JPEG frames.

### 2.2 Evaluation Protocol

All experiments use the standard Charades evaluation protocol:

- **Primary metric**: mean Average Precision (mAP) computed over all 157 classes on the full validation set (*Full-val-MAP*).
- **Secondary metric**: sampled validation mAP (*Sampled-val-MAP*), computed on a random subset for faster per-epoch monitoring.
- **Per-class AP**: class-level AP values reported at the best epoch for fine-grained analysis.
- **Early stopping**: patience-based with configurable threshold (`--min-delta`). Training halts when the validation mAP fails to improve by at least `min-delta` for `patience` consecutive epochs.

### 2.3 Dataset Anomalies

Several structural properties of the Charades annotation set affect interpretation of results across all experiments:

**Structurally ambiguous class pairs**: c104/c105 (*turning on/off a light*) map to visually identical frames; c010/c011 (*sitting at/on a table*) differ only in the preposition; c043/c044 (*taking from a box / putting in a box*) share the same hand–box interaction with reversed temporal direction. These classes impose a hard ceiling on achievable mAP.

**Extreme class imbalance**: c060 (*opening a box*) has only 46 training instances; c136 (*putting a glass somewhere*) has 35 training vs 143 test instances; c141 (*holding a towel*) has 506 training vs 20 test instances. These imbalances cause unstable per-class AP estimates and disproportionate influence on macro-averaged mAP.

---

## 3. Phase 1 — Visual Feature Baselines

### 3.1 Experimental Design

Phase 1 establishes definitive baselines for two visual backbones (CLIP ViT-B/16 and DINOv3 ViT-L/16) and investigates the impact of regularisation on each. The experimental matrix is:

| Config | Backbone | Dropout | Drop-path | Weight decay | Patience |
|--------|----------|:-------:|:---------:|:------------:|:--------:|
| clip/seed0 ⭐ | CLIP | 0.0 | 0.0 | 0.01 | 50 (none) |
| clip/seed1 | CLIP | 0.0 | 0.0 | 0.01 | 50 (none) |
| clip/seed2 | CLIP | 0.0 | 0.0 | 0.01 | 50 (none) |
| clip_reg_v1 | CLIP | 0.1 | 0.05 | 0.05 | 15 |
| clip_reg_v2 | CLIP | 0.0 | 0.0 | 0.05 | 20 |
| dinov3/seed0 | DINOv3 CLS | 0.0 | 0.0 | 0.01 | 50 (none) |
| dinov3_reg_v1 | DINOv3 CLS | 0.2 | 0.1 | 0.05 | 12 |
| dinov3_reg_v2 ⭐ | DINOv3 CLS | 0.05 | 0.05 | 0.05 | 15 |

Three seeds were run for the CLIP baseline to assess variance; single seeds for all other configurations.

### 3.2 Results

| Config | Best epoch | Full-val mAP | Sampled-val mAP | Stop epoch |
|--------|:----------:|:------------:|:----------------:|:----------:|
| **clip/seed0** ⭐ | 13 | **32.40** | 33.43 | 50 |
| clip/seed1 | 13 | ~32.1 | ~33.2 | 50 |
| clip/seed2 | 15 | ~31.8 | ~32.9 | 50 |
| clip_reg_v1 | 15 | 29.17 | 29.83 | 30 |
| clip_reg_v2 | 13 | 28.91 | 29.49 | 33 |
| dinov3/seed0 | 13 | 25.44 | 25.94 | 50 |
| dinov3_reg_v1 | 14 | 24.56 | 25.16 | 26 |
| **dinov3_reg_v2** ⭐ | 13 | **25.21** | 25.82 | 28 |

**Definitive baselines**: CLIP original at 32.40 mAP; DINOv3 CLS reg_v2 at 25.21 mAP.

### 3.3 Analysis and Conclusions from Phase 1

**Finding 1 — CLIP does not benefit from regularisation.** Both regularised CLIP configurations (reg_v1: 29.17, reg_v2: 28.91) substantially underperform the unregularised baseline (32.40), with a drop of −3.2 to −3.5 mAP. This is counter-intuitive given the severe overfitting observed in the baseline (train mAP reaches 99.52 by epoch 49 while val mAP declines to 28.73). The explanation lies in the nature of CLIP features: the pre-trained CLIP embedding space carries a strong **semantic-linguistic prior** that implicitly regularises the representation. The features are already well-structured for action classification; explicit regularisation constrains the model's capacity without reducing the generalisation gap.

**Finding 2 — DINOv3 reg_v2 is the optimal DINOv3 configuration.** Mild regularisation (drop=0.05, dp=0.05, wd=0.05) produces a validation curve that is more stable than the unregularised version (25.21 vs 25.44 mAP, a negligible −0.23 difference) but with substantially earlier convergence and reduced post-peak degradation. The unregularised DINOv3 run showed an apparent advantage on certain postural classes, but this was identified as an **overfitting artefact**: with reg_v2, CLIP wins on all postural classes.

**Finding 3 — CLIP dominates DINOv3 by +7.19 mAP overall.** The performance gap is substantial and systematic. CLIP's advantage derives from its vision-language pre-training, which provides semantically grounded features that align well with Charades' action taxonomy. The Charades class names are essentially natural language descriptions (*"cooking something"*, *"holding a vacuum"*), and CLIP's features encode exactly this type of semantic information.

**Finding 4 — DINOv3 excels on visually specific object-interaction classes.** Despite the overall deficit, DINOv3 outperforms CLIP on a targeted subset of classes where fine-grained visual appearance (texture, shape, spatial configuration) is more discriminative than semantic description:

| Class | CLIP AP | DINOv3 AP | Δ |
|-------|:-------:|:---------:|:-:|
| c143: Opening a refrigerator (v3) | 60.2 | 88.1 | +27.9 |
| c099: Putting a broom somewhere | 27.4 | 53.0 | +25.6 |
| c137: Holding a vacuum (v2) | 51.3 | 74.2 | +22.9 |
| c071: Putting a blanket somewhere (v2) | 18.5 | 38.8 | +20.3 |

These classes are characterised by **visually distinctive objects** (refrigerator, broom, vacuum, blanket) whose spatial appearance is more reliably captured by DINOv3's self-supervised visual features than by CLIP's language-aligned representations.

**Finding 5 — CLIP wins on semantically described classes.** Classes defined by their semantic label rather than their visual instantiation are strongly favoured by CLIP:

- *Cooking something* (c037): 79.0 CLIP vs ~45 DINOv3
- *Talking on a phone* (c117): 75.9 CLIP vs ~35 DINOv3
- *Working on a laptop* (c016): 75.9 CLIP vs ~40 DINOv3

These classes have high intra-class visual variability (cooking can look very different across videos) but consistent semantic descriptions that CLIP captures effectively.

### 3.4 CLIP Training Dynamics

The CLIP baseline exhibits a distinctive training trajectory:

| Epoch | Train mAP | Val mAP | Train–Val gap |
|------:|----------:|--------:|--------------:|
| 0 | ~2.0 | ~2.5 | −0.5 |
| 13 | 40.43 | **32.40** | 8.0 |
| 25 | ~75.0 | ~30.5 | ~44.5 |
| 49 | 99.52 | 28.73 | 70.8 |

The best validation performance occurs at epoch 13, with a moderate train–val gap of 8.0 points. After epoch 13, the model memorises the training set (reaching 99.52 train mAP) while validation degrades monotonically. This trajectory establishes the baseline overfitting pattern against which all subsequent experiments are compared.

---

## 4. Phase 2A — DINOv3 Dense Feature Investigation

### 4.1 Motivation

The DINOv3 ViT-L/16 encoder produces a sequence of 197 tokens per image: one CLS token (global representation) and 196 patch tokens (14×14 spatial grid, each representing a local image region). The Phase 1 baseline used only the CLS token, discarding **99.5% of the encoder's output**. Phase 2A investigates whether the spatial information encoded in the patch tokens provides complementary discriminative signal for action detection.

The hypothesis is grounded in recent literature:

- **Jose et al. (CVPR 2025, arXiv:2412.16334)** demonstrated that concatenating DINOv2's CLS token with the mean of patch tokens improves both global classification (+1.8% on ImageNet) and dense prediction (+3.2% on ADE20K), suggesting that the CLS and patch representations encode partially non-redundant information.
- **COMM (Jiang et al., arXiv:2310.08825)** showed that DINOv2's shallow layers preserve low-level spatial detail while deep layers encode high-level semantics, motivating the use of final-layer patch tokens as a source of complementary fine-grained visual information.
- **Talk2DINO (Barsellotti et al., arXiv:2411.19331)** demonstrated that DINOv2 attention maps provide meaningful spatial alignment with semantic concepts, motivating attention-weighted pooling mechanisms.

### 4.2 Feature Extraction

Three DINOv3 feature variants were extracted:

**CLS-only (baseline)**: `last_hidden_state[:, 0, :]` → [1, 1024]. The standard global representation.

**Mean-patch**: `last_hidden_state[:, 1:, :].mean(dim=1)` → [1, 1024]. The spatial average of all 196 patch tokens, capturing a uniform summary of local image regions without the CLS token's global aggregation.

**Combined (CLS ‖ mean_patch)**: concatenation of the CLS token and the mean-patch vector → [1, 2048]. This preserves both the global semantic representation and the spatially-averaged local information.

The extraction was performed using the modified `dinov3_feature_extractor.py`, which was extended with a `--pooling` CLI argument accepting `{cls, mean_patch, combined}`. The `combined` pooling mode was implemented as:

```python
elif pooling == "combined":
    h = outputs.last_hidden_state           # (1, 197, 1024)
    cls_tok = h[:, 0, :]                    # (1, 1024)
    mean_p = h[:, 1:, :].mean(dim=1)        # (1, 1024)
    feat = torch.cat([cls_tok, mean_p], dim=-1)  # (1, 2048)
```

All features were extracted at the same temporal resolution as the CLS baseline (~1.5fps, window_size=16 average pooling), ensuring direct comparability. Extraction was parallelised across 4–8 GPU nodes using the `--shard_id` / `--num_shards` mechanism, with atomic saves and skip_existing for fault-tolerant resume.

### 4.3 Training Configurations

| Experiment | Feature | in_feat_dim | Dropout | Drop-path | Weight decay | Patience |
|------------|---------|:-----------:|:-------:|:---------:|:------------:|:--------:|
| 2A.0 (baseline) | CLS | 1024 | 0.05 | 0.05 | 0.05 | 15 |
| 2A.1 | mean_patch | 1024 | 0.05 | 0.05 | 0.05 | 15 |
| 2A.2 | combined | 2048 | 0.1 | 0.1 | 0.05 | 15 |

Configuration 2A.2 uses elevated regularisation (drop=0.1, dp=0.1) to account for the doubled input dimensionality (2048 vs 1024), following the same rationale established in Phase 1 for DINOv3 regularisation.

### 4.4 Results

| Experiment | Best epoch | Full-val mAP | Sampled-val mAP | Stop epoch | Δ vs CLS baseline |
|------------|:----------:|:------------:|:----------------:|:----------:|:---------:|
| 2A.0 — CLS (baseline) | 13 | **25.21** | 25.82 | 28 | — |
| 2A.1 — mean_patch | — | < 25.21 | — | — | negative |
| 2A.2 — combined | 15 | 25.20 | 25.87 | 30 | **−0.01** |

### 4.5 Detailed Analysis of the Combined (2A.2) Training Trajectory

The combined feature training trajectory provides insight into the dynamics of the 2048-dimensional input:

| Epoch | Train mAP | Val mAP | Train–Val gap |
|------:|----------:|--------:|--------------:|
| 0 | 1.77 | 2.39 | −0.62 |
| 5 | 7.52 | 13.96 | −6.44 |
| 10 | 22.78 | 23.95 | −1.17 |
| **15** | **32.63** | **25.20** | **7.43** |
| 20 | 44.89 | 23.95 | 20.94 |
| 25 | 57.97 | 22.77 | 35.20 |
| 30 | 70.14 | 21.44 | 48.70 |

**Observation 1 — Faster early convergence**: the combined features reach val mAP 23.95 at epoch 10, compared to epoch 13 for CLS-only. The additional spatial information from patch tokens accelerates the initial learning phase.

**Observation 2 — Identical peak performance**: despite the faster convergence, the peak validation mAP (25.20 at epoch 15) is statistically indistinguishable from the CLS baseline (25.21 at epoch 13). The patch tokens provide no additional discriminative information at peak.

**Observation 3 — Comparable overfitting trajectory**: the train–val gap at epoch 30 (48.70) is comparable to the CLS baseline's overfitting pattern, scaled by the slightly higher model capacity (2048→256 projection vs 1024→256).

**Observation 4 — Block-level analysis**: the per-block validation mAP at best epoch shows consistent improvement across blocks (Block1: 23.04, Block2: 24.21, Block3: 24.62, Final: 25.20), indicating that all three SSM blocks contribute positively. This is in contrast to the skeleton experiment where Block 3 showed degradation (see Section 5.6).

### 4.6 Interpretation — Why Dense Features Do Not Help

The null result for Phase 2A is interpretable through several complementary lenses:

**Hypothesis A — Temporal resolution is too coarse for spatial information.** At ~1.5fps (one feature per 16 frames), each feature represents an average over approximately 0.67 seconds of video. At this temporal granularity, the spatial detail encoded by patch tokens is averaged away: a hand moving from left to right across 16 frames produces a blurred spatial signal in the mean-pooled representation. The CLS token, being a global summary, is more robust to this temporal averaging than spatially-resolved patch features.

**Hypothesis B — CLS already captures sufficient information.** DINOv3's CLS token is trained via self-supervised distillation to capture the global semantics of the image. For Charades' action taxonomy — which is primarily defined by scene-level semantics and object co-occurrence — this global representation may already encode all discriminative information available in the image. The patch tokens provide spatial resolution that is not required for these particular action classes.

**Hypothesis C — The mean-pooling destroys the spatial structure.** The value of patch tokens lies in their spatial arrangement, not in their average. Mean-pooling 196 patch tokens into a single vector discards the spatial layout entirely — the resulting representation is a "bag of patches" that loses information about where objects are relative to the body. More sophisticated aggregation methods (attention pooling, spatial pyramid pooling) might preserve this structure, but the null result from combined features (which preserves the CLS global signal) suggests that the additional spatial information is genuinely redundant for this task.

### 4.7 Decision: Phase 2A Closed

Based on the null result for both mean_patch and combined features, the planned attention-weighted pooling experiment (2A.3) was **not pursued**. The rationale is conservative: if concatenating the full CLS and mean_patch representations (preserving both information streams without any lossy aggregation) produces identical performance to CLS alone, then a more sophisticated but still patch-derived aggregation is unlikely to yield meaningful improvement. The experimental resources were redirected to the more promising skeleton fusion pathway (Phase 2B).

---

## 5. Phase 2B — Skeleton Feature Integration

### 5.1 Motivation

The Phase 1 per-class analysis identified a systematic gap in both CLIP and DINOv3 representations: classes discriminated by **body kinematics** rather than visual appearance or object identity are consistently underserved. Two failure groups were identified:

**Group A — State/direction verbs** (mean CLIP AP: 13.3): actions defined by a directional trajectory or state transition. Examples: *turning on/off a light* (c104/c105: AP 8.7/2.3), *throwing a bag somewhere* (c024: AP 7.1), where the discriminative signal lies in the body motion trajectory rather than the visual scene.

**Group B — Postural transitions** (mean CLIP AP: 35.8): actions defined by the trajectory of the centre of mass or joint angular velocities. Examples: *sneezing* (c153: AP 17.8, rapid trunk flexion), *running* (c150: AP 18.9, periodic gait), *standing up* (c154: AP 36.8, CoM ascent).

These classes require **kinematic information** that is fundamentally absent from appearance-based features. Skeleton representations, which encode joint positions and their temporal evolution, are designed to capture precisely this type of information.

### 5.2 SCDNet Backbone

The skeleton features used in this work were extracted using SCDNet (Skeleton-aware Compositional Dynamic Network, Wu et al., AAAI 2024, arXiv:2309.05834), a Graph Neural Network that processes human body keypoints through spatial-temporal graph convolutions. Key properties:

- **Input**: 2D/3D joint keypoints from a pose estimator (17–25 joints per frame)
- **Processing**: Compositional decomposition of the skeleton graph, capturing both full-body dynamics and part-specific motion patterns
- **Output**: 4096-dimensional embedding per frame, approximately L2-normalised, value range [−1.0, +1.0]
- **Coverage**: 9848 files covering the complete Charades train+test set

### 5.3 Temporal Alignment

A critical preprocessing step was required before training: the skeleton features are extracted at ~24fps (one embedding per video frame), while the visual features operate at ~1.5fps (one embedding per 16-frame window). The temporal mismatch factor of approximately 16× was resolved through **non-overlapping average pooling with window_size=16**, identical to the pooling strategy used during visual feature extraction.

The alignment was validated across a systematic sample of 20 videos, confirming a consistent ratio of 15.67–16.00 between raw skeleton frame count and CLIP window count. After alignment, the skeleton features in `charades_scdnet_w16/` have shape `[N, 4096]` with N identical to the corresponding CLIP feature count for each video.

**One anomalous video** was detected: `5UNDJ` (22 skeleton windows vs 292 CLIP windows), attributed to SCDNet pose estimation failure on the source video. This video is handled via zero-tensor fallback in the fusion dataloader.

The extraction pipeline (`extract_scdnet_features.py`) reads directly from the 93GB zip archive without full decompression, applies the window pooling, verifies alignment against CLIP features, and saves atomically with resume support. Throughput: ~8.4 videos/second, completing the full dataset in approximately 20 minutes.

### 5.4 Codebase Modifications

Three modifications were required to support skeleton features in MS-Temba:

**Modification 1 — Backbone routing** (`MSTemba_main.py`): Added `scdnet → 4096` to the backbone-to-dimension mapping, alongside an explicit `-in_feat_dim 4096` CLI override.

**Modification 2 — Auto-transpose whitelist** (`charades_dataloader.py`): The dataloader's feature orientation heuristic uses a hardcoded whitelist of known feature dimensions `(256, 512, 768, 1024, 2048)` to determine whether a loaded array requires transposition. Without modification, skeleton features with D=4096 would be **silently transposed incorrectly** for any video with exactly 256, 512, 768, or 1024 temporal windows. The fix adds `4096` to the whitelist. This bug would have affected approximately 2–5% of videos, producing corrupted inputs that would degrade mAP without any visible error message.

**Modification 3 — Training script** (`run_charades_scdnet_seed0.sh`): New script with elevated regularisation (drop=0.1, dp=0.1, wd=0.05, patience=15) reflecting the higher input dimensionality.

### 5.5 Training Configuration

| Parameter | Skeleton (2B.1) | CLIP baseline | Comparison |
|-----------|:---------------:|:-------------:|:----------:|
| Backbone | scdnet | clip | — |
| in_feat_dim | 4096 | 768 | 5.3× larger |
| Projection params | 1,049K | 197K | 5.3× more |
| Dropout | 0.1 | 0.0 | Regularised |
| Drop-path | 0.1 | 0.0 | Regularised |
| Weight decay | 0.05 | 0.01 | 5× stronger |
| Patience | 15 | 50 (none) | Early stopping |
| All other params | identical | — | Controlled |

### 5.6 Results

| Metric | Skeleton (2B.1) | CLIP baseline | DINOv3 CLS reg_v2 |
|--------|:---------------:|:-------------:|:------------------:|
| **Best val mAP** | **9.46** | 32.40 | 25.21 |
| Sampled val mAP | 9.73 | 33.43 | 25.82 |
| Best epoch | 21 | 13 | 13 |
| Early stop epoch | 36 | 50 (none) | 28 |
| Train mAP at best | 13.23 | 40.43 | — |
| Train mAP at stop | 52.45 | 99.52 | — |
| Train–val gap (best) | 3.77 | 8.03 | — |
| Train–val gap (stop) | 45.19 | 70.79 | — |

The skeleton single-stream result (9.46 mAP) falls into the lowest of the three pre-experiment prediction scenarios (Scenario C: mAP < 15), indicating that the 4096-dimensional SCDNet embeddings have limited standalone discriminative power for the full Charades taxonomy within the MS-Temba framework.

### 5.7 Training Dynamics

The skeleton training trajectory reveals four distinct phases:

**Phase I — Extended warmup (epochs 0–9)**: Val mAP rises from 2.36 to 8.51 over 10 epochs, substantially slower than CLIP (which reaches its plateau by epoch 5). The slower convergence reflects the difficulty of learning a useful 256-dimensional representation from the 4096-dimensional input through a single linear projection.

**Phase II — Narrow plateau (epochs 10–21)**: Val mAP fluctuates within a 1-point band (8.51–9.46) across 12 epochs, while train mAP doubles from 6.74 to 13.23. The model has exhausted the generalisable information in the skeleton features and spends the plateau memorising training-specific patterns.

**Phase III — Post-plateau overfitting (epochs 22–36)**: Train mAP accelerates from 15.23 to 52.45 while val mAP monotonically declines from 9.13 to 7.26. The train–val gap widens from 6.10 to 45.19 points.

**Phase IV — Early stopping (epoch 36)**: Training halted after 15 consecutive epochs without improvement. Best checkpoint at epoch 21.

**Block-level analysis** at the best epoch provides additional diagnostic information:

| Block | Train mAP | Val mAP | Sampled val mAP |
|-------|----------:|--------:|:---------------:|
| Block 1 (256-dim) | 10.87 | 8.58 | 8.88 |
| Block 2 (384-dim) | 11.99 | **9.07** | 9.28 |
| Block 3 (576-dim) | 12.80 | 8.92 | 9.23 |
| Final (interaction) | 13.23 | **9.46** | 9.73 |

Block 2 achieves the highest per-block validation mAP (9.07), while Block 3 already shows slight degradation (8.92). This pattern — where additional model depth improves training but not validation — indicates that the bottleneck is upstream at the input projection, not in the temporal modelling capacity of the SSM blocks.

### 5.8 Per-Class Analysis

The per-class AP distribution at the best epoch is heavily right-skewed:

| Statistic | Value |
|-----------|:-----:|
| Mean AP | 9.73 |
| Median AP | 6.35 |
| Standard deviation | 9.91 |
| Classes with AP > 20 | 24 (15.3%) |
| Classes with AP > 10 | 53 (33.8%) |
| Classes with AP < 2 | 26 (16.6%) |

#### 5.8.1 Skeleton Strengths — Postural and Kinematic Classes

The top-performing skeleton classes share a common characteristic: they are defined by **whole-body posture, gross motor patterns, or sustained body configuration** that is directly encoded in the skeleton topology.

| Rank | Class | Skel AP | CLIP AP | Δ (Skel−CLIP) | Kinematic cue |
|-----:|-------|:-------:|:------:|:--------:|---|
| 1 | c151: Closing a closet | 52.3 | 23.5 | **+28.8** | Arm extension + forward lean |
| 2 | c059: Drinking | 51.8 | 46.0 | +5.8 | Hand-to-mouth trajectory |
| 3 | c011: Sitting on a bed | 43.0 | 36.7 | +6.3 | Seated posture with low CoM |
| 4 | c123: Walking | 35.8 | 18.7 | **+17.1** | Periodic bilateral gait cycle |
| 5 | c154: Sitting down | 31.6 | 30.5 | +1.1 | Descending CoM trajectory |
| 6 | c097: Watching TV | 30.5 | 30.5 | 0.0 | Static seated, forward-facing |
| 7 | c061: Eating something | 26.8 | 21.7 | +5.1 | Repetitive hand-to-mouth arc |
| 8 | c124: Running somewhere | 22.5 | 18.9 | +3.6 | High-frequency gait pattern |
| 9 | c133: Undressing | 21.9 | 18.5 | +3.4 | Bilateral arm-cross motion |
| 10 | c118: Holding groceries | 15.9 | 7.1 | **+8.8** | Bilateral arm-load posture |

The most striking skeleton advantages occur on c151 (*closing a closet*: +28.8), c123 (*walking*: +17.1), and c118 (*holding groceries*: +8.8) — all defined by distinctive full-body configurations that the skeleton captures with high fidelity but that CLIP's semantic features underspecify.

#### 5.8.2 Skeleton Weaknesses — Object-Defined and Fine-Grained Classes

Three systematic failure modes account for the 26 classes with AP < 2:

**Failure mode 1 — Object-discriminated throwing classes**: c045 (*throwing a book*: 0.17), c085 (*throwing clothes*: 0.26), c064 (*throwing a pillow*: 0.64), c031 (*throwing a bag*: 0.73). All share an identical ballistic arm trajectory; the discriminative signal lies entirely in the **thrown object**, which is invisible to skeleton features. This is the most fundamental limitation of skeleton-only approaches.

**Failure mode 2 — Fine-grained manipulation**: c060 (*opening a box*: 0.37), c039 (*opening a window*: 0.78), c066 (*putting a picture*: 0.96). These actions require distinguishing hand-object interaction geometries that are below the spatial resolution of the 17–25 joint skeleton topology.

**Failure mode 3 — Minimal body motion**: c138 (*turning on a TV*: 0.91), c095 (*playing with phone*: 0.97), c103 (*working at a table*: 1.06). The body remains essentially stationary; the discriminative signal lies in the device being manipulated. The skeleton embedding for these actions is indistinguishable from "sitting still".

#### 5.8.3 Complementarity Quantification

| Metric | Count | Percentage |
|--------|------:|:----------:|
| Classes where Skeleton AP > CLIP AP | 18 | 11.5% |
| Classes where Skeleton AP > CLIP AP by ≥5 points | 8 | 5.1% |
| Classes where CLIP AP > Skeleton AP | 139 | 88.5% |
| Classes where CLIP AP > Skeleton AP by ≥20 points | 42 | 26.8% |

The complementarity pattern is **highly asymmetric**: CLIP dominates on the vast majority of classes, but skeleton features provide genuine, non-redundant information on a targeted subset. This asymmetry has direct implications for fusion architecture design: the gate must be initialised to favour CLIP and should only up-weight skeleton when kinematic information is specifically needed.

### 5.9 Diagnosis — Root Cause of Low Standalone Performance

The primary bottleneck is the **aggressive input projection compression** (4096→256, a 16× factor):

| Backbone | Input dim | Internal dim | Compression ratio | Projection params | Val mAP |
|----------|:---------:|:------------:|:-----------------:|:-----------------:|:-------:|
| CLIP | 768 | 256 | 3.0× | 197K | 32.40 |
| DINOv3 | 1024 | 256 | 4.0× | 262K | 25.21 |
| SCDNet | 4096 | 256 | **16.0×** | 1,049K | 9.46 |

Evidence supporting the compression bottleneck diagnosis:

1. **Train mAP reaches 52.45**: the model *can* extract discriminative information from skeleton features when overfitting is permitted — the information is present in the features but the generalisation pathway is too narrow.
2. **Val mAP plateaus at ~9.3–9.5 despite continued training improvement**: a fixed generalisation bottleneck rather than a data quality issue.
3. **Block 2 outperforms Block 3 on validation**: additional model depth does not compensate for information lost at the input stage.

However, the compression ratio alone does not fully explain the result. A secondary factor is the **inherent information content** of the skeleton features relative to the task: the 157-class Charades taxonomy is primarily visual/semantic, with only ~15% of classes having strong kinematic discriminability. The skeleton features are well-suited for this 15% but carry limited signal for the remaining 85%.

---

## 6. Cross-Phase Comparative Analysis

### 6.1 Unified Results Table

| Experiment | Phase | Backbone | in_feat_dim | Best mAP | Δ vs CLIP | Key finding |
|------------|-------|----------|:-----------:|:--------:|:---------:|-------------|
| CLIP orig ⭐ | 1 | CLIP ViT-B/16 | 768 | **32.40** | — | Definitive baseline |
| CLIP reg_v1 | 1 | CLIP ViT-B/16 | 768 | 29.17 | −3.23 | Regularisation hurts |
| CLIP reg_v2 | 1 | CLIP ViT-B/16 | 768 | 28.91 | −3.49 | Regularisation hurts |
| DINOv3 CLS reg_v2 ⭐ | 1 | DINOv3 ViT-L/16 | 1024 | **25.21** | −7.19 | Best DINOv3 config |
| DINOv3 mean_patch | 2A | DINOv3 ViT-L/16 | 1024 | <25.21 | <−7.19 | Patch-only worse |
| DINOv3 combined | 2A | DINOv3 ViT-L/16 | 2048 | 25.20 | −7.20 | Patches add nothing |
| SCDNet skeleton | 2B | SCDNet GNN | 4096 | 9.46 | −22.94 | Complementary signal |

### 6.2 Overfitting Comparison

| Experiment | Best epoch | Train mAP (best) | Val mAP (best) | Gap (best) | Train mAP (stop) | Gap (stop) |
|------------|:----------:|:----------------:|:--------------:|:----------:|:----------------:|:----------:|
| CLIP orig | 13 | 40.4 | 32.4 | 8.0 | 99.5 (ep49) | 70.8 |
| DINOv3 CLS reg_v2 | 13 | — | 25.2 | — | — | — |
| DINOv3 combined | 15 | 32.6 | 25.2 | 7.4 | 70.1 (ep30) | 48.7 |
| SCDNet skeleton | 21 | 13.2 | 9.5 | 3.8 | 52.5 (ep36) | 45.2 |

A consistent pattern emerges: all experiments exhibit severe overfitting, with the train–val gap exceeding 45 points by the time of early stopping or training completion. This suggests that the Charades training set (7985 videos) is fundamentally small for an 18M-parameter model, regardless of input feature type. The key differentiator is the **peak validation mAP**, which is determined by feature quality rather than model capacity.

### 6.3 Information Content Hierarchy

The experimental results establish a clear hierarchy of feature informativeness for Charades:

```
CLIP (768-dim, 32.40 mAP)
  >> DINOv3 CLS (1024-dim, 25.21 mAP)
    = DINOv3 CLS + patch (2048-dim, 25.20 mAP)
      >> SCDNet skeleton (4096-dim, 9.46 mAP)
```

This hierarchy reflects the alignment between each feature space and the Charades task:

1. **CLIP** features are pre-trained with language supervision that directly encodes action-relevant semantic concepts. The Charades class labels *are* natural language descriptions, giving CLIP a structural advantage.
2. **DINOv3** features are self-supervised on visual properties (texture, shape, spatial layout). These capture fine-grained visual details but lack the semantic grounding that CLIP provides.
3. **SCDNet** features encode kinematic dynamics (joint trajectories, body posture, motion patterns). These are highly informative for the ~15% of classes defined by body motion but uninformative for the ~85% of classes defined by visual appearance or object identity.

Critically, this hierarchy applies only to **standalone** performance. The per-class complementarity analysis (Section 5.8.3) demonstrates that the hierarchy does not hold uniformly across classes — skeleton features outperform CLIP on 18 classes, providing non-redundant information that cannot be recovered from visual features alone.

---

## 7. Synthesis and Key Findings

### 7.1 Principal Findings

**PF1 — CLIP is the dominant single-stream backbone for Charades within MS-Temba.** At 32.40 mAP, CLIP outperforms DINOv3 by 7.19 points and skeleton by 22.94 points. The language-supervised pre-training provides a structural advantage for a dataset whose class taxonomy is defined by natural language descriptions.

**PF2 — DINOv3 patch tokens do not provide additional discriminative information beyond the CLS token at the temporal resolution used.** Both mean-patch (2A.1) and CLS+mean_patch (2A.2) fail to improve over CLS-only, with the combined configuration producing statistically identical performance (25.20 vs 25.21 mAP). This null result eliminates the dense DINOv3 feature pathway and redirects resources to skeleton fusion.

**PF3 — Skeleton features have low standalone discriminative power but exhibit targeted complementarity with CLIP.** The 9.46 mAP standalone result masks a highly non-uniform per-class profile: 24 classes exceed 20 AP (including 52.3 for *closing a closet* and 51.8 for *drinking*), while 26 classes fall below 2 AP. The 8 classes where skeleton exceeds CLIP by ≥5 AP are precisely those defined by gross body kinematics.

**PF4 — The complementarity between CLIP and skeleton features is asymmetric and class-dependent.** CLIP dominates 88.5% of classes; skeleton wins on 11.5%. This asymmetry necessitates a fusion architecture with a visual-biased default that selectively incorporates skeleton information when kinematic cues are relevant.

**PF5 — Overfitting is a universal challenge on Charades regardless of feature type.** All configurations show train–val gaps exceeding 45 points by the end of training. The 7985-video training set is insufficient for the 18M-parameter MS-Temba model. Early stopping is essential, and any fusion architecture that increases parameters must be designed with strong regularisation.

**PF6 — The skeleton input projection bottleneck (4096→256, 16× compression) exacerbates the standalone performance deficit.** The block-level analysis shows that deeper SSM blocks do not improve validation performance for skeleton features, indicating that the information loss occurs at the input stage rather than in the temporal modelling.

### 7.2 Implications for Fusion Design

The findings impose several constraints on the fusion architecture:

1. **Visual-biased initialisation**: the gate or attention mechanism must default to CLIP dominance (~62–70% weight) and learn to selectively up-weight skeleton only where kinematic information is discriminative.

2. **Independent projections**: CLIP and skeleton features must be projected independently to the shared 256-dimensional space, rather than concatenated before projection. Concatenation (4864-dim input) would exacerbate the compression problem and create a dimensionality imbalance where skeleton features occupy 84% of the input but contribute useful information for only 15% of classes.

3. **Visual projection warm-starting**: the CLIP projection can be initialised from the trained CLIP baseline checkpoint, preserving the well-learned visual representation while the skeleton projection and gate are trained from scratch.

4. **Differential learning rates**: the skeleton projection should use a reduced learning rate (0.2× base) to prevent gradient instability from the high-dimensional input.

5. **Anomalous video handling**: a binary mask mechanism is required to handle videos with missing or truncated skeleton features (e.g., 5UNDJ), forcing the gate to 1.0 (full visual weight) for these cases.

---

## 8. Future Work — Fusion Architecture and Roadmap

### 8.1 Planned Experiments

The fusion experiments are ordered by complexity and expected information gain:

#### 8.1.1 Score-Level Fusion (2B.2) — Immediate

Average the logits of independently trained CLIP and skeleton models at inference time:

```python
logits_fused = α · logits_clip + (1 − α) · logits_skel
# Sweep α ∈ {0.7, 0.8, 0.85, 0.9, 0.95}
```

**Zero code changes required.** Both checkpoints already exist. This experiment validates complementarity and establishes a fusion lower bound. Expected gain: +0.3 to +0.6 mAP vs CLIP.

#### 8.1.2 Gated Dual Projector (2B.5) — Primary Experiment

The recommended fusion architecture:

```python
class GatedDualProjection(nn.Module):
    def __init__(self, vis_dim, skel_dim, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_vis  = nn.Sequential(
            nn.Linear(vis_dim, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(drop_rate)
        )
        self.proj_skel = nn.Sequential(
            nn.Linear(skel_dim, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(drop_rate)
        )
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.Sigmoid()
        )

    def forward(self, f_vis, f_skel, skel_mask=None):
        h_v = self.proj_vis(f_vis)
        h_s = self.proj_skel(f_skel)
        g   = self.gate(torch.cat([h_v, h_s], dim=-1))
        if skel_mask is not None:
            g = g.masked_fill(skel_mask.unsqueeze(-1).unsqueeze(-1), 1.0)
        return g * h_v + (1.0 - g) * h_s
```

The gate learns per-frame, per-dimension modality weights. For frames where visual appearance dominates (cooking scenes, phone interactions), g → 1.0 (full CLIP); for frames where kinematics dominate (walking, throwing, sitting down), g → 0.0 (full skeleton). The parameter overhead is approximately 131K (+0.7%). Expected gain: +1.1 to +2.6 mAP vs CLIP.

#### 8.1.3 Ablation Experiments (2B.3, 2B.4)

Two ablation experiments isolate the gating mechanism's contribution:

- **Feature concatenation (2B.3)**: `[f_CLIP ‖ f_skel]` → `Linear(4864 → 256)`. Lower bound for feature-level fusion.
- **Additive dual projector (2B.4)**: `proj_vis(f_CLIP) + proj_skel(f_skel)`. Equal-weight fusion without gating.

#### 8.1.4 Cross-Attention Fusion (2B.6/2B.7) — Conditional

Conditional on gated fusion results (proceed if ΔmAP ≥ +1.5):

- **Visual-as-Query (2B.7)**: CLIP features query skeleton features, asking "given this scene, which kinematic dimensions are relevant?" This direction is preferred given the asymmetric complementarity.
- **Skeleton-as-Query (2B.6)**: Skeleton features query CLIP features. Motivated by CLIP-MG (Xiang et al., 2025).

#### 8.1.5 CLIP + DINOv3 Fusion (2C) and Three-Stream (2D)

Despite the null result of Phase 2A for DINOv3 dense features, score-level and gated fusion between CLIP and DINOv3 CLS remain valid experiments — DINOv3 showed advantages on specific object-interaction classes in Phase 1. The three-stream fusion (CLIP + DINOv3 + skeleton) represents the ultimate experimental target.

### 8.2 Expected Impact

| Strategy | Estimated mAP | ΔmAP vs CLIP | Confidence |
|----------|:------------:|:------------:|:----------:|
| Score-level CLIP+skel (α=0.9) | 32.7–33.0 | +0.3 to +0.6 | High |
| Additive dual projector | 33.0–34.0 | +0.6 to +1.6 | Medium |
| **Gated dual projector** | **33.5–35.0** | **+1.1 to +2.6** | **Medium–High** |
| Cross-attention | 33.5–35.5 | +1.1 to +3.1 | Low–Medium |
| Three-stream gated | 35.0–39.0 | +2.6 to +6.6 | Low |

### 8.3 Literature Support for the Fusion Approach

The proposed gated fusion architecture is supported by several recent works in skeleton–visual fusion for action recognition:

- **CLIP-MG** (Xiang et al., 2025, arXiv:2506.16385): skeleton-as-Query cross-attention directs CLIP's spatial attention to body-relevant regions, achieving +16.5pp on NTU RGB+D 120.
- **SkeletonCLIP++** (Lin et al., 2024): CLIP semantic embeddings weight skeleton frames via Weighted Frame Integration, using visual context as a temporal curriculum for skeleton processing.
- **Zhu et al.** (ACM TOMM 2022, arXiv:2202.11374): two-stage fusion (early spatial attention + late cross-attention) achieves +1.5 to +3.0 mAP on multi-label benchmarks.
- **HCMFN** (Hu et al., 2024): skeleton features as spatial anchors for visual ROI pooling, achieving +2.1 mAP on fine-grained manipulation classes.

A common finding across these works is that **the fusion direction matters**: given the performance asymmetry observed in this project (32.4 vs 9.5 mAP), the visual-as-Query or visual-biased-gate approach is more appropriate than skeleton-as-Query, ensuring that the well-learned visual representations are preserved while skeleton information is selectively incorporated.

### 8.4 Implementation Requirements

The transition from single-stream to dual-stream training requires the following codebase modifications:

1. **Dataloader modification** (`charades_dataloader.py`): extend to load features from two directories simultaneously (CLIP + skeleton), returning both tensors per sample. Handle temporal dimension mismatches and missing skeleton files.
2. **Model modification** (`models_MSTemba.py`): replace the single `InputProjection` with `GatedDualProjection`, accepting two input tensors.
3. **Training script** (`MSTemba_main.py`): add CLI arguments for secondary feature path, skeleton mask threshold, gate bias initialisation, and per-module learning rate specification.

---

## 9. References

1. Pramanik et al., "MS-Temba: Multi-Scale Temporal Mamba for Temporal Action Detection," arXiv:2501.06138, 2025.
2. Wu et al., "SCDNet: Skeleton-aware Compositional Dynamic Network for Action Recognition," AAAI 2024, arXiv:2309.05834.
3. Xiang et al., "CLIP-MG: Skeleton-Guided Multi-Granularity CLIP for Action Recognition," arXiv:2506.16385, 2025.
4. Lin et al., "SkeletonCLIP++: CLIP-Enhanced Skeleton-Based Action Recognition," 2024.
5. Zhu et al., "Two-Stage Skeleton–Visual Fusion for Action Recognition," ACM TOMM 2022, arXiv:2202.11374.
6. Hu et al., "HCMFN: Hierarchical Cross-Modal Fusion Network for Action Detection," 2024.
7. Jose et al., "Revisiting DINOv2 for Dense Visual Tasks," CVPR 2025, arXiv:2412.16334.
8. Jiang et al., "COMM: Complementary Multi-Modal Features for Dense Prediction," arXiv:2310.08825, 2023.
9. Barsellotti et al., "Talk2DINO: Aligning Language with DINOv2 via Attention Maps," arXiv:2411.19331, 2024.

---

## Appendix A — Experiment Registry

| Experiment ID | Phase | Config | Best mAP | Status |
|---------------|-------|--------|:--------:|:------:|
| clip/seed0 | 1 | drop=0, dp=0, wd=0.01 | 32.40 | ✅ |
| clip/seed1 | 1 | drop=0, dp=0, wd=0.01 | ~32.1 | ✅ |
| clip/seed2 | 1 | drop=0, dp=0, wd=0.01 | ~31.8 | ✅ |
| clip_reg_v1 | 1 | drop=0.1, dp=0.05, wd=0.05 | 29.17 | ✅ |
| clip_reg_v2 | 1 | drop=0, dp=0, wd=0.05 | 28.91 | ✅ |
| dinov3/seed0 | 1 | drop=0, dp=0, wd=0.01 | 25.44 | ✅ |
| dinov3_reg_v1 | 1 | drop=0.2, dp=0.1, wd=0.05 | 24.56 | ✅ |
| dinov3_reg_v2 | 1 | drop=0.05, dp=0.05, wd=0.05 | 25.21 | ✅ |
| dinov3_meanpatch | 2A | drop=0.05, dp=0.05, wd=0.05 | <25.21 | ✅ |
| dinov3_combined | 2A | drop=0.1, dp=0.1, wd=0.05 | 25.20 | ✅ |
| scdnet/seed0 | 2B | drop=0.1, dp=0.1, wd=0.05 | 9.46 | ✅ |
| score_fusion | 2B | CLIP+skel logit avg | — | ⏳ |
| gated_fusion | 2B | CLIP+skel gated | — | ⏳ |

## Appendix B — Feature Directory Paths

| Feature | Path relative to repo root |
|---------|---------------------------|
| CLIP | `data/hf_features/Temporal_Action_Detection/charades_features_clip/` |
| DINOv3 CLS | `data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps/` |
| DINOv3 mean_patch | `data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps_meanpatch/` |
| DINOv3 combined | `data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps_combined/` |
| Skeleton SCDNet | `data/hf_features/Temporal_Action_Detection/charades_scdnet_w16/` |
| Video frames | `data/charades_frames_24fps/` |

## Appendix C — Skeleton Per-Class AP vs CLIP (Top 30 Absolute Difference)

| Rank | Class | Skeleton AP | CLIP AP | Δ (Skel−CLIP) | Winner |
|-----:|-------|:-----------:|:------:|:--------:|:------:|
| 1 | c046: Opening refrigerator | 1.6 | 60.2 | −58.6 | CLIP |
| 2 | c143: Opening refrigerator (v3) | 3.5 | 88.1 | −84.6 | CLIP |
| 3 | c117: Talking on phone | 2.6 | 75.9 | −73.3 | CLIP |
| 4 | c088: Holding a vacuum | 2.9 | 74.2 | −71.3 | CLIP |
| 5 | c137: Holding a vacuum (v2) | 8.8 | 74.2 | −65.4 | CLIP |
| 6 | c037: Cooking something | 8.6 | 79.0 | −70.4 | CLIP |
| 7 | c016: Working on laptop | 18.4 | 75.9 | −57.5 | CLIP |
| 8 | c065: Holding a broom | 9.1 | 75.4 | −66.3 | CLIP |
| 9 | c058: Closing refrigerator | 2.7 | 60.2 | −57.5 | CLIP |
| 10 | c145: Closing refrigerator (v2) | 3.5 | 60.2 | −56.7 | CLIP |
| 11 | c151: Closing a closet | **52.3** | 23.5 | **+28.8** | **Skeleton** |
| 12 | c123: Walking | **35.8** | 18.7 | **+17.1** | **Skeleton** |
| 13 | c118: Holding groceries | **15.9** | 7.1 | **+8.8** | **Skeleton** |
| 14 | c125: Closing book (v2) | **31.3** | 22.5 | **+8.8** | **Skeleton** |
| 15 | c011: Sitting on bed | **43.0** | 36.7 | **+6.3** | **Skeleton** |

This table illustrates the extreme asymmetry of the complementarity: CLIP's advantages are large in absolute terms (60–85 AP points on refrigerator, vacuum, and cooking classes) while skeleton's advantages are more modest but consistent (6–29 AP points on postural and kinematic classes). The fusion architecture must reflect this asymmetry.

---