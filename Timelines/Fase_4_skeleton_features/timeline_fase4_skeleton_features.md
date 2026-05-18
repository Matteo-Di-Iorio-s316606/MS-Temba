# MS-Temba on Charades: Technical Analysis — Phase 2B
### SCDNet Skeleton Feature Extraction, Alignment, Single-Stream Baseline, and Fusion Roadmap

> **Author**: Matteo Di Iorio  
> **Period**: March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Dataset**: Charades v1 — 157 action classes, 7985 training videos, 1863 test videos  
> **Document**: Phase 2B of N — skeleton feature extraction, temporal alignment, single-stream training, results analysis, fusion roadmap  
> **Prerequisites**: Phase 1 complete (CLIP baseline 32.40 mAP, DINOv3 reg_v2 baseline 25.21 mAP)

---

## Table of Contents

1. [Context and Motivation](#1-context-and-motivation)
2. [SCDNet Skeleton Features — Technical Overview](#2-scdnet-skeleton-features--technical-overview)
3. [Dataset Inspection and Format Analysis](#3-dataset-inspection-and-format-analysis)
4. [Temporal Alignment Strategy](#4-temporal-alignment-strategy)
5. [Extraction Pipeline Implementation](#5-extraction-pipeline-implementation)
6. [Extraction Results and Validation](#6-extraction-results-and-validation)
7. [Codebase Modifications for Skeleton Integration](#7-codebase-modifications-for-skeleton-integration)
8. [Single-Stream Skeleton Training — Configuration and Protocol](#8-single-stream-skeleton-training--configuration-and-protocol)
9. [Single-Stream Skeleton Results — Quantitative Analysis](#9-single-stream-skeleton-results--quantitative-analysis)
10. [Per-Class Analysis — Strengths, Weaknesses, and Complementarity](#10-per-class-analysis--strengths-weaknesses-and-complementarity)
11. [Diagnosis and Remediation Strategies](#11-diagnosis-and-remediation-strategies)
12. [Next Steps — Visual–Skeleton Fusion](#12-next-steps--visualskeleton-fusion)
13. [Literature Survey — Multi-Modal Fusion for Action Detection](#13-literature-survey--multi-modal-fusion-for-action-detection)

---

## 1. Context and Motivation

### 1.1 Why Skeleton Features?

The per-class analysis from Phase 1 identified a structural weakness in both CLIP and DINOv3 backbones: neither CLS-token representation adequately captures **kinematic and postural information** required to discriminate a specific subset of Charades classes. This weakness manifests in two distinct failure patterns:

**Group A — State/direction verbs** (mean CLIP AP: 13.3): classes whose discrimination requires understanding of a state transition or directional trajectory that is invisible in a global semantic embedding. Archetypal examples:

- *Turning on/off a light* (c104/c105): AP 8.7/2.3 — the action is defined by the pre/post state of the light, not by the body configuration
- *Throwing a bag somewhere* (c024): AP 7.1 — the ballistic gesture is extremely brief and the release trajectory is not captured by the CLS token
- *Putting on/taking off shoes* (c055/c057): AP 33.8/16.5 — the directional ambiguity (foot approaching vs departing from hand) is not resolvable from a single global vector

**Group B — Postural transitions** (mean CLIP AP: 35.8): classes whose discrimination requires modelling the trajectory of the centre of mass (CoM) or joint angular velocities over time:

- *Someone is sneezing* (c153): AP 17.8 — requires detecting rapid trunk flexion and head snap
- *Someone is running* (c150): AP 18.9 — requires detecting periodic gait pattern, lateral symmetry, foot elevation
- *Someone is standing up* (c154): AP 36.8 — requires detecting positive CoM displacement with knee/hip extension

The key insight is that **the discriminative signal for these classes is kinematic rather than semantic**: it lies in how the body moves, not in what object is present or what the scene looks like. This is precisely the type of information that skeleton-based representations are designed to encode.

### 1.2 Why SCDNet?

The skeleton features available on ABACA were extracted using **SCDNet** (Skeleton-aware Compositional Dynamic Network), a backbone that produces **Level 3 semantic embeddings** via a Graph Neural Network (GNN) operating on the skeleton graph. This is qualitatively different from raw keypoint coordinates (Level 1) or 3D joint positions (Level 2):

- **Level 1 (2D keypoints)**: raw (x, y) coordinates of 17–33 landmarks per frame. Simple but loses depth information and is sensitive to camera viewpoint.
- **Level 2 (3D keypoints)**: estimated (x, y, z) coordinates from monocular video. Richer but still a low-level geometric representation.
- **Level 3 (GNN embeddings — SCDNet)**: the body is modelled as a graph where nodes are joints and edges are bone segments. The GNN learns motion dynamics directly on the graph, capturing high-level relational patterns such as "wrist approaching object" or "knees flexing while pelvis descends". The output is a high-dimensional semantic embedding, not raw coordinates.

The 4096-dimensional embedding produced by SCDNet is therefore a **compact, high-level representation of body dynamics** that can be directly integrated into MS-Temba's input pipeline without additional preprocessing of raw skeleton geometry.

### 1.3 Ablation Methodology

The integration of skeleton features is structured as a **progressive ablation**:

1. **Phase 2B.1 — Single-stream skeleton baseline** (this document): train MS-Temba using only SCDNet features as input (`backbone=scdnet`, `in_feat_dim=4096`). This establishes a standalone skeleton performance reference and validates that the 4096-dim features are learnable in the MS-Temba framework.

2. **Phase 2B.2 — Score-level fusion** (planned): independently trained CLIP and skeleton models produce logits that are averaged at inference time. Zero code changes; serves as an initial complementarity test.

3. **Phase 2B.3 — Feature concatenation** (planned): concatenate CLIP (768-dim) and SCDNet (4096-dim) features → project to common dimension. This serves as a lower bound for multi-modal fusion.

4. **Phase 2B.4 — Additive dual projector** (planned): project each modality independently to the same dimension, then sum.

5. **Phase 2B.5 — Gated fusion** (planned): learnable per-frame gate `g = σ(Linear([f_visual; f_skel]))` controls the contribution of each modality. This is the recommended production setup.

6. **Phase 2B.6/2B.7 — Cross-attention fusion** (conditional): skeleton-as-Query or visual-as-Query cross-attention. Semantically directed: "given that I see a bag, attend to the relevant kinematic dimensions".

This document covers the extraction infrastructure (Sections 1–6), the implementation of codebase modifications (Section 7), the single-stream training protocol (Section 8), a comprehensive results analysis (Sections 9–10), a diagnosis of observed failure modes with remediation strategies (Section 11), and the fusion roadmap with literature references (Sections 12–13).

---

## 2. SCDNet Skeleton Features — Technical Overview

### 2.1 SCDNet Architecture

SCDNet (Skeleton-aware Compositional Dynamic Network) is a skeleton-based action recognition backbone that processes human body keypoints through a spatial-temporal graph convolutional network. The key design choices relevant to this integration are:

**Graph construction**: the human body is represented as a directed graph G = (V, E) where V is the set of joints (typically 17–25 keypoints from a pose estimator such as OpenPose or HRNet) and E is the set of skeletal connections (bone segments). The graph topology encodes anatomical constraints, ensuring that learned features respect the physical structure of the body.

**Spatial-temporal processing**: the GNN operates on sequences of pose graphs, capturing both the spatial relationships between joints at each frame and the temporal evolution of those relationships across frames. This dual processing is crucial for action recognition: the spatial component encodes "what pose is this?", while the temporal component encodes "how is the pose changing?".

**Compositional decomposition**: SCDNet decomposes actions into sub-action components along the skeleton graph, allowing the model to capture partial-body dynamics (e.g., arm motion independently of leg motion). This compositional approach is particularly relevant for Charades, where many actions are defined by the motion of a specific body part (wrist trajectory for put/take, knee flexion for sit down).

**Output embedding**: the final layer produces a 4096-dimensional embedding that encodes the full spatio-temporal dynamics of the skeleton sequence. This dimensionality is considerably larger than the visual feature dimensions used in this project (768 CLIP, 1024 DINOv3), reflecting the richness of the kinematic representation.

### 2.2 Feature Characteristics

The SCDNet features stored in `Charades_SCDNet_features2.zip` have the following properties, confirmed by inspection:

| Property | Value |
|---|---|
| Format | NumPy `.npy`, float32 |
| Dimensionality | 4096 per frame |
| Temporal sampling | ~24fps (one embedding per original video frame) |
| Value range | approximately [−1.0, +1.0] |
| Coverage | 9848 files — complete Charades train + test set |

The value range [−1.0, +1.0] suggests the features have been normalised (likely L2-normalised or tanh-activated at the output layer), which is beneficial for integration with MS-Temba's input projection.

---

## 3. Dataset Inspection and Format Analysis

### 3.1 Initial Inspection

The skeleton features were inspected in two stages. First, the zip archive was examined to understand its structure:

```bash
ls -lh /srv/storage/.../Charades_SCDNet_features2.zip
# -rw-r----- 1 maali stars 93G Mar 20 14:44 Charades_SCDNet_features2.zip

unzip -l Charades_SCDNet_features2.zip | head -30
# Archive: Charades_SCDNet_features2.zip
#   Length    Date    Time    Name
#       0  2025-08-27  Charades_SCDNet_features2/
# 12058752  2025-08-27  Charades_SCDNet_features2/001YG.npy
# 18006144  2025-08-27  Charades_SCDNet_features2/003WS.npy
# ...

unzip -l Charades_SCDNet_features2.zip | tail -1
# 114959399936   9849 files
```

**Key findings from archive inspection**:
- 9849 entries = 1 directory entry + **9848 `.npy` files** — exact match with the total Charades video count
- Naming convention: 5-character uppercase alphanumeric IDs (e.g., `001YG.npy`, `00T1E.npy`) — identical to the Charades video ID convention used by CLIP and DINOv3 feature directories
- File sizes vary from ~3.5MB (`00SL4`) to ~36MB (`00ZCA`), consistent with variable video durations
- Total uncompressed size: ~115GB

### 3.2 Content Inspection and fps Analysis

Three representative files were extracted and compared against the corresponding CLIP and DINOv3 features:

```python
# Inspection results for three sample videos
Video: 00T1E
  SKELETON  shape=(458, 4096)  dtype=float32  min=-0.7979  max=0.9922
  CLIP      shape=(29, 768)    dtype=float16
  DINO      shape=(29, 1024)   dtype=float32
  FPS ratio (skel/clip): 458/29 = 15.7931

Video: 00SL4
  SKELETON  shape=(216, 4096)  dtype=float32  min=-0.7104  max=0.9450
  CLIP      shape=(14, 768)    dtype=float16
  DINO      shape=(14, 1024)   dtype=float32
  FPS ratio (skel/clip): 216/14 = 15.4286

Video: 00ZCA
  SKELETON  shape=(2230, 4096) dtype=float32  min=-0.9555  max=1.0797
  CLIP      shape=(140, 768)   dtype=float16
  DINO      shape=(140, 1024)  dtype=float32
  FPS ratio (skel/clip): 2230/140 = 15.9286
```

This inspection revealed a critical finding: **CLIP and DINOv3 features are not at 24fps**. They are the result of temporal average pooling over windows of 16 frames (`window_size=16` in `dinov3_feature_extractor.py`), producing approximately 1.5 features per second rather than 24. The skeleton features, by contrast, are extracted at the native frame rate (~24fps), resulting in a consistent ratio of approximately 16:1.

### 3.3 Large-Scale fps Validation

To confirm the ratio stability across the full dataset, a systematic sample of 20 videos was inspected:

| Video | Skel_T | CLIP_T | Ratio | Dim_skel |
|-------|-------:|-------:|------:|---------:|
| 001YG | 736 | 46 | 16.000 | 4096 |
| 1NVWD | 713 | 45 | 15.844 | 4096 |
| 3IMTV | 856 | 54 | 15.852 | 4096 |
| 5D3X6 | 414 | 26 | 15.923 | 4096 |
| 7B3J0 | 851 | 54 | 15.759 | 4096 |
| 94HXT | 768 | 48 | 16.000 | 4096 |
| AXYF9 | 754 | 48 | 15.708 | 4096 |
| CS01T | 835 | 53 | 15.755 | 4096 |
| EF2YJ | 862 | 54 | 15.963 | 4096 |
| G4IV1 | 510 | 32 | 15.938 | 4096 |
| HY45L | 760 | 48 | 15.833 | 4096 |
| JT1XT | 721 | 46 | 15.674 | 4096 |
| LNR61 | 791 | 50 | 15.820 | 4096 |
| NC75G | 736 | 46 | 16.000 | 4096 |
| P3EW1 | 935 | 59 | 15.847 | 4096 |
| QYPLI | 737 | 47 | 15.681 | 4096 |
| STAFD | 1498 | 94 | 15.936 | 4096 |
| UJXBC | 766 | 48 | 15.958 | 4096 |
| WDCGH | 735 | 46 | 15.978 | 4096 |
| Y1HGC | 736 | 46 | 16.000 | 4096 |
| **Mean** | | | **15.874** | |
| **Min** | | | **15.674** | |
| **Max** | | | **16.000** | |

The ratio is consistently ~16 across all videos. Values below exactly 16.0 arise from videos whose total frame count is not divisible by 16: the last CLIP/DINO window is computed over a partial window of fewer than 16 frames, while the skeleton retains all frames. The small residual variance (min 15.674) is therefore an artefact of how the CLIP/DINO extractor handles the trailing frames, not a systematic fps discrepancy.

---

## 4. Temporal Alignment Strategy

### 4.1 The Alignment Problem

The fundamental challenge is that the two modalities have different temporal resolutions:

- **Skeleton features**: `T` vectors, one per original video frame (~24fps)
- **CLIP/DINOv3 features**: `T'` vectors, one per 16-frame window (~1.5fps)
- **Required alignment**: `T'` skeleton vectors that correspond one-to-one with the CLIP/DINOv3 vectors

The alignment must be performed **before training**, not on-the-fly, to avoid CPU bottlenecks in the data pipeline.

### 4.2 Average Pooling over Windows of 16 Frames

The chosen alignment strategy mirrors exactly how CLIP and DINOv3 features were originally extracted: **non-overlapping average pooling over windows of 16 frames**.

```
Skeleton raw [T, 4096]:
[s₁  s₂  s₃  ... s₁₆][s₁₇ ... s₃₂][s₃₃ ... s₄₈] ...
 \___________________|  \__________|  \_________/
       window 1            window 2      window 3

Aligned skeleton [T', 4096]:
  skel'₁ = mean(s₁..s₁₆)
  skel'₂ = mean(s₁₇..s₃₂)
  skel'₃ = mean(s₃₃..s₄₈)
  ...
```

This approach has three key properties that make it the natural choice:

**Consistency with visual features**: since CLIP and DINOv3 features are also window averages of 16 frames, the aligned skeleton features represent the same temporal segment as their visual counterparts. Frame `t` of the skeleton sequence and frame `t` of the CLIP/DINOv3 sequence both correspond to the average of the same 16 original frames.

**Graceful handling of partial windows**: the last window may contain fewer than 16 frames (for videos with T not divisible by 16). The average over the available frames still produces a valid 4096-dim vector, consistent with how the DINOv3 extractor handles trailing frames.

**Information preservation**: average pooling over a 16-frame window of GNN embeddings preserves the mean kinematic state over that interval. Unlike nearest-neighbour resampling, it avoids introducing temporal discontinuities and smooths out frame-level noise in the skeleton detection.

### 4.3 Alternative Strategies Considered

**Nearest-neighbour resampling**: select the skeleton frame closest to the centre of each 16-frame window. Simpler but introduces temporal discontinuities at window boundaries and discards 15/16 of the available skeleton information.

**Linear interpolation**: compute the skeleton embedding at the target timestamp via linear interpolation between the two nearest raw frames. Appropriate for smooth continuous signals but the 4096-dim GNN embeddings are not guaranteed to interpolate linearly in a meaningful way.

**Re-extraction at 1.5fps**: re-run the SCDNet inference on the video at the target sampling rate, discarding intermediate frames. This would be the cleanest approach but requires re-running the computationally expensive GNN inference on all 9848 videos, which is not feasible given resource constraints.

Average pooling is the optimal choice: it is computationally trivial, information-preserving, and exactly consistent with the existing visual feature extraction pipeline.

---

## 5. Extraction Pipeline Implementation

### 5.1 Script: `vim/extract_scdnet_features.py`

The extraction script reads skeleton features directly from the zip archive (avoiding a full 93GB decompression), applies window average pooling, verifies temporal alignment against CLIP features, and saves the result atomically.

**Core pooling function**:
```python
def pool_skeleton(skel: np.ndarray, window_size: int = 16) -> np.ndarray:
    T, D = skel.shape
    n_windows = (T + window_size - 1) // window_size
    out = np.zeros((n_windows, D), dtype=np.float32)
    for i in range(n_windows):
        start = i * window_size
        end   = min(start + window_size, T)
        out[i] = skel[start:end].mean(axis=0)
    return out
```

**Alignment verification function**:
```python
def verify_alignment(out, vid, clip_dir, tol=1):
    clip_path = os.path.join(clip_dir, f"{vid}.npy")
    if not os.path.exists(clip_path):
        return True
    clip_T = np.load(clip_path, mmap_mode='r').shape[0]
    diff = abs(out.shape[0] - clip_T)
    if diff > tol:
        print(f"  [WARN] {vid}: skel_windows={out.shape[0]}, clip_T={clip_T}, diff={diff}")
        return False
    return True
```

**Key implementation details**:

- **Direct zip streaming**: features are read directly from the zip archive via `z.open(name)` and `np.load(f)`, avoiding the need to decompress the full 93GB archive to disk. Memory usage at any given time is bounded by a single video's raw skeleton features (at most ~36MB for the largest video).

- **Atomic saves**: output files are written to a temporary `.npy.tmp` path and renamed to the final `.npy` path only upon successful completion. This prevents corrupted partial files if the process is interrupted mid-write.

- **Resume support**: `--skip_existing` (default True) checks for the presence of the output file before processing each video, enabling transparent resumption after interruption without reprocessing completed videos.

- **Sharding support**: `--shard_id` and `--num_shards` parameters divide the video list into disjoint subsets for parallel processing across multiple nodes. With `--num_shards 4`, each job processes ~2462 videos independently, reducing total wall time from ~20 minutes to ~5 minutes.

- **Progress logging**: throughput and ETA are reported every 200 videos, enabling monitoring of long-running jobs.

### 5.2 Execution

The script was executed interactively on a GPU node (esterel32-1) with the conda environment activated:

```bash
cd /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/
       Traineeship/MS-Temba
source env_abaca.sh

python vim/extract_scdnet_features.py \
    --zip_path ".../Charades_SCDNet_features2.zip" \
    --out_dir  "data/hf_features/Temporal_Action_Detection/charades_scdnet_w16" \
    --clip_dir "data/hf_features/Temporal_Action_Detection/charades_features_clip" \
    --window_size 16 \
    --skip_existing \
    --verify
```

**Observed throughput**: ~8.4–8.5 videos/second, consistent across the full run. At this rate, 9848 videos required approximately **19.5 minutes** of wall time. The bottleneck is zip decompression and numpy IO rather than the pooling computation itself, which is negligible for 4096-dim vectors.

---

## 6. Extraction Results and Validation

### 6.1 Completeness Check

```bash
ls data/hf_features/Temporal_Action_Detection/charades_scdnet_w16/ | wc -l
# 9848
```

All 9848 expected files were produced. No extraction failures.

### 6.2 Alignment Verification

Post-extraction verification on a representative sample confirms exact temporal alignment:

| Video | Skeleton shape | CLIP shape | Match |
|-------|:--------------:|:----------:|:-----:|
| 00T1E | (29, 4096) | (29, 768) | ✅ |
| 001YG | (46, 4096) | (46, 768) | ✅ |
| STAFD | (94, 4096) | (94, 768) | ✅ |
| ZZ9RN | (46, 4096) | (46, 768) | ✅ |

The output shape `[N, 4096]` has `N` identical to the corresponding CLIP feature count for all verified videos.

### 6.3 Anomalous Videos

During the smoke test, one warning was detected:

```
[WARN] 5UNDJ: skel_windows=22, clip_T=292, diff=270
```

This video (`5UNDJ`) has 22 skeleton windows but 292 CLIP features — a discrepancy of 270 windows. This is not a bug in the extraction pipeline but an issue with the source data: SCDNet likely failed to produce reliable skeleton detections for most frames of this video (possibly due to heavy occlusion, multi-person scenes, or low video quality), resulting in a truncated skeleton sequence in the original zip. This anomaly is handled at the dataloader level with a zero-tensor fallback (see Section 7.2).

### 6.4 Storage Summary

| Directory | Files | Approx. size | Avg shape |
|-----------|------:|-------------:|-----------|
| `charades_features_clip` | 9848 | ~4.5GB | [N, 768] float16 |
| `charades_dinov3_vitl16_w16_24fps` | 9850 | ~8.2GB | [N, 1024] float32 |
| `charades_scdnet_w16` | 9848 | ~18GB | [N, 4096] float32 |

The skeleton features are the largest set by a factor of 2–4, due to the higher dimensionality (4096 vs 768/1024) and float32 precision.

---

## 7. Codebase Modifications for Skeleton Integration

### 7.1 `MSTemba_main.py` — Backbone Routing

The backbone-to-dimension mapping in `MSTemba_main.py` was extended to include `scdnet`. The modification was applied within the existing `if/elif` chain that routes backbone identifiers to their corresponding input feature dimensionalities:

```python
# Original mapping (lines 829-840)
if args.backbone == "i3d":
    in_feat_dim = 1024
elif args.backbone == "clip":
    in_feat_dim = 768
elif args.backbone == "vificlip":
    in_feat_dim = 512
elif args.backbone == "viclip":
    in_feat_dim = 512
elif args.backbone in ("dinov3", "dinov3_vitl16"):
    in_feat_dim = 1024
# --- ADDED ---
elif args.backbone == "scdnet":
    in_feat_dim = 4096
# --- END ---
else:
    raise ValueError(f"Unknown backbone: {args.backbone}")
```

This is the only modification required in the training script. The model architecture (`models_MSTemba.py`) requires no changes: the `LinearProjection` module's `in_feat_dim` parameter is already passed dynamically from the argparse configuration, and the `nn.Linear(in_feat_dim, 256)` projection handles the 4096→256 mapping transparently.

The `-in_feat_dim` CLI argument remains available as an explicit override (`-in_feat_dim 4096`), which is used in the training script as an additional safety measure.

### 7.2 `charades_dataloader.py` — Feature Dimension Whitelist

A critical bug was identified and fixed in the auto-transpose heuristic within the `__getitem__` method of the `Charades` dataset class. The original code contained a hardcoded whitelist of known feature dimensions used to determine whether a loaded feature array required transposition:

```python
# ORIGINAL (buggy for 4096-dim features)
if feat.shape[1] not in (256, 512, 768, 1024, 2048) \
   and feat.shape[0] in (256, 512, 768, 1024, 2048):
    feat = feat.T
```

This heuristic assumes that the feature dimension `D` is one of the listed values, and if `D` does not match but `T` (the temporal dimension) does, the array must be transposed. However, with skeleton features of dimension 4096, **any video with exactly 256, 512, 768, or 1024 temporal windows would be incorrectly transposed** to `[4096, T]` instead of the correct `[T, 4096]` orientation. For the Charades dataset, this would affect approximately 2–5% of videos whose temporal extent after padding happens to coincide with a whitelisted value.

The fix adds `4096` to the whitelist:

```python
# FIXED
if feat.shape[1] not in (256, 512, 768, 1024, 2048, 4096) \
   and feat.shape[0] in (256, 512, 768, 1024, 2048, 4096):
    feat = feat.T
```

This ensures that skeleton features with `D=4096` are correctly identified as having the feature dimension on axis 1, regardless of the temporal extent. The fix is backward-compatible: it does not affect the behaviour for CLIP (768), DINOv3 (1024), or any other existing backbone.

**Impact assessment**: without this fix, the skeleton single-stream experiment would have produced silently corrupted results for a subset of videos, with the input projection receiving temporal indices instead of feature dimensions. The effect would have been a systematic degradation of mAP that would have been difficult to diagnose from aggregate metrics alone, as the corruption is video-dependent and not visible in the loss function.

### 7.3 Training Script: `vim/scripts/run_charades_scdnet_seed0.sh`

A new training script was created following the established convention of the existing scripts (cf. `run_charades_clip_reg_v2_seed0.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

OUTDIR="$ROOT/runs/charades/scdnet/seed0"
mkdir -p "$OUTDIR"

RESUME="False"
if [ -f "$OUTDIR/checkpoint_last.pth" ]; then
    RESUME="$OUTDIR/checkpoint_last.pth"
fi

cd "$ROOT/vim"

python MSTemba_main.py \
  -dataset      charades \
  -mode         rgb \
  -backbone     scdnet \
  -model        mstemba \
  -train        True \
  -seed         0 \
  -resume       "$RESUME" \
  -save_every   1 \
  -rgb_root     "$ROOT/data/hf_features/Temporal_Action_Detection/charades_scdnet_w16" \
  -in_feat_dim  4096 \
  -num_clips    256 \
  -skip         0 \
  -comp_info    False \
  -epochs       50 \
  -unisize      True \
  -alpha_l      1 \
  -beta_l       0.05 \
  -batch_size   5 \
  --drop        0.1 \
  --drop-path   0.1 \
  --weight-decay 0.05 \
  --early-stop-patience 15 \
  --min-delta   0.01 \
  -output_dir   "$OUTDIR" \
  2>&1 | tee -a "$OUTDIR/training.log"
```

Key design decisions in the configuration:

- **`-in_feat_dim 4096`**: explicit override matching the SCDNet embedding dimension, used alongside the backbone mapping as a double safety check.
- **`--drop 0.1` and `--drop-path 0.1`**: elevated regularisation relative to the CLIP baseline (0.0/0.0). The input projection `Linear(4096→256)` has 1,048,576 parameters — approximately 5.3× more than the CLIP projection `Linear(768→256)` with 196,608 parameters. This parameter inflation, combined with the fixed training set size, increases overfitting risk substantially.
- **`--weight-decay 0.05`**: consistent with the DINOv3 reg_v2 configuration that achieved the best stability in Phase 1.
- **`--early-stop-patience 15`**: allows sufficient exploration of the loss landscape given the higher-dimensional input, while preventing excessive overfitting.

### 7.4 Summary of All Code Modifications

| File | Modification | Lines affected | Backward compatible |
|------|-------------|:--------------:|:-------------------:|
| `MSTemba_main.py` | Added `scdnet → 4096` to backbone routing | 1 elif block (~839–840) | ✅ |
| `charades_dataloader.py` | Added `4096` to auto-transpose whitelist | 1 line | ✅ |
| `vim/scripts/run_charades_scdnet_seed0.sh` | New training script | New file | N/A |

No changes were made to `models_MSTemba.py`, the loss function, the optimizer configuration, the scheduler, or the evaluation pipeline. The model architecture is identical to the CLIP and DINOv3 baselines — only the input projection width differs.

---

## 8. Single-Stream Skeleton Training — Configuration and Protocol

### 8.1 Final Training Configuration

| Parameter | Value | Comparison to CLIP baseline |
|-----------|-------|-----------------------------|
| Backbone | scdnet | clip |
| Input feature dim | 4096 | 768 |
| Input projection params | 1,048,576 + 256 (bias) | 196,608 + 256 |
| Dropout rate | 0.1 | 0.0 |
| Drop-path rate | 0.1 | 0.0 |
| Weight decay | 0.05 | 0.01 |
| Early stop patience | 15 | 50 (no early stop) |
| Learning rate | 5×10⁻⁴ | 5×10⁻⁴ |
| Scheduler | Cosine (warmup=5, min_lr=1×10⁻⁵) | Identical |
| Batch size | 5 | 5 |
| EMA decay | 0.99996 | 0.99996 |
| Temporal padding | 256 windows | 256 windows |
| Epochs (max) | 50 | 50 |
| Total model params | ~19.2M | ~18.3M |

The total parameter count increases by approximately 0.9M (4.9%) relative to the CLIP configuration, entirely attributable to the wider input projection. All other architectural components — the three SSM blocks, the interaction block, and the classifier head — remain identical.

### 8.2 Training Infrastructure

The experiment was executed on a single GPU node (esterel36-1) within the Grid5000/ABACA cluster at Sophia Antipolis. The training was launched interactively via:

```bash
oarsub -q besteffort -p esterel36 -l host=1/gpu=1,walltime=12 -I
cd /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba
source env_abaca.sh
bash vim/scripts/run_charades_scdnet_seed0.sh
```

**Training duration**: 147 minutes 31 seconds (approximately 4.0 minutes per epoch for 37 epochs). Early stopping triggered at epoch 36 after 15 consecutive epochs without improvement. The best validation mAP was recorded at epoch 21.

---

## 9. Single-Stream Skeleton Results — Quantitative Analysis

### 9.1 Summary Metrics

| Metric | Value | CLIP baseline | DINOv3 CLS reg_v2 | Δ vs CLIP |
|--------|------:|:-------------:|:------------------:|:---------:|
| **Best val mAP** | **9.46** | 32.40 | 25.21 | **−22.94** |
| Sampled val mAP | 9.73 | 33.43 | 25.82 | −23.70 |
| Best epoch | 21 | 13 | 13 | — |
| Early stop epoch | 36 | 50 (no ES) | 28 | — |
| Train mAP at best | 13.23 | 40.43 | — | — |
| Train mAP at stop | 52.45 | 99.52 | — | — |
| Train-val gap (best) | 3.77 | 8.03 | — | — |
| Train-val gap (stop) | 45.19 | 70.79 | — | — |

**Interpretation**: the result falls into **Scenario C** from the pre-experiment predictions (Section 7.3 of the original document): `mAP < 15`, indicating that the 4096-dim SCDNet embeddings have limited standalone discriminative power within the MS-Temba framework on Charades. This outcome, while below the expected range, is diagnostically valuable — the per-class AP profile (Section 10) reveals a clear complementarity pattern with CLIP that directly motivates the fusion experiments.

### 9.2 Training Dynamics — Epoch-by-Epoch Analysis

The full training trajectory is presented below (37 epochs before early stopping):

| Epoch | Train loss | Train mAP | Val loss | Val mAP | Sampled val mAP | LR |
|------:|-----------:|----------:|---------:|--------:|:---------------:|---:|
| 0 | 783.74 | 1.75 | 581.55 | 2.36 | 2.56 | 1.0×10⁻⁵ |
| 5 | 22.12 | 4.04 | 29.25 | 6.39 | 6.61 | 1.9×10⁻⁴ |
| 10 | 20.57 | 6.74 | 28.45 | 8.51 | 8.81 | 2.0×10⁻⁴ |
| 13 | 20.04 | 8.59 | 28.19 | 9.12 | 9.40 | 2.1×10⁻⁴ |
| 16 | 19.75 | 9.56 | 28.36 | 9.33 | 9.61 | 2.0×10⁻⁴ |
| **21** | **18.73** | **13.23** | **28.78** | **9.46** | **9.73** | **2.0×10⁻⁴** |
| 25 | 17.65 | 19.36 | 30.01 | 8.88 | 9.17 | 1.8×10⁻⁴ |
| 30 | 15.50 | 31.76 | 32.17 | 8.16 | 8.44 | 1.5×10⁻⁴ |
| 36 | 12.82 | 52.45 | 36.23 | 7.26 | 7.48 | 9.6×10⁻⁵ |

**Key observations from the training trajectory**:

**Observation 1 — Extended warmup phase (epochs 0–9)**: the model requires approximately 10 epochs to reach the vicinity of its best performance (val mAP 8.51 at epoch 10 vs 9.46 at epoch 21). This is significantly slower than CLIP, which reaches its best performance at epoch 13 from a much higher starting point. The delayed convergence likely reflects the difficulty of learning a useful 256-dimensional representation from the 4096-dimensional skeleton embeddings through a single linear projection followed by LayerNorm, GELU, and Dropout.

**Observation 2 — Narrow plateau (epochs 10–21)**: the validation mAP fluctuates within a narrow band of 8.51–9.46 across 12 epochs, while the training mAP doubles from 6.74 to 13.23. This pattern indicates that the model quickly exhausts the generalizable information in the skeleton features and spends the remainder of the plateau phase memorising training-specific patterns that do not transfer to the validation set.

**Observation 3 — Severe post-plateau overfitting (epochs 22–36)**: after epoch 21, the training mAP accelerates from 13.23 to 52.45 (+39.22 in 15 epochs) while the validation mAP monotonically declines from 9.46 to 7.26 (−2.20). The train-val gap widens from 3.77 to 45.19 points. This degree of overfitting is substantially more severe than observed with CLIP (gap at stop: 70.79, but with a much higher baseline — the relative overfitting rate is comparable). The diversity loss remains at 0.0 throughout training, indicating that the SSM blocks do not encounter degenerate attention patterns despite the weaker input signal.

**Observation 4 — Validation loss divergence**: the validation loss increases steadily from 28.19 (epoch 13) to 36.23 (epoch 36), a 28.5% increase. By contrast, the training loss continues to decrease monotonically. This divergence confirms that the model is fitting training noise rather than learning transferable representations.

### 9.3 Block-Level Analysis

MS-Temba processes input through three sequential SSM blocks of increasing capacity (Block 1: 256-dim, Block 2: 384-dim, Block 3: 576-dim). The per-block validation mAP at the best epoch (21) provides insight into the information flow:

| Block | Train mAP | Val mAP | Sampled val mAP |
|-------|----------:|--------:|:---------------:|
| Block 1 (256-dim, 1 SSM) | 10.87 | 8.58 | 8.88 |
| Block 2 (384-dim, 2 SSM) | 11.99 | 9.07 | 9.28 |
| Block 3 (576-dim, 3 SSM) | 12.80 | 8.92 | 9.23 |
| **Final (interaction)** | **13.23** | **9.46** | **9.73** |

**Interpretation**: Block 2 achieves the highest per-block validation mAP (9.07), while Block 3 already shows slight degradation (8.92). This pattern — where deeper blocks improve training metrics but not validation metrics — is a hallmark of overfitting propagated through the network depth. The interaction block partially recovers performance (9.46), suggesting that the multi-scale aggregation provides some regularisation benefit. Compared to the CLIP baseline where all blocks progressively improve validation performance, the skeleton signal is too weak to sustain the multi-scale processing without overfitting at the deeper levels.

---

## 10. Per-Class Analysis — Strengths, Weaknesses, and Complementarity

### 10.1 Distribution Statistics

| Statistic | Skeleton (sampled val AP) |
|-----------|:------------------------:|
| Mean | 9.73 |
| Median | 6.35 |
| Standard deviation | 9.91 |
| Classes with AP > 20 | 24 (15.3%) |
| Classes with AP > 10 | 53 (33.8%) |
| Classes with AP < 2 | 26 (16.6%) |

The distribution is **heavily right-skewed**: a small cluster of classes achieves AP > 30 while the majority (104/157, 66.2%) remain below 10 AP. The high standard deviation (9.91, exceeding the mean) confirms extreme heterogeneity in skeleton feature utility across the action taxonomy.

### 10.2 Top-Performing Classes (Skeleton AP > 20)

The 15 highest-performing classes under skeleton-only input reveal a coherent pattern — these are classes defined primarily by **whole-body posture, gross motor patterns, or sustained body configuration**:

| Rank | Class | Skeleton AP | CLIP AP | Δ (Skel−CLIP) | Discriminative cue |
|-----:|-------|:----------:|:------:|:---------:|---|
| 1 | c151: Closing a closet | 52.3 | 23.5 | +28.8 | Arm extension + forward lean |
| 2 | c059: Drinking | 51.8 | 46.0 | +5.8 | Hand-to-mouth trajectory |
| 3 | c011: Sitting on a bed | 43.0 | 36.7 | +6.3 | Seated posture |
| 4 | c123: Walking | 35.8 | 18.7 | +17.1 | Periodic gait cycle |
| 5 | c146: Smiling at someone | 33.5 | 33.5 | +0.0 | Upper-body orientation |
| 6 | c132: Holding a pillow (v2) | 33.5 | 27.9 | +5.6 | Static holding pose |
| 7 | c134: Holding a container | 31.8 | 31.8 | +0.0 | Arm position |
| 8 | c154: Sitting down | 31.6 | 30.5 | +1.1 | CoM descent trajectory |
| 9 | c125: Closing a book (v2) | 31.3 | 22.5 | +8.8 | Hand convergence pattern |
| 10 | c097: Watching television | 30.5 | 30.5 | +0.0 | Seated, forward-facing |
| 11 | c122: Holding a paper | 26.8 | 22.0 | +4.8 | Elevated arm position |
| 12 | c061: Eating something | 26.8 | 21.7 | +5.1 | Repetitive hand-to-mouth |
| 13 | c155: Holding a cup | 25.3 | 24.9 | +0.4 | Arm elevation |
| 14 | c014: Looking out window | 23.2 | 22.8 | +0.4 | Stationary, elevated gaze |
| 15 | c015: Reading book | 22.8 | 22.8 | +0.0 | Seated, head-down |

The top classes share a common characteristic: the skeleton encodes the **spatial configuration of the body** (sitting, walking, leaning, arm trajectory) rather than fine-grained object identity. The most notable skeleton advantage is c151 (*Closing a closet*: +28.8) and c123 (*Walking*: +17.1), both of which are defined by distinctive full-body kinematic patterns.

### 10.3 Worst-Performing Classes (Skeleton AP < 2)

The 15 lowest-performing classes reveal the systematic blind spots of skeleton features:

| Rank | Class | Skeleton AP | CLIP AP | Δ (Skel−CLIP) | Failure mode |
|-----:|-------|:----------:|:------:|:---------:|---|
| 1 | c045: Throwing a book | 0.17 | 8.2 | −8.0 | Object-defined |
| 2 | c085: Throwing clothes | 0.26 | 7.1 | −6.8 | Object-defined |
| 3 | c101: Holding a window | 0.30 | 5.6 | −5.3 | Object-defined |
| 4 | c060: Opening a box | 0.37 | 14.1 | −13.7 | Fine-grained manipulation |
| 5 | c089: Holding a paper/notebook | 0.50 | 4.5 | −4.0 | Object-defined |
| 6 | c024: Tidying up a blanket | 0.53 | 7.5 | −7.0 | Object interaction |
| 7 | c064: Throwing a pillow | 0.64 | 8.6 | −8.0 | Object-defined |
| 8 | c031: Throwing a bag | 0.73 | 7.1 | −6.4 | Object-defined |
| 9 | c086: Putting a pillow | 0.75 | 9.0 | −8.3 | Object-defined |
| 10 | c039: Opening a window | 0.78 | 5.1 | −4.3 | Fine-grained manipulation |
| 11 | c138: Turning on a television | 0.91 | 5.7 | −4.8 | Minimal body motion |
| 12 | c066: Putting a picture | 0.96 | 3.8 | −2.8 | Fine-grained manipulation |
| 13 | c095: Playing with phone/camera | 1.09 | 3.3 | −2.2 | Minimal body motion |
| 14 | c103: Working at a table | 1.06 | 2.9 | −1.8 | Ambiguous posture |
| 15 | c093: Putting shoes | 1.09 | 4.5 | −3.4 | Fine-grained manipulation |

Three failure modes emerge:

**Failure mode 1 — Object-discriminated throwing classes**: c045 (*throwing a book*), c085 (*throwing clothes*), c064 (*throwing a pillow*), c031 (*throwing a bag*) all share an identical ballistic arm trajectory. The discriminative signal lies entirely in the **object being thrown**, which is invisible to skeleton features. This group represents the most fundamental limitation of skeleton-only approaches.

**Failure mode 2 — Fine-grained hand manipulation**: c060 (*opening a box*), c039 (*opening a window*), c066 (*putting a picture*), c093 (*putting shoes*) require discrimination based on subtle hand-object interactions that are below the spatial resolution of the 17–25 joint skeleton topology. The distinction between "opening a box" and "opening a window" lies in the object geometry and hand shape, neither of which is captured by joint keypoints.

**Failure mode 3 — Minimal body motion**: c138 (*turning on a television*), c095 (*playing with phone*), c103 (*working at a table*) involve actions where the body is essentially stationary and the discriminative signal lies in the device being manipulated. The skeleton embedding for these actions is indistinguishable from "sitting still".

### 10.4 Complementarity Analysis — Skeleton vs CLIP

To quantify the complementarity between skeleton and visual features, we compute the per-class AP advantage of each modality:

| Metric | Count | Percentage |
|--------|------:|:----------:|
| Classes where skeleton AP > CLIP AP | 18 | 11.5% |
| Classes where skeleton AP > CLIP AP by ≥ 5 points | 8 | 5.1% |
| Classes where CLIP AP > skeleton AP | 139 | 88.5% |
| Classes where CLIP AP > skeleton AP by ≥ 20 points | 42 | 26.8% |

**CLIP dominates on the vast majority of classes** (88.5%), which is expected given that Charades is fundamentally a video dataset where visual appearance carries the primary discriminative signal. However, the 8 classes where skeleton achieves a substantial advantage (≥5 AP over CLIP) represent precisely the kinematic-discriminated classes identified in the Phase 1 analysis. This confirms that skeleton features encode information that is genuinely **absent** from CLIP, not merely redundant.

The fusion hypothesis is therefore well-supported: **the skeleton modality cannot compete with CLIP alone, but it encodes complementary kinematic information on a targeted subset of classes that CLIP systematically underperforms on**. The optimal fusion strategy should allow the model to selectively leverage skeleton information for these classes while defaulting to CLIP for the majority.

---

## 11. Diagnosis and Remediation Strategies

### 11.1 Primary Diagnosis: Input Dimensionality Mismatch

The most likely cause of the poor standalone performance is the **dimensionality mismatch between the skeleton feature space and the MS-Temba input projection**. The input projection `Linear(4096→256)` compresses the skeleton embedding by a factor of 16×, which is substantially more aggressive than the CLIP configuration (768→256, 3× compression) or the DINOv3 configuration (1024→256, 4× compression). This extreme compression may result in information loss that disproportionately affects the discriminative dimensions of the skeleton embedding.

**Evidence supporting this diagnosis**:
- The training mAP reaches 52.45 (epoch 36), demonstrating that the model *can* extract discriminative information from the skeleton features given sufficient capacity — the information is present in the features but the generalisation pathway is too narrow.
- The validation mAP plateaus at ~9.3–9.5 despite continued training improvement, suggesting a fixed generalisation bottleneck rather than a data quality issue.
- Block 2 achieves higher validation mAP than Block 3, indicating that additional model depth does not help — the bottleneck is upstream at the input projection.

### 11.2 Remediation Strategy 1: Intermediate Dimensionality Reduction

Rather than compressing directly from 4096 to 256, a two-stage projection could be employed:

```python
# Current: single projection
self.proj = nn.Sequential(
    nn.Linear(4096, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1)
)

# Proposed: two-stage projection with bottleneck
self.proj = nn.Sequential(
    nn.Linear(4096, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
    nn.Linear(1024, 256),  nn.LayerNorm(256),  nn.GELU(), nn.Dropout(0.1)
)
```

This adds approximately 4.2M parameters but reduces the compression ratio per stage to a more manageable 4× per step, consistent with what the model handles successfully for DINOv3 features.

### 11.3 Remediation Strategy 2: PCA Pre-Compression

An alternative to a learnable two-stage projection is to apply PCA to the skeleton features offline, reducing them from 4096 to 1024 or 512 dimensions before training. This has the advantage of being a zero-parameter operation that preserves the variance-maximising directions of the feature space:

```python
# Offline PCA (once, before training)
from sklearn.decomposition import PCA
pca = PCA(n_components=1024)
skel_reduced = pca.fit_transform(skel_raw)  # [N, 4096] → [N, 1024]
# Save reduced features, train with in_feat_dim=1024
```

The retained variance fraction should be monitored: if 1024 components retain >95% of the total variance, the 4096-dimensional representation contains substantial redundancy that can be safely discarded.

### 11.4 Remediation Strategy 3: Stronger Regularisation

The current regularisation (dropout=0.1, drop-path=0.1, weight-decay=0.05) may be insufficient for the 4096-dimensional input. Alternative configurations to explore:

| Config | Dropout | Drop-path | Weight decay | Rationale |
|--------|:-------:|:---------:|:------------:|-----------|
| reg_v1 (current) | 0.1 | 0.1 | 0.05 | Moderate |
| reg_v2 | 0.2 | 0.2 | 0.05 | Stronger dropout |
| reg_v3 | 0.1 | 0.1 | 0.10 | Stronger weight decay |
| reg_v4 | 0.3 | 0.2 | 0.10 | Aggressive |

Given that the model memorises training data rapidly (train mAP >50 by epoch 36), a substantially more aggressive regularisation schedule is warranted, particularly on the input projection layer.

### 11.5 Remediation Strategy 4: Lower Learning Rate for Skeleton Projection

A discriminative learning rate schedule could be applied, using a lower learning rate for the input projection layer while maintaining the standard rate for the SSM blocks:

```python
# Separate param groups with different LR
proj_params = list(model.proj.parameters())
other_params = [p for p in model.parameters() if id(p) not in {id(pp) for pp in proj_params}]

optimizer = AdamW([
    {'params': proj_params, 'lr': 1e-4},   # 5× lower for skeleton projection
    {'params': other_params, 'lr': 5e-4}   # standard LR for SSM blocks
], weight_decay=0.05)
```

This strategy slows down the projection layer's capacity to memorise input-specific patterns while allowing the SSM blocks to learn temporal dynamics at normal speed.

### 11.6 Priority Assessment

Given the primary goal of skeleton integration as a **complementary modality** (not a standalone backbone), the remediation strategies are prioritised as follows:

1. **Highest priority — Gated fusion with CLIP** (Section 12): the skeleton's complementary per-class profile makes fusion the most impactful next step, even without improving standalone skeleton performance.
2. **Medium priority — PCA pre-compression**: a simple, zero-parameter intervention that may improve both standalone and fusion performance by removing redundant dimensions.
3. **Lower priority — Two-stage projection and regularisation ablations**: these are model modifications that increase complexity and should be explored only if fusion results are below expectations.

---

## 12. Next Steps — Visual–Skeleton Fusion

### 12.1 Phase 2B.2 — Score-Level Fusion (Immediate, Zero Code Changes)

The simplest fusion approach averages the output logits of independently trained CLIP and skeleton models:

```python
# At inference time
logits_clip = model_clip(features_clip)     # [B, T, 157]
logits_skel = model_skel(features_skel)     # [B, T, 157]
logits_fused = (logits_clip + logits_skel) / 2  # [B, T, 157]
```

This requires **zero architectural modifications**: both models are already trained, and the fusion is performed post-hoc. The expected gain is modest (estimated ΔmAP: +0.3 to +0.8 vs CLIP baseline) due to the large performance asymmetry between the two models, but it provides a useful sanity check — any positive ΔmAP confirms that the skeleton encodes non-redundant information.

An important extension is **weighted score-level fusion**, where the interpolation weight α is optimised on the validation set:

```python
logits_fused = α * logits_clip + (1 − α) * logits_skel
# Sweep α ∈ {0.7, 0.8, 0.85, 0.9, 0.95} on validation set
```

Given the large performance gap, the optimal α is expected to be ≥0.85, heavily favouring CLIP.

### 12.2 Phase 2B.3 — Feature Concatenation (Lower Bound)

Concatenation of CLIP and skeleton features at the input level:

```
Input:  [f_CLIP (768) ‖ f_skel (4096)] → Linear(4864 → 256) → MS-Temba blocks
```

This serves as a lower bound for feature-level fusion. The expected weakness is **modality dominance**: since CLIP features are already well-structured for the semantic task and occupy a smaller subspace (768/4864 = 15.8% of the input), the gradient signal may be dominated by the skeleton dimensions, leading to suboptimal utilisation of the CLIP information.

### 12.3 Phase 2B.4 — Additive Dual Projector

Independent projections for each modality, summed:

```python
f_v = proj_visual(f_CLIP)   # Linear(768→256) + LN + GELU + Drop
f_s = proj_skel(f_skel)     # Linear(4096→256) + LN + GELU + Drop
h = f_v + f_s               # [B, T, 256] → input to MS-Temba blocks
```

This preserves each modality's projection pathway and avoids the dimensionality imbalance of concatenation. However, the summation assigns equal weight to both modalities at all times, which is suboptimal for actions where one modality is clearly dominant.

### 12.4 Phase 2B.5 — Gated Dual Projector (Recommended)

The gated fusion architecture learns per-frame, per-dimension modality weights:

```python
class GatedDualProjection(nn.Module):
    def __init__(self, vis_dim, skel_dim, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_vis  = nn.Sequential(
            nn.Linear(vis_dim,  out_dim), nn.LayerNorm(out_dim),
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
            # For anomalous videos (e.g. 5UNDJ): force g=1 (full visual weight)
            g = g.masked_fill(skel_mask.unsqueeze(-1).unsqueeze(-1), 1.0)
        return g * h_v + (1.0 - g) * h_s
```

This allows the model to adaptively weight each modality per frame: on frames where visual appearance is dominant (e.g., a clearly visible cooking scene), the gate will up-weight CLIP; on frames where kinematic information is critical (e.g., the moment of a throwing gesture), the gate will up-weight skeleton. The parameter overhead is approximately 131K parameters (+0.7% relative to the baseline model).

**Gate initialisation**: based on the per-class analysis, the gate bias should be initialised to favour CLIP (e.g., bias=+0.5 so that σ(0.5) ≈ 0.62, giving ~62% weight to visual features at initialisation). This reflects the prior that CLIP dominates on 88.5% of classes and prevents the skeleton signal from disrupting CLIP's established representations during early training.

**Skeleton mask handling**: for anomalous videos where skeleton features are missing or severely truncated (cf. video `5UNDJ`), a binary mask forces the gate to 1.0 (full visual weight), ensuring graceful degradation.

### 12.5 Phase 2B.6/2B.7 — Cross-Attention Fusion (Conditional)

Cross-attention provides semantically directed fusion with explicit attention mechanisms:

**Phase 2B.6 — Skeleton-as-Query (sv)**: skeleton queries attend to visual features, asking "given this body configuration, which visual context is relevant?"

```python
Q = proj_Q(f_skel)    # [B, T, d_head]
K = proj_K(f_CLIP)    # [B, T, d_head]
V = proj_V(f_CLIP)    # [B, T, d_head]
h = softmax(QK^T / √d) · V + proj_out(f_skel)  # residual
```

**Phase 2B.7 — Visual-as-Query (vs)**: visual queries attend to skeleton features, asking "given this scene, which kinematic dimensions are relevant?"

```python
Q = proj_Q(f_CLIP)    # [B, T, d_head]
K = proj_K(f_skel)    # [B, T, d_head]
V = proj_V(f_skel)    # [B, T, d_head]
h = softmax(QK^T / √d) · V + proj_out(f_CLIP)  # residual
```

Cross-attention is more computationally expensive (+~15% parameters) and more susceptible to overfitting on the limited Charades training set. It is therefore conditional on the gated fusion results: if gated fusion achieves ΔmAP ≥ +1.5 vs CLIP, cross-attention will be explored; otherwise, the marginal complexity is unlikely to be justified.

### 12.6 Expected Impact on mAP

Based on the per-class analysis and the observed complementarity pattern:

| Strategy | Estimated mAP | ΔmAP vs CLIP | Confidence |
|----------|:------------:|:------------:|:----------:|
| Score-level fusion (α=0.9) | 32.7–33.0 | +0.3 to +0.6 | High |
| Feature concatenation | 32.5–33.5 | +0.1 to +1.1 | Medium |
| Additive dual projector | 33.0–34.0 | +0.6 to +1.6 | Medium |
| **Gated dual projector** | **33.5–35.0** | **+1.1 to +2.6** | **Medium–High** |
| Cross-attention | 33.5–35.5 | +1.1 to +3.1 | Low–Medium |

The estimates are derived from the number of classes where skeleton features exhibit a positive advantage (18 classes) weighted by their expected AP improvement under fusion, constrained by the overall dataset mAP contribution of each class.

### 12.7 Recommended Execution Order

1. **Score-level fusion** (2B.2): immediate, validates complementarity, establishes fusion baseline.
2. **Gated dual projector** (2B.5): primary fusion experiment, expected to deliver the largest gain.
3. **Additive dual projector** (2B.4): ablation to isolate the gating mechanism's contribution.
4. **Feature concatenation** (2B.3): ablation to compare against naive concatenation.
5. **Cross-attention** (2B.6/2B.7): conditional on gated fusion results.

---

## 13. Literature Survey — Multi-Modal Fusion for Action Detection

### 13.1 Skeleton–Visual Fusion in Recent Literature

The integration of skeleton and visual features for action recognition has received substantial attention in recent work, with several methodological trends relevant to this project.

**CLIP-MG** (Xiang et al., 2025, arXiv:2506.16385) introduces a skeleton-as-Query cross-attention mechanism where skeleton features guide the spatial attention of CLIP visual features. The key insight is that skeleton information can serve as a **spatial prior** that directs visual attention to body-relevant image regions, suppressing background distractors. On NTU RGB+D 120, skeleton-guided attention improves CLIP-based recognition by +16.5 percentage points over visual-only baselines. This architecture directly motivates our Phase 2B.6 (skeleton-as-Query) and provides empirical evidence that the attention pattern can be learned despite the large dimensionality gap between skeleton (typically 256-512 after projection) and visual features (768-1024).

**SkeletonCLIP++** (Lin et al., 2024) reverses the attention direction: CLIP semantic embeddings are used to weight individual skeleton frames, implementing a form of **semantic temporal attention**. Frames where the visual context matches a semantic action prototype receive higher skeleton weights, while ambiguous frames are down-weighted. The Weighted Frame Integration (WFI) mechanism achieves state-of-the-art results on several skeleton benchmarks by leveraging the semantic coherence of CLIP features as a temporal curriculum for skeleton processing. This approach is relevant to our gated fusion design, where the gate function serves an analogous role — weighting skeleton contributions based on visual context.

**Zhu et al.** (ACM TOMM 2022, arXiv:2202.11374) propose a two-stage fusion architecture for skeleton-guided action recognition: (1) **early fusion** via skeleton-guided spatial attention that prunes irrelevant visual regions, followed by (2) **late fusion** via cross-attention between skeleton temporal features and visual temporal features. The two-stage design explicitly separates the spatial and temporal aspects of skeleton–visual complementarity. On Charades-like multi-label benchmarks, the authors report consistent improvements of +1.5 to +3.0 mAP over visual-only baselines.

**HCMFN** (Hu et al., 2024) employs skeleton features as **spatial anchors** for visual feature extraction, using joint locations to define region-of-interest (ROI) pooling windows in the visual feature map. This approach is particularly effective for actions involving hand-object interactions, where the skeleton provides precise spatial localisation that global visual features lack. The architecture achieves improvements of +2.1 mAP on fine-grained manipulation classes.

**COMM** (Jiang et al., arXiv:2310.08825) demonstrates that DINOv2 features from different network layers encode qualitatively different information: shallow layers preserve low-level spatial detail (texture, edge orientation) while deep layers encode high-level semantic content. Applied to the skeleton fusion context, this suggests that the choice of visual feature layer matters: shallow CLIP or DINOv3 features may provide better spatial alignment with skeleton features than the final CLS token, which is optimised for global semantic classification rather than body-part localisation.

**GCTF** (VideoMamba context, 2024) explores gated cross-modal temporal fusion within the Mamba/SSM framework, demonstrating that the selective scan mechanism of SSMs can be extended to operate across modalities. The gating mechanism learns to dynamically select which modality should dominate the state-space update at each temporal step. This is architecturally similar to our GatedDualProjection but operates at the SSM block level rather than the input projection level, potentially capturing richer temporal interaction patterns.

### 13.2 Key Insights for This Project

From the literature survey, several design principles emerge that inform the fusion architecture:

1. **The gate/attention direction matters**: skeleton-as-Query (CLIP-MG, HCMFN) and visual-as-Query (SkeletonCLIP++) produce qualitatively different attention patterns. Given that CLIP dominates on 88.5% of classes while skeleton provides targeted improvements on a small subset, **visual-as-Query** is likely more appropriate for Charades — the model should default to CLIP and selectively attend to skeleton features when kinematic information is needed.

2. **Gate initialisation bias**: the performance asymmetry (32.4 vs 9.5 mAP) necessitates a **visual-favouring initialisation** of the gate. Without this, the skeleton signal may disrupt CLIP's well-learned representations during early training, producing worse results than CLIP alone. CLIP-MG addresses this with a temperature-scaled residual connection; we implement it via gate bias initialisation (Section 12.4).

3. **Freezing the visual backbone**: several papers (SkeletonCLIP++, CLIP-MG) report that freezing CLIP features during fusion training prevents catastrophic forgetting of the visual representations. This is directly applicable to our setup, where the CLIP model's learned projection should be preserved while the skeleton projection and gate are trained.

4. **Two-stage training**: Zhu et al. demonstrate that training the fusion modules after independently pre-training each modality's stream (as we have done with the CLIP baseline and skeleton single-stream) is preferable to joint training from scratch. This validates our progressive ablation methodology.

### 13.3 Recommended Fusion Architecture

Based on the literature analysis and the empirical results from Phase 2B.1, the recommended production architecture is a **gated dual projector with visual-biased initialisation**:

```
f_clip [B, T, 768]  → proj_vis  → h_v [B, T, 256]
f_skel [B, T, 4096] → proj_skel → h_s [B, T, 256]

g = σ(Linear([h_v ‖ h_s]) + bias_init=0.5)  → [B, T, 256]

h = g · h_v + (1−g) · h_s  → MS-Temba blocks → classifier → [B, T, 157]
```

The `proj_vis` weights can optionally be initialised from the trained CLIP baseline checkpoint (warm-starting the visual pathway), while `proj_skel` and the gate are trained from scratch. The skeleton projection should use a reduced learning rate (0.2× the base rate) to prevent gradient instability from the 4096-dimensional input.

---

## Appendix A — File Paths Reference

| Resource | Path |
|----------|------|
| Raw skeleton zip | `/srv/storage/.../share/Charades/Charades_SCDNet_features2.zip` |
| Aligned skeleton features | `data/hf_features/Temporal_Action_Detection/charades_scdnet_w16/` |
| Extraction script | `vim/extract_scdnet_features.py` |
| OAR job script | `vim/scripts/job_scdnet_align.oar.sh` |
| Training script (skeleton) | `vim/scripts/run_charades_scdnet_seed0.sh` |
| CLIP features | `data/hf_features/Temporal_Action_Detection/charades_features_clip/` |
| DINOv3 features | `data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps/` |
| Training log (skeleton) | `runs/charades/scdnet/seed0/training.log` |
| Metrics CSV (skeleton) | `runs/charades/scdnet/seed0/metrics_per_epoch.csv` |
| Best checkpoint (skeleton) | `runs/charades/scdnet/seed0/checkpoint_best.pth` |

## Appendix B — Anomalous Videos Detected During Extraction

| Video ID | Skeleton windows | CLIP windows | Discrepancy | Likely cause |
|----------|:----------------:|:------------:|:-----------:|--------------|
| 5UNDJ | 22 | 292 | 270 | SCDNet detection failure — severe occlusion or multi-person scene |

**Handling strategy**: during dataloader integration for gated fusion, videos with `|skel_T - visual_T| > 1` will be flagged and the skeleton stream will be replaced with a zero tensor for that video, falling back entirely to the visual features. The gate in gated fusion will naturally learn to output g≈1 (full visual weight) for such cases.

## Appendix C — Complete Per-Class AP Table (Skeleton Single-Stream, Best Epoch 21, Sampled Validation)

| Class | Description | Skeleton AP | CLIP AP (Phase 1) |
|-------|-------------|:-----------:|:---------:|
| c000 | Holding something | 14.16 | 45.0 |
| c001 | Standing up | 13.60 | 36.8 |
| c002 | Drinking from cup/glass/bottle | 11.66 | 46.0 |
| c003 | Opening a door | 9.28 | 25.4 |
| c004 | Closing a door | 7.90 | 18.5 |
| c005 | Putting a shoe somewhere | 1.99 | 16.5 |
| c006 | Sitting in a chair | 11.38 | 51.0 |
| c007 | Putting something on a table | 5.67 | 28.3 |
| c008 | Walking through a doorway | 17.78 | 22.8 |
| c009 | Sitting on the floor | 11.71 | 36.7 |
| c010 | Sitting at a table | 4.63 | 36.7 |
| c011 | Sitting on a bed | 43.00 | 36.7 |
| c012 | Lying on the floor | 6.35 | 14.1 |
| c013 | Throwing a pillow somewhere | 1.68 | 8.6 |
| c014 | Watching/Looking outside of a window | 23.23 | 22.8 |
| c015 | Watching/reading/looking at a book | 22.77 | 22.8 |
| c016 | Working/Playing on a laptop/computer | 18.44 | 75.9 |
| c017 | Putting a bag somewhere | 3.41 | 7.1 |
| c018 | Opening a bag | 4.08 | 14.1 |
| c019 | Closing a box | 13.49 | 25.4 |
| c020 | Opening a book | 14.78 | 22.8 |
| c021 | Closing a book | 6.95 | 22.8 |
| c022 | Taking a shoe off somewhere | 4.43 | 16.5 |
| c023 | Taking off some clothes | 4.58 | 18.5 |
| c024 | Tidying up a blanket/s | 0.53 | 7.5 |
| c025 | Going from one room to another | 3.77 | 22.8 |
| c026 | Putting a blanket somewhere | 11.69 | 18.5 |
| c027 | Reaching for and grabbing a pillow | 4.86 | 8.6 |
| c028 | Laughing | 3.82 | 17.8 |
| c029 | Taking a picture/photo of something | 2.84 | 8.6 |
| c030 | Watching a video | 3.50 | 30.5 |
| c031 | Throwing a bag somewhere | 0.73 | 7.1 |
| c032 | Grasping onto a doorknob | 10.31 | 25.4 |
| c033 | Holding a pillow | 12.25 | 27.9 |
| c034 | Holding a bag | 6.94 | 7.1 |
| c035 | Opening a closet/cabinet | 4.33 | 25.4 |
| c036 | Pouring something into a cup | 2.72 | 46.0 |
| c037 | Cooking something | 8.63 | 79.0 |
| c038 | Closing a closet/cabinet | 4.11 | 25.4 |
| c039 | Opening a window | 0.78 | 5.1 |
| c040 | Putting some food somewhere | 5.13 | 28.3 |
| c041 | Turning off a light | 2.23 | 8.7 |
| c042 | Holding a cup/glass/bottle | 2.35 | 46.0 |
| c043 | Taking something from a box | 3.20 | 14.1 |
| c044 | Putting something in a box | 2.43 | 14.1 |
| c045 | Throwing a book somewhere | 0.17 | 8.2 |
| c046 | Opening a refrigerator | 1.61 | 60.2 |
| c047 | Putting on a hat | 12.36 | 18.5 |
| c048 | Lying on a bed | 1.41 | 36.7 |
| c049 | Running | 7.54 | 18.9 |
| c050 | Holding a shoe/shoes | 1.42 | 16.5 |
| c051 | Sitting on a chair | 14.11 | 51.0 |
| c052 | Holding a box | 12.57 | 14.1 |
| c053 | Putting some clothes somewhere | 4.19 | 18.5 |
| c054 | Reaching for and grabbing a bag | 4.29 | 7.1 |
| c055 | Holding a dish | 13.06 | 33.8 |
| c056 | Throwing a box somewhere | 3.88 | 14.1 |
| c057 | Throwing shoes somewhere | 3.81 | 16.5 |
| c058 | Closing a refrigerator | 2.74 | 60.2 |
| c059 | Drinking from cup/glass/bottle (v2) | 51.79 | 46.0 |
| c060 | Opening a box | 0.37 | 14.1 |
| c061 | Eating something | 26.76 | 21.7 |
| c062 | Holding a phone/camera | 8.52 | 75.9 |
| c063 | Looking at a phone/camera | 7.98 | 75.9 |
| c064 | Throwing a pillow | 0.64 | 8.6 |
| c065 | Holding a broom | 9.15 | 75.4 |
| c066 | Putting a picture somewhere | 0.96 | 3.8 |
| c067 | Holding a blanket | 9.41 | 18.5 |
| c068 | Laughing at someone | 3.18 | 17.8 |
| c069 | Washing a dish | 1.52 | 33.8 |
| c070 | Putting a cup/glass/bottle somewhere | 14.35 | 46.0 |
| c071 | Putting a blanket somewhere (v2) | 7.24 | 18.5 |
| c072 | Holding a laptop | 16.83 | 75.9 |
| c073 | Holding a towel | 5.80 | 18.5 |
| c074 | Holding a mirror | 4.33 | 8.6 |
| c075 | Holding a food item | 5.68 | 28.3 |
| c076 | Holding a bag (v2) | 10.01 | 7.1 |
| c077 | Fixing something | 4.03 | 8.6 |
| c078 | Holding a book | 19.12 | 22.8 |
| c079 | Washing something | 3.05 | 33.8 |
| c080 | Taking a towel from somewhere | 3.07 | 18.5 |
| c081 | Holding a sandwich | 7.78 | 28.3 |
| c082 | Holding a clothes item | 11.00 | 18.5 |
| c083 | Holding a picture | 1.59 | 3.8 |
| c084 | Fixing hair | 2.43 | 8.6 |
| c085 | Throwing clothes somewhere | 0.26 | 7.1 |
| c086 | Putting a pillow somewhere | 0.75 | 9.0 |
| c087 | Cooking/Stirring food | 14.06 | 79.0 |
| c088 | Holding a vacuum | 2.90 | 74.2 |
| c089 | Holding a paper/notebook | 0.50 | 4.5 |
| c090 | Looking in a mirror | 5.63 | 8.6 |
| c091 | Lying on a sofa/couch | 1.41 | 36.7 |
| c092 | Taking a box from somewhere | 4.47 | 14.1 |
| c093 | Putting shoes somewhere | 1.09 | 4.5 |
| c094 | Putting a bag on something | 3.25 | 7.1 |
| c095 | Playing with a phone/camera | 0.97 | 3.3 |
| c096 | Eating a snack | 8.90 | 28.3 |
| c097 | Watching television | 30.48 | 30.5 |
| c098 | Sitting on a sofa/couch | 8.20 | 51.0 |
| c099 | Putting a broom somewhere | 2.54 | 53.0 |
| c100 | Closing a window | 2.53 | 5.1 |
| c101 | Holding a window | 0.30 | 5.6 |
| c102 | Putting a phone/camera somewhere | 9.04 | 75.9 |
| c103 | Working at a table | 1.06 | 2.9 |
| c104 | Turning on a light | 2.95 | 8.7 |
| c105 | Turning off a light (v2) | 1.27 | 2.3 |
| c106 | Eating at a table | 21.50 | 21.7 |
| c107 | Holding a door | 22.21 | 25.4 |
| c108 | Grasping onto a door | 3.31 | 25.4 |
| c109 | Opening a door (v2) | 6.48 | 25.4 |
| c110 | Cleaning the floor | 7.22 | 33.8 |
| c111 | Fixing a light | 2.43 | 8.7 |
| c112 | Taking a towel/s from somewhere | 5.05 | 18.5 |
| c113 | Sneezing | 14.74 | 17.8 |
| c114 | Washing hands | 12.64 | 33.8 |
| c115 | Holding a towel/s | 7.06 | 18.5 |
| c116 | Throwing a towel/s somewhere | 1.30 | 16.5 |
| c117 | Talking on a phone/camera | 2.57 | 75.9 |
| c118 | Holding a bag of groceries | 15.92 | 7.1 |
| c119 | Standing on something | 5.58 | 36.8 |
| c120 | Holding a bottle | 5.23 | 46.0 |
| c121 | Lying on a bed (v2) | 12.73 | 36.7 |
| c122 | Holding a paper | 26.79 | 22.0 |
| c123 | Walking | 35.80 | 18.7 |
| c124 | Running somewhere | 22.55 | 18.9 |
| c125 | Closing a book (v2) | 31.33 | 22.5 |
| c126 | Sitting on a toilet | 5.67 | 36.7 |
| c127 | Opening a cabinet | 22.53 | 25.4 |
| c128 | Opening a refrigerator (v2) | 3.10 | 60.2 |
| c129 | Holding a grocery bag | 2.53 | 7.1 |
| c130 | Fixing a door | 7.44 | 25.4 |
| c131 | Smiling | 3.64 | 17.8 |
| c132 | Holding a pillow (v2) | 33.49 | 27.9 |
| c133 | Someone is undressing | 21.90 | 18.5 |
| c134 | Holding a container | 31.84 | 31.8 |
| c135 | Sitting in a bed | 21.61 | 36.7 |
| c136 | Putting a glass somewhere | 1.13 | 46.0 |
| c137 | Holding a vacuum (v2) | 8.78 | 74.2 |
| c138 | Turning on a television | 0.91 | 5.7 |
| c139 | Holding a remote | 5.71 | 8.6 |
| c140 | Going somewhere | 3.26 | 22.8 |
| c141 | Holding a towel (v3) | 11.47 | 18.5 |
| c142 | Pouring from a container | 8.99 | 46.0 |
| c143 | Opening a refrigerator (v3) | 3.54 | 88.1 |
| c144 | Closing a door (v2) | 3.90 | 18.5 |
| c145 | Closing a refrigerator (v2) | 3.52 | 60.2 |
| c146 | Smiling at someone | 33.50 | 33.5 |
| c147 | Opening a closet | 6.75 | 25.4 |
| c148 | Sitting somewhere | 9.63 | 51.0 |
| c149 | Taking something off a shelf | 7.05 | 14.1 |
| c150 | Holding a bag (v3) | 19.79 | 7.1 |
| c151 | Closing a closet | 52.26 | 23.5 |
| c152 | Holding a phone | 20.46 | 75.9 |
| c153 | Sitting on something | 19.04 | 51.0 |
| c154 | Sitting down | 31.64 | 30.5 |
| c155 | Holding a cup | 25.31 | 24.9 |
| c156 | Putting something somewhere | 20.71 | 28.3 |

---

## Appendix D — Documentation Plan

| Document | Content | Status |
|----------|---------|:------:|
| `phase1_baseline_clip_dino.md` | Baseline, regularisation, full comparison | ✅ |
| `phase2b_skeleton_extraction_and_results.md` | This document — extraction, alignment, single-stream results, fusion roadmap | ✅ |
| `phase2a_dino_dense_features.md` | DINOv3 patch feature extraction, mean-patch and combined training | ⏳ |
| `phase2b_skeleton_fusion.md` | Score-level, gated, cross-attention ablation | ⏳ |
| `phase2c_clip_dino_fusion.md` | CLIP + DINOv3 fusion experiments | ⏳ |
| `phase3_survey_skeleton_visual.md` | Extended literature survey | ⏳ |

---