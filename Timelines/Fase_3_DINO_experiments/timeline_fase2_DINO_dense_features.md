# MS-Temba on Charades: Technical Analysis — Phase 2A
### DINOv3 Dense Patch Features: From CLS-Only to Spatially Structured Representations

> **Author**: Matteo Di Iorio  
> **Period**: March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes, GPU H100/A100  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Dataset**: Charades v1 — 157 action classes, 7985 training videos, 1863 test videos  
> **Document**: Phase 2A of N — DINOv3 dense patch feature extraction, architectural adaptation, ablation design  
> **Prerequisites**: Phase 1 complete (CLIP baseline 32.40 mAP, DINOv3 CLS-only reg_v2 baseline 25.21 mAP)

---

## Table of Contents

1. [Context and Motivation](#1-context-and-motivation)
2. [DINOv3 Internal Representations: CLS Token vs Patch Tokens](#2-dinov3-internal-representations-cls-token-vs-patch-tokens)
3. [Why CLS-Only is Suboptimal for DINOv3](#3-why-cls-only-is-suboptimal-for-dinov3)
4. [Dense Feature Extraction Strategies](#4-dense-feature-extraction-strategies)
5. [Temporal Alignment and Storage](#5-temporal-alignment-and-storage)
6. [Architectural Modifications to MS-Temba](#6-architectural-modifications-to-ms-temba)
7. [Ablation Design](#7-ablation-design)
8. [Expected Impact and Per-Class Analysis](#8-expected-impact-and-per-class-analysis)
9. [Implementation Plan](#9-implementation-plan)
10. [Next Steps](#10-next-steps)

---

## 1. Context and Motivation

### 1.1 The CLS-Only Bottleneck

The Phase 1 experiments established a significant performance gap between CLIP (32.40 mAP) and DINOv3 (25.21 mAP reg_v2) on the Charades dataset. The analysis in the comparison document identified three contributing factors, with the second being the most tractable:

1. **Semantic alignment gap** (~50% of gap): CLIP is pre-aligned with the Charades action vocabulary through contrastive text supervision; DINOv3 is not.
2. **CLS token suboptimality for DINOv3** (~40% of gap): the competitive advantage of DINOv3 lies in its patch-level representations, which are not used in the current CLS-only setup.
3. **Dimensionality pressure** (~10% of gap): 1024-dim input with identical training data leads to higher overfitting risk.

Factor 2 is the primary target of Phase 2A. The current DINOv3 pipeline extracts a single 1024-dimensional CLS token per frame — a global summary of the entire image — and discards the 256 individual patch embeddings that encode localised spatial information. This design decision was a pragmatic simplification during the Phase 1 experiments, but it systematically discards the information that makes DINOv3 structurally different from CLIP.

### 1.2 What the Patch Tokens Contain

DINOv3 ViT-L/16 processes a 224×224 image by dividing it into a 14×14 grid of non-overlapping 16×16-pixel patches. Each patch is independently projected into a 1024-dimensional embedding, then refined through 24 transformer layers where patches attend to each other and to the CLS token. The result is 256 patch tokens, each carrying information about a specific 16×16-pixel spatial region of the image, enriched by global context through self-attention.

Unlike CLIP, where the CLS token is explicitly trained to be compatible with global textual descriptions and therefore aggregates information in a semantically compressed way, DINOv3's training objective — self-supervised distillation with Gram anchoring — explicitly preserves the **patch-to-patch relational structure**. The Gram anchoring mechanism regularises the similarity matrix between patch tokens, ensuring that spatial coherence is maintained throughout training. This means DINOv3's patch tokens encode localised visual structure that the CLS token, by construction, cannot fully transmit.

For Charades action recognition, this distinction is consequential: many of the 51 classes where DINOv3 already outperforms CLIP (even with CLS-only) involve spatially specific visual patterns — *opening a refrigerator* (+27.9), *holding a vacuum* (+22.9), *putting a broom somewhere* (+25.6). These are precisely the classes where the spatial distribution of visual elements (object shape, hand position, body-object relationship) is more informative than a global semantic label. By using dense patch features, we directly address the information bottleneck that limits DINOv3's performance on these classes.

### 1.3 Relationship to Phase 2B (Skeleton Features)

Phase 2A and Phase 2B are complementary rather than competing interventions. They address different types of missing information:

- **Phase 2A (DINOv3 dense features)**: recovers spatial structure within each frame — where objects and body parts are located, how they are arranged relative to each other.
- **Phase 2B (skeleton features)**: recovers kinematic structure across frames — how the body is moving, the trajectory of joints, postural transitions.

A longer-term goal (Phase 2C) is to combine all three streams: CLIP (semantic), DINOv3 dense (spatial), and skeleton (kinematic). Phase 2A and 2B provide the individual validated components for this multi-stream architecture.

---

## 2. DINOv3 Internal Representations: CLS Token vs Patch Tokens

### 2.1 Vision Transformer Architecture in DINOv3

To understand why patch tokens are superior for this task, it is necessary to briefly review how a Vision Transformer (ViT) processes an image and what each type of token contains.

Given an input image I ∈ ℝ^(H×W×C), the ViT:

1. **Patchification**: divides the image into N non-overlapping patches of size P×P pixels. For DINOv3 ViT-L/16 with 224×224 input and P=16: N = (224/16)² = 14² = **196 patches**.
2. **Linear projection**: each flattened patch (P²·C = 768 values for RGB) is linearly projected into a D=1024 dimensional token embedding.
3. **Position encoding**: learnable 2D position embeddings are added to each patch token, encoding spatial location.
4. **Prepend CLS token**: a special learnable [CLS] token is prepended to the sequence: [x_cls, x_1, x_2, ..., x_N].
5. **Transformer layers**: L=24 self-attention layers process the full sequence [x_cls, x_1, ..., x_N]. Each layer allows every token to attend to every other token, enabling global information exchange.
6. **Output**: the final hidden states [h_cls, h_1, h_2, ..., h_N] ∈ ℝ^((N+1)×D).

The current pipeline extracts only h_cls — a single 1024-dim vector — and discards h_1, ..., h_196 (the 196 patch token embeddings).

```
Input image [224×224×3]
      │
      ▼ Patchification (14×14 grid, P=16)
[196 patch embeddings] + [1 CLS token]
      │
      ▼ 24 Transformer layers (self-attention)
      │
      ├── h_cls  [1024-dim]   ← CURRENT USE (CLS-only)
      │
      └── h_1...h_196  [196×1024-dim]  ← UNUSED (dense patch tokens)
```

### 2.2 Information Content of CLS vs Patch Tokens

The **CLS token** h_cls is designed to be a global summary of the entire image, aggregated through all transformer layers. After 24 layers of self-attention, h_cls has attended to all 196 patch tokens and incorporates information from the entire image. However, this aggregation is necessarily lossy: the 1024-dim CLS embedding cannot fully encode the spatial layout of 196 independently meaningful regions. The CLS token captures *what is in the image globally*, but the specific spatial arrangement of elements — which object is where, how they relate spatially, what the body configuration looks like — is compressed and potentially lost.

The **patch tokens** h_i carry localised information about a specific 16×16-pixel region, enriched by context from the full image through self-attention. Each patch token encodes:
- Local texture and colour of its spatial region
- Contextual information from adjacent patches (via local attention patterns)
- Global context from distant patches (via long-range attention)
- Spatial position (through positional encodings)

Critically, because DINOv3's Gram anchoring mechanism explicitly regularises the similarity matrix between patch tokens, the relational structure between patches is preserved through training. Two spatially adjacent patches will have similar embeddings if they belong to the same semantic region; two patches from semantically distinct regions (e.g., a hand touching an object vs the background wall) will have dissimilar embeddings.

For action recognition in Charades, this patch-level structure is particularly valuable:

- **Hand-object relationships**: the relative position and similarity of hand patches and object patches encodes information about grasping, reaching, or placing
- **Body configuration**: the spatial arrangement of patches corresponding to different body parts (torso, arms, legs) encodes postural information
- **Object identity and location**: distinct object categories are spatially localised in specific patches, rather than globally mixed in the CLS token

### 2.3 The Gram Anchoring Mechanism and Dense Feature Quality

A key motivation for using DINOv3 specifically (rather than earlier DINO variants or other ViTs) is the **Gram anchoring** training procedure, which directly addresses the degradation of dense features observed in large-scale self-supervised training.

In standard self-supervised distillation (DINO v1/v2), the training objective focuses on aligning the student's CLS token distribution with the teacher's. During extended training at large scale, the patch-level representations tend to degrade — individual patch tokens become less informative and less distinguishable, as the optimisation pressure concentrates on the globally aggregated CLS token.

Gram anchoring adds a regularisation term that constrains the **Gram matrix** of patch tokens — the N×N matrix of pairwise cosine similarities between all patch token pairs — to remain consistent with a reference computed from a clean, minimally augmented view. Formally:

```
L_gram = ||G_student - G_teacher||²_F

where G[i,j] = cos_sim(h_i, h_j)  for patch tokens i, j ∈ {1,...,N}
```

This regularisation explicitly preserves the relational geometry of the patch space: if two patches were similar (or dissimilar) in the reference view, they remain similar (or dissimilar) after augmentation. The practical effect is that DINOv3's patch tokens maintain much higher spatial discriminativity than earlier self-supervised ViTs, making them suitable for dense prediction tasks such as semantic segmentation and depth estimation — and, by extension, for spatially structured action recognition.

---

## 3. Why CLS-Only is Suboptimal for DINOv3

### 3.1 The Aggregation Asymmetry Between CLIP and DINOv3

A critical asymmetry exists between how CLS token quality maps to task performance for CLIP vs DINOv3:

**For CLIP**: the CLS token is the primary output of interest. The entire training objective optimises the CLS token to be maximally compatible with textual descriptions. The patch tokens are a byproduct of the transformer computation, not directly optimised. CLIP's CLS token is therefore a well-calibrated global representation for semantic tasks, and using it alone captures the backbone's primary competency.

**For DINOv3**: the CLS token is a global aggregation, but it is not the backbone's primary competitive advantage. DINOv3's design papers and evaluation benchmarks emphasise its performance on dense prediction tasks (ADE20K segmentation, NYU depth estimation, correspondence) where patch tokens are directly used. The CLS token of DINOv3 is competent for image-level classification, but it does not carry the full information that distinguishes DINOv3 from other ViTs. Using CLS-only for DINOv3 is analogous to using only the [EOS] token of a language model for text classification — technically valid, but discarding most of the model's representational power.

### 3.2 Empirical Evidence from Phase 1

The Phase 1 results provide indirect evidence for this asymmetry. DINOv3 CLS-only achieves 25.21 mAP vs CLIP's 32.40, a gap of 7.19 points. Yet DINOv3 ViT-L/16 is a substantially larger and more parameter-rich model than CLIP ViT-B/16 (307M vs 87M parameters). If DINOv3 were simply a worse backbone, one would not expect the specific pattern of per-class advantages observed: DINOv3 achieves its largest wins on classes involving spatially specific object manipulations (*refrigerator* +27.9, *vacuum* +22.9, *broom* +25.6), which are precisely the classes where spatial patch structure is most informative. This suggests that even the CLS token of DINOv3 carries some patch-level geometric information implicitly, but that the full patch representation would unlock substantially more.

### 3.3 The Scale of the Unused Information

Quantitatively, the information discarded by using CLS-only is substantial. For each frame:

```
CLS-only:  1 × 1024 = 1,024 dimensions used
Patch (196): 196 × 1024 = 200,704 dimensions available
Unused ratio: (200,704 - 1,024) / 200,704 ≈ 99.5% of DINOv3's output discarded
```

Even with aggressive spatial pooling (e.g., mean over all 196 patches), the aggregated representation captures fundamentally different information from the CLS token: it reflects the average visual content across all spatial locations, rather than a globally optimised summary token. The combination of CLS + mean-patch provides a 2048-dim representation that independently encodes both the global semantic summary and the mean spatial content.

---

## 4. Dense Feature Extraction Strategies

### 4.1 Overview of Approaches

Three approaches are considered for integrating DINOv3 patch information, ordered by implementation cost and expected representational richness:

**Approach 1 — Mean patch pooling**: compute the arithmetic mean of all 196 patch token embeddings for each frame. Produces a single 1024-dim vector per frame representing the average visual content across all spatial locations.

**Approach 2 — CLS + mean patch concatenation**: concatenate the CLS token (1024-dim) with the mean-pooled patch tokens (1024-dim), producing a 2048-dim vector per frame. This combines the globally optimised semantic summary (CLS) with the spatially averaged content representation (mean patch).

**Approach 3 — Spatial patch aggregation with attention pooling**: compute a weighted average of the 196 patch tokens using attention scores derived from the CLS token. The CLS token acts as a query, and each patch token's contribution is weighted by its relevance to the global image summary.

### 4.2 Approach 1: Mean Patch Pooling

The mean-pooled patch representation captures the **average spatial content** of the frame:

```
f_mean = (1/N) · Σᵢ hᵢ    for i ∈ {1, ..., 196}
                             shape: [1024]
```

**Advantages**:
- Simple, deterministic, no additional learned parameters
- Preserves average information about all spatial regions equally
- Directly comparable to the CLS token in downstream tasks, enabling clean ablations
- Storage requirement identical to CLS-only (same dimensionality)

**Limitations**:
- Treats all spatial regions equally, including background regions with no discriminative information
- A frame where only the upper-left quadrant is informative (e.g., a hand reaching for an object) will have its signal diluted by 75% of uninformative background patches
- Loses spatial layout information: the average of "hand patches" and "object patches" does not encode their relative spatial position

### 4.3 Approach 2: CLS + Mean Patch Concatenation (Recommended Baseline)

Concatenating the CLS token with mean-pooled patches creates a 2048-dim representation that combines complementary information sources:

```
f_combined = [h_cls ‖ f_mean]    shape: [2048]

h_cls  [1024]: globally optimised semantic summary
f_mean [1024]: spatially averaged visual content
```

**Why this is more powerful than either alone**:

The CLS token and mean-patch vectors are not redundant. CLS is optimised through the self-supervised objective to be a globally discriminative summary token, and its representations are shaped by the full training process. Mean-patch represents the arithmetic average of localised spatial content — it captures the dominant textures and colour statistics of the scene without the semantic compression of the CLS aggregation. Empirically, in DINOv3's own evaluation benchmarks, linear probing on the concatenation of CLS and register tokens outperforms either alone on several tasks.

**Storage requirement**: 2048-dim float32 × 9848 videos × mean ~50 frames/video ≈ **~16GB** — double the CLS-only storage but feasible.

### 4.4 Approach 3: Attention-Weighted Patch Pooling

A more principled aggregation uses the CLS-to-patch attention scores from the final transformer layer as weights:

```
aᵢ = softmax(h_cls · hᵢᵀ / √d_k)    for i ∈ {1, ..., 196}
f_attn = Σᵢ aᵢ · hᵢ                  shape: [1024]
```

The attention weights aᵢ reflect how much each patch contributed to the CLS token's final representation. Patches with high attention scores — typically those containing semantically salient objects or body parts — contribute more to the aggregated representation. This provides a form of **spatial saliency-based pooling** that concentrates representational capacity on the most task-relevant regions.

**Advantages over mean pooling**:
- Naturally down-weights uninformative background regions
- The attention pattern is implicitly class-aware: for a frame of someone holding a phone, patches covering the phone and hand region will receive higher attention than patches covering the background wall
- The resulting 1024-dim vector is directly comparable in dimension to the CLS-only baseline

**Limitations**:
- Requires access to the attention maps from the final transformer layer, not just the output tokens — necessitates a modified forward pass through the DINOv3 backbone
- The attention weights are derived from a self-supervised objective and may not perfectly align with task-relevant saliency for Charades

**Implementation note**: in HuggingFace's DINOv3 implementation (`AutoModel`), attention maps can be extracted by passing `output_attentions=True` to the forward call and indexing the final layer's cross-attention between CLS and patch tokens. This requires a slight modification to the `encode_image_global` function in `dinov3_feature_extractor.py`.

### 4.5 Strategy Selection for Phase 2A

The recommended progression for Phase 2A is:

1. **First experiment**: mean patch pooling (Approach 1) — establishes whether patch content alone improves over CLS
2. **Second experiment**: CLS + mean patch concatenation (Approach 2) — establishes the value of combining both representations
3. **Third experiment** (if time permits): attention-weighted pooling (Approach 3) — explores saliency-based aggregation

The CLS + mean patch concatenation (Approach 2) is expected to be the most effective and is therefore prioritised as the primary baseline for subsequent fusion with skeleton features.

---

## 5. Temporal Alignment and Storage

### 5.1 Existing Feature Extraction Pipeline

The current DINOv3 feature extraction (`vim/dinov3_feature_extractor.py`) uses the following pipeline:

```python
def encode_image_global(image, model, processor, pooling='cls'):
    """Current implementation — CLS-only path."""
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    if pooling == 'cls':
        return outputs.last_hidden_state[:, 0, :]     # [1, 1024] — CLS token
    elif pooling == 'mean_patch':
        return outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # [1, 1024] — mean of patches
```

The `pooling='mean_patch'` path is already implemented in the extractor but was never used for the Charades extraction. Phase 2A activates this path and adds the concatenation option.

The temporal processing is handled by `temporal_average_pool`:

```python
def temporal_average_pool(features, window_size=16):
    """
    Given a list of per-frame features, average over non-overlapping
    windows of window_size frames.
    Input:  list of T arrays, each [D]
    Output: array [ceil(T/window_size), D]
    """
```

This function remains unchanged — the window_size=16 pooling applies identically to mean-patch features as it does to CLS features.

### 5.2 Modifications to the Extraction Script

The modified extraction pipeline for CLS + mean patch concatenation:

```python
def encode_image_combined(image, model, processor):
    """
    Extracts CLS token and mean-pooled patch tokens, concatenates them.
    Returns: [2048] float32 — [CLS(1024) || mean_patch(1024)]
    """
    inputs = processor(images=image, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.last_hidden_state   # [1, 197, 1024]
    cls_token     = hidden_states[:, 0, :]      # [1, 1024]
    patch_tokens  = hidden_states[:, 1:, :]     # [1, 196, 1024]
    mean_patch    = patch_tokens.mean(dim=1)    # [1, 1024]
    
    combined = torch.cat([cls_token, mean_patch], dim=-1)  # [1, 2048]
    return combined.squeeze(0).cpu().numpy().astype(np.float32)
```

The temporal average pooling over 16-frame windows is then applied to the 2048-dim per-frame vectors identically to the current pipeline.

### 5.3 Output Format and Storage

**Mean patch only** (Approach 1):
```
Output shape: [T', 1024] float32
Directory: charades_dinov3_vitl16_w16_24fps_meanpatch/
Storage: ~8.2GB (identical to CLS-only)
```

**CLS + mean patch** (Approach 2, recommended):
```
Output shape: [T', 2048] float32
Directory: charades_dinov3_vitl16_w16_24fps_combined/
Storage: ~16.4GB
```

The naming convention and file structure remain identical to the existing DINOv3 CLS-only features: one `.npy` file per video, named with the 5-character Charades video ID.

### 5.4 Computational Cost of Re-extraction

Re-extracting features requires running the DINOv3 ViT-L/16 forward pass on all 9848 × ~800 frames = ~7.9M frames. At the observed throughput of the original extraction (~30 videos/min on GPU), re-extraction of all features will require approximately **5.5 hours** on a single GPU node.

This cost is incurred once and the features are stored permanently, making subsequent training runs of any fusion configuration essentially free from the extraction perspective.

---

## 6. Architectural Modifications to MS-Temba

### 6.1 Input Projection Layer

The primary architectural impact of switching from CLS-only (1024-dim) to CLS + mean patch (2048-dim) is on the **input projection layer** in `MSTemba.__init__`:

**Current (CLS-only, 1024-dim)**:
```python
self.proj = nn.Sequential(
    nn.Linear(1024, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(p=drop_rate)
)
```

**Modified (CLS + mean patch, 2048-dim)**:
```python
self.proj = nn.Sequential(
    nn.Linear(2048, 256),    # only this line changes
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(p=drop_rate)
)
```

The change is limited to updating `in_feat_dim` from 1024 to 2048, passed via the `--in_feat_dim 2048` argument. All subsequent layers (Block 1, 2, 3, interaction block, classification head) operate on the 256-dim projected space and remain completely unchanged. This is the key architectural advantage of MS-Temba's design: the input projection absorbs the dimensionality change, making the modification fully contained.

### 6.2 Regularisation Implications

Increasing `in_feat_dim` from 1024 to 2048 doubles the number of parameters in the input projection layer. Following the empirical pattern from Phase 1 — where doubling from 512 (CLIP) to 1024 (DINOv3) required explicit regularisation (reg_v2) — a further doubling to 2048 will likely require correspondingly stronger regularisation.

The projection `Linear(2048 → 256)` has 2048 × 256 + 256 = **524,544 parameters** (vs 262,400 for DINOv3 CLS-only). With only 7985 training videos, the ratio of projection parameters to training samples is approximately **66:1**, which is extremely high. Strong regularisation is essential:

| Parameter | DINOv3 CLS-only (reg_v2) | DINOv3 dense (proposed) | Rationale |
|-----------|:------------------------:|:-----------------------:|-----------|
| `--drop` | 0.05 | **0.1** | Larger input space → more dropout |
| `--drop-path` | 0.05 | **0.1** | Idem |
| `--weight-decay` | 0.05 | **0.05** | Maintain L2 pressure |
| `--early-stop-patience` | 15 | **15** | Same patience |
| `--in_feat_dim` | 1024 | **2048** | Combined representation |

### 6.3 Optional: Intermediate Projection for Dimensionality Reduction

An alternative to direct `Linear(2048 → 256)` is a two-stage projection that first reduces dimensionality and then projects to the target:

```python
self.proj = nn.Sequential(
    nn.Linear(2048, 512),      # Stage 1: reduce to CLIP-equivalent dimension
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(p=drop_rate),
    nn.Linear(512, 256),       # Stage 2: project to MS-Temba internal dim
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(p=drop_rate)
)
```

This two-stage approach:
- Adds only ~1M parameters but provides a learnable dimensionality reduction before the main projection
- Allows the model to learn which 2048-dim subspaces are most informative for the task before compressing to 256
- Has been shown empirically to improve performance in multi-modal fusion architectures where the input dimensions are very large

Whether to use single-stage or two-stage projection will be determined by the ablation results: if direct `Linear(2048→256)` shows instability or poor convergence, the two-stage variant will be tested.

---

## 7. Ablation Design

### 7.1 Experiment Matrix

The Phase 2A ablation is structured as a progressive set of experiments, each building on the previous:

| Experiment | Config | `in_feat_dim` | Feature type | Expected mAP | Purpose |
|------------|--------|:-------------:|:------------:|:------------:|---------|
| 2A.0 (baseline) | DINOv3 CLS reg_v2 | 1024 | CLS only | 25.21 ✅ | Reference |
| 2A.1 | DINOv3 mean patch | 1024 | mean(patches) only | ? | Is patch alone better than CLS? |
| 2A.2 | DINOv3 CLS + mean patch | 2048 | CLS ‖ mean(patches) | ? | Does combination help? |
| 2A.3 | DINOv3 attn pooling | 1024 | attention-weighted patches | ? | Does saliency weighting help? |
| 2A.4 | DINOv3 CLS + attn pooling | 2048 | CLS ‖ attn-pooled patches | ? | Best combination? |

Experiments 2A.1 and 2A.2 are the highest priority. 2A.3 and 2A.4 are conditional on time availability and whether 2A.1/2A.2 show meaningful improvements.

### 7.2 Diagnostic Questions

Each experiment is designed to answer a specific diagnostic question:

**2A.1 vs 2A.0**: *Does mean patch pooling alone outperform CLS-only?*
- If yes (mean > CLS): the patch representations carry more task-relevant information than the CLS token for Charades. The CLS aggregation is lossy in a way that hurts performance.
- If no (CLS > mean): the CLS token's globally optimised representation is superior, and patch averaging introduces noise.

**2A.2 vs 2A.0 and 2A.1**: *Does concatenation outperform either individual representation?*
- If yes: the two representations are genuinely complementary — CLS captures something that mean patch does not, and vice versa. Full combination is warranted.
- If approximately equal to max(2A.0, 2A.1): one representation dominates and concatenation does not add value.

**2A.1 vs 2A.3**: *Does saliency-based pooling outperform uniform mean pooling?*
- If yes (attn > mean): the background patches are adding noise that uniform pooling does not filter. Saliency-based pooling provides a meaningful quality improvement.
- If approximately equal: the attention weights do not provide a useful signal for spatial filtering, possibly because DINOv3's self-supervised objective does not strongly specialise attention to task-relevant patches.

### 7.3 Per-Class Analysis Protocol

For each experiment, the per-class AP profile at the best checkpoint will be compared against the CLS-only baseline to identify:

1. **Classes that improve with dense features**: particularly expected in the ~51 DINOv3-winning classes identified in Phase 1, especially those involving spatially specific object manipulations.
2. **Classes that regress**: if dense features introduce noise for semantically descriptive classes (*cooking*, *talking on phone*), this will be visible in per-class AP drops.
3. **New DINOv3 wins vs CLIP**: classes that were CLIP-dominated in Phase 1 but switch to DINOv3 with dense features, indicating that the spatial information resolves a previously ambiguous case.

### 7.4 Hypothesised Per-Class Impact

Based on the spatial structure of DINOv3's patch tokens and the per-class analysis from Phase 1, the following class groups are expected to show the largest improvements with dense features:

**High expected gain — spatially specific object manipulations**:

| Class | CLIP AP | DINOv3 CLS AP | Expected DINOv3 dense AP | Rationale |
|-------|--------:|:-------------:|:------------------------:|-----------|
| Opening a refrigerator (c143) | 60.2 | 88.1 | ~88–92 | Already high; patch features encode refrigerator door geometry |
| Holding a vacuum (c137) | 51.3 | 74.2 | ~76–82 | Vacuum's distinctive shape is better encoded in patch space |
| Putting a broom somewhere (c099) | 27.4 | 53.0 | ~58–65 | Broom's elongated spatial footprint benefits from patch encoding |
| Opening a closet/cabinet (c113) | 41.2 | 51.4 | ~55–62 | Cabinet door spatial geometry → patch advantage |
| Taking a dish from somewhere (c120) | 15.9 | 29.3 | ~33–40 | Hand-dish spatial relationship |

**Moderate expected gain — hand-object interactions**:

| Class | CLIP AP | DINOv3 CLS AP | Expected DINOv3 dense AP | Rationale |
|-------|--------:|:-------------:|:------------------------:|-----------|
| Putting a bag somewhere (c022) | 40.6 | 39.2 | ~45–52 | Directional reach encoded in patch spatial layout |
| Taking a bag from somewhere (c023) | 25.4 | 23.8 | ~30–38 | Idem (opposite direction) |
| Opening a door (c008) | 39.8 | 41.2 | ~44–50 | Door edge position relative to hand patch |
| Taking food from somewhere (c063) | 29.2 | 36.3 | ~38–44 | Object-hand spatial proximity |

**Low or negative expected gain — semantically descriptive classes**:

| Class | CLIP AP | DINOv3 CLS AP | Expected DINOv3 dense AP | Rationale |
|-------|--------:|:-------------:|:------------------------:|-----------|
| Someone is cooking (c147) | 79.0 | 5.5 | ~8–15 | No amount of spatial information recovers the missing semantic prior |
| Talking on phone (c019) | 75.9 | 5.0 | ~10–18 | Idem |
| Working on a laptop (c052) | 75.9 | 68.6 | ~70–75 | Already high with CLS; marginal gain from patches |

---

## 8. Expected Impact and Per-Class Analysis

### 8.1 Global mAP Estimates

The expected mAP improvement from CLS-only (25.21) for each approach:

| Approach | Expected ΔmAP | Expected total mAP | Confidence |
|----------|:-------------:|:-----------------:|:----------:|
| 2A.1: mean patch only | +0.5 to +2.0 | ~25.7 to 27.2 | Medium |
| 2A.2: CLS + mean patch | +1.5 to +4.0 | ~26.7 to 29.2 | Medium-High |
| 2A.3: attention pooling | +1.0 to +3.0 | ~26.2 to 28.2 | Low-Medium |
| 2A.4: CLS + attn pooling | +2.0 to +5.0 | ~27.2 to 30.2 | Medium |

The upper end of the 2A.2 range (29.2 mAP) would bring DINOv3 within ~3 points of CLIP (32.40), a substantial reduction of the gap. However, the semantic alignment gap (CLIP's textual supervision prior) cannot be eliminated through dense features alone, making it unlikely that DINOv3 dense fully matches CLIP on the overall mAP metric.

### 8.2 Where Dense Features are Expected to Close the CLIP-DINOv3 Gap

The most significant impact is expected on the classes where DINOv3 already shows advantages in CLS-only mode (51/157 classes). Dense features should amplify these existing advantages and potentially extend them to additional classes:

- **DINOv3 winning classes** (CLS advantage > 10 AP): deeper spatial encoding should increase margins
- **Near-parity classes** (|CLIP − DINOv3| < 5 AP): spatial information may tip the balance toward DINOv3
- **Classes requiring fine-grained spatial discrimination** (put vs take, open vs close): patch-level direction of reach may help

Critically, the semantically descriptive classes where CLIP has overwhelming advantages (AP > 60 CLIP, AP < 20 DINOv3) are unlikely to be reversed by dense features alone. The fundamental issue is the absence of language-grounded semantic priors in DINOv3's training, which no amount of spatial feature engineering can fully compensate for.

### 8.3 Implications for the CLIP + DINOv3 Fusion Strategy

The per-class AP profile of DINOv3 dense features will directly inform the hybrid routing strategy for the eventual CLIP + DINOv3 fusion:

- Classes where DINOv3 dense > CLIP: route to DINOv3 stream or weight DINOv3 higher in the fusion gate
- Classes where CLIP > DINOv3 dense: route to CLIP stream or weight CLIP higher

This per-class routing is more nuanced than the initial analysis suggested, because dense features may change which classes DINOv3 wins on. The Phase 2A experiments are therefore a prerequisite for designing an informed fusion strategy.

---

## 9. Implementation Plan

### 9.1 Step 1 — Modify `dinov3_feature_extractor.py`

The existing extractor requires three modifications:

**Modification A**: add `encode_image_combined` function that returns CLS ‖ mean_patch:

```python
def encode_image_combined(model, processor, image, device, amp_dtype):
    """
    Returns concatenation of CLS token and mean-pooled patch tokens.
    Output: numpy array [2048] float32.
    """
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.autocast(device_type='cuda', dtype=amp_dtype):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
    hidden = outputs.last_hidden_state.float()   # [1, 197, 1024]
    cls    = hidden[:, 0, :]                     # [1, 1024]
    mean_p = hidden[:, 1:, :].mean(dim=1)        # [1, 1024]
    return torch.cat([cls, mean_p], dim=-1).squeeze(0).cpu().numpy()
```

**Modification B**: add `--pooling` argument to the CLI:
```
--pooling: choices=['cls', 'mean_patch', 'combined', 'attn'],
           default='cls'
```

**Modification C**: update `--save_dir` default to reflect the pooling strategy in the directory name (e.g., `charades_dinov3_vitl16_w16_24fps_combined`).

### 9.2 Step 2 — Create New Launch Scripts

Two new extraction scripts are needed:

`vim/scripts/extract_dino_meanpatch.sh` — for Approach 1 (mean patch only, 1024-dim)  
`vim/scripts/extract_dino_combined.sh` — for Approach 2 (CLS + mean patch, 2048-dim)

Each script runs the modified `dinov3_feature_extractor.py` on a GPU node with appropriate OAR parameters (GPU required, ~5.5 hours walltime).

### 9.3 Step 3 — Create Training Scripts

`vim/scripts/run_charades_dino_meanpatch_seed0.sh`:
```bash
python MSTemba_main.py \
  -backbone dinov3_vitl16 \
  -rgb_root ".../charades_dinov3_vitl16_w16_24fps_meanpatch" \
  -in_feat_dim 1024 \
  --drop 0.05 --drop-path 0.05 --weight-decay 0.05 \
  --early-stop-patience 15 ...
```

`vim/scripts/run_charades_dino_combined_seed0.sh`:
```bash
python MSTemba_main.py \
  -backbone dinov3_vitl16_combined \
  -rgb_root ".../charades_dinov3_vitl16_w16_24fps_combined" \
  -in_feat_dim 2048 \
  --drop 0.1 --drop-path 0.1 --weight-decay 0.05 \
  --early-stop-patience 15 ...
```

### 9.4 Step 4 — Update `MSTemba_main.py`

Add `dinov3_vitl16_combined` and `dinov3_vitl16_meanpatch` to the backbone → in_feat_dim mapping in the argparse section:

```python
feat_dim_map = {
    'clip':                    512,   # or 768 depending on variant
    'i3d':                    1024,
    'dinov3_vitl16':          1024,   # CLS-only
    'dinov3_vitl16_meanpatch': 1024,  # mean patch only
    'dinov3_vitl16_combined':  2048,  # CLS + mean patch
    'scdnet':                 4096,   # Phase 2B
}
if args.in_feat_dim is None:
    args.in_feat_dim = feat_dim_map.get(args.backbone, 512)
```

No changes to the model architecture (`models_MSTemba.py`) are required — the input projection already accepts `in_feat_dim` as a parameter and will automatically adapt to 2048-dim input.

---

## 10. Next Steps

### 10.1 Immediate Actions

1. **Modify `dinov3_feature_extractor.py`**: add `encode_image_combined` and `--pooling` CLI argument
2. **Run mean patch extraction** on a GPU node (~5.5h): produces `charades_dinov3_vitl16_w16_24fps_meanpatch/`
3. **Run combined extraction** on a GPU node (~5.5h): produces `charades_dinov3_vitl16_w16_24fps_combined/`
4. **Train 2A.1** (mean patch only, `in_feat_dim=1024`)
5. **Train 2A.2** (CLS + mean patch, `in_feat_dim=2048`)
6. **Analyse per-class AP profiles** and compare against CLS-only baseline

### 10.2 Decision Points

After completing 2A.1 and 2A.2, two key decisions will be made:

**Decision 1**: Which DINOv3 representation to use as the visual baseline for skeleton fusion (Phase 2B.2)?
- If 2A.2 significantly outperforms 2A.0: use CLS + mean patch as the visual stream in all subsequent fusion experiments
- If 2A.2 provides marginal improvement: continue with CLS-only to avoid the additional 4096-dim computation overhead

**Decision 2**: Whether to pursue attention-weighted pooling (2A.3/2A.4)?
- Only if 2A.1 significantly outperforms 2A.0 AND there is meaningful gap between 2A.1 and 2A.2, suggesting that spatial selectivity matters

### 10.3 Connection to the Full Multi-Modal Pipeline

Phase 2A and Phase 2B together establish the individual stream baselines that feed into the eventual multi-modal fusion (Phase 2C):

```
Phase 2A: DINOv3 dense → best DINOv3 representation R_dino
Phase 2B: Skeleton     → best skeleton representation R_skel (4096-dim, window-aligned)

Phase 2C: CLIP (512) + R_dino (1024 or 2048) + R_skel (4096)
          → Gated fusion or Cross-attention
          → MS-Temba temporal processing
          → Classification (157 classes)
```

The expected cumulative ΔmAP from combining all improvements:

| Configuration | Expected mAP | ΔmAP vs CLIP baseline |
|---------------|:------------:|:---------------------:|
| CLIP baseline (Phase 1) | 32.40 | — |
| DINOv3 dense alone (Phase 2A) | ~27–30 | below CLIP |
| CLIP + skeleton (Phase 2B) | ~33.4–36.4 | +1.0 to +4.0 |
| CLIP + DINOv3 dense + skeleton (Phase 2C) | ~34–38 | +1.6 to +5.6 |

### 10.4 Documentation Plan

| Document | Content | Status |
|----------|---------|:------:|
| `phase1_baseline_clip_dino.md` | Baseline, regularisation, full comparison | ✅ |
| `phase2a_dino_dense_features.md` | This document — dense feature design and plan | ✅ |
| `phase2a_dino_dense_results.md` | Training results for 2A.1 and 2A.2 | ⏳ |
| `phase2b_skeleton_extraction.md` | SCDNet extraction and alignment | ✅ |
| `phase2b_skeleton_singlestream.md` | Single-stream skeleton training results | ⏳ |
| `phase2b_skeleton_fusion.md` | Early fusion, gated fusion, cross-attention ablation | ⏳ |
| `phase2c_multimodal_fusion.md` | CLIP + DINOv3 dense + skeleton combined | ⏳ |

---

## Appendix A — DINOv3 Feature Dimension Reference

| Pooling strategy | Output dim | Storage (9848 videos, ~50 frames) | `in_feat_dim` |
|------------------|:----------:|:---------------------------------:|:-------------:|
| CLS only | 1024 | ~8.2 GB | 1024 |
| Mean patch only | 1024 | ~8.2 GB | 1024 |
| CLS + mean patch | 2048 | ~16.4 GB | 2048 |
| Attention pooling | 1024 | ~8.2 GB | 1024 |
| CLS + attn pooling | 2048 | ~16.4 GB | 2048 |

## Appendix B — Existing Feature Directories

| Directory | Shape | Status |
|-----------|-------|:------:|
| `charades_dinov3_vitl16_w16_24fps/` | [N, 1024] float32 | ✅ exists (CLS-only) |
| `charades_dinov3_vitl16_w16_24fps_meanpatch/` | [N, 1024] float32 | ⏳ to extract |
| `charades_dinov3_vitl16_w16_24fps_combined/` | [N, 2048] float32 | ⏳ to extract |

---
