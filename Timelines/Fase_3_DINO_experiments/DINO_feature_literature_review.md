# MS-Temba on Charades: Literature Review and Implementation Analysis
### DINOv3 Dense Feature Adaptation and CLIP–DINOv3 Fusion Strategies

> **Author**: Matteo Di Iorio  
> **Period**: March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Document**: Phase 2A supplementary — literature review, cross-paper comparison, detailed fusion implementation  
> **Companion documents**: `phase2a_dino_dense_features.md` (technical plan), `deep_mstemba_clip_dino_comparison_updated.md` (Phase 1 results)

---

## Table of Contents

1. [Introduction: The Two Fundamental Problems](#1-introduction-the-two-fundamental-problems)
2. [DINOv3 Adaptation for Temporal Action Detection](#2-dinov3-adaptation-for-temporal-action-detection)
3. [Literature: Fusing CLIP and DINOv2/v3](#3-literature-fusing-clip-and-dinov2v3)
4. [Detailed Fusion Architectures for MS-Temba](#4-detailed-fusion-architectures-for-ms-temba)
5. [Implementation Details for Each Strategy](#5-implementation-details-for-each-strategy)
6. [Ablation Design Grounded in the Literature](#6-ablation-design-grounded-in-the-literature)
7. [Expected Contributions per Strategy](#7-expected-contributions-per-strategy)
8. [References](#8-references)

---

## 1. Introduction: The Two Fundamental Problems

The Phase 1 results established that CLIP and DINOv3 CLS-only exhibit a 7.19 mAP gap on Charades (32.40 vs 25.21), with qualitatively different per-class profiles. The analysis traced this gap to two orthogonal problems that Phase 2A addresses:

**Problem 1 — DINOv3 is not being used as designed.** DINOv3's primary competitive advantage over CLIP — and indeed the central contribution of the DINOv3 work — is the quality of its **patch-level dense representations**, preserved through Gram anchoring. The current pipeline uses only the CLS token, discarding 196 out of 197 output tokens per frame (~99.5% of available information). This is analogous to using a depth estimation model and discarding the depth map in favour of a single global scalar. Adapting DINOv3 for MS-Temba requires extracting and integrating patch features.

**Problem 2 — CLIP and DINOv3 encode fundamentally different but complementary information.** CLIP encodes semantic-linguistic priors; DINOv3 encodes spatial-structural geometry. On Charades, they perform best on disjoint class subsets: CLIP dominates on semantically descriptive classes (*cooking*, *talking on phone*, AP > 60), DINOv3 leads on spatially specific object manipulations (*refrigerator*, *vacuum*, *broom*, Δ > 20 AP). A fusion strategy that combines both encoders stands to outperform either individually on a much larger fraction of the 157 classes.

These two problems have been studied extensively in the broader vision literature, but not in the specific context of temporal action detection with Mamba-based architectures. This document reviews the most relevant prior work and derives concrete, literature-grounded implementation strategies for MS-Temba.

---

## 2. DINOv3 Adaptation for Temporal Action Detection

### 2.1 DINOv2 on Video: Evidence from the Original Paper

The DINOv2 paper (Oquab et al., 2023; arXiv:2304.07193) evaluates its features on three video action recognition benchmarks — UCF-101, Kinetics-400, and Something-Something v2 (SSv2) — despite the model being trained exclusively on static images. The evaluation protocol is instructive: for UCF-101 and Kinetics-400, 8 evenly spaced frames are selected from each video and a linear classifier is trained on the average of their features; for SSv2, concatenation is preferred over averaging to retain more temporal information. DINOv2 matches OpenCLIP on UCF and Kinetics with a +2.5% advantage, and clearly outperforms it on dense prediction tasks where OpenCLIP features show many artifacts.

This result has three direct implications for the MS-Temba adaptation:

**Implication 1**: DINOv2 features transfer well to video without any video-specific training — a pure frame-level extractor with temporal pooling suffices for action recognition, validating the approach already used in Phase 1.

**Implication 2**: the performance gap between averaging and concatenation on SSv2 (a dataset that requires temporal reasoning rather than appearance matching) supports the hypothesis that the temporal ordering of features carries discriminative information that averaging destroys. This is directly handled by MS-Temba's temporal SSM architecture, which processes the full feature sequence rather than averaging it.

**Implication 3**: on segmentation, DINOv2 produces dense features that are quantitatively and qualitatively superior to CLIP. This is the key result motivating the Phase 2A patch-level extraction: the same property that makes DINOv2 superior on segmentation — localised, spatially coherent patch representations — is expected to benefit the spatial action classes in Charades.

### 2.2 Temporal DINO: Self-Supervised Temporal Extension

Teeti et al. (2023; arXiv:2308.04589) propose **Temporal-DINO**, a self-supervised video strategy that extends the DINO student-teacher framework to temporal sequences. The approach employs two models: a student processing past frames and a teacher processing both past and future frames, enabling a broader temporal context. During training, the teacher guides the student to learn future context by observing only past frames.

While Temporal-DINO is not directly applicable to the MS-Temba setup (it requires retraining the backbone, which is not feasible given resource constraints), it provides a conceptual grounding for why DINOv3 features, though extracted frame-by-frame, can be processed temporally by MS-Temba's SSMs. The temporal SSM blocks in MS-Temba effectively implement a form of sequential distillation: each frame's state is conditioned on all preceding frame states through the hidden state recurrence. The role of the student in Temporal-DINO is analogous to MS-Temba's classifier: it learns to predict the correct class from a causally ordered sequence of feature representations, guided by the supervision signal from the action labels.

The key difference is that Temporal-DINO operates in the feature space of DINO itself, while MS-Temba operates on pre-extracted features that are fixed. This makes the feature quality — and specifically, whether patch information is included — all the more critical: the temporal model cannot compensate for information that was discarded at extraction time.

### 2.3 DINO-World: DINOv2 Patch Tokens as Temporal State Space

The DINO-World model (arXiv:2507.19468) provides the most direct evidence for why patch tokens — rather than the CLS token — are the appropriate representation for temporal modelling. DINO-World is a latent-space world model trained to predict future frames in the latent space of DINOv2. It leverages a frozen DINOv2 encoder and trains a future predictor on a large-scale uncurated video dataset; crucially, the predictor operates on patch tokens in the latent space, not on the CLS token.

This design choice is theoretically motivated: the CLS token, by construction, aggregates all spatial information into a single vector and loses the spatial structure that makes DINOv2 distinctive. Patch tokens, on the other hand, are spatially indexed: patch token h_i encodes the visual content of a specific 16×16 region of the image, and the temporal evolution of a specific patch's content is therefore a localised, interpretable signal. For a video of someone opening a refrigerator, the patch tokens covering the refrigerator handle will evolve distinctively from frame to frame in a way that the CLS token does not isolate.

The practical adaptation for MS-Temba is straightforward: mean-pooling the 196 patch tokens into a single 1024-dim vector before temporal processing is a computationally tractable approximation of the full patch-level representation. It sacrifices spatial indexing but preserves the average content of all spatial regions — a useful intermediate between CLS-only and full 196-token processing (which would be computationally prohibitive as a stored feature representation).

### 2.4 The Frame-to-Sequence Adaptation Challenge

A recurring challenge in adapting image-trained DINOv3 to video is the mismatch between the spatial processing of the ViT and the temporal processing required for action detection. DINOv3 ViT-L/16 processes a single 224×224 frame as a 14×14 grid of non-overlapping patches, with no temporal dimension. MS-Temba's SSM blocks, conversely, process sequences of feature vectors over time.

The adaptation in MS-Temba's current design handles this by treating the temporal sequence of extracted per-frame features as the input sequence to the SSM. This is the same paradigm used by Video Mamba Suite and other Mamba-based video architectures for action localization, cited in the MS-Temba paper itself. MS-Temba's architecture is composed of a Visual Backbone, a Temporal Encoder consisting of Temporal Mamba Blocks, a Temporal Mamba Fuser, and a Classification Head. Closest to the MS-Temba approach, Video Mamba Suite performs action localization but relies on frame sampling and compressing all information within the architecture, limiting it to videos around 3 minutes — while MS-Temba targets untrimmed videos exceeding 40 minutes.

The key insight is that MS-Temba's temporal processing is **backbone-agnostic**: the architecture downstream of the `InputProjection` block does not care whether the incoming features are from CLIP, DINOv3 CLS, or DINOv3 dense. What matters is that the `InputProjection` correctly maps the input dimensionality to the internal 256-dim space. This architectural property — the complete decoupling of feature extraction from temporal modelling — is precisely what makes Phase 2A tractable: switching from CLS (1024-dim) to CLS + mean-patch (2048-dim) requires only updating `in_feat_dim` in the input projection.

---

## 3. Literature: Fusing CLIP and DINOv2/v3

### 3.1 The Complementarity Hypothesis

The central empirical motivation for fusing CLIP and DINOv3 is that they encode complementary information. This complementarity has been extensively documented in the literature, though in contexts different from temporal action detection. Understanding the mechanism of complementarity — rather than simply observing it — is essential for designing effective fusion strategies.

**CLIP encodes**: global semantic-linguistic structure, concept-level information (what the scene "means"), cross-modal compatibility with text descriptions, and object-level identity (what objects are present).

**DINOv3 encodes**: spatial-structural geometry (where objects are and how they relate), local texture and appearance details, patch-to-patch relational consistency (preserved by Gram anchoring), and scene composition (how the scene is organised).

These two information types are, in the vocabulary of information theory, largely non-redundant: knowing the semantic label of a scene does not specify its spatial layout, and knowing the spatial layout does not specify the semantic label. This non-redundancy is the theoretical basis for expecting fusion gains.

### 3.2 COMM: The First Systematic CLIP + DINOv2 Fusion Study

The most directly relevant work is **COMM** (Jiang et al., 2023/2024; arXiv:2310.08825), which presents the first systematic investigation of combining CLIP and DINOv2 as dual visual encoders. Considering fine-grained pixel information in DINOv2 and global semantic information in CLIP, COMM proposes to fuse the visual embeddings of these two models to enhance visual capabilities. Surprisingly, when equipped with an MLP layer for alignment, the vision-only model DINOv2 shows promise as a visual branch, attributed to the fine-grained localisation information captured by DINOv2. The analysis shows that shallow layer features of CLIP offer particular advantages for fine-grained tasks such as grounding and region understanding, while deep features from DINOv2 provide richer pixel-level features.

The COMM architecture (Figure 4 in the paper) is particularly relevant: features from both encoders are independently extracted, a Multi-Level Feature Merging (MFM) module combines shallow and deep features within each encoder, and then the DINOv2 features are aligned with an MLP before being concatenated with the CLIP features. The concatenated representation is then passed through a linear layer before being fed to the downstream model.

**Direct applicability to MS-Temba**: the COMM architecture maps almost exactly to the proposed Phase 2A.2 strategy (CLS + mean-patch concatenation) — the main difference being that COMM combines features from two separate encoder runs, while the Phase 2A.2 approach concatenates CLS and mean-patch from a single DINOv3 forward pass. The fusion of CLIP and DINOv3 (Phase 2A.4, discussed below) follows the COMM paradigm more closely.

**Key finding from COMM**: the merging of shallow features from DINOv2 leads to significant performance degradation (worse than deep features only), while for CLIP, shallow features are beneficial. This asymmetry reflects the training objectives: CLIP's shallow layers encode fine-grained structural details useful for grounding, while DINOv2's shallow layers lack sufficient semantic information and should not be merged.

For MS-Temba, this finding implies that when using DINOv3 features, only the **final layer's** patch tokens should be used (as in the Phase 2A.2 design), not intermediate layer activations. The intermediate layer activations of DINOv3 would provide lower-quality patch representations that could degrade performance.

### 3.3 Talk2DINO: Aligning CLIP Text Embeddings to DINOv2 Patch Space

**Talk2DINO** (Barsellotti et al., 2024; arXiv:2411.19331) approaches the CLIP–DINOv2 complementarity from a different angle: instead of fusing features in a shared representation space, it learns a mapping from CLIP's text embedding space into DINOv2's patch embedding space. Talk2DINO aligns the textual embeddings of CLIP to the patch-level features of DINOv2 through a learned mapping function without the need to fine-tune the underlying backbones. At training time, DINOv2 attention maps are exploited to selectively align local visual patches with textual embeddings, using the head that best aligns with the provided caption.

The significance for the MS-Temba context is conceptual rather than directly implementable. Talk2DINO demonstrates that **DINOv2's patch space is semantically structurable**: given the right learning signal, the patch tokens can be made compatible with text-level semantic categories. This is precisely what the Phase 2A adaptation attempts to do implicitly — by training MS-Temba's classification head on top of DINOv3 patch features, the model is effectively learning which dimensions of the patch feature space are relevant for each of the 157 Charades classes.

The key difference from Talk2DINO is that the alignment is supervised by action classification labels rather than text-image contrastive loss. This is potentially less precise — the labels are coarser than image captions — but is directly optimised for the Charades classification objective, which is ultimately what we care about.

**Implication for DINOv3 × CLIP fusion**: Talk2DINO's attention-weighted alignment mechanism suggests that when fusing CLIP and DINOv3 features, the attention maps of DINOv3 can serve as a natural weighting mechanism — patches with high attention scores are more semantically salient and should receive higher weight in the aggregated representation. This directly motivates the attention-weighted pooling strategy (Approach 3 in Phase 2A).

### 3.4 dino.txt: CLS + Patch Average as the Optimal DINOv2 Representation

**dino.txt** (Jose et al., CVPR 2025; arXiv:2412.16334) provides the most direct empirical support for the Phase 2A.2 design (CLS + mean-patch concatenation). dino.txt successfully trains a CLIP-like model on top of a frozen DINOv2 backbone by proposing key ingredients to improve performance on both global and dense tasks, including concatenating the [CLS] token with the patch average to train the alignment, along with curating data using both text and image modalities.

The central finding of dino.txt is that neither CLS alone nor patch average alone produces the best results on tasks requiring both global semantic understanding and dense spatial discrimination — it is their concatenation that achieves state-of-the-art on zero-shot classification (global task) and open-vocabulary segmentation (dense task) simultaneously. The method proposes concatenating the [CLS] token with the patch average to train the alignment and curating data using both text and image modalities.

This result directly validates the Phase 2A.2 hypothesis: concatenating CLS and mean-patch provides a 2048-dim representation that is superior to either component alone for tasks spanning both global and local discrimination. The MS-Temba classification objective spans both types: global classes like *cooking* (semantically defined, global appearance) and local classes like *holding a vacuum* (spatially defined, requires encoding object shape) — making the combined representation theoretically optimal.

**Quantitative evidence from dino.txt**: the paper reports that on ImageNet-1k zero-shot classification, the CLS + patch average combination outperforms CLS-only by 1.8% top-1 accuracy with the same training. On ADE20K open-vocabulary segmentation (a dense task), the improvement is 3.2% mIoU. The gains are larger on dense tasks, consistent with the hypothesis that patch information contributes most to spatially-specific discrimination.

### 3.5 COMM Follow-up: DM-Fuse and Multi-Level Features

A recent medical imaging extension of the COMM dual-encoder paradigm — DM-Fuse (2025) — further validates the approach and provides additional architectural insights. DM-Fuse integrates multi-level features from two complementary vision encoders (CLIP and DINOv2) to mitigate the loss of fine-grained details and visual bias. For each encoder, features are extracted from multiple layers and categorised into shallow and deep groups representing fine-grained and global abstract representations respectively. CLIP trained via image-text contrastive learning prioritises cross-modal consistency and object-level features, whereas DINOv2 emphasises intrinsic image structures, excelling at capturing textures, contours, and local relational patterns.

The DM-Fuse paper explicitly codifies the complementarity principle: CLIP captures **cross-modal semantic consistency** while DINOv2 captures **intrinsic structural properties**. Applied to Charades: CLIP is appropriate for classes whose discrimination depends on semantic labelling (*cooking*, *talking on phone*), while DINOv3 is appropriate for classes whose discrimination depends on structural properties (*opening a refrigerator*, *holding a vacuum*). A fusion mechanism that can adaptively weight these contributions per frame and per class is therefore the ideal architecture.

### 3.6 Gated Fusion in Mamba-Based Dual-Stream Architectures

The Gated Class Token Fusion mechanism from **Dual Branch VideoMamba** (arXiv:2506.03162) provides a direct architectural precedent for gated fusion in Mamba-based video models. The paper introduces a Gated Class Token Fusion mechanism that combines information between two parallel Mamba branches, performed at each layer in the network to provide a form of continuous fusion. The two branches use distinct scanning strategies, with the intention of separately extracting spatial and temporal features from video inputs.

While the Dual Branch VideoMamba fuses spatial and temporal Mamba branches (rather than CLIP and DINOv3 streams), the architectural principle is directly transferable: a per-position gate controls the mixing of two parallel feature streams, and the gate is computed from the concatenation of both streams. Applied to CLIP + DINOv3 fusion in MS-Temba:

```
At each temporal position t:
  f_clip_t  = CLIP feature at time t        [768-dim]
  f_dino_t  = DINOv3 dense feature at t     [2048-dim]
  
  proj_clip = Linear(768 → 256)(f_clip_t)   [256-dim]
  proj_dino = Linear(2048 → 256)(f_dino_t)  [256-dim]
  
  gate_t    = σ(W_g · [proj_clip; proj_dino]) [256-dim]
  
  h_t       = gate_t ⊙ proj_clip + (1 - gate_t) ⊙ proj_dino
```

The gate is a learnable sigmoid function that interpolates between the two projected representations at each temporal position and each feature dimension. This allows the model to learn, for example, that at a frame showing someone holding a broom, the DINOv3 dimensions encoding the broom's elongated shape should dominate, while at a frame showing someone talking on the phone, the CLIP dimensions encoding the semantic concept should dominate.

### 3.7 Fusion-Mamba: Cross-Modal Fusion in Hidden State Space

**Fusion-Mamba** (arXiv:2404.09146) provides a more sophisticated alternative: instead of fusing features before the SSM, it performs fusion within the SSM's hidden state space. Fusion-Mamba introduces a carefully designed Mamba-based structure to integrate cross-modality features in a hidden state space. The mapping-based deep feature fusion method effectively reduces spatial disparities through dual-direction gated attention, which suppresses redundant features and captures complementary information among modalities.

The key insight is that fusion within the hidden state allows the temporal dynamics of one modality to condition the temporal dynamics of the other — a form of cross-modal temporal attention that is not possible with pre-SSM concatenation. For the CLIP + DINOv3 case, this would mean that the temporal evolution of CLIP features (which captures semantic-level scene changes) could gate the temporal evolution of DINOv3 features (which captures structural changes). This is architecturally richer than gated input fusion but also more complex to implement and more susceptible to overfitting on Charades's limited training data.

**Recommendation**: Fusion-Mamba-style in-state-space fusion is reserved for a potential Phase 2C experiment, as it requires significant architectural changes to MS-Temba's SSM blocks. The gated input fusion described above is the recommended starting point.

---

## 4. Detailed Fusion Architectures for MS-Temba

### 4.1 Taxonomy of Fusion Points

Before detailing each strategy, it is useful to establish where in the MS-Temba pipeline fusion can occur:

```
                    MS-Temba Pipeline
                    
[Features on disk]
       │
       ▼
  DataLoader         ← POINT A: Feature concatenation at load time
       │
       ▼
  InputProjection    ← POINT B: Projection layer fusion (separate projectors per modality)
       │
       ▼
  Block 1 (SSM)      ← POINT C: Gated fusion between projected representations
       │
       ▼
  Block 2 (SSM×2)    ← POINT D: Cross-attention between block outputs
       │
       ▼
  Block 3 (SSM×3)
       │
       ▼
  Interaction Block
       │
       ▼
  Classifier Head
```

Each fusion point entails different tradeoffs:
- **Point A**: maximum simplicity, but forces a single linear projection to learn from a heterogeneous concatenated input
- **Point B**: separate projectors for each modality, allowing modality-specific dimensionality reduction before fusion
- **Point C**: fusion in the 256-dim projected space, with a lightweight gate
- **Point D**: deep fusion with cross-attention, maximum expressivity but highest overfitting risk

### 4.2 Strategy 1 — Feature-Level Concatenation (Point A)

The simplest strategy: concatenate CLIP and DINOv3 features at load time and treat the result as a single high-dimensional input.

**Architecture**:
```
f_clip  [N, 768]
f_dino  [N, 2048]  (CLS + mean-patch)
           │
      Concatenate along feature dim
           │
      f_fused [N, 2816]
           │
   InputProjection: Linear(2816 → 256)
           │
   MS-Temba blocks (unchanged)
```

**Codebase changes**:
- `charades_dataloader.py`: load both feature files, concatenate along dim=1
- `MSTemba_main.py`: add `--backbone clip_dino_concat` with `in_feat_dim=2816`
- `models_MSTemba.py`: no changes (InputProjection already accepts arbitrary `in_feat_dim`)

**Regularisation requirements**: `Linear(2816 → 256)` has 2816×256+256 = 721,152 parameters. With 7985 training videos, this is a high-risk projection. Strong regularisation is essential: `drop=0.15, drop_path=0.1, weight_decay=0.05`.

**Advantage**: zero architectural complexity, serves as the lower bound for more sophisticated fusion.  
**Disadvantage**: single projection must simultaneously learn to compress semantic (CLIP) and spatial (DINOv3) features into 256 dimensions, which may conflict.

### 4.3 Strategy 2 — Dual Projector with Additive Fusion (Point B)

Two separate projectors, one per modality, with additive (sum) combination:

**Architecture**:
```
f_clip  [N, 768]  →  proj_clip: Linear(768 → 256)  →  h_clip  [N, 256]
f_dino  [N, 2048] →  proj_dino: Linear(2048 → 256) →  h_dino  [N, 256]
                                                              │
                                                        h = h_clip + h_dino
                                                              │
                                                   MS-Temba blocks (unchanged)
```

**Codebase changes** (in `models_MSTemba.py`):
```python
class DualProjection(nn.Module):
    def __init__(self, clip_dim=768, dino_dim=2048, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_clip = nn.Sequential(
            nn.Linear(clip_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(dino_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate)
        )
    
    def forward(self, f_clip, f_dino):
        return self.proj_clip(f_clip) + self.proj_dino(f_dino)
```

**Parameter count**: 768×256+256 + 2048×256+256 = 197,376 + 524,544 = **721,920 parameters** — slightly higher than concatenation but with much better gradient flow, since each projector optimises independently for its own modality.

**Advantage**: each modality is projected in a specialised subspace before fusion, reducing cross-modal interference.  
**Disadvantage**: additive fusion treats both modalities symmetrically at every temporal position, without adaptive weighting.

### 4.4 Strategy 3 — Gated Dual Projector (Point B–C)

The recommended production strategy, implementing the gated fusion principle from COMM and Dual Branch VideoMamba. The gate learns to weight each modality's contribution per temporal position and per feature dimension:

**Architecture**:
```
f_clip  [N, 768]  →  proj_clip [N, 256]  ─────────────────┐
f_dino  [N, 2048] →  proj_dino [N, 256]  ─────────────────┤
                                                           │
                                gate_input = [proj_clip; proj_dino] [N, 512]
                                g = σ(W_g · gate_input)   [N, 256]
                                                           │
                                h = g ⊙ proj_clip + (1 − g) ⊙ proj_dino
                                                           │
                                           MS-Temba blocks (unchanged)
```

**Implementation** (in `models_MSTemba.py`):
```python
class GatedDualProjection(nn.Module):
    def __init__(self, clip_dim=768, dino_dim=2048, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_clip = nn.Sequential(
            nn.Linear(clip_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate)
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(dino_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate)
        )
        # Gate: takes concatenation of both projections [2*out_dim → out_dim]
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, f_clip, f_dino):
        h_c = self.proj_clip(f_clip)          # [B, T, 256]
        h_d = self.proj_dino(f_dino)          # [B, T, 256]
        g   = self.gate(torch.cat([h_c, h_d], dim=-1))   # [B, T, 256]
        return g * h_c + (1.0 - g) * h_d     # [B, T, 256]
```

**Parameter count**: 197,376 (proj_clip) + 524,544 (proj_dino) + 256×2×256+256 = 131,328 (gate) = **853,248 total** — a 5% increase over the single-stream CLIP baseline (~18M parameters). This is negligible.

**Theoretical justification**: the gate g ∈ (0,1)^256 operates independently per feature dimension, allowing the model to learn that some dimensions should always come from CLIP (e.g., semantic class-discriminative dimensions), others from DINOv3 (e.g., spatial configuration dimensions), and others from a weighted combination. This dimension-wise gating is strictly more expressive than scalar gating (one gate per temporal position) or channel-wise gating (one gate per feature map).

**Initialisation strategy**: the gate should be initialised to g ≈ 0.5 (uniform mixing). This is achieved by initialising W_g with small weights (~1e-3), ensuring that at the start of training both modalities contribute equally and the gate specialises gradually. Initialising g ≈ 1 (full CLIP) or g ≈ 0 (full DINOv3) would bias the gradient flow toward the initialised modality and slow down learning.

### 4.5 Strategy 4 — Cross-Attention Fusion (Point D)

The most expressive fusion strategy, directly inspired by the Talk2DINO attention-alignment mechanism. CLIP features serve as **Query** (seeking semantic grounding), DINOv3 patch features serve as **Key** and **Value** (providing spatial content):

**Architecture**:
```
f_clip  [N, 768]  →  proj_Q: Linear(768 → 256)  →  Q  [N, 256]
f_dino  [N, 2048] →  proj_K: Linear(2048 → 256) →  K  [N, 256]
                  →  proj_V: Linear(2048 → 256) →  V  [N, 256]

Attention weights: A = softmax(QKᵀ / √256)  [N, N]
Attended output:   Z = AV                    [N, 256]
Residual:          h = LayerNorm(Z + Q)      [N, 256]
                              │
                  MS-Temba blocks (unchanged)
```

**Semantic interpretation**: the attention mechanism asks "given the semantic concept that CLIP identifies in this frame (Q), which spatial regions of the DINOv3 representation (K, V) are most relevant?" On a frame of someone holding a vacuum, CLIP's Query would attend to the DINOv3 Key dimensions encoding the vacuum's shape and the hand's position, extracting those dimensions as the Value. On a frame of someone talking on the phone, CLIP's Query would attend to the DINOv3 Key dimensions encoding the hand-head configuration.

**Parameter count**: 768×256 + 2×(2048×256) = 196,608 + 1,048,576 = **1,245,184** — a ~7% increase over baseline. Still well within acceptable bounds.

**Overfitting risk**: cross-attention has N² computational complexity and much larger effective capacity than gated fusion. With only 7985 training videos, this is a higher overfitting risk than the gated strategy. Recommended only if gated fusion shows clear saturation.

---

## 5. Implementation Details for Each Strategy

### 5.1 Dataloader Changes: Loading Two Feature Modalities

All multi-modal strategies require loading two feature files per video. The current `charades_dataloader.py` loads a single `.npy` file from `rgb_root`. The proposed modification adds a second optional feature directory:

```python
class CharadesDataset(Dataset):
    def __init__(self, ..., rgb_root, dino_root=None, ...):
        self.dino_root = dino_root
        ...
    
    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        
        # Primary features (CLIP)
        clip_feat = np.load(os.path.join(self.rgb_root, f"{vid_id}.npy"))
        
        # Secondary features (DINOv3 dense) — optional
        if self.dino_root is not None:
            dino_feat = np.load(os.path.join(self.dino_root, f"{vid_id}.npy"))
        else:
            dino_feat = None
        
        # Temporal pooling to num_clips=256 (unchanged logic)
        clip_feat = self._temporal_pool(clip_feat)
        if dino_feat is not None:
            dino_feat = self._temporal_pool(dino_feat)
        
        return clip_feat, dino_feat, labels, ...
```

**Handling mismatched feature counts**: although Phase 2A features are extracted with the same pipeline and should be aligned, a defensive check ensures that `|T_clip - T_dino| <= 1`. If the mismatch exceeds 1 (analogous to the `5UNDJ` skeleton anomaly), the secondary features are replaced with zeros and a warning is logged.

### 5.2 MSTemba_main.py Changes

Addition of new backbone identifiers and corresponding routing logic:

```python
# In the backbone → configuration mapping:
fusion_configs = {
    'clip':                  {'in_feat_dim': 768,  'fusion': 'single'},
    'dinov3_cls':            {'in_feat_dim': 1024, 'fusion': 'single'},
    'dinov3_combined':       {'in_feat_dim': 2048, 'fusion': 'single'},
    'clip_dino_concat':      {'in_feat_dim': 2816, 'fusion': 'concat'},
    'clip_dino_gated':       {'in_feat_dim': None, 'fusion': 'gated',
                              'clip_dim': 768, 'dino_dim': 2048},
    'clip_dino_crossattn':   {'in_feat_dim': None, 'fusion': 'crossattn',
                              'clip_dim': 768, 'dino_dim': 2048},
}
```

The `fusion` key selects which projection module is instantiated in `MSTemba.__init__`.

### 5.3 Models_MSTemba.py Changes

The `MSTemba.__init__` is modified to conditionally instantiate the appropriate projection module based on the `fusion` argument:

```python
class MSTemba(nn.Module):
    def __init__(self, ..., in_feat_dim=768, fusion='single',
                 clip_dim=768, dino_dim=2048, ...):
        super().__init__()
        
        if fusion == 'single':
            self.proj = nn.Sequential(
                nn.Linear(in_feat_dim, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop_rate)
            )
        elif fusion == 'gated':
            self.proj = GatedDualProjection(
                clip_dim=clip_dim, dino_dim=dino_dim,
                out_dim=256, drop_rate=drop_rate
            )
        elif fusion == 'crossattn':
            self.proj = CrossAttentionFusion(
                clip_dim=clip_dim, dino_dim=dino_dim,
                out_dim=256, drop_rate=drop_rate
            )
        
        # All downstream blocks remain unchanged
        self.block1 = ...
        ...
```

The `forward()` method requires a conditional branch:

```python
def forward(self, features, dino_features=None):
    if dino_features is not None:
        x = self.proj(features, dino_features)   # dual-input projection
    else:
        x = self.proj(features)                  # single-input projection (unchanged)
    # rest of forward: block1, block2, block3, fuser, classifier
    ...
```

This design preserves full backward compatibility: the single-stream CLIP and DINOv3 experiments from Phase 1 run without modification.

---

## 6. Ablation Design Grounded in the Literature

### 6.1 Full Experiment Matrix

Combining the Phase 2A dense feature experiments with the multi-modal fusion strategies:

| ID | Config | Backbone | `in_feat_dim` | Fusion | Lit. grounding |
|----|--------|----------|:-------------:|:------:|---------------|
| 2A.0 | DINOv3 CLS reg_v2 (baseline) | dinov3_cls | 1024 | single | Phase 1 |
| 2A.1 | DINOv3 mean patch | dinov3_meanpatch | 1024 | single | DINOv2 (Oquab et al.) |
| 2A.2 | DINOv3 CLS + mean patch | dinov3_combined | 2048 | single | dino.txt (Jose et al.) |
| 2A.3 | DINOv3 attn pooling | dinov3_attn | 1024 | single | Talk2DINO (Barsellotti et al.) |
| 2B.0 | CLIP only (baseline) | clip | 768 | single | Phase 1 |
| 2C.1 | CLIP + DINOv3 concat | clip_dino_concat | 2816 | concat | COMM baseline |
| 2C.2 | CLIP + DINOv3 additive | clip_dino_add | — | additive | COMM variant |
| 2C.3 | CLIP + DINOv3 gated | clip_dino_gated | — | gated | GCTF (VideoMamba) |
| 2C.4 | CLIP + DINOv3 cross-attn | clip_dino_xattn | — | crossattn | Talk2DINO principle |

### 6.2 Order of Execution

The experiments should be run in the following order to maximise learning from each step before committing computational resources:

**Phase 2A (DINOv3 dense features)**:
1. Run 2A.1 (mean patch) → determines if patch features alone are useful
2. Run 2A.2 (CLS + mean patch) → determines if combination helps (expected: yes, per dino.txt)
3. If 2A.1 meaningfully outperforms 2A.0: run 2A.3 (attention pooling)

**Phase 2C (CLIP + DINOv3 fusion)**:
4. Run 2C.1 (concat) → establishes if any combination gain exists
5. Run 2C.3 (gated) → the recommended production strategy
6. If 2C.3 > 2C.1 by ≥ 1 mAP: run 2C.4 (cross-attention) → explore upper bound

The concat baseline (2C.1) is essential: if concatenation does not improve over the best single-stream result, more complex fusion is unlikely to help either.

### 6.3 Diagnostic Decision Tree

```
2A.1 mean patch > 2A.0 CLS?
├── YES (+1 mAP):  patch features carry useful info beyond CLS
│   ├── 2A.2 CLS+patch > max(2A.0, 2A.1)?
│   │   ├── YES (+1 mAP): representations are complementary → USE 2A.2 for fusion
│   │   └── NO: pick the better single representation for fusion
│   └── Run 2A.3 (attention pooling) to test saliency-based filtering
└── NO:  CLS token is already optimal for MS-Temba on Charades
    └── Proceed with 2A.0 (CLS-only) for fusion experiments

2C.1 concat > best 2A result?
├── YES: fusion adds value → run 2C.3 (gated) and 2C.4 (cross-attn)
└── NO:  no fusion gain → report null result, proceed to Phase 2B (skeleton)
```

---

## 7. Expected Contributions per Strategy

### 7.1 DINOv3 Dense Features (Phase 2A)

| Strategy | Targeted class group | Expected ΔmAP | Mechanism |
|----------|---------------------|:-------------:|-----------|
| 2A.1 mean patch | Spatially specific objects | +0.5 to +2.0 | Patch content encodes object shape better than CLS |
| 2A.2 CLS + mean patch | All DINOv3-winning classes | +1.5 to +4.0 | CLS + spatial = semantic + structural |
| 2A.3 attn pooling | Salient object classes | +1.0 to +3.0 | Saliency filtering reduces background noise |

### 7.2 CLIP + DINOv3 Fusion (Phase 2C)

| Strategy | Targeted class group | Expected ΔmAP vs CLIP | Mechanism |
|----------|---------------------|:---------------------:|-----------|
| 2C.1 concat | Mixed | +0.5 to +2.0 | Lower bound: proves complementarity |
| 2C.3 gated | All 157 classes | +1.5 to +4.5 | Per-dimension adaptive weighting |
| 2C.4 cross-attn | Spatially specific + semantic | +2.0 to +5.5 | Semantic-guided spatial attention |

### 7.3 Cumulative Expected Performance

| Configuration | Expected mAP | ΔmAP vs CLIP baseline |
|---------------|:------------:|:---------------------:|
| CLIP baseline (Phase 1) | 32.40 | — |
| DINOv3 CLS-only (Phase 1) | 25.21 | −7.19 |
| DINOv3 CLS + mean patch (2A.2) | 26.7–29.2 | −3.2 to −5.7 vs CLIP |
| CLIP + DINOv3 gated (2C.3) | 33.9–36.9 | +1.5 to +4.5 |
| CLIP + DINOv3 cross-attn (2C.4) | 34.4–37.9 | +2.0 to +5.5 |
| CLIP + DINOv3 + Skeleton (Phase 2C full) | 35–39 | +2.6 to +6.6 |

The CLIP + DINOv3 gated fusion (2C.3) is the primary target: it represents the most theoretically grounded, practically implementable, and computationally feasible strategy. The expected +1.5 to +4.5 mAP gain over the CLIP baseline would bring MS-Temba substantially closer to the state-of-the-art on Charades multi-label temporal action detection.

---

## 8. References

The following papers are cited in this document, ordered by relevance to the MS-Temba adaptation:

**Core DINOv2/v3 references**:
- Oquab et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR. arXiv:2304.07193.
- Jose et al. (CVPR 2025). *DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment*. arXiv:2412.16334.
- Teeti et al. (2023). *Temporal DINO: A Self-Supervised Video Strategy to Enhance Action Prediction*. arXiv:2308.04589.
- *Back to the Features: DINO as a Foundation for Video World Models* (DINO-World). arXiv:2507.19468.

**CLIP + DINOv2 fusion**:
- Jiang et al. (2023/2024). *From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models* (COMM). arXiv:2310.08825.
- Barsellotti et al. (2024). *Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation* (Talk2DINO). arXiv:2411.19331.
- DM-Fuse (2025). *Enhancing Medical MLLMs with Dual Vision Encoders and MoE-based Modality Projector*. ScienceDirect.

**Gated and cross-modal fusion in Mamba architectures**:
- *Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection*. arXiv:2506.03162.
- *Fusion-Mamba: Cross-Modal Fusion in Hidden State Space*. arXiv:2404.09146.

**MS-Temba original paper**:
- Pramanik et al. (2025). *MS-Temba: Multi-Scale Temporal Mamba for Efficient Temporal Action Detection*. arXiv:2501.06138.

---