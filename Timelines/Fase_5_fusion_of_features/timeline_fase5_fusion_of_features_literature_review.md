# MS-Temba on Charades: Literature Review and Implementation Analysis
### Skeleton Feature Adaptation, Visual–Kinematic Fusion, and State of the Art

> **Author**: Matteo Di Iorio  
> **Period**: March 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — esterel nodes  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Document**: Phase 2B supplementary — skeleton feature state of the art, adaptation to MS-Temba, fusion strategies  
> **Companion documents**: `phase2b_skeleton_extraction.md` (extraction pipeline), `mstemba_phase2a_dino_clip_literature_fusion.md` (CLIP–DINOv3 fusion)  
> **Status**: Initial version — to be updated upon completion of single-stream and fusion experiments

---

## Table of Contents

1. [Introduction and Problem Framing](#1-introduction-and-problem-framing)
2. [Skeleton Representations: A Taxonomic Overview](#2-skeleton-representations-a-taxonomic-overview)
3. [SCD-Net: The Source Backbone](#3-scd-net-the-source-backbone)
4. [From Raw Skeleton to GNN Embedding: The Representation Pipeline](#4-from-raw-skeleton-to-gnn-embedding-the-representation-pipeline)
5. [The Complementarity of Skeleton and Visual Features](#5-the-complementarity-of-skeleton-and-visual-features)
6. [Literature: Visual–Skeleton Fusion Architectures](#6-literature-visualskeleton-fusion-architectures)
7. [Literature: Skeleton + CLIP Fusion](#7-literature-skeleton--clip-fusion)
8. [Detailed Fusion Architectures for MS-Temba](#8-detailed-fusion-architectures-for-ms-temba)
9. [Implementation Details](#9-implementation-details)
10. [Ablation Design and Experiment Matrix](#10-ablation-design-and-experiment-matrix)
11. [Expected Impact per Strategy](#11-expected-impact-per-strategy)
12. [Limitations and Open Questions](#12-limitations-and-open-questions)
13. [References](#13-references)

---

## 1. Introduction and Problem Framing

### 1.1 Why Skeleton Features Belong in the MS-Temba Pipeline

The Phase 1 per-class analysis identified a persistent failure pattern that neither CLIP nor DINOv3 can resolve through improved visual representations alone: a subset of Charades classes whose discrimination depends fundamentally on **body kinematics** — how the body moves, not what it looks like or what object is present. These classes fall into two identifiable groups:

**Group A — State/direction verbs**: actions defined by the trajectory or direction of a body part, where the semantic difference lies in a subtle kinematic distinction invisible in a single visual frame. The archetypal cases are *Turning on a light* (c104, AP 8.7) vs *Turning off a light* (c105, AP 2.3): both are visually near-identical in a CLS token or patch feature, but the direction of wrist motion before and after the key frame is the defining cue.

**Group B — Postural transitions**: actions requiring detection of a temporal change in body configuration rather than a static pose. *Someone is sneezing* (c153, AP 17.8) requires detecting the characteristic rapid trunk flexion and head snap sequence; *Someone is standing up* (c154, AP 36.8) requires detecting upward CoM displacement with concurrent knee and hip extension.

For Group A classes, even the best possible visual feature cannot resolve the direction ambiguity — the critical information is in the motion trajectory, not in the appearance of any single frame. For Group B classes, while a very high quality visual backbone might partially infer the transition from appearance cues, the kinematic trajectory provides a more direct and reliable signal.

The skeleton features extracted via SCD-Net provide exactly this kinematic information, encoded in a 4096-dimensional GNN embedding that captures both the spatial configuration of the skeleton graph at each frame and the temporal evolution of that configuration across frames.

### 1.2 The Specific Challenge: Adapting GNN Embeddings to MS-Temba

The adaptation challenge is fundamentally different from the DINOv3 dense feature adaptation. DINOv3 and MS-Temba share the same paradigm — both are frame-level, appearance-based representations — and the only obstacle was recovering the spatially structured patch features that were discarded in the CLS-only pipeline.

Skeleton features, by contrast, represent a **fundamentally different modality**: they encode body geometry in a graph-structured space, they are extracted by a model with a completely different inductive bias (GNN graph convolution vs ViT self-attention), and they carry no information about object appearance, scene context, or semantic labels. The fusion challenge is therefore not about recovering discarded information but about **bridging two orthogonal representation spaces** — one semantic-visual, one kinematic-structural.

### 1.3 Relationship to the DINOv3 Fusion Work

The skeleton fusion work is related to but distinct from the CLIP–DINOv3 fusion described in the Phase 2A companion document. The key differences in fusion design philosophy are:

| Property | CLIP + DINOv3 | CLIP/DINOv3 + Skeleton |
|----------|:-------------:|:----------------------:|
| Modality relationship | Both visual, different semantics | Visual vs kinematic |
| Information overlap | High (both encode scene appearance) | Low (nearly orthogonal) |
| Temporal alignment | Identical (both from same frame) | Same window, different signal |
| Expected gate behaviour | Class-dependent mixing | Strong specialisation expected |
| Dominant failure mode | Semantic gap (DINOv3) | Missing kinematic signal (both) |
| Expected complementarity | Moderate (similar failure classes) | High (disjoint failure classes) |

This orthogonality makes skeleton fusion theoretically more powerful but also more challenging: the two representations live in very different metric spaces and the model must learn to bridge them.

---

## 2. Skeleton Representations: A Taxonomic Overview

### 2.1 Three Levels of Skeletal Representation

The skeleton literature distinguishes three increasingly abstract levels of skeletal representation, corresponding to different stages of the pose estimation and feature extraction pipeline:

**Level 1 — Raw keypoints**: 2D or 3D coordinates (x, y) or (x, y, z) of anatomical landmarks (joints), typically 17–33 per frame. Produced by pose estimators such as OpenPose, HRNet, or MediaPipe. Simple and interpretable but sensitive to camera viewpoint, occlusion, and estimation errors. The primary input to GCN-based models.

**Level 2 — Geometric representations**: derived quantities that are invariant to certain nuisance factors. Examples include bone vectors (directed vectors from parent to child joint), joint angles (computed from triplets of connected joints), and relative joint positions (joint positions minus the centre of mass). These representations can be rotation or translation-invariant depending on the derivation, and multi-stream GCN models commonly operate on several such representations simultaneously. The 2s-AGCN model (Shi et al., 2019) popularised the joint-and-bone two-stream approach.

**Level 3 — GNN/Transformer embeddings**: high-dimensional vector representations produced by the final layers of a GNN-based encoder. These encodings abstract away all explicit geometric structure and instead capture high-level motion patterns, interaction dynamics, and action-discriminative features in a dense vector space. The SCD-Net features available in this project fall into this category.

**The key implication for MS-Temba**: because the features stored in `charades_scdnet_w16/` are already Level 3 embeddings, they can be directly fed into MS-Temba's temporal processing pipeline without any graph-specific processing. The GNN has already performed the spatial graph convolution; MS-Temba's SSM blocks operate on the resulting sequence of per-frame embeddings as a standard temporal sequence.

### 2.2 Evolution of Skeleton-Based Action Recognition

**RNN-based era (2015–2018)**: early deep learning approaches treated skeleton sequences as temporal sequences of joint vectors and applied LSTMs or GRUs. The main limitation was the failure to model the spatial structure of the skeleton graph — LSTM processes a flat vector and cannot encode the tree topology of the body.

**ST-GCN (Yan et al., AAAI 2018)**: the landmark paper that introduced Spatial-Temporal Graph Convolutional Networks for skeleton-based action recognition. ST-GCN proposed to model the human body as a directed graph G = (V, E) where nodes are joints and edges are anatomically defined bone connections. The key contribution was the spatial-temporal graph convolution operation that simultaneously convolves over the spatial neighbourhood of each joint and along the temporal dimension. ST-GCN automatically learns both spatial and temporal patterns from data, moving beyond the limitations of hand-crafted traversal rules.

**2s-AGCN (Shi et al., CVPR 2019)**: extended ST-GCN to two streams (joint and bone) with an adaptive graph topology — rather than a fixed anatomical adjacency matrix, the graph topology is learned from data. The two-stream fusion at the output level showed that joint and bone streams carry complementary information.

**CTR-GCN (Chen et al., ICCV 2021)**: Channel-wise Topology Refinement Graph Convolutional Network. The key innovation was learning a separate graph topology per feature channel rather than sharing a single topology across all channels. This allows different channels to focus on different joint relationships — some channels may attend to upper-body connections, others to lower-body, etc. CTR-GCN became a widely used baseline.

**Transformer-based approaches (2021–present)**: Transformer architectures have been adapted for skeleton sequences, operating on flattened joint sequences rather than graph-structured inputs. The ST-TR model (Plizzari et al., 2021) and subsequent works show that self-attention can model long-range joint correlations that GCN's local aggregation misses. However, the lack of graph structural inductive bias often requires more data or pre-training.

**Self-supervised approaches (2022–present)**: contrastive learning frameworks (SkeletonCLR, AimCLR, SCD-Net) train skeleton encoders without action labels by defining positive/negative pairs from augmented views of the same sequence. SCD-Net (Wu et al., AAAI 2024) extends this by disentangling spatial and temporal clues, enabling more discriminative representations without supervision.

### 2.3 The Multi-Stream Paradigm

A defining characteristic of the state of the art in skeleton-based action recognition is the use of **multiple streams** processing different kinematic representations of the same sequence. The standard approach uses 4–6 streams: joint positions, bone vectors, joint velocities (first-order temporal differences of joint positions), bone velocities (first-order temporal differences of bone vectors), and sometimes acceleration terms. Each stream is processed independently by a GCN, and the final predictions are fused by score-level averaging. The multi-stream approach consistently yields 2–5% accuracy gains over any single stream, confirming that different kinematic representations carry non-redundant information.

In the MS-Temba context, the SCD-Net features are single-stream embeddings. If SCD-Net was trained using a multi-stream approach internally, the 4096-dim embedding may already integrate information from multiple kinematic representations. However, the specifics of the SCD-Net training setup on the Charades features are not documented in the available materials and would require investigation.

---

## 3. SCD-Net: The Source Backbone

### 3.1 Architecture and Training Paradigm

SCD-Net (Spatiotemporal Clues Disentanglement Network, Wu et al., AAAI 2024; arXiv:2309.05834) is a self-supervised skeleton-based action recognition framework that produces the 4096-dimensional embeddings used in this project. Its design is grounded in contrastive learning with a novel spatiotemporal disentanglement mechanism.

The core observation motivating SCD-Net is that most existing contrastive learning approaches for skeleton sequences encode spatiotemporal representations in an **entangled** manner: spatial structure (which joints are active, how the body is configured) and temporal dynamics (how the configuration evolves over time) are mixed into a single representation without explicit separation. This entanglement makes it difficult to learn discriminative features, because positive pairs (augmented views of the same sequence) must be similar in both spatial and temporal dimensions simultaneously.

SCD-Net addresses this by integrating a **decoupling module** with the feature extractor to derive explicit clues from spatial and temporal domains respectively. Formally:

```
Input: skeleton sequence X ∈ ℝ^(T × V × C)
  T = number of frames
  V = number of joints (typically 17–25)
  C = coordinate channels (3 for 3D, 2 for 2D + confidence)

Spatial extractor: f_s(X) → z_s ∈ ℝ^(D_s)  (spatial clue)
Temporal extractor: f_t(X) → z_t ∈ ℝ^(D_t) (temporal clue)

Combined embedding: z = [z_s || z_t] ∈ ℝ^(D_s + D_t)
```

The disentanglement is enforced during contrastive training: a **global anchor** is constructed from the original sequence, and the spatial/temporal clues are trained to interact with this anchor through separate contrastive objectives. This encourages the spatial clue to capture structure-relevant features and the temporal clue to capture motion-relevant features.

### 3.2 The Masking Strategy

SCD-Net incorporates a **masking strategy with structural constraints**, adapting the masked image modelling paradigm (VideoMAE) to skeleton sequences. Rather than masking random patches, the masking respects the graph topology of the skeleton: connected components of the skeleton graph are masked together, ensuring that the model cannot trivially reconstruct masked joints from immediately adjacent neighbours. This structural masking forces the model to learn higher-order motion patterns rather than local interpolation.

### 3.3 The Output Embedding

The 4096-dimensional embedding produced by SCD-Net represents a concatenation of the spatial and temporal clues after the self-attention refinement stage. This dimensionality is substantially larger than typical GCN outputs (usually 256 or 512) because SCD-Net was designed for rich representation learning rather than efficient inference. The large dimensionality reflects the richness of the disentangled spatiotemporal representation.

**Key properties of the SCD-Net embeddings** (confirmed by Phase 2B.0 inspection):
- Shape: [T, 4096], one embedding per original video frame (~24fps native)
- Value range: approximately [−1.0, +1.0], consistent with L2-normalised output
- After window average pooling (window_size=16): shape [N, 4096] aligned to CLIP/DINOv3

The L2-normalisation of the output is a standard self-supervised training convention that constrains the embedding to a hypersphere, facilitating contrastive learning and providing a well-behaved metric structure. For MS-Temba, this normalisation is beneficial: the input projection `Linear(4096 → 256)` receives normalised inputs, which stabilises gradient flow.

### 3.4 SCD-Net vs Earlier Skeleton Backbones

| Property | ST-GCN | 2s-AGCN | CTR-GCN | SCD-Net |
|----------|:------:|:-------:|:-------:|:-------:|
| Training supervision | Supervised | Supervised | Supervised | Self-supervised |
| Output dimension | 256 | 256 | 256 | 4096 |
| Disentanglement | No | No | Channel-wise | Spatial/temporal |
| Streams | 1 | 2 (joint+bone) | Multiple | 1 (disentangled) |
| Masking augmentation | No | No | No | Yes (structural) |
| Value normalisation | Softmax (logits) | Softmax | Softmax | L2 (embedding) |

The self-supervised training of SCD-Net has an important implication: the embeddings are **not supervised toward any specific action taxonomy**. SCD-Net was trained on NTU RGB+D (120 classes, indoor daily activities) and the embeddings are therefore biased toward the kinematic patterns common in that dataset. Transfer to Charades — a very different dataset (household activities, multi-label, longer videos) — may result in some classes being well-represented and others being poorly discriminated in the SCD-Net embedding space. This is a hypothesis to be evaluated in the single-stream baseline experiment.

---

## 4. From Raw Skeleton to GNN Embedding: The Representation Pipeline

### 4.1 The Full Feature Derivation Chain

To understand what information the SCD-Net embeddings contain, it is instructive to trace the full pipeline from raw video pixels to the 4096-dim vector stored on disk:

```
Raw video frames  [H × W × 3]
         │
         ▼ Pose estimator (OpenPose / HRNet)
Joint keypoints   [T × V × 2]  (2D, per frame)
         │
         ▼ Depth lifting (optional) or 3D estimation
3D skeleton       [T × V × 3]  (if available)
         │
         ▼ Graph construction: body as G=(V,E)
         │  V = joints (nodes), E = bone connections (edges)
         │
         ▼ Spatial-Temporal GCN
Intermediate features [T × V × C']  (per-joint per-frame)
         │
         ▼ Graph-level pooling (mean/max over V)
Frame-level features  [T × C'']  (per-frame)
         │
         ▼ SCD-Net decoupling + self-attention
Spatial clue z_s  [D_s]  │
Temporal clue z_t [D_t]  │
         │
         ▼ Concatenation
SCD-Net embedding [T × 4096]  ← stored in Charades_SCDNet_features2.zip
         │
         ▼ Window average pooling (window_size=16)
Aligned embedding  [N × 4096]  ← stored in charades_scdnet_w16/
```

**What is lost at each stage**:
- Pose estimator → pixel appearance, colour, texture, background
- 2D→3D lifting → depth ambiguity (if monocular)
- Graph pooling → per-joint spatial location information (partially)
- SCD-Net encoding → raw coordinates, absolute position in scene

**What is preserved**:
- Body configuration (which joints are where, relative to each other)
- Motion patterns (how joints move over time)
- Kinematic structure (joint angles, velocities, trajectories)
- Spatial-temporal disentanglement (explicit spatial and temporal clues)

This derivation chain makes explicit why skeleton features are complementary to visual features: they discard precisely what visual features preserve (appearance, context, object identity) and preserve precisely what visual features discard (kinematic structure, body geometry).

### 4.2 The Window Alignment: Average Pooling of Kinematic States

The window average pooling operation applied during extraction (window_size=16, matching CLIP/DINOv3) has a specific kinematic interpretation. Averaging 16 consecutive frame-level embeddings over ~0.67 seconds produces a **mean kinematic state** over that temporal window. For smooth, sustained actions (holding an object, watching television, sitting), this average is highly representative of the action state and loses little information. For brief, transient actions (throwing, turning on a light), the average may significantly dilute the peak kinematic signal.

This temporal dilution is a potential weakness of the current setup for Group A classes (state/direction verbs), where the critical kinematic cue — the brief moment of state transition — may be averaged with surrounding frames of lower discriminative value. The multi-resolution temporal analysis discussed in Section 8 directly addresses this limitation.

---

## 5. The Complementarity of Skeleton and Visual Features

### 5.1 Formal Characterisation of Complementarity

The complementarity between skeleton and visual features can be characterised along several dimensions:

**Information-theoretic complementarity**: Let I(f_vis; Y) be the mutual information between visual features and action labels, and I(f_skel; Y) the mutual information between skeleton features and action labels. True complementarity requires that I(f_vis, f_skel; Y) > max(I(f_vis; Y), I(f_skel; Y)), i.e., the joint representation contains more information about the labels than either individual representation. This is expected when the two features have low mutual information I(f_vis; f_skel) — when they encode largely independent aspects of the action.

For CLIP and SCD-Net, I(f_vis; f_skel) is expected to be very low: CLIP features encode semantic-linguistic global concepts, while SCD-Net features encode kinematic patterns derived from keypoint graphs. The only shared information would be partial — e.g., both might encode some rough body configuration cues.

**Per-class complementarity from Phase 1**: the empirical evidence from Phase 1 directly characterises which classes are expected to benefit most from adding skeleton features:

| Class group | CLIP AP | Reason for CLIP failure | Expected skeleton contribution |
|-------------|:-------:|:-----------------------:|:-----------------------------:|
| Turn on/off light (c104/c105) | 8.7/2.3 | State ambiguity | Wrist trajectory direction |
| Throwing actions (c024,c025,c058,c074) | 7–15 | Brief gesture | Arm ballistic trajectory |
| Sneezing (c153) | 17.8 | Rapid postural transition | Trunk-head angular velocity |
| Standing up (c154) | 36.8 | Postural transition | Upward CoM + knee extension |
| Standing to sitting (c151) | 60.4 | — | Downward CoM + hip flexion |
| Running (c150) | 18.9 | Gait pattern | Periodic lower-limb trajectory |

### 5.2 The Two Sides of the Complementarity

While the above analysis motivates skeleton integration, it is equally important to identify classes where skeleton features are expected to be **less useful or potentially harmful**:

**Semantically defined classes**: *Someone is cooking* (c147, CLIP AP 79.0), *Working on a laptop* (c145, AP 58.4), *Watching television* (c132, AP 61.3). These classes are fundamentally defined by the presence of specific objects and their semantic context, not by any particular body configuration. A person cooking can have many different body poses; the key discriminative cue is the food and the kitchen context. Skeleton features for these classes will encode generic "standing in the kitchen" or "sitting at a desk" configurations, which add no discriminative power and may introduce noise.

**Object-defined classes with visual specificity**: *Opening a refrigerator* (c143, DINOv3 AP 88.1), *Holding a vacuum* (c137, DINOv3 AP 74.2). These classes are already well-handled by visual features (particularly DINOv3 dense) because the discriminative cue is in the object's appearance. The body configuration for these classes is generic (reaching forward, standing with arm extended), offering limited additional discriminative power.

This analysis suggests that the optimal fusion gate should learn to: (a) heavily weight skeleton features on Group A/B classes, (b) ignore skeleton features on semantically-defined classes, and (c) apply moderate weighting on object-manipulation classes. This per-class specialisation is precisely what the gated fusion mechanism can learn.

---

## 6. Literature: Visual–Skeleton Fusion Architectures

### 6.1 The Two-Modality Complementarity Problem

The skeleton-RGB fusion literature has extensively documented that these two modalities are complementary in action recognition. The RGB video modality contains not only temporal information but also abundant spatial information, such as the description of the human limbs and human-object interaction. The skeleton sequence modality naturally lacks spatial information, making it difficult to predict the action precisely in scenes with human-object interaction. These properties make the two modalities genuinely complementary for a comprehensive understanding of human actions.

This observation is directly relevant to Charades: the 7985 training videos contain a wide variety of actions spanning from purely kinematic (sneezing, running, standing up) to purely appearance-based (cooking, watching television) to combined (opening a refrigerator, which requires both the spatial context of the refrigerator and the arm extension motion). A single-modality model will inevitably fail on one of these subsets; a properly designed fusion model has the potential to succeed on all three.

### 6.2 Two-Stage Fusion: Skeleton Attention + Cross-Modal Attention

The most relevant architectural framework from the literature is the **two-stage fusion** approach proposed by Zhu et al. (2022; arXiv:2202.11374). The framework operates as follows:

**Stage 1 — Early fusion via skeleton attention**: the skeleton sequence is projected onto the RGB frames to generate a spatial attention mask, guiding the visual encoder to focus on the limb movement regions. The skeleton attention module computes a spatial heatmap by mapping joint coordinates onto the image plane; this heatmap weights the visual feature extraction, suppressing background and irrelevant regions.

**Stage 2 — Late fusion via cross-modal attention**: the final skeleton feature and the final visual feature are fused using a cross-attention module that exploits the correlation between the two modalities.

```
Skeleton sequence [T × V × C] ───────────────────────────┐
        │                                                  │
        ▼ Skeleton attention module                        │
   Spatial heatmap [H × W]                                │
        │                                                  │
        ▼ Guides visual encoder                           │
Visual features [T × D_v] ──────────────────────────────┐ │
                                                         │ │
                                                    Cross-Attention
                                                         │
                                               Fused features [T × D_f]
```

**Applicability to MS-Temba**: the Stage 1 component (skeleton-guided visual attention) requires pixel-level skeleton projection, which is not possible in the current setup where visual features are pre-extracted as fixed-length vectors without spatial indexing. However, the Stage 2 component — cross-modal attention between final skeleton and visual features — is directly applicable and corresponds to the cross-attention fusion strategy described in Section 8.

### 6.3 Human-Centric Multimodal Fusion (HCMFN)

Hu et al. (2024) proposed a human-centric approach to multimodal fusion that transforms RGB, optical flow, and depth data into person-centric images using skeleton keypoints as spatial anchors. The skeleton data guides the cropping and alignment of visual data around the person's body, ensuring that the visual features encode person-specific appearance rather than scene context. This approach addresses a key limitation of standard visual backbones: they encode a fixed-size crop of the full image, which may dedicate significant representational capacity to background regions that are irrelevant to the action.

While the full HCMFN pipeline is not applicable in the current setup (visual features are already extracted), the underlying principle is valuable: **skeleton information can guide attention** toward the body region in visual features, even post-extraction, through a learned cross-modal attention mechanism. This is the principle behind the skeleton-as-Query fusion strategy (Section 8.5).

### 6.4 MAF-Net: Two-Stage Self-Attention + Cross-Modal Attention

MAF-Net (2025) proposes a fusion framework with self-attention modules for within-modality feature enhancement followed by cross-modal attention for between-modality fusion. The model employs a late fusion strategy to combine skeletal and RGB features, allowing for more effective capture of spatial and temporal dependencies. Within each modality, a multi-head self-attention module suppresses noise and enhances discriminative features before cross-modal fusion.

The self-attention within-modality enhancement is directly applicable to the MS-Temba setting: before computing the cross-modal gate or attention, applying a lightweight self-attention layer to both the projected skeleton features and the projected visual features would allow each modality to enhance its internal structure before fusion. MS-Temba's temporal SSM blocks already provide a form of temporal self-attention within each modality if processed sequentially; the key question is whether explicit within-modality enhancement before fusion would further improve results.

### 6.5 The Multi-Stream Perspective: Score-Level vs Feature-Level Fusion

The skeleton literature distinguishes between two fundamental fusion paradigms with different tradeoffs:

**Score-level (late) fusion**: each modality is processed by a complete independent model, and the final action scores (logits or probabilities) are combined by weighted averaging. This is the most common approach in high-performance skeleton models and consistently achieves state-of-the-art by combining 4–6 streams. The advantage is that each modality's model can be optimised independently; the disadvantage is that no cross-modal interaction occurs during feature learning, limiting the model's ability to exploit correlated kinematic and visual patterns.

**Feature-level (early or intermediate) fusion**: modality features are combined at some intermediate layer of the model, allowing cross-modal interaction during feature learning. More complex to optimise but potentially captures synergies that score-level fusion misses.

For MS-Temba, both approaches are feasible and complementary:
- **Score-level**: train CLIP-based and skeleton-based MS-Temba independently, then combine their output logits. Simple, robust, no architectural changes.
- **Feature-level**: combine features before or within the MS-Temba temporal blocks. More complex but potentially higher-performance.

The ablation design in Section 10 tests both approaches, starting with score-level as a lower bound.

---

## 7. Literature: Skeleton + CLIP Fusion

### 7.1 SkeletonCLIP++ — Semantic Guidance for Skeleton Models

**SkeletonCLIP++** (2024) represents the most directly relevant fusion of skeleton features with CLIP semantics. The framework extends skeleton-based action recognition by incorporating CLIP's semantic information to provide richer supervision beyond binary action labels. The key contribution is a **Weighted Frame Integration (WFI)** mechanism that shifts video feature computation from simple averaging to a weighted frame approach, where CLIP similarity scores determine the temporal weighting. Frames that are most semantically representative of the action class (according to CLIP) receive higher weight in the temporal aggregation.

A second contribution, **Contrastive Sample Identification (CSI)**, introduces a discriminative task where the model learns to identify the most similar negative sample among positive ones — enhancing the ability to distinguish between closely related actions. The approach shows particular improvements on smaller datasets, which is directly relevant to the Charades training set (7985 videos, relatively small for multi-label TAD).

**Applicability to MS-Temba**: SkeletonCLIP++ operates at training time and requires CLIP embeddings as a supervision signal. In the MS-Temba context, a simpler adaptation is to use CLIP's per-class text embeddings as attention priors for the skeleton features: for a frame where CLIP assigns high probability to "holding a broom", the skeleton features encoding broom-related arm configurations should be up-weighted. This is the semantic-guided skeleton attention strategy discussed in Section 8.6.

### 7.2 CLIP-MG: Skeleton-Guided CLIP Attention

**CLIP-MG** (2025; arXiv:2506.16385) takes the opposite approach from SkeletonCLIP++: rather than using CLIP to guide skeleton processing, it uses skeleton keypoints to guide CLIP's visual attention. Specifically, a pose-guided semantic attention mechanism uses skeletal cues to steer CLIP towards where the gesture is taking place. The skeleton encoder generates a pose-guided query vector, which is then used in a cross-attention mechanism with CLIP's patch tokens to focus visual attention on gesture-relevant image regions.

The architectural design is precisely:
```
Skeleton keypoints → pose encoder → pose query Q_pose [D]
CLIP patch tokens  → patch encoder → patch K,V [N × D]

Attended visual features: Z = softmax(Q_pose · Kᵀ / √D) · V
Final: [CLIP CLS || Z]  → classification head
```

The ablation results are striking: removing the pose branch entirely reduces accuracy from 61.82% to 45.30% on iMiGUE micro-gesture recognition (-16.52 percentage points), confirming that skeleton data carries substantial complementary information even for visual recognition tasks. The significant gain from adding skeleton information motivates the same approach for Charades.

**Direct adaptation to MS-Temba**: the CLIP-MG principle can be adapted as follows. Instead of using raw skeleton keypoints as the query, the already-extracted SCD-Net embeddings serve as queries in a cross-attention over the visual features:

```
SCD-Net embedding [N, 4096] → proj_Q: Linear(4096 → 256) → Q [N, 256]
CLIP feature [N, 768]       → proj_KV: Linear(768 → 256)  → K,V [N, 256]

At each temporal position t:
  A_t = softmax(q_t · K^T / √256)  [N]  (temporal cross-attention)
  Z_t = A_t · V                    [256]
  h_t = proj_out(Z_t) + proj_clip(f_clip_t)  (residual)
```

### 7.3 The Semantic Limitation of Skeleton-Only Models

A consistent finding across the skeleton literature is that skeleton-only models — even state-of-the-art multi-stream GCN models — fail systematically on actions requiring **visual context or object identity**. This is acknowledged directly in the HCMFN paper: skeleton-based approaches often encounter difficulties when distinguishing between similar actions, as the absence of interaction information between individuals and objects makes it hard for skeleton-based methods to resolve ambiguities. For example, "drinking from a glass" and "talking on the phone" may have similar arm-to-face configurations and are difficult to distinguish from skeleton data alone.

For Charades, this limitation maps directly to the CLIP-dominant classes: *cooking* (AP 79.0 CLIP), *working on a laptop* (AP 58.4), *watching television* (AP 61.3). The skeleton features alone will not be competitive on these classes, and any fusion strategy must learn to rely on visual features for them.

---

## 8. Detailed Fusion Architectures for MS-Temba

### 8.1 Taxonomy of Fusion Points

As established in the Phase 2A companion document, fusion in MS-Temba can occur at four distinct points in the pipeline:

```
[Features on disk]          [Skeleton features on disk]
       │                              │
       ▼                              ▼
  DataLoader         ← POINT A: Concatenation at load time
       │
       ▼
  InputProjection    ← POINT B: Dual projectors, separate per modality
       │
       ▼
  Block 1 (SSM)      ← POINT C: Gated mixing between projected features
       │
       ▼
  Block 2, 3 (SSM)   ← POINT D: Cross-attention between temporal states
       │
       ▼
  Classifier Head    ← POINT E: Score-level (logit) fusion
```

For skeleton fusion, Point E (score-level) becomes an important additional option, since skeleton and visual models have traditionally been combined at the score level in the skeleton literature.

### 8.2 Strategy 0 — Score-Level Fusion (Baseline Lower Bound)

The simplest and most robust approach: train independent CLIP-based and skeleton-based MS-Temba models, then combine their output logits at inference time.

**Architecture**:
```
f_clip  [N, 768]  → MS-Temba_clip  → logits_clip  [157]
f_skel  [N, 4096] → MS-Temba_skel  → logits_skel  [157]

logits_fused = α · logits_clip + (1-α) · logits_skel
```

The mixing weight α can be:
- Fixed at 0.5 (equal weight)
- Tuned on a validation set via grid search
- Learned by a small logistic regression model trained on the validation set
- Per-class: α_c = argmax_{α} AP_c(val) — optimal per-class mixing

**Expected MAP**: if CLIP achieves 32.40 mAP and skeleton achieves S mAP, and they are genuinely complementary, score-level fusion with optimal α should achieve approximately max(CLIP, skeleton) + complementarity bonus ≈ 32.40 + 1–3 mAP.

**Advantages**: zero architectural changes, serves as a rigorous lower bound for feature-level fusion.  
**Disadvantages**: no cross-modal interaction during feature learning; each model learns independently without awareness of the other modality.

### 8.3 Strategy 1 — Feature Concatenation (Point A)

Concatenate visual and skeleton features before any projection, treating the combined vector as a single high-dimensional input.

**Architecture**:
```
f_clip  [N, 768]
f_skel  [N, 4096]
           │
      Concatenate along feature dim
           │
      f_fused [N, 4864]
           │
   InputProjection: Linear(4864 → 256) + LayerNorm + GELU + Dropout
           │
   MS-Temba blocks (unchanged)
```

**Parameter count**: 4864 × 256 + 256 = **1,245,184** for the input projection alone — 6× larger than the CLIP-only projection. With 7985 training videos, this extremely high ratio (156:1 parameters-per-training-sample) necessitates aggressive regularisation: `drop=0.2, drop_path=0.2, weight_decay=0.1`.

**Alternative — CLIP + DINOv3 dense + Skeleton concatenation**:
If Phase 2A results favour DINOv3 dense (2048-dim) over CLS-only (1024-dim), the full three-stream concatenation becomes:
```
f_clip  [N, 768]
f_dino  [N, 2048]
f_skel  [N, 4096]
           │
      f_fused [N, 6912]  → Linear(6912 → 256)
```
This would require extremely strong regularisation and is unlikely to be optimal without modality-specific projections.

### 8.4 Strategy 2 — Dual Projector with Additive Fusion (Point B)

Separate lightweight projection networks for each modality, combined by element-wise summation:

**Architecture**:
```
f_vis  [N, 768 or 2048]  →  proj_vis: Linear(D_v → 256)  →  h_vis  [N, 256]
f_skel [N, 4096]          →  proj_skel: Linear(4096 → 256) →  h_skel [N, 256]
                                                                      │
                                                              h = h_vis + h_skel
                                                                      │
                                                          MS-Temba blocks
```

**Rationale**: each projector independently compresses its modality into the 256-dim working space of MS-Temba. Addition assumes equal contribution of both modalities at every temporal position — a strong and likely incorrect assumption, since the relative importance of skeleton vs visual features varies by action class and by temporal position within the action.

### 8.5 Strategy 3 — Gated Dual Projector (Recommended)

The gated mechanism allows per-position, per-dimension adaptive weighting, directly motivated by the COMM/GCTF literature and adapted for the larger dimensionality asymmetry (4096 vs 768):

**Architecture**:
```
f_vis  [N, D_v]   →  proj_vis  [N, 256]  ──────────────┐
f_skel [N, 4096]  →  proj_skel [N, 256]  ──────────────┤
                                                        │
                        gate_input = [h_vis; h_skel]   [N, 512]
                        g = σ(W_g · gate_input)        [N, 256]
                                                        │
                        h = g ⊙ h_vis + (1 − g) ⊙ h_skel
                                                        │
                                          MS-Temba blocks
```

**Implementation** (in `models_MSTemba.py`):
```python
class VisualSkeletonGatedFusion(nn.Module):
    def __init__(self, vis_dim=768, skel_dim=4096, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_vis = nn.Sequential(
            nn.Linear(vis_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate)
        )
        self.proj_skel = nn.Sequential(
            nn.Linear(skel_dim, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate)
        )
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, f_vis, f_skel, skel_mask=None):
        """
        skel_mask: boolean tensor [B] — True for anomalous videos (5UNDJ-type)
                   where skeleton features are zero-padded. Gate forced to 1
                   (full visual) for these videos.
        """
        h_v = self.proj_vis(f_vis)    # [B, T, 256]
        h_s = self.proj_skel(f_skel)  # [B, T, 256]
        g   = self.gate(torch.cat([h_v, h_s], dim=-1))  # [B, T, 256]
        
        if skel_mask is not None:
            # Force gate to 1.0 (full visual) for videos with missing skeleton
            g = g.masked_fill(skel_mask.unsqueeze(-1).unsqueeze(-1), 1.0)
        
        return g * h_v + (1.0 - g) * h_s   # [B, T, 256]
```

**The skel_mask parameter**: this handles the anomalous video case (5UNDJ) identified during extraction, where the skeleton feature count differs drastically from the visual feature count. Rather than dropping these videos entirely, the zero-padded skeleton is masked out and the model falls back to visual-only for those examples.

**Expected gate specialisation**: through training, the gate is expected to learn:
- g ≈ 0 (full skeleton weight) for Group A/B kinematic classes at frames containing the kinematic event
- g ≈ 1 (full visual weight) for semantically-defined classes and frames with no distinctive kinematic content
- g ≈ 0.5 (mixed) for combined classes where both modalities contribute

This gate specialisation is directly analogous to the CLIP–DINOv3 gate described in the Phase 2A companion document, but with a stronger expected specialisation because CLIP and skeleton are more orthogonal than CLIP and DINOv3.

### 8.6 Strategy 4 — Skeleton-as-Query Cross-Attention

Inspired by the CLIP-MG architecture, the skeleton embedding acts as a contextual query that attends over the visual features. The semantic interpretation: "given the kinematic state I observe (skeleton), which aspects of the visual scene are most relevant?"

**Architecture**:
```
f_skel [N, 4096] →  proj_Q: Linear(4096 → 256)  →  Q [N, 256]
f_vis  [N, D_v]  →  proj_K: Linear(D_v → 256)   →  K [N, 256]
                 →  proj_V: Linear(D_v → 256)   →  V [N, 256]

At temporal position t:
  a_t = softmax(q_t · K^T / √256)    [N] temporal self-attention
  z_t = a_t · V                      [256]
  h_t = LayerNorm(proj_out(z_t) + proj_vis(f_vis[t]))  residual
```

**Semantic interpretation**: the skeleton query q_t encodes the kinematic state at time t (body configuration + motion dynamics). The attention weights a_t select which temporal positions of the visual features are most relevant given that kinematic state. This is a form of **kinematically-guided visual temporal attention**: the model learns to find the visual frames that are most consistent with the observed body motion.

Note that when N is the sequence length and attention is computed across all N positions, this is a form of temporal cross-attention with O(N²) complexity. For the window-pooled sequences (N ≈ 50 on average), this is computationally trivial.

### 8.7 Strategy 5 — Visual-as-Query Cross-Attention (Symmetric Alternative)

The symmetric alternative where visual features act as Query and skeleton as Key/Value:

```
f_vis  [N, D_v]  →  proj_Q: Linear(D_v → 256)   →  Q [N, 256]
f_skel [N, 4096] →  proj_K: Linear(4096 → 256)  →  K [N, 256]
                 →  proj_V: Linear(4096 → 256)  →  V [N, 256]

h = softmax(QK^T / √256) · V + proj_vis(f_vis)  (residual)
```

**Semantic interpretation**: "given what I see visually (visual Query), which aspects of the observed body motion (skeleton Key/Value) are most relevant to my understanding?" This prioritises visual semantics over kinematic structure, which may be more appropriate for Charades where most classes are visually-defined and skeleton provides a supplementary kinematic signal.

---

## 9. Implementation Details

### 9.1 Dataloader Changes

The `CharadesDataset` requires an additional optional `skel_root` parameter:

```python
class CharadesDataset(Dataset):
    def __init__(self, ..., rgb_root, skel_root=None, ...):
        self.skel_root = skel_root
        ...
    
    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        
        # Primary visual features
        vis_feat = np.load(os.path.join(self.rgb_root, f"{vid_id}.npy"))
        
        # Skeleton features (optional)
        skel_feat = None
        skel_valid = True
        if self.skel_root is not None:
            skel_path = os.path.join(self.skel_root, f"{vid_id}.npy")
            if os.path.exists(skel_path):
                skel_feat = np.load(skel_path)
                # Validate alignment
                if abs(skel_feat.shape[0] - vis_feat.shape[0]) > 1:
                    # Anomalous video (5UNDJ-type): use zero skeleton
                    skel_feat = np.zeros((vis_feat.shape[0], 4096), dtype=np.float32)
                    skel_valid = False
            else:
                skel_feat = np.zeros((vis_feat.shape[0], 4096), dtype=np.float32)
                skel_valid = False
        
        # Temporal pooling to num_clips=256 (unchanged)
        vis_feat  = self._temporal_pool(vis_feat)
        if skel_feat is not None:
            skel_feat = self._temporal_pool(skel_feat)
        
        return vis_feat, skel_feat, skel_valid, labels, ...
```

### 9.2 MSTemba_main.py Changes

New backbone identifiers for skeleton fusion configurations:

```python
fusion_configs = {
    # Single-stream baselines
    'clip':              {'in_feat_dim': 768,  'fusion': 'single'},
    'scdnet':            {'in_feat_dim': 4096, 'fusion': 'single'},
    # Visual + skeleton fusion
    'clip_skel_score':   {'fusion': 'score'},      # post-hoc logit fusion
    'clip_skel_concat':  {'in_feat_dim': 4864, 'fusion': 'concat'},
    'clip_skel_add':     {'vis_dim': 768, 'skel_dim': 4096, 'fusion': 'add'},
    'clip_skel_gated':   {'vis_dim': 768, 'skel_dim': 4096, 'fusion': 'gated'},
    'clip_skel_xattn_sv': {'vis_dim': 768, 'skel_dim': 4096, 'fusion': 'xattn_sv'},
    'clip_skel_xattn_vs': {'vis_dim': 768, 'skel_dim': 4096, 'fusion': 'xattn_vs'},
    # Three-stream (CLIP + DINOv3 dense + skeleton)
    'clip_dino_skel_gated': {'vis_dim': 2816, 'skel_dim': 4096, 'fusion': 'gated'},
}
```

### 9.3 Models_MSTemba.py Changes

The `MSTemba.__init__` is extended with a `fusion` parameter selecting the projection module:

```python
class MSTemba(nn.Module):
    def __init__(self, ..., fusion='single', vis_dim=768, skel_dim=4096,
                 in_feat_dim=768, ...):
        super().__init__()
        
        if fusion == 'single':
            self.proj = nn.Sequential(
                nn.Linear(in_feat_dim, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(drop_rate)
            )
        elif fusion == 'gated':
            self.proj = VisualSkeletonGatedFusion(
                vis_dim=vis_dim, skel_dim=skel_dim,
                out_dim=256, drop_rate=drop_rate
            )
        elif fusion == 'xattn_sv':
            self.proj = CrossAttentionFusion(
                query_dim=skel_dim, kv_dim=vis_dim,
                out_dim=256, drop_rate=drop_rate
            )
        elif fusion == 'xattn_vs':
            self.proj = CrossAttentionFusion(
                query_dim=vis_dim, kv_dim=skel_dim,
                out_dim=256, drop_rate=drop_rate
            )
        # All downstream blocks unchanged
        ...
    
    def forward(self, vis_features, skel_features=None, skel_mask=None):
        if skel_features is not None:
            x = self.proj(vis_features, skel_features, skel_mask)
        else:
            x = self.proj(vis_features)
        # rest of forward
        ...
```

---

## 10. Ablation Design and Experiment Matrix

### 10.1 Full Experiment Matrix

| ID | Config | Backbone(s) | `in_feat_dim` | Fusion | Lit. grounding |
|----|--------|-------------|:-------------:|:------:|---------------|
| 2B.0 | CLIP baseline | clip | 768 | single | Phase 1 |
| 2B.1 | Skeleton single-stream | scdnet | 4096 | single | SCD-Net (Wu et al.) |
| 2B.2 | Score-level fusion | clip + scdnet | — | score | Multi-stream literature |
| 2B.3 | Feature concatenation | clip_skel_concat | 4864 | concat | Lower bound |
| 2B.4 | Additive dual projector | clip_skel_add | — | add | Baseline |
| 2B.5 | Gated dual projector | clip_skel_gated | — | gated | GCTF, COMM |
| 2B.6 | Skeleton-as-Query CA | clip_skel_xattn_sv | — | xattn_sv | CLIP-MG |
| 2B.7 | Visual-as-Query CA | clip_skel_xattn_vs | — | xattn_vs | Symmetric alternative |
| 2B.8 | CLIP + DINOv3 + Skeleton | clip_dino_skel | — | gated ×2 | Full 3-stream |

### 10.2 Order of Execution

**Priority 1 — Single-stream skeleton baseline** (2B.1, already next step):
- Establishes whether SCD-Net features are discriminative on Charades
- Validates MS-Temba with in_feat_dim=4096
- Provides per-class AP profile for understanding skeleton strengths

**Priority 2 — Score-level fusion** (2B.2, immediate after 2B.1):
- Requires no code changes beyond running both models and combining outputs
- Establishes whether any complementarity exists before attempting feature-level fusion

**Priority 3 — Gated fusion** (2B.5, primary feature-level experiment):
- Theoretically best single-experiment fusion strategy
- Includes the skel_mask mechanism for handling anomalous videos

**Priority 4 — Cross-attention** (2B.6 or 2B.7):
- Only if gated fusion shows clear improvement over score-level
- The query direction (skeleton-as-Query vs visual-as-Query) to be determined by the relative strength of the single-stream experiments

### 10.3 Diagnostic Decision Tree

```
2B.1 skeleton mAP > 20?
├── YES: skeleton features are reasonably discriminative on Charades
│   ├── Per-class: which Group A/B classes does skeleton win on?
│   ├── Run 2B.2 score-level fusion
│   │   ├── 2B.2 > max(CLIP, skel) + 0.5 mAP?
│   │   │   ├── YES: complementarity confirmed → proceed to feature-level
│   │   │   │   └── Run 2B.5 (gated) → if gain > 2B.2: run 2B.6
│   │   │   └── NO: weak complementarity → report and reassess
│   │   └── Consider per-class score mixing with optimal α per class
└── NO: skeleton features have limited discriminative power
    ├── Reason A: SCD-Net domain mismatch (NTU → Charades)
    ├── Reason B: window averaging dilutes peak kinematic signal
    ├── Reason C: 4096-dim overfitting with 7985 training videos
    └── Action: test with LoRA-style fine-tuning or smaller projection
```

---

## 11. Expected Impact per Strategy

### 11.1 Skeleton Single-Stream (2B.1)

The skeleton single-stream mAP on Charades is expected to be substantially lower than CLIP (32.40) but potentially competitive on the kinematic classes. Based on the transfer characteristics of self-supervised skeleton models and the Charades class distribution:

**Conservative scenario (mAP 10–15)**: SCD-Net trained on NTU RGB+D transfers poorly to Charades due to the domain gap (controlled indoor activities vs naturalistic household activities). Skeleton features provide almost no information for semantically-defined classes (~60% of Charades by class count).

**Base scenario (mAP 15–22)**: Moderate transfer — skeleton features are competitive on kinematic classes (Groups A, B) but near-random on semantic classes. The mean AP over all 157 classes reflects the mixture of ~30 skeleton-competitive and ~127 skeleton-poor classes.

**Optimistic scenario (mAP 22–28)**: SCD-Net features are highly transferable across activity datasets. Some semantic classes are partially discriminable via skeleton (e.g., *watching television* → seated forward-facing posture).

### 11.2 Fusion Strategies

| Strategy | Expected ΔmAP vs CLIP | Expected ΔmAP vs best single | Mechanism |
|----------|:---------------------:|:----------------------------:|-----------|
| 2B.2 Score-level | +1.0 to +3.5 | +0.5 to +2.5 | Complementary predictions |
| 2B.3 Concat | +0.5 to +2.5 | +0.2 to +2.0 | Lower bound |
| 2B.4 Additive | +0.8 to +3.0 | +0.5 to +2.5 | Modality-specific projections |
| 2B.5 Gated | **+1.5 to +4.5** | +1.0 to +3.5 | Adaptive per-class weighting |
| 2B.6 Skel-as-Q CA | +1.5 to +5.0 | +1.0 to +4.0 | Kinematic-guided visual attention |

The gated dual projector (2B.5) is the primary recommendation. The expected +1.5 to +4.5 mAP gain over CLIP would bring MS-Temba to approximately **33.9–36.9 mAP**, representing a significant improvement over the current state.

### 11.3 Cumulative Three-Stream Estimate

Combining CLIP, DINOv3 dense (Phase 2A), and skeleton (Phase 2B) in a unified gated fusion:

| Configuration | Expected mAP | ΔmAP vs CLIP |
|---------------|:------------:|:------------:|
| CLIP baseline | 32.40 | — |
| + DINOv3 dense gated (2C.3) | 33.9–36.9 | +1.5 to +4.5 |
| + Skeleton gated (2B.5) | 33.9–36.9 | +1.5 to +4.5 |
| CLIP + DINOv3 + Skeleton gated | 35.4–39.4 | +3.0 to +7.0 |

The three-stream architecture is the ultimate target, expected to address the three orthogonal failure modes of the CLIP baseline: semantic gap (DINOv3 dense), spatial structure (DINOv3 patch), and kinematic information (skeleton).

---

## 12. Limitations and Open Questions

### 12.1 The Charades Anomaly: 5UNDJ

The single detected skeleton anomaly (5UNDJ: skel_windows=22 vs clip_T=292) suggests an SCDNet detection failure on that video. While the zero-tensor fallback handles this gracefully, it raises the question of whether other videos have subtler anomalies — partial detection failures that produce physically implausible but non-zero skeleton embeddings. A quality filter based on the reconstruction confidence of the pose estimator would address this but requires access to the raw pose estimation outputs.

### 12.2 Domain Gap: NTU RGB+D → Charades

SCD-Net was trained on NTU RGB+D (120 class categories, controlled indoor environment, fixed camera positions, explicit single-person actions). Charades is qualitatively different: naturalistic household activities, egocentric-like camera perspectives, multi-person scenes in some videos, multi-label annotations, and longer videos with more temporal ambiguity. The domain gap between these two datasets is an open question to be quantified by the 2B.1 single-stream experiment.

If the domain gap is severe (mAP < 15), a potential remedy is to fine-tune the SCD-Net embedding extractor on Charades action labels — but this requires the raw pose estimations for all 9848 videos, which are not available in the current setup.

### 12.3 The Temporal Resolution Mismatch

The window average pooling of skeleton features (16 frames per window) may dilute brief kinematic events that are the primary discriminative cue for Group A classes. A 16-frame window at 24fps covers ~0.67 seconds — long enough to average out a fast wrist turn or a brief arm throw trajectory. An alternative extraction at smaller window size (e.g., window_size=8) would double the temporal resolution at the cost of doubling the sequence length and storage.

This is a worthwhile experiment if the single-stream results show that Group A classes (direction/state verbs) particularly underperform relative to expectations.

### 12.4 Multi-Stream Skeleton Representations

The current setup uses a single SCD-Net stream. The state-of-the-art in skeleton recognition uses 4–6 streams (joint, bone, joint velocity, bone velocity, etc.), and score-level fusion over these streams consistently provides large gains. If resources permit, extracting additional streams from the same Charades_SCDNet_features2.zip source — or re-extracting with a multi-stream backbone — would likely improve the skeleton single-stream baseline substantially.

---

## 13. References

**SCD-Net and skeleton backbone**:
- Wu et al. (AAAI 2024). *SCD-Net: Spatiotemporal Clues Disentanglement Network for Self-supervised Skeleton-based Action Recognition*. arXiv:2309.05834.
- Yan et al. (AAAI 2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (ST-GCN). arXiv:1801.07455.
- Shi et al. (CVPR 2019). *Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition* (2s-AGCN).
- Chen et al. (ICCV 2021). *Channel-wise Topology Refinement Graph Convolutional Network for Skeleton-Based Action Recognition* (CTR-GCN).

**Visual–skeleton fusion**:
- Zhu et al. (ACM TOMM 2022). *Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition*. arXiv:2202.11374.
- Hu et al. (2024). *Human-Centric Multimodal Fusion Network for Robust Action Recognition* (HCMFN). Engineering Applications of AI.
- MAF-Net (2025). *A Multimodal Data Fusion Approach for Human Action Recognition*. PLOS ONE.
- Wu et al. (2024). *SCD-Net: Spatiotemporal Clues Disentanglement Network*. AAAI 2024.

**Skeleton + CLIP fusion**:
- SkeletonCLIP++ (2024). *Advancing Human Motion Recognition with SkeletonCLIP++*. PMC/MDPI.
- CLIP-MG (2025). *CLIP-MG: Guiding Semantic Attention with Skeletal Pose Features and RGB Data for Micro-Gesture Recognition*. arXiv:2506.16385.

**Mamba-based skeleton models**:
- ActionMamba (2025). *Action Spatial-Temporal Aggregation Network Based on Mamba and GCN for Skeleton-Based Action Recognition*. MDPI Electronics.
- TSkel-Mamba. *Temporal Dynamic Modeling via State Space Model for Human Skeleton-based Action Recognition*.
- Parts-Mamba. *Augmenting Joint Context with Part-Level Scanning for Occluded Human Skeleton*.

**MS-Temba original paper**:
- Pramanik et al. (2025). *MS-Temba: Multi-Scale Temporal Mamba for Efficient Temporal Action Detection*. arXiv:2501.06138.

---
