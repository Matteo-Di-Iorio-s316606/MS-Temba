# MS-Temba on Charades: Complete Analysis — CLIP vs DINOv3

> **Dataset**: Charades v1 — 157 domestic action classes, 7985 training videos, 1863 test videos  
> **Model**: MS-Temba (Multi-Scale Temporal Mamba)  
> **Backbone A**: CLIP ViT (512-dim features) — Best checkpoint **ep13**, Full-val-MAP **32.40**, Sampled-val-MAP **33.43**  
> **Backbone B**: DINOv3 ViT (1024-dim features, reg v2) — Best checkpoint **ep13**, Full-val-MAP **25.21**, Sampled-val-MAP **25.82**  
> **Note**: DINOv3 results refer to the `dinov3_vitl16_reg_v2` configuration (drop=0.05, drop_path=0.05, weight_decay=0.05, early_stop_patience=15), which constitutes the definitive DINOv3 baseline after a series of regularisation experiments (see Section 4.2).

---

## Table of Contents

1. [CLIP as Feature Extraction Backbone](#1-clip-as-feature-extraction-backbone)
2. [DINOv3 as Feature Extraction Backbone](#2-dinov3-as-feature-extraction-backbone)
3. [Direct Comparison: CLIP vs DINOv3](#3-direct-comparison-clip-vs-dinov3)
4. [Training Configuration](#4-training-configuration)
5. [Global Results](#5-global-results)
6. [Full Training Curves (ep 0–49)](#6-full-training-curves-ep-049)
   - 6.1 CLIP Training Curves
   - 6.2 DINOv3 Training Curves (reg v2)
7. [Per-Class Table (157 classes)](#7-per-class-table-157-classes)
8. [AP Distribution Analysis](#8-ap-distribution-analysis)
9. [Per-Object Analysis (38 categories)](#9-per-object-analysis-38-categories)
10. [Per-Verb Analysis (33 categories)](#10-per-verb-analysis-33-categories)
11. [Confusable Class Pairs](#11-confusable-class-pairs)
12. [Compositional Structure: Object × Verb Mapping](#12-compositional-structure-object--verb-mapping)
13. [Structurally Ambiguous Classes and Dataset Issues](#13-structurally-ambiguous-classes-and-dataset-issues)
14. [Recommendations and Future Directions](#14-recommendations-and-future-directions)
15. [Skeleton Feature Integration with CLIP](#15-skeleton-feature-integration-with-clip)

---

# 1. CLIP as Feature Extraction Backbone

## 1.1 Type of Features Produced

CLIP (*Contrastive Language–Image Pre-training*, OpenAI) is a **Vision Transformer (ViT)** trained on approximately 400 million image–text pairs. In this context it is used exclusively as a **visual feature extractor**: the model's `encode_image()` is called to obtain a global embedding of dimension **512** for each video frame, which MS-Temba receives as input and processes temporally.

The ViT divides each frame into non-overlapping patches (typically 16×16 pixels for ViT-B/16 or 14×14 for ViT-L/14) and projects them into a latent space. A special **[CLS]** token is added which, after N self-attention layers, aggregates information from all patches. The output of `encode_image()` is precisely this CLS token, linearly projected into a shared 512-dimensional space — the same space in which text embeddings are projected. The result is a **single global vector per frame**: no individual patch or region embeddings are exported to MS-Temba, only a single compact representation of the entire image.

This architectural choice has direct implications: everything that does not surface in the CLS aggregation — local patch distribution, fine spatial relationships, postural configuration — is irreversibly lost before MS-Temba receives the data.

## 1.2 How Pretraining Shapes the Representation

Contrastive training optimises the visual encoder so that images are close, in embedding space, to their natural textual descriptions. Formally, given a batch of N (image, text) pairs, the objective maximises the cosine similarity of the N correct pairs and minimises it for the N²−N incorrect ones. This has a direct consequence on the geometry of the 512-dim space: vectors organise themselves around **linguistically frequent semantic concepts** in the web corpus — objects, categories, usage contexts, typical actions. Two frames are close in embedding space not because they share the same visual composition, but because they could be described with the same words.

For MS-Temba this has an important effect: the temporal sequence of input features is already strongly separated by global semantics. Mamba receives, for each frame, a point in a space where *cooking* is already distant from *holding a phone*, and *lying on a bed* is already distant from *walking through a doorway*. The contribution of MS-Temba therefore reduces, in large part, to distinguishing in time the transition patterns between these concepts, rather than building semantic representations from scratch.

An important corollary is that separability in CLIP space is asymmetric with respect to verbs: actions sharing the same object but with different verbs (*putting a bag* vs *taking a bag*, *opening a door* vs *closing a door*) tend to produce similar embeddings, because the textual corpus describes both with very similar words in the same context. Verb-dependent discrimination therefore falls almost entirely on MS-Temba.

## 1.3 Strengths for Feature Extraction in This Context

**Compact semantic discriminativity.** At 512 dimensions, CLIP features already provide a strongly separated signal for semantically distinct Charades classes. This reduces the risk of overfitting in MS-Temba's downstream classifier: a compact, well-structured feature space requires less data to train a reliable classifier compared to a high-dimensional space with unsupervised geometry. Classes such as *cooking* (AP 79.0), *talking on the phone* (AP 75.9), *working on a laptop* (AP 75.9), or *holding a broom* (AP 75.4) achieve very high APs precisely because these categories correspond to concepts clearly anchored in the pretraining text corpus: the backbone has already separated these clusters before MS-Temba intervenes.

**Web-scale semantic coverage.** Pretraining on 400M image–text pairs exposes the backbone to a vast variety of domestic actions, many of which naturally fall within the Charades vocabulary. As a result, the produced features tend to be informative even for classes that are not very frequent in the training set, as long as they are linguistically well-characterised — a direct advantage in a dataset like Charades, where the class distribution is heavily imbalanced and many actions have fewer than 100 training occurrences.

**Temporal signal stability.** Since CLIP features are anchored to stable semantic concepts, the frame-by-frame sequence that MS-Temba receives tends to be coherent and low-noise for high-AP classes: scenes of *cooking*, *watching television*, or *lying on a bed* produce very similar CLS vectors across consecutive frames, making it easier for the temporal model to detect the start and end of the action.

## 1.4 Limitations for Feature Extraction in This Context

**Loss of spatial resolution.** Compression into the CLS token is the most important structural limitation. MS-Temba receives, for each frame, a single 512-dim vector: information about the subject's posture, limb arrangement, spatial hand–object relationship, and body configuration are all aggregated in this single point, with irreversible loss of local detail. The ViT's self-attention can in principle capture patch-to-patch relationships, but was not trained to preserve this local geometry in the CLS token, because the training signal was compatibility with global textual descriptions. This systematically penalises classes where the discriminative signal is configurational rather than nominal: *standing to sitting* (AP 60.4 vs 23.2 DINOv3), *someone is running* (AP 18.9 vs 9.1 DINOv3), *someone is sneezing* (AP 17.8 vs 11.8 DINOv3) — in the real data, DINOv3 does not resolve these classes better than CLIP, contrary to what was initially hypothesised.

**Verb-dependent ambiguity.** As noted, class pairs sharing the same object with opposing verbs produce very similar embeddings. This is not only a separability problem in feature space, but structurally reflects the fact that the textual corpus rarely distinguishes *put* and *take* with visually informative descriptions. MS-Temba must therefore learn to discriminate these pairs almost exclusively from temporal dynamics, without a useful geometric prior from the feature — which explains why classes such as *closing a door* (AP 30.3) vs *opening a door* (AP 39.8) or *putting a bag* (AP 40.6) vs *taking a bag* (AP 25.4) show heterogeneous performance despite similar training frequencies.

**Semantic saturation for visually rich classes.** Some Charades classes are visually very specific but semantically generic in natural language — for example *throwing* is a rare and visually violent verb that the web corpus associates with contexts very different from domestic settings. This produces uninformative features for all classes with *throw* as a verb, as confirmed by the mean AP of v025 (throw) of only 10.2, the lowest among all verbs.

---

# 2. DINOv3 as Feature Extraction Backbone

## 2.1 Type of Features Produced

DINOv3 (Meta/FAIR) is a family of **Vision Transformers** trained with purely visual self-supervised learning on a proprietary web-scale dataset (LVD-1689M). In this context the **ViT-L/14** variant is used — 24 transformer layers, patch size 14×14, internal dimension 1024 — which produces embeddings of dimension **1024**. Unlike CLIP, DINOv3 is designed to produce both a global embedding (CLS token) and high-quality **dense per-patch features**; in the current setup however MS-Temba receives exclusively the 1024-dimensional CLS vector per frame, leaving the richness of patch-level representations unexploited.

Compared to CLIP, the feature is more capacious (1024 vs 512) but entirely devoid of linguistic grounding: it represents the image according to a geometry learned purely from visual statistical structure, without any label or textual description ever guiding training.

## 2.2 How Pretraining Shapes the Representation

DINOv3's self-supervised training is based on a **student-teacher procedure with EMA** (Exponential Moving Average): the teacher is a moving average of the student's weights, and the objective is to have the student, under strongly augmented views of the image, reproduce the teacher's activation distributions on minimally distorted views. Without text or labels, the learning signal comes entirely from the internal structure of the images themselves.

A crucial characteristic of DINOv3 compared to previous versions is the **Gram anchoring** mechanism: during training, not only the global feature is regularised but the similarity matrix between patch tokens — i.e. patch-to-patch relationships. In practice, the encoder is pushed to preserve *how different regions of the image relate to each other*, not just which global embedding it produces. This directly affects the quality of the CLS token: while producing a single global vector, the aggregation occurs in a ViT whose internal layers have learned to maintain local spatial coherence. The result is a CLS token that, compared to CLIP's, carries more implicit information about the geometric structure of the scene.

For MS-Temba this translates into a CLS vector organised around the **visual structure of the scene** rather than its nominal meaning: two frames are close in embedding space if the scene is arranged similarly — same subject posture, same body orientation, same spatial subject–object relationship — regardless of how it would be described in words.

## 2.3 Strengths for Feature Extraction in This Context

**Sensitivity to spatial configuration — updated and corrected data.** DINOv3's CLS token implicitly preserves more information about the geometric structure of the scene compared to CLIP.

With complete data over 157 classes, DINOv3 outperforms CLIP on **51/157 classes** at the per-class best checkpoint, but the pattern is not postural. The largest wins concern **object manipulation actions with visually specific objects that have low salience in CLIP's web corpus**:

| Class | CLIP AP₁₃ | DINOv3 AP@best | Δ |
|-------|----------:|--------------:|---:|
| Opening a refrigerator (c143) | 60.2 | 88.1 | +27.9 |
| Putting a broom somewhere (c099) | 27.4 | 53.0 | +25.6 |
| Holding a vacuum (c137) | 51.3 | 74.2 | +22.9 |
| Putting a blanket (c071) | 18.5 | 38.8 | +20.3 |
| Taking a vacuum from somewhere (c138) | 20.9 | 37.9 | +17.0 |
| Taking some clothes (c002) | 38.6 | 54.7 | +16.1 |
| Lying on a bed (c134) | 69.9 | 82.8 | +12.9 |

Postural classes go in the opposite direction: *standing to sitting* (DINOv3: 23.2 vs CLIP 60.4), *sneezing* (11.8 vs 17.8), *running* (9.1 vs 18.9) — CLIP wins on all of them.

**Representational richness at 1024 dimensions.** The higher dimensionality offers in principle more degrees of freedom to encode subtle visual information. In a multi-label task like Charades — where multiple actions can coexist in the same frame — a larger space can in theory host discriminative signals for heterogeneous classes without them "overwriting" each other in the embedding. This theoretical advantage is however conditional on sufficient data to train the downstream classifier without saturating the space's capacity.

**Unexploited potential of dense features.** An important structural property of DINOv3 — not exploited in the current setup — is the quality of its **per-patch features**. Unlike CLIP, DINOv3 was explicitly optimised to produce spatially coherent and semantically discriminative local feature maps. Passing to MS-Temba not only the CLS token but also a subset of patch tokens (or a weighted mean thereof) could recover information about the spatial distribution of scene elements that the CLS alone does not transmit — particularly for classes with high postural variance.

## 2.4 Limitations for Feature Extraction in This Context

**Absence of nominal semantic prior.** The most relevant limitation when compared to CLIP is that DINOv3's feature space is not organised around linguistic concepts. For Charades classes whose discrimination depends on a strong, visually generic nominal concept — *cooking* (AP ~5.5 vs 79.0 CLIP), *talking on the phone* (AP ~5.0 vs 75.9), *watching television* (AP ~48.1 vs 61.3) — the backbone offers no prior advantage in embedding separation. MS-Temba receives features whose space does not in any way reflect the semantic taxonomy of Charades: visually very different but semantically close classes, or visually similar but semantically distant ones, may be at arbitrary distances in DINOv3 space. The downstream classifier must therefore build semantic separation from scratch, with all the data requirements and overfitting risk this entails.

**Accelerated overfitting from higher capacity.** The 1024-dimensionality makes the linear classifier (or any downstream head) far more susceptible to memorising the training set compared to a 512-dim feature. In the data this effect is evident even in the regularised configuration (reg v2): DINOv3 shows a train–val gap of ~52 pt at ep28 (stop) vs ~79 pt for CLIP at ep13 (best). The problem is structural: in a 1024-dim space, a linear classifier has twice the parameters per class compared to a 512-dim one, and with only ~50 test examples per class on average, memorisation is nearly inevitable without regularisation. The reg v2 configuration (drop=0.05, wd=0.05) mitigates but does not eliminate this problem.

**Mismatch between backbone and temporal model capacity.** MS-Temba has ~18M total parameters — a relatively compact model. Receiving 1024-dim features instead of 512-dim doubles the input size without changing the temporal model's capacity: the risk is that MS-Temba does not have enough capacity to effectively exploit the additional dimensions, while simultaneously suffering from overfitting of the final classifier due to higher dimensionality. A learnable 1024→512 projection before MS-Temba's input could resolve this mismatch, making the two configurations comparable in terms of pressure on the downstream classifier.

---

# 3. Direct Comparison: CLIP vs DINOv3

CLIP and DINOv3 encode two fundamentally different types of visual information, and this difference is the main interpretive key of the Charades results.

### Summary Comparison Table

| Aspect | CLIP (512-dim) | DINOv3 (1024-dim, reg v2) | Interpretation |
|--------|:--------------:|:-------------------------:|----------------|
| **Pretraining paradigm** | Contrastive image-text | Self-supervised visual distillation | CLIP learns image–text correspondence; DINOv3 learns visual regularities without language |
| **Semantic grounding** | **Strong** | Absent / implicit | CLIP benefits from easily nameable classes; DINOv3 receives no direct linguistic prior |
| **Local spatial sensitivity** | Moderate | **High** | DINOv3 is better suited to capture configurations of visually specific objects |
| **Representation granularity** | Global-semantic | Dense-structural | CLIP favours overall meaning; DINOv3 favours the arrangement of visual details |
| **Feature dimensionality** | 512 | 1024 | DINOv3 offers more capacity, but with higher overfitting risk |
| **Robustness on semantically descriptive classes** | **High** | Moderate | CLIP is favoured when the class name coincides with frequent natural language concepts |
| **Robustness on postural/body classes** | Moderate | **Low** | Real data show DINOv3 does not excel on postural classes |
| **Classes AP ≥ 60** | **22** | 5 | CLIP produces more strong classes |
| **Classes AP < 20** | 49 | **~90** | DINOv3 reg v2 is weak on ~57% of classes |
| **Best checkpoint** | ep13 | ep13 | = (same for both) |
| **Regularisation** | None (orig) | drop=0.05, wd=0.05 (reg v2) | DINO requires regularisation; CLIP does not |
| **Post-peak stability** | Moderate (−3.7 at ep49) | **Improved** (−4.0 at ep28 vs −6.6 without reg) | Reg v2 contains DINOv3's collapse |
| **Main advantage** | Semantically recognisable classes | Manipulation actions with specific objects (refrigerator, vacuum, broom) | Pattern differs from initial hypothesis |
| **Main limitation** | Directional verbs, postural actions | Absence of semantic prior, no real postural advantage | Partially complementary weaknesses |

### Comparative Discussion

From a **feature signal** standpoint, CLIP and DINOv3 represent two nearly complementary extremes. CLIP produces a representation centred on the linguistic meaning of the scene: two images are close in embedded space if they could be described with the same words. DINOv3 produces a representation centred on visual structure: two images are close if the arrangement of elements in the scene is similar. In Charades — a dataset with many easily verbalisable domestic actions — this difference is directly reflected in the per-class AP distribution.

Quantitative results show that **CLIP dominates in terms of overall coverage**: global best-mAP 32.40 vs 25.21 (gap +7.19), 22 classes with AP ≥ 60 vs 5, and only 49 classes with AP < 20 vs ~90 for DINOv3 out of 157 classes. This pattern is consistent with a dataset in which most classes are defined by easily nameable object–action combinations (*holding a phone*, *cooking*, *watching television*) for which CLIP's semantic prior offers a structural advantage.

**DINOv3** shows clear advantages on **51/157 classes** at the per-class best, but the real pattern differs from the original hypothesis: the largest wins concern manipulation actions with visually specific objects — *opening a refrigerator* (+27.9), *putting a broom* (+25.6), *holding a vacuum* (+22.9) — not postural classes.

**On the regularisation effect for DINOv3.** The transition from the original configuration (no reg, mAP 25.44) to reg v2 (drop=0.05, wd=0.05, mAP 25.21) introduces a minimal loss on the best checkpoint (−0.23 mAP) in exchange for a more stable post-peak curve: the collapse from best to stop is reduced from −6.6 pt (orig, ep13→ep49) to −4.0 pt (reg v2, ep13→ep28). The reg v2 configuration is therefore preferable as the definitive baseline because it offers more reliable and comparable results for future experiments.

### Interpretive Summary

Overall, **CLIP constitutes the most effective backbone as a general choice on Charades**, thanks to its greater compatibility with semantically descriptive classes, better category coverage, and lower tendency to overfit. **DINOv3** retains a role as a complementary backbone for the ~51 classes where its per-class advantage is real (visually specific objects: refrigerator, vacuum, broom). Hybrid routing remains feasible; the impact on Full-val-MAP is difficult to estimate precisely before testing, but DINOv3-winning classes represent ~33% of classes with average frequency.

---

## 4. Training Configuration

### 4.1 CLIP (original configuration — definitive baseline)

| Parameter | Value |
|-----------|-------|
| Dataset | Charades v1 (157 classes) |
| Training split | 7985 videos |
| Test split | 1863 videos |
| Epochs | 50 (0–49) |
| Batch size | 5 |
| Num clips | 256 |
| Optimiser | AdamW (lr=5e-4, wd=0.01) |
| Scheduler | Cosine (warmup=5 ep, min_lr=1e-5) |
| CLIP feat dim | 512 |
| Dropout | 0.0 |
| Drop-path | 0.0 |
| Total parameters | ~18M |
| Loss | Multi-label BCE + Diversity Loss (β=0.05) |
| Seed | 0 |
| Early stopping | No |

### 4.2 DINOv3 (reg v2 configuration — definitive baseline)

The DINOv3 configuration required a series of regularisation experiments to identify the optimal setup. Three rounds were conducted:

| Configuration | drop | drop_path | weight_decay | Best mAP | Stop ep | Notes |
|---------------|:----:|:---------:|:------------:|:--------:|:-------:|-------|
| orig (no reg) | 0.0 | 0.0 | 0.01 | 25.44 | 50 | Severe post-ep13 collapse (−6.6 mAP) |
| reg v1 | 0.2 | 0.1 | 0.05 | 24.56 | 26 | Dropout too aggressive |
| **reg v2 (definitive)** | **0.05** | **0.05** | **0.05** | **25.21** | **28** | Near parity on best, stable curve |

**Final reg v2 parameters**:

| Parameter | Value |
|-----------|-------|
| Dataset | Charades v1 (157 classes) |
| DINOv3 feat dim | 1024 |
| Dropout | 0.05 |
| Drop-path | 0.05 |
| Weight decay | 0.05 |
| Early stop patience | 15 |
| Min delta | 0.01 |
| Early stop epoch | 28 |
| All other parameters | Identical to CLIP (above) |

**Architectural note**: the regularisation modifications required code changes to `models_MSTemba.py`: `LinearProjection` was extended with a `drop_rate` parameter propagated to all inter-block instances, and `InputProjection` was converted from `nn.Linear` to `nn.Sequential(Linear, LayerNorm, GELU, Dropout)` to ensure dropout also acts on the model's input — the critical point for 1024-dim features.

---

## 5. Global Results

### 5.1 mAP per Checkpoint

| Metric | CLIP | DINOv3 reg v2 | Δ |
|--------|:----:|:-------------:|:-:|
| Best epoch | **13** | **13** | = |
| Full-val-MAP (best ep) | **32.40** | 25.21 | +7.19 |
| Sampled-val-MAP (best ep) | **33.43** | 25.82 | +7.61 |
| Train-MAP @ best ep | 40.43 | 33.08 | — |
| Full-val-MAP @ stop ep | 28.73 (ep49) | 21.22 (ep28) | — |
| Train–val gap @ best ep | +8.0 pt | +7.9 pt | ≈ |
| Train–val gap @ stop ep | ~70.8 pt (ep49) | ~56.4 pt (ep28) | — |

### 5.2 Per-Class AP Distribution (Best Checkpoint)

| AP Range | CLIP ep13 | DINOv3 reg v2 ep13 |
|----------|:---------:|:------------------:|
| ≥80 (Excellent) | 0 | 1 |
| 60–79 | 22 | 4 |
| 40–59 | 30 | 10 |
| 20–39 | 56 | 41 |
| <20 (Low) | 49 | ~101 |

---

## 6. Full Training Curves (ep 0–49)

## 6.1 CLIP Training Curves (Epochs 0–49)

All values extracted from `training_CLIP.log` — EMA model, original configuration.

| Ep | Train-MAP | Full-val-MAP | Sampled-val-MAP | Gap (T−V) |
|----|----------:|-------------:|----------------:|----------:|
|  0 |   1.82 |   2.45 |   2.61 |  -0.63 |
|  1 |   1.74 |   2.44 |   2.65 |  -0.70 |
|  2 |   2.13 |   3.66 |   3.79 |  -1.53 |
|  3 |   3.64 |   9.02 |   9.33 |  -5.38 |
|  4 |   9.69 |  18.92 |  19.70 |  -9.23 |
|  5 |  15.08 |  23.35 |  23.99 |  -8.27 |
|  6 |  19.32 |  26.58 |  27.35 |  -7.26 |
|  7 |  22.16 |  27.40 |  28.00 |  -5.24 |
|  8 |  25.80 |  30.06 |  30.87 |  -4.26 |
|  9 |  28.44 |  31.19 |  32.00 |  -2.75 |
| 10 |  31.76 |  31.45 |  32.22 |  +0.31 |
| 11 |  32.75 |  31.99 |  32.78 |  +0.76 |
| 12 |  36.20 |  32.24 |  33.22 |  +3.96 |
| 13 |  40.43 |  32.40 |  33.43 |  +8.03 | ← **BEST**
| 14 |  44.06 |  31.52 |  32.44 | +12.54 |
| 15 |  48.44 |  32.09 |  33.29 | +16.35 |
| 16 |  53.34 |  31.66 |  32.70 | +21.68 |
| 17 |  55.72 |  31.25 |  32.34 | +24.47 |
| 18 |  61.34 |  31.12 |  31.98 | +30.22 |
| 19 |  66.27 |  30.75 |  31.71 | +35.52 |
| 20 |  68.49 |  31.09 |  32.05 | +37.40 |
| 21 |  74.51 |  30.74 |  31.66 | +43.77 |
| 22 |  77.73 |  30.45 |  31.57 | +47.28 |
| 23 |  81.49 |  30.04 |  30.99 | +51.45 |
| 24 |  84.46 |  29.99 |  30.92 | +54.47 |
| 25 |  87.44 |  29.79 |  30.77 | +57.65 |
| 26 |  88.52 |  29.92 |  30.83 | +58.60 |
| 27 |  91.48 |  30.06 |  31.06 | +61.42 |
| 28 |  92.98 |  29.80 |  30.76 | +63.18 |
| 29 |  93.38 |  29.81 |  30.68 | +63.57 |
| 30 |  95.44 |  29.61 |  30.53 | +65.83 |
| 31 |  96.25 |  29.53 |  30.42 | +66.72 |
| 32 |  96.97 |  29.27 |  30.23 | +67.70 |
| 33 |  97.00 |  29.32 |  30.22 | +67.68 |
| 34 |  97.32 |  29.24 |  30.16 | +68.08 |
| 35 |  97.88 |  29.21 |  30.12 | +68.67 |
| 36 |  97.87 |  29.12 |  30.05 | +68.75 |
| 37 |  98.50 |  29.23 |  30.14 | +69.27 |
| 38 |  98.39 |  29.17 |  30.08 | +69.22 |
| 39 |  98.90 |  29.08 |  30.02 | +69.82 |
| 40 |  98.76 |  29.08 |  30.06 | +69.68 |
| 41 |  99.02 |  29.12 |  30.07 | +69.90 |
| 42 |  99.03 |  28.95 |  29.85 | +70.08 |
| 43 |  98.99 |  28.96 |  29.89 | +70.03 |
| 44 |  99.21 |  28.90 |  29.83 | +70.31 |
| 45 |  99.24 |  28.84 |  29.76 | +70.40 |
| 46 |  99.19 |  28.87 |  29.78 | +70.32 |
| 47 |  99.33 |  28.76 |  29.67 | +70.57 |
| 48 |  99.21 |  28.74 |  29.66 | +70.47 |
| 49 |  99.52 |  28.73 |  29.68 | +70.79 |

> **Note**: val MAP peaks at ep13 (32.40) then decreases monotonically. Train MAP continues to rise to 99.52. The model enters massive overfitting after ep20. The cause is structural — 7985 videos with 157 imbalanced classes are insufficient for an ~18M parameter model with zero regularisation.

## 6.2 DINOv3 reg v2 Training Curves (Epochs 0–28)

Values extracted from `metrics_per_epoch.csv` of the `dinov3_vitl16_reg_v2/seed0` configuration. Training stopped at ep28 due to early stopping (patience=15, min_delta=0.01).

| Ep | Train-MAP | Full-val-MAP | Sampled-val-MAP | Gap (T−V) | is_best |
|----|----------:|-------------:|----------------:|----------:|:-------:|
|  0 |   1.79 |   2.42 |   2.63 |  -0.63 | ✓ |
|  1 |   1.89 |   2.36 |   2.51 |  -0.47 | |
|  2 |   1.84 |   3.44 |   3.56 |  -1.60 | ✓ |
|  3 |   2.64 |   5.09 |   5.20 |  -2.45 | ✓ |
|  4 |   5.13 |  12.04 |  12.47 |  -6.91 | ✓ |
|  5 |  10.85 |  16.25 |  16.73 |  -5.40 | ✓ |
|  6 |  14.55 |  19.87 |  20.49 |  -4.68 | ✓ |
|  7 |  17.11 |  21.17 |  21.69 |  -4.06 | ✓ |
|  8 |  20.30 |  22.81 |  23.30 |  -2.51 | ✓ |
|  9 |  23.18 |  23.88 |  24.34 |  -0.70 | ✓ |
| 10 |  25.50 |  24.06 |  24.58 |  +1.44 | ✓ |
| 11 |  27.51 |  24.70 |  25.37 |  +2.81 | ✓ |
| 12 |  31.22 |  24.38 |  24.91 |  +6.84 | |
| 13 |  33.08 |  25.21 |  25.82 |  +7.87 | ✓ | ← **BEST**
| 14 |  34.44 |  24.73 |  25.37 | +9.71 | |
| 15 |  36.34 |  24.39 |  25.06 | +11.95 | |
| 16 |  40.15 |  24.38 |  25.01 | +15.77 | |
| 17 |  42.56 |  24.03 |  24.89 | +18.53 | |
| 18 |  45.59 |  23.73 |  24.41 | +21.86 | |
| 19 |  49.10 |  23.31 |  24.07 | +25.79 | |
| 20 |  51.57 |  23.10 |  23.81 | +28.47 | |
| 21 |  56.09 |  23.18 |  23.94 | +32.91 | |
| 22 |  58.20 |  22.58 |  23.20 | +35.62 | |
| 23 |  62.22 |  22.49 |  23.29 | +39.73 | |
| 24 |  65.42 |  22.16 |  22.91 | +43.26 | |
| 25 |  68.98 |  21.84 |  22.59 | +47.14 | |
| 26 |  71.03 |  21.58 |  22.38 | +49.45 | |
| 27 |  73.43 |  21.34 |  22.03 | +52.09 | |
| 28 |  66.89 |  21.22 |  21.98 | +45.67 | | ← Early stop

> **Note**: the reg v2 configuration peaks at ep13 (25.21), identical to the original. The train–val gap at ep28 (stop) is ~52 pt — lower than the ~79 pt of the original at ep49, confirming the stabilising effect of regularisation. The post-peak collapse is contained: −4.0 mAP over 15 epochs vs −6.6 mAP over 36 epochs in the original. Regularisation did not modify the best checkpoint position (ep13 in all runs) but limited subsequent deterioration.

---

## 7. Per-Class Table (157 Classes)

**Column legend**:
- **Tr**: training set occurrences; **Te**: test set; **Tot**: total
- **AP₀**: AP at epoch 0 (random baseline); **AP₁₃**: AP at CLIP best checkpoint (ep13)
- **AP_peak**: maximum CLIP AP reached in any epoch 0–49; **Pk_ep**: CLIP peak epoch
- **AP₄₉**: CLIP AP at last epoch; **Δ13→49**: drop from best checkpoint to ep49 (negative = deterioration)
- **AP_DINOv3**: DINOv3 reg v2 AP at per-class best checkpoint
- **Winner**: backbone with highest AP at respective best checkpoint

| Code | Class | Obj | Verb | Tr | Te | Tot | AP₀ | AP₁₃ | AP_peak | Pk_ep | AP₄₉ | Δ13→49 | AP_DINOv3 | Winner |
|------|-------|-----|------|---:|---:|----:|----:|-----:|--------:|------:|-----:|-------:|----------:|--------|
| c000 | Holding some clothes | clothes | hold | 635 | 227 | 862 | 5.4 | 41.3 | 42.5 | ep10 | 32.9 | +8.3 | — | CLIP* |
| c001 | Putting clothes somewhere | clothes | put | 640 | 209 | 849 | 3.4 | 32.2 | 34.3 | ep11 | 27.6 | +4.5 | 46.5 | DINOv3 |
| c002 | Taking some clothes from somewhere | clothes | take | 479 | 180 | 659 | 3.1 | 38.6 | 41.4 | ep14 | 32.5 | +6.1 | 54.7 | DINOv3 |
| c003 | Throwing clothes somewhere | clothes | throw | 303 | 113 | 416 | 1.7 | 20.7 | 21.9 | ep18 | 14.9 | +5.8 | 30.2 | DINOv3 |
| c004 | Tidying some clothes | clothes | tidy | 266 | 95 | 361 | 3.0 | 36.7 | 41.4 | ep22 | 32.4 | +4.3 | 18.6 | CLIP |
| c005 | Washing some clothes | clothes | wash | 81 | 23 | 104 | 1.4 | 19.5 | 26.5 | ep10 | 21.3 | -1.8 | 8.5 | CLIP |
| c006 | Closing a door | door | close | 618 | 204 | 822 | 2.7 | 30.3 | 30.3 | ep13 | 18.9 | +11.4 | 16.6 | CLIP |
| c007 | Fixing a door | door | fix | 27 | 28 | 55 | 1.0 | 33.4 | 33.5 | ep12 | 21.6 | +11.8 | 5.3 | CLIP |
| c008 | Opening a door | door | open | 784 | 273 | 1057 | 3.6 | 39.8 | 43.5 | ep16 | 38.3 | +1.5 | 41.2 | DINOv3 |
| c009 | Putting something on a table | table | put | 879 | 285 | 1164 | 3.4 | 23.1 | 25.6 | ep16 | 17.2 | +6.0 | 28.2 | DINOv3 |
| c010 | Sitting on a table | table | sit | 46 | 16 | 62 | 0.6 | 3.8 | 8.5 | ep20 | 7.9 | -4.2 | 8.5 | DINOv3 |
| c011 | Sitting at a table | table | sit | 728 | 197 | 925 | 11.4 | 75.5 | 78.2 | ep21 | 75.4 | +0.1 | 82.5 | DINOv3 |
| c012 | Tidying up a table | table | tidy | 202 | 72 | 274 | 1.5 | 38.3 | 42.2 | ep12 | 36.1 | +2.3 | 10.7 | CLIP |
| c013 | Washing a table | table | wash | 42 | 14 | 56 | 0.3 | 8.4 | 25.2 | ep18 | 13.8 | -5.4 | 23.8 | DINOv3 |
| c014 | Working at a table | table | work | 241 | 96 | 337 | 7.1 | 61.4 | 61.4 | ep13 | 48.5 | +12.9 | 18.6 | CLIP |
| c015 | Holding a phone/camera | phone/camera | hold | 1055 | 265 | 1320 | 13.3 | 73.2 | 73.3 | ep6 | 70.4 | +2.8 | 17.8 | CLIP |
| c016 | Playing with a phone/camera | phone/camera | play | 673 | 187 | 860 | 6.2 | 73.2 | 73.9 | ep6 | 67.4 | +5.8 | 51.2 | CLIP |
| c017 | Putting a phone/camera somewhere | phone/camera | put | 219 | 71 | 290 | 0.9 | 14.5 | 15.9 | ep16 | 9.5 | +5.0 | 9.6 | CLIP |
| c018 | Taking a phone/camera from somewhere | phone/camera | take | 298 | 115 | 413 | 1.9 | 25.7 | 26.8 | ep21 | 19.3 | +6.4 | 21.8 | CLIP |
| c019 | Talking on a phone/camera | phone/camera | talk | 217 | 42 | 259 | 1.4 | 75.9 | 79.5 | ep22 | 77.8 | -1.8 | 5.0 | CLIP |
| c020 | Holding a bag | bag | hold | 678 | 186 | 864 | 8.1 | 61.5 | 61.8 | ep11 | 58.8 | +2.7 | 10.8 | CLIP |
| c021 | Opening a bag | bag | open | 356 | 91 | 447 | 2.0 | 32.6 | 36.7 | ep10 | 31.1 | +1.5 | 40.7 | DINOv3 |
| c022 | Putting a bag somewhere | bag | put | 280 | 74 | 354 | 1.5 | 40.6 | 40.6 | ep13 | 25.1 | +15.5 | 39.2 | CLIP |
| c023 | Taking a bag from somewhere | bag | take | 278 | 76 | 354 | 1.3 | 25.4 | 25.7 | ep10 | 18.9 | +6.5 | 23.8 | CLIP |
| c024 | Throwing a bag somewhere | bag | throw | 89 | 21 | 110 | 0.2 | 7.1 | 15.6 | ep25 | 8.9 | -1.7 | 7.5 | DINOv3 |
| c025 | Closing a book | book | close | 187 | 68 | 255 | 0.6 | 30.9 | 34.4 | ep12 | 26.7 | +4.2 | 6.2 | CLIP |
| c026 | Holding a book | book | hold | 615 | 211 | 826 | 8.4 | 68.2 | 71.9 | ep11 | 66.3 | +2.0 | 56.1 | CLIP |
| c027 | Opening a book | book | open | 280 | 96 | 376 | 2.9 | 28.5 | 31.3 | ep11 | 27.4 | +1.1 | 35.7 | DINOv3 |
| c028 | Putting a book somewhere | book | put | 248 | 84 | 332 | 1.1 | 20.7 | 23.9 | ep27 | 18.8 | +2.0 | 22.5 | DINOv3 |
| c029 | Smiling at a book | book | smile | 92 | 54 | 146 | 0.9 | 27.2 | 33.5 | ep20 | 27.5 | -0.3 | 18.8 | CLIP |
| c030 | Taking a book from somewhere | book | take | 257 | 97 | 354 | 1.2 | 35.9 | 36.3 | ep11 | 24.8 | +11.1 | 26.7 | CLIP |
| c031 | Throwing a book somewhere | book | throw | 58 | 21 | 79 | 0.2 | 2.7 | 8.0 | ep6 | 2.6 | +0.2 | 8.1 | DINOv3 |
| c032 | Watching/Reading a book | book | watch | 483 | 150 | 633 | 4.2 | 67.1 | 70.0 | ep11 | 65.8 | +1.2 | 35.9 | CLIP |
| c033 | Holding a towel/s | towel | hold | 575 | 208 | 783 | 5.7 | 38.0 | 41.4 | ep20 | 37.9 | +0.2 | 11.8 | CLIP |
| c034 | Putting a towel/s somewhere | towel | put | 276 | 118 | 394 | 1.9 | 17.8 | 23.5 | ep20 | 15.9 | +1.9 | 29.4 | DINOv3 |
| c035 | Taking a towel/s from somewhere | towel | take | 286 | 109 | 395 | 2.0 | 13.6 | 17.4 | ep8 | 10.6 | +3.0 | 26.2 | DINOv3 |
| c036 | Throwing a towel/s somewhere | towel | throw | 154 | 65 | 219 | 0.7 | 13.6 | 13.6 | ep13 | 9.4 | +4.2 | 12.4 | CLIP |
| c037 | Tidying up a towel/s | towel | tidy | 125 | 64 | 189 | 1.8 | 14.7 | 22.6 | ep17 | 16.8 | -2.1 | 10.0 | CLIP |
| c038 | Washing something with a towel | towel | wash | 232 | 77 | 309 | 2.0 | 13.4 | 18.7 | ep16 | 12.6 | +0.7 | 9.0 | CLIP |
| c039 | Closing a box | box | close | 96 | 46 | 142 | 0.4 | 7.3 | 10.0 | ep8 | 6.5 | +0.7 | 4.8 | CLIP |
| c040 | Holding a box | box | hold | 331 | 115 | 446 | 3.0 | 42.2 | 42.2 | ep13 | 37.1 | +5.2 | 35.8 | CLIP |
| c041 | Opening a box | box | open | 181 | 68 | 249 | 1.1 | 26.4 | 26.4 | ep13 | 15.8 | +10.6 | 20.3 | CLIP |
| c042 | Putting a box somewhere | box | put | 201 | 72 | 273 | 1.2 | 20.6 | 22.7 | ep18 | 17.6 | +3.0 | 29.1 | DINOv3 |
| c043 | Taking a box from somewhere | box | take | 180 | 77 | 257 | 1.7 | 18.3 | 19.3 | ep22 | 14.0 | +4.2 | 32.4 | DINOv3 |
| c044 | Taking something from a box | box | take | 154 | 58 | 212 | 1.9 | 21.3 | 25.3 | ep9 | 15.5 | +5.8 | 11.7 | CLIP |
| c045 | Throwing a box somewhere | box | throw | 34 | 6 | 40 | 0.1 | 8.7 | 8.7 | ep13 | 0.4 | +8.3 | 18.1 | DINOv3 |
| c046 | Closing a laptop | laptop | close | 100 | 32 | 132 | 0.5 | 15.1 | 18.6 | ep19 | 9.6 | +5.5 | 5.0 | CLIP |
| c047 | Holding a laptop | laptop | hold | 259 | 105 | 364 | 4.1 | 73.1 | 74.3 | ep15 | 63.7 | +9.4 | 60.5 | CLIP |
| c048 | Opening a laptop | laptop | open | 99 | 30 | 129 | 0.6 | 15.8 | 22.1 | ep23 | 16.6 | -0.9 | 16.7 | DINOv3 |
| c049 | Putting a laptop somewhere | laptop | put | 91 | 49 | 140 | 0.8 | 36.6 | 37.8 | ep16 | 33.1 | +3.5 | 26.9 | CLIP |
| c050 | Taking a laptop from somewhere | laptop | take | 82 | 37 | 119 | 0.6 | 15.6 | 20.6 | ep8 | 14.3 | +1.3 | 21.2 | DINOv3 |
| c051 | Watching a laptop | laptop | watch | 347 | 131 | 478 | 5.6 | 64.5 | 66.0 | ep9 | 60.8 | +3.8 | 33.5 | CLIP |
| c052 | Working/Playing on a laptop | laptop | play | 341 | 95 | 436 | 6.8 | 75.9 | 78.8 | ep15 | 70.4 | +5.5 | 68.6 | CLIP |
| c053 | Holding a shoe/shoes | shoe | hold | 233 | 87 | 320 | 2.4 | 32.8 | 35.9 | ep19 | 29.7 | +3.1 | 5.2 | CLIP |
| c054 | Putting shoes somewhere | shoe | put | 177 | 73 | 250 | 1.3 | 21.1 | 21.1 | ep13 | 12.6 | +8.5 | 22.1 | DINOv3 |
| c055 | Putting on shoe/shoes | shoe | dress | 162 | 42 | 204 | 0.8 | 33.8 | 40.8 | ep16 | 28.5 | +5.3 | 14.3 | CLIP |
| c056 | Taking shoes from somewhere | shoe | take | 160 | 60 | 220 | 1.1 | 33.2 | 33.2 | ep13 | 22.9 | +10.3 | 26.4 | CLIP |
| c057 | Taking off some shoes | shoe | undress | 196 | 63 | 259 | 0.9 | 16.5 | 16.5 | ep20 | 12.6 | +3.8 | 15.9 | CLIP |
| c058 | Throwing shoes somewhere | shoe | throw | 73 | 32 | 105 | 0.5 | 6.2 | 15.0 | ep19 | 4.5 | +1.8 | 11.7 | DINOv3 |
| c059 | Sitting in a chair | chair | sit | 1260 | 368 | 1628 | 15.4 | 73.6 | 78.3 | ep12 | 69.1 | +4.6 | 51.3 | CLIP |
| c060 | Standing on a chair | chair | stand | 46 | 7 | 53 | 0.6 | 33.2 | 41.9 | ep15 | 3.4 | +29.8 | 1.3 | CLIP |
| c061 | Holding some food | food | hold | 1075 | 367 | 1442 | 11.5 | 63.1 | 63.3 | ep11 | 61.2 | +1.9 | 27.1 | CLIP |
| c062 | Putting some food somewhere | food | put | 495 | 211 | 706 | 4.8 | 24.7 | 27.5 | ep8 | 21.0 | +3.7 | 32.1 | DINOv3 |
| c063 | Taking food from somewhere | food | take | 683 | 245 | 928 | 3.9 | 29.2 | 29.5 | ep18 | 24.1 | +5.1 | 36.3 | DINOv3 |
| c064 | Throwing food somewhere | food | throw | 52 | 22 | 74 | 0.4 | 1.9 | 3.8 | ep7 | 1.0 | +0.9 | 2.8 | DINOv3 |
| c065 | Eating a sandwich | sandwich | eat | 459 | 111 | 570 | 2.9 | 46.3 | 46.5 | ep12 | 37.3 | +9.0 | 9.2 | CLIP |
| c066 | Making a sandwich | sandwich | make | 29 | 13 | 42 | 0.6 | 7.8 | 14.7 | ep18 | 8.2 | -0.4 | 12.1 | DINOv3 |
| c067 | Holding a sandwich | sandwich | hold | 400 | 120 | 520 | 3.3 | 46.5 | 50.0 | ep21 | 41.9 | +4.5 | 12.8 | CLIP |
| c068 | Putting a sandwich somewhere | sandwich | put | 154 | 46 | 200 | 0.8 | 11.4 | 12.7 | ep18 | 7.7 | +3.7 | 14.1 | DINOv3 |
| c069 | Taking a sandwich from somewhere | sandwich | take | 161 | 57 | 218 | 2.0 | 18.6 | 18.6 | ep13 | 10.0 | +8.7 | 13.0 | CLIP |
| c070 | Holding a blanket | blanket | hold | 443 | 174 | 617 | 6.5 | 50.8 | 57.2 | ep15 | 46.7 | +4.1 | 10.0 | CLIP |
| c071 | Putting a blanket somewhere | blanket | put | 205 | 97 | 302 | 1.4 | 18.5 | 23.2 | ep10 | 15.2 | +3.2 | 38.8 | DINOv3 |
| c072 | Snuggling with a blanket | blanket | snuggle | 323 | 109 | 432 | 3.9 | 57.6 | 68.6 | ep29 | 66.0 | -8.4 | 24.7 | CLIP |
| c073 | Taking a blanket from somewhere | blanket | take | 200 | 79 | 279 | 1.0 | 33.1 | 33.1 | ep13 | 27.6 | +5.5 | 23.4 | CLIP |
| c074 | Throwing a blanket somewhere | blanket | throw | 105 | 61 | 166 | 0.9 | 13.5 | 14.4 | ep14 | 9.6 | +3.8 | 16.3 | DINOv3 |
| c075 | Tidying up a blanket/s | blanket | tidy | 139 | 59 | 198 | 1.5 | 21.4 | 28.7 | ep6 | 18.3 | +3.1 | 27.2 | DINOv3 |
| c076 | Holding a pillow | pillow | hold | 385 | 116 | 501 | 3.6 | 53.0 | 53.0 | ep13 | 50.5 | +2.5 | 13.8 | CLIP |
| c077 | Putting a pillow somewhere | pillow | put | 171 | 58 | 229 | 1.0 | 22.1 | 22.9 | ep16 | 18.3 | +3.8 | 24.2 | DINOv3 |
| c078 | Snuggling with a pillow | pillow | snuggle | 159 | 67 | 226 | 1.7 | 43.1 | 46.7 | ep11 | 37.8 | +5.2 | 21.5 | CLIP |
| c079 | Taking a pillow from somewhere | pillow | take | 161 | 51 | 212 | 0.5 | 12.9 | 19.4 | ep14 | 16.6 | -3.7 | 21.8 | DINOv3 |
| c080 | Throwing a pillow somewhere | pillow | throw | 134 | 28 | 162 | 0.3 | 24.4 | 24.4 | ep13 | 22.9 | +1.5 | 12.2 | CLIP |
| c081 | Putting something on a shelf | shelf | put | 503 | 154 | 657 | 2.2 | 21.4 | 25.7 | ep22 | 23.1 | -1.7 | 11.0 | CLIP |
| c082 | Tidying a shelf | shelf | tidy | 228 | 92 | 320 | 2.4 | 19.7 | 28.2 | ep12 | 22.7 | -3.0 | 24.5 | DINOv3 |
| c083 | Reaching for and grabbing a picture | picture | take | 74 | 45 | 119 | 1.6 | 7.9 | 12.5 | ep12 | 6.1 | +1.7 | 6.3 | CLIP |
| c084 | Holding a picture | picture | hold | 133 | 57 | 190 | 1.7 | 16.9 | 20.7 | ep12 | 11.0 | +5.9 | 12.9 | CLIP |
| c085 | Laughing at a picture | picture | laugh | 28 | 15 | 43 | 0.2 | 13.0 | 24.1 | ep17 | 12.2 | +0.8 | 4.1 | CLIP |
| c086 | Putting a picture somewhere | picture | put | 62 | 30 | 92 | 0.5 | 17.8 | 18.4 | ep8 | 10.7 | +7.2 | 8.0 | CLIP |
| c087 | Taking a picture of something | phone/camera | photograph | 231 | 71 | 302 | 1.2 | 26.4 | 32.5 | ep8 | 23.3 | +3.1 | 5.7 | CLIP |
| c088 | Watching/looking at a picture | picture | watch | 198 | 78 | 276 | 1.9 | 16.4 | 19.1 | ep16 | 11.8 | +4.6 | 6.8 | CLIP |
| c089 | Closing a window | window | close | 53 | 10 | 63 | 0.2 | 10.2 | 12.8 | ep8 | 3.1 | +7.1 | 1.7 | CLIP |
| c090 | Opening a window | window | open | 87 | 17 | 104 | 0.3 | 29.7 | 29.7 | ep13 | 10.3 | +19.5 | 27.8 | CLIP |
| c091 | Washing a window | window | wash | 39 | 7 | 46 | 0.3 | 35.1 | 68.7 | ep20 | 43.5 | -8.3 | 40.9 | DINOv3 |
| c092 | Watching outside of a window | window | watch | 305 | 88 | 393 | 2.1 | 40.1 | 46.2 | ep12 | 36.4 | +3.7 | 25.2 | CLIP |
| c093 | Holding a mirror | mirror | hold | 94 | 24 | 118 | 0.9 | 28.9 | 33.6 | ep14 | 16.4 | +12.5 | 2.5 | CLIP |
| c094 | Smiling in a mirror | mirror | smile | 111 | 42 | 153 | 0.7 | 24.8 | 29.9 | ep16 | 12.5 | +12.3 | 23.6 | CLIP |
| c095 | Washing a mirror | mirror | wash | 37 | 9 | 46 | 0.3 | 7.4 | 16.2 | ep7 | 8.4 | -0.9 | 4.7 | CLIP |
| c096 | Watching in a mirror | mirror | watch | 449 | 134 | 583 | 3.3 | 39.2 | 44.4 | ep12 | 34.5 | +4.7 | 40.5 | DINOv3 |
| c097 | Walking through a doorway | doorway | walk | 1444 | 439 | 1883 | 4.9 | 43.8 | 44.9 | ep21 | 38.8 | +5.0 | 35.3 | CLIP |
| c098 | Holding a broom | broom | hold | 364 | 106 | 470 | 2.8 | 75.4 | 78.3 | ep19 | 67.7 | +7.7 | 7.9 | CLIP |
| c099 | Putting a broom somewhere | broom | put | 100 | 44 | 144 | 0.5 | 27.4 | 31.9 | ep22 | 21.1 | +6.3 | 53.0 | DINOv3 |
| c100 | Taking a broom from somewhere | broom | take | 127 | 53 | 180 | 0.5 | 34.9 | 42.3 | ep12 | 22.4 | +12.5 | 44.6 | DINOv3 |
| c101 | Throwing a broom somewhere | broom | throw | 25 | 8 | 33 | 0.1 | 1.5 | 2.3 | ep28 | 1.3 | +0.1 | 9.1 | DINOv3 |
| c102 | Tidying up with a broom | broom | tidy | 237 | 56 | 293 | 1.5 | 63.0 | 70.6 | ep19 | 61.8 | +1.2 | 60.8 | CLIP |
| c103 | Fixing a light | light | fix | 36 | 15 | 51 | 1.3 | 24.5 | 31.9 | ep19 | 24.4 | +0.1 | 1.0 | CLIP |
| c104 | Turning on a light | light | turn | 207 | 73 | 280 | 0.8 | 8.7 | 10.1 | ep27 | 5.0 | +3.7 | 7.5 | CLIP |
| c105 | Turning off a light | light | turn | 189 | 50 | 239 | 0.2 | 2.3 | 5.9 | ep36 | 4.1 | -1.8 | 7.9 | DINOv3 |
| c106 | Drinking from a cup/glass/bottle | cup | drink | 1134 | 300 | 1434 | 5.8 | 65.5 | 66.5 | ep14 | 60.1 | +5.4 | 19.9 | CLIP |
| c107 | Holding a cup/glass/bottle | cup | hold | 1014 | 380 | 1394 | 11.0 | 67.2 | 69.0 | ep10 | 62.0 | +5.2 | 63.5 | CLIP |
| c108 | Pouring into a cup/glass/bottle | cup | pour | 276 | 56 | 332 | 1.0 | 20.9 | 27.3 | ep9 | 20.0 | +0.9 | 8.0 | CLIP |
| c109 | Putting a cup/glass/bottle somewhere | cup | put | 481 | 211 | 692 | 2.0 | 22.3 | 27.1 | ep14 | 21.0 | +1.3 | 30.4 | DINOv3 |
| c110 | Taking a cup/glass/bottle from somewhere | cup | take | 536 | 240 | 776 | 3.3 | 27.5 | 29.4 | ep10 | 22.1 | +5.5 | 37.5 | DINOv3 |
| c111 | Washing a cup/glass/bottle | cup | wash | 49 | 20 | 69 | 0.4 | 12.7 | 12.7 | ep13 | 4.8 | +7.9 | 4.8 | CLIP |
| c112 | Closing a closet/cabinet | closet | close | 410 | 134 | 544 | 1.2 | 19.4 | 19.4 | ep13 | 13.6 | +5.7 | 13.6 | CLIP |
| c113 | Opening a closet/cabinet | closet | open | 595 | 199 | 794 | 3.5 | 41.2 | 41.2 | ep13 | 37.5 | +3.7 | 51.4 | DINOv3 |
| c114 | Tidying up a closet/cabinet | closet | tidy | 218 | 83 | 301 | 2.7 | 29.3 | 34.7 | ep8 | 21.9 | +7.4 | 22.2 | CLIP |
| c115 | Holding a paper/notebook | paper | hold | 324 | 145 | 469 | 4.2 | 28.4 | 42.6 | ep10 | 30.0 | -1.6 | 11.4 | CLIP |
| c116 | Putting their paper/notebook somewhere | paper | put | 168 | 59 | 227 | 0.9 | 6.2 | 11.2 | ep11 | 5.7 | +0.5 | 16.5 | DINOv3 |
| c117 | Taking paper/notebook from somewhere | paper | take | 175 | 60 | 235 | 0.9 | 12.4 | 15.0 | ep11 | 10.2 | +2.2 | 15.2 | DINOv3 |
| c118 | Holding a dish | dish | hold | 763 | 267 | 1030 | 7.8 | 38.2 | 42.1 | ep10 | 34.9 | +3.3 | 16.4 | CLIP |
| c119 | Putting a dish somewhere | dish | put | 452 | 145 | 597 | 2.8 | 14.3 | 17.1 | ep21 | 13.5 | +0.8 | 28.1 | DINOv3 |
| c120 | Taking a dish from somewhere | dish | take | 375 | 146 | 521 | 2.3 | 15.9 | 16.6 | ep15 | 15.3 | +0.7 | 29.3 | DINOv3 |
| c121 | Washing a dish | dish | wash | 112 | 30 | 142 | 1.6 | 54.2 | 54.2 | ep13 | 39.8 | +14.4 | 19.5 | CLIP |
| c122 | Lying on a sofa/couch | sofa | lie | 160 | 62 | 222 | 3.5 | 46.9 | 48.6 | ep9 | 44.3 | +2.6 | 7.7 | CLIP |
| c123 | Sitting on sofa/couch | sofa | sit | 484 | 189 | 673 | 6.5 | 59.9 | 60.4 | ep10 | 51.2 | +8.7 | 51.0 | CLIP |
| c124 | Lying on the floor | floor | lie | 168 | 48 | 216 | 1.3 | 55.1 | 59.2 | ep17 | 52.2 | +3.0 | 4.0 | CLIP |
| c125 | Sitting on the floor | floor | sit | 395 | 142 | 537 | 5.0 | 54.1 | 57.5 | ep16 | 52.4 | +1.7 | 41.7 | CLIP |
| c126 | Throwing something on the floor | floor | throw | 308 | 135 | 443 | 1.5 | 11.4 | 14.4 | ep16 | 6.9 | +4.5 | 10.3 | CLIP |
| c127 | Tidying something on the floor | floor | tidy | 478 | 152 | 630 | 4.2 | 53.1 | 58.2 | ep14 | 51.5 | +1.6 | 36.5 | CLIP |
| c128 | Holding some medicine | medicine | hold | 282 | 71 | 353 | 2.0 | 16.4 | 19.7 | ep29 | 18.1 | -1.6 | 5.5 | CLIP |
| c129 | Taking/consuming some medicine | medicine | eat | 162 | 37 | 199 | 0.7 | 9.7 | 14.3 | ep17 | 8.8 | +0.8 | 12.5 | DINOv3 |
| c130 | Putting groceries somewhere | groceries | put | 207 | 80 | 287 | 1.9 | 34.0 | 35.3 | ep16 | 24.9 | +9.1 | 5.6 | CLIP |
| c131 | Laughing at television | tv | laugh | 46 | 13 | 59 | 0.3 | 11.2 | 25.1 | ep40 | 23.8 | -12.6 | 1.6 | CLIP |
| c132 | Watching television | tv | watch | 353 | 108 | 461 | 2.8 | 61.3 | 65.9 | ep18 | 63.3 | -2.0 | 48.1 | CLIP |
| c133 | Awakening in bed | bed | awaken | 98 | 45 | 143 | 1.2 | 49.0 | 59.5 | ep21 | 49.7 | -0.7 | 10.4 | CLIP |
| c134 | Lying on a bed | bed | lie | 257 | 78 | 335 | 3.5 | 69.9 | 73.3 | ep9 | 65.6 | +4.3 | 82.8 | DINOv3 |
| c135 | Sitting in a bed | bed | sit | 315 | 335 | 650 | 6.2 | 59.1 | 59.1 | ep25 | 56.4 | +2.6 | 44.7 | CLIP |
| c136 | Fixing a vacuum | vacuum | fix | 35 | 143 | 178 | 0.4 | 20.3 | 32.6 | ep8 | 8.4 | +11.9 | 1.1 | CLIP |
| c137 | Holding a vacuum | vacuum | hold | 213 | 9 | 222 | 2.1 | 51.3 | 62.0 | ep10 | 52.7 | -1.4 | 74.2 | DINOv3 |
| c138 | Taking a vacuum from somewhere | vacuum | take | 57 | 69 | 126 | 0.6 | 20.9 | 26.1 | ep15 | 20.3 | +0.6 | 37.9 | DINOv3 |
| c139 | Washing their hands | hands | wash | 81 | 25 | 106 | 1.0 | 26.2 | 35.0 | ep10 | 21.2 | +5.0 | 1.6 | CLIP |
| c140 | Fixing a doorknob | doorknob | fix | 31 | 33 | 64 | 0.4 | 38.5 | 47.1 | ep25 | 39.4 | -0.9 | 2.1 | CLIP |
| c141 | Grasping onto a doorknob | doorknob | grasp | 506 | 20 | 526 | 4.0 | 40.4 | 40.6 | ep15 | 29.6 | +10.8 | 37.1 | CLIP |
| c142 | Closing a refrigerator | refrigerator | close | 192 | 238 | 430 | 0.5 | 48.5 | 54.3 | ep21 | 50.5 | -2.0 | 4.7 | CLIP |
| c143 | Opening a refrigerator | refrigerator | open | 250 | 45 | 295 | 0.6 | 60.2 | 60.6 | ep16 | 55.4 | +4.8 | 88.1 | DINOv3 |
| c144 | Fixing their hair | hair | fix | 165 | 58 | 223 | 1.0 | 30.5 | 34.6 | ep14 | 18.1 | +12.3 | 4.9 | CLIP |
| c145 | Working on paper/notebook | paper | work | 237 | 60 | 297 | 2.4 | 58.4 | 63.1 | ep8 | 54.4 | +3.9 | 14.3 | CLIP |
| c146 | Awakening somewhere | None | awaken | 245 | 74 | 319 | 1.9 | 53.1 | 61.0 | ep8 | 52.9 | +0.1 | 7.5 | CLIP |
| c147 | Someone is cooking | food | cook | 338 | 85 | 423 | 2.1 | 79.0 | 81.7 | ep17 | 77.7 | +1.3 | 5.5 | CLIP |
| c148 | Someone is dressing | clothes | dress | 332 | 76 | 408 | 3.4 | 50.0 | 50.0 | ep19 | 44.2 | +5.8 | 7.4 | CLIP |
| c149 | Someone is laughing | None | laugh | 588 | 120 | 708 | 4.4 | 52.4 | 53.8 | ep15 | 45.2 | +7.2 | 15.2 | CLIP |
| c150 | Someone is running | None | run | 357 | 211 | 568 | 2.5 | 18.9 | 22.6 | ep24 | 19.6 | -0.8 | 9.1 | CLIP |
| c151 | Standing to sitting | None | sit | 964 | 112 | 1076 | 4.0 | 60.4 | 60.9 | ep17 | 58.2 | +2.2 | 23.2 | CLIP |
| c152 | Someone is smiling | None | smile | 1029 | 279 | 1308 | 7.4 | 49.5 | 50.6 | ep14 | 41.6 | +8.0 | 23.5 | CLIP |
| c153 | Someone is sneezing | None | sneeze | 643 | 393 | 1036 | 2.7 | 17.8 | 25.5 | ep14 | 21.2 | -3.4 | 11.8 | CLIP |
| c154 | Someone is standing up | None | stand | 1341 | 175 | 1516 | 7.3 | 36.8 | 36.8 | ep13 | 33.6 | +3.1 | 27.9 | CLIP |
| c155 | Someone is undressing | clothes | undress | 395 | 381 | 776 | 2.8 | 56.7 | 61.3 | ep17 | 59.2 | -2.6 | 7.7 | CLIP |
| c156 | Someone is eating | food | eat | 936 | 134 | 1070 | 9.5 | 58.8 | 60.1 | ep12 | 51.8 | +7.1 | 24.2 | CLIP |

---

## 8. AP Distribution Analysis

### 8.1 Top-20 CLIP Classes (AP₁₃ descending)

| Rank | Code | Class | Tr | AP₁₃ | AP₄₉ | Δ |
|------|------|-------|----|-----:|-----:|--:|
| 1 | c147 | Someone is cooking | 338 | 79.0 | 77.7 | +1.3 |
| 2 | c019 | Talking on a phone/camera | 217 | 75.9 | 77.8 | -1.8 |
| 3 | c052 | Working/Playing on a laptop | 341 | 75.9 | 70.4 | +5.5 |
| 4 | c011 | Sitting at a table | 728 | 75.5 | 75.4 | +0.1 |
| 5 | c098 | Holding a broom | 364 | 75.4 | 67.7 | +7.7 |
| 6 | c059 | Sitting in a chair | 1260 | 73.6 | 69.1 | +4.6 |
| 7 | c015 | Holding a phone/camera | 1055 | 73.2 | 70.4 | +2.8 |
| 8 | c016 | Playing with a phone/camera | 673 | 73.2 | 67.4 | +5.8 |
| 9 | c047 | Holding a laptop | 259 | 73.1 | 63.7 | +9.4 |
| 10 | c134 | Lying on a bed | 257 | 69.9 | 65.6 | +4.3 |
| 11 | c026 | Holding a book | 615 | 68.2 | 66.3 | +2.0 |
| 12 | c107 | Holding a cup/glass/bottle | 1014 | 67.2 | 62.0 | +5.2 |
| 13 | c032 | Watching/Reading a book | 483 | 67.1 | 65.8 | +1.2 |
| 14 | c106 | Drinking from a cup/glass/bottle | 1134 | 65.5 | 60.1 | +5.4 |
| 15 | c051 | Watching a laptop | 347 | 64.5 | 60.8 | +3.8 |
| 16 | c061 | Holding some food | 1075 | 63.1 | 61.2 | +1.9 |
| 17 | c102 | Tidying up with a broom | 237 | 63.0 | 61.8 | +1.2 |
| 18 | c020 | Holding a bag | 678 | 61.5 | 58.8 | +2.7 |
| 19 | c014 | Working at a table | 241 | 61.4 | 48.5 | +12.9 |
| 20 | c132 | Watching television | 353 | 61.3 | 63.3 | -2.0 |

### 8.2 Bottom-20 CLIP Classes (AP₁₃ ascending)

| Rank | Code | Class | Tr | AP₁₃ | AP_peak | Pk_ep | AP_DINOv3 |
|------|------|-------|----|-----:|--------:|------:|----------:|
| 1 | c101 | Throwing a broom somewhere | 25 | 1.5 | 2.3 | ep28 | 9.1 |
| 2 | c064 | Throwing food somewhere | 52 | 1.9 | 3.8 | ep7 | 2.8 |
| 3 | c105 | Turning off a light | 189 | 2.3 | 5.9 | ep36 | 7.9 |
| 4 | c031 | Throwing a book somewhere | 58 | 2.7 | 8.0 | ep6 | 8.1 |
| 5 | c010 | Sitting on a table | 46 | 3.8 | 8.5 | ep20 | 8.5 |
| 6 | c116 | Putting their paper/notebook somewhere | 168 | 6.2 | 11.2 | ep11 | 16.5 |
| 7 | c058 | Throwing shoes somewhere | 73 | 6.2 | 15.0 | ep19 | 11.7 |
| 8 | c024 | Throwing a bag somewhere | 89 | 7.1 | 15.6 | ep25 | 7.5 |
| 9 | c039 | Closing a box | 96 | 7.3 | 10.0 | ep8 | 4.8 |
| 10 | c095 | Washing a mirror | 37 | 7.4 | 16.2 | ep7 | 4.7 |
| 11 | c066 | Making a sandwich | 29 | 7.8 | 14.7 | ep18 | 12.1 |
| 12 | c083 | Reaching for and grabbing a picture | 74 | 7.9 | 12.5 | ep12 | 6.3 |
| 13 | c013 | Washing a table | 42 | 8.4 | 25.2 | ep18 | 23.8 |
| 14 | c045 | Throwing a box somewhere | 34 | 8.7 | 8.7 | ep13 | 18.1 |
| 15 | c104 | Turning on a light | 207 | 8.7 | 10.1 | ep27 | 7.5 |
| 16 | c129 | Taking/consuming some medicine | 162 | 9.7 | 14.3 | ep17 | 12.5 |
| 17 | c089 | Closing a window | 53 | 10.2 | 12.8 | ep8 | 1.7 |
| 18 | c131 | Laughing at television | 46 | 11.2 | 25.1 | ep40 | 1.6 |
| 19 | c068 | Putting a sandwich somewhere | 154 | 11.4 | 12.7 | ep18 | 14.1 |
| 20 | c126 | Throwing something on the floor | 308 | 11.4 | 14.4 | ep16 | 10.3 |

### 8.3 Classes Where DINOv3 Outperforms CLIP

The real DINOv3 pattern: manipulation actions with visually specific objects (vacuum, refrigerator, broom) and put/take actions on objects with low salience in CLIP's web corpus.

| Code | Class | AP_CLIP | AP_DINOv3 reg v2 ep13 | AP_DINOv3 best | DINOv3 advantage |
|------|-------|--------:|----------------------:|---------------:|:----------------:|
| c143 | Opening a refrigerator | 60.2 | 83.8 | 88.1 | +27.9 |
| c099 | Putting a broom somewhere | 27.4 | 47.7 | 53.0 | +25.6 |
| c137 | Holding a vacuum | 51.3 | 59.3 | 74.2 | +22.9 |
| c071 | Putting a blanket somewhere | 18.5 | 29.9 | 38.8 | +20.3 |
| c138 | Taking a vacuum from somewhere | 20.9 | 27.7 | 37.9 | +17.0 |
| c002 | Taking some clothes | 38.6 | 48.6 | 54.7 | +16.1 |
| c013 | Washing a table | 8.4 | 17.0 | 23.8 | +15.4 |
| c001 | Putting clothes somewhere | 32.2 | 45.3 | 46.5 | +14.3 |
| c043 | Taking a box from somewhere | 18.3 | 31.3 | 32.4 | +14.1 |
| c119 | Putting a dish somewhere | 14.3 | 21.7 | 28.1 | +13.8 |
| c120 | Taking a dish from somewhere | 15.9 | 26.9 | 29.3 | +13.4 |
| c134 | Lying on a bed | 69.9 | 72.1 | 82.8 | +12.9 |
| c035 | Taking a towel | 13.6 | 26.0 | 26.2 | +12.6 |
| c034 | Putting a towel | 17.8 | 29.4 | 29.4 | +11.6 |
| c116 | Putting their paper/notebook | 6.2 | 16.4 | 16.5 | +10.3 |
| c113 | Opening a closet/cabinet | 41.2 | 51.4 | 51.4 | +10.2 |
| c110 | Taking a cup/glass/bottle | 27.5 | 35.4 | 37.5 | +10.0 |
| c100 | Taking a broom from somewhere | 34.9 | 33.3 | 44.6 | +9.7 |
| c003 | Throwing clothes | 20.7 | 25.8 | 30.2 | +9.5 |
| c045 | Throwing a box somewhere | 8.7 | 18.1 | 18.1 | +9.4 |

---

## 9. Per-Object Analysis (38 Categories)

Mean AP₁₃ CLIP for groups of classes sharing the same object (sorted by descending mean AP).

| Obj | Label | #Classes | Mean AP CLIP | Max AP | Min AP | Total occ. |
|-----|-------|--------:|-------------:|-------:|-------:|----------:|
| o002 | bed | 3 | 59.3 | 69.9 | 49.0 | 1,128 |
| o028 | refrigerator | 2 | 54.3 | 60.2 | 48.5 | 725 |
| o007 | chair | 2 | 53.4 | 73.6 | 33.2 | 1,681 |
| o032 | sofa/couch | 2 | 53.4 | 59.9 | 46.9 | 895 |
| o025 | phone/camera | 6 | 48.2 | 75.9 | 14.5 | 3,444 |
| o014 | doorway | 1 | 43.8 | 43.8 | 43.8 | 1,883 |
| o015 | floor | 4 | 43.4 | 55.1 | 11.4 | 1,826 |
| o016 | food | 6 | 42.8 | 79.0 | 1.9 | 4,643 |
| o020 | laptop | 7 | 42.4 | 75.9 | 15.1 | 1,798 |
| o000 | None | 7 | 41.3 | 60.4 | 17.8 | 6,531 |
| o006 | broom | 5 | 40.4 | 75.4 | 1.5 | 1,120 |
| o013 | doorknob | 2 | 39.4 | 40.4 | 38.5 | 590 |
| o009 | clothes | 8 | 36.9 | 56.7 | 19.5 | 4,435 |
| o034 | television | 2 | 36.2 | 61.3 | 11.2 | 520 |
| o010 | cup/glass/bottle | 6 | 36.0 | 67.2 | 12.7 | 4,697 |
| o004 | book | 8 | 35.2 | 68.2 | 2.7 | 3,001 |
| o033 | table | 6 | 35.1 | 75.5 | 3.8 | 2,818 |
| o012 | door | 3 | 34.5 | 39.8 | 30.3 | 1,934 |
| o017 | groceries | 1 | 34.0 | 34.0 | 34.0 | 287 |
| o001 | bag | 5 | 33.5 | 61.5 | 7.1 | 2,129 |
| o003 | blanket | 6 | 32.5 | 57.6 | 13.5 | 1,994 |
| o027 | pillow | 5 | 31.1 | 53.0 | 12.9 | 1,330 |
| o036 | vacuum | 3 | 30.8 | 51.3 | 20.3 | 526 |
| o011 | dish | 4 | 30.7 | 54.2 | 14.3 | 2,290 |
| o018 | hair | 1 | 30.5 | 30.5 | 30.5 | 223 |
| o008 | closet/cabinet | 3 | 29.9 | 41.2 | 19.4 | 1,639 |
| o037 | window | 4 | 28.8 | 40.1 | 10.2 | 606 |
| o024 | paper/notebook | 4 | 26.4 | 58.4 | 6.2 | 1,228 |
| o019 | hands | 1 | 26.2 | 26.2 | 26.2 | 106 |
| o029 | sandwich | 5 | 26.1 | 46.5 | 7.8 | 1,550 |
| o023 | mirror | 4 | 25.1 | 39.2 | 7.4 | 900 |
| o031 | shoe | 6 | 23.9 | 33.8 | 6.2 | 1,358 |
| o005 | box | 7 | 20.7 | 42.2 | 7.3 | 1,619 |
| o030 | shelf | 2 | 20.5 | 21.4 | 19.7 | 977 |
| o035 | towel | 6 | 18.5 | 38.0 | 13.4 | 2,289 |
| o026 | picture | 5 | 14.4 | 17.8 | 7.9 | 720 |
| o022 | medicine | 2 | 13.0 | 16.4 | 9.7 | 552 |
| o021 | light | 3 | 11.8 | 24.5 | 2.3 | 570 |

---

## 10. Per-Verb Analysis (33 Categories)

| Verb | Label | #Classes | Mean AP₁₃ | Mean AP₄₉ | Mean Δ | Interpretation |
|------|-------|--------:|----------:|----------:|-------:|----------------|
| v002 | cook | 1 | 79.0 | 77.7 | +1.3 | High textual semantics (CLIP) |
| v024 | talk | 1 | 75.9 | 77.8 | -1.8 | |
| v014 | play | 2 | 74.5 | 68.9 | +5.6 | Confused with hold (v008) |
| v004 | drink | 1 | 65.5 | 60.1 | +5.4 | |
| v032 | work | 2 | 59.9 | 51.5 | +8.4 | Confused with v018 (sit) and v008 (hold) |
| v010 | lie | 3 | 57.3 | 54.0 | +3.3 | Horizontal posture → good generalisation |
| v018 | sit | 7 | 55.2 | 52.9 | +2.2 | Static posture → easy to generalise |
| v000 | awaken | 2 | 51.0 | 51.3 | -0.3 | |
| v021 | snuggle | 2 | 50.3 | 51.9 | -1.6 | Characteristic physical contact |
| v008 | hold | 20 | 48.3 | 44.5 | +3.8 | Prominent object in hand → CLIP very strong |
| v031 | watch | 6 | 48.1 | 45.4 | +2.7 | Gaze towards target → low variance |
| v029 | walk | 1 | 43.8 | 38.8 | +5.0 | |
| v003 | dress | 2 | 41.9 | 36.3 | +5.5 | |
| v007 | grasp | 1 | 40.4 | 29.6 | +10.8 | |
| v005 | eat | 3 | 38.3 | 32.6 | +5.6 | |
| v028 | undress | 2 | 36.6 | 35.9 | +0.6 | |
| v022 | stand | 2 | 35.0 | 18.5 | +16.5 | Postural transition → unstable |
| v026 | tidy | 8 | 34.5 | 32.7 | +1.8 | |
| v012 | open | 8 | 34.3 | 29.0 | +5.2 | |
| v019 | smile | 3 | 33.8 | 27.2 | +6.6 | |
| v006 | fix | 5 | 29.4 | 22.4 | +7.0 | Requires mechanical understanding |
| v013 | photograph | 1 | 26.4 | 23.3 | +3.1 | Confused with hold phone (c015) |
| v009 | laugh | 3 | 25.5 | 27.0 | -1.5 | Facial expression → poor |
| v023 | take | 19 | 23.2 | 18.3 | +4.9 | |
| v001 | close | 7 | 23.1 | 18.4 | +4.7 | |
| v016 | put | 20 | 22.4 | 18.0 | +4.4 | |
| v030 | wash | 8 | 22.1 | 20.7 | +1.4 | Repetitive manual action → recognisable |
| v015 | pour | 1 | 20.9 | 20.0 | +0.9 | |
| v017 | run | 1 | 18.9 | 19.6 | -0.8 | |
| v020 | sneeze | 1 | 17.8 | 21.2 | -3.4 | |
| v025 | throw | 11 | 10.2 | 7.5 | +2.7 | Brief gesture → few occurrences, rare in corpus |
| v011 | make | 1 | 7.8 | 8.2 | -0.4 | Very rare (sandwich only) |
| v027 | turn | 2 | 5.5 | 4.6 | +0.9 | Requires pre/post state → problematic |

---

## 11. Confusable Class Pairs

### 11.1 Pairs with Same (Object, Verb) — Structural Ambiguity

| Pair | Class A | Class B | AP_A | AP_B | Problem |
|------|---------|---------|-----:|-----:|---------|
| c010/c011 | Sitting on a table | Sitting at a table | 3.8 | 75.5 | Same code (o033,v018) — 'sitting ON table' vs 'sitting AT table'. Purely prepositional distinction, impossible from visual features. |
| c104/c105 | Turning on a light | Turning off a light | 8.7 | 2.3 | Same code (o021,v027) — requires understanding of light state before/after. |
| c043/c044 | Taking a box from somewhere | Taking something from a box | 18.3 | 21.3 | Same code (o005,v023) — direction of action is ambiguous. |

### 11.2 Visually Similar Pairs (Shared Object)

| Code A | Code B | Class A | Class B | AP_A | AP_B | Common element |
|--------|--------|---------|---------|-----:|-----:|----------------|
| c006 | c008 | Closing a door | Opening a door | 30.3 | 39.8 | door |
| c097 | c141 | Walking through a doorway | Grasping onto a doorknob | 43.8 | 40.4 | doorway/doorknob |
| c015 | c019 | Holding a phone/camera | Talking on a phone/camera | 73.2 | 75.9 | phone |
| c047 | c052 | Holding a laptop | Working/Playing on a laptop | 73.1 | 75.9 | laptop |
| c059 | c123 | Sitting in a chair | Sitting on sofa/couch | 73.6 | 59.9 | sit |
| c098 | c102 | Holding a broom | Tidying up with a broom | 75.4 | 63.0 | broom |
| c132 | c131 | Watching television | Laughing at television | 61.3 | 11.2 | tv |
| c133 | c134 | Awakening in bed | Lying on a bed | 49.0 | 69.9 | bed |

---

## 12. Compositional Structure: Object × Verb Mapping

### 12.1 Objects with Highest Mean AP CLIP

| Rank | Object | #Classes | Mean AP₁₃ |
|------|--------|--------:|----------:|
| 1 | bed (o002) | 3 | 59.3 |
| 2 | refrigerator (o028) | 2 | 54.3 |
| 3 | chair (o007) | 2 | 53.4 |
| 4 | sofa/couch (o032) | 2 | 53.4 |
| 5 | phone/camera (o025) | 6 | 48.2 |
| 6 | doorway (o014) | 1 | 43.8 |
| 7 | floor (o015) | 4 | 43.4 |
| 8 | food (o016) | 6 | 42.8 |
| 9 | laptop (o020) | 7 | 42.4 |
| 10 | None (o000) | 7 | 41.3 |

### 12.2 Objects with Lowest Mean AP CLIP

| Rank | Object | #Classes | Mean AP₁₃ | Reason |
|------|--------|--------:|----------:|--------|
| 1 | shoe (o031) | 6 | 23.9 | |
| 2 | box (o005) | 7 | 20.7 | |
| 3 | shelf (o030) | 2 | 20.5 | |
| 4 | towel (o035) | 6 | 18.5 | |
| 5 | picture (o026) | 5 | 14.4 | Low frequency, high variance |
| 6 | medicine (o022) | 2 | 13.0 | Small object, difficult to localise |
| 7 | light (o021) | 3 | 11.8 | Light state on/off indistinguishable |

### 12.3 Structurally Duplicate (Object, Verb) Pairs

| Pairs | Classes | AP_A | AP_B | Training ratio | Analysis |
|-------|---------|-----:|-----:|:--------------:|---------|
| (o033, v018) | c010 vs c011 | 3.8 | 75.5 | 46 vs 728 | Sitting *on* vs *at* a table — impossible from visual features. The AP gap is entirely explained by frequency. |
| (o021, v027) | c104 vs c105 | 8.7 | 2.3 | 207 vs 189 | Turning *on* vs *off* a light — requires understanding of light state. |
| (o005, v023) | c043 vs c044 | 18.3 | 21.3 | 180 vs 154 | Taking *a box from somewhere* vs *something from a box* — ambiguous direction. |

---

## 13. Structurally Ambiguous Classes and Dataset Issues

| Code | Class | Problem | Train | Test | AP_CLIP | AP_DINOv3 |
|------|-------|---------|------:|-----:|--------:|----------:|
| c010 | Sitting on a table | Structurally identical to c011 (same mapping). AP perpetually low due to rarity. | 46 | 16 | 3.8 | 8.5 |
| c104 | Turning on a light | Identical mapping to c105. Near-zero AP for both. | 207 | 73 | 8.7 | 7.5 |
| c105 | Turning off a light | Identical mapping to c104. Near-zero AP for both. | 189 | 50 | 2.3 | 7.9 |
| c101 | Throwing a broom somewhere | Only 33 total occurrences. AP never above 2.3. | 25 | 8 | 1.5 | 9.1 |
| c064 | Throwing food somewhere | 74 total occurrences. AP always < 5. | 52 | 22 | 1.9 | 2.8 |
| c085 | Laughing at a picture | 43 occurrences. Subtle emotion. | 28 | 15 | 13.0 | 4.1 |
| c031 | Throwing a book somewhere | 79 occurrences. Very rare action. | 58 | 21 | 2.7 | 8.1 |
| c045 | Throwing a box somewhere | 40 occurrences. Collapse at ep49: AP = 0.4. | 34 | 6 | 8.7 | 18.1 |
| c060 | Standing on a chair | 53 occurrences. Extreme collapse from 33.2 (ep13) to 3.4 (ep49). | 46 | 7 | 33.2 | 1.3 |
| c141 | Grasping onto a doorknob | Anomalous split: 506 train but only 20 test. | 506 | 20 | 40.4 | 37.1 |
| c136 | Fixing a vacuum | Inverted ratio: 143 test vs 35 train. | 35 | 143 | 20.3 | 1.1 |

---

## 14. Recommendations and Future Directions

### 14.1 For Low-AP Classes
- **Duplicate classes (c010/c011, c104/c105, c043/c044)**: merging or exclusion from evaluation, or addition of a dedicated state-detection module.
- **Very rare classes (c101, c064, c085, total < 80)**: over-sampling, synthetic augmentation, or exclusion from mAP for a more honest assessment of the model's real capabilities.
- **State verbs (v027: turn on/off)**: require a dedicated temporal module that compares the initial and final frames of the action.

### 14.2 For DINOv3

- **Dense features (high priority)**: in the current setup only the CLS token is used. DINOv3's patch features — its real competitive advantage — are not exploited. Extracting and integrating mean-patch or attention-pooled patch features is the most promising next step for DINOv3 (see Section 14.4).
- **Feature projection**: reduce 1024→512 with a trainable projection layer before the MS-Temba backbone to align model capacity with CLIP's.
- **Hybrid routing**: use CLIP for semantically descriptive classes and DINOv3 for classes where the real advantage is confirmed (c143 +27.9, c099 +25.6, c137 +22.9, c071 +20.3, c138 +17.0). **Not recommended** for routing to DINOv3 on c151, c153, c154, c150, c059: real data shows CLIP wins on all these classes.

### 14.3 Architecture

- **Two-head compositional classifier**: add a second classifier (separate object + verb) in parallel to the global classifier, exploiting Charades' compositional mapping.
- **Per-verb temporal modules**: verbs such as v025 (throw), v027 (turn), and v022 (stand) require short temporal sequence analysis — consider denser local attention on these.

### 14.4 Planned Next Steps

In order of priority for subsequent phases:

1. **DINOv3 Dense Features (Phase 2A)**: extract DINOv3's patch features (mean-pooling or attention-pooling over 256 patch tokens) to obtain a spatially structured representation. Estimate mAP impact via ablation with `in_feat_dim=1024` (CLS) vs `in_feat_dim=2048` (CLS+mean_patch). The modification impacts `dinov3_feature_extractor.py`, the dataloader, and MS-Temba's input pipeline.

2. **SCDNet Skeleton Features (Phase 2B)**: integrate pre-extracted skeleton features available on ABACA (`Charades_SCDNet_features2.zip`). The prerequisite is temporal alignment (fps), detailed in Section 15. Recommended fusion strategy: gated fusion (see Section 15.4.3b).

3. **State-of-the-art survey (Phase 3)**: systematic analysis of the literature on combining visual and skeleton features for TAD/TAR, focusing on: modal fusion strategies, robustness to occlusions, benchmark datasets (NTU RGB+D, Charades, Toyota Smarthome), recent models (SkeletonBERT, MotionBERT, HiCo, PoseFormer, UNIK).

---

# 15. Skeleton Feature Integration with CLIP

> **Focus**: this section analyses the integration of skeleton features into the MS-Temba+CLIP pipeline, motivated by the structural limitations of CLIP's CLS token in encoding body movement, and estimates the expected performance impact.

## 15.1 Motivation: CLIP's Postural Gap

As analysed in previous sections, CLIP's CLS token compresses each frame into a single 512-dim vector optimised for textual compatibility. This compression is lossy with respect to body signals: limb configuration, centre of mass (CoM) trajectory, joint angular velocities, relative segment orientation — all information that contrastive training has no incentive to explicitly preserve.

The data confirm this: the 26 classes identified as primary candidates for skeleton features have a **mean CLIP AP of 35.0**, substantially aligned with the overall mean (32.4), but with a much wider distribution: 7 classes below AP 20 (many linked to verbs *throw*, *turn*, *run*) and a cluster of postural classes which, despite being "semantically understandable" by CLIP, remain low precisely due to verb-directional ambiguity that the CLS token does not resolve.

The missing signal is largely **kinematic**: it is not enough to know that a person is standing near a bag — one needs to know whether the wrist is moving away from the object (put) or towards it (take), whether the CoM is descending (sitting down) or rising (standing up), whether the torso tilts forward with knees flexing (sneezing) or remains vertical.

## 15.2 Types of Skeleton Features

Three levels of representation are available, with different computational costs and available information:

**Level 1 — 2D Keypoints** (e.g. OpenPose, MediaPipe Pose): (x, y) coordinates of 17–33 body landmarks per frame. Simple to extract, no depth required, but lose depth information and are subject to occlusions. Appropriate as a low-cost baseline.

**Level 2 — 3D Keypoints** (e.g. MotionBERT, MixSTE): (x, y, z) coordinates estimated monocularly from video. Capture volumetric body orientation and 3D limb trajectories — much more informative for reach/grasp actions and postural transitions.

**Level 3 — Semantic Graph Neural Network Features** (e.g. ST-GCN, PoseConv3D): body represented as a graph where nodes are joints and edges are bone segments. The GNN learns motion dynamics directly on the graph space — captures high-level relationships such as "hand approaches foot" or "knees bend while pelvis descends".

**Note on SCDNet**: skeleton features available on ABACA (`Charades_SCDNet_features2.zip`) are extracted with SCDNet (Skeleton-aware Compositional Dynamic Network), a backbone producing Level 3 GNN embeddings. Their sampling fps must be verified before integration to ensure alignment with the 24fps visual features.

## 15.3 Classes That Would Benefit Most

### 15.3.1 Group A — State and Direction Verbs (expected gain: high)

| Code | Class | AP CLIP | Gap to resolve | Relevant skeleton feature |
|------|-------|--------:|----------------|--------------------------|
| c105 | Turning off a light | 2.3 | Light state pre/post not visible in CLS | Wrist trajectory + final position relative to wall |
| c104 | Turning on a light | 8.7 | Same | Same (opposite direction) |
| c024 | Throwing a bag somewhere | 7.1 | Ballistic gesture — brief and fast | Shoulder angular acceleration + release point |
| c126 | Throwing something on floor | 11.4 | Same | Trunk rotation + descending wrist trajectory |
| c057 | Taking off shoes | 16.5 | Bend-down indistinguishable from put-on | Knee angle + foot/hand contact |
| c055 | Putting on shoes | 33.8 | Same (opposite direction) | Same |

### 15.3.2 Group B — Postural Transitions (expected gain: medium-high)

| Code | Class | AP CLIP | Gap to resolve | Relevant skeleton feature |
|------|-------|--------:|----------------|--------------------------|
| c153 | Someone is sneezing | 17.8 | Sudden trunk flexion + head snap | Trunk angular velocity + head tilt |
| c150 | Someone is running | 18.9 | Gait pattern — CLIP CLS does not encode periodicity | Step frequency, lateral symmetry, foot elevation |
| c154 | Someone is standing up | 36.8 | CoM rise from seated position | CoM Δy positive, knee/hip extension |
| c151 | Standing to sitting | 60.4 | CoM descent toward chair | CoM Δy negative, knee/hip flexion |
| c133 | Awakening in bed | 49.0 | Rise from horizontal position | Torso angle relative to bed plane |
| c060 | Standing on a chair | 33.2 | CoM higher than normal | High absolute CoM + extended legs |

### 15.3.3 Group C — Put/Take and Reach Direction Disambiguation (expected gain: medium)

| Code | Class | AP CLIP | Gap to resolve | Relevant skeleton feature |
|------|-------|--------:|----------------|--------------------------|
| c022 | Putting a bag somewhere | 40.6 | Release trajectory | Increasing hand-object distance (put) vs decreasing (take) |
| c023 | Taking a bag from somewhere | 25.4 | Grasp trajectory | Same (opposite direction) |
| c006 | Closing a door | 30.3 | Push vs pull | Wrist approaching door (push) |
| c008 | Opening a door | 39.8 | Same | Wrist pulling back (pull) |
| c043 | Taking a box from somewhere | 18.3 | Ambiguous action direction | Hand-to-object vs object-to-hand trajectory |
| c044 | Taking something from a box | 21.3 | Mirror of above | Same |

### 15.3.4 Group D — Global Body Actions (expected gain: low-medium)

| Code | Class | AP CLIP | Skeleton contribution |
|------|-------|--------:|----------------------|
| c097 | Walking through a doorway | 43.8 | Body orientation relative to opening |
| c149 | Someone is laughing | 52.4 | Shoulder/trunk shaking pattern |
| c152 | Someone is smiling | 49.5 | Only with facial joints (limited) |
| c155 | Someone is undressing | 56.7 | Arm trajectory away from body |
| c148 | Someone is dressing | 50.0 | Arm trajectory towards body |

## 15.4 Fusion Strategies with CLIP

Fusing CLIP and skeleton features in MS-Temba is not a trivial architectural choice: the point at which the two modalities are integrated determines how much cross-modal information the model can exploit, how susceptible it is to overfitting, and how expensive it is at training and inference.

### 15.4.1 Early Fusion
Immediate concatenation at the input level: `[CLIP_CLS (512) ‖ skel (D)] → Linear proj → 512`. Simple to implement but subject to *modality dominance*: gradients tend to ignore the skeleton stream on classes where CLIP is already strong (~106/157). Use as ablation baseline to verify that skeleton is informative.

### 15.4.2 Late Fusion
Two separate MS-Temba models → logit fusion via learnable gate. Maximum stream specialisation, but no cross-modal interaction during processing. Cost ×2 parameters. Use as ablation comparison.

### 15.4.3 Mid-Level Fusion (Sum or Gated)
Separate projections for each stream → fusion at common dimension `d=512`.

**Sum fusion**: `h = Linear_CLIP(clip) + Linear_Skel(skel)`. Avoids CLIP dominance but does not learn inter-modal interactions. Mid-level baseline.

**Gated fusion** (recommended):
```
g = σ(Linear_gate([f_CLIP; f_Skel]))    # gate ∈ (0,1)^d
h = g ⊙ f_CLIP + (1−g) ⊙ f_Skel
```
The gate learns per frame how much to weight each modality on each dimension. Consistent with the adaptive selection mechanism of Temba-Blocks. Low cost (+~2% parameters). **Recommended setup for the first production experiment.**

### 15.4.4 Dual-Branch Fusion
Two parallel branches of Temba-Blocks → fusion in MS-Fuser. Maximum expressiveness, models different temporal scales per modality. Cost ~×2. Use only if gated fusion is already confirmed and sufficient regularisation is available (high overfitting risk on Charades with 7985 videos).

### 15.4.5 Cross-Attention Fusion
`Q = proj_Q(CLIP_CLS)`, `K = V = proj_KV(skel)`. The attention mechanism selects which kinematic dimensions are relevant given the current semantic context. Explicit semantics: CLIP queries skeleton — "given that I see a bag, retrieve from the skeleton feature the information about hand trajectory". Superior to gated fusion for contextual interaction, but more expensive and susceptible to overfitting.

### 15.4.6 Summary Comparison

| Strategy | Cross-modal interaction | Parameter cost | Overfitting risk | Recommended |
|----------|:-----------------------:|:--------------:|:----------------:|:-----------:|
| **Early Fusion** | ✗ (CLIP dominance) | +~1% | Low | Ablation baseline |
| **Late Fusion** | ✗ (none) | ×2 | Low | Ablation comparison |
| **Sum Fusion** | Partial (additive) | +~1% | Low | Mid-level baseline |
| **Gated Fusion** | ✓ (learnable, symmetric) | +~2% | Low–Medium | **Recommended setup** |
| **Dual-Branch** | ✓ (multi-scale, separate) | ~×2 | High | Only with pre-training |
| **Cross-Attention** | ✓ (directional CLIP→Skel) | +~15% | Medium–High | Advanced final setup |

**Recommended progression**: Early → Gated → Cross-Attention.

## 15.5 Skeleton Feature Representation and Preprocessing

**Mandatory normalisation** before fusion with CLIP:
1. Subtract mid-hip position from all joints (camera position invariance)
2. Divide by average torso length (subject scale invariance)
3. Sample at the same frequency as CLIP frames (24fps) via interpolation if necessary

**fps alignment for SCDNet**: features on ABACA may have been extracted at a different fps than 24. Before integration, inspect the format: `python -c "import zipfile; z=zipfile.ZipFile('Charades_SCDNet_features2.zip'); print(z.namelist()[:5])"` and compare the number of frames with the corresponding CLIP features.

**Missing skeleton fallback**: not all frames produce reliable detections (occlusions, multi-person, low video quality). With gated fusion, force `g = 1` (full weight on CLIP) when the skeleton confidence score is below threshold.

## 15.6 Expected mAP Impact Estimates

### 15.6.1 Estimates per Group

| Group | #Classes | Mean CLIP AP | Expected gain (gated) | Expected gain (cross-att) |
|-------|:--------:|:------------:|:---------------------:|:-------------------------:|
| A — State/direction verbs | 6 | 13.3 | +10–14 AP/class | +15–20 AP/class |
| B — Postural transitions | 6 | 35.8 | +8–12 AP/class | +12–18 AP/class |
| C — Reach direction disambiguation | 6 | 29.2 | +6–10 AP/class | +10–15 AP/class |
| D — Global body actions | 8 | 50.7 | +2–5 AP/class | +4–8 AP/class |

### 15.6.2 Global ΔmAP Estimate

| Scenario | Strategy | Mean gain/class | Benefited classes | Estimated ΔmAP |
|----------|----------|:---------------:|:-----------------:|:--------------:|
| **Conservative** | Gated, 2D skel | +6 AP | 26 | **+1.0** |
| **Base** | Gated, 3D skel | +11 AP | 26 | **+1.8** |
| **Optimistic** | Cross-Att, GNN | +17 AP | 26 | **+2.8** |
| **Ideal** | Cross-Att + multi-scale | +24 AP | 26 | **+4.0** |

### 15.6.3 Estimated AP Distribution Post-Skeleton Integration

| AP Range | Current CLIP | CLIP + Skeleton (base) | CLIP + Skeleton (optimistic) | Δ classes (base) |
|----------|:-----------:|:----------------------:|:----------------------------:|:----------------:|
| ≥ 60 | 22 | ~24 | ~27 | +2 |
| 40–59 | 30 | ~33 | ~35 | +3 |
| 20–39 | 56 | ~57 | ~55 | +1 |
| < 20 | 49 | ~43 | ~40 | −6 |

## 15.7 Implementation Criticalities

**Overfitting on rare classes.** The Group A and B classes that would benefit most from skeleton are often the rarest in training: c104/c105 (~200 train), c060 (46 train), c024 (89 train). Adding a skeleton stream increases model capacity without increasing training data. With gated fusion, recommended `drop=0.15, drop_path=0.1`.

## 15.8 Implementation Priorities

1. **Inspect SCDNet features** already available on ABACA: format, fps, dimensionality.
2. **Early fusion ablation** as lower bound: concat (CLIP 512 + skel D → proj 512). If ΔmAP < +0.2 on Groups A and B, the problem lies in skeleton feature quality.
3. **Sum fusion vs gated fusion ablation**: if gated outperforms sum by > +0.3 mAP, learnable interactions are important.
4. **Upgrade to 3D features** (MotionBERT or equivalent) if gated + 2D shows confirmed gain.
5. **Cross-attention CLIP→Skeleton** only if previous steps confirm skeleton value and sufficient regularisation has been added.
6. **In parallel**: kinematic auxiliary supervision for Group A (predict wrist motion vector direction). No additional inference-time cost.

---