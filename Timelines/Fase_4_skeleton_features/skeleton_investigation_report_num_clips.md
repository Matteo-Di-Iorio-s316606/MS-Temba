# Technical Report: Root Cause Analysis of Skeleton Feature Under-performance in MS-Temba on Charades

**Author:** Matteo Di Iorio
**Date:** April 22, 2026
**Project:** Master's Thesis — Temporal Action Detection with Multi-Modal Fusion
**Model:** MS-Temba (Multi-Scale Temporal Mamba)
**Dataset:** Charades (157 classes, multi-label TAD)

---

## Abstract

This report documents a systematic investigation into the under-performance of skeleton features (SCD-Net, pre-trained on NTU) when integrated into the MS-Temba architecture for Temporal Action Detection on the Charades dataset. Three baseline configurations have reached a performance ceiling around 8.5–9.5 mAP, an approximately threefold degradation relative to the CLIP baseline (29.0 mAP). Previous hypotheses attributed this gap primarily to domain shift (NTU → Charades). This report argues that the dominant causes are instead (i) a fixed-length input window (`num_clips = 256`) that catastrophically interacts with the high native frame rate of skeleton features, producing a systematic evaluation bias, and (ii) a label granularity mismatch that introduces structural noise when features are densely sampled. Empirical measurement on the Charades test split shows that 34.5% of ground-truth actions are entirely unreachable by the model at evaluation time and an additional 45.0% are only partially observable. A set of prioritized experiments is proposed to decouple these confounding factors.

---

## 1. Problem Setting and Prior Results

### 1.1 Architecture summary

MS-Temba processes a temporal feature sequence $X \in \mathbb{R}^{B \times D_{\text{in}} \times T \times 1 \times 1}$ through three hierarchical Mamba stages operating on progressively re-sampled subsequences (modulo-2 and modulo-3 strided views), followed by a scale-fusion and a classification head at each timestep. Crucially, the temporal axis $T$ is **preserved throughout the forward pass**: the model outputs per-timestep logits $Y \in \mathbb{R}^{B \times T \times C}$ with $C = 157$.

The model therefore implements a dense, temporally-local classifier: every output position $t$ attempts to predict the multi-label action set active at that timestep. There is no temporal pooling that would abstract away the input rate.

### 1.2 Baseline results

Three skeleton configurations have been evaluated on Charades:

| Configuration | Effective $T$ (mean) | Best mAP | Peak epoch |
|---|---|---|---|
| Skeleton w=16 (SCD-Net windowed) | ~45 | 9.46 | 21 |
| Skeleton w=1 native, no interpolation | ~750 | 8.52 | 15 |
| Skeleton w=1 interpolated to CLIP FPS | ~45 | 8.52 | — |
| CLIP (reference) | ~47 | 29.00 | 13 |

All skeleton variants exhibit rapid overfitting after the peak (train mAP reaches 15.83 by epoch 30 while validation collapses to 8.10), and all converge to a narrow band around 8.5–9.5 mAP regardless of temporal resolution or interpolation strategy. This apparent invariance to the upstream resolution has been interpreted as evidence of a fundamental domain-gap ceiling.

### 1.3 Motivation for re-investigation

The invariance of the final mAP across such different temporal configurations is suspicious. If the temporal resolution were truly orthogonal to performance, the model would be effectively discarding the temporal signal it receives — a conclusion that contradicts the very rationale for using a Mamba-based temporal model. An alternative reading is that **a downstream mechanism is clamping all configurations into the same operating regime**, making the upstream distinctions irrelevant. This report identifies that mechanism.

---

## 2. The `num_clips` Mechanism

### 2.1 Definition

The hyper-parameter `num_clips = 256` defines the maximum temporal length processed by the model in a single forward pass. In the dataloader (`charades_dataloader.py`), it drives the clipping logic:

```python
if len(features) > num_clips and num_clips > 0:
    if self.split == "testing":
        random_index = 0
    else:
        random_index = random.choice(range(0, len(features) - num_clips))
    features = features[random_index: random_index + num_clips: 1]
    labels   = labels[random_index: random_index + num_clips: 1]
    hmap     = hmap[random_index: random_index + num_clips: 1]
```

Two distinct behaviors emerge:

- **Training**: a random window of 256 consecutive timesteps is sampled. Over multiple epochs, different subsequences are shown to the model, providing a form of temporal data augmentation.
- **Testing**: `random_index = 0` is hard-coded. The model always evaluates on the **first 256 timesteps** of each video. Any action located beyond this window is, by construction, never presented to the model at evaluation time.

### 2.2 Feature-rate sensitivity

The number of timesteps per video is determined by the feature extractor's sampling rate:

$$T_{\text{video}} = \lfloor \text{duration} \times \text{FPS}_{\text{feat}} \rfloor$$

The maximum video duration that fits entirely in the 256-timestep window is therefore:

$$D_{\max} = \frac{\text{num\_clips}}{\text{FPS}_{\text{feat}}}$$

For the three relevant feature types:

| Feature | FPS | $D_{\max}$ (seconds) |
|---|---|---|
| CLIP | 1.51 | **169.5** |
| Skeleton w=16 | ~1.5 | ~170 |
| Skeleton w=1 (native) | 24.05 | **10.65** |

Charades videos have a mean duration of 30.1 s and range from 2.4 s to 179 s. For CLIP, the 169.5 s threshold exceeds all but one or two outlier videos: the dataset is, in practice, entirely visible to the model. For native-rate skeleton, the 10.65 s threshold is below the median video duration: **the majority of the dataset is clipped, and at evaluation time, always clipped from the beginning**.

### 2.3 Empirical measurement

To quantify the impact, we measured, on the Charades test split, how many ground-truth actions fall entirely or partially beyond the first 10.65 seconds. The diagnostic script is reproduced below:

```python
import json

with open('data/charades.json') as f:
    data = json.load(f)

fps_skel = 24.0
num_clips = 256
max_sec_visible = num_clips / fps_skel  # 10.67s

total, invisible, partial = 0, 0, 0
for vid, meta in data.items():
    if meta['subset'] != 'testing':
        continue
    for cls, st, en in meta['actions']:
        total += 1
        if st >= max_sec_visible:
            invisible += 1
        elif en > max_sec_visible:
            partial += 1
```

Results:

```
Total test actions:             16,691
Completely invisible (>10.7s):   5,752  (34.5%)
Partially invisible:             7,517  (45.0%)
Fully visible:                   3,422  (20.5%)
```

**More than one third of test actions are structurally unreachable** by the skeleton model. An additional 45% are only partially within the evaluation window, meaning the model sees the action onset but the class-defining motion content (often concentrated in the latter portion of the clip) is truncated.

### 2.4 Interpretation

The mean Average Precision metric, computed per-class and averaged, is not robust to systematic absence of positive evidence. When a positive instance cannot appear in the model's input, the model cannot produce a true positive for it at evaluation: the precision-recall curve is capped at a recall $r < 1$ proportional to the fraction of visible actions. Aggregating over all classes, this produces an **upper bound on achievable mAP** that is substantially lower than the true model capacity on fully-visible data.

Under an optimistic assumption of uniform performance degradation proportional to action visibility, the observed mAP can be modeled as:

$$\text{mAP}_{\text{obs}} \approx \text{mAP}_{\text{ideal}} \cdot (p_{\text{full}} + \alpha \cdot p_{\text{partial}})$$

with $p_{\text{full}} = 0.205$, $p_{\text{partial}} = 0.450$, and $\alpha \in [0, 1]$ a partial-credit coefficient. Even with a generous $\alpha = 0.5$:

$$\text{mAP}_{\text{obs}} \approx \text{mAP}_{\text{ideal}} \cdot (0.205 + 0.225) = 0.430 \cdot \text{mAP}_{\text{ideal}}$$

Reverse-engineering from the observed 8.52 mAP yields a latent $\text{mAP}_{\text{ideal}} \approx 19.8$. The inequality is rough, but its direction is clear: the current evaluation setup is mathematically incapable of surfacing more than approximately half of whatever modeling capacity the skeleton features genuinely possess.

---

## 3. Automatic FPS Adaptation in the Label Construction Pipeline

Before proceeding to the second structural cause, it is important to address an intervention that has been informally proposed as a potential fix: generating a skeleton-specific ground-truth file in which action boundaries are pre-converted to frame indices, so as to "match" the skeleton frame rate. This section traces the data flow through `make_dataset` to show that such an intervention would be redundant — the existing pipeline already performs the FPS adaptation automatically at load time — and to clarify exactly where the label-construction pipeline does and does not contribute to the observed problems.

### 3.1 Input: ground truth in absolute seconds

The `charades.json` file stores each action as a triple `[class_id, start_sec, end_sec]` with timestamps in continuous, feature-agnostic seconds:

```json
"7UPGT": {
    "subset": "training",
    "duration": 23.21,
    "actions": [
        [149, 16.0, 22.2],
        [143,  0.0,  5.0],
        [ 61, 14.2, 24.0]
    ]
}
```

These timestamps describe when events occur in real time, independently of any downstream representation chosen for the video.

### 3.2 The `make_dataset` procedure

The core label-construction logic in `charades_dataloader.py` is:

```python
num_feat = fts.shape[0]                          # (1) read from features
label = np.zeros((num_feat, num_classes))        # (2) allocate label grid
fps = num_feat / float(duration)                 # (3) compute FPS from data

for ann in actions:
    cls, st, en = int(ann[0]), float(ann[1]), float(ann[2])
    for fr in range(num_feat):
        t = fr / fps                             # (4) frame → seconds
        if t > st and t < en:
            label[fr, cls] = 1                   # (5) binary gate
```

The critical line is (3): `fps` is not a hyper-parameter and is not read from any configuration file. It is **derived at runtime from the shape of the feature tensor and the video duration**. Whatever feature extractor produced the `.npy` being loaded therefore determines the FPS used for label construction, and does so on a per-video basis.

### 3.3 Worked example

Consider the video `7UPGT` (duration 23.21 s) under two different feature extractors, for the action `[143, 0.0, 5.0]`:

**Under CLIP features** (`charades_features_clip/7UPGT.npy`, shape `[35, 768]`):
- `num_feat = 35`
- `fps = 35 / 23.21 ≈ 1.508`
- Frame-to-time: `t = fr / 1.508`
- Frames matching the action: `fr` such that `0.0 < fr/1.508 < 5.0`, i.e. `fr ∈ {1, 2, …, 7}`
- Result: `label[1:8, 143] = 1`

**Under native skeleton features** (`charades_scdnet_full/7UPGT.npy`, shape `[558, 4096]`):
- `num_feat = 558`
- `fps = 558 / 23.21 ≈ 24.04`
- Frame-to-time: `t = fr / 24.04`
- Frames matching the action: `fr` such that `0.0 < fr/24.04 < 5.0`, i.e. `fr ∈ {1, 2, …, 120}`
- Result: `label[1:121, 143] = 1`

In both cases, the **same action boundaries in seconds** produce correctly-aligned label grids at the **correct feature-specific granularity**. No modification to the source file is required. The only state that changes between the two runs is the `root` directory argument passed to the `Charades` dataset constructor.

### 3.4 Why a pre-computed `charades_skeleton.json` is not warranted

Generating a separate ground-truth file with frame-indexed actions, for example:

```json
"7UPGT": {
    "duration": 23.21,
    "fps": 24.04,
    "actions_frames": [[143, 1, 120], ...]
}
```

would introduce three problems and solve none:

1. **Hard-coded FPS.** Any change to the feature extractor — including the planned DINOv2 experiments — would require regenerating the file. The current code handles this automatically.
2. **Loss of precision.** Converting seconds to frames in advance is a lossy operation that irreversibly rounds every boundary to the feature grid. Performing the conversion on demand keeps the original seconds as the source of truth.
3. **Duplication of state.** Two ground-truth files describing the same videos invite silent drift between them, which is a classic source of reproducibility bugs.

### 3.5 What the pipeline does and does not solve

The automatic FPS adaptation correctly places label edges at the appropriate frame indices for each feature type. It is mathematically exact: a feature vector at position $fr$ corresponds to real time $fr / \text{fps}$, and the labeling rule `st < t < en` is applied directly in the time domain.

What the pipeline does **not** address is how much noise lives in those edges in the first place. Adapting the grid scales the uncertainty to the correct temporal density; it does not remove it. This distinction sets up the analysis in Section 4: the pipeline is correct by construction with respect to temporal alignment, and the problems identified in this report are downstream of this correctness.

The takeaway for the skeleton investigation is therefore twofold:

- The FPS mismatch between CLIP and skeleton is **not** a source of error in itself, because the label grid adapts.
- Any residual training-time problems must be sought elsewhere — specifically in how the downstream model interacts with sequences of very different lengths (Section 2) and in how the loss function treats fine-grained binary boundaries that inherit sub-second annotator noise (Section 4).

---

## 4. Secondary Issue: Label–Feature Granularity Mismatch

A second, independent problem becomes visible once `num_clips` is accounted for. Although Section 3 establishes that the label grid is constructed correctly at whatever FPS the features impose, it does not eliminate the fact that the *information content* of the annotation is bounded by the human labeling process that produced it.

Charades annotations are produced by human annotators operating at effectively 1–2 Hz, and carry an intrinsic boundary uncertainty of roughly $\pm 0.5$–$1$ s. These seconds-level uncertainties are then propagated verbatim into the per-frame label grid by the code shown in Section 3.2:

```python
fps = num_feat / float(duration)
for fr in range(num_feat):
    t = fr / fps
    if t > st and t < en:
        label[fr, cls] = 1
```

At CLIP's rate (1.5 FPS), a 1 s annotation uncertainty corresponds to $\approx 1.5$ timesteps — a negligible fraction of the sequence. At the native skeleton rate (24 FPS), the **same** uncertainty corresponds to 24 timesteps. When the model optimizes a per-timestep binary cross-entropy against these labels, it is asked to learn boundaries that do not exist at that temporal granularity in the ground truth. This imposes an irreducible noise floor on the training signal, which explains the **fast-onset overfitting pattern** observed in every skeleton training run: the model memorizes the noise in the training labels while validation, which suffers from the same noise on different videos, cannot benefit from the memorization.

This hypothesis is further consistent with the observed invariance between `w=1 native` and `w=1 interpolated from CLIP`: both configurations ultimately produce label grids at incompatible scales — the former has 24 FPS labels, the latter has 1.5 FPS labels after interpolation but inherits the same annotator noise. What differs is the spatial pattern of noise, not its magnitude.

Note that this is emphatically **not** a flaw in `make_dataset` itself: the procedure faithfully reproduces, at the feature's own rate, the information present in the source annotation. The problem is epistemic rather than implementational — at high FPS, the per-frame label is more precise than the source annotation warrants.

---

## 5. Revised Causal Model

The original attribution (60% pooling / 30% capacity / 10% domain gap) was synthesized post-hoc and not isolated by controlled experiments. A more defensible decomposition, grounded in measurement, is:

| Cause | Evidence | Estimated impact |
|---|---|---|
| `num_clips`-induced evaluation bias | 34.5% of test actions structurally invisible; deterministic clipping at $t=0$ in testing | **High** — caps achievable mAP independent of model quality |
| Label granularity mismatch | Overfitting onset at epoch 15; train/val divergence from epoch 5; invariance of mAP to input resolution | **Medium–High** — produces noise floor in training |
| Domain gap NTU → Charades | Qualitative (backbone trained on lab-controlled data) | **Unknown** — cannot be isolated until the two causes above are removed |

Critically, the domain-gap hypothesis **cannot be validated** against the current results because it is confounded with the two structural issues. Any claim about domain gap must first eliminate the other two causes.

---

## 6. Proposed Experimental Program

The experiments below are ordered by epistemic value per unit of compute. Each is designed to isolate a single causal factor.

### 6.1 E1 — Sliding-window inference (removes evaluation bias)

Modify only the evaluation path. Split each test video into overlapping 256-timestep windows, run inference on each window, and aggregate predictions at the corresponding absolute timesteps (mean or max over overlapping positions). Training is unchanged.

**What it tests:** the contribution of `num_clips` to the observed mAP ceiling, independent of training and feature quality.

**Expected result:**
- If skeleton w=1 rises from 8.52 to $\sim 12$–$14$ mAP, the evaluation bias accounts for the majority of the gap.
- If it remains flat, the issue is upstream (feature quality, training noise).

**Cost:** 2–3 hours of implementation, one re-evaluation pass.

### 6.2 E2 — Increase `num_clips` to 768 (removes training bias)

Set `num_clips = 768`, which covers $768 / 24 = 32$ s at 24 FPS — exceeding the mean video duration. Verify GPU memory (A40 at 48 GB should tolerate this with batch size 3–5) and re-train for 50 epochs.

**What it tests:** the contribution of truncation during training, on top of the fix already provided by E1 at test time.

**Expected result:** if E1 showed improvement and E2 adds further improvement, the compound effect confirms that both training and evaluation were bottlenecked. If E2 shows no improvement over E1, the bottleneck is exclusively at evaluation.

**Cost:** one full training run (~12 hours on A40).

### 6.3 E3 — Temporal pooling of skeleton to 3 FPS (attacks both causes)

In the dataloader, apply 1D average pooling with kernel and stride of 8 on the skeleton feature tensor before any other processing. This reduces the effective FPS from 24 to 3, which:

1. Reduces $T_{\text{skel}}$ for a 30 s video from 720 to 90, bringing virtually every video entirely within `num_clips = 256`.
2. Attenuates the label granularity mismatch: at 3 FPS, a $\pm 0.5$ s annotator uncertainty spans 1.5 timesteps, comparable to CLIP's regime.
3. Acts as a denoising prior through temporal averaging.

**What it tests:** whether the skeleton features contain useful information once the two structural issues are removed simultaneously.

**Expected result:**
- If mAP rises to 11–14: confirms that the skeleton signal exists but was obscured. This is the most likely outcome given the diagnostic.
- If mAP remains at ~9: the residual gap can be attributed to domain shift with confidence, because alternative explanations have been controlled.

**Cost:** one training run (~8 hours, reduced due to shorter sequences).

### 6.4 E4 — Stronger regularization (attacks overfitting)

Fix dropout to 0.3, drop_path to 0.3, weight decay to 0.1 (from current 0.1/0.1/0.05). Re-train skeleton w=1.

**What it tests:** whether part of the ceiling is driven by capacity–data mismatch (4096-dim features over 157 classes on ~8 k training videos is a high-capacity regime).

**Expected result:** modest improvement (+0.5–1.0 mAP), useful as a regularization ablation for the thesis.

**Cost:** one training run.

### 6.5 E5 — Visibility-stratified evaluation (diagnostic)

Using the best existing checkpoint, compute per-video mAP and stratify by the visibility fraction of each video's actions (fully visible / partially visible / invisible) under `num_clips = 256`. This is a post-hoc analysis without re-training.

**What it tests:** direct validation of the causal model in Section 5. Fully-visible videos should show mAP well above the aggregate 8.52, while partially-visible videos should drop sharply.

**Expected result:** a monotonic gradient in mAP across strata, validating that `num_clips` is the dominant confounder.

**Cost:** 1–2 hours.

---

## 7. Recommended Sequence

The principle of the program is to **first remove confounders, then interpret residuals**. In practice:

1. **E5** (stratified evaluation, 1 h): provides an immediate, training-free estimate of the `num_clips` effect and a reference baseline before any intervention.
2. **E1** (sliding-window inference, 2–3 h): provides a concrete upper bound on what the existing trained model can achieve under unbiased evaluation.
3. **E3** (temporal pooling, 8 h training): attacks both identified structural causes in a single, well-motivated experiment.
4. **E2** (`num_clips = 768`, 12 h training): conditional on E3 — run only if E3 does not fully close the gap, to isolate the contribution of training-time truncation from evaluation-time truncation.
5. **E4** (regularization): run as a secondary ablation regardless of outcome, for the thesis.

Steps 1–3 should be executed within approximately one compute-day. At that point, the residual gap between skeleton and CLIP will be interpretable — either as a genuine domain-gap ceiling (defensible, publishable as a negative result with rigorous methodology) or as a further tractable problem.

---

## 8. On the "New Ground-Truth File" Proposal

For completeness, we restate the conclusion of Section 3 in the context of the experimental plan. The proposal to generate a skeleton-specific ground-truth JSON (e.g. `charades_skeleton.json` with pre-computed frame indices) is not warranted: as shown in detail in Section 3, `make_dataset` derives the per-video FPS at runtime from `num_feat / duration` and constructs the label grid directly in the time domain. Any feature extractor — CLIP, skeleton, DINOv2, or future additions — is handled transparently by the same source file.

Should a future intervention aim to address the annotation-boundary noise identified in Section 4, the correct location for the change is inside `make_dataset` itself — replacing the hard `if t > st and t < en` gate with a linear or Gaussian ramp at boundaries over a fixed $\delta$ (e.g., 0.5 s) to produce soft labels. This modifies the label generation logic, not the ground-truth source file.

---

## 9. Implications for the Thesis Narrative

The current draft Chapter 4 frames the skeleton investigation as a "rigorous negative result": skeleton modalities hit a domain-gap ceiling that cannot be overcome. The findings in this report suggest that framing is premature and, more importantly, **methodologically weak**: a reviewer will ask why the domain-gap conclusion was drawn when two confounders were not controlled.

A stronger narrative, which the proposed experiments can support, is:

> We investigated skeleton features as a complementary modality for TAD on Charades and observed an apparent performance ceiling near 9 mAP. Through systematic decomposition, we identified two structural confounders in the pipeline — a fixed-window evaluation that renders 34.5% of test actions unreachable, and a label granularity mismatch driven by the feature rate — and showed that controlling for both recovers the signal to X mAP. The residual gap to CLIP, of Y mAP, we attribute to domain shift and discuss as a direction for future work.

This framing converts a thin negative claim into a substantive diagnostic contribution, regardless of the final numerical outcome.

---

## 10. Summary of Findings

1. The observed 8.5–9.5 mAP ceiling for skeleton features on Charades is **not diagnostic of a domain-gap ceiling**, because two structural confounders operate simultaneously and explain a large fraction of the gap independently.

2. The `num_clips = 256` window interacts catastrophically with the 24 FPS skeleton rate: **34.5% of test actions are entirely invisible to the model**, and **45.0% are partially truncated**. Deterministic first-window evaluation (`random_index = 0` in test) makes this an evaluation-time bias, not merely a training-time one.

3. The 24 FPS label grid, derived from human annotations with $\pm 1$ s boundary uncertainty, introduces ~24 timesteps of boundary noise per annotation, explaining the rapid overfitting pattern consistent across every skeleton training run.

4. The invariance of mAP across upstream configurations (w=16, w=1 native, w=1 interpolated) is explained by both confounders operating as a bottleneck downstream of the feature-extraction choice.

5. The `make_dataset` procedure already adapts the label grid to any feature FPS at runtime through `fps = num_feat / duration`. The FPS mismatch between modalities is therefore **not a source of error in itself**, and generating a skeleton-specific ground-truth file is unnecessary. Any future intervention on label noise belongs inside `make_dataset` (soft labels), not in a duplicated source file.

6. The proposed experimental program (E1–E5) controls the remaining confounders systematically and can be executed within approximately one compute-day. Its outcome is diagnostically informative regardless of sign.

---

## Appendix A — Diagnostic Script Output (Reproducibility)

```
Total test actions:             16,691
Completely invisible (>10.7s):   5,752  (34.5%)
Partially invisible:             7,517  (45.0%)
Fully visible:                   3,422  (20.5%)
```

Environment: Grid5000 Sophia, Python 3.9.2, `charades.json` as of April 22, 2026.

## Appendix B — Key Source Locations

- Dataloader clipping logic: `vim/charades_dataloader.py`, `Charades.__getitem__`, lines implementing the `if len(features) > num_clips` block.
- Label generation from absolute seconds (per-video FPS derivation): `vim/charades_dataloader.py`, `make_dataset`, inner loop over actions.
- Model forward pass (confirms $T$ preservation): `vim/models_MSTemba.py`, `MSTemba.forward_features`.
- Training configuration: `vim/MSTemba_main.py`, argument parser and training loop.