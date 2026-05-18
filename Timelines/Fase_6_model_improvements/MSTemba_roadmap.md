# MS-Temba: Roadmap of Proposed Improvements

*A technical report on architectural, semantic and engineering extensions for dense Temporal Action Detection on long untrimmed videos*

**Matteo Di Iorio**  
*Master's Research Project — Internal Document for Supervisor Review*

---

# Abstract
MS-Temba (Sinha et al., CVPR 2026) introduces dilated state-space models for dense Temporal Action Detection (TAD) on long untrimmed videos, achieving state-of-the-art results on TSU and Charades benchmarks with only 17M parameters — a five-fold reduction over MS-TCT. Despite its strengths, the design exposes several open limitations: hard-coded dilation rates with no content-aware adaptation, uniform supervision across multi-scale blocks, recurrent decay on very long sequences (T ≈ 2 500), a frozen visual backbone, and the absence of any semantic prior on action classes. This document presents a structured roadmap of twelve proposed extensions, organised into six thematic axes. For each proposal we describe the limitation it addresses, the architectural intervention, the expected outcome, and we explicitly trace the intellectual lineage of the idea to the literature — distinguishing between how the source paper applies the concept in its original domain and how we adapt it to dense TAD. The document concludes with a novelty ranking of the proposals and a recommended implementation timeline structured for clean ablation analysis.

# 1. Introduction
## 1.1 Background
Temporal Action Detection in densely labelled, untrimmed video sequences is a fundamentally harder problem than the more common action classification task. Real-world Activities of Daily Living (ADL) recordings span tens of minutes, contain dozens of overlapping action instances per video, and require simultaneous handling of brief atomic events (a 2-second drink) and long sustained activities (a 15-minute reading session). The architectural choices that succeed in standard video understanding — sparse frame sampling, global pooling, self-attention over short windows — break down at the temporal scales relevant to ADL.

MS-Temba addresses this challenge by extending Mamba's selective state-space mechanism with dilated scanning. Each Temba block processes the sequence at a distinct temporal stride, and a multi-scale Mamba fuser aggregates the resulting representations. The design achieves linear computational complexity in sequence length while preserving fine-grained temporal structure — a combination unattainable with pure Transformer or pure convolutional baselines.

## 1.2 Open limitations of MS-Temba
A careful reading of the paper, combined with analysis of the public implementation, reveals seven distinct limitations that motivate this roadmap:

- Dilation rates are hard-coded as η = k for the k-th block, irrespective of content.

- Auxiliary supervision is applied uniformly across blocks despite their emergent scale specialisation (Figure 5 of the paper).

- The recurrent state-space backbone suffers from progressive information decay over sequences of T ≈ 2 500 segments (≈40 minutes).

- Detection is reduced to per-frame binary cross-entropy classification; no explicit boundary modelling is performed.

- The visual backbone (I3D or CLIP) is fully frozen, preventing adaptation to the ADL distribution.

- The final classification head learns class representations from scratch, ignoring the available semantic structure between action labels.

- The Mamba-1 backend caps the SSM state dimension at N = 16, limiting hidden-state capacity for very long sequences.

## 1.3 Structure of this document
Each of the following six themes addresses one or more of these limitations. Section 2 introduces adaptive temporal scaling. Section 3 develops three complementary mechanisms for long-range context. Section 4 introduces LLM-derived semantic priors. Section 5 covers cross-modal vision-language fusion. Section 6 presents a critical counterpoint on a tempting but ultimately unsuitable use of LLMs at inference time. Section 7 collects architectural enablers — engineering upgrades that unlock the gains of the other themes. Section 8 ranks the proposals by novelty and presents a recommended implementation timeline.

# 2. Theme 1 — Adaptive Temporal Scales
## 2.1 Gated Multi-Scale Dilations
#### *Problem addressed*
MS-Temba's k-th block applies a single dilation rate η = k, fixed at design time. Within a video, however, the temporal scale that best resolves an action varies frame-by-frame. A 2-second 'Drink from Cup' embedded in a 15-minute 'Use Laptop' would benefit from η = 1 in its immediate neighbourhood and η ≥ 3 elsewhere. The current architecture forces all blocks to process every token at their respective static rate, with no mechanism for content-dependent scale selection.

#### *Proposed intervention*
Each Temba block is extended to host three parallel dilated SSM branches with rates η ∈ {1, k, 2k}. A small gating network — a two-layer MLP with hidden dimension D_k / 4 — receives the input token feature and produces a softmax distribution g_t ∈ Δ³ over the three branches. The block output becomes the gate-weighted combination:

```
z_k(t)  =  Σ_m  g_t[m] · DilatedSSM_{η_m}(z_{k−1})(t)
```

At inference, Gumbel-softmax with straight-through estimator produces a sparse gate, activating only the dominant branch per token; computational cost therefore remains comparable to the original single-branch block. The gate logits are initialised with a warm-start bias that favours η = k in the k-th block at epoch zero, exactly reproducing MS-Temba's behaviour, before allowing data-driven deviations to emerge during training.

#### *Modules modified*
- `models/temba_block.py` — replace the single `DilatedSSM` call with a three-branch parallel block and append the gating MLP.

- `models/ms_fuser.py` — a companion gate replaces the uniform sum z^f = Σ_k z̃_k with z^f_t = Σ_k w_t[k] · z̃_k(t), where w_t is a per-frame distribution over blocks.

#### *Expected outcome*
An estimated improvement of +1 to +2.5 mAP on TSU, with the largest gains expected on classes with high intra-class temporal variance (e.g. 'Read'). Parameter overhead: approximately 1.5M additional weights (mostly in the gates). Inference cost is unchanged thanks to sparse activation.

#### *Intellectual lineage*
**Source paper:** *PDAN — Pyramid Dilated Attention Network (Dai et al., WACV 2021).* PDAN organises dilated attention layers in a static pyramid, with each layer operating at a fixed dilation. Our adaptation differs in three ways: we replace attention with selective state-space scanning, we make the dilation selection content-aware via a learnable gate, and we operate within Mamba's linear-time regime rather than attention's quadratic one. The Mixture-of-Experts gating pattern is inspired by sparse-MoE literature (Shazeer et al., 2017) but applied here at the temporal-scale level rather than the parameter level.

## 2.2 Scale-Specific Auxiliary Loss
#### *Problem addressed*
Figure 5 of the MS-Temba paper documents an emergent phenomenon: Block 1 — the smallest dilation — naturally specialises on short actions, and Block 3 on long actions, despite all blocks receiving the same supervision target Y. The current auxiliary loss L_aux(k) = BCE(ŷ_k, Y) introduces a gradient conflict: Block 1 is penalised for failing to detect 15-minute 'Use Laptop' instances that exceed its effective receptive field, and Block 3 is penalised for missing micro-actions it cannot temporally resolve.

#### *Proposed intervention*
Off-line, we partition the ground-truth instances into three duration buckets according to a scheme aligned with the block dilations:

- Block 1 (η = 1): instances with duration < 5 seconds (e.g., Drink from Cup, Sit Down).

- Block 2 (η = 2): instances with duration between 5 and 30 seconds (e.g., Walk, a single cooking step).

- Block 3 (η = 3): instances with duration > 30 seconds (e.g., Read, Use Laptop, Watch TV).

For each block k, we construct a filtered target Y^(k) containing only instances from the k-th bucket. The auxiliary loss becomes:

```
L_aux^(k)  =  BCE( ŷ_k ,  Y^(k) )
```

#### *Modules modified*
- `train.py` — pre-compute the duration buckets at dataset load time; apply a per-block mask to the auxiliary loss.

- No architectural changes; zero additional parameters.

#### *Expected outcome*
A modest mAP improvement (+0.5 to +1) accompanied by a substantially stronger interpretability story: the per-bucket mAP for each block — currently emergent — becomes a clean ablation that directly demonstrates scale specialisation. The bucket boundaries themselves become a tunable hyperparameter eligible for further ablation. The proposal is highly synergistic with the gated multi-scale of Section 2.1: the duration-filtered signal sharpens the supervision that the gate learns to route on.

#### *Intellectual lineage*
**Source: original contribution.**The idea has loose analogues in feature-pyramid-network detection literature (Lin et al., 2017), where different pyramid levels are trained against different object scales, but the application to dense multi-label TAD with duration-conditioned filtering has not — to our knowledge — been documented. The motivation is supplied by MS-Temba's own Figure 5, but the paper does not exploit the observation.

# 3. Theme 2 — Long-Range Context
Recurrent state-space models compress information at every step. Over sequences of T ≈ 2 500 tokens — the standard configuration for TSU — early-frame information decays significantly by the time it reaches late-frame decisions. The three proposals in this section address this decay through complementary mechanisms: local attention for fine-grained precision, a global attention pass for distant-frame communication, and a persistent memory bank for compressed long-term storage.

## 3.1 Sliding Window Attention on Block 1
#### *Problem addressed*
Short actions (Drink from Cup, Sit Down, Take Object) require precise frame-level boundary localisation. The recurrent SSM in Block 1 — although operating at the finest dilation η = 1 — still compresses local information through its hidden-state update, which can blur boundary cues. The action-conditioned precision at τ = 0 in Table 2 of MS-Temba quantifies this loss.

#### *Proposed intervention*
We replace the entirety of Block 1 with a Sliding Window Attention block. Each token attends to W = 32 tokens centred on its position (W/2 left and W/2 right) — corresponding to approximately 16 seconds with the standard segment stride. The implementation uses 4-head multi-head attention with relative position encoding (RoPE), maintaining the same input and output dimensions as the original Block 1 to ensure drop-in compatibility with MS-Fuser. Blocks 2 and 3 retain their dilated SSM design unchanged.

#### *Modules modified*
- `models/extensions/swa_block.py` — new module containing the sliding-window attention block.

- `models/ms_temba.py` — Block 1 is replaced by an instance of SWABlock when a config flag is enabled.

#### *Expected outcome*
Improved boundary precision on short actions, reflected in the action-conditioned precision P_AC at τ = 0 in Table 2 of MS-Temba. Parameter overhead: approximately 0.5–1 M weights. Computational cost remains linear in T because the attention is windowed.

#### *Intellectual lineage*
**Source paper:** *Samba — Simple Hybrid State Space Models for Efficient Unlimited Context (Ren et al., Microsoft, 2024).* Samba interleaves Mamba and Sliding Window Attention layers in a language-modelling architecture, with the observation that a small dose of local attention compensates for the SSM's recurrent compression on fine-grained dependencies. We adapt the same principle to the temporal axis of video: rather than alternating mechanisms layer-by-layer, we replace one specific block — the finest-scale block — with SWA, on the grounds that fine temporal precision is exactly where the SSM is weakest. The novelty lies in the per-scale specialisation: different mechanisms for different temporal resolutions, rather than the same hybrid pattern repeated throughout the network.

## 3.2 Global Shared Attention Block
#### *Problem addressed*
The bidirectional SSM scan in MS-Temba propagates information forward and backward, but the effective dependency between distant frames still degrades with distance. A 'Read' instance at minute 5 and another 'Read' instance at minute 35 of a TSU video have no direct communication channel through the recurrent state: information must survive 2 500 step updates in each direction. Table 2 of MS-Temba reveals that action-conditioned metrics at large τ (τ = 20, τ = 40) remain a weakness.

#### *Proposed intervention*
Between MS-Fuser and the classification head, we insert a single global self-attention block. The block operates once over the entire fused sequence using 4 attention heads at the fused embedding dimension E = 576, with a residual connection: y_final = y^f + GSA(y^f). To avoid the O(T²) cost on T = 2 500, the attention is chunked: the sequence is divided into chunks of 512 tokens with 64-token overlap, attention is computed within each chunk, and the outputs are merged. The effective complexity is O(T · 512) — linear in T for fixed chunk size.

#### *Modules modified*
- `models/extensions/global_shared_attention.py` — new module.

- `models/ms_temba.py` — insert the GSA block between MS-Fuser and the classification head.

#### *Expected outcome*
Targeted improvement on the action-conditioned metrics at τ ∈ {20, 40} in Table 2 of MS-Temba, which directly measure long-range temporal dependency. Parameter overhead: approximately 1.5M weights.

#### *Intellectual lineage*
**Source paper:** *Zamba — A Compact 7B SSM Hybrid Model (Glorioso et al., Zyphra, 2024).* Zamba demonstrates that a single attention block, shared across many SSM layers, is sufficient to recover Transformer-level long-context capability at minimal parameter cost. The original setting is autoregressive language modelling; we transpose the idea to dense per-frame prediction by inserting the shared block once, non-recurrently, on top of the fused representation. The chunked-attention adaptation is necessary because T = 2 500 makes naive global attention prohibitive — a constraint absent in Zamba's per-token decoding regime.

## 3.3 Compressible Memory Tokens
#### *Problem addressed*
Even with sliding-window attention (Section 3.1) and a single global attention pass (Section 3.2), the architecture provides no persistent storage that bypasses the recurrent dynamics entirely. Information available at the start of a 40-minute video must either flow through the SSM recurrence (subject to decay) or through the single GSA pass (which is a one-shot operation, not a learnable memory).

#### *Proposed intervention*
We introduce M = 64 learnable memory tokens m_1, …, m_M ∈ ℝ^E that function as a compressible summary of the video. Two operations are added in parallel to the main pipeline:

Write step (compresses global context into the memory bank):

```
m_i  ←  m_i  +  Σ_t  α_{t,i} · y_t,    α_{t,i} = softmax_t( m_i · y_t )
```

Read step (injects the memory summary into every frame):

```
ŷ_t  =  y_t  +  Σ_i  β_{i,t} · m_i,    β_{i,t} = softmax_i( y_t · m_i )
```

An orthogonality regulariser Σ_{i≠j} (m_i · m_j)² discourages collapse — without it the memory tokens tend to converge to a single representation.

#### *Modules modified*
- `models/extensions/memory_bank.py` — new module implementing the bank, the write and read attention, and the regulariser.

- `models/ms_temba.py` — wire the memory bank in parallel between MS-Fuser and the classification head.

#### *Expected outcome*
The decay distance between any two frames becomes O(1) through the memory channel, complementing the O(T) recurrent path. The proposal is particularly effective for recurring actions far apart in time. Parameter overhead is negligible — M × E ≈ 36 K weights — but the optimisation is more delicate than the other proposals: the orthogonality regulariser is essential, and M is a sensitive hyperparameter.

#### *Intellectual lineage*
**Source papers:** *TALLFormer (Cheng & Bertasius, ECCV 2022) *and *MEGA — Moving Average Equipped Gated Attention (Ma et al., ICLR 2023).* TALLFormer addresses memory cost in long-video Transformers by maintaining a long-memory module that compresses past activations; MEGA introduces a gated moving-average mechanism for long-context language modelling. Our adaptation borrows the compressed-summary idea but reformulates it as a small set of learnable slots updated via bidirectional attention, with explicit orthogonality regularisation. The combination with multi-scale dilated SSMs has not — to our knowledge — appeared in the TAD literature. A contemporaneous work, MambaTAD (arXiv 2511.17929, November 2025), addresses the same decay problem via a diagonal-masked bidirectional SSM module; memory tokens are an orthogonal solution that could be combined.

# 4. Theme 3 — Semantic Priors from Large Language Models
The final classification head of MS-Temba maps the fused per-frame representation to a vector of C class scores through a learned linear layer W ∈ ℝ^{C × E}. This formulation discards available semantic structure: 'Drink from Cup' and 'Drink from Glass' are independent rows of W, and any co-occurrence prior between classes must be discovered from training-set statistics. The three proposals in this section inject external linguistic knowledge into the model, all computed strictly off-line so that inference cost remains unchanged.

## 4.1 CLIP-Based Class Prompts
#### *Problem addressed*
The from-scratch classifier wastes the structural information available through pre-trained text encoders. Two semantically related actions are forced to develop independent representations during training, which hurts particularly on long-tail classes where training data is sparse.

#### *Proposed intervention*
The linear classifier is replaced by cosine similarity between the fused visual representation and a textual prototype for each class. For each class c, a prompt of the form 'a person {action}' is encoded through the frozen CLIP text encoder to produce a 768-dim embedding, then projected through a small learnable MLP to the fused embedding dimension:

```
logit_{t,c}  =  cos( y^f_t ,  ẽ_c ) / τ,    ẽ_c = MLP( CLIP_text(prompt_c) )
```

The temperature τ is a learnable scalar. The prompts and their CLIP embeddings are pre-computed once; only the projection MLP and τ remain trainable. Adding a new class amounts to writing a new prompt — no retraining is required for the rest of the model.

#### *Modules modified*
- `models/extensions/prompt_classifier.py` — new module replacing the original linear head.

- A precomputed file with class embeddings ẽ_c is shipped with the model.

#### *Expected outcome*
An expected +0.5 to +1 mAP gain, concentrated on long-tail classes whose few training instances previously produced a poorly-learned row of W. Inference cost is identical to the original head. A practical secondary benefit is the natural support for zero-shot evaluation on unseen classes.

#### *Intellectual lineage*
**Source paper:** *CLIP — Learning Transferable Visual Models from Natural Language Supervision (Radford et al., ICML 2021).* CLIP itself uses cosine similarity between image and text embeddings for zero-shot classification of natural images. Our adaptation transposes the mechanism to the output head of a temporal model whose input is a sequence-level visual representation rather than a single image embedding. The 'prompt as classifier' pattern has since been adopted across many downstream tasks (CoOp, MaPLe), but its application to dense multi-label TAD with a learned projection over fixed CLIP text embeddings is, to our knowledge, untested in this specific setting.

## 4.2 LLM-Augmented Class Descriptions
#### *Problem addressed*
A bare prompt like 'a person reading' captures a fraction of what differentiates the action from visually similar ones (e.g., 'a person using a tablet'). Pose, gaze direction, object involvement, typical duration, and common co-occurrences are all absent. CLIP's text encoder alone cannot supply this richness from a short template.

#### *Proposed intervention*
Off-line, for each class c we query a large language model (Llama-3, GPT-4, or Claude) with a structured prompt that requests five rich descriptions covering posture, hand motion, objects, duration, and co-occurrence patterns, contextualised for ADL scenarios. Each description is encoded through CLIP-text to produce five 768-dim vectors, which are aggregated in one of two modes:

- Mean aggregation: a single enriched prototype ẽ_c = mean over the five embeddings.

- Prototype set: the five embeddings are kept individually, and inference uses nearest-prototype matching (à la Prototypical Networks).

#### *Modules modified*
- A new off-line preparation script generates and caches the class descriptions.

- `models/extensions/prompt_classifier.py` — supports loading enriched prototypes or prototype sets transparently in place of the bare CLIP prompts.

#### *Expected outcome*
An additional +0.5 to +1.5 mAP over the bare-prompt baseline of Section 4.1, with the largest gains on visually ambiguous classes that benefit from object and context cues. The cost is entirely off-line; inference is unchanged.

#### *Intellectual lineage*
**Source papers:** *DCLIP (Menon & Vondrick, 2023) and CuPL (Pratt et al., 2023).* Both works query GPT-3 to enrich CLIP class prompts for zero-shot image classification. We extend the pattern to dense temporal prediction on long ADL videos, with prompts specifically tailored to elderly-care scenarios and with explicit mention of co-occurrence patterns that the source works do not consider. The use of a prototype set rather than a mean prototype is borrowed from few-shot learning literature (Snell et al., Prototypical Networks, NeurIPS 2017).

## 4.3 LLM Co-occurrence Re-ranking
#### *Problem addressed*
Co-occurrence priors between action classes carry strong signal in ADL settings: 'Walk' and 'Use Phone' frequently co-occur, while 'Cook' and 'Watch TV' essentially never do. MS-Temba must discover these structures implicitly from training-set statistics, which are biased by dataset composition and can be unreliable for rarely-observed class pairs.

#### *Proposed intervention*
Off-line, for every pair of classes (c_i, c_j) we query an LLM with a structured natural-language prompt requesting the likelihood, on a 0–10 scale, of co-occurrence within a 30-second window in an elderly person's daily activities. The responses populate a matrix C ∈ ℝ^{C × C}; for TSU's 51 classes this amounts to 2 601 floating-point values, a negligible storage cost. At inference, the per-frame logits ŷ_t are re-ranked additively:

```
ŷ'_t  =  ŷ_t  +  λ · ( C · ŷ_t )
```

with λ a single learnable scalar. The first term preserves the per-frame visual evidence; the second term boosts classes that are compatible with the highest-confidence predictions and suppresses those that are incompatible.

#### *Modules modified*
- A new off-line script generates and stores the co-occurrence matrix.

- `models/extensions/cooc_reranker.py` — applies the additive re-ranking before the final sigmoid.

#### *Expected outcome*
Targeted improvement on the action-conditioned metrics of Table 2 of MS-Temba: estimated +1 to +1.5 mAP_AC at τ ∈ {0, 20, 40}. The mechanism is interpretable — the matrix C is human auditable as JSON — and updatable: priors can be refined without retraining the model.

#### *Intellectual lineage*
**Source papers:** *MLAD (Tirupattur et al., CVPR 2021) *and *CTRN (Dai et al., BMVC 2021).* Both works model multi-label class dependencies through trainable graph or attention modules whose structure is estimated from training data. Our adaptation replaces the data-driven graph with an LLM-derived matrix, eliminating dataset bias at the cost of a one-time off-line preparation. To the best of our knowledge, this is the first application of LLM-sourced co-occurrence priors to dense TAD. The closest related work in another field is Menon & Vondrick's use of LLM descriptions for zero-shot classification — but they query the LLM for class-level attributes, not pair-level relationships.

# 5. Theme 4 — Cross-Modal Vision-Language Fusion
## 5.1 Frame-Level Caption Fusion (CLIP-It Style)
#### *Problem addressed*
MS-Temba operates on visual features alone. Actions that are visually similar but semantically distinct — 'Pour from Bottle' and 'Drink from Cup', both featuring a hand near the mouth — are difficult to disambiguate purely from CLS tokens. Object identity, intent, and scene context are often discriminative but compressed away by the visual backbone.

#### *Proposed intervention*
Off-line, for each video we sample frames at the same stride used by the segment pooling. For each sampled frame we query a Multimodal LLM (BLIP-2, LLaVA-1.5, or Florence-2) with a prompt such as 'Describe what the person is doing in this image, including objects and posture'. The resulting caption is encoded through the CLIP text encoder, producing a textual feature sequence parallel to the visual feature sequence.

Before the first Temba block, a cross-attention layer fuses the two streams: the visual tokens v_t serve as queries, the textual tokens t_t as keys and values. The augmented visual representation v'_t = v_t + CrossAttention(v_t, t_t) is then fed to the unchanged Temba pipeline. Only the cross-attention parameters (≈ 2 M weights) are trainable; the MLLM and CLIP text encoder are frozen and used purely as feature extractors.

#### *Modules modified*
- A new off-line caption-generation pipeline; the captions and their CLIP embeddings are cached per dataset.

- `models/extensions/caption_fusion.py` — implements the cross-attention layer.

- `models/ms_temba.py` — inserts the fusion layer between the backbone projection and Temba Block 1.

#### *Expected outcome*
An estimated +1 to +2.5 mAP improvement, concentrated on visually ambiguous classes for which the textual modality supplies the disambiguating information (object identity, posture cues). Caption quality varies across MLLMs, which becomes a controlled design knob in the ablation.

#### *Intellectual lineage*
**Source paper:** *CLIP-It — Language-Guided Video Summarisation (Narasimhan, Rohrbach & Darrell, NeurIPS 2021).* CLIP-It generates frame-level captions with an MLLM, embeds them via CLIP text, and uses cross-attention to fuse them with visual features for video summarisation. The same recipe has been demonstrated for video summarisation in MS-Temba's Section 6, but not — to our knowledge — for dense TAD. The key difference in our application is that the textual stream serves as a per-frame disambiguator for classification, rather than as global context for importance ranking.

# 6. Theme 5 — Critical Counterpoint: LLM as Post-Processor
## 6.1 The tempting idea
A natural extension of the LLM-augmented direction (Section 4) is to use an LLM at inference time as a post-processor for MS-Temba's predictions. Given the model's action segments (start, end, class, confidence), an LLM might flag false positives, correct misaligned boundaries, and re-rank confidence based on temporal common-sense. The intuition — that the LLM can apply 'reasoning' over the sequence in a way the dense classifier cannot — is compelling on its face.

## 6.2 Why we reject it
Despite the apparent attractiveness, four arguments make this approach unsuitable for our setting. We document them here explicitly because they reinforce the design choices of the approved themes.

#### *Contradiction of the efficiency narrative*
MS-Temba's most distinctive contribution is its computational efficiency: 17 M parameters, 3.46 GFLOPs, and 51 samples per second on a single GPU. These properties make the model suitable for edge deployment in smart-home and patient-monitoring scenarios — the explicit use cases of the paper. Introducing a 7B-parameter language model at inference time inflates the computational footprint by two orders of magnitude and forfeits this distinctive advantage.

#### *Unreliability of MLLMs on dense TAD*
Section C of the supplementary material of MS-Temba documents that TimeChat and Video-LLaVA fail to detect dense, concurrent actions on long ADL videos — even when explicitly fine-tuned on timestamp-aware data. Models that fail as primary detectors are unlikely to perform well as correctors; the more probable outcome is the introduction of false corrections that degrade rather than improve the baseline predictions.

#### *Latency and reproducibility*
An LLM call adds seconds of latency per video, accumulating to hours over the TSU test set. Furthermore, the results depend on the exact LLM version and prompt formulation, both of which are non-stationary: models are deprecated, APIs change, and prompts drift. Reproducibility across submissions and over time becomes problematic.

#### *Misaligned research mileage*
Even in the most favourable case, an LLM post-processor is a wrapper around the existing architecture; it does not improve MS-Temba's internal representation. A paper built around it would tell a story about prompting, not about TAD architecture — a misalignment with the stated research goals of this project.

## 6.3 What we do instead
The proposals of Section 4 obtain the same semantic benefit — leveraging LLM knowledge — without paying any of the four costs above. The LLM is queried off-line, once per dataset, to produce class prototypes (Section 4.1, 4.2) or co-occurrence priors (Section 4.3). Inference remains pure, fast, and reproducible. The principle we adopt is: the LLM belongs in the training pipeline, not in the inference path.

# 7. Theme 6 — Architectural Enablers
The two proposals in this section are engineering upgrades rather than novel architectural contributions. Their novelty is low, but their impact is high: they unlock the gains of the more ambitious themes by providing the necessary computational and representational headroom. Both should be implemented first in the development timeline.

## 7.1 Migration to Mamba-2 (State Space Duality)
#### *Problem addressed*
MS-Temba's implementation uses Mamba-1 with state dimension N = 16 — a hard constraint of the original CUDA kernel. For sequences of T = 2 500 tokens, this 16-dim hidden state is the primary bottleneck on information capacity: the model cannot retain rich representations of the past.

#### *Proposed intervention*
Mamba-2 reformulates selective state-space models as structured matrix multiplications via the State Space Duality (SSD) framework. The same authors as Mamba-1 (Tri Dao and Albert Gu) released an updated implementation, mamba-ssm ≥ 2.0, which is a near-drop-in replacement. The migration consists of replacing instances of the Mamba module with Mamba2 and increasing the state dimension from N = 16 to N = 64 or N = 128, exploiting matmul-based execution on Tensor Cores at the same wall-clock speed.

#### *Modules modified*
- *requirements.txt* — update to mamba-ssm ≥ 2.0.

- `models/temba_block.py` — replace `Mamba` with `Mamba2`, set d_state = 64 and headdim = 64.

- Re-tune the learning rate (a roughly 25% reduction has been reported in similar migrations).

#### *Expected outcome*
A standalone improvement of +0.5 to +1.5 mAP from the increased state capacity alone. More importantly, the upgrade is a prerequisite for the gated multi-scale of Section 2.1 (which needs additional capacity to host three parallel branches without degradation) and the memory tokens of Section 3.3 (which require a richer hidden state to be effective).

#### *Intellectual lineage*
**Source paper:** *Mamba-2 — Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Dao & Gu, ICML 2024).* The paper introduces the SSD framework which connects selective SSMs to structured-mask linear attention, and provides an efficient matmul-based algorithm. Our use is purely an engineering migration: we adopt the newer backend without modifying the SSD framework itself. The novelty of this proposal is therefore low; however, the practical impact is large, as it unblocks two of the higher-novelty proposals.

## 7.2 LoRA Fine-Tuning of the Visual Backbone
#### *Problem addressed*
MS-Temba's visual backbone (I3D or CLIP-L/14) is fully frozen throughout training. The consequence is that backbone features carry no adaptation to the ADL distribution — domestic scenes, side views of cooking, low-illumination interiors — which is under-represented in the backbone's pre-training corpus. The +7.9 mAP gap in Table 1 of MS-Temba between I3D (36.1) and CLIP (44.0) suggests that backbone quality is a dominant factor in performance, yet the paper provides no clean ablation isolating architectural contributions from backbone contributions.

#### *Proposed intervention*
We inject Low-Rank Adaptation (LoRA) layers of rank r ∈ {4, 8} on the query, key, value, and output projection matrices of every Transformer block in the CLIP-L/14 backbone. The original weights remain frozen; only the LoRA adapters are trainable, adding approximately 1–2 M parameters versus the 87 M frozen weights of the backbone. Gradient checkpointing on the backbone enables training within 24 GB of GPU memory.

#### *Modules modified*
- A LoRA adapter is added to each attention layer of the backbone, in-place at load time.

- `train.py` — gradients propagate through the LoRA adapters; backbone weights remain frozen.

#### *Expected outcome*
A standalone gain of +1 to +3 mAP from backbone adaptation, with the additional methodological value of finally providing a clean ablation between 'pure architectural contribution' and 'backbone contribution' — a disentanglement that MS-Temba itself never performs.

#### *Intellectual lineage*
**Source paper:** *LoRA — Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022).* LoRA was introduced for parameter-efficient fine-tuning of large language models, with the original experiments on GPT-3. Its application to vision backbones is now standard practice in the parameter-efficient fine-tuning literature (e.g., VeRA, AdaLoRA), and applying it to CLIP for downstream visual tasks is well-established. Our specific application is therefore not novel; the contribution is the inclusion of the long-missing backbone-adaptation ablation in TAD with multi-scale Mamba.

# 8. Novelty Ranking and Strategic Synthesis
The twelve proposals do not lie on a single axis of value: some are high-novelty research contributions, others are engineering enablers; some carry implementation risk, others are drop-in changes. Table 8.1 summarises the trade-offs and ranks the proposals on three independent dimensions: novelty (how original the contribution is relative to the literature), expected gain (typical improvement on TSU mAP), and implementation effort.

| **Rank** | **Proposal** | **Novelty** | **Gain (mAP)** | **Effort** | **Notes** |
| --- | --- | --- | --- | --- | --- |
| **1** | 2.1 Gated Multi-Scale Dilations | **Very High** | +1 to +2.5 | Medium | Central architectural innovation; requires Mamba-2 |
| **2** | 3.3 Compressible Memory Tokens | **Very High** | hard to predict | Research-grade | Highest research mileage; optimisation is delicate |
| **3** | 4.3 LLM Co-occurrence Re-ranking | **High** | +1 to +1.5 (AC) | Drop-in | First LLM-sourced class-pair priors in TAD |
| **4** | 3.1 Sliding Window Attention (Block 1) | **High** | boundary precision | Medium | First per-scale Samba-style hybrid in TAD |
| **5** | 3.2 Global Shared Attention | **High** | AC at τ ≥ 20 | Medium | Single Zamba-style block on the fused representation |
| **6** | 2.2 Scale-Specific Auxiliary Loss | **High** | +0.5 to +1 | Drop-in | Strong interpretability story; zero new parameters |
| **7** | 5.1 Frame-Level Caption Fusion | **Medium** | +1 to +2.5 | Medium | First CLIP-It style fusion for dense TAD |
| **8** | 4.2 LLM-Augmented Descriptions | **Medium** | +0.5 to +1.5 | Drop-in | Extension of 4.1 via richer prompts; off-line cost |
| **9** | 4.1 CLIP-Based Class Prompts | **Low** | +0.5 to +1 | Drop-in | Foundation for 4.2 and 4.3; enables zero-shot |
| **10** | 7.2 LoRA Backbone Fine-Tuning | **Low** | +1 to +3 | Easy | Highest absolute gain; supplies the missing ablation |
| **11** | 7.1 Mamba-2 Backend Migration | **Low** | +0.5 to +1.5 | Drop-in | Best ROI per effort; prerequisite for 2.1 and 3.3 |
| **—** | *6 LLM Post-Processor (rejected)* | N/A | uncertain | Heavy | *Explicitly discussed and rejected (Section 6)* |

*Table 8.1 — Novelty, expected gain, and effort for each proposal. Rank is determined primarily by novelty, with expected gain and effort serving as tie-breakers. ‘AC’ denotes action-conditioned metrics from Table 2 of MS-Temba.*

# 9. Recommended Implementation Timeline
The implementation order is dictated by two principles: (i) infrastructure proposals should precede architectural ones, because they unlock the necessary headroom; and (ii) clean ablation requires that each proposal be evaluated against the cumulative state of all previously-implemented proposals, not against the original baseline alone. The timeline below is structured to permit a publication-grade ablation table at the end of each phase.

## 9.1 Phase 1 — Infrastructure (Weeks 1–4)
Implement the two architectural enablers from Section 7. Reproduce the MS-Temba baseline first; establish that the I3D and CLIP backbones reach within ±0.5 mAP of the published numbers. Then migrate the backend to Mamba-2 and report the standalone gain. Finally, add LoRA on the CLIP backbone and report the cumulative gain. The deliverable at the end of Phase 1 is a clean three-row ablation table: baseline, +Mamba-2, +Mamba-2 +LoRA.

## 9.2 Phase 2 — Adaptive Temporal Scales (Weeks 5–8)
Add the scale-specific auxiliary loss of Section 2.2 first, since it requires no architectural change. Then implement the gated multi-scale of Section 2.1, with the warm-start ensuring that the initial behaviour matches the original architecture. Per-bucket mAP results should be reported to demonstrate that the blocks specialise as intended. The deliverable is an ablation table extending the Phase 1 one with +aux-filtered and +gated-multi-scale rows.

## 9.3 Phase 3 — Long-Range Context (Weeks 9–13)
Implement the Sliding Window Attention of Section 3.1 first, since it is the most localised change. Then add the Global Shared Attention of Section 3.2. Action-conditioned metrics at τ ∈ {0, 20, 40} are the primary evaluation focus for this phase. The deliverable extends the ablation table and adds a dedicated table on action-conditioned metrics.

## 9.4 Phase 4 — Semantic Priors (Weeks 14–17)
Implement Section 4.1 (CLIP prompts) first, then Section 4.2 (LLM-augmented descriptions), then Section 4.3 (co-occurrence module). Each step extends the previous one, allowing a clean incremental ablation. Long-tail class performance should be reported separately, since these proposals are most effective on classes with few training instances.

## 9.5 Phase 5 — Cross-Modal and Research-Grade Extensions (Weeks 18–24)
Implement the caption fusion of Section 5.1 and the memory tokens of Section 3.3. The memory tokens are intentionally placed last because they carry the highest implementation risk and their outcome is most uncertain — placing them at the end avoids derailing the more predictable middle phases. If memory tokens fail to deliver consistent gains, they can be moved to future work without compromising the rest of the contribution.

## 9.6 Ablation Study Strategy
The final ablation table should follow an additive, not subtractive, layout: starting from the baseline, each row adds one proposal on top of the previous configuration. This layout directly answers the reviewer question 'is each contribution necessary?'. A complementary leave-one-out table, in which each row removes one proposal from the full configuration, answers the alternative reviewer question 'is each contribution sufficient?'. Both should be reported on TSU; Charades can be used as a sanity check with a reduced subset of configurations.

Action-conditioned metrics from Table 2 of MS-Temba should be reported separately, because several proposals (notably 3.2 Global Shared Attention and 4.3 LLM Co-occurrence Re-ranking) target specifically this evaluation regime. Per-class mAP analysis, with classes stratified by training-set frequency, is essential for the proposals of Section 4 whose gains concentrate on long-tail classes.

# 10. Summary and Outlook
This document has presented twelve proposals for improving MS-Temba, organised into six thematic axes and one critical counterpoint. The proposals span the full spectrum from high-novelty research contributions (the gated multi-scale of Section 2.1 and the memory tokens of Section 3.3) to engineering enablers (the Mamba-2 migration of Section 7.1 and the LoRA fine-tuning of Section 7.2). For each, we have traced the intellectual lineage to its source paper, distinguishing between the original application of the idea and our specific adaptation to dense Temporal Action Detection.

Three paper trajectories are coherent within this roadmap. An 'architectural paper' would combine the proposals of Themes 1, 2 and 6 — a pure-vision contribution emphasising the content-aware multi-scale and hybrid-attention story. An 'LLM-augmented paper' would combine Themes 1, 3, 4 and 6 — building on the architectural one with the semantic-priors track. A full thesis would cover all themes including the memory-tokens extension as a research-grade future-work direction.

The recommended timeline organises the implementation into five phases of approximately four weeks each, with the structure designed to produce a publication-quality ablation study at the end of each phase. Phase 1 establishes the infrastructure and supplies the long-missing backbone-adaptation ablation; Phase 2 delivers the central architectural innovation; Phases 3 and 4 build out the long-range and semantic-priors tracks respectively; Phase 5 explores the highest-risk and highest-novelty extensions. The plan is internally robust to the partial failure of any single proposal: the rejection of a phase-5 idea would not compromise the rest, and the additive ablation structure ensures that each contribution is independently auditable.

Should the proposals deliver their expected gains, the resulting MS-Temba v2 would weigh approximately 25–28 M parameters — still well below MS-TCT's 87 M — while addressing the seven limitations enumerated in Section 1.2. The path from the present roadmap to a follow-up publication is concrete and tractable, and represents the planned trajectory of the master's research project documented here.

*End of document — Matteo Di Iorio, internal report for supervisor review.*

Matteo Di Iorio   ·   page   of