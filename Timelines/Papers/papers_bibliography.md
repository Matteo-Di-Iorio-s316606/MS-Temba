# Paper Bibliography for MS-Temba Future Work

Lista completa dei paper rilevanti per il progetto, organizzati per area tematica. Per ogni paper trovi un breve abstract che ne descrive il contenuto e l'utilità per lo studio.

---

## 1. Temporal Action Detection — Baseline e ambito diretto

### MS-TCT (Dai et al., CVPR 2022)
*Multi-Scale Temporal ConvTransformer for Action Detection.* Architettura ibrida ConvTransformer per TAD multi-label denso, costruita per Charades e TSU. Combina una gerarchia di temporal convolutions per le relazioni locali con un Transformer encoder per le interazioni globali, e introduce una "instance-center" branch per la posizione relativa rispetto al centro dell'istanza. Con 87M parametri è il **competitor diretto principale** di MS-Temba, che lo supera con 17M parametri.

### PDAN (Dai et al., WACV 2021)
*Pyramid Dilated Attention Network for Action Detection.* Pioniere del concetto di "dilated receptive field" applicato all'attention temporale. Stack di blocchi di dilated attention organizzati a piramide, con dilatazioni crescenti per coprire scale temporali multiple. È il **diretto antenato concettuale** di MS-Temba (stessa intuizione di dilatazioni multi-scala, ma su attention invece che SSM).

### TGM (Piergiovanni & Ryoo, ICML 2019)
*Temporal Gaussian Mixture Layer for Videos.* Layer convoluzionale i cui kernel sono mixture di Gaussiane apprendibili lungo la dimensione temporale, parametrizzate da μ e σ invece che dai singoli pesi. Riferimento "ultra-leggero" della Tabella 1 di MS-Temba (solo 2M parametri).

### Super-event (Piergiovanni & Ryoo, CVPR 2018)
*Learning Latent Super-Events to Detect Multiple Activities in Videos.* Introduce soft attention temporale che impara filtri a scala variabile per catturare contesti di alto livello (es. "una partita di calcio contiene passaggi, tiri, ecc."). Primo lavoro sul context multi-scale per detection multi-label.

### MLAD (Tirupattur et al., CVPR 2021)
*Modeling Multi-Label Action Dependencies for Temporal Action Localization.* Transformer encoder che cattura esplicitamente le dipendenze tra classi di azioni in video multi-label, sia in senso temporale sia cross-class. **Cruciale**: introduce le metriche Action-Conditioned (PAC, RAC, F1AC, mAPAC) usate in Tabella 2 di MS-Temba. Riferimento per il problema della co-occorrenza inter-classe.

### CTRN (Dai et al., BMVC 2021)
*Class Temporal Relational Network for Action Detection.* Modella le dipendenze tra classi di azioni nel tempo costruendo un grafo class-temporal e ragionando su di esso. Esplicita la struttura relazionale tra etichette, complementare al puro temporal modeling. Ispira la nostra **proposta T3.c** (LLM co-occurrence module).

### HAAN (Gao et al., ACM MM 2023)
*Human Action Aware Network for Multi-Label Temporal Action Detection.* Detection multi-label che incorpora prior sulla struttura del corpo umano e sul movimento per disambiguare azioni concorrenti. Iniezione di prior antropologici (parts del corpo) nel feature space, utile in ADL dove le interazioni con oggetti sono frequenti.

### DualDETR (Zhu et al., CVPR 2024)
*Dual DETRs for Multi-Label Temporal Action Detection.* Estende il paradigma DETR (set-prediction con learnable queries) al setting multi-label, usando due decoder paralleli — uno per le boundaries e uno per la classificazione. **Disaccoppia esplicitamente** la localizzazione dalla classificazione, problema che MS-Temba accorpa nella sola BCE per-frame. Principale baseline DETR-style della Tabella 1.

### AAN (Dai et al., BMVC 2023)
*Attributes-Aware Network for Temporal Action Detection.* Detection guidato da attributi semantici (oggetti, scene, parts del corpo) estratti dai frame e fusi col branch principale. Multi-task learning con supervisione ausiliaria di attributi — particolarmente forte con backbone CLIP.

### ActionFormer (Zhang et al., ECCV 2022)
*Localizing Moments of Actions with Transformers.* TAL anchor-free, single-shot, basato su Transformer con local self-attention multi-scala e light-weight decoder che classifica ogni momento e ne stima i confini. Stabilisce il **paradigma "FCOS-style"** per TAD: predizione densa per-frame con regressione esplicita degli offset di start/end. Modello di riferimento per la possibile aggiunta di una boundary regression head a MS-Temba.

### TALLFormer (Cheng & Bertasius, ECCV 2022)
*Temporal Action Localization with Long-memory Transformer.* Affronta il problema dell'alto costo di memoria GPU nel processare video lunghi proponendo un Transformer end-to-end con memoria di lungo termine. **Ispirazione diretta** per la nostra proposta T2.c (memory tokens).

### TTM (Ryoo et al., CVPR 2023)
*Token Turing Machines.* Modello sequenziale per video basato su un meccanismo di memoria ispirato alle Neural Turing Machines, con read/write tokens su una memoria persistente. Riapplica concetti di Turing Machine al dominio Transformer. Baseline ViViT della Tabella 1 di MS-Temba.

### PointTAD (Tan et al., NeurIPS 2022)
*Multi-Label Temporal Action Detection with Learnable Query Points.* Detection multi-label con learnable query points invece di anchor box, ispirato a Sparse R-CNN nel temporale. Prima estensione del paradigma point-based detection a TAD multi-label.

### Coarse-Fine Networks (Kahatapitiya & Ryoo, CVPR 2021)
*Coarse-Fine Networks for Temporal Activity Detection in Videos.* Architettura a due rami che processa il video a risoluzioni temporali diverse (coarse e fine) e le fonde. Approccio multi-scala temporale alternativo, baseline su Charades.

---

## 2. State Space Models e Mamba — Fondazionali

### S4 (Gu, Goel & Ré, ICLR 2022)
*Efficiently Modeling Long Sequences with Structured State Spaces.* Introduce gli Structured State Space sequence models, che riformulano gli SSM continui in modo discretizzato e parametrizzato (matrice A diagonale + low-rank, HiPPO initialization) per processare sequenze lunghissime in tempo lineare. **Base teorica di tutto il filone Mamba**.

### Mamba (Gu & Dao, 2023)
*Linear-Time Sequence Modeling with Selective State Spaces.* Estende S4 introducendo **selective state space parameters**: le matrici A, B, C dipendono dall'input, abilitando il modello a selezionare cosa propagare lungo la sequenza. Aggiunge un algoritmo hardware-aware (parallel scan). **Spina dorsale di MS-Temba**.

### Mamba-2 / SSD (Dao & Gu, ICML 2024)
*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* Connette teoricamente SSM e linear attention attraverso matrici strutturate (State Space Duality). Permette algoritmi più veloci basati su matmul invece di scan, e sblocca state dim 64-128 vs 16 di Mamba-1. **Cruciale per la nostra proposta T6.a** (drop-in upgrade del backend).

### Vision Mamba / Vim (Zhu et al., ICML 2024)
*Efficient Visual Representation Learning with Bidirectional State Space Model.* Prima architettura pure-SSM per la visione, con **bidirectional scanning** (forward + backward) per spezzare la natura causale di Mamba. 2.8× più veloce di DeiT con 86.8% meno memoria GPU. MS-Temba eredita direttamente la scelta del bidirectional scan.

### VMamba (Liu et al., NeurIPS 2024)
*VMamba: Visual State Space Model.* Estende Mamba a immagini 2D introducendo il Cross-Scan Module (CSM) che esegue scan in 4 direzioni. L'idea che la "direzione di scan" sia un design choice cruciale per l'adattamento di Mamba a domini diversi.

---

## 3. Mamba per video e task temporali

### VideoMamba (Li et al., ECCV 2024)
*State Space Model for Efficient Video Understanding.* Adatta Mamba al video classification con scansione spatio-temporale, scalando a sequenze molto lunghe mantenendo efficienza lineare. Prima dimostrazione che Mamba è competitivo con video Transformer su action recognition. MS-Temba usa la sua configurazione SSM (state-dim=16) come riferimento.

### Video Mamba Suite (Chen et al., 2024)
*State Space Model as a Versatile Alternative for Video Understanding.* Toolbox che valuta Mamba su un'ampia gamma di task video: classification, retrieval, action localization (HACS, GTEA), grounding. **Il paper più vicino a MS-Temba sul terreno della localizzazione**, ma limitato a video di ~3 minuti.

### VideoMambaPro / Snakes and Ladders (Lu et al., ICCV 2025)
*Two Steps Up for VideoMamba.* Identifica due limiti specifici di VideoMamba (token decay e cross-token confusion) e li mitiga con una combinazione di residui e Mamba a due passi. Diagnosi precisa dei fallimenti di Mamba su video lunghi e remediation chirurgica.

### TranS4mer (Islam et al., CVPR 2023)
*Efficient Movie Scene Detection using State-Space Transformers.* Combina S4 con self-attention per scene detection in film, modellando dipendenze temporali a scala di ore. **Primo ibrido SSM+Attention** per long-form video understanding — apre la strada alle soluzioni miste che ispirano i nostri Temi 2.a e 2.b.

### Manta (Zatsarynna et al., CVPR 2025)
*Diffusion Mamba for Efficient and Effective Stochastic Long-Term Dense Action Anticipation.* Action anticipation di lungo termine combinando diffusion models e Mamba per produrre sequenze di azioni future. Mostra che Mamba può modellare distribuzioni stocastiche di sequenze.

### MOGO (Liu et al., NeurIPS 2025)
*Mamba Only Glances Once: A Lightweight Framework for Efficient Video Action Detection.* Framework leggero per video action detection spazio-temporale (atomica) basato su Mamba, ottimizzato per inference rapida. Single-pass detection con Mamba.

### MuSE (Tang et al., 2024)
*Mamba is Efficient Multi-scale Learner for Text-Video Retrieval.* Adatta Mamba al retrieval text-video con encoding multi-scala dei video. **Lavoro più vicino concettualmente a MS-Temba** (multi-scale + Mamba), ma in dominio retrieval invece che detection.

### Vivim (Yang et al., 2024)
*A Video Vision Mamba for Medical Video Object Segmentation.* Prima estensione di Mamba a dense prediction temporale (segmentazione frame-by-frame in video medici), problema strutturalmente simile a TAD denso.

### MambaTAD (arXiv 2511.17929, novembre 2025)
*Diagonal-Masked Bidirectional State-Space Module for Temporal Action Detection.* Affronta esplicitamente il problema del decay del contesto temporale in TAD. Introduce DMBSS (Diagonal-Masked Bidirectional State-Space) per facilitare global feature fusion, una global feature fusion head che raffina la detection progressivamente, e uno state-space temporal adapter (SSTA) per fine-tuning end-to-end. **Lavoro contemporaneo a MS-Temba**, da citare obbligatoriamente in qualsiasi follow-up.

### SBM (MDPI Mathematics, 2025)
*Separated Bidirectional Mamba for Temporal Action Localization.* TAL con SSM che modellano frame changes come state transitions, separando i flussi forward e backward. Riconcettualizza il task TAL come "rilevamento di transizioni di stato".

### State-Sensitive Mamba (Neurocomputing, 2025)
*State-Sensitive Mamba for Temporal Action Localization.* TAL con sensibilità esplicita ai cambi di stato (boundaries) e centroid sequence enhancement. Estende la lunghezza della sequenza a 5× rispetto al modello precedente sfruttando il costo lineare e la sensibilità ai boundary di stato.

---

## 4. Architetture ibride Mamba + Attention (LLM scale)

### Jamba (AI21 Labs, 2024)
*A Hybrid Transformer-Mamba Language Model.* Primo modello ibrido SSM+Attention production-grade. Interleaves blocchi di Transformer e Mamba con rapporto 1:7 (un layer di attention ogni 7 di Mamba) come sweet spot empirico. Aggiunge MoE per scalare la capacità senza far esplodere i parametri attivi. **Conferma sperimentalmente** che un piccolo numero di layer di attention sparsi tra molti layer Mamba migliora la qualità.

### Jamba-1.5 (AI21 Labs, 2024)
*Hybrid Transformer-Mamba Models at Scale.* Versioni instruction-tuned dell'architettura Jamba (94B e 12B parametri), con context window di 256K token. Conferma che la formula 1:7 regge a scala di 398B parametri totali. Introduce ExpertsInt8 quantization.

### Samba (Microsoft, 2024)
*Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* Architettura ibrida che combina layer-wise Mamba con **Sliding Window Attention**. Comprime selettivamente la sequenza in stati nascosti ricorrenti mantenendo la capacità di richiamare con precisione le memorie recenti tramite attention locale. **Ispirazione diretta** della nostra proposta T2.a.

### Zamba / Zamba2 (Zyphra, 2024)
*A Compact 7B SSM Hybrid Model.* Ibrido SSM-transformer da 7B con **un singolo modulo di attention condiviso** tra molti layer SSM. Costo parametrico minimo, beneficio di full-attention. Zamba2 usa Mamba-2 e LoRA per specializzare il blocco condiviso. **Ispirazione diretta** della nostra proposta T2.b.

### Nemotron-H (NVIDIA, 2025)
*Hybrid Mamba-Attention Architecture.* Famiglia di ibridi sviluppati da NVIDIA con focus su efficienza di training e inference su hardware NVIDIA. Co-design hardware/architettura: layer di attention strategicamente posizionati per massimizzare l'utilizzo di Tensor Cores, layer SSM per ridurre la memoria KV-cache.

### YOCO (Microsoft, 2024)
*You Only Cache Once: Decoder-Decoder Architectures for Language Models.* Architettura "decoder-decoder" dove il primo decoder è SSM (efficiente, lineare) e produce KV-cache una sola volta; il secondo decoder è Transformer e riusa quella cache. Pattern interessante per dense prediction adattabile al MS-Fuser.

### MambaVision (NVIDIA, 2024)
*A Hybrid Mamba-Transformer Vision Backbone.* Backbone gerarchico che combina blocchi conv → Mamba → attention, ognuno nel proprio "regime ottimale" (low-level → mid-range → global). **Modello concettuale ideale** per estendere MS-Temba sull'asse temporale invece che spaziale.

### Caduceus (Schiff et al., ICML 2024)
*Bi-Directional Equivariant Long-Range DNA Sequence Modeling.* Mamba bidirezionale per genomica, con reverse-complement equivariance built-in. Esempio di come iniettare inductive biases specifici del dominio nello state-space.

---

## 5. Architetture concorrenti / alternative

### RWKV (Peng et al., 2023)
*Reinventing RNNs for the Transformer Era.* Architettura ricorrente con time-mix e channel-mix che si comporta come Transformer in training (parallelo) e come RNN in inference (lineare). Predecessore concettuale di Mamba con motivazione simile.

### RetNet (Microsoft, 2023)
*Retentive Network: A Successor to Transformer for Large Language Models.* Retention mechanism simultaneamente parallelo (training), ricorrente (inference) e chunkwise-recurrent (long context). Mamba-2 mostra che RetNet è un caso speciale di SSD con maschera strutturata.

### xLSTM (Hochreiter et al., 2024)
*Extended Long Short-Term Memory.* Riproposizione moderna di LSTM con exponential gating e matrix memory (sLSTM e mLSTM), competitive con Transformer e Mamba a parità di compute. Alternativa al SSM come building block.

### Hyena (Poli et al., ICML 2023)
*Towards Larger Convolutional Language Models.* Lunghe convoluzioni implicite parametrizzate da reti neurali + gating, alternativa sub-quadratica all'attention. Predecessore concettuale di Mamba senza la selezione data-dependent.

---

## 6. Backbones visivi e foundation models

### I3D (Carreira & Zisserman, CVPR 2017)
*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.* Inflated 3D ConvNet, primo backbone video pre-addestrato su Kinetics. Backbone "tradizionale" usato in MS-Temba per il confronto storico.

### CLIP (Radford et al., ICML 2021)
*Learning Transferable Visual Models from Natural Language Supervision.* Foundation model multimodale image-text addestrato su 400M coppie con contrastive learning. Backbone primario di MS-Temba per le SOTA performance, e fonte degli embedding testuali per la nostra proposta T3.a.

### ViViT (Arnab et al., ICCV 2021)
*A Video Vision Transformer.* Estensione di ViT al video con factorized space-time attention. Backbone alternativo nella Tabella 1 di MS-Temba (TTM).

### X3D (Feichtenhofer, CVPR 2020)
*Expanding Architectures for Efficient Video Recognition.* Espansione progressiva di reti video 2D per ottenere architetture efficienti. Baseline nel confronto Coarse-Fine di MS-Temba.

---

## 7. Multimodal LLM e video understanding

### CLIP-It (Narasimhan, Rohrbach & Darrell, NeurIPS 2021)
*Language-Guided Video Summarization.* Genera frame-level captions con un MLLM, embed con CLIP-text, cross-attend con feature visive. **Ispirazione diretta** della nostra proposta T4 (caption fusion).

### Video-LLaVA (Lin et al., 2023)
*Learning United Visual Representation by Alignment Before Projection.* MLLM video con allineamento delle rappresentazioni visive prima della proiezione nel language model. Citato in MS-Temba come esempio di MLLM che fallisce su TAD denso (suppl. §C).

### TimeChat (Ren et al., 2023)
*A Time-Sensitive Multimodal Large Language Model for Long Video Understanding.* MLLM video addestrato esplicitamente sui timestamp dei video. Citato in MS-Temba come esempio che, nonostante il training sui timestamp, fallisce a rilevare azioni concorrenti in video complessi.

### Video-ChatGPT (Maaz et al., ACL 2024)
*Towards Detailed Video Understanding via Large Vision and Language Models.* MLLM video per comprensione dettagliata. Esempio della linea di ricerca MLLM-detector che il paper originale critica esplicitamente.

### A2Summ (He et al., CVPR 2023)
*Align and Attend: Multimodal Summarization with Dual Contrastive Losses.* Modella temporal correspondence tra captions testuali e feature visive per summarization. Baseline nella Tabella 7 di MS-Temba.

### LLMVS (Lee et al., CVPR 2025)
*Video Summarization with Large Language Models.* Genera frame-level captions con un MLLM, poi query un LLM per produrre frame-level importance scores. State-of-the-art in video summarization superato da MS-Temba.

---

## 8. Tecniche di efficient fine-tuning

### LoRA (Hu et al., ICLR 2022)
*Low-Rank Adaptation of Large Language Models.* Adattamento di modelli pre-addestrati tramite low-rank decomposition delle matrici di update, congelando i pesi originali. **Strumento centrale** della nostra proposta T6.b per fine-tunare il backbone CLIP a costo di 1-2M parametri trainable invece di 87M.

---

## 9. Datasets di riferimento

### TSU — Toyota Smarthome Untrimmed (Dai et al., IEEE TPAMI 2022)
*Real-World Untrimmed Videos for Activity Detection.* Dataset ADL untrimmed con 51 classi di azioni quotidiane, fino a 5 azioni co-occorrenti per frame, durata media dei video di 21 minuti. **Benchmark primario di MS-Temba**.

### Charades (Sigurdsson et al., ECCV 2016)
*Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding.* Dataset large-scale di 9848 video untrimmed con 157 classi di azioni e 66500 annotazioni. Media di 6.8 istanze di azione per video. **Secondo benchmark di MS-Temba**.

### TVSum (Song et al., CVPR 2015)
*Summarizing Web Videos using Titles.* 50 video di 10 generi (documentari, vlog, ecc.) annotati per importance da 15-20 individui. Benchmark per video summarization.

### SumMe (Gygli et al., ECCV 2014)
*Creating Summaries from User Videos.* 25 video da 30s a 6 minuti registrati con camera egocentrica. Benchmark per video summarization.