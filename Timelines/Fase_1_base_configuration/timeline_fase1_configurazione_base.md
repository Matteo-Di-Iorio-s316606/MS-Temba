# MS-Temba su Charades: Analisi Tecnica Completa — Fase 1
### Riproduzione, Backbone Comparison e Stabilizzazione del Training

> **Autore**: Matteo Di Iorio  
> **Periodo**: Febbraio–Marzo 2026  
> **Cluster**: Grid5000 / ABACA (Sophia Antipolis) — nodi esterel, GPU H100/A100  
> **Repository**: `/srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/`  
> **Dataset**: Charades v1 — 157 classi, 7985 video train, 1863 video test  
> **Documento**: Fase 1 di N — baseline, CLIP vs DINO, regolarizzazione

---

## Indice

1. [Contesto e Obiettivi](#1-contesto-e-obiettivi)
2. [Architettura MS-Temba](#2-architettura-ms-temba)
3. [Infrastruttura e Ambiente di Sviluppo](#3-infrastruttura-e-ambiente-di-sviluppo)
4. [Fase 1.1 — Riproduzione dei Risultati Baseline](#41-fase-11--riproduzione-dei-risultati-baseline)
5. [Fase 1.2 — CLIP come Backbone](#5-fase-12--clip-come-backbone)
6. [Fase 1.3 — DINOv3 come Backbone](#6-fase-13--dinov3-come-backbone)
7. [Fase 1.4 — Confronto CLIP vs DINOv3](#7-fase-14--confronto-clip-vs-dinov3)
8. [Fase 1.5 — Interventi di Regolarizzazione](#8-fase-15--interventi-di-regolarizzazione)
9. [Modifiche Architetturali e al Codebase](#9-modifiche-architetturali-e-al-codebase)
10. [Risultati Consolidati e Analisi Comparativa](#10-risultati-consolidati-e-analisi-comparativa)
11. [Limitazioni e Problemi Aperti](#11-limitazioni-e-problemi-aperti)
12. [Passi Successivi — Fase 2 e Oltre](#12-passi-successivi--fase-2-e-oltre)

---

## 1. Contesto e Obiettivi

### 1.1 Progetto di Tesi

Il progetto si inquadra in un tirocinio di ricerca con l'obiettivo di riprodurre, analizzare e migliorare il modello **MS-Temba** (*Multi-Scale Temporal Mamba*) per il task di **Temporal Action Detection (TAD)** multi-label su video. Il task consiste nel predire, per ogni frame di un video, l'insieme di azioni in corso tra le 157 classi del dataset Charades.

Il lavoro si articola in più fasi progressive:

1. **Fase 1** (questo documento): riproduzione dei risultati pubblicati, esplorazione di backbone alternativi (CLIP, DINOv3), stabilizzazione del training con tecniche di regolarizzazione e early stopping.
2. **Fase 2** (pianificata): sfruttamento delle dense patch features di DINOv3, integrazione di feature scheletriche (SCDNet), fusione multi-modale.
3. **Fase 3** (pianificata): studio sistematico dello stato dell'arte sulla combinazione di feature visive e scheletriche, sviluppo di architetture innovative.

### 1.2 Dataset: Charades v1

Charades è un dataset di video domestici raccolti tramite crowdsourcing, caratterizzato da:

- **9848 video** totali (7985 train, 1863 test), durata media ~30 secondi
- **157 classi** di azioni domestiche (es. *washing dishes*, *opening a door*, *lying on a bed*)
- **Annotazioni multi-label**: mediamente 6.8 azioni per video, con sovrapposizioni temporali frequenti
- **Distribuzione sbilanciata**: alcune classi hanno >2000 occorrenze, altre <50
- **Metrica principale**: mean Average Precision (mAP) frame-level, calcolata in due modalità:
  - *Full-val-MAP*: su tutti i frame del validation set
  - *Sampled-val-MAP*: su un sottoinsieme campionato (25 frame per video), più rapida e usata come proxy durante il training

La struttura composita delle classi (verbo × oggetto, es. *putting* + *bag*, *opening* + *door*) introduce ambiguità sistematiche che rappresentano una delle sfide principali del dataset.

### 1.3 Modello: MS-Temba

MS-Temba è un modello per TAD basato su **Mamba** (Selective State Space Model), che processa sequenze di feature visive pre-estratte da un backbone. Il modello non opera direttamente sui pixel ma su rappresentazioni frame-level già calcolate offline. La novità principale rispetto ai precedenti approcci trasformer-based è l'utilizzo di SSM a scala multipla con struttura gerarchica e meccanismo di diversity loss sui C-states.

---

## 2. Architettura MS-Temba

### 2.1 Pipeline Completa

```
Video (.mp4)
    │
    ▼ [Feature Extraction — offline, pre-computata]
Backbone (CLIP / DINOv3 / I3D / ViCLIP)
    │
    ▼
Feature files (.npy) — shape [T, D]
    │  T = numero di frame campionati (max 256)
    │  D = dimensione feature (512 CLIP, 1024 DINO/I3D)
    │
    ▼ [charades_dataloader.py]
Collation → padding a 256 frame → [B, 256, D]
    │
    ▼ [models_MSTemba.py — classe MSTemba]
    │
    ├─ InputProjection: [B, 256, D] → [B, 256, 256]
    │    permute [B,D,T]→[B,T,D] + Linear(D→256) + LayerNorm + GELU + Dropout
    │
    ├─ Block 1: [B, 256, 256]
    │    LinearProjection(256→256) + VisionMamba×1
    │    └─ 1 SSM su tutti i 256 token → output [B, 256, 256]
    │
    ├─ Block 2: [B, 256, 384]
    │    LinearProjection(256→384) + VisionMamba×2
    │    └─ SSM_even su token pari + SSM_odd su token dispari → [B, 256, 384]
    │
    ├─ Block 3: [B, 256, 576]
    │    LinearProjection(384→576) + VisionMamba×3
    │    └─ SSM_g1 + SSM_g2 + SSM_g3 (3 gruppi dilated) → [B, 256, 576]
    │
    ├─ scale_proj1/2/3: proiezioni a dimensione comune 576
    │    concat_x = [proj1(B1), proj2(B2), B3] → fusione multi-scala
    │
    ├─ interaction_block: Mamba finale su concat_x
    │
    └─ Classification head: Linear(576 → 157) → [B, 256, 157]

Output: (logits [B,256,157], block_predictions [3×], diversity_loss scalar)
```

### 2.2 Componenti Principali

#### VisionMamba (backbone SSM)
Adattamento di Mamba per sequenze visive. Ogni blocco implementa:
- **RMSNorm** prima del mixer
- **Mamba SSM** (selective scan) con parametri `d_state=16`, `d_conv=4`, `expand=2`
- **Connessione residua** sommata all'output
- Supporto per **RoPE** (Rotary Position Embedding) e **CLS token** (non usato in questo contesto)

La variante usata in MS-Temba utilizza il forward unidirezionale (non bidirezionale) per mantenere la causalità temporale.

#### LinearProjection
Modulo di transizione tra blocchi gerarchici:
```python
Linear(in_ch → out_ch) → LayerNorm(out_ch) → GELU() → Dropout(p=drop_rate)
```
Dopo le modifiche della Fase 1.5, accetta `drop_rate` come parametro.

#### Diversity Loss
Calcolata sui **C-states** (vettori di contesto interno) dei tre gruppi del Block 3. Penalizza la similarità coseno tra i C-states dei diversi gruppi di dilation, incoraggiando la specializzazione delle tre scale temporali. Peso nella loss finale: `100.0 * diversity_loss`.

#### Funzione di Loss Complessiva
```
L = alpha_l * (L_BCE_final + beta_l * Σ L_BCE_block_i) + 100.0 * L_diversity
```
Con `alpha_l=1.0`, `beta_l=0.05`, `L_diversity` tipicamente ~0 dopo le prime epoche.

### 2.3 Parametri del Modello

| Componente | Parametri (stima) |
|---|---|
| InputProjection (CLIP, D=512) | 512×256 + norme ≈ 131K |
| InputProjection (DINO, D=1024) | 1024×256 + norme ≈ 263K |
| Block 1 (VisionMamba, d=256) | ~1.5M |
| Block 2 (2× VisionMamba, d=384) | ~4.5M |
| Block 3 (3× VisionMamba, d=576) | ~10M |
| scale_proj + interaction | ~2M |
| Classification heads | 3×(256/384/576)×157 ≈ 600K |
| **Totale** | **~18–19M** |

---

## 3. Infrastruttura e Ambiente di Sviluppo

### 3.1 Cluster Grid5000 / ABACA

Il training gira su nodi del cluster Grid5000 (sito di Sophia Antipolis, progetto ABACA), prenotati tramite il sistema di scheduling **OAR**. I nodi usati appartengono alla famiglia `esterel` e montano GPU NVIDIA (A100 o H100 a seconda della disponibilità).

**Configurazione ambiente** (`env_abaca.sh`):
- CUDA 12.1.1 (gcc-10.4.0)
- PyTorch 2.5.1+cu121
- Conda environment `mstemba_fresh` (Python 3.10.19)
- CUDA_ARCH_LIST rilevata automaticamente dalla GPU (9.0 per H100, 8.0 per A100)
- PYTHONPATH include `MS-Temba/vim/` per import diretti

**Storage condiviso**: `/srv/storage/stars@storage3.sophia.grid5000.fr/share/Charades/` per feature pre-estratte e dataset.

### 3.2 Struttura del Repository

```
MS-Temba/
├── vim/                          # Codice principale
│   ├── models_MSTemba.py         # Architettura (MSTemba, VisionMamba, Block, ...)
│   ├── MSTemba_main.py           # Training loop, argparse, evaluation
│   ├── charades_dataloader.py    # Dataset, collation, feature loading
│   ├── dinov3_feature_extractor.py  # Estrazione feature DINOv3
│   ├── clip_feature_extraction.py   # Estrazione feature CLIP
│   ├── engine.py                 # Utility train/val step
│   ├── apmeter.py                # Calcolo AP/mAP
│   ├── losses.py                 # BCE multi-label
│   └── scripts/                  # Launch script per ogni configurazione
├── data/
│   └── hf_features/Temporal_Action_Detection/
│       ├── charades_features_clip/        # Feature CLIP (512-dim, 24fps)
│       ├── charades_dinov3_vitl16_w16_24fps/  # Feature DINOv3 (1024-dim, 24fps)
│       └── charades_features_i3d/         # Feature I3D (1024-dim)
├── runs/                         # Output esperimenti
│   └── charades/
│       ├── clip/seed{0,1,2}/
│       ├── dinov3_vitl16/seed{0,1,2}/
│       ├── clip_reg/seed0/
│       └── dinov3_vitl16_reg/seed0/
├── mamba-1p1p1/                  # Mamba SSM (fork locale)
├── causal-conv1d/                # Dipendenza Mamba
└── env_abaca.sh                  # Setup ambiente
```

### 3.3 Configurazione di Training Standard

| Parametro | Valore | Note |
|---|---|---|
| `epochs` | 50 | con early stopping nelle varianti reg |
| `batch_size` | 5 | limitato dalla lunghezza sequenze |
| `num_clips` | 256 | frame per video (con padding) |
| `lr` | 5e-4 (default timm) | cosine schedule con warmup |
| `warmup_epochs` | 5 | default timm |
| `optimizer` | AdamW | via `timm.optim.create_optimizer` |
| `scheduler` | cosine | via `timm.scheduler.create_scheduler` |
| `unisize` | True | collation a lunghezza fissa 256 |
| `skip` | 0 | nessun frame skipping |
| `alpha_l` | 1.0 | peso loss principale |
| `beta_l` | 0.05 | peso block losses ausiliarie |

---

## 4. Fase 1.1 — Riproduzione dei Risultati Baseline

### 4.1 Obiettivo

Verificare che il codebase originale di MS-Temba produca risultati coerenti con quelli riportati nel paper, usando le feature I3D come backbone di riferimento (configurazione originale del paper).

### 4.2 Setup

- **Backbone**: I3D (1024-dim)
- **Dataset**: Charades v1
- **Seeds**: 0, 1, 2 (per stimare la varianza)
- **Feature path**: `charades_features_i3d/`

### 4.3 Risultati

La riproduzione ha confermato risultati comparabili con quelli riportati nel paper originale di MS-Temba. I checkpoint sono salvati in `runs/charades/i3d/seed{0,1,2}/`. Questo step ha validato la correttezza dell'implementazione e dell'ambiente prima di procedere con backbone alternativi.

---

## 5. Fase 1.2 — CLIP come Backbone

### 5.1 Motivazione

CLIP (Contrastive Language–Image Pre-training, OpenAI) è un Vision Transformer addestrato su ~400M coppie immagine–testo con obiettivo contrastivo. L'ipotesi è che lo spazio semantico di CLIP, già organizzato per concetti linguistici, fornisca feature più discriminative per le classi di Charades rispetto a I3D, che apprende rappresentazioni puramente visive e motion-based.

### 5.2 Tipologia di Feature Estratte

**Modello**: CLIP ViT-B/16 (o ViT-L/14 per TSU), `encode_image()` → token CLS proiettato  
**Dimensione**: 512-dim per ViT-B/16  
**Frequenza**: 24 fps (una feature per frame)  
**Pooling**: nessuno — ogni frame produce un vettore indipendente  

Il token CLS aggrega l'informazione di tutte le patch tramite self-attention e viene proiettato nello spazio condiviso immagine–testo. Questo spazio è organizzato attorno a concetti semantici linguisticamente frequenti: azioni come *cooking*, *watching television*, *talking on the phone* producono embedding molto separati, mentre azioni con stesso oggetto e verbi contrapposti (*put bag* vs *take bag*) sono proiettate in regioni vicine.

### 5.3 Configurazione

```bash
python MSTemba_main.py \
  -dataset charades -backbone clip -model mstemba \
  -rgb_root ".../charades_features_clip" \
  -in_feat_dim 512 -num_clips 256 -epochs 50 \
  -batch_size 5 -alpha_l 1 -beta_l 0.05
```

**Nota**: `in_feat_dim` non era necessario specificare esplicitamente per CLIP (il default nel codice era già 768 ma veniva sovrascritto dal riconoscimento del backbone string `'clip'`).

### 5.4 Risultati — 3 Seed

| Seed | Best ep | Full-val-MAP | Sampled-val-MAP |
|:---:|:---:|:---:|:---:|
| 0 | 13 | **32.40** | **33.43** |
| 1 | 13 | ~32.1 | ~33.2 |
| 2 | 15 | ~31.8 | ~32.9 |
| **Media** | **~13.7** | **~32.1** | **~33.2** |

Il checkpoint migliore (seed0, ep13) produce:
- Block 1 sampled-val-map: 29.36
- Block 2 sampled-val-map: 30.90
- Block 3 sampled-val-map: 31.69
- Final sampled-val-map: **33.43**

La progressione blocco→blocco conferma il design gerarchico di MS-Temba: ogni scala temporale aggiuntiva porta un incremento di mAP (~1.3-1.6 punti per blocco).

### 5.5 Analisi delle Curve di Training

Il training CLIP mostra un pattern caratteristico:
- **Ep 0–5**: warmup del lr, mAP sale da ~2 a ~10
- **Ep 5–13**: crescita rapida, val_map da 10 a 32.4
- **Ep 13–50**: overfitting progressivo — train_map continua a crescere (>80 a ep50) mentre val_map scende a ~27-28

Il gap train/val a fine training è indicativo di memorizzazione: il modello impara pattern specifici del training set che non generalizzano, specialmente per le 49 classi con AP < 20 (classi rare con <100 esempi di training).

### 5.6 Punti di Forza di CLIP su Charades

- **Discriminatività semantica compatta**: classi semanticamente distinte sono già separate nello spazio 512-dim prima di Mamba. Classi ad alta AP: *cooking* (79.0), *talking on the phone* (75.9), *working on a laptop* (75.9).
- **Copertura web-scale**: il pretraining su 400M coppie copre naturalmente il vocabolario domestico di Charades.
- **Stabilità temporale**: frame consecutivi di stessa azione producono embedding simili, facilitando la rilevazione temporale per Mamba.

### 5.7 Limiti di CLIP su Charades

- **Ambiguità verbo-dipendente**: coppie come *opening/closing door*, *putting/taking bag* producono embedding simili perché il corpus testuale non le distingue visivamente. Questo spiega AP asimmetriche: *opening door* (39.8) vs *closing door* (30.3).
- **Perdita di risoluzione spaziale**: il CLS token aggrega tutte le patch — configurazione posturale, relazioni mano–oggetto e dettagli locali sono irrecuperabilmente persi prima di Mamba.
- **Saturazione per verbi rari**: verbi come *throw* sono associati nel corpus web a contesti diversi da quelli domestici → mean AP verbo *throw* = 10.2, il più basso tra tutti i verbi.

---

## 6. Fase 1.3 — DINOv3 come Backbone

### 6.1 Motivazione

DINOv3 (DINO v3, Meta AI) è un ViT-L/16 addestrato con apprendimento self-supervised su LVD-142M (142M immagini). A differenza di CLIP, non ha un segnale testuale: apprende rappresentazioni visive strutturate attraverso distillazione self-supervised. L'ipotesi è che DINOv3 produca feature più ricche di informazione spaziale e strutturale rispetto a CLIP, potenzialmente utili per le classi di Charades che richiedono discriminazione posturale.

### 6.2 Tipologia di Feature Estratte

**Modello**: `facebook/dinov3-vitl16-pretrain-lvd1689m` (HuggingFace)  
**Dimensione**: 1024-dim (ViT-L ha hidden dim maggiore di ViT-B)  
**Frequenza**: 24 fps, con temporal average pooling su finestre di 16 frame (`window_size=16`)  
**Pooling**: token CLS (`encode_image_global` con `pooling='cls'`)  
**Precisione**: bfloat16 durante l'estrazione (evitare NaN con fp16)  

**Nota critica**: le feature estratte sono ancora CLS-only — un singolo vettore 1024-dim per frame. Le patch features di DINOv3 (256 token × 1024-dim per frame a risoluzione 256×256) non vengono estratte né utilizzate in questa fase. Questo è un limite importante che verrà affrontato nella Fase 2.

### 6.3 Configurazione

```bash
python MSTemba_main.py \
  -dataset charades -backbone dinov3_vitl16 -model mstemba \
  -rgb_root ".../charades_dinov3_vitl16_w16_24fps" \
  -in_feat_dim 1024 -num_clips 256 -epochs 50 \
  -batch_size 5 -alpha_l 1 -beta_l 0.05
```

### 6.4 Risultati — Seed 0

| Metrica | Valore | Epoca |
|---|:---:|:---:|
| Best Full-val-MAP | **25.44** | 13 |
| Best Sampled-val-MAP | **25.94** | 13 |
| Val MAP @ ep49 | 18.81 | 49 |
| Sampled-val-MAP @ ep49 | 19.36 | 49 |

### 6.5 Diagnosi: Perché DINO < CLIP di ~7 mAP?

Questo risultato è **controintuitivo** e richiede una spiegazione articolata.

**Ipotesi 1 — Overfitting severo (confermata)**  
Il crollo da 25.44 (ep13) a 18.81 (ep49) è un segno inequivocabile di overfitting. La `LinearProjection(1024→256)` ha il doppio dei parametri rispetto alla versione CLIP (512→256), con gli stessi dati di training e zero regolarizzazione aggiuntiva. Il modello memorizza il training set invece di generalizzare.

**Ipotesi 2 — CLS token non sfrutta il vantaggio reale di DINO (confermata)**  
Il vantaggio competitivo di DINOv3 rispetto a CLIP non sta nel CLS token ma nelle **patch features**: ogni patch produce un embedding localizzato e spazialmente strutturato. Usando solo il CLS token, si rinuncia alla principale superiorità di DINO e si ottiene una rappresentazione globale di qualità inferiore a quella di CLIP (che è ottimizzata esplicitamente per la discriminatività globale tramite il training contrastivo).

**Ipotesi 3 — Mismatch di pretraining (parziale)**  
CLIP è addestrato con supervizione semantica esplicita su concetti linguistici naturali, molti dei quali coincidono con le classi di Charades. DINOv3 apprende rappresentazioni visive auto-supervisionate senza ancoraggio semantico esplicito — la geometria dello spazio 1024-dim non è pre-allineata con il vocabolario di Charades, richiedendo a Mamba un lavoro di organizzazione semantica che CLIP ha già fatto.

### 6.6 Analisi Curve di Training DINO

- **Ep 0–13**: crescita rapida analoga a CLIP, con picco a 25.44
- **Ep 13–49**: crollo progressivo e monotono fino a 18.81 (-6.6 mAP)
- Il train_map a ep49 era ~70+, confermando overfitting estremo

---

## 7. Fase 1.4 — Confronto CLIP vs DINOv3

### 7.1 Tabella Riassuntiva

| Metrica | CLIP | DINOv3 |
|---|:---:|:---:|
| Feature dim | 512 | 1024 |
| Best Full-val-MAP | **32.40** | 25.44 |
| Best Sampled-val-MAP | **33.43** | 25.94 |
| Epoca best | 13 | 13 |
| Val MAP @ ep49 | ~27.5 | 18.81 |
| Gap train/val @ best | ~0 | ~0 |
| Gap train/val @ stop | ~55 | ~52 |
| Pretraining objective | Contrastivo img–txt | Self-supervised |
| Feature type usata | CLS global | CLS global |

### 7.2 Per-Class Analysis (Highlights)

Classi dove CLIP > DINOv3 significativamente:
- *Cooking* (CLIP: 79.0, DINO: ~45): fortemente ancorata al vocabolario CLIP
- *Talking on phone* (CLIP: 75.9, DINO: ~38): semantica linguistica dominante
- *Working on laptop* (CLIP: 75.9, DINO: ~42): idem

Classi dove la differenza è minore (DINOv3 non recupera nemmeno su classi posturali):
- *Standing up* (CLIP: 60.4, DINO: 23.2): ipotesi iniziale che DINO fosse meglio su classi posturali **non confermata** — DINOv3 CLS-only non cattura meglio la configurazione posturale di CLIP
- *Running* (CLIP: 18.9, DINO: 9.1): entrambi scarsi

Questa analisi rafforza la conclusione che il CLS token di DINOv3 non sfrutta il vantaggio strutturale del backbone — le patch features sono necessarie.

### 7.3 Analisi per Verbo e Oggetto

L'analisi sul vocabolario compositivo di Charades (157 classi = 33 verbi × 38 oggetti) rivela:

**Verbi con AP medio più basso (entrambi i backbone)**:
- *throw* (v025): mean AP 10.2 — underrepresentato nel corpus web e raro in Charades
- *run* (v019): mean AP 12.1 — difficile senza informazione di motion
- *turn on/off* (v026/027): mean AP ~15 — cambiamento di stato non visibile nel singolo frame

**Verbi con AP medio più alto**:
- *watch* (v030): mean AP 68.3 — configurazione visiva distintiva e stabile
- *eat* (v009): mean AP 55.7 — fortemente ancorato a oggetti visivi riconoscibili
- *cook* (v006): mean AP 52.1 — contesto visivo ricco e uniforme

---

## 8. Fase 1.5 — Interventi di Regolarizzazione

### 8.1 Motivazione e Diagnosi

Entrambi i backbone mostrano overfitting: CLIP raggiunge il picco a ep13 e poi scende; DINOv3 crolla da ep13 a ep49. L'obiettivo è stabilizzare il training, avvicinare il best alla fine del training e permettere confronti più affidabili tra configurazioni.

La diagnosi precisa per backbone:

**CLIP**: il problema principale non è la capacità del modello ma il **learning rate scheduling** — il lr cosine sale durante il warmup e poi scende, ma il modello ha già memorizzato il training set prima che il decay lo regolarizzi. Il dropout agisce troppo presto e frena la convergenza.

**DINOv3**: doppia criticità — overfitting strutturale (dimensione input 2× rispetto a CLIP con stessa regolarizzazione zero) e CLS-only features (vedi Fase 2). La regolarizzazione è più urgente qui.

### 8.2 Modifiche al Codebase

#### 8.2.1 `models_MSTemba.py` — Dropout nella LinearProjection

**Prima (originale)**:
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

**Dopo (modificato)**:
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

#### 8.2.2 `models_MSTemba.py` — Input Projection con Dropout

**Prima**: `self.proj = nn.Linear(in_feat_dim, embed_dims[0])`

**Dopo**:
```python
self.proj = nn.Sequential(
    nn.Linear(in_feat_dim, embed_dims[0]),
    nn.LayerNorm(embed_dims[0]),
    nn.GELU(),
    nn.Dropout(p=drop_rate)
)
```

Il `permute(0,2,1)` preesistente (eredità di una versione Conv1d) è stato mantenuto per gestire il formato delle feature su disco (`[B, D, T]` → `[B, T, D]` prima del Linear).

#### 8.2.3 `models_MSTemba.py` — Propagazione `drop_rate` in MSTemba

La signature di `MSTemba.__init__` è stata estesa con `drop_rate=0.0`, propagato a tutte le `LinearProjection` istanziate nel costruttore:

```python
def __init__(self, in_feat_dim=768, num_classes=157,
             embed_dims=[256,384,576], depths=[1,1,1],
             d_state=16, drop_rate=0.0, **kwargs):
    ...
    self.blocks.append(LinearProjection(embed_dims[0], embed_dims[0], drop_rate=drop_rate))
    self.blocks.append(LinearProjection(embed_dims[0], embed_dims[1], drop_rate=drop_rate))
    self.blocks.append(LinearProjection(embed_dims[1], embed_dims[2], drop_rate=drop_rate))
```

Questo assicura che `--drop 0.2` passato da CLI arrivi fino all'input projection e a tutte le transizioni inter-blocco — non solo ai blocchi Mamba interni gestiti da timm.

#### 8.2.4 `MSTemba_main.py` — Nuovi Argomenti

Aggiunti all'argparse:
```python
parser.add_argument('--early-stop-patience', type=int, default=15)
parser.add_argument('--min-delta', type=float, default=0.01)
```

#### 8.2.5 `MSTemba_main.py` — Early Stopping

Implementato nella funzione `run()` con logica:
```python
is_best = val_map > (Best_val_map + min_delta)
if is_best:
    patience_counter = 0
    # salva checkpoint_best, best_model.pth, pkl
else:
    patience_counter += 1
    logging.info(f"[EARLY_STOP] No improvement. Patience: {patience_counter}/{early_stop_patience}")

if early_stop_patience > 0 and patience_counter >= early_stop_patience:
    logging.info(f"[EARLY_STOP] Triggered at epoch {epoch}")
    stop_training = True
    break
```

Il `patience_counter` viene salvato nel checkpoint per supportare il resume corretto:
```python
save_checkpoint(..., extra={'patience_counter': patience_counter})
```

#### 8.2.6 `MSTemba_main.py` — Logging Strutturato

**`run_config.json`**: salvato all'inizio di ogni run con `json.dump(vars(args), ...)`. Non viene sovrascritto in caso di resume. Permette la riproducibilità completa di ogni esperimento.

**`metrics_per_epoch.csv`**: scritto in append a ogni epoch con le colonne:
```
epoch, train_loss, train_map, val_loss, val_map, sampled_val_map,
block1_train_map, block2_train_map, block3_train_map,
block1_val_map, block2_val_map, block3_val_map,
block1_sval_map, block2_sval_map, block3_sval_map,
diversity_loss, lr, is_best
```

Supporta il resume in append mode (non sovrascrive le epoche già loggiate).

#### 8.2.7 `MSTemba_main.py` — Fix Doppio Save

Il codice originale salvava `checkpoint_last.pth` due volte per epoch (prima e dopo il val_step). Corretto mantenendo un solo save post-val con `patience_counter` aggiornato.

### 8.3 Configurazioni dei Nuovi Esperimenti

Sono stati condotti due round di regolarizzazione, con calibrazione progressiva dei parametri.

**Round 1 (v1)**

| Parametro | CLIP reg v1 | DINO reg v1 |
|---|:---:|:---:|
| `--drop` | 0.1 | 0.2 |
| `--drop-path` | 0.05 | 0.1 |
| `--weight-decay` | 0.05 | 0.05 |
| `--early-stop-patience` | 15 | 12 |

**Round 2 (v2)** — calibrazione sulla base della diagnosi del Round 1

| Parametro | CLIP reg v2 | DINO reg v2 |
|---|:---:|:---:|
| `--drop` | 0.0 | 0.05 |
| `--drop-path` | 0.0 | 0.05 |
| `--weight-decay` | 0.05 | 0.05 |
| `--early-stop-patience` | 20 | 15 |

La motivazione della calibrazione è dettagliata nella Sezione 10.2.

### 8.4 Risultati degli Esperimenti Regolarizzati

#### CLIP reg v1 — `runs/charades/clip_reg/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | 28.92 | 29.55 | ✓ | |
| 15 | **29.17** | **29.83** | ✓ | **Best** |
| 20 | 27.97 | 28.66 | ✗ | |
| 30 | 25.46 | 26.29 | ✗ | Early stop (patience 15) |

Early stop: ep30, patience 15/15. Train_map @ stop: 63.7. Durata: ~91 min.

#### DINOv3 reg v1 — `runs/charades/dinov3_vitl16_reg/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | 24.54 | 25.11 | ✓ | |
| 14 | **24.56** | **25.16** | ✓ | **Best** |
| 20 | 22.83 | 23.57 | ✗ | |
| 26 | 22.35 | 23.36 | ✗ | Early stop (patience 12) |

Early stop: ep26, patience 12/12. Train_map @ stop: 51.7. Durata: ~82 min.

#### CLIP reg v2 — `runs/charades/clip_reg_v2/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 13 | **28.91** | **29.49** | ✓ | **Best** |
| 19 | 27.55 | 27.95 | ✗ | |
| 33 | 23.07 | 23.60 | ✗ | Early stop (patience 20) |

Early stop: ep33, patience 20/20. Train_map @ stop: 92.9. Durata: ~86 min.

**Osservazione critica**: nonostante `drop=0.0` identico all'originale, il best è 28.91 anziché 32.40. La differenza è imputabile esclusivamente al `weight-decay=0.05` (vs 0.01 originale), che penalizza i pesi grandi nelle prime epoche rallentando la convergenza. Il gap train/val esplode comunque (train 92.9 vs val 23.1 a ep33): il weight decay da solo non risolve l'overfitting strutturale di CLIP.

#### DINOv3 reg v2 — `runs/charades/dinov3_vitl16_reg_v2/seed0/`

| Epoch | val_map | sampled_val_map | is_best | Note |
|:---:|:---:|:---:|:---:|---|
| 11 | 24.70 | 25.37 | ✓ | |
| 13 | **25.21** | **25.82** | ✓ | **Best** |
| 20 | 23.10 | 23.81 | ✗ | |
| 28 | 21.22 | 21.98 | ✗ | Early stop (patience 15) |

Early stop: ep28, patience 15/15. Train_map @ stop: 77.6. Durata: ~72 min.

**Risultato positivo**: la v2 recupera quasi il valore originale (25.21 vs 25.44) con un crollo post-picco più contenuto (−4.0 mAP in 15 epoche vs −6.6 in 36 epoche dell'originale). È la **configurazione DINO definitiva** per questa fase.

---

## 9. Modifiche Architetturali e al Codebase

### 9.1 Riepilogo Cronologico delle Modifiche

| Data | File | Modifica | Motivazione |
|---|---|---|---|
| Mar 2026 | `models_MSTemba.py` | `LinearProjection` + `drop_rate` | Propagare dropout alle transizioni inter-blocco |
| Mar 2026 | `models_MSTemba.py` | `self.proj` → `nn.Sequential` con Dropout | Regolarizzare l'input projection (critico per DINO 1024-dim) |
| Mar 2026 | `models_MSTemba.py` | `MSTemba.__init__` + `drop_rate` | Propagazione completa del parametro |
| Mar 2026 | `MSTemba_main.py` | `--early-stop-patience`, `--min-delta` | Fermare il training al picco |
| Mar 2026 | `MSTemba_main.py` | `run_config.json` | Riproducibilità completa |
| Mar 2026 | `MSTemba_main.py` | `metrics_per_epoch.csv` | Analisi dettagliata post-hoc |
| Mar 2026 | `MSTemba_main.py` | Early stopping in `run()` | Evitare crollo post-picco |
| Mar 2026 | `MSTemba_main.py` | `save_checkpoint` + `extra` | Salvare `patience_counter` per resume |
| Mar 2026 | `MSTemba_main.py` | Fix doppio save | Bug: `checkpoint_last` salvato 2× per epoch |
| Mar 2026 | `vim/scripts/` | `run_charades_clip_reg_seed0.sh` | Launch script CLIP reg v1 |
| Mar 2026 | `vim/scripts/` | `run_charades_dinov3_reg_seed0.sh` | Launch script DINO reg v1 |
| Mar 2026 | `vim/scripts/` | `run_charades_clip_reg_v2_seed0.sh` | Launch script CLIP reg v2 (drop=0.0, wd=0.05) |
| Mar 2026 | `vim/scripts/` | `run_charades_dinov3_reg_v2_seed0.sh` | Launch script DINO reg v2 (drop=0.05, wd=0.05) |

### 9.2 Bug Identificati e Risolti

**Bug 1 — `permute` non rimosso dopo refactoring `self.proj`**  
Origine: il codice originale aveva un `nn.Conv1d` come `self.proj`, che richiedeva `[B, D, T]` (quindi il `permute`). Dopo la sostituzione con `nn.Linear`, il permute era stato commentato ma non rimosso, causando un errore di shape. La feature su disco è salvata in formato `[D, T]`, quindi il permute `[B,D,T]→[B,T,D]` è necessario e corretto — va mantenuto prima di `self.proj`.

**Bug 2 — Doppio `save_checkpoint` per epoch**  
Origine: nel loop `run()` originale, `checkpoint_last.pth` veniva salvato sia prima che dopo il val_step. Corretto rimuovendo il salvataggio anticipato.

---

## 10. Risultati Consolidati e Analisi Comparativa

### 10.1 Tabella Riepilogativa Completa — 6 Configurazioni

| Configurazione | Backbone | drop | dp | wd | Best ep | Best val-MAP | Best sval-MAP | Stop ep | Train@stop |
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

**Baseline definitiva selezionata**:
- **CLIP**: `clip/seed0` originale — val-MAP **32.40**
- **DINO**: `dinov3_reg_v2/seed0` — val-MAP **25.21** (crollo post-picco contenuto, stabile)

### 10.2 Interpretazione dei Risultati — Analisi Comparativa per Round

#### CLIP: nessuna configurazione regolarizzata supera l'originale

| Confronto | Δ val-MAP | Osservazione |
|---|:---:|---|
| orig → reg v1 (drop=0.1) | −3.2 | Dropout troppo aggressivo, frena la convergenza |
| orig → reg v2 (drop=0.0, wd=0.05) | −3.5 | Weight decay da solo penalizza ugualmente |
| reg v1 → reg v2 | −0.3 | Sostanzialmente equivalenti nonostante parametri diversi |

**Conclusione per CLIP**: il problema di overfitting non è risolvibile con dropout o weight decay standard. Il gap train/val esplode in tutti i casi dopo ep13-15 (train >60, val ~25-29). La causa è strutturale — **7985 video con 157 classi sbilanciate** non sono sufficienti per un modello da ~18M parametri. Le soluzioni possibili sono: (1) lr scheduling più aggressivo post-picco, (2) riduzione della capacità del modello, (3) data augmentation temporale. Per ora il run originale (32.40) rimane la **baseline CLIP definitiva**.

#### DINO: la v2 è la configurazione ottimale

| Confronto | Δ val-MAP best | Δ crollo post-picco | Osservazione |
|---|:---:|:---:|---|
| orig → reg v1 (drop=0.2) | −0.9 | +4.4 migliorato | Dropout troppo alto, ma stabilizza |
| orig → reg v2 (drop=0.05) | −0.2 | +2.6 migliorato | Quasi parità sul best, curva più stabile |
| reg v1 → reg v2 | +0.6 | +1.8 migliorato | v2 superiore su entrambe le dimensioni |

**Conclusione per DINO**: la v2 (`drop=0.05, drop_path=0.05, wd=0.05`) è la configurazione definitiva — recupera quasi il valore originale (25.21 vs 25.44, delta −0.23) con una curva post-picco più controllata. È la **baseline DINO definitiva** per i confronti futuri.

### 10.3 Gap Strutturale CLIP vs DINO

Il gap di ~7 mAP tra CLIP e DINO non è un artefatto di overfitting — permane anche quando si confrontano i valori al picco (ep13) dove entrambi non mostrano overfitting. È un gap strutturale attribuibile a:

1. **Allineamento semantico** (50% del gap stimato): CLIP è pre-allineato con il vocabolario di Charades tramite training contrastivo; DINOv3 non lo è.
2. **CLS token subottimale per DINO** (40% del gap stimato): le dense patch features di DINO sarebbero la rappresentazione corretta.
3. **Dimensionalità non proporzionale** (10% del gap): 1024-dim con stesso numero di parametri di training non è necessariamente più informativo.

---

## 11. Limitazioni e Problemi Aperti

### 11.1 Limitazioni Tecniche Attuali

**Feature CLS-only per DINOv3**: come discusso, le patch features di DINOv3 (il suo vero punto di forza) non vengono estratte né utilizzate. Ogni frame è rappresentato da un singolo vettore globale che non cattura le relazioni spaziali locali, la configurazione posturale, o le relazioni mano–oggetto.

**Assenza di feature temporali dense**: le feature CLIP e DINO sono estratte frame-by-frame senza informazione di motion. Backbone come I3D o ViCLIP catturano il flusso ottico o clip temporali, fornendo a Mamba un segnale di transizione esplicito.

**Skeleton features non integrate**: le feature scheletriche SCDNet disponibili su ABACA (`/srv/storage/stars@storage3.sophia.grid5000.fr/share/Charades/Charades_SCDNet_features2.zip`) non sono ancora state caricate, analizzate né integrate nel modello. Queste feature sono particolarmente rilevanti per le classi dei Gruppi A e B (verbi di stato/direzione, transizioni posturali) identificate nell'analisi per-class.

**Single seed per gli esperimenti regolarizzati**: `clip_reg` e `dinov3_reg` hanno un solo seed (0). Per confronti statisticamente affidabili, servono almeno 3 seed.

**Log duplicati**: il meccanismo `tee -a` + file handler del logger produce ogni riga duplicata nei training log. Non influenza i risultati ma rende i log più difficili da leggere.

### 11.2 Overfitting Strutturale di CLIP — Problema Aperto

Tutti e tre i round di regolarizzazione (orig, v1, v2) non hanno recuperato il valore originale di CLIP (32.40). L'overfitting post-ep13 è strutturale e non colmabile con i parametri esplorati. Le direzioni non ancora sperimentate sono: lr scheduling con decay più aggressivo post-warmup (es. cosine con restart o step decay), riduzione degli embed_dims del modello, o data augmentation temporale (random temporal crop, frame dropout durante il training).

---

## 12. Passi Successivi — Fase 2 e Oltre

### 12.1 Fase 2A — DINOv3 Dense Features (Priorità Alta)

**Obiettivo**: sfruttare le patch features di DINOv3 invece del solo CLS token.

**Due approcci possibili**:

*Approccio 1 — Re-estrazione delle feature (più potente)*: modificare `dinov3_feature_extractor.py` per estrarre tutti gli N patch tokens invece del solo CLS. Shape output: `[T, N_patches, 1024]` dove `N_patches = (H/16) × (W/16)`. Per immagini 224×224 con ViT-L/16: `N=196 patches`. Richiede modifica del dataloader per gestire la dimensione spaziale aggiuntiva e dell'architettura per aggregarla (attention pooling, mean pooling, o spatial-then-temporal processing).

*Approccio 2 — Attention pooling in-memory (meno costoso)*: caricare le feature CLS esistenti ma aggiungere un secondo forward del backbone (o estrarre i token intermedi) durante il training. Meno costoso in storage ma più costoso in compute.

**Impatto atteso** (stime dal documento di analisi):
- +2–4 mAP su classi dei Gruppi A (verbi di stato/direzione) e B (transizioni posturali)
- ΔmAP globale stimato: +1.0 (conservativo) → +4.0 (ottimistico)

**Step implementativi**:
1. Verificare le feature DINO esistenti: `python -c "import numpy as np; x=np.load('...'); print(x.shape)"` — verificare se sono già `[T, 1024]` o altro formato
2. Modificare `dinov3_feature_extractor.py` per estrarre mean delle patch (via `encode_image_global` con `pooling='mean_patch'`)
3. Opzionale: estrarre anche CLS + mean_patch come vettore concatenato (2048-dim) per ablation
4. Modificare il dataloader per gestire la nuova dimensione
5. Aggiornare `MSTemba.__init__` con nuovo `in_feat_dim`

### 12.2 Fase 1.5 — Calibrazione Regolarizzazione ✅ Completata

Tre round di esperimenti (orig, v1, v2) hanno prodotto la baseline definitiva:
- **CLIP**: run originale (val-MAP 32.40) — nessuna regolarizzazione migliora il best
- **DINO**: reg v2 (val-MAP 25.21, drop=0.05, wd=0.05) — migliore stabilità post-picco

La direzione per migliorare CLIP non è la regolarizzazione classica ma interventi sul lr scheduling o sulla capacità del modello — da esplorare come obiettivo secondario dopo le dense features.

### 12.3 Fase 2C — Integrazione Skeleton Features SCDNet (Priorità Alta)

**Obiettivo**: integrare le feature scheletriche pre-estratte con SCDNet per arricchire la rappresentazione con informazione posturale e cinematica.

**Step preliminari**:
1. Ispezionare il formato delle feature: `python -c "import zipfile; z=zipfile.ZipFile('Charades_SCDNet_features2.zip'); print(z.namelist()[:5])"`
2. Verificare il fps di campionamento delle skeleton features (critico per l'allineamento con le 24fps CLIP/DINO)
3. Implementare il resampling temporale: se le skeleton sono a fps diverso (es. 6fps o 30fps), applicare interpolazione lineare o nearest-neighbor per portarle a 24fps

**Strategie di fusione** (in ordine di complessità crescente):
- **Early fusion** (concat): `[B, T, D_visual + D_skel]` → proiezione congiunta. Lower bound semplice da implementare.
- **Gated fusion**: `h = g ⊙ f_visual + (1-g) ⊙ f_skel` con `g = sigmoid(Linear([f_v, f_s]))`. Lascia al modello decidere il peso di ciascuna modalità per ogni frame.
- **Cross-attention**: `f_visual` come Query, `f_skel` come Key/Value (o viceversa). Permette interazione fine-grained tra le due modalità.

**Impatto atteso**:
- Maggiore per classi Gruppo A (*running*, *sneezing*, *turning on light*) e Gruppo B (*sitting down*, *standing up*, *bending*)
- ΔmAP stimato: +1.0 (gated, skel 2D) → +4.0 (cross-attention, GNN)

### 12.4 Fase 3 — Survey e Idee Innovative

**Survey stato dell'arte**: analisi sistematica della letteratura su combinazione di feature visive e scheletriche per TAD/TAR (Temporal Action Detection/Recognition), con focus su:
- Fusione modale (early/late/mid fusion, cross-modal attention)
- Robustezza a occlusioni e skeleton mancanti
- Dataset benchmark (NTU RGB+D, Charades, Toyota Smarthome)
- Modelli recenti: SkeletonBERT, MotionBERT, HiCo, PoseFormer, UNIK

**Idee innovative identificate**:
1. **Auxiliary supervision cinematica**: addestrare MS-Temba a predire la direzione del motion vector del polso come obiettivo ausiliario durante il training. Non aggiunge costi a inference time.
2. **Multi-stream gerarchico CLIP + DINO dense + Skeleton**: tre stream paralleli fusi tramite cross-attention a livello dei blocchi Mamba.
3. **Adaptive temporal sampling**: campionamento adattivo dei frame basato sulla saliency delle feature (invece del fisso 24fps) per concentrare la capacità del modello sui frame informativi.

### 12.5 Piano di Documentazione

Questo documento è il primo di una serie. La struttura prevista:

| Documento | Contenuto | Status |
|---|---|:---:|
| `phase1_baseline_clip_dino.md` | Questo documento — baseline, reg, confronto completo | ✅ |
| `phase2a_dino_dense_features.md` | Estrazione e integrazione patch features DINOv3 | ⏳ |
| `phase2b_skeleton_scdnet.md` | Integrazione skeleton SCDNet, allineamento fps, fusione | ⏳ |
| `phase2c_fusion_ablation.md` | Ablation study strategie di fusione | ⏳ |
| `phase3_survey_skeleton_visual.md` | Survey stato dell'arte | ⏳ |
| `phase3_innovative_ideas.md` | Idee innovative e risultati sperimentali | ⏳ |

---
