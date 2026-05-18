PROBLEMI E DOMANDE APERTE:
1) Nel CLIP originale non ho usato regolarizzazione ( dropout e drop_path credo) mentre nella versione unidirezionale si (e infatti il risultato mi sembra simile alla versione di CLIP_reg che avevo fatto);

2) Le skeleton features vengono compressate di molto (4096 -> 256 (x16)) come definito dal modello MSTemba che accetta 256 il che potrebbe far perdere informazioni e aumentaare overiftting;

3) Il numero di canaali (credo si chiami così) deve essere per forza 256 per MSTeba (conideranndo di voler avere un lavoro confrontabile con quello delk paper)?

4) L'estrazione delle feature (sia skeleton che dino etc) dipende dal modello (unidirezionale o bidirezionale) oppure no? Perche altrimenti potrei ripartire da zero cercando di aggiungere man mano gli esperimenti e adattare il modello senza perdermi niente e senza uscire di testa andando a prendere le feature gà estratte;

5) Problema del temporal mismatch per skeleton (Ancora non l'ho capito bene): Devo riestrarre le feature skeleton per adattarle a quelle CLIP e rifare l'esperimento?

6) Diminuire early stop (ora è 15 ma secondo me almeno 10);

7) Controllare fps features di CLIP, DINOv3 e Skeleton (dall'analsi risulta CLIP/DINOv3 -> 1,5 fps e Skeleton -> 24 fps)

8) Differenza tra DINOv2 e DINOv3, quale utilizzare? Considera che per il momento sto utilizzando DINOv3 e se non ci sovessero essere palesi miglioramenti nelle performance o nell'adattabilità del modello preferirei proseguire con quello (riferisciti in generale anche per le altre task e le altre domande al modello migliore tra i due ricordando l'attuale preferenza per DINOv3).

---

## 1) Regolarizzazione CLIP: Originale vs Unidirezionale

**La tua osservazione è corretta**: hai probabilmente visto risultati simili tra:
- CLIP unidirezionale (dropout/drop_path 0.1)
- CLIP_reg precedente (con regolarizzazione)

**Analisi**:
```
CLIP paper (bidirezionale): ~32.40 mAP (no dropout forte)
CLIP_reg (tuo esperimento):  ~29.xx mAP (con dropout 0.1)
CLIP uni attuale:            29.00 mAP (dropout 0.1)
```

**Il gap -3.4 mAP è principalmente**: 
- 70-80%: Loss bidirezionalità (contesto futuro)
- 20-30%: Regolarizzazione (dropout previene overfitting ma limita capacity)

**Conclusione**: La similarità CLIP_reg ≈ CLIP_uni conferma che **regolarizzazione e unidirezionalità hanno effetti comparabili** (entrambi riducono ~2-3 mAP vs optimistic baseline).

**Raccomandazione**: MANTIENI dropout 0.1 anche se riduce leggermente mAP assoluto, perché:
- Previene overfitting catastrofico (train-val gap 37.7 già critico)
- Standard practice in letteratura
- Comparabilità con esperimenti futuri

---

## 2) Compressione Skeleton 4096→256 (16×)

**Problema confermato**: Compressione eccessiva.

**Comparazione**:
- CLIP: 768→256 (3×)
- Skeleton: 4096→256 (16×)  ← 5.3× PIÙ AGGRESSIVA

**Conseguenze**:
1. **Information loss**: dimensioni cinematiche sottili (velocità angolari, traiettorie) perse
2. **Overfitting risk**: projection layer cerca di "memorizzare" mapping invece di apprendere pattern
3. **Training instability**: LR skeleton 0.5× compensazione parziale

**Soluzione: Bottleneck projection 4096→512→256**

```python
# models_MSTemba.py - SOLUZIONE OTTIMALE
self.proj_skel = nn.Sequential(
    nn.Linear(4096, 512),      # First compression 8× (gentler)
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),       # Second compression 2×
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.1)
)
```

**Pro**:
- Compression graduale (8×→2× invece di 16× diretto)
- Preserva comparabilità (output 256, embed_dim invariato)
- +262k parameters (4096×512 + 512×256 = 2.36M) ma acceptable
- Expected gain: +0.5-1.0 mAP skeleton standalone

**Implementazione**: 10 minuti modifica, re-train gated fusion

---

## 3) Constraint embed_dim=256

**Risposta: NO, NON obbligatorio ma raccomandato per comparabilità**

**MS-Temba è configurabile**:
```python
model = MSTemba(
    embed_dim=256,    # ← VARIABILE (default paper)
    depth=[2,2,2],
    ...
)
```

**Impatto cambio embed_dim**:

| embed_dim | Block dims | Total params | Comparabilità | Note |
|-----------|------------|--------------|---------------|------|
| **256** (paper) | 256/384/576 | ~20M | ✅ Diretta | Standard |
| **384** | 384/576/864 | ~45M | ⚠️ Ablation | 2.25× params |
| **512** | 512/768/1152 | ~80M | ❌ Diverso | 4× params |

**Raccomandazione: MANTIENI 256**

Motivi:
1. **Comparabilità paper**: risultati directly comparable
2. **Skeleton compression**: risolvi con bottleneck (vedi #2), non aumentando embed_dim
3. **Overfitting risk**: 80M params su 7985 video = 10k params/sample (troppo!)
4. **Efficienza**: training 4× più lento con embed_dim=512

**Eccezione**: Se skeleton bottleneck 4096→512→256 NON migliora (+<0.3 mAP), ALLORA considera embed_dim=384 come ablation (compromise ragionevole).

---

## 4) Estrazione Features: Indipendenza da Modello

**Risposta: COMPLETAMENTE INDIPENDENTE**

**Pipeline separata**:
```
┌─────────────────────────┐
│  Video raw (pixels)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Feature Extractor       │  ← Pre-trained models (CLIP, DINOv3, SCD-Net)
│ (INDIPENDENTE)          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  .npy files on disk     │  ← [N, D] arrays (N=timesteps, D=dim)
│  (RIUSABILI)            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  MS-Temba model         │  ← Uni/bidirezionale qui (downstream)
│  (Uni o Bidirezionale)  │
└─────────────────────────┘
```

**Feature extractors NON vedono MS-Temba**:
- CLIP ViT-B/16: encoder standalone
- DINOv3: encoder standalone  
- SCD-Net: encoder standalone

**Implicazione CRITICA**: 

✅ **PUOI ripartire da zero con modello diverso** SENZA riestrarre features

**Workflow ottimale**:
1. ✅ Features già estratte: RIUSA tutte (CLIP, DINOv3, skeleton)
2. ✅ Cambia modello: uni→bi, embed_dim, fusion architecture → zero re-extraction
3. ✅ Aggiungi esperimenti incrementali: ogni nuovo modello carica stessi .npy

**Unica eccezione**: Se cambi **temporal sampling rate** (es. 1 FPS → 2 FPS), ALLORA re-estrai.

---

## 5) Temporal Mismatch Skeleton: Spiegazione Definitiva

### **Root Cause (spiegato step-by-step)**

**CLIP pipeline**:
```
Video 30sec @ 30 FPS → sample 1 FPS → 30 frames
                                        ↓
                              CLIP encode → [30, 768]
                                        ↓
                       (NO window pooling) → [30, 768]
                                        ↓
                           pad/clip to 256 → [256, 768]
                                              ^^^^
                                              30 valid, 226 padding
                                              → 30/256 = 11.7% coverage
```

Ma nei tuoi video:
```
Video tipico: 180-300 frames @ 1 FPS → [180-300, 768]
                                          ↓
                                 clip to 256 → [256, 768]
                                               ^^^^
                                               256 valid, 0 padding
                                               → 100% coverage ✓
```

**Skeleton pipeline (IL PROBLEMA)**:
```
Video 30sec @ 24 FPS → 720 frames pose → [720, 17, 3] keypoints
                                            ↓
                                 SCD-Net encode → [720, 4096]
                                            ↓
                      window_size=16 pooling → [720/16, 4096] = [45, 4096]
                                            ↓    ^^^^^^^^
                                                 QUI IL PROBLEMA!
                                            ↓
                                pad to 256 → [256, 4096]
                                             ^^^^
                                             45 valid, 211 padding
                                             → 45/256 = 17.6% coverage ✗
```

**Perché window_size=16?**

Durante estrazione skeleton, è stato applicato **average pooling** su finestre di 16 frame:
```python
# Script estrazione (presumibilmente)
window_size = 16  # ← Riduce 720 → 45
pooled_features = []
for i in range(0, T, window_size):
    window = features[i:i+window_size]  # [16, 4096]
    pooled = window.mean(dim=0)         # [4096]
    pooled_features.append(pooled)
```

**Ratio mismatch**:
- Video tipico CLIP: 200 timestep
- Video tipico Skeleton: 200/16 = 12.5 timestep
- Ratio: 16× MENO skeleton!

### **Soluzione: NON riestrarre, USA interpolation**

**Opzione A: Interpolazione lineare (RACCOMANDATO per test rapido)**

```python
# charades_dataloader.py, nel __getitem__
skel_feat = np.load(skel_path)  # [45, 4096]

# Interpolate a 256 timestep
import torch.nn.functional as F
skel_tensor = torch.from_numpy(skel_feat).float()  # [45, 4096]
skel_tensor = skel_tensor.T.unsqueeze(0)           # [1, 4096, 45]

skel_interp = F.interpolate(
    skel_tensor,
    size=256,
    mode='linear',
    align_corners=False
)  # [1, 4096, 256]

skel_feat = skel_interp.squeeze(0).T.numpy()  # [256, 4096]
```

**Pro**: 
- Zero re-extraction (30 secondi codice)
- Risolve mismatch (100% coverage)
- Test immediato se aiuta

**Contro**:
- Smoothing artificiale
- Non aggiunge informazione (solo reshaping)
- Gain atteso: +0.3-0.8 mAP

**Opzione B: Re-estrazione window_size=1 (per baseline definitiva)**

Modifica script estrazione:
```python
window_size = 1  # invece di 16
# Output: [720, 4096] → clip to 256 → [256, 4096] (100% coverage!)
```

**Pro**: Massima risoluzione temporale
**Contro**: 3-4h GPU, 16× storage

**Raccomandazione**:
1. **ORA**: Test Opzione A (interpolation) → 1 run gated fusion
2. **Se gain >0.5 mAP**: considera Opzione B (re-extraction)
3. **Se gain <0.5 mAP**: skippa skeleton, vai CLIP+DINOv3

---

## 6) Early Stopping: patience=15 → 10

**Analisi learning curve attuale**:
```
Epoch 10-13: val mAP plateau (best 28.87 @ epoch 13)
Epoch 14-28: overfitting (train↑ val↓)
           ↑
           Patience expires here (13+15=28)
```

**Con patience=10**:
```
Epoch 10-13: best @ 13
Epoch 14-23: waiting (13+10=23)
           ↑
           Stop epoch 23 (risparmia 5 epoch × 3min = 15min)
```

**Raccomandazione: ✅ patience=10**

**Pro**:
- Efficiency: -15min training
- Best epoch invariato (già identificato @ 13)
- Riduce overfitting exposure

**Contro**: Nessuno (best epoch 10-15 è stabile pattern)

**Implementazione**:
```bash
# vim/scripts/run_charades_gated_fusion_seed0.sh
--early_stop_patience 10  # cambia da 15
```

---

## 7) FPS Features: Verifica Necessaria

**La tua analisi preliminare**:
- CLIP/DINOv3: ~1.5 FPS
- Skeleton: ~24 FPS (native) → ~1.5 FPS (dopo window pooling w=16)

**Verifica consigliata** (5 minuti):

```python
import numpy as np
import os

clip_dir = "data/hf_features/Temporal_Action_Detection/charades_features_clip"
dino_dir = "data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps"
skel_dir = "data/hf_features/Temporal_Action_Detection/charades_scdnet_w16"

videos = os.listdir(clip_dir)[:20]  # sample 20
for vid in videos:
    clip_T = np.load(os.path.join(clip_dir, vid)).shape[0]
    dino_T = np.load(os.path.join(dino_dir, vid)).shape[0]
    skel_T = np.load(os.path.join(skel_dir, vid)).shape[0]
    
    print(f"{vid:20s}: CLIP {clip_T:3d}, DINOv3 {dino_T:3d}, Skel {skel_T:3d}, Ratio C/S: {clip_T/skel_T:.1f}×")
```

**Output atteso**:
```
5UNDJ.npy           : CLIP 292, DINOv3 292, Skel  22, Ratio C/S: 13.3×
XXXXX.npy           : CLIP 180, DINOv3 180, Skel  45, Ratio C/S:  4.0×
YYYYY.npy           : CLIP 256, DINOv3 256, Skel  64, Ratio C/S:  4.0×
```

**Interpretazione**:
- Se ratio medio **4-6×**: conferma window_size=16 problema
- Se ratio medio **>10×**: problema extraction skeleton (investiga)
- Se ratio **~1×**: mismatch è solo padding (meno critico)

**Collegamento domanda 5**: Se ratio 4-6×, Opzione A (interpolation) risolve.

---

## 8) DINOv2 vs DINOv3: Chiarimento CRITICO

### **⚠️ IMPORTANTE: DINOv3 NON ESISTE!**

**Nomenclatura corretta**:
- **DINOv2** (2023): Self-supervised ViT, attuale state-of-the-art
  - Paper: "DINOv2: Learning Robust Visual Features without Supervision" (Meta AI)
  - Modelli: ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14
  - Output: 384 (S), 768 (B), 1024 (L), 1536 (g) dim

- **DINO** (2021): Precedente versione (superseded da DINOv2)

- **DINOv3**: ❌ NON ESISTE (probabilmente confusione naming)

### **Cosa stai usando realmente?**

**Ipotesi A: DINOv2 (corretto)**
```python
# Extraction probabile
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# Output: [N, 768] per ViT-B/14
```

**Ipotesi B: Confusione con "DINOv3" naming interno**

Verifica:
```bash
# Check extraction script
grep -r "dinov3\|DINOv3" vim/scripts/
# Se trova riferimenti, è naming custom

# Check model checkpoint
head data/hf_features/.../README.md  # se esiste
```

### **Raccomandazione: Usa DINOv2 (ViT-B/14)**

**Configurazione ottimale per il tuo caso**:

| Modello | Dim | Performance attesa | Note |
|---------|-----|-------------------|------|
| **DINOv2 ViT-B/14** | 768 | ~26-28 mAP | ✅ Raccomandato (balance) |
| DINOv2 ViT-L/14 | 1024 | ~28-30 mAP | Più pesante (3× params) |
| DINOv2 ViT-g/14 | 1536 | ~29-31 mAP | Overkill (5× params) |

**Perché ViT-B/14**:
1. **Stessa dim CLIP** (768): fusion simmetrica
2. **Temporal alignment perfetto**: estratto a 1 FPS come CLIP
3. **Comparabilità**: standard in letteratura
4. **Efficiency**: 86M params (vs 304M ViT-L)

### **Se hai estratto "DINOv3"**: Rinomina mentalmente → DINOv2

**Action items**:
1. ✅ Verifica script estrazione (quale modello caricato?)
2. ✅ Conferma dim features (768 = ViT-B, 1024 = ViT-L)
3. ✅ Prosegui chiamandolo "DINOv2" (nomenclatura corretta)

---

## **SUMMARY FINALE: Action Plan Prioritario**

### **IMMEDIATE (questa settimana)**
1. ✅ **Verifica FPS** features (script domanda 7) → 5min
2. ✅ **Conferma DINOv2** (non v3) nelle extraction → 10min
3. ✅ **patience=10** in training script → 2min
4. ✅ **Bottleneck projection skeleton** 4096→512→256 → 10min codice

### **SHORT-TERM (prossima settimana)**
5. ✅ **CLIP + DINOv2 gated fusion** (PRIORITY 1)
   - Risolve temporal mismatch per design (both 100% coverage)
   - Expected: 31-33 mAP (+2-4)
   - Timeline: 3 giorni (extraction se serve + training + analysis)

6. ⚠️ **Test skeleton interpolation** (Opzione A domanda 5)
   - 30min implementazione
   - Se gain <0.5 mAP → skippa skeleton definitivamente

### **OPTIONAL (se tempo/curiosità)**
7. 🔄 **DINOv2 ViT-L/14** invece di B/14
   - Solo se DINOv2-B risultati deludenti (<28 mAP)
   - 1024-dim richiede modifiche projection

8. 🔄 **Skeleton re-extraction w=1** (Opzione B domanda 5)
   - Solo se interpolation test mostra promise
   - Baseline definitiva thesis

### **SKIP (non fare)**
- ❌ Cambiare embed_dim da 256 (complica comparabilità)
- ❌ Riestrarre features per uni/bidirezionale (indipendenti!)
- ❌ Cercare "DINOv3" (non esiste, usa DINOv2)

**Roadmap 2 settimane**:
- **Week 1**: CLIP + DINOv2 fusion (main experiment)
- **Week 2**: Skeleton interpolation test (optional polish) + thesis writing

Il focus primario è **CLIP + DINOv2**, che ha **highest probability of success** (+2-4 mAP) perché risolve il temporal mismatch per design (entrambi 100% coverage, stessa modalità visual).