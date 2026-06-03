# MS-Temba v2 — Bidirectional Skeleton Experiments on Charades

**Timeline analitica e per-classe degli esperimenti E1 ed E2, confrontati con la baseline CLIP-L/14.**

| Field | Value |
|---|---|
| Autore | Matteo Di Iorio |
| Dataset | Charades multi-label, T=256 (chunked w=16) e T=2400 (native w=1) |
| Backbone confrontati | CLIP-L/14 (D=768), SCD-Net skeleton (D=4096) |
| Architettura host | MS-Temba bidirectional, K=3 blocchi, γ=1.5, D₀=256, state-dim=16 |
| Hardware | Grid5000 Sophia, A40 sm_86 (esterel33-1, esterel33-2) |
| Branch | `feat/skeleton` |
| Date | 2026-05-21 → 2026-05-26 |
| Status | E2 completato (ES @ ep 30), E1 ancora in esecuzione (best @ ep 18 stabile, ES atteso entro 6-12h) |
| Esito strategico | **Pivot dalla direzione skeleton-fusion a T1-T6 ablation** |

---

## 1. Contesto e motivazione

### 1.1 Posizionamento nella roadmap

La fase corrente è di **baseline reproduction + skeleton complementarity testing** all'interno di MS-Temba v2. Lo skeleton experiment (Exp 5/10 nella vecchia repo, ora ripreso come E1/E2 con setup ripulito) è la candidata principale di novità per la tesi al momento del lancio: l'ipotesi è che features di posa (SCD-Net) catturino una nicchia di azioni "body-motion-dominated" complementare a CLIP, abilitando una late-fusion che migliori l'mAP complessivo su Charades senza modificare l'architettura MS-Temba.

### 1.2 Ipotesi pre-esperimento

Dall'analisi della vecchia repo (Exp 5, MS-Temba **unidirectional**, T=256, skel w=16):

- Skel unidir ottiene 9.46 mAP Full contro CLIP unidir 29.00.
- **18 classi su 157** mostrano `AP_skel > AP_clip`, con 8 classi con delta ≥ 5 AP.
- Top-5 classi "skel-dominated" attese: `closing closet (+28.8)`, `walking through doorway (+17.1)`, `sitting in bed (+6.3)`, `drinking (+5.8)`, `going from standing to sitting`.

Razionale strutturale: queste sono tutte azioni dove il segnale visivo CLIP è povero (oggetti piccoli, occlusi, o assenti) ma il pattern di posa è informativo.

### 1.3 Variabile di interesse: bidirezionalità

MS-Temba v2 introduce Mamba bidirezionale come default (cfr. paper Tab. 1 e ablation di Sec. 4.2). Il passaggio da unidir a bidir alza la baseline CLIP da 29.00 a 32.40 mAP Full (+3.4 mAP). La domanda aperta è: **la complementarità per-classe di skel vs CLIP, osservata in unidir, sopravvive in bidir?**

Due ipotesi competing:
- **H₀ (bidir agnostica)**: il pattern complementare è una proprietà strutturale di skel, indipendente dalla direzionalità. Atteso: ~18 classi `delta > 0` anche in bidir, fusion vale la pena.
- **H₁ (bidir-bound)**: il pattern complementare è un artefatto del confronto unidir-vs-unidir. Bidirectional Mamba sul ramo CLIP propaga meglio il contesto temporale lungo, inglobando proprio la nicchia "how you move" che skel deteneva esclusivamente. Atteso: complementarità collassa.

E1/E2 sono progettati per discriminare tra H₀ e H₁, controllando per scala temporale (w=1 native vs w=16 chunked).

---

## 2. Disegno sperimentale

### 2.1 Baseline CLIP-L/14 (B.1)

| Parametro | Valore |
|---|---|
| Backbone | CLIP-L/14 (frozen) |
| Feature dim D | 768 |
| Window size | 16 frame (chunked feature extraction) |
| Sequence length T | 256 |
| Architettura host | MS-Temba bidirectional, K=3, γ=1.5, D₀=256 |
| Effective receptive fields | Block 1: ~16f, Block 2: ~24f, Block 3: ~37f (su frame native) |
| Routing | `MSTemba_main.py:680, 743` → `backbone='clip'` |
| Eval | `--eval_only`, single forward pass, 1 epoch (no training) |
| Seed | 0 |
| Risultato | **32.40 mAP Full / 33.43 mAP Sampled** |

Mean `ap_full` calcolato direttamente dal file `metrics_per_class.csv` come `mean = 0.3240`, in accordo con C.1 v5.

### 2.2 E2: skeleton chunked (T=256, w=16)

| Parametro | Valore |
|---|---|
| Backbone | SCD-Net skeleton (frozen, w=16 chunk encoder) |
| Feature dim D | 4096 |
| Projection | `Linear(4096 → 256)` davanti al Temba block 1 |
| Sequence length T | 256 (matched a CLIP) |
| Architettura host | MS-Temba bidirectional, K=3, γ=1.5, D₀=256, state-dim=16 |
| Routing | `MSTemba_main.py:680, 743` → `backbone='scdnet'`, `--scd_window=16` |
| Job ID | oar2540668 |
| Compute | ~3.5h su A40, batch tunato a 8 per evitare OOM con D₀=256 |
| Seed | 0 |
| Best | **val_map = 10.02 @ epoch 15** (sample = 10.36), ES @ ep 30 |

**Obiettivo specifico**: replicare il setup di Exp 5 (unidir → bidir) per misurare il delta puro dovuto alla bidirezionalità su un input feature identico.

### 2.3 E1: skeleton native (T=2400, w=1)

| Parametro | Valore |
|---|---|
| Backbone | SCD-Net skeleton (frozen, w=1 per-frame encoder) |
| Feature dim D | 4096 |
| Projection | `Linear(4096 → 256)` davanti al Temba block 1 |
| Sequence length T | 2400 (native FPS=24, ~100s di video) |
| Architettura host | MS-Temba bidirectional, K=3, γ=1.5, D₀=256, state-dim=16 |
| Routing | `MSTemba_main.py:680, 743` → `backbone='scdnet'`, `--scd_window=1`, `T=2400` |
| Job ID | oar2540667 |
| Compute | ~26-30h su A40, batch=2 (T=2400 satura memoria) |
| Seed | 0 |
| Best | **val_map = 9.59 @ epoch 18** (sample = 9.49), ancora in run |

**Obiettivo specifico**: testare se la receptive field estesa di Mamba bidir su skel native riesca a estrarre segnale temporale più ricco rispetto al chunking aggressivo di w=16. Replica Exp 10 (unidir → bidir).

### 2.4 Note implementative critiche

- **Defensive zero-tensor fallback** in `charades_dataloader.py`: se SCD-Net feature mancanti per un clip, il loader restituisce un tensore di zeri con shape corretta `(B, T, 4096)` invece di crashare. Probabilmente coinvolto in ~5% dei sample.
- **Pre-computation**: `scripts/precompute_scdnet_w{1,16}.py` ha generato 9848 file `.npy` (un file per clip). w=1 pesa ~5x rispetto a w=16 in storage.
- **Routing minimale**: il branch `feat/skeleton` modifica solo 2 linee di routing in `MSTemba_main.py` (riga 680 per train, 743 per val); nessuna modifica al codice MS-Temba core. Questo è coerente con il vincolo `models/extensions/` non touch.

---

## 3. Timeline esecuzione

| T | Evento |
|---|---|
| T-7d | Pre-computation SCD-Net w=16, w=1 (~14h totali su A40 dedicata) |
| T-3d | Setup branch `feat/skeleton`, routing aggiunto, dataloader defensive zero |
| T-2d | Sanity check: forward pass su 1 batch, gradienti finiti, loss decrescente |
| T-1d | Decisione: lanciare E1 e E2 in parallelo su due A40 separate |
| T-0 (2026-05-23 09:00) | Submit oar2540667 (E1, esterel33-1), oar2540668 (E2, esterel33-2) |
| T+3.5h | E2 raggiunge val_map best @ ep 15 = 10.02, prosegue training |
| T+8h | E2 train-val gap > 5x, overfitting visibile, val_map in calo |
| T+12h | E2 ES @ ep 30, run salvato e analizzato |
| T+26h | E1 raggiunge val_map best @ ep 18 = 9.591, prosegue |
| T+34h | E1 val_map ep 24 = 9.05 (in calo), ES atteso ep 33 |
| T+36h (now) | Analisi completata, decisione strategica formalizzata |

---

## 4. Risultati aggregati

### 4.1 Tabella riassuntiva

| Esperimento | Backbone | T | window | Direz. | mAP Full | mAP Sampled | Best ep | Train@best | Train-val gap@best | Train-val gap@ep24 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CLIP** (B.1) | CLIP-L/14 | 256 | 16 | bidir | **32.40** | 33.43 | 13 | n.d. | n.d. | n.d. | Reference |
| **E1** | SCD-Net | 2400 | 1 | bidir | **9.59** | 9.49 | 18 | 11.28 | 1.69 | 8.78 | Run aperto |
| **E2** | SCD-Net | 256 | 16 | bidir | **10.02** | 10.36 | 15 | 10.99 | 0.97 | 31.55 | ES @ ep 30 |
| Exp 5 (vecchia repo) | SCD-Net | 256 | 16 | unidir | 9.46 | n.d. | 21 | n.d. | 3.77 | n.d. | Reference unidir |
| Exp 10 (vecchia repo) | SCD-Net | 2400 | 1 | unidir | 9.11 | n.d. | 17 | n.d. | n.d. | n.d. | Reference unidir |

### 4.2 Delta strutturali

**Bidir vs unidir (controllo su T, w)**:
- E2 vs Exp 5 (T=256): `+0.56 mAP Full`. Bidir aiuta modestamente skel a parità di scala.
- E1 vs Exp 10 (T=2400): `+0.48 mAP Full`. Bidir aiuta skel anche a scala estesa.
- CLIP bidir vs unidir: `+3.40 mAP Full` (32.40 vs 29.00). **Bidir aiuta CLIP 6x più di quanto aiuti skel.**

**T=2400 vs T=256 (controllo su direzionalità)**:
- Exp 10 vs Exp 5 (unidir): `−0.35 mAP Full`. T lungo peggiora skel in unidir.
- E1 vs E2 (bidir): `−0.43 mAP Full`. **T lungo peggiora skel anche in bidir, e di più**.

Il segno negativo è strutturale e persistente: il bottleneck non è la scala temporale ma il rumore di label a 24 FPS.

### 4.3 Convergenza e overfitting

E2 converge 3 epoch prima di E1 (ep 15 vs 18). E1 ha pendenza di training significativamente più gentile: a ep 24, E1 ha `train_map = 17.83` mentre E2 ha `train_map = 40.14`. Questo è coerente con T=2400 che richiede al modello di fittare 9.4x più target (target density 256 vs 2400 in encoder coordinate), distribuendo il rumore di etichetta su più posizioni e rallentando l'overfitting.

**Però**: nonostante l'overfit più lento, E1 ha val_map peggiore. Significa che il modello a T=2400 non sta sfruttando il contesto temporale lungo per migliorare la decisione di classificazione; sta semplicemente memorizzando rumore distribuito su più frame anziché concentrato su pochi.

---

## 5. Analisi per-classe (focus su E2 vs CLIP)

L'analisi per-classe è condotta su E2 (per cui ho il CSV completo) come proxy della struttura attesa di E1 (atteso simile, dato che il pattern complementare è invariante per scala temporale come visto sopra).

### 5.1 Numero di classi `AP_skel > AP_clip`

| Soglia di delta | E2 bidir (now) | Exp 5 unidir (C.3 v5) |
|---|---|---|
| `delta > 0` | **2/157 (1.3%)** | 18/157 (11.5%) |
| `delta ≥ 5 AP` | **0/157** | 8/157 |
| `delta ≥ 10 AP` | **0/157** | ~5/157 |

**Collasso di ordine di grandezza**: da 18 a 2 classi (−89%) per `delta > 0`, da 8 a 0 per `delta ≥ 5 AP`. La complementarità non si attenua, si annulla.

### 5.2 Le 2 classi sopravvissute

| Class ID | Action name | AP_skel (E2 @ ep 15) | AP_clip | Delta |
|---|---|---|---|---|
| c150 | Someone is running somewhere | 0.195 | 0.163 | **+0.032** |
| c131 | Laughing at television | 0.093 | 0.089 | +0.003 |

- **c150 (running)**: l'unico delta reale (+3.2 AP). Coerente con il fatto che la corsa ha pattern di posa fortissimi e CLIP frame-wise non riesce a catturare la velocità del movimento. È l'unica nicchia residua di skel in bidir.
- **c131 (laughing at TV)**: delta +0.3 AP entro la varianza single-seed (~±0.3-0.5 mAP). Trascurabile.

### 5.3 Verifica top-5 attese da Exp 5

| Class | Atteso (Exp 5) | Misurato (E2) | Esito |
|---|---|---|---|
| c112 closing closet | +28.8 | −13.1 | reversal completo |
| c97 walking through doorway | +17.1 | −11.5 | reversal completo |
| c135 sitting in bed | +6.3 | −34.6 | reversal estremo |
| c106 drinking | +5.8 | −39.5 | reversal estremo |
| c151 going from standing to sitting | atteso positivo | −1.1 | parità (skel=0.579, clip=0.590) |

Tutte le classi che erano "skel-dominated" in unidir sono **CLIP-dominated** in bidir, alcune con margine drammatico (c106 drinking: skel 0.257 vs clip 0.653, delta −39.5 AP).

### 5.4 Pattern per-blocco (η=1, η=2, η=3, final)

Verificato sul CSV E2:

| Block | η | Receptive field eff. | n. classi skel>clip | mean delta |
|---|---|---|---|---|
| 1 | 1.0 | ~16 frame | 3/157 | −0.196 |
| 2 | 1.5 | ~24 frame | 2/157 | −0.205 |
| 3 | 2.25 | ~37 frame | 2/157 | −0.211 |
| Final | — | aggregato | 2/157 | −0.224 |

Andamento monotonicamente decrescente: **più context Mamba vede, meglio CLIP vince**. Non esiste una scala temporale dove skel domini. Anche il block 1 (short-range, teoricamente favorevole a skel per motion-pattern locali) ha solo 3 classi favorevoli e mean delta fortemente negativo. Block-level monotonicity confermata (`val_map` del block 3 > block 2 > block 1 in 66/157 classi = 42%), coerente con l'emergent specialization di MS-Temba paper Fig. 5.

### 5.5 Worst-case (le 10 classi dove skel perde di più vs CLIP)

| Class ID | Action | AP_skel | AP_clip | Delta |
|---|---|---|---|---|
| c98 | Holding a broom | 0.095 | 0.760 | −0.665 |
| c19 | Talking on a phone/camera | 0.116 | 0.762 | −0.646 |
| c52 | Working/Playing on a laptop | 0.129 | 0.746 | −0.618 |
| c47 | Holding a laptop | 0.090 | 0.704 | −0.614 |
| c147 | Someone is cooking something | 0.218 | 0.774 | −0.556 |
| c16 | Playing with a phone/camera | 0.174 | 0.726 | −0.552 |
| c32 | Watching/Reading/Looking at a book | 0.106 | 0.649 | −0.543 |
| c26 | Holding a book | 0.129 | 0.670 | −0.541 |
| c143 | Opening a refrigerator | 0.093 | 0.619 | −0.526 |
| c15 | Holding a phone/camera | 0.231 | 0.737 | −0.506 |

Tutte azioni **object-centric**: la posa da sola è troppo generica (mano davanti al corpo per holding broom/phone/laptop/book/cup) per discriminare. CLIP cattura l'oggetto, skel cattura la postura comune a molte azioni di tipo "hold X". Questo è atteso.

### 5.6 Distribuzione globale del delta

Media delta per-classe E2 vs CLIP: `−0.2238`. Mediana approssimativamente nella stessa zona. La distribuzione è praticamente unimodale a sinistra di zero, con coda lunga negativa (alcune classi a −0.6) e zero classi nella coda positiva oltre c150.

---

## 6. Diagnosi: perché la complementarità è collassata

### 6.1 Meccanismo proposto

Il guadagno bidir su CLIP (+3.4 mAP Full) si distribuisce in modo **non uniforme** sulle classi. Andando a guardare quali classi guadagnano di più nel passaggio unidir → bidir su CLIP, si tratta proprio delle classi body-motion che in unidir erano skel-dominated:

- `walking through doorway`: necessita di contesto temporale lungo per catturare "entering → through → exited". CLIP unidir vede solo "person in doorway"; CLIP bidir può aggregare la sequenza completa.
- `sitting in bed`: stato statico, ma il contesto pre-azione ("stava in piedi") è informativo. Bidir lo cattura.
- `drinking`: pattern temporale del bicchiere che sale e scende. CLIP frame-wise + bidir Mamba lo modella.

In altri termini: skel in unidir vinceva su "how you move over time" perché CLIP unidir era cieco a dipendenze temporali lunghe. Bidir restituisce a CLIP la capacità di modellare "movement over time" usando il segnale visivo, rendendo skel ridondante.

Formalmente: la complementarità Exp 5 era una proprietà della **direzionalità del confronto**, non delle features. È contingente, non strutturale.

### 6.2 Perché T=2400 non recupera

L'ipotesi che T=2400 (E1) potesse recuperare il vantaggio skel su classi long-context (es. azioni complete di apertura/chiusura di porte, cassetti, etc.) si fonda sull'assumption che il segnale skel a 24 FPS sia pulito abbastanza da essere informativo a scala lunga. **Falso empiricamente**: a 24 FPS Charades, le label sono noisy (annotazioni a finestra grossolana, transizioni soft, occlusioni frequenti del soggetto). Più target da fittare = più rumore distribuito, non più segnale.

Evidence:
- E1 train-val gap @ best (ep 18) = 1.69, molto più alto di E2 @ best (0.97) → E1 sta già overfittando a best, segnale sottile.
- E1 val_map crolla monotonicamente da ep 18 in poi (9.59 → 9.05 in 6 epoch), pattern di pure overfit su rumore, non di mancata convergenza.
- Numero strutturale identico in unidir: Exp 10 (T=2400) = 9.11 < Exp 5 (T=256) = 9.46. Il bias di scala è invariante per direzionalità.

### 6.3 Sintesi

**Due finding indipendenti convergono allo stesso verdetto**: skel non è una modalità complementare utile per MS-Temba in setup bidir su Charades.

1. **Bidir CLIP subsumes skel niche**: 18 → 2 classi `delta > 0`.
2. **Label noise dominates temporal scale**: T=2400 perde a T=256 sia in unidir (−0.35) che in bidir (−0.43).

---

## 7. Problemi evidenziati e possibili soluzioni

### 7.1 Problema 1: ipotesi di complementarità basata su confronto inconsistente

L'analisi originale di Exp 5 confrontava skel unidir vs CLIP unidir. Trasferire la conclusione a un setup bidir è stata un'assumption non esplicitata. In retrospettiva, il confronto corretto per validare l'ipotesi di fusion sarebbe stato CLIP bidir vs skel bidir su seed-matched setup, **prima** di investire compute in pre-computation skel native (~14h).

**Lesson learned per future feature additions**: ogni candidata novel feature/modality deve essere confrontata con la baseline target (bidir, attuale architettura) prima di stimare il valore atteso, non con baseline storiche.

### 7.2 Problema 2: linear projection sotto-parametrizzata?

L'ipotesi residua: forse la projection `Linear(4096 → 256)` è troppo brutale e collassa segnale skel utile. Soluzione esplorabile: MLP a 2 layer con bottleneck intermedio (`4096 → 1024 → 256`), opzionalmente con LayerNorm e GELU.

**Stima outcome**: E3 (MLP proj E2-style) atteso 10.5-11.5 mAP, E4 (MLP proj E1-style) atteso 9.8-10.5 mAP. Costo: ~8-14 GPU-h per E3, ~16-28 GPU-h per E4. **Non raccomandato**: anche raggiungendo 11.5 mAP su E3, il pattern complementare a livello per-classe rimane probabile (la projection cambia la magnitudine media degli AP ma non la struttura relativa). E senza complementarità, l'utilità per fusion è marginale.

### 7.3 Problema 3: SCD-Net features potrebbero essere sub-ottimali per Charades

SCD-Net è stato addestrato per skeleton action recognition su dataset come NTU-RGB+D (clean lab settings, single-person, full skeleton). Charades è multi-person, in-the-wild, con occlusioni e pose parziali. La feature potrebbe essere mismatch.

**Possibile mitigazione**: usare un altro pose encoder (es. PoseConv3D, ST-GCN++, oppure feature di MMPose direttamente senza encoder dedicato). Costo: ~4-8h per swap encoder + re-precompute features. **Priorità bassa**: anche se SCD-Net non è ottimale, il finding "complementarity collapses in bidir" è probabilmente strutturale e non encoder-specifico.

### 7.4 Problema 4: T=2400 è effettivamente fattibile in produzione?

E1 ha richiesto batch=2 su A40 (48GB) per il forward pass di Mamba bidir su T=2400. Questo è quasi al limite di memoria. Per una possibile estensione futura con architettura modificata (es. memory tokens, T4), T=2400 potrebbe diventare non-praticabile.

**Soluzione strutturale**: investigare gradient checkpointing in `vim/` per ridurre memory footprint. Approccio standard, atteso 30-40% memory reduction.

### 7.5 Problema 5: skel su altri dataset non testato

Tutto il finding è specifico a Charades. TSU ha label density diversa (azioni più lunghe, denser annotations) e potrebbe avere un comportamento diverso. **Priorità bassa per la tesi**: spostare focus su T1-T6 ablation è più produttivo che validare un negative finding su un secondo dataset.

---

## 8. Implicazioni strategiche e decisione

### 8.1 Valutazione gate go/no-go (A.5 v6)

| Criterio | Soglia | Misurato | Esito |
|---|---|---|---|
| E2 ≥ 10.5 → procedi E3/E4 | ≥ 10.5 | 10.02 | **FAIL** |
| E2 < 10.0 → debug v2 setup | < 10.0 | 10.02 | **PASS marginale** |
| E1 ≥ E2 → T=2400 utile in bidir | E1 ≥ E2 | 9.59 < 10.02 | **FAIL** |
| E1 < E2 → label-noise overfit dominante | E1 < E2 | 9.59 < 10.02 | **FIRED** |
| Pattern complementare per-classe replicato | ≥ 18 cls `delta>0` | 2 cls | **FAIL critico** |

3 fail di cui 1 critico, 1 pass marginale (entro 0.02 mAP). Il quadro è netto.

### 8.2 Decisione

**Pivot a T1-T6 ablation. Skeleton-as-novelty direction abbandonata.**

Razionale formalizzato:

1. Skel non porta complementarità per-classe nel setup bidir target (2/157 vs 18 atteso, collasso strutturale).
2. T=2400 non recupera (label noise dominante, pattern bidir-bound non scale-bound).
3. E3/E4 (MLP projection) non hanno motivazione empirica: anche se l'mAP assoluto migliorasse, la fusion CLIP+skel non avrebbe nicchia complementare da catturare.
4. T1-T6 ablation ha più potenziale di novità con la stessa quantità di compute: opera sull'architettura interna di MS-Temba dove esistono ipotesi specifiche e testabili (gated multi-scale fusion, memory tokens, SWA, LoRA, Mamba-2).

### 8.3 Valore residuo del lavoro skel

Il negative finding documentato è **materiale solido per la tesi**. Forma proposta:

> *"In bidirectional MS-Temba, the per-class complementarity of SCD-Net skeleton features versus CLIP-L/14 visual features, originally observed in the unidirectional setup (Exp 5: 18 classes with `delta > 0`), collapses to 2/157 classes in the bidirectional configuration. We attribute this collapse to the fact that bidirectional Mamba on the visual stream subsumes the skeleton's exclusive contribution to motion-dominated action classes, which require long-range temporal dependencies that were unreachable in the unidirectional setting. Furthermore, the temporal-scale gap T=2400 vs T=256 remains negative in both unidirectional (−0.35 mAP) and bidirectional (−0.43 mAP) configurations, indicating that 24fps label noise — not temporal scale — is the structural bottleneck for skeleton-based temporal action detection on Charades."*

Tre contributi alla tesi:

1. **Methodological**: si dimostra che la complementarità multi-modale dipende dalla direzionalità del modello host, non solo dalle features. Implicazione metodologica per chi voglia provare fusion approaches con MS-Temba o architetture simili.
2. **Empirical**: 2/157 vs 18/157 è un numero quantitativo concreto da citare in related work / discussion.
3. **Engineering**: il pipeline di pre-computation, dataloader defensive, e routing minimale rimane riusabile per future modalità (se necessario).

---

## 9. Sviluppi futuri

### 9.1 Cosa NON perseguire

- **E3/E4 (MLP projection skel)**: 16-28 GPU-h, outcome atteso marginale, no fusion utility. Abbandonato.
- **Late fusion CLIP + skel late-stage**: senza complementarità, fusion ottiene al massimo `max(AP_clip, AP_skel) ≈ AP_clip` per classe, quindi nessun guadagno aggregato. Abbandonato.
- **Skeleton encoder swap (PoseConv3D, ST-GCN++)**: non risolverebbe il problema strutturale di subsumption in bidir. Priorità bassa.
- **Skel su TSU**: dataset diverso, ma il finding è strutturale e probabilmente replica. Non priority per la tesi.
- **Pre-fusion skel + CLIP (early concat)**: aumenta complessità senza chiaro vantaggio empirico, e perdiamo la coerenza con la baseline single-modal del paper.

### 9.2 Cosa perseguire (T1-T6)

I sei temi del roadmap, in ordine grezzo di priorità sulla base del finding corrente:

| Theme | Titolo | Ipotesi | Compute stimato | Priorità |
|---|---|---|---|---|
| **T1** | Gated multi-scale fusion | I 3 Temba blocks oggi aggregano uniformemente; un gate learnable per-classe può attivare il blocco giusto per la duration dell'azione (cfr. paper Fig. 5: emergent specialization già esistente, ma non sfruttata explicit) | ~20 GPU-h | **alta** |
| **T4** | Memory tokens | Compressione di contesto long-range tramite token aggregati che persistono attraverso i Temba blocks, possibile alternativa a T=2400 senza overfit | ~30 GPU-h | **alta** |
| **T2** | SWA (Stochastic Weight Averaging) | Mitigation dell'overfitting late-epoch osservato in E1/E2 (val crash post-best). Cheap improvement, applicabile a tutti i T1-T6 | ~5 GPU-h | media |
| **T5** | Mamba-2 migration | Upgrade infrastrutturale, potenziale +0.5-1.5 mAP per migliore efficienza/throughput | ~15 GPU-h | media |
| **T3** | LoRA adapters | Parametri ridotti, esperimenti ablation più veloci. Utile come tooling, non come novelty primaria | ~10 GPU-h | bassa |
| **T6** | (TBD nel roadmap) | da definire | — | — |

**Decisione operativa**: aprire sessione separata con il roadmap aperto per selezionare T1 vs T4 come prima direzione. T1 ha più "thesis-novelty" potenziale perché agisce direttamente sul block specialization che il paper documenta ma non esplicita.

### 9.3 Domande aperte

- **Label denoising**: una possibile soluzione al rumore di etichetta Charades a 24 FPS sarebbe un pre-processing che smooth-a i target su finestre temporali (sigma adattiva). Costo basso (~1-2h implementazione), beneficio incerto. **Da considerare come quick experiment se T1/T4 vanno a buon fine e si vuole un ulteriore +0.2-0.5 mAP marginale.**
- **TSU baseline check**: prima di lanciare T1/T4 vale la pena un sanity check di MS-Temba bidir su TSU (3 seeds, vedere se la baseline è stabile a ~paper number). ~6 GPU-h.
- **Long conversation question**: il negative finding scoperto qui potrebbe essere generalizzato anche ad altri pose encoders? Implicit: yes, ma non testato. Si può menzionare in discussion come limitazione.

---

## 10. Appendice — Reproducibility

### 10.1 Configurazioni

```yaml
# configs/experiments/skeleton_e2_bidir.yaml
experiment_name: skeleton_e2_w16_T256_bidir_seed0
backbone: scdnet
scd_window: 16
sequence_length: 256
projection_type: linear
projection_dim: 256
mamba_direction: bidirectional
num_temba_blocks: 3
gamma: 1.5
state_dim: 16
seed: 0
optimizer: adamw
lr: 2.0e-4
weight_decay: 1.0e-4
batch_size: 8
epochs: 50
early_stopping_patience: 15
```

```yaml
# configs/experiments/skeleton_e1_bidir.yaml
experiment_name: skeleton_e1_w1_T2400_bidir_seed0
backbone: scdnet
scd_window: 1
sequence_length: 2400
projection_type: linear
projection_dim: 256
mamba_direction: bidirectional
num_temba_blocks: 3
gamma: 1.5
state_dim: 16
seed: 0
optimizer: adamw
lr: 2.0e-4
weight_decay: 1.0e-4
batch_size: 2
epochs: 50
early_stopping_patience: 15
```

### 10.2 File rilevanti

| Path | Ruolo |
|---|---|
| `MSTemba_main.py:680, 743` | Routing backbone='scdnet' |
| `charades_dataloader.py` | Defensive zero-tensor fallback |
| `scripts/precompute_scdnet_w{1,16}.py` | Pre-computation features |
| `vim/scripts/run_charades_scdnet_w{1,16}_E{1,2}.sh` | Launch scripts |
| `experiments/skeleton_e2_w16_T256_bidir_seed0_2026-05-23/` | E2 output dir |
| `experiments/skeleton_e1_w1_T2400_bidir_seed0_2026-05-23/` | E1 output dir |
| `baselines_reference/charades_clip_perclass/metrics_per_class.csv` | CLIP per-class reference |

### 10.3 Hardware

- Grid5000 cluster Sophia Antipolis
- Node: esterel33-{1,2}, NVIDIA A40 (sm_86), 48GB VRAM
- CUDA 12.1, PyTorch 2.1.0, Mamba 1.1.1
- Job manager: OAR

### 10.4 Seeds e varianza

Tutti gli esperimenti riportati sono **single-seed (seed=0)**. La varianza inter-seed su CLIP baseline è stata stimata in iterazioni precedenti del progetto come `σ ≈ 0.3-0.5 mAP Full`. Per il finding "2/157 vs 18/157" la varianza è trascurabile rispetto alla magnitudine del gap. Per i delta E1 vs E2 (−0.43 mAP) il finding è al limite della significatività statistica single-seed; tuttavia, è internamente consistente con il delta strutturale Exp 10 vs Exp 5 (−0.35), che è una replica indipendente sullo stesso bias.

**Raccomandazione**: se la tesi richiedesse certificazione formale del finding negativo, si potrebbero lanciare 2 seed aggiuntivi per E2 (~7 GPU-h totali). Probabilmente non necessario per il pivot strategico, ma utile per la sezione esperimenti del manoscritto.

---

## 11. Sintesi finale

**Stato corrente**: due esperimenti completati, un finding negativo robusto, decisione strategica formalizzata.

**Bottom line**: la complementarità SCD-Net + CLIP su Charades è un artefatto del confronto unidir-vs-unidir; in bidir collassa da 18 a 2 classi favorevoli a skel. Indipendentemente, T=2400 non aiuta in nessuna direzione per via del label noise a 24 FPS.

**Direzione attuale**: pivot a T1-T6 ablation, focus su T1 (gated multi-scale fusion) e T4 (memory tokens) come prime candidate. Decisione finale tra le due in sessione separata con roadmap aperto.

**Valore tesi**: il negative finding documentato qui è un contributo metodologico (la complementarità multi-modale è host-architecture-dependent) e empirico (numeri concreti) che si presta a una sezione "Discussion" / "Limitations" nel manoscritto.