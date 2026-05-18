# MS-Temba su Charades: Analisi Completa CLIP vs DINOv3

> **Dataset**: Charades v1 — 157 classi di azioni domestiche, 7985 video di training, 1863 di test  
> **Modello**: MS-Temba (Multi-Scale Temporal Mamba)  
> **Backbone A**: CLIP ViT (512-dim features) — Best checkpoint **ep13**, Full-val-MAP **32.40**, Sampled-val-MAP **33.43**  
> **Backbone B**: DINOv3 ViT (1024-dim features, reg v2) — Best checkpoint **ep13**, Full-val-MAP **25.21**, Sampled-val-MAP **25.82**  
> **Nota**: i risultati DINOv3 si riferiscono alla configurazione `dinov3_vitl16_reg_v2` (drop=0.05, drop_path=0.05, weight_decay=0.05, early_stop_patience=15), che costituisce la baseline definitiva per DINOv3 dopo una serie di esperimenti di regolarizzazione (cfr. Sezione 4.2).

---

## Indice

1. [CLIP come backbone per feature extraction](#1-clip-come-backbone-per-feature-extraction)
2. [DINOv3 come backbone per feature extraction](#2-dinov3-come-backbone-per-feature-extraction)
3. [Confronto diretto tra CLIP e DINOv3](#3-confronto-diretto-tra-clip-e-dinov3)
4. [Configurazione di Training](#4-configurazione-di-training)
5. [Risultati Globali](#5-risultati-globali)
6. [Curve di Training Completo (ep 0–49)](#6-curve-di-training-completo)
   - 6.1 Curve di Training CLIP
   - 6.2 Curve di Training DINOv3 (reg v2)
7. [Tabella Completa per Classe (157 classi)](#7-tabella-completa-per-classe-157-classi)
8. [Analisi Distribuzione AP](#8-analisi-distribuzione-ap)
9. [Analisi per Oggetto (38 categorie)](#9-analisi-per-oggetto-38-categorie)
10. [Analisi per Verbo (33 categorie)](#10-analisi-per-verbo-33-categorie)
11. [Coppie di Classi Confondibili](#11-coppie-di-classi-confondibili)
12. [Struttura Compositiva: Mapping Oggetto × Verbo](#12-struttura-compositiva-mapping-oggetto--verbo)
13. [Classi con Ambiguità Strutturale e Dataset Issue](#13-classi-con-ambiguità-strutturale-o-dataset-issue)
14. [Raccomandazioni e Sviluppi Futuri](#14-raccomandazioni-e-sviluppi-futuri)
15. [Integrazione di Feature Skeleton con CLIP](#15-integrazione-di-feature-skeleton-con-clip)

---

# 1. CLIP come backbone per feature extraction

## 1.1 Tipologia di feature prodotte

CLIP (*Contrastive Language–Image Pre-training*, OpenAI) è un **Vision Transformer (ViT)** addestrato su circa 400 milioni di coppie immagine–testo. In questo contesto viene usato esclusivamente come **estrattore visivo**: si utilizza l'`encode_image()` del modello per ottenere un embedding globale di dimensione **512** per ogni frame del video, che MS-Temba riceve come input e processa temporalmente.

Il ViT suddivide ogni frame in patch non sovrapposte (tipicamente 16×16 pixel per ViT-B/16 o 14×14 per ViT-L/14) e le proietta in uno spazio latente. Viene aggiunto un token speciale **[CLS]** che, dopo N layer di self-attention, aggrega l'informazione di tutte le patch. L'output dell'`encode_image()` è proprio questo token CLS, proiettato linearmente in uno spazio condiviso di dimensione 512 — lo stesso spazio in cui sono proiettati gli embedding testuali. Il risultato è un **vettore globale per frame**: non vi sono embedding individuali per patch o regioni esportati a MS-Temba, ma un'unica rappresentazione compatta dell'intera immagine.

Questa scelta architetturale ha implicazioni dirette: tutto ciò che non emerge nell'aggregazione CLS — distribuzione locale delle patch, relazioni spaziali fini, configurazione posturale — è irreversibilmente perso prima che MS-Temba riceva il dato.

## 1.2 Come il pretraining plasma la rappresentazione

Il training contrastivo ottimizza l'encoder visivo affinché le immagini siano vicine, nello spazio embedding, alle loro descrizioni testuali naturali. Formalmente, dato un batch di N coppie (immagine, testo), l'obiettivo massimizza la similarità coseno delle N coppie corrette e minimizza quella delle N²−N coppie errate. Questo ha una conseguenza diretta sulla geometria dello spazio 512-dim: i vettori si organizzano attorno a **concetti semantici linguisticamente frequenti** nel corpus web — oggetti, categorie, contesti d'uso, azioni tipiche. Due frame risultano vicini nello spazio embedding non perché abbiano la stessa composizione visiva, ma perché potrebbero essere descritti con le stesse parole.

Per MS-Temba ciò ha un effetto importante: la sequenza temporale di feature in input è già fortemente separata per semantica globale. Il modello Mamba riceve, per ciascun frame, un punto in uno spazio in cui *cooking* è già lontano da *holding a phone*, e *lying on a bed* è già lontano da *walking through a doorway*. Il contributo di MS-Temba si riduce quindi, in larga misura, a distinguere nel tempo i pattern di transizione tra questi concetti, piuttosto che costruire da zero la rappresentazione semantica dell'azione.

Un corollario importante è che la separabilità nello spazio CLIP è asimmetrica rispetto ai verbi: azioni con lo stesso oggetto ma verbi diversi (*putting a bag* vs *taking a bag*, *opening a door* vs *closing a door*) tendono a produrre embedding simili tra loro, perché il corpus testuale descrive entrambe con parole molto vicine e nello stesso contesto. La distinzione verbo-dipendente è quindi un compito che grava quasi interamente su MS-Temba.

## 1.3 Punti di forza per l'estrazione di feature in questo contesto

**Discriminatività semantica compatta.** A 512 dimensioni, la feature CLIP offre un segnale già fortemente separato per le classi semanticamente distinte di Charades. Questo riduce il rischio di overfitting nel classificatore downstream di MS-Temba: uno spazio di feature compatto e ben strutturato richiede meno dati per addestrare un classificatore affidabile rispetto a uno spazio ad alta dimensionalità con geometria non supervisionata. Classi come *cooking* (AP 79.0), *talking on the phone* (AP 75.9), *working on a laptop* (AP 75.9) o *holding a broom* (AP 75.4) raggiungono AP altissimi proprio perché queste categorie corrispondono a concetti chiaramente ancorati nel corpus testuale di pretraining: il backbone ha già separato questi cluster prima che MS-Temba intervenga.

**Copertura semantica web-scale.** Il pretraining su 400M di coppie immagine–testo espone il backbone a una vastissima varietà di azioni domestiche, molte delle quali rientrano naturalmente nel vocabolario di Charades. Di conseguenza, la feature prodotta tende a essere informativa anche per classi non frequentissime nel training set, purché siano linguisticamente ben caratterizzate — un vantaggio diretto in un dataset come Charades, dove la distribuzione delle classi è fortemente sbilanciata e molte azioni hanno meno di 100 occorrenze di training.

**Stabilità temporale del segnale.** Poiché le feature CLIP sono ancorate a concetti semantici stabili, la sequenza frame-by-frame che MS-Temba riceve tende a essere coerente e poco rumorosa per le classi ad alta AP: scene di *cooking*, *watching television* o *lying on a bed* producono vettori CLS molto simili tra frame consecutivi, rendendo più facile per il modello temporale rilevare l'inizio e la fine dell'azione.

## 1.4 Limiti per l'estrazione di feature in questo contesto

**Perdita di risoluzione spaziale.** La compressione nel token CLS è il limite strutturale più importante. MS-Temba riceve per ogni frame un unico vettore 512-dim: le informazioni sulla postura del soggetto, sulla disposizione degli arti, sulla relazione spaziale mano–oggetto e sulla configurazione corporea sono tutte aggregate in questo singolo punto, con perdita irreversibile del dettaglio locale. Il self-attention del ViT, pur potendo in linea di principio catturare relazioni patch-to-patch, non è stato addestrato a preservare questa geometria locale nel token CLS, perché il segnale di training era la compatibilità con descrizioni testuali globali. Questo penalizza sistematicamente le classi in cui il segnale discriminativo è configurazionale piuttosto che nominale: *standing to sitting* (AP 60.4 vs 23.2 DINOv3), *someone is running* (AP 18.9 vs 9.1 DINOv3), *someone is sneezing* (AP 17.8 vs 11.8 DINOv3): nei dati reali DINOv3 non risolve queste classi meglio di CLIP, contrariamente a quanto ipotizzato in precedenza.

**Ambiguità verbo-dipendente.** Come anticipato, coppie di classi con stesso oggetto e verbi contrapposti producono embedding molto simili. Questo non è solo un problema di separabilità nello spazio feature, ma strutturalmente riflette il fatto che il corpus testuale raramente distingue *put* e *take* con descrizioni visivamente informative. MS-Temba deve quindi imparare a discriminare queste coppie quasi esclusivamente dalle dinamiche temporali, senza un prior geometrico utile dalla feature — e questo spiega perché classi come *closing a door* (AP 30.3) vs *opening a door* (AP 39.8) o *putting a bag* (AP 40.6) vs *taking a bag* (AP 25.4) mostrino performance disomogenee malgrado frequenze di training simili.

**Saturazione semantica per classi visivamente ricche.** Alcune classi di Charades sono visivamente molto specifiche ma semanticamente generiche nel linguaggio naturale — ad esempio *throwing* è un verbo raro e visivamente violento che il corpus web associa a contesti molto diversi da quelli domestici. Questo produce feature poco informative per tutte le classi con `throw` come verbo, come confermato dal mean AP di v025 (throw) pari a soli 10.2, il più basso tra tutti i verbi.

---

# 2. DINOv3 come backbone per feature extraction

## 2.1 Tipologia di feature prodotte

DINOv3 (Meta/FAIR) è una famiglia di **Vision Transformer** addestrati con self-supervised learning puramente visivo su un dataset web-scale proprietario (LVD-1689M). In questo contesto viene impiegata la variante **ViT-L/14** — 24 layer di transformer, patch size 14×14, dimensione interna 1024 — che produce embedding di dimensione **1024**. A differenza di CLIP, DINOv3 è progettato per produrre sia un embedding globale (token CLS) sia **feature dense per ogni patch** di alta qualità; nel setup attuale tuttavia MS-Temba riceve esclusivamente il vettore CLS da 1024 dimensioni per frame, lasciando inutilizzata la ricchezza delle rappresentazioni patch-level.

Rispetto a CLIP, la feature è più capiente (1024 vs 512) ma priva di qualsiasi ancoraggio linguistico: rappresenta l'immagine secondo una geometria appresa interamente dalla struttura visiva statistica, senza che nessuna etichetta o descrizione testuale abbia mai guidato il training.

## 2.2 Come il pretraining plasma la rappresentazione

Il training self-supervised di DINOv3 si basa su una procedura **student-teacher con EMA** (Exponential Moving Average): il teacher è una media mobile dei pesi dello studente, e l'obiettivo è far sì che lo studente, su viste fortemente augmentate dell'immagine, riproduca le distribuzioni di attivazione del teacher su viste minimalmente distorte. Non essendoci testo né etichette, il segnale di apprendimento proviene interamente dalla struttura interna delle immagini stesse.

Una caratteristica cruciale di DINOv3 rispetto alle versioni precedenti è il meccanismo di **Gram anchoring**: durante il training, viene regolarizzata non solo la feature globale, ma la matrice di similarità tra i token di patch — ovvero le relazioni patch-to-patch. In pratica, l'encoder viene spinto a preservare *come le diverse regioni dell'immagine stanno tra loro*, non solo quale embedding globale produca. Questo ha un effetto diretto sulla qualità del token CLS: pur producendo un singolo vettore globale, l'aggregazione avviene in un ViT i cui layer interni hanno imparato a mantenere la coerenza spaziale locale. Il risultato è un CLS token che, rispetto a quello di CLIP, porta con sé più informazione implicita sulla struttura geometrica della scena.

Per MS-Temba questo si traduce in un vettore CLS organizzato attorno alla **struttura visiva della scena** più che al suo significato nominale: due frame risultano vicini nello spazio embedding se la scena è disposta in modo simile — stessa postura del soggetto, stesso orientamento del corpo, stessa relazione spaziale soggetto–oggetto — indipendentemente da come sarebbe descritta a parole.

## 2.3 Punti di forza per l'estrazione di feature in questo contesto

**Sensibilità alla configurazione spaziale — dati aggiornati e corretti.** Il CLS token di DINOv3 preserva implicitamente più informazione sulla struttura geometrica della scena rispetto a CLIP.

Con i dati completi su 157 classi, DINOv3 supera CLIP su **51/157 classi** al best checkpoint per-class, ma il pattern non è posturale. Le vittorie più nette riguardano **azioni di manipolazione di oggetti visivamente specifici con bassa salienza nel corpus web**:

| Classe | CLIP AP₁₃ | DINOv3 AP@best | Δ |
|--------|----------:|--------------:|---:|
| Opening a refrigerator (c143) | 60.2 | 88.1 | +27.9 |
| Putting a broom somewhere (c099) | 27.4 | 53.0 | +25.6 |
| Holding a vacuum (c137) | 51.3 | 74.2 | +22.9 |
| Putting a blanket (c071) | 18.5 | 38.8 | +20.3 |
| Taking a vacuum (c138) | 20.9 | 37.9 | +17.0 |
| Taking some clothes (c002) | 38.6 | 54.7 | +16.1 |
| Lying on a bed (c134) | 69.9 | 82.8 | +12.9 |

Le classi posturali vanno in senso opposto: *standing to sitting* (DINOv3: 23.2 vs CLIP 60.4), *sneezing* (11.8 vs 17.8), *running* (9.1 vs 18.9) — CLIP vince su tutte.

**Ricchezza rappresentazionale a 1024 dimensioni.** La maggiore dimensionalità offre in linea di principio più gradi di libertà per codificare informazioni visive sottili. In un task multi-label come Charades — dove più azioni possono coesistere nello stesso frame — uno spazio più ampio può in teoria ospitare segnali discriminativi per classi eterogenee senza che si "sovrascrivano" reciprocamente nell'embedding. Questo vantaggio teorico è però condizionato dalla disponibilità di dati sufficienti per addestrare il classificatore downstream senza saturare la capacità dello spazio.

**Potenziale non sfruttato delle feature dense.** Una proprietà strutturale importante di DINOv3 — che nel setup attuale non viene sfruttata — è la qualità delle sue **feature per patch**. A differenza di CLIP, DINOv3 è stato esplicitamente ottimizzato per produrre mappe di feature spazialmente coerenti e semanticamente discriminative a livello locale. Passare a MS-Temba non solo il CLS token ma anche un sottoinsieme dei token di patch (o una loro media pesata per regione) potrebbe recuperare informazioni sulla distribuzione spaziale degli elementi nella scena che il solo CLS non trasmette — in particolare per classi a forte varianza posturale.

## 2.4 Limiti per l'estrazione di feature in questo contesto

**Assenza di prior semantico nominale.** Il limite più rilevante nel confronto con CLIP è che lo spazio di feature DINOv3 non è organizzato attorno a concetti linguistici. Per le classi di Charades la cui discriminazione dipende da un concetto nominale forte e visivamente generico — *cooking* (AP ~5.5 vs 79.0 CLIP), *talking on the phone* (AP ~5.0 vs 75.9), *watching television* (AP ~48.1 vs 61.3) — il backbone non offre alcun vantaggio a priori nella separazione degli embedding. MS-Temba riceve feature il cui spazio non riflette in alcun modo la tassonomia semantica di Charades: classi visivamente molto diverse ma semanticamente vicine, oppure visivamente simili ma semanticamente distanti, possono risultare a distanze arbitrarie nello spazio DINOv3. Il classificatore downstream deve quindi costruire da zero la separazione semantica, con tutto il peso che questo comporta in termini di dati necessari e rischio di overfitting.

**Overfitting accelerato dalla maggiore capacità.** La dimensionalità 1024 rende il classificatore lineare (o qualsiasi testa downstream) molto più suscettibile alla memorizzazione del training set rispetto a una feature 512-dim. Nei dati questo effetto è evidente anche nella configurazione regolarizzata (reg v2): DINOv3 mostra un gap train–val di ~52 pt a ep28 (stop) contro ~70 pt di CLIP a ep13 (best). Il problema è strutturale: in uno spazio 1024-dim, un classificatore lineare ha il doppio dei parametri per classe rispetto a uno 512-dim, e con soli ~50 esempi di test per classe in media la memorizzazione è quasi inevitabile in assenza di regolarizzazione. La configurazione reg v2 (drop=0.05, wd=0.05) mitiga ma non elimina questo problema.

**Mismatch tra capacità del backbone e del modello temporale.** MS-Temba ha ~18M parametri totali — un modello relativamente contenuto. Ricevere in ingresso feature 1024-dim invece di 512-dim raddoppia la dimensione dell'input ma non cambia la capacità del modello temporale: il rischio è che MS-Temba non abbia abbastanza capacità per sfruttare efficacemente le dimensioni aggiuntive, mentre al contempo subisce l'overfitting del classificatore finale a causa della maggiore dimensionalità. Una proiezione apprendibile 1024→512 prima dell'ingresso in MS-Temba potrebbe risolvere questo mismatch, rendendo le due configurazioni comparabili in termini di pressione sul classificatore downstream.

---

# 3. Confronto diretto tra CLIP e DINOv3

CLIP e DINOv3 codificano due tipologie fondamentalmente diverse di informazione visiva, e questa differenza è la chiave interpretativa principale dei risultati su Charades.

### Tabella comparativa sintetica

| Aspetto | CLIP (512-dim) | DINOv3 (1024-dim, reg v2) | Interpretazione |
|---------|:--------------:|:-------------------------:|-----------------|
| **Paradigma di pre-training** | Contrastive image-text | Self-supervised visual distillation | CLIP apprende corrispondenza immagine–testo; DINOv3 apprende regolarità visive senza linguaggio |
| **Ancoraggio semantico** | **Forte** | Assente / implicito | CLIP beneficia di classi facilmente nominabili; DINOv3 non riceve alcun prior linguistico diretto |
| **Sensibilità spaziale locale** | Moderata | **Alta** | DINOv3 è più adatto a cogliere configurazioni di oggetti visivamente specifici |
| **Granularità della rappresentazione** | Globale-semantica | Densa-strutturale | CLIP privilegia il significato complessivo; DINOv3 la disposizione dei dettagli visivi |
| **Dimensione delle feature** | 512 | 1024 | DINOv3 offre maggiore capacità, ma con più rischio di overfitting |
| **Robustezza su classi semanticamente descrittive** | **Alta** | Moderata | CLIP è favorito quando il nome coincide con concetti frequenti nel linguaggio naturale |
| **Robustezza su classi posturali/corporee** | Moderata | **Bassa** | Con i dati reali DINOv3 non eccelle sulle classi posturali |
| **Classi AP ≥ 60** | **22** | 5 | CLIP produce più classi forti |
| **Classi AP < 20** | 49 | **~90** | DINOv3 reg v2 è debole su ~57% delle classi |
| **Best checkpoint** | ep13 | ep13 | = (stesso per entrambi) |
| **Regolarizzazione** | nessuna (orig) | drop=0.05, wd=0.05 (reg v2) | DINO richiede regolarizzazione; CLIP no |
| **Stabilità post-picco** | Moderata (−3.7 a ep49) | **Migliorata** (−4.0 a ep28 vs −6.6 senza reg) | Reg v2 contiene il crollo di DINOv3 |
| **Vantaggio principale** | Classi semanticamente riconoscibili | Azioni manipolazione oggetti specifici (frigorifero, aspirapolvere, scopa) | Pattern diverso da quanto ipotizzato inizialmente |
| **Limite principale** | Verbi direzione, azioni posturali | Assenza prior semantico, nessun vantaggio posturale reale | Debolezze parzialmente complementari |

### Discussione comparativa

Dal punto di vista del **segnale di feature**, CLIP e DINOv3 rappresentano due estremi quasi complementari. CLIP produce una rappresentazione centrata sul significato linguistico della scena: due immagini sono vicine nello spazio embedded se potrebbero essere descritte con le stesse parole. DINOv3 produce una rappresentazione centrata sulla struttura visiva: due immagini sono vicine se la disposizione degli elementi nella scena è simile. In Charades — un dataset con molte azioni domestiche facilmente verbalizzabili — questa differenza si riflette direttamente nella distribuzione degli AP per classe.

I risultati quantitativi mostrano che **CLIP domina in termini di copertura complessiva**: best-mAP globale 32.40 vs 25.21 (gap +7.19), 22 classi con AP ≥ 60 contro 5, e solo 49 classi con AP < 20 contro ~90 di DINOv3 su 157 classi. Questo pattern è coerente con un dataset in cui la maggior parte delle classi è definita da combinazioni oggetto–azione facilmente nominabili (*holding a phone*, *cooking*, *watching television*) per cui il prior semantico di CLIP offre un vantaggio strutturale.

**DINOv3** mostra vantaggi netti su **51/157 classi** al best-per-class, ma il pattern reale è diverso dall'ipotesi originale: le vittorie maggiori riguardano azioni di manipolazione con oggetti visivamente specifici — *opening a refrigerator* (+27.9), *putting a broom* (+25.6), *holding a vacuum* (+22.9) — non classi posturali.

**Sull'effetto della regolarizzazione per DINOv3.** Il passaggio dalla configurazione originale (senza reg, mAP 25.44) alla reg v2 (drop=0.05, wd=0.05, mAP 25.21) introduce una perdita minima sul best checkpoint (−0.23 mAP) in cambio di una curva post-picco più stabile: il crollo da best a stop si riduce da −6.6 pt (orig, ep13→ep49) a −4.0 pt (reg v2, ep13→ep28). La configurazione reg v2 è quindi preferibile come baseline definitiva perché offre risultati più affidabili e confrontabili con futuri esperimenti.

### Sintesi interpretativa

Nel complesso, **CLIP costituisce il backbone più efficace come scelta generale su Charades**, grazie alla sua maggiore compatibilità con classi semanticamente descrittive, alla migliore copertura delle categorie e a una minore tendenza all'overfitting. **DINOv3** mantiene un ruolo come backbone complementare per le ~51 classi dove il suo vantaggio per-classe è reale (oggetti visivamente specifici: frigorifero, aspirapolvere, scopa). Il routing ibrido rimane percorribile; l'impatto sul Full-val-MAP è difficile da stimare con precisione prima di testarlo, ma le classi DINOv3-winning rappresentano ~33% delle classi con frequenza media.

---

## 4. Configurazione di Training

### 4.1 CLIP (configurazione originale — baseline definitiva)

| Parametro | Valore |
|-----------|--------|
| Dataset | Charades v1 (157 classi) |
| Training split | 7985 video |
| Test split | 1863 video |
| Epoche | 50 (0–49) |
| Batch size | 5 |
| Num clips | 256 |
| Ottimizzatore | AdamW (lr=5e-4, wd=0.01) |
| Scheduler | Cosine (warmup=5 ep, min_lr=1e-5) |
| CLIP feat dim | 512 |
| Dropout | 0.0 |
| Drop-path | 0.0 |
| Parametri totali | ~18M |
| Loss | BCE multi-label + Diversity Loss (β=0.05) |
| Seed | 0 |
| Early stopping | No |

### 4.2 DINOv3 (configurazione reg v2 — baseline definitiva)

La configurazione DINOv3 ha richiesto una serie di esperimenti di regolarizzazione per identificare il setup ottimale. Tre round sono stati condotti:

| Configurazione | drop | drop_path | weight_decay | Best mAP | Stop ep | Note |
|----------------|:----:|:---------:|:------------:|:--------:|:-------:|------|
| orig (no reg) | 0.0 | 0.0 | 0.01 | 25.44 | 50 | Crollo severo post-ep13 (−6.6 mAP) |
| reg v1 | 0.2 | 0.1 | 0.05 | 24.56 | 26 | Dropout troppo aggressivo |
| **reg v2 (definitiva)** | **0.05** | **0.05** | **0.05** | **25.21** | **28** | Quasi parità sul best, curva stabile |

**Parametri finali reg v2**:

| Parametro | Valore |
|-----------|--------|
| Dataset | Charades v1 (157 classi) |
| DINOv3 feat dim | 1024 |
| Dropout | 0.05 |
| Drop-path | 0.05 |
| Weight decay | 0.05 |
| Early stop patience | 15 |
| Min delta | 0.01 |
| Early stop epoch | 28 |
| Tutti gli altri parametri | Identici a CLIP (sopra) |

**Nota architetturale**: le modifiche di regolarizzazione hanno richiesto interventi al codice di `models_MSTemba.py`: la `LinearProjection` è stata estesa con un parametro `drop_rate` propagato a tutte le istanze inter-blocco, e l'`InputProjection` è stata convertita da `nn.Linear` a `nn.Sequential(Linear, LayerNorm, GELU, Dropout)` per garantire che il dropout agisca anche sull'ingresso del modello — il punto critico per la feature 1024-dim.

---

## 5. Risultati Globali

### 5.1 MAP per Checkpoint

| Metrica | CLIP | DINOv3 reg v2 | Δ |
|---------|:----:|:-------------:|:-:|
| Best epoch | **13** | **13** | = |
| Full-val-MAP (best ep) | **32.40** | 25.21 | +7.19 |
| Sampled-val-MAP (best ep) | **33.43** | 25.82 | +7.61 |
| Train-MAP @ best ep | 40.43 | 33.08 | — |
| Full-val-MAP @ stop ep | 28.73 (ep49) | 21.22 (ep28) | — |
| Gap train–val @ best ep | +8.0 pt | +7.9 pt | ≈ |
| Gap train–val @ stop ep | ~70.8 pt (ep49) | ~56.4 pt (ep28) | — |

### 5.2 Distribuzione AP per Classe (Best Checkpoint)

| Fascia AP | CLIP ep13 | DINOv3 reg v2 ep13 |
|-----------|:---------:|:------------------:|
| ≥80 (Eccellente) | 0 | 1 |
| 60–79 | 22 | 4 |
| 40–59 | 30 | 10 |
| 20–39 | 56 | 41 |
| <20 (Bassa) | 49 | ~101 |

---

## 6. Curve di Training Completo

## 6.1 Curve di Training CLIP (Epoche 0–49)

Tutti i valori estratti dal `training_CLIP.log` — EMA model, configurazione originale.

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

> **Nota**: il val MAP raggiunge il picco a ep13 (32.40) poi scende monotonicamente. Il train MAP continua a salire fino a 99.52. Il modello entra in overfitting massiccio dopo ep20. La causa è strutturale — 7985 video con 157 classi sbilanciate non sono sufficienti per un modello da ~18M parametri con regolarizzazione zero.

## 6.2 Curve di Training DINOv3 reg v2 (Epoche 0–28)

Valori estratti dal `metrics_per_epoch.csv` della configurazione `dinov3_vitl16_reg_v2/seed0`. Il training si è fermato a ep28 per early stopping (patience=15, min_delta=0.01).

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

> **Nota**: la configurazione reg v2 mostra il picco a ep13 (25.21), identico all'originale. Il gap train–val a ep28 (stop) è ~52 pt — inferiore ai ~79 pt dell'originale a ep49, confermando l'effetto stabilizzante della regolarizzazione. Il crollo post-picco è contenuto: −4.0 mAP in 15 epoche vs −6.6 mAP in 36 epoche dell'originale. La regolarizzazione non ha modificato la posizione del best checkpoint (ep13 in tutti i run) ma ha limitato il deterioramento successivo.

---

## 7. Tabella Completa per Classe (157 classi)

**Legenda colonne**:
- **Tr**: occorrenze nel training set; **Te**: nel test set; **Tot**: totale
- **AP₀**: AP all'epoca 0 (baseline random); **AP₁₃**: AP al best checkpoint CLIP (ep13)
- **AP_peak**: AP massima CLIP raggiunta in qualsiasi epoca 0–49; **Pk_ep**: epoca del picco CLIP
- **AP₄₉**: AP CLIP all'ultima epoca; **Δ13→49**: drop dal best checkpoint a ep49 (negativo = peggioramento)
- **AP_DINOv3**: AP DINOv3 reg v2 al best checkpoint per-class
- **Winner**: backbone con AP più alta al rispettivo best checkpoint

| Code | Classe | Obj | Verb | Tr | Te | Tot | AP₀ | AP₁₃ | AP_peak | Pk_ep | AP₄₉ | Δ13→49 | AP_DINOv3 | Winner |
|------|--------|-----|------|---:|---:|----:|----:|-----:|--------:|------:|-----:|-------:|----------:|--------|
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

## 8. Analisi Distribuzione AP

### 8.1 Top-20 Classi CLIP (AP₁₃ decrescente)

| Rank | Code | Classe | Tr | AP₁₃ | AP₄₉ | Δ |
|------|------|--------|----|-----:|-----:|--:|
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

### 8.2 Bottom-20 Classi CLIP (AP₁₃ crescente)

| Rank | Code | Classe | Tr | AP₁₃ | AP_peak | Pk_ep | AP_DINOv3 |
|------|------|--------|----|-----:|--------:|------:|----------:|
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

### 8.3 Classi dove DINOv3 supera CLIP

Il pattern reale di DINOv3: azioni di manipolazione con oggetti visivamente specifici (aspirapolvere, frigorifero, scopa) e azioni di put/take su oggetti con bassa salienza nel corpus web di CLIP.

| Code | Classe | AP_CLIP | AP_DINOv3 reg v2 ep13 | AP_DINOv3 best | Vantaggio DINOv3 |
|------|--------|--------:|----------------------:|---------------:|-----------------:|
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

## 9. Analisi per Oggetto (38 categorie)

Media AP₁₃ CLIP per gruppi di classi che condividono lo stesso oggetto (ordinato per mean AP decrescente).

| Obj | Label | #Classi | Mean AP CLIP | Max AP | Min AP | Tot occ. |
|-----|-------|--------:|-------------:|-------:|-------:|---------:|
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

## 10. Analisi per Verbo (33 categorie)

| Verb | Label | #Classi | Mean AP₁₃ | Mean AP₄₉ | Mean Δ | Interpretazione |
|------|-------|--------:|----------:|----------:|-------:|-----------------|
| v002 | cook | 1 | 79.0 | 77.7 | +1.3 | Alta semantica testuale (CLIP) |
| v024 | talk | 1 | 75.9 | 77.8 | -1.8 | |
| v014 | play | 2 | 74.5 | 68.9 | +5.6 | Confuso con hold (v008) |
| v004 | drink | 1 | 65.5 | 60.1 | +5.4 | |
| v032 | work | 2 | 59.9 | 51.5 | +8.4 | Confuso con v018 (sit) e v008 (hold) |
| v010 | lie | 3 | 57.3 | 54.0 | +3.3 | Postura orizzontale → buona generalizzazione |
| v018 | sit | 7 | 55.2 | 52.9 | +2.2 | Postura statica → facile da generalizzare |
| v000 | awaken | 2 | 51.0 | 51.3 | -0.3 | |
| v021 | snuggle | 2 | 50.3 | 51.9 | -1.6 | Contatto fisico caratteristico |
| v008 | hold | 20 | 48.3 | 44.5 | +3.8 | Oggetto prominente in mano → CLIP molto forte |
| v031 | watch | 6 | 48.1 | 45.4 | +2.7 | Sguardo verso target → bassa varianza |
| v029 | walk | 1 | 43.8 | 38.8 | +5.0 | |
| v003 | dress | 2 | 41.9 | 36.3 | +5.5 | |
| v007 | grasp | 1 | 40.4 | 29.6 | +10.8 | |
| v005 | eat | 3 | 38.3 | 32.6 | +5.6 | |
| v028 | undress | 2 | 36.6 | 35.9 | +0.6 | |
| v022 | stand | 2 | 35.0 | 18.5 | +16.5 | Transizione posturale → instabile |
| v026 | tidy | 8 | 34.5 | 32.7 | +1.8 | |
| v012 | open | 8 | 34.3 | 29.0 | +5.2 | |
| v019 | smile | 3 | 33.8 | 27.2 | +6.6 | |
| v006 | fix | 5 | 29.4 | 22.4 | +7.0 | Richiede comprensione meccanica |
| v013 | photograph | 1 | 26.4 | 23.3 | +3.1 | Confuso con hold phone (c015) |
| v009 | laugh | 3 | 25.5 | 27.0 | -1.5 | Espressione facciale → scarsa |
| v023 | take | 19 | 23.2 | 18.3 | +4.9 | |
| v001 | close | 7 | 23.1 | 18.4 | +4.7 | |
| v016 | put | 20 | 22.4 | 18.0 | +4.4 | |
| v030 | wash | 8 | 22.1 | 20.7 | +1.4 | Azione manuale ripetuta → riconoscibile |
| v015 | pour | 1 | 20.9 | 20.0 | +0.9 | |
| v017 | run | 1 | 18.9 | 19.6 | -0.8 | |
| v020 | sneeze | 1 | 17.8 | 21.2 | -3.4 | |
| v025 | throw | 11 | 10.2 | 7.5 | +2.7 | Gesto breve → poche occorrenze, raro nel corpus |
| v011 | make | 1 | 7.8 | 8.2 | -0.4 | Rarissimo (solo sandwich) |
| v027 | turn | 2 | 5.5 | 4.6 | +0.9 | Richiede stato pre/post → problematico |

---

## 11. Coppie di Classi Confondibili

### 11.1 Coppie con Stesso (Oggetto, Verbo) — Ambiguità Strutturale

| Coppia | Classe A | Classe B | AP_A | AP_B | Problema |
|--------|----------|----------|-----:|-----:|---------|
| c010/c011 | Sitting on a table | Sitting at a table | 3.8 | 75.5 | c010 vs c011: stesso codice (o033,v018) — 'sitting ON table' vs 'sitting AT table'. Distinzione puramente preposizionale, impossibile da feature visive. |
| c104/c105 | Turning on a light | Turning off a light | 8.7 | 2.3 | Stesso codice (o021,v027) — richiede comprensione dello stato luce prima/dopo. |
| c043/c044 | Taking a box from somewhere | Taking something from a box | 18.3 | 21.3 | Stesso codice (o005,v023) — direzione dell'azione ambigua. |

### 11.2 Coppie Visivamente Simili (Oggetto Condiviso)

| Code A | Code B | Classe A | Classe B | AP_A | AP_B | Oggetto comune |
|--------|--------|----------|----------|-----:|-----:|---------------|
| c006 | c008 | Closing a door | Opening a door | 30.3 | 39.8 | door |
| c097 | c141 | Walking through a doorway | Grasping onto a doorknob | 43.8 | 40.4 | doorway/doorknob |
| c015 | c019 | Holding a phone/camera | Talking on a phone/camera | 73.2 | 75.9 | phone |
| c047 | c052 | Holding a laptop | Working/Playing on a laptop | 73.1 | 75.9 | laptop |
| c059 | c123 | Sitting in a chair | Sitting on sofa/couch | 73.6 | 59.9 | sit |
| c098 | c102 | Holding a broom | Tidying up with a broom | 75.4 | 63.0 | broom |
| c132 | c131 | Watching television | Laughing at television | 61.3 | 11.2 | tv |
| c133 | c134 | Awakening in bed | Lying on a bed | 49.0 | 69.9 | bed |

---

## 12. Struttura Compositiva: Mapping Oggetto × Verbo

### 12.1 Oggetti con Maggior Mean AP CLIP

| Rank | Oggetto | #Classi | Mean AP₁₃ |
|------|---------|--------:|----------:|
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

### 12.2 Oggetti con Minor Mean AP CLIP

| Rank | Oggetto | #Classi | Mean AP₁₃ | Motivo |
|------|---------|--------:|----------:|--------|
| 1 | shoe (o031) | 6 | 23.9 | |
| 2 | box (o005) | 7 | 20.7 | |
| 3 | shelf (o030) | 2 | 20.5 | |
| 4 | towel (o035) | 6 | 18.5 | |
| 5 | picture (o026) | 5 | 14.4 | Bassa frequenza, alta varianza |
| 6 | medicine (o022) | 2 | 13.0 | Oggetto minuto, difficile da localizzare |
| 7 | light (o021) | 3 | 11.8 | Luce non distinguibile on/off |

### 12.3 Coppie (Oggetto, Verbo) Strutturalmente Duplicate

| Coppie | Classi | AP_A | AP_B | Divario train | Analisi |
|--------|--------|-----:|-----:|:-------------:|---------|
| (o033, v018) | c010 vs c011 | 3.8 | 75.5 | 46 vs 728 | Sitting *on* vs *at* a table — impossibile da features visive. Il divario AP è interamente spiegato dalla frequenza. |
| (o021, v027) | c104 vs c105 | 8.7 | 2.3 | 207 vs 189 | Turning *on* vs *off* a light — richiede comprensione dello stato luce. |
| (o005, v023) | c043 vs c044 | 18.3 | 21.3 | 180 vs 154 | Taking *a box from somewhere* vs *something from a box* — direzione ambigua. |

---

## 13. Classi con Ambiguità Strutturale o Dataset Issue

| Code | Classe | Problema | Train | Test | AP_CLIP | AP_DINOv3 |
|------|--------|----------|------:|-----:|--------:|----------:|
| c010 | Sitting on a table | Strutturalmente identica a c011 (stesso mapping). AP perpetuamente bassa per rarità. | 46 | 16 | 3.8 | 8.5 |
| c104 | Turning on a light | Identico mapping a c105. AP near-zero per entrambe. | 207 | 73 | 8.7 | 7.5 |
| c105 | Turning off a light | Identico mapping a c104. AP near-zero per entrambe. | 189 | 50 | 2.3 | 7.9 |
| c101 | Throwing a broom somewhere | Solo 33 occorrenze totali. AP mai sopra 2.3. | 25 | 8 | 1.5 | 9.1 |
| c064 | Throwing food somewhere | 74 occorrenze totali. AP sempre < 5. | 52 | 22 | 1.9 | 2.8 |
| c085 | Laughing at a picture | 43 occorrenze. Emozione sottile. | 28 | 15 | 13.0 | 4.1 |
| c031 | Throwing a book somewhere | 79 occorrenze. Azione rarissima. | 58 | 21 | 2.7 | 8.1 |
| c045 | Throwing a box somewhere | 40 occorrenze. Collapse a ep49: AP = 0.4. | 34 | 6 | 8.7 | 18.1 |
| c060 | Standing on a chair | 53 occorrenze. Collapse estremo da 33.2 (ep13) a 3.4 (ep49). | 46 | 7 | 33.2 | 1.3 |
| c141 | Grasping onto a doorknob | Split anomalo: 506 train ma solo 20 test. | 506 | 20 | 40.4 | 37.1 |
| c136 | Fixing a vacuum | Ratio invertito: 143 test vs 35 train. | 35 | 143 | 20.3 | 1.1 |

---

## 14. Raccomandazioni e Sviluppi Futuri

### 14.1 Per le Classi a Bassa AP
- **Classi duplicate (c010/c011, c104/c105, c043/c044)**: merging o esclusione dall'evaluation, oppure aggiunta di un modulo state-detection dedicato.
- **Classi rarissime (c101, c064, c085, tot < 80)**: over-sampling, synthetic augmentation, o esclusione da mAP per un'analisi più onesta delle capacità reali del modello.
- **Verbi di stato (v027: turn on/off)**: richiedono un modulo temporale dedicato che confronti il frame iniziale e finale dell'azione.

### 14.2 Per DINOv3

- **Dense features (priorità alta)**: nel setup attuale viene usato solo il CLS token. Le patch features di DINOv3 — il suo reale vantaggio competitivo — non vengono sfruttate. L'estrazione e l'integrazione di mean-patch o attention-pooled patch features è il passo successivo più promettente per DINOv3 (cfr. sezione 14.4).
- **Feature projection**: ridurre 1024→512 con un layer di proiezione trainabile prima del backbone MS-Temba per allineare la capacità del modello a quella di CLIP.
- **Hybrid routing**: usare CLIP per le classi semanticamente descrittive e DINOv3 per le classi dove il vantaggio reale è confermato (c143 +27.9, c099 +25.6, c137 +22.9, c071 +20.3, c138 +17.0). **Sconsigliato** il routing verso DINOv3 per c151, c153, c154, c150, c059: con i dati reali CLIP vince su tutte queste classi.

### 14.3 Architettura

- **Two-head compositional classifier**: aggiungere un secondo classificatore (oggetto + verbo separati) in parallelo al classificatore globale, sfruttando il mapping compositivo di Charades.
- **Per-verb temporal modules**: verbi come v025 (throw), v027 (turn) e v022 (stand) richiedono analisi delle sequenze temporali corte — considerare attention locale più densa su questi.

### 14.4 Passi Successivi Pianificati

In ordine di priorità per le fasi successive del progetto:

1. **DINOv3 Dense Features (Fase 2A)**: estrarre le patch features di DINOv3 (mean-pooling o attention-pooling sui 256 token di patch) per ottenere una rappresentazione spazialmente strutturata. Stimare l'impatto su mAP tramite ablation con `in_feat_dim=1024` (CLS) vs `in_feat_dim=2048` (CLS+mean_patch). La modifica impatta `dinov3_feature_extractor.py`, il dataloader e la pipeline di input di MS-Temba.

2. **Skeleton Features SCDNet (Fase 2B)**: integrare le feature scheletriche pre-estratte disponibili su ABACA (`Charades_SCDNet_features2.zip`). Il prerequisito è l'allineamento temporale (fps), dettagliato nella Sezione 15. La strategia di fusione consigliata è la gated fusion (cfr. Sezione 15.4.3b).

3. **Survey stato dell'arte (Fase 3)**: analisi sistematica della letteratura su combinazione di feature visive e scheletriche per TAD/TAR, con focus su: strategie di fusione modale, robustezza a occlusioni, dataset benchmark (NTU RGB+D, Charades, Toyota Smarthome), modelli recenti (SkeletonBERT, MotionBERT, HiCo, PoseFormer, UNIK).

---

# 15. Integrazione di Feature Skeleton con CLIP

> **Focus**: questa sezione analizza l'inserimento di feature skeleton nel pipeline MS-Temba+CLIP, motivato dai limiti strutturali del token CLS di CLIP sulla codifica del movimento corporeo, e stima l'impatto atteso sulle performance.

## 15.1 Motivazione: il gap posturale di CLIP

Come analizzato nelle sezioni precedenti, il token CLS di CLIP comprime ogni frame in un singolo vettore 512-dim ottimizzato per la compatibilità testuale. Questa compressione è lossy rispetto al segnale corporeo: configurazione degli arti, traiettoria del baricentro (CoM), velocità angolari delle giunture, orientamento relativo segmenti — tutte informazioni che il training contrastivo non ha incentivo a preservare esplicitamente.

I dati lo confermano: le 26 classi identificate come candidate primarie per le skeleton features hanno una **media AP CLIP di 35.0**, sostanzialmente allineata all'overall (32.4), ma con una distribuzione molto più ampia: 7 classi sotto AP 20 (molte legate a verbi *throw*, *turn*, *run*) e un cluster di classi posturali che, nonostante siano "comprensibili" da CLIP semanticamente, restano basse proprio per l'ambiguità verbo-direzionale che il CLS token non risolve.

Il segnale mancante è in larga parte **cinematico**: non basta sapere che una persona è in piedi vicino a una borsa — bisogna sapere se il polso si sta allontanando dall'oggetto (put) o avvicinandosi (take), se il CoM si abbassa (sitting down) o si alza (standing up), se il tronco si inclina in avanti con le ginocchia che si flettono (sneezing) o rimane verticale.

## 15.2 Tipologie di feature skeleton

Esistono tre livelli di rappresentazione utilizzabili, con diverso costo computazionale e informazione disponibile:

**Livello 1 — Keypoints 2D** (es. OpenPose, MediaPipe Pose): coordinate (x, y) di 17–33 landmark corporei per frame. Semplici da estrarre, non richiedono profondità, ma perdono informazione di profondità e sono soggetti a occlusioni. Appropriati come baseline a basso costo.

**Livello 2 — Keypoints 3D** (es. MotionBERT, MixSTE): coordinate (x, y, z) stimate da monoculare o da video. Catturano l'orientamento volumetrico del corpo e le traiettorie 3D degli arti — molto più informative per le azioni di reach/grasp e per le transizioni posturali.

**Livello 3 — Feature semantiche da Graph Neural Network** (es. ST-GCN, PoseConv3D): rappresentazione del corpo come grafo dove i nodi sono le giunture e gli archi sono i segmenti ossei. La GNN apprende dinamiche di movimento direttamente sullo spazio del grafo — cattura relazioni di alto livello come "la mano si avvicina al piede" o "le ginocchia si piegano mentre il bacino scende".

**Nota su SCDNet**: le feature skeleton disponibili su ABACA (`Charades_SCDNet_features2.zip`) sono estratte con SCDNet (Skeleton-aware Compositional Dynamic Network), un backbone che produce embedding GNN di livello 3. Il loro fps di campionamento è da verificare prima dell'integrazione per garantire l'allineamento con le feature visive a 24fps.

## 15.3 Classi che beneficerebbero maggiormente

### 15.3.1 Gruppo A — Verbi di stato e direzione (guadagno atteso: alto)

| Code | Classe | AP CLIP | Gap da risolvere | Feature skeleton rilevante |
|------|--------|--------:|------------------|---------------------------|
| c105 | Turning off a light | 2.3 | Stato luce pre/post non visibile nel CLS | Traiettoria polso + posizione finale rispetto al muro |
| c104 | Turning on a light | 8.7 | Idem | Idem (direzione opposta) |
| c024 | Throwing a bag somewhere | 7.1 | Gesto balistico — breve e veloce | Accelerazione angolare spalla + release point |
| c126 | Throwing something on floor | 11.4 | Idem | Rotazione tronco + traiettoria polso discendente |
| c057 | Taking off shoes | 16.5 | Bend-down non distinguibile da put-on | Angolo ginocchio + contatto piede/mano |
| c055 | Putting on shoes | 33.8 | Idem (direzione opposta) | Idem |

### 15.3.2 Gruppo B — Transizioni posturali (guadagno atteso: medio-alto)

| Code | Classe | AP CLIP | Gap da risolvere | Feature skeleton rilevante |
|------|--------|--------:|------------------|---------------------------|
| c153 | Someone is sneezing | 17.8 | Flessione improvvisa tronco + head snap | Velocità angolare tronco + inclinazione testa |
| c150 | Someone is running | 18.9 | Gait pattern — CLIP CLS non codifica periodicità | Frequenza passo, simmetria laterale, elevazione piede |
| c154 | Someone is standing up | 36.8 | Risalita CoM da posizione seduta | CoM Δy positivo, estensione ginocchio/anca |
| c151 | Standing to sitting | 60.4 | Discesa CoM verso sedia | CoM Δy negativo, flessione ginocchio/anca |
| c133 | Awakening in bed | 49.0 | Risalita da posizione orizzontale | Angolo busto rispetto al piano del letto |
| c060 | Standing on a chair | 33.2 | Elevazione CoM più alta del normale | CoM assoluto alto + gambe estese |

### 15.3.3 Gruppo C — Disambiguazione put/take e reach direction (guadagno atteso: medio)

| Code | Classe | AP CLIP | Gap da risolvere | Feature skeleton rilevante |
|------|--------|--------:|------------------|---------------------------|
| c022 | Putting a bag somewhere | 40.6 | Release trajectory | Distanza mano-oggetto crescente (put) vs decrescente (take) |
| c023 | Taking a bag from somewhere | 25.4 | Grasp trajectory | Idem (direzione opposta) |
| c006 | Closing a door | 30.3 | Push vs pull | Polso in avvicinamento alla porta (push) |
| c008 | Opening a door | 39.8 | Idem | Polso che tira indietro (pull) |
| c043 | Taking a box from somewhere | 18.3 | Direzione azione ambigua | Hand-to-object vs object-to-hand trajectory |
| c044 | Taking something from a box | 21.3 | Idem speculare | Idem |

### 15.3.4 Gruppo D — Azioni corporee globali (guadagno atteso: basso-medio)

| Code | Classe | AP CLIP | Contributo skeleton |
|------|--------|--------:|---------------------|
| c097 | Walking through a doorway | 43.8 | Orientamento corpo rispetto all'apertura |
| c149 | Someone is laughing | 52.4 | Pattern scossa spalle/tronco |
| c152 | Someone is smiling | 49.5 | Solo con joint facciali (limitato) |
| c155 | Someone is undressing | 56.7 | Traiettoria braccia verso l'esterno del corpo |
| c148 | Someone is dressing | 50.0 | Traiettoria braccia verso l'interno del corpo |

## 15.4 Strategie di fusione con CLIP

La fusione tra feature CLIP e feature skeleton in MS-Temba non è una scelta architetturale banale: il punto in cui le due modalità vengono integrate determina quanta informazione cross-modale il modello riesce a sfruttare, quanto è suscettibile all'overfitting e quanto è costoso a training e inference.

### 15.4.1 Early Fusion
Concatenazione immediata a livello di input: `[CLIP_CLS (512) ‖ skel (D)] → Linear proj → 512`. Semplice da implementare ma soggetta a *modality dominance*: il gradiente tende a ignorare lo stream skeleton su classi dove CLIP è già forte (~106/157). Usare come baseline ablation per verificare che skeleton sia informativo.

### 15.4.2 Late Fusion
Due MS-Temba separati → logit fusion via gate apprendibile. Massima specializzazione per stream, ma nessuna interazione cross-modale durante l'elaborazione. Costo ×2 parametri. Usare come confronto ablation.

### 15.4.3 Mid-Level Fusion (Sum o Gated)
Proiezioni separate per ciascun stream → fusione a dimensione comune `d=512`.

**Sum fusion**: `h = Linear_CLIP(clip) + Linear_Skel(skel)`. Evita la dominance CLIP ma non apprende interazioni tra modalità. Baseline mid-level.

**Gated fusion** (consigliata):
```
g = σ(Linear_gate([f_CLIP; f_Skel]))    # gate ∈ (0,1)^d
h = g ⊙ f_CLIP + (1−g) ⊙ f_Skel
```
Il gate apprende per ogni frame quanto pesare ciascuna modalità su ciascuna dimensione. Coerente con il meccanismo di selezione adattiva dei Temba-Blocks. Basso costo (+~2% parametri). **Setup raccomandato per il primo esperimento produttivo.**

### 15.4.4 Dual-Branch Fusion
Due branch parallele di Temba-Blocks → fusione nell'MS-Fuser. Massimo potere espressivo, modella scale temporali diverse per ciascuna modalità. Costo ~×2. Da usare solo se gated fusion è già confermata e si dispone di regolarizzazione aggressiva (alto rischio overfitting su Charades con 7985 video).

### 15.4.5 Cross-Attention Fusion
`Q = proj_Q(CLIP_CLS)`, `K = V = proj_KV(skel)`. Il meccanismo di attention seleziona quali dimensioni cinematiche sono rilevanti dato il contesto semantico corrente. Semantica esplicita: CLIP interroga skeleton — "dato che vedo una borsa, recupera dalla feature skeleton le informazioni sulla traiettoria della mano". Superiore alla gated fusion per interazione contestuale, ma più costoso e suscettibile a overfitting.

### 15.4.6 Confronto riassuntivo

| Strategia | Interazione cross-modale | Costo parametri | Rischio overfitting | Consigliata |
|-----------|:------------------------:|:---------------:|:-------------------:|:-----------:|
| **Early Fusion** | ✗ (dominance CLIP) | +~1% | Basso | Baseline ablation |
| **Late Fusion** | ✗ (nessuna) | ×2 | Basso | Confronto ablation |
| **Sum Fusion** | Parziale (additiva) | +~1% | Basso | Baseline mid-level |
| **Gated Fusion** | ✓ (learnable, simmetrica) | +~2% | Basso–Medio | **Setup raccomandato** |
| **Dual-Branch** | ✓ (multi-scala, separata) | ~×2 | Alto | Solo con pre-training |
| **Cross-Attention** | ✓ (direzionale CLIP→Skel) | +~15% | Medio–Alto | Setup finale avanzato |

**Progressione consigliata**: Early → Gated → Cross-Attention.

## 15.5 Feature skeleton: rappresentazione e preprocessing

**Normalizzazione obbligatoria** prima della fusione con CLIP:
1. Sottrarre la posizione del mid-hip da tutti i joint (invarianza alla posizione camera)
2. Dividere per la lunghezza media del torso (invarianza alla scala del soggetto)
3. Campionare alla stessa frequenza dei frame CLIP (24fps) tramite interpolazione se necessario

**Allineamento fps per SCDNet**: le feature su ABACA sono state estratte a fps potenzialmente diversi da 24. Prima dell'integrazione, ispezionare il formato: `python -c "import zipfile; z=zipfile.ZipFile('Charades_SCDNet_features2.zip'); print(z.namelist()[:5])"` e confrontare il numero di frame con le feature CLIP corrispondenti.

**Fallback skeleton mancante**: non tutti i frame producono rilevamenti affidabili (occlusioni, multi-persona). Con gated fusion, forzare `g = 1` (peso totale su CLIP) quando il confidence score skeleton è sotto soglia.

## 15.6 Stima dell'impatto atteso su mAP

### 15.6.1 Stima per gruppo

| Gruppo | #Classi | AP CLIP medio | Guadagno atteso (gated) | Guadagno atteso (cross-att) |
|--------|:-------:|:-------------:|:-----------------------:|:---------------------------:|
| A — Verbi stato/direzione | 6 | 13.3 | +10–14 AP/classe | +15–20 AP/classe |
| B — Transizioni posturali | 6 | 35.8 | +8–12 AP/classe | +12–18 AP/classe |
| C — Disambiguazione reach | 6 | 29.2 | +6–10 AP/classe | +10–15 AP/classe |
| D — Azioni corporee globali | 8 | 50.7 | +2–5 AP/classe | +4–8 AP/classe |

### 15.6.2 Stima globale ΔmAP

| Scenario | Strategia | Guadagno medio/classe | Classi beneficiate | ΔmAP stimato |
|----------|-----------|:---------------------:|:------------------:|:------------:|
| **Conservativo** | Gated, skel 2D | +6 AP | 26 | **+1.0** |
| **Base** | Gated, skel 3D | +11 AP | 26 | **+1.8** |
| **Ottimistico** | Cross-Att, GNN | +17 AP | 26 | **+2.8** |
| **Ideale** | Cross-Att + multi-scale | +24 AP | 26 | **+4.0** |

### 15.6.3 Distribuzione AP stimata post-integrazione skeleton

| Fascia AP | CLIP attuale | CLIP + Skeleton (base) | CLIP + Skeleton (ottimistico) | Δ classi (base) |
|-----------|:-----------:|:---------------------:|:-----------------------------:|:---------------:|
| ≥ 60 | 22 | ~24 | ~27 | +2 |
| 40–59 | 30 | ~33 | ~35 | +3 |
| 20–39 | 56 | ~57 | ~55 | +1 |
| < 20 | 49 | ~43 | ~40 | −6 |

## 15.7 Criticità implementative

**Overfitting su classi rare.** Le classi dei Gruppi A e B più beneficiate da skeleton sono spesso le più rare in training: c104/c105 (~200 train), c060 (46 train), c024 (89 train). Aggiungere uno stream skeleton aumenta la capacità del modello senza aumentare i dati di training. Con gated fusion si raccomanda `drop=0.15, drop_path=0.1`.

## 15.8 Priorità di implementazione

1. **Ispezionare le feature SCDNet** già disponibili su ABACA: formato, fps, dimensionalità.
2. **Ablation early fusion** come lower bound: concat (CLIP 512 + skel D → proj 512). Se ΔmAP < +0.2 su Gruppi A e B, il problema è nella qualità delle feature skeleton.
3. **Ablation sum fusion vs gated fusion**: se gated supera sum di > +0.3 mAP, le interazioni learnable sono importanti.
4. **Upgrade a feature 3D** (MotionBERT o equivalente) se gated + 2D mostra guadagno confermato.
5. **Cross-attention CLIP→Skeleton** solo se i passi precedenti confermano il valore skeleton e si è aggiunta regolarizzazione sufficiente.
6. **In parallelo**: auxiliary supervision cinematica per Gruppo A (predire direzione motion vector del polso). Non aggiunge costi a inference time.

---
