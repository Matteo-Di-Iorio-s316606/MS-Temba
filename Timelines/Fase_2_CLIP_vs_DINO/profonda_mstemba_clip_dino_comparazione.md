# MS-Temba su Charades: Analisi Completa CLIP vs DINOv3

> **Dataset**: Charades v1 — 157 classi di azioni domestiche, 7985 video di training, 1863 di test  
> **Modello**: MS-Temba (Multi-Scale Temporal Mamba)  
> **Backbone A**: CLIP ViT (512-dim features) — Best checkpoint **ep13**, Full-val-MAP **32.40**, Sampled-val-MAP **33.43**  
> **Backbone B**: DINOv3 ViT-L/16 (1024-dim features, reg v2) — Best checkpoint **ep13**, Full-val-MAP **25.21**, Sampled-val-MAP **25.82**  
> **Run CLIP**: `runs/charades/clip/seed0` — configurazione originale, nessuna regolarizzazione  
> **Run DINOv3**: `runs/charades/dinov3_vitl16_reg_v2/seed0` — drop=0.05, drop_path=0.05, wd=0.05, early stop patience=15 (ep28)  
> **Nota su DINOv3**: i risultati si riferiscono alla configurazione reg_v2, baseline definitiva dopo una serie di esperimenti di regolarizzazione (orig → reg v1 → reg v2). Il run originale DINOv3 (senza reg) mostrava picco identico (mAP 25.44 a ep13) ma crollo severo a ep49 (18.81, −6.6 mAP). La reg v2 stabilizza la curva (−4.0 mAP a ep28) con perdita minima sul best (−0.23 mAP). I dati per-classe del run originale mostravano valori anomali su alcune classi posturali (es. sneezing 67.9, running 54.6) che con reg_v2 non sono confermati: questi erano probabilmente artefatti di overfitting precoce. I dati reg_v2 su tutte 157 classi sono considerati più affidabili e vengono usati in questa analisi.  
> **Data aggiornamento**: 25 marzo 2026

---

## Indice

1. [Architettura CLIP vs DINOv3](#architettura-clip-vs-dinov3)
2. [Configurazione di Training](#configurazione-di-training)
3. [Risultati Globali](#risultati-globali)
4. [Curve di Training Completo](#curve-di-training-completo)
5. [Tabella Completa per Classe (157 classi)](#tabella-completa-per-classe-157-classi)
6. [Analisi Distribuzione AP](#analisi-distribuzione-ap)
7. [Analisi per Oggetto (38 categorie)](#analisi-per-oggetto-38-categorie)
8. [Analisi per Verbo (33 categorie)](#analisi-per-verbo-33-categorie)
9. [Coppie di Classi Confondibili](#coppie-di-classi-confondibili)
10. [Dinamiche di Overfitting: Late Bloomers e Collapse](#dinamiche-di-overfitting-late-bloomers-e-collapse)
11. [Analisi del Picco Reale vs Checkpoint Best](#analisi-del-picco-reale-vs-checkpoint-best)
12. [Struttura Compositiva: Mapping Oggetto × Verbo](#struttura-compositiva-mapping-oggetto--verbo)
13. [Classi con Ambiguità Strutturale](#classi-con-ambiguita-strutturale)
14. [Raccomandazioni e Sviluppi Futuri](#raccomandazioni-e-sviluppi-futuri)

---

# Architettura CLIP vs DINOv3

## 1. CLIP: architettura, principio di apprendimento, punti di forza e limiti

### 1.1 Introduzione

CLIP (*Contrastive Language–Image Pre-training*) è un modello multimodale proposto da OpenAI con l'obiettivo di apprendere rappresentazioni visive trasferibili direttamente da supervisione in linguaggio naturale. L'idea di fondo è sostituire, almeno in parte, la classica supervisione chiusa di tipo ImageNet-style con un segnale molto più aperto: grandi collezioni di coppie immagine–testo provenienti dal web. Nel lavoro originale, CLIP viene addestrato su circa **400 milioni di coppie immagine–testo** e dimostra che un obiettivo contrastivo semplice ma scalabile consente di ottenere rappresentazioni visive capaci di generalizzare bene in regime **zero-shot** su oltre 30 benchmark, inclusi compiti di OCR, classificazione fine-grained, geo-localizzazione e action recognition video. In particolare, il paper riporta che il miglior modello CLIP raggiunge **76.2%** di accuratezza top-1 su ImageNet in zero-shot, eguagliando il ResNet-50 supervisionato originale senza usare i 1.28 milioni di esempi annotati di ImageNet.

L'importanza di CLIP non risiede soltanto nei risultati numerici, ma nel cambiamento di paradigma che introduce. Nei modelli visivi classici, l'encoder è addestrato a discriminare un insieme fisso e chiuso di etichette. In CLIP, invece, il modello apprende a misurare la compatibilità fra immagini e descrizioni testuali; ciò rende il linguaggio naturale l'interfaccia di inferenza e trasforma il testo in uno spazio semantico utilizzabile come classificatore dinamico. Questa proprietà spiega perché CLIP sia diventato un riferimento centrale per zero-shot classification, retrieval e, più in generale, per i foundation models multimodali.

### 1.2 Architettura

Dal punto di vista strutturale, CLIP è composto da due encoder distinti: un **image encoder** e un **text encoder**. L'image encoder può essere una variante di **ResNet** oppure un **Vision Transformer (ViT)**; nel paper originale vengono addestrate più famiglie di modelli, tra cui RN50, RN101, RN50x4, RN50x16, RN50x64, oltre a ViT-B/32, ViT-B/16 e ViT-L/14. Il text encoder è invece un **Transformer** autoregressivo modificato, con tokenizzazione BPE lower-cased, vocabolario di 49,152 token e una lunghezza massima della sequenza fissata a 76 token utili più token speciali. La rappresentazione testuale finale è ricavata dall'attivazione del token finale e poi proiettata linearmente nello spazio multimodale condiviso.

Un dettaglio architetturale rilevante è che CLIP **non** usa una testa di proiezione non lineare in stile SimCLR; gli autori dichiarano di utilizzare soltanto una **proiezione lineare** dalle rappresentazioni unimodali allo spazio di embedding condiviso. Questo punto è importante perché suggerisce che gran parte del potere del modello non derivi da una testa contrastiva complessa, ma dalla qualità degli encoder, dalla scala dei dati e dalla natura della loss. In altri termini, CLIP non costruisce il suo vantaggio su un sofisticato "alignment head", bensì sulla capacità di apprendere una geometria semantica comune tra immagine e testo.

Nel repository ufficiale, l'API riflette esattamente questa fattorizzazione: `encode_image()` produce le feature visive, `encode_text()` produce le feature testuali e la chiamata congiunta restituisce i logit immagine-testo, definiti come **similarità coseno** tra le due rappresentazioni, scalate da una costante. Questa formulazione rende CLIP estremamente pratico: il modello può essere usato sia come estrattore di feature, sia come classificatore zero-shot costruito on the fly a partire da prompt testuali.

### 1.3 Obiettivo di apprendimento

Il cuore di CLIP è una loss **contrastiva simmetrica** applicata su mini-batch di coppie immagine–testo allineate. Dato un batch di N immagini e N testi corrispondenti, il modello calcola tutte le similarità immagine-testo nel batch, ottenendo una matrice N×N. Gli elementi diagonali rappresentano le coppie corrette; tutti gli altri sono negativi impliciti. L'addestramento massimizza la similarità delle coppie reali e minimizza quella delle coppie scorrette tramite una cross-entropy simmetrica sulle due direzioni, immagine→testo e testo→immagine.

Questo schema ha due conseguenze fondamentali. Primo, l'encoder visivo non apprende semplicemente a separare classi discrete, ma a posizionare l'immagine in uno spazio in cui essa sia vicina alle sue descrizioni linguistiche plausibili. Secondo, il linguaggio naturale funge da forma di supervisione estremamente ricca: una didascalia non dice solo "cane", ma può includere attributi, relazioni, azioni, contesto, stile e composizione. Proprio questa ampiezza semantica è una delle ragioni per cui CLIP trasferisce bene a compiti non visti in training.

### 1.4 Meccanismo di inferenza zero-shot

La procedura zero-shot di CLIP è elegante perché riusa direttamente il task di pretraining. In inferenza, si costruisce un insieme di descrizioni testuali corrispondenti alle classi target, le si passa nel text encoder per ottenere un set di embedding e queste embedding vengono usate come pesi impliciti di un classificatore lineare. L'immagine viene codificata dall'image encoder, si misura la similarità coseno con ogni embedding testuale e si applica una softmax. Il paper osserva che questa procedura è formalmente interpretabile come una regressione logistica multinomiale con input e pesi normalizzati e con temperature scaling.

Uno degli aspetti più influenti emersi nel lavoro originale è il ruolo del **prompt engineering**. Gli autori mostrano che usare semplicemente il nome della classe spesso è subottimale; un template come *"A photo of a {label}."* migliora già la performance, e su ImageNet l'uso di prompt più contestualizzati e di ensembling di prompt porta a guadagni sostanziali. Nel paper si riporta che il solo prompt base migliora ImageNet di **1.3 punti**, mentre l'ensemble di prompt multipli migliora di un ulteriore **3.5%**, per un guadagno complessivo vicino a **5 punti** rispetto all'uso del solo nome della classe.

### 1.5 Perché CLIP funziona bene

Il primo grande punto di forza di CLIP è la **generalizzazione zero-shot**. Il modello è stato progettato proprio per evitare che ogni nuovo task richieda fine-tuning supervisionato o una testa classificativa dedicata. Questa proprietà è stata verificata empiricamente su un ampio insieme di benchmark eterogenei.

Il secondo punto di forza è la **ricchezza semantica** della rappresentazione. Poiché l'addestramento costringe le immagini a essere allineate a testi naturali, le feature di CLIP tendono a catturare informazione concettuale, contestuale e lessicale molto più di quanto facciano encoder visivi puramente self-supervised o supervisionati su etichette chiuse. Questo spiega perché CLIP sia particolarmente efficace su classi descrivibili verbalmente in modo naturale e su compiti in cui il lessico delle categorie è vicino alla distribuzione testuale del pretraining.

Il terzo punto di forza è la **flessibilità operativa**. Lo stesso modello può essere impiegato per classificazione zero-shot, retrieval cross-modale, image search, feature extraction e linear probing.

Infine, CLIP mostra segnali interessanti su compiti dinamici e verb-centric. Nel paper gli autori sottolineano che zero-shot CLIP supera un ResNet-50 su due benchmark di action recognition video, con un vantaggio di **14.5%** su Kinetics700 e di **7.7%** rispetto alle feature di ResNet-50 su UCF101.

### 1.6 Debolezze strutturali di CLIP

Il principale limite di CLIP è che la sua forza semantica globale non implica automaticamente una forte sensibilità ai **dettagli spaziali fini**. L'unità di apprendimento è la coppia immagine–testo a livello globale; il modello non è addestrato esplicitamente ad allineare regioni locali a frammenti testuali, né a rappresentare in modo preciso micro-differenze strutturali.

Una seconda debolezza riguarda i task che richiedono **conteggio**, **fine-grained discrimination** o comprensione di stati molto vicini. La model card ufficiale dice esplicitamente che CLIP "currently struggles" in compiti come **fine-grained classification** e **counting objects**. Questo si collega direttamente all'action recognition: distinguere *turning on* da *turning off a light* non richiede soltanto semantica globale, ma comprensione di differenze di stato, causalità locale e talvolta temporalità implicita.

Una terza debolezza è la dipendenza dal **prompt** e dalla **tassonomia di classe**. Il paper mostra chiaramente che le prestazioni possono variare molto a seconda della formulazione testuale, e la model card avverte che l'uso in scenari non accuratamente testati è sconsigliato proprio per l'alta variabilità rispetto alle classi scelte e a come vengono nominate.

Una quarta debolezza è la sensibilità a **bias e fairness issues**. La model card riporta che OpenAI ha osservato disparità significative rispetto a razza e genere in test di denigration risk e che tali disparità possono cambiare in funzione della costruzione delle classi. Inoltre, il dataset di training, derivando in larga parte da dati internet pubblici, riflette in modo sbilanciato popolazioni più connesse al web.

### 1.7 Lettura interpretativa: che tipo di rappresentazione apprende CLIP?

Una buona sintesi concettuale è la seguente: CLIP apprende una rappresentazione visiva che non risponde alla domanda "a quale classe chiusa appartiene questa immagine?", ma a una domanda più generale del tipo **"quale testo plausibile descrive questa immagine?"**. La feature CLIP è quindi, in larga misura, una feature **semantico-linguistica**: tende a privilegiare oggetti, contesto, affordance, composizione globale e concetti che hanno un buon ancoraggio lessicale.

D'altra parte, proprio questa natura rende CLIP meno naturalmente adatto a rappresentare aspetti puramente geometrici, metrici o fisici che il linguaggio descrive raramente con precisione sufficiente. In un confronto con rappresentazioni self-supervised puramente visuali, questo si traduce spesso in una tensione classica: **CLIP vede bene "che cosa" e "in quale contesto", ma non sempre "come è configurato esattamente"**.

### 1.8 Implicazioni specifiche per action recognition

Nel contesto dell'action recognition, CLIP è particolarmente promettente quando l'azione è ben nominabile e semanticamente forte, ad esempio *talking on the phone*, *cooking*, *reading*, *playing guitar*. In questi casi, il linguaggio di pretraining fornisce un prior molto utile: non solo l'oggetto coinvolto, ma anche il contesto tipico dell'azione.

Tuttavia, quando il riconoscimento dipende da **transizioni di stato**, differenze sottili di postura, relazioni mano-oggetto molto fini o dinamiche temporali brevi, CLIP mostra i suoi limiti. Il pretraining originario è infatti centrato su immagini statiche e allineamento globale con testo; non è ottimizzato per modellare evoluzioni temporali o distinguere configurazioni quasi identiche.

### 1.9 Conclusione

CLIP rappresenta uno spartiacque perché dimostra che la supervisione in linguaggio naturale, se scalata correttamente, può produrre rappresentazioni visive generali, trasferibili e sorprendentemente competitive in zero-shot. La sua architettura è semplice: due encoder, uno spazio condiviso, una loss contrastiva. Il suo impatto, però, è profondo: trasforma il linguaggio in classificatore, rende la semantica testuale parte integrante dell'inferenza e inaugura una nuova famiglia di foundation models multimodali.

Allo stesso tempo, CLIP non è un sostituto universale di ogni rappresentazione visiva. Le sue debolezze su conteggio, fine-grained discrimination, dettaglio locale, prompt sensitivity e fairness ne definiscono con chiarezza i confini. In sintesi, CLIP è fortissimo quando il problema è "semanticamente nominabile" e relativamente globale; è meno convincente quando il compito richiede una comprensione visiva precisa, localizzata o strettamente strutturale.

---

## 2. DINOv3: architettura, principio di apprendimento, punti di forza e limiti

### 2.1 Introduzione

DINOv3 rappresenta l'evoluzione più recente della linea DINO di Meta per il **self-supervised visual pretraining** e si propone come un **vision foundation model** generalista, addestrato senza etichette manuali su larga scala. L'obiettivo centrale del lavoro è mostrare che un backbone puramente visivo, se scalato correttamente in termini di dati, modello e ottimizzazione, può produrre rappresentazioni ad alta trasferibilità e soprattutto **dense features** di qualità molto elevata, utili non solo per compiti globali di classificazione ma anche per segmentazione, depth estimation e corrispondenza locale.

Dal punto di vista concettuale, DINOv3 si colloca quasi all'opposto di CLIP. Se CLIP costruisce rappresentazioni visive allineate al linguaggio naturale, DINOv3 apprende una geometria dello spazio visivo **senza testo**, partendo esclusivamente dalla struttura statistica delle immagini. Questo implica che la semantica appresa da DINOv3 è meno "nominale" e meno ancorata a categorie linguistiche esplicite, ma potenzialmente più fedele alla struttura spaziale, alle relazioni locali e alla continuità patch-to-patch dell'immagine.

### 2.2 Architettura generale e famiglia di modelli

Nel repository ufficiale, DINOv3 è rilasciato come una **famiglia di backbone** piuttosto ampia. Sono disponibili modelli **ViT** pre-addestrati su un dataset web-scale chiamato **LVD-1689M**, con taglie che vanno da **ViT-S/16** fino a **ViT-7B/16**, oltre a versioni **ConvNeXt** distilled e a modelli addestrati su dati satellitari (**SAT-493M**).

In questo contesto viene usata la variante **ViT-L/16** con dimensione interna **1024**, che produce embedding di dimensione 1024. Questa maggiore dimensionalità rispetto a CLIP (512) offre in linea di principio più gradi di libertà per codificare informazioni visive sottili, ma aumenta il rischio di overfitting nel downstream a basso numero di campioni.

### 2.3 Principio di apprendimento: self-distillation visiva senza etichette

Il principio di base di DINOv3 resta quello della **self-distillation student-teacher** tipica della famiglia DINO: uno studente apprende a riprodurre target generati da un teacher costruito come media esponenziale dei parametri, evitando l'uso di annotazioni manuali e sfruttando viste multiple dell'immagine.

Il contributo distintivo di DINOv3 è la nuova fase di training chiamata **Gram anchoring**, pensata per correggere la degradazione delle dense features durante training lunghi e molto scalati. L'idea è regolarizzare la **struttura delle similarità tra patch** — la matrice di Gram delle feature locali. In altre parole, ciò che viene preservato non è il valore assoluto di ciascun embedding locale, ma la geometria relativa delle relazioni patch-to-patch. Gli autori mostrano che questa strategia riesce a "riparare" dense features degradate anche se applicata tardi nel training, con benefici evidenti su benchmark come ADE20K.

### 2.4 Natura della rappresentazione appresa

La rappresentazione di DINOv3 è, in prima approssimazione, una rappresentazione **visiva strutturale** più che linguistico-semantica. Mentre in CLIP ogni embedding visivo è ottimizzato per essere compatibile con descrizioni testuali, in DINOv3 l'apprendimento è interamente governato dalla coerenza tra viste e dalla regolarità della geometria visiva interna. Ciò porta a feature che tendono a essere più adatte a compiti dove la qualità della **mappa spaziale** conta molto: segmentazione, corrispondenza densa, stima della profondità, raggruppamento di regioni semanticamente affini.

In termini interpretativi, si può dire che DINOv3 apprende una nozione di somiglianza visiva che privilegia **coerenza spaziale, continuità locale e organizzazione della scena**. Per questo DINOv3 tende a risultare particolarmente competitivo quando il segnale discriminativo è presente nella configurazione stessa dell'immagine, più che nel suo "nome" linguistico.

### 2.5 Punti di forza

- **Dense features di alta qualità**: il vantaggio competitivo più importante di DINOv3 rispetto a CLIP non sta nel token CLS ma nelle patch features. Ogni patch produce un embedding localizzato e spazialmente strutturato — una proprietà che nel setup attuale (CLS-only) non viene sfruttata e rappresenta il principale potenziale non ancora utilizzato.
- **Sensibilità alla configurazione spaziale**: il CLS token di DINOv3, pur non contenendo patch features esplicite, preserva implicitamente più informazione sulla struttura geometrica della scena. I dati confermano vantaggi netti su azioni di manipolazione con oggetti visivamente specifici (frigorifero, aspirapolvere, scopa).
- **Potenziale rappresentazionale a 1024 dimensioni**: in scenari con dati sufficienti, la dimensionalità maggiore offre più gradi di libertà per codificare informazioni visive sottili.

### 2.6 Limiti

- **Assenza di prior semantico nominale**: lo spazio di feature DINOv3 non è organizzato attorno a concetti linguistici. Per classi la cui discriminazione dipende da un concetto nominale forte (cooking, talking on phone, watching television), il backbone non offre alcun vantaggio a priori.
- **Overfitting accelerato dalla maggiore capacità**: la dimensionalità 1024 rende il classificatore molto più suscettibile alla memorizzazione del training set. Il gap train–val a ep28 (stop reg_v2) è ~52 pt — inferiore ai ~71 pt di CLIP a ep49, ma ancora molto elevato. La regolarizzazione mitiga ma non elimina il problema.
- **Mismatch tra capacità backbone e modello temporale**: MS-Temba ha ~18M parametri. Ricevere in ingresso feature 1024-dim invece di 512-dim raddoppia la dimensione dell'input senza aumentare la capacità del modello temporale.
- **CLS token non sfrutta il vantaggio reale di DINOv3**: il vantaggio competitivo di DINOv3 risiede nelle patch features, non nel CLS token. Usando solo il CLS token si rinuncia alla principale superiorità del backbone.

---

## 3. Confronto CLIP vs DINOv3: Sintesi Interpretativa

### 3.1 Differenze fondamentali nello spazio di feature

CLIP e DINOv3 codificano due tipologie fondamentalmente diverse di informazione visiva. CLIP produce una rappresentazione centrata sul **significato linguistico** della scena: due immagini sono vicine nello spazio embedded se potrebbero essere descritte con le stesse parole. DINOv3 produce una rappresentazione centrata sulla **struttura visiva**: due immagini sono vicine se la disposizione degli elementi nella scena è simile.

In Charades — un dataset con molte azioni domestiche facilmente verbalizzabili — questa differenza si riflette direttamente nella distribuzione degli AP per classe. CLIP domina su classi con forte ancoraggio lessicale (*cooking*, *talking on phone*, *watching television*). DINOv3 mostra vantaggi netti su azioni di manipolazione con oggetti visivamente specifici che hanno bassa salienza nel corpus web di CLIP.

### 3.2 Pattern di vantaggio di DINOv3: correzione rispetto all'analisi precedente

**Nota critica**: i dati del run DINOv3 originale (senza regolarizzazione) mostravano valori anomali molto alti su alcune classi posturali — in particolare c153 (sneezing, 67.9), c150 (running, 54.6), c154 (standing up, 55.8), c151 (standing to sitting, 69.1) — che suggerivano un forte vantaggio posturale di DINOv3. I dati completi del run reg_v2 (baseline definitiva) **non confermano questo pattern**: CLIP vince su tutte queste classi. I valori anomali del run originale erano probabilmente artefatti di overfitting precoce su quei pattern visivi specifici, che la regolarizzazione ha eliminato.

Il pattern reale di DINOv3 reg_v2 su 157 classi è diverso: i vantaggi maggiori riguardano **azioni di manipolazione con oggetti visivamente specifici** che hanno bassa salienza nel corpus web di CLIP:

| Classe | CLIP | DINOv3 reg_v2 | Δ DINOv3 |
|--------|:----:|:-------------:|:--------:|
| Opening a refrigerator (c143) | 60.2 | 88.1 | +27.9 |
| Putting a broom somewhere (c099) | 27.4 | 53.0 | +25.6 |
| Holding a vacuum (c137) | 51.3 | 74.2 | +22.9 |
| Putting a blanket somewhere (c071) | 18.5 | 38.8 | +20.3 |
| Taking a vacuum from somewhere (c138) | 20.9 | 37.9 | +17.0 |
| Taking some clothes (c002) | 38.6 | 54.7 | +16.1 |
| Lying on a bed (c134) | 69.9 | 82.8 | +12.9 |

Le classi posturali vanno in senso opposto: *standing to sitting* (DINOv3 23.2 vs CLIP 60.4), *sneezing* (11.8 vs 17.8), *running* (9.1 vs 18.9) — CLIP vince su tutte.

### 3.3 Dinamiche di overfitting a confronto

Il fatto che DINOv3 reg_v2 mostri un overfitting più contenuto rispetto al run originale (−4.0 mAP da best a stop vs −6.6) conferma che la regolarizzazione era necessaria. Tuttavia il gap train–val rimane grande (~52 pt a ep28): il problema è strutturale — 7985 video con 157 classi sbilanciate non sono sufficienti per uno spazio 1024-dim senza forte regolarizzazione.

CLIP, invece, beneficia di una struttura rappresentazionale più fortemente regolarizzata dal linguaggio, che funge da prior semantico stabilizzante. Anche senza dropout o weight decay aggiuntivi, il suo gap train–val post-best è meno catastrofico (da 32.40 a ~29 in 36 epoche, vs collasso di DINOv3 originale).

### 3.4 Sintesi interpretativa

Nel complesso, i risultati confermano che **CLIP costituisce il backbone più efficace come scelta generale su Charades**: best-mAP 32.40 vs 25.21 (gap +7.19), 22 classi con AP ≥ 60 contro 5, e solo 49 classi con AP < 20 contro ~101 per DINOv3 reg_v2.

**DINOv3** mantiene un ruolo come backbone complementare per le ~51 classi dove il suo vantaggio è reale, ma il pattern è specifico: oggetti visivamente caratteristici con bassa salienza nel corpus web (frigorifero, aspirapolvere, scopa, alcune azioni put/take su oggetti specifici). Non è, come inizialmente ipotizzato, un backbone con vantaggio posturale generale.

I due modelli catturano **informazioni diverse e parzialmente complementari**, suggerendo che una combinazione — tramite late fusion, gated fusion o cross-attention — potrebbe sfruttare simultaneamente il prior semantico-linguistico di CLIP e la sensibilità strutturale di DINOv3 su oggetti specifici.

---

## 4. Configurazione di Training

### 4.1 CLIP — Configurazione originale (baseline definitiva)

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
| EMA decay | 0.99996 |
| Dropout | 0.0 |
| Drop path | 0.0 |
| CLIP feat dim | 512 |
| Parametri totali | ~18M |
| Loss | BCE multi-label + Diversity Loss (β=0.05) |
| Seed | 0 |
| Early stopping | No |

> **Nota**: Assenza totale di dropout e drop-path rende il modello interamente dipendente dall'early stopping come meccanismo anti-overfitting. Come mostrerà la Sezione 4, il training MAP raggiunge ~99.5% all'ep49 mentre il val MAP crolla a 28.7% — un gap di ~70 punti percentuali. Esperimenti di regolarizzazione (reg v1: drop=0.1/wd=0.05; reg v2: drop=0.0/wd=0.05) non hanno migliorato il best mAP, confermando che il problema di CLIP è strutturale — il prior semantico-linguistico già regolarizza implicitamente la geometria dello spazio feature.

### 4.2 DINOv3 — Configurazione reg v2 (baseline definitiva)

| Parametro | CLIP | DINOv3 reg v2 | Δ / Nota |
|-----------|:----:|:-------------:|----------|
| DINOv3 feat dim | 512 | 1024 | Doppia dimensionalità |
| Dropout | 0.0 | **0.05** | Aggiunto per reg |
| Drop path | 0.0 | **0.05** | Aggiunto per reg |
| Weight decay | 0.01 | **0.05** | Aumentato per reg |
| Early stop patience | — | **15** | Ferma a ep28 |
| Min delta | — | **0.01** | Soglia miglioramento |
| Tutti gli altri | = | = | Identici a CLIP |

**Cronologia esperimenti DINOv3**:

| Config | drop | dp | wd | Best mAP | Stop ep | Nota |
|--------|:----:|:--:|:--:|:--------:|:-------:|------|
| orig | 0.0 | 0.0 | 0.01 | 25.44 | 50 | Crollo −6.6 mAP post-ep13 |
| reg v1 | 0.2 | 0.1 | 0.05 | 24.56 | 26 | Dropout troppo aggressivo |
| **reg v2** | **0.05** | **0.05** | **0.05** | **25.21** | **28** | Quasi parità best, curva stabile |

> **Nota architetturale**: le modifiche di regolarizzazione hanno richiesto interventi a `models_MSTemba.py`: la `LinearProjection` è stata estesa con `drop_rate`, propagato a tutte le istanze inter-blocco e all'`InputProjection` (convertita da `nn.Linear` a `nn.Sequential(Linear, LayerNorm, GELU, Dropout)`).

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
| Gap train–val @ stop ep | ~70.8 pt (ep49) | ~52.4 pt (ep28) | — |

### 5.2 Distribuzione AP per Classe (Best Checkpoint) — 157 classi complete

| Fascia AP | CLIP ep13 | DINOv3 reg_v2 ep13 |
|-----------|:---------:|:------------------:|
| ≥80 (Eccellente) | 0 | 1 (c143: 88.1) |
| 60–79 | 22 | 4 |
| 40–59 | 30 | 10 |
| 20–39 | 56 | 41 |
| <20 (Bassa) | 49 | ~101 |

> **Nota**: a differenza della versione precedente di questa analisi (con dati DINOv3 parziali su 44/157 classi), la tabella ora copre tutte 157 classi con dati reg_v2 completi. La distribuzione DINOv3 risulta molto più sbilanciata verso le fasce basse di quanto i dati parziali suggerissero.

---

## 6. Curve di Training Completo

## 6.1 Curve di Training CLIP (Epoche 0–49)

Tutti i valori estratti dal `training.log` — EMA model, configurazione originale.

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

> **Osservazione chiave**: Il val MAP raggiunge il picco a ep13 (32.40) poi scende monotonicamente. Il train MAP continua a salire fino a 99.52. Il modello entra in overfitting massiccio dopo ep20. Esperimenti di regolarizzazione (reg v1 e v2) non hanno migliorato il best mAP, suggerendo che l'overfitting di CLIP è strutturale — legato alla limitata dimensione del training set (7985 video) rispetto alla capacità del modello (~18M parametri).

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
| 14 |  34.44 |  24.73 |  25.37 |  +9.71 | |
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
| 28 |  66.89 |  21.22 |  21.98 | +45.67 | | ← Early stop (patience 15)

> **Osservazione chiave**: la curva reg_v2 mostra lo stesso picco a ep13 dell'originale (25.21 vs 25.44, −0.23 mAP) ma con crollo post-picco più contenuto (−4.0 mAP in 15 epoche vs −6.6 in 36 epoche). Il gap train–val a ep28 (~52 pt) è inferiore a quello dell'originale (~79 pt a ep49), confermando l'effetto stabilizzante della regolarizzazione. Il pattern strutturale — picco precoce, poi overfitting progressivo — è identico a CLIP, confermando che il problema è principalmente legato alla dimensione del dataset.

---

## 7. Tabella Completa per Classe (157 classi)

**Legenda colonne**:
- **Tr**: occorrenze nel training set; **Te**: nel test set; **Tot**: totale
- **AP₀**: AP all'epoca 0 (baseline random); **AP₁₃**: AP al best checkpoint CLIP (ep13)
- **AP_peak**: AP massima CLIP raggiunta in qualsiasi epoca 0–49; **Pk_ep**: epoca del picco CLIP
- **AP₄₉**: AP CLIP all'ultima epoca; **Δ13→49**: drop dal best checkpoint a ep49 (negativo = peggioramento)
- **AP_DINOv3**: AP DINOv3 reg_v2 al suo best checkpoint per-class
- **Winner**: backbone con AP più alta al rispettivo best checkpoint

| Code | Classe | Obj | Verb | Tr | Te | Tot | AP₀ | AP₁₃ | AP_peak | Pk_ep | AP₄₉ | Δ13→49 | AP_DINOv3 | Winner |
|------|--------|-----|------|---:|---:|----:|----:|-----:|--------:|------:|-----:|-------:|----------:|--------|
| c000 | Holding some clothes | clothes | hold | 635 | 227 | 862 | 5.4 | 41.3 | 42.5 | ep10 | 32.9 | +8.3 | — | CLIP |
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
| c024 | Throwing a bag somewhere | bag | throw | 89 | 21 | 110 | 0.2 | 7.1 | 15.6 | ep25 | 8.9 | -1.7 | 7.5 | CLIP |
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
| c072 | Snuggling with a blanket | blanket | snuggle | 323 | 109 | 432 | 3.9 | 57.6 | 68.6 | ep29 | 66.0 | -8.4 🟢 | 24.7 | CLIP |
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
| c091 | Washing a window | window | wash | 39 | 7 | 46 | 0.3 | 35.1 | 68.7 | ep20 | 43.5 | -8.3 🟢 | 40.9 | CLIP |
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
| c106 | Drinking from a cup/glass/bottle | cup/glass/bottle | drink | 1134 | 300 | 1434 | 5.8 | 65.5 | 66.5 | ep14 | 60.1 | +5.4 | 19.9 | CLIP |
| c107 | Holding a cup/glass/bottle | cup/glass/bottle | hold | 1014 | 380 | 1394 | 11.0 | 67.2 | 69.0 | ep10 | 62.0 | +5.2 | 63.5 | CLIP |
| c108 | Pouring into a cup/glass/bottle | cup/glass/bottle | pour | 276 | 56 | 332 | 1.0 | 20.9 | 27.3 | ep9 | 20.0 | +0.9 | 8.0 | CLIP |
| c109 | Putting a cup/glass/bottle somewhere | cup/glass/bottle | put | 481 | 211 | 692 | 2.0 | 22.3 | 27.1 | ep14 | 21.0 | +1.3 | 30.4 | DINOv3 |
| c110 | Taking a cup/glass/bottle from somewhere | cup/glass/bottle | take | 536 | 240 | 776 | 3.3 | 27.5 | 29.4 | ep10 | 22.1 | +5.5 | 37.5 | DINOv3 |
| c111 | Washing a cup/glass/bottle | cup/glass/bottle | wash | 49 | 20 | 69 | 0.4 | 12.7 | 12.7 | ep13 | 4.8 | +7.9 | 4.8 | CLIP |
| c112 | Closing a closet/cabinet | closet/cabinet | close | 410 | 134 | 544 | 1.2 | 19.4 | 19.4 | ep13 | 13.6 | +5.7 | 13.6 | CLIP |
| c113 | Opening a closet/cabinet | closet/cabinet | open | 595 | 199 | 794 | 3.5 | 41.2 | 41.2 | ep13 | 37.5 | +3.7 | 51.4 | DINOv3 |
| c114 | Tidying up a closet/cabinet | closet/cabinet | tidy | 218 | 83 | 301 | 2.7 | 29.3 | 34.7 | ep8 | 21.9 | +7.4 | 22.2 | CLIP |
| c115 | Holding a paper/notebook | paper/notebook | hold | 324 | 145 | 469 | 4.2 | 28.4 | 42.6 | ep10 | 30.0 | -1.6 | 11.4 | CLIP |
| c116 | Putting their paper/notebook somewhere | paper/notebook | put | 168 | 59 | 227 | 0.9 | 6.2 | 11.2 | ep11 | 5.7 | +0.5 | 16.5 | DINOv3 |
| c117 | Taking paper/notebook from somewhere | paper/notebook | take | 175 | 60 | 235 | 0.9 | 12.4 | 15.0 | ep11 | 10.2 | +2.2 | 15.2 | DINOv3 |
| c118 | Holding a dish | dish | hold | 763 | 267 | 1030 | 7.8 | 38.2 | 42.1 | ep10 | 34.9 | +3.3 | 16.4 | CLIP |
| c119 | Putting a dish somewhere | dish | put | 452 | 145 | 597 | 2.8 | 14.3 | 17.1 | ep21 | 13.5 | +0.8 | 28.1 | DINOv3 |
| c120 | Taking a dish from somewhere | dish | take | 375 | 146 | 521 | 2.3 | 15.9 | 16.6 | ep15 | 15.3 | +0.7 | 29.3 | DINOv3 |
| c121 | Washing a dish | dish | wash | 112 | 30 | 142 | 1.6 | 54.2 | 54.2 | ep13 | 39.8 | +14.4 | 19.5 | CLIP |
| c122 | Lying on a sofa/couch | sofa/couch | lie | 160 | 62 | 222 | 3.5 | 46.9 | 48.6 | ep9 | 44.3 | +2.6 | 7.7 | CLIP |
| c123 | Sitting on sofa/couch | sofa/couch | sit | 484 | 189 | 673 | 6.5 | 59.9 | 60.4 | ep10 | 51.2 | +8.7 | 51.0 | CLIP |
| c124 | Lying on the floor | floor | lie | 168 | 48 | 216 | 1.3 | 55.1 | 59.2 | ep17 | 52.2 | +3.0 | 4.0 | CLIP |
| c125 | Sitting on the floor | floor | sit | 395 | 142 | 537 | 5.0 | 54.1 | 57.5 | ep16 | 52.4 | +1.7 | 41.7 | CLIP |
| c126 | Throwing something on the floor | floor | throw | 308 | 135 | 443 | 1.5 | 11.4 | 14.4 | ep16 | 6.9 | +4.5 | 10.3 | CLIP |
| c127 | Tidying something on the floor | floor | tidy | 478 | 152 | 630 | 4.2 | 53.1 | 58.2 | ep14 | 51.5 | +1.6 | 36.5 | CLIP |
| c128 | Holding some medicine | medicine | hold | 282 | 71 | 353 | 2.0 | 16.4 | 19.7 | ep29 | 18.1 | -1.6 | 5.5 | CLIP |
| c129 | Taking/consuming some medicine | medicine | eat | 162 | 37 | 199 | 0.7 | 9.7 | 14.3 | ep17 | 8.8 | +0.8 | 12.5 | DINOv3 |
| c130 | Putting groceries somewhere | groceries | put | 207 | 80 | 287 | 1.9 | 34.0 | 35.3 | ep16 | 24.9 | +9.1 | 5.6 | CLIP |
| c131 | Laughing at television | television | laugh | 46 | 13 | 59 | 0.3 | 11.2 | 25.1 | ep40 | 23.8 | -12.6 🟢 | 1.6 | CLIP |
| c132 | Watching television | television | watch | 353 | 108 | 461 | 2.8 | 61.3 | 65.9 | ep18 | 63.3 | -2.0 | 48.1 | CLIP |
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
| c145 | Working on paper/notebook | paper/notebook | work | 237 | 60 | 297 | 2.4 | 58.4 | 63.1 | ep8 | 54.4 | +3.9 | 14.3 | CLIP |
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

> 🔴 = crollo severo dopo il best checkpoint (Δ > −20); 🟢 = continua a migliorare dopo ep13 (picco reale >5pt sopra AP₁₃)

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

### 8.3 Classi dove DINOv3 reg_v2 supera CLIP — Pattern Reale

> **Nota importante**: questa sezione è stata interamente rivista rispetto alla versione precedente. I dati del run DINOv3 originale (senza reg) mostravano vantaggi enormi su classi posturali (c153 sneezing +50.1, c150 running +35.7, c154 standing up +19.0) che i dati reg_v2 **non confermano**. CLIP vince su tutte quelle classi con reg_v2. Il pattern reale di DINOv3 è diverso: vantaggi su azioni di manipolazione con oggetti visivamente specifici a bassa salienza nel corpus web di CLIP.

| Code | Classe | AP_CLIP | AP_DINOv3 reg_v2 | Vantaggio DINOv3 |
|------|--------|--------:|------------------:|------------------:|
| c143 | Opening a refrigerator | 60.2 | 88.1 | +27.9 |
| c099 | Putting a broom somewhere | 27.4 | 53.0 | +25.6 |
| c137 | Holding a vacuum | 51.3 | 74.2 | +22.9 |
| c071 | Putting a blanket somewhere | 18.5 | 38.8 | +20.3 |
| c138 | Taking a vacuum from somewhere | 20.9 | 37.9 | +17.0 |
| c002 | Taking some clothes from somewhere | 38.6 | 54.7 | +16.1 |
| c013 | Washing a table | 8.4 | 23.8 | +15.4 |
| c001 | Putting clothes somewhere | 32.2 | 46.5 | +14.3 |
| c043 | Taking a box from somewhere | 18.3 | 32.4 | +14.1 |
| c119 | Putting a dish somewhere | 14.3 | 28.1 | +13.8 |
| c120 | Taking a dish from somewhere | 15.9 | 29.3 | +13.4 |
| c134 | Lying on a bed | 69.9 | 82.8 | +12.9 |
| c035 | Taking a towel/s from somewhere | 13.6 | 26.2 | +12.6 |
| c034 | Putting a towel/s somewhere | 17.8 | 29.4 | +11.6 |
| c116 | Putting their paper/notebook | 6.2 | 16.5 | +10.3 |
| c113 | Opening a closet/cabinet | 41.2 | 51.4 | +10.2 |
| c110 | Taking a cup/glass/bottle | 27.5 | 37.5 | +10.0 |
| c100 | Taking a broom from somewhere | 34.9 | 44.6 | +9.7 |
| c003 | Throwing clothes somewhere | 20.7 | 30.2 | +9.5 |
| c045 | Throwing a box somewhere | 8.7 | 18.1 | +9.4 |

**Classi posturali — CLIP vince su tutte con reg_v2** (erano erroneamente in lista DINOv3 nella versione precedente):

| Code | Classe | AP_CLIP | AP_DINOv3 reg_v2 | Winner corretto |
|------|--------|--------:|------------------:|:---------------:|
| c153 | Someone is sneezing | 17.8 | 11.8 | **CLIP** |
| c150 | Someone is running | 18.9 | 9.1 | **CLIP** |
| c154 | Someone is standing up | 36.8 | 27.9 | **CLIP** |
| c151 | Standing to sitting | 60.4 | 23.2 | **CLIP** |
| c059 | Sitting in a chair | 73.6 | 51.3 | **CLIP** |

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
| v022 | stand | 2 | 35.0 | 18.5 | +16.5 | Transizione posturale → instabile nel tempo |
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
| v025 | throw | 11 | 10.2 | 7.5 | +2.7 | Gesto breve → poche occ., raro nel corpus |
| v011 | make | 1 | 7.8 | 8.2 | -0.4 | Rarissimo (solo sandwich) |
| v027 | turn | 2 | 5.5 | 4.6 | +0.9 | Richiede stato pre/post → problematico |

---

## 11. Coppie di Classi Confondibili

### 11.1 Coppie con Stesso (Oggetto, Verbo) — Ambiguità Strutturale

| Coppia | Classe A | Classe B | AP_A | AP_B | Problema |
|--------|----------|----------|-----:|-----:|---------|
| c010/c011 | Sitting on a table | Sitting at a table | 3.8 | 75.5 | c010 vs c011: stesso codice (o033,v018) — 'sitting ON table' vs 'sitting AT table'. Distinzione puramente preposizionale, impossibile da feature visive. Sia CLIP che DINOv3 soffrono. |
| c104/c105 | Turning on a light | Turning off a light | 8.7 | 2.3 | Stesso codice (o021,v027) — richiede comprensione dello stato luce prima/dopo. Entrambi i backbone sotto AP 8. |
| c043/c044 | Taking a box from somewhere | Taking something from a box | 18.3 | 21.3 | Stesso codice (o005,v023) — direzione dell'azione ambigua. |

### 11.2 Coppie Visivamente Simili (Oggetto Condiviso)

| Code A | Code B | Classe A | Classe B | AP_A (CLIP) | AP_B (CLIP) | Oggetto comune |
|--------|--------|----------|----------|------------:|------------:|---------------|
| c006 | c008 | Closing a door | Opening a door | 30.3 | 39.8 | door |
| c097 | c141 | Walking through a doorway | Grasping onto a doorknob | 43.8 | 40.4 | doorway/doorknob |
| c015 | c019 | Holding a phone/camera | Talking on a phone/camera | 73.2 | 75.9 | phone |
| c047 | c052 | Holding a laptop | Working/Playing on a laptop | 73.1 | 75.9 | laptop |
| c059 | c123 | Sitting in a chair | Sitting on sofa/couch | 73.6 | 59.9 | sit |
| c098 | c102 | Holding a broom | Tidying up with a broom | 75.4 | 63.0 | broom |
| c132 | c131 | Watching television | Laughing at television | 61.3 | 11.2 | tv |
| c133 | c134 | Awakening in bed | Lying on a bed | 49.0 | 69.9 | bed |

---

## 12. Dinamiche di Overfitting: Late Bloomers e Collapse

Classi con le dinamiche più interessanti tra ep13 e ep49 in CLIP.

### 12.1 Classi che Continuano a Migliorare Dopo ep13 (Late Bloomers 🟢)

| Code | Classe | AP₁₃ | AP_peak | Pk_ep | AP₄₉ | Guadagno netto |
|------|--------|-----:|--------:|------:|-----:|--------------:|
| c072 | Snuggling with a blanket | 57.6 | 68.6 | ep29 | 66.0 | +10.9 |
| c091 | Washing a window | 35.1 | 68.7 | ep20 | 43.5 | +8.4 |
| c131 | Laughing at television | 11.2 | 25.1 | ep40 | 23.8 | +12.6 |
| c132 | Watching television | 61.3 | 65.9 | ep18 | 63.3 | +2.0 |
| c142 | Closing a refrigerator | 48.5 | 54.3 | ep21 | 50.5 | +2.0 |
| c153 | Someone is sneezing | 17.8 | 25.5 | ep14 | 21.2 | +3.4 |

> **Nota**: Il pattern dei late bloomers suggerisce che un ensemble di checkpoint (ep13 + ep20 + ep29) recupererebbe diversi punti MAP, specialmente per c072, c091 e c131. Le stesse classi mostrano comportamenti diversi con DINOv3 reg_v2 (curva più corta, early stop a ep28).

### 12.2 Classi con Collapse Severo Dopo ep13 (🔴)

| Code | Classe | AP₁₃ | AP₄₉ | Δ | Causa probabile |
|------|--------|-----:|-----:|--:|-----------------|
| c060 | Standing on a chair | 33.2 | 3.4 | −29.8 | Solo 46 train — memorizzazione seguita da forgetting |
| c090 | Opening a window | 29.7 | 10.3 | −19.5 | Solo 87 train — bassa frequenza |
| c022 | Putting a bag somewhere | 40.6 | 25.1 | −15.5 | Ambiguità con c023 (put vs take) |
| c121 | Washing a dish | 54.2 | 39.8 | −14.4 | AP alto al best poi calo progressivo |
| c131 | Laughing at television | 11.2 | 23.8 | ← late bloomer | Eccezionalmente migliora dopo ep13 |
| c144 | Fixing their hair | 30.5 | 18.1 | −12.3 | Gesto fine-grained, bassa frequenza |
| c094 | Smiling in a mirror | 24.8 | 12.5 | −12.3 | Postura specifica ma dataset piccolo |

---

## 13. Analisi del Picco Reale vs Checkpoint Best

Il best checkpoint globale (ep13, MAP=32.40) non è il picco ottimale per ogni classe. La tabella seguente mostra le classi con il maggiore divario tra AP_peak (raggiungibile) e AP₁₃ (riportato).

| Code | Classe | AP₁₃ (riportato) | AP_peak (massimo) | Pk_ep | Gain |
|------|--------|------------------:|------------------:|------:|-----:|
| c091 | Washing a window | 35.1 | 68.7 | ep20 | +33.5 |
| c013 | Washing a table | 8.4 | 25.2 | ep18 | +16.8 |
| c115 | Holding a paper/notebook | 28.4 | 42.6 | ep10 | +14.2 |
| c131 | Laughing at television | 11.2 | 25.1 | ep40 | +13.9 |
| c136 | Fixing a vacuum | 20.3 | 32.6 | ep8 | +12.4 |
| c085 | Laughing at a picture | 13.0 | 24.1 | ep17 | +11.2 |
| c072 | Snuggling with a blanket | 57.6 | 68.6 | ep29 | +11.0 |
| c137 | Holding a vacuum | 51.3 | 62.0 | ep10 | +10.7 |
| c133 | Awakening in bed | 49.0 | 59.5 | ep21 | +10.5 |
| c139 | Washing their hands | 26.2 | 35.0 | ep10 | +8.9 |
| c058 | Throwing shoes somewhere | 6.2 | 15.0 | ep19 | +8.8 |
| c095 | Washing a mirror | 7.4 | 16.2 | ep7 | +8.8 |
| c060 | Standing on a chair | 33.2 | 41.9 | ep15 | +8.7 |
| c140 | Fixing a doorknob | 38.5 | 47.1 | ep25 | +8.6 |
| c082 | Tidying a shelf | 19.7 | 28.2 | ep12 | +8.5 |
| c024 | Throwing a bag somewhere | 7.1 | 15.6 | ep25 | +8.4 |
| c146 | Awakening somewhere | 53.1 | 61.0 | ep8 | +7.9 |
| c037 | Tidying up a towel/s | 14.7 | 22.6 | ep17 | +7.8 |
| c153 | Someone is sneezing | 17.8 | 25.5 | ep14 | +7.7 |
| c102 | Tidying up with a broom | 63.0 | 70.6 | ep19 | +7.5 |
| c100 | Taking a broom from somewhere | 34.9 | 42.3 | ep12 | +7.4 |
| c103 | Fixing a light | 24.5 | 31.9 | ep19 | +7.4 |
| c075 | Tidying up a blanket/s | 21.4 | 28.7 | ep6 | +7.3 |
| c005 | Washing some clothes | 19.5 | 26.5 | ep10 | +7.0 |
| c055 | Putting on shoe/shoes | 33.8 | 40.8 | ep16 | +7.0 |

> **Implicazione pratica**: Un ensemble dei checkpoint ep13 + ep20 + ep29 recupererebbe diversi punti MAP senza alcuna modifica architetturale. Le classi c091, c072, c131 e c133 beneficerebbero massimamente.

---

## 14. Struttura Compositiva: Mapping Oggetto × Verbo

Ogni classe Charades è definita come una coppia *(oggetto, verbo)*. La performance del modello riflette questa struttura.

### 14.1 Oggetti con Maggior Mean AP CLIP

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

### 14.2 Oggetti con Minor Mean AP CLIP

| Rank | Oggetto | #Classi | Mean AP₁₃ | Motivo |
|------|---------|--------:|----------:|--------|
| 1 | shoe (o031) | 6 | 23.9 | |
| 2 | box (o005) | 7 | 20.7 | |
| 3 | shelf (o030) | 2 | 20.5 | |
| 4 | towel (o035) | 6 | 18.5 | |
| 5 | picture (o026) | 5 | 14.4 | Bassa frequenza, alta varianza |
| 6 | medicine (o022) | 2 | 13.0 | Oggetto minuto, difficile da localizzare |
| 7 | light (o021) | 3 | 11.8 | Luce non distinguibile on/off |

### 14.3 Coppie (Oggetto, Verbo) Strutturalmente Duplicate

| Coppie | Classi | AP_A | AP_B | Divario train | Analisi |
|--------|--------|-----:|-----:|:-------------:|---------|
| (o033, v018) | c010 vs c011 | 3.8 | 75.5 | 46 vs 728 | Sitting *on* vs *at* a table — impossibile da features visive. Il divario AP è interamente spiegato dalla frequenza (c011 = 15.8× più frequente). |
| (o021, v027) | c104 vs c105 | 8.7 | 2.3 | 207 vs 189 | Turning *on* vs *off* a light — richiede comprensione dello stato luce pre/post azione. AP entrambe < 9. |
| (o005, v023) | c043 vs c044 | 18.3 | 21.3 | 180 vs 154 | Taking *a box from somewhere* vs *something from a box* — direzione dell'azione ambigua visivamente. |

---

## 15. Classi con Ambiguità Strutturale o Dataset Issue

| Code | Classe | Problema | Train | Test | AP_CLIP | AP_DINOv3 |
|------|--------|----------|------:|-----:|--------:|----------:|
| c010 | Sitting on a table | Strutturalmente identica a c011 (stesso mapping). AP perpetuamente bassa per rarità. | 46 | 16 | 3.8 | 8.5 |
| c104 | Turning on a light | Identico mapping a c105. Entrambe near-zero AP. | 207 | 73 | 8.7 | 7.5 |
| c105 | Turning off a light | Identico mapping a c104. Entrambe near-zero AP. | 189 | 50 | 2.3 | 7.9 |
| c101 | Throwing a broom somewhere | Solo 33 occorrenze totali. AP mai sopra 2.3 (CLIP). DINOv3 9.1 — leggero vantaggio ma sempre basso. | 25 | 8 | 1.5 | 9.1 |
| c064 | Throwing food somewhere | 74 occorrenze totali. AP sempre < 5 per entrambi. | 52 | 22 | 1.9 | 2.8 |
| c085 | Laughing at a picture | 43 occorrenze. Emozione sottile. DINOv3 4.1 vs CLIP 13.0 — CLIP vince comunque. | 28 | 15 | 13.0 | 4.1 |
| c031 | Throwing a book somewhere | 79 occorrenze. Azione rarissima. DINOv3 8.1 vs CLIP 2.7. | 58 | 21 | 2.7 | 8.1 |
| c045 | Throwing a box somewhere | 40 occorrenze. AP CLIP₄₉ = 0.4 — collapse totale. DINOv3 18.1 — vantaggio netto su classi rarissime. | 34 | 6 | 8.7 | 18.1 |
| c060 | Standing on a chair | 53 occorrenze. Collapse estremo CLIP da 33.2 (ep13) a 3.4 (ep49). | 46 | 7 | 33.2 | 1.3 |
| c141 | Grasping onto a doorknob | Split anomalo: 506 train ma solo 20 test. | 506 | 20 | 40.4 | 37.1 |
| c136 | Fixing a vacuum | Ratio invertito: 143 test vs 35 train. | 35 | 143 | 20.3 | 1.1 |

---

## 16. Raccomandazioni e Sviluppi Futuri

### 16.1 Contro l'Overfitting

1. **Dropout e Drop-path per CLIP**: la configurazione attuale ha entrambi a 0.0. Gli esperimenti di regolarizzazione non hanno migliorato il best mAP, suggerendo che il prior semantico-linguistico di CLIP già regolarizza implicitamente. L'overfitting strutturale richiede approcci diversi: lr scheduling più aggressivo post-picco o riduzione della capacità del modello.
2. **Multi-checkpoint ensemble**: la media delle predizioni di ep13 + ep20 + ep29 recupererebbe diversi punti MAP per i late bloomers (c072, c091, c131) senza costi architetturali.
3. **Per-class checkpoint selection**: salvare il best checkpoint per ogni classe individualmente e usare un ensemble pesato al test time.
4. **Mixup/CutMix**: augmentation nello spazio delle features temporali.

### 16.2 Per le Classi a Bassa AP

5. **Classi duplicate (c010/c011, c104/c105, c043/c044)**: merging o esclusione dall'evaluation, oppure aggiunta di un modulo state-detection dedicato.
6. **Classi rarissime (c101, c064, c085, tot < 80)**: over-sampling, synthetic augmentation, o esclusione da mAP per un'analisi più onesta.
7. **Verbi di stato (v027: turn on/off)**: richiedono un modulo temporale dedicato che confronti il frame iniziale e finale dell'azione.

### 16.3 Per DINOv3

8. **Dense features (priorità alta)**: nel setup attuale viene usato solo il CLS token. Le patch features di DINOv3 — il suo reale vantaggio competitivo — non vengono sfruttate. L'estrazione e l'integrazione di mean-patch o attention-pooled patch features è il passo successivo più promettente.
9. **Feature projection**: ridurre 1024→512 con un layer di proiezione trainabile prima del backbone MS-Temba per allineare la capacità del modello a quella di CLIP.
10. **Hybrid routing corretto**: usare CLIP per le classi semanticamente descrittive e DINOv3 per le classi dove il vantaggio reg_v2 è confermato (c143 +27.9, c099 +25.6, c137 +22.9, c071 +20.3, c138 +17.0). **Non usare DINOv3** per il routing su classi posturali (c151, c153, c154, c150, c059): con i dati reg_v2 CLIP vince su tutte.

### 16.4 Architettura

11. **Two-head compositional classifier**: aggiungere un secondo classificatore (oggetto + verbo separati) in parallelo al classificatore globale, sfruttando il mapping compositivo di Charades.
12. **Per-verb temporal modules**: verbi come v025 (throw), v027 (turn) e v022 (stand) richiedono analisi delle sequenze temporali corte — considerare attention locale più densa su questi.

### 16.5 Passi Successivi (Fase 2)

In ordine di priorità:

1. **DINOv3 Dense Features (Fase 2A)**: estrarre le patch features di DINOv3 (mean-pooling o attention-pooling sui token di patch) per ottenere una rappresentazione spazialmente strutturata. Impatto atteso: +1–4 mAP su classi di manipolazione oggetti e potenzialmente sulle classi posturali (che DINOv3 CLS-only non risolve).
2. **Skeleton Features SCDNet (Fase 2B)**: integrare le feature scheletriche pre-estratte disponibili su ABACA (`Charades_SCDNet_features2.zip`). Strategia raccomandata: gated fusion. Impatto atteso: +1–4 mAP su classi dei Gruppi A (verbi di stato) e B (transizioni posturali).
3. **Survey stato dell'arte (Fase 3)**: analisi sistematica della letteratura su combinazione di feature visive e scheletriche per TAD/TAR.

---

## Fonti e Metadati

- **Dataset**: Charades v1, Allen Institute for AI — Sigurdsson et al. (2016)
- **Modello**: MS-Temba — Pramanik et al. — https://github.com/thearkaprava/MS-Temba
- **Log di training CLIP**: `runs/charades/clip/seed0/training.log` (50 epoche, seed=0)
- **Metriche DINOv3**: `runs/charades/dinov3_vitl16_reg_v2/seed0/metrics_per_epoch.csv` (28 epoche, reg_v2)
- **AP DINOv3 per-class**: estratte dall'analisi completa reg_v2 su tutte 157 classi (marzo 2026)
- **Esperimenti regolarizzazione**: `runs/charades/dinov3_vitl16/`, `runs/charades/dinov3_vitl16_reg/`, `runs/charades/dinov3_vitl16_reg_v2/`
- **File di configurazione usati**: `Charades_v1_classes.txt`, `Charades_v1_mapping.txt`, `Charades_v1_objectclasses.txt`, `Charades_v1_verbclasses.txt`, `Charades_v1_train.csv`, `Charades_v1_test.csv`