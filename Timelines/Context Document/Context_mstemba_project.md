# Context — MS-Temba Research

**Owner**: Matteo Di Iorio (master's thesis, traineeship at Inria Sophia)  
**Project**: MS-Temba research extension — TAD on densely-labeled ADL benchmarks  
**Document version**: v5 (per-class baselines complete, ready for skeleton design)  
**Last updated**: 2026-05-21

> **Come usare questo documento**: è il briefing che ogni nuova chat di Claude deve leggere prima.  
> Sezione A è lo stato corrente (aggiornare a fine sessione).  
> Sezione B è il setup tecnico (cambia di rado).  
> Sezione C è lo storico esperimenti (append-only, mai cancellare).

---

## A. Stato corrente

### A.1 Active repo & branch
- **Repo locale**: `/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2/`
- **GitHub**: `git@github.com:Matteo-Di-Iorio-s316606/MS-Temba.git` (fork di `thearkaprava/MS-Temba`)
- **Branch attiva**: `reproduce-baseline`
- **Convention**: `main` traccia upstream (immodificata), `reproduce-baseline` = setup + infra training + fix minori + per-class eval, `feat/<nome>` per macro-esperimenti

### A.2 Dove sono nel piano
- ✅ Setup repo nuova (MS-Temba-v2 da zero, fork pulito, env separato)
- ✅ Baseline references importate dalla vecchia repo (logs + checkpoint + script storici, 6 baseline)
- ✅ Symlink feature dataset dalla vecchia repo
- ✅ Env conda `mstemba_v2` funzionante, pipeline validata via sanity check
- ✅ Sanity check Charades I3D in OAR batch (script self-contained, env+module nel body, OAR directives in header)
- ✅ Infrastruttura training: `CheckpointManager` (atomic save, resume), `EarlyStopper`, `OARSignalHandler` (SIGUSR2)
- ✅ Per-class metrics logger: CSV append-friendly (summary + per-class) + JSON ricostruito a `finalize()`
- ✅ **Decisione EMA col supervisor (2026-05-20)**: NON usare EMA (vedi A.4 per stato implementativo)
- ✅ **Branch `-eval_only` in `MSTemba_main.py`** (2026-05-20): carica checkpoint via `-resume`, fa una val pass, scrive `metrics_per_class.csv` + JSON, exit. Auto-detect formato (v2 payload vs legacy flat state_dict).
- ✅ **Branch dataset `multithumos`** aggiunto nel main (riusa `charades_dataloader.Charades`, classes=65, JSON `data/multithumos.json`)
- ✅ **Per-class baselines (6 dataset)** generate via `-eval_only` da checkpoint legacy: 4/6 match esatto con C.1, 2/6 (TSU) con gap su `sampled` mAP (vedi A.4)
- ⏳ **Skeleton bidirezionale (Fase 4 redux)**: design dual-stream nativo vs late fusion sui logit. Niente codice ancora.
- ⏳ Ablation T1–T6: roadmap esistente (vedi `Timelines/Fase_6_model_improvements/MSTemba_roadmap.md`)

### A.3 Prossimi task concreti
1. **Design skeleton bidirezionale** (no GPU richiesto, design + parameter budget):
   - Decidere architettura: skeleton come secondo input nel dataloader (dual-stream nativo, fusione precoce a livello D=256) **vs** skeleton come secondo modello full MS-Temba con late fusion sui logit.
   - Stimare VRAM su TSU (skeleton D=4096, T=2500) prima del primo dry-run.
   - Riferimenti: `Timelines/Fase_4_skeleton_features/` (analisi vecchia repo, bottleneck Linear(4096→256)) e `Timelines/Fase_5_fusion_of_features/`.
   - Output: nota di design + nome branch (`feat/skeleton-bidir` o variante).
2. **Implementazione skeleton bidir** dopo design approvato. Confronto baseline = colonna `ap_full` del per-class CSV RGB già generato (NON `ap_sampled` per TSU, vedi A.4).
3. **Ablation T1–T6**: solo dopo skeleton funzionante.
4. **Backlog (non bloccante)**: investigare gap TSU sampled mAP in `-eval_only` (vedi A.4). Tutte le ipotesi plausibili escluse, causa sconosciuta. Per il confronto skel-vs-RGB intra-v2 il gap è ininfluente (entrambi usano la stessa pipeline `-eval_only`).

### A.4 Blockers / known issues
- **EMA — decisione supervisor presa, implementazione parziale**: il supervisor (2026-05-20) ha confermato che neither la repo originale né i numeri C.1 usano EMA aggiornato. Decisione: NON usare EMA. **Stato attuale**: nessuna rimozione applicata (default argparse ancora `model_ema=True`, ma `ema_state: None` nei checkpoint v2 perché `update()` non viene mai chiamato — il payload è inerte). **Decisione operativa**: si è scelto di non toccare la riga di default per evitare regressioni; convivere col codice morto. La porta resta aperta per riattivare EMA con `update()` corretto se serve (e.g. ablation T-*).
- **TSU sampled mAP gap in `-eval_only`**: TSU I3D Δ_sampled = -3.31, TSU CLIP Δ_sampled = -12.54 vs C.1. Causa **non identificata** dopo indagine (escluse: code drift `val_step`/`run_network`/`dataloader`/`utils`/`apmeter`, feature path, truncation a num_clips=2500 — solo 6/185 video troncati, RNG varianza — risultato deterministico tra run successivi). Charades e MultiTHUMOS match esatto, gap solo su TSU sampled. **Decisione**: accettare. I CSV per-classe TSU sono internamente consistenti (deterministici, stessa pipeline come sarà per skeleton) → usabili per confronto intra-v2.
- **`val_dataloader(shuffle=True)`** (`MSTemba_main.py`, `load_data`): semanticamente sbagliato (validation deve essere deterministica). Bug pre-esistente identico in vecchia repo. **NON FIXARE ora** — cambierebbe i numeri delle 4 baseline che combaciano e dovresti rigenerare tutto. Eventuale fix futuro richiede rigenerazione di tutti i 6 `metrics_per_class.csv`.
- Solo coda `besteffort` accessibile al tuo account su Sophia. Possibile richiesta upgrade ad abaca da inoltrare via supervisor.

### A.5 Note operative sui per-class CSV
- **Localizzazione**: `experiments/<dataset>_<backbone>_baseline_perclass_<timestamp>_oar<jobid>/metrics_per_class_final.json` + `metrics_per_class.csv` (157/65/51 righe). Colonne: `epoch, class_id, ap_full, ap_sampled, block_1_full, block_1_sampled, block_2_full, block_2_sampled, block_3_full, block_3_sampled`.
- **Per Charades e MultiTHUMOS**: usare indifferentemente `ap_full` o `ap_sampled` (entrambi match con C.1).
- **Per TSU**: usare **`ap_full`** per il confronto skel-vs-RGB. Il `ap_sampled` è internamente coerente ma non confrontabile col paper (vedi A.4).

---

## B. Setup tecnico

### B.1 Compute environment
- **Cluster**: Grid5000 Sophia. La "famiglia esterel" sono **43+ cluster OAR distinti** (`esterel2` … `esterel44`), non un cluster unico. Hardware eterogeneo, da verificare per ogni submission.
- **GPU effettivamente disponibili per mstemba** (compute capability ≥ 7.0 richiesta dai kernel `mamba_ssm`):
  | Cluster | GPU | VRAM | Compute | Note |
  |---|---|---|---|---|
  | esterel16 | RTX 2080 Ti | 11 GiB | 7.5 | Turing, sufficiente per Charades; **insufficiente per TSU num_clips=2500** |
  | esterel17 | A6000 | 48 GiB | 8.6 | Ampere, ideale per TSU |
  | esterel35 (n-2) | A40 | 46 GiB | 8.6 | Ampere, equivalente A6000 |
  | esterel19+ | da verificare con Monika | — | — | controllare di volta in volta |
- **Cluster NON utilizzabili**: esterel2-15 (Pascal GTX 1080/1080Ti, compute 6.1), esterel3-4 (Maxwell TITAN X, compute 5.2). Kernel mamba_ssm 1.1.1 falliscono.
- **Coda accessibile**: solo `besteffort` (job killabile, mitigato da checkpoint manager + SIGUSR2 handler).
- **Frontend**: `mdiiorio@fsophia.grid5000.fr` (no GPU, solo sottomissione e gestione job).
- **Monika** (vedere GPU libere live): https://intranet.grid5000.fr/oar/Sophia/monika-prod.cgi

### B.2 Conda env `mstemba_v2`
- Python 3.10.13
- torch 2.1.1+cu118 + torchvision 0.16.1 + torchaudio 2.1.1
- mamba_ssm 1.1.1 (editable, source `mamba-1p1p1/` locale)
- causal-conv1d 1.1.1 (da PyPI source, NON la 1.0.0 della cartella locale — API mismatch)
- setuptools `<70` (pin obbligatorio: torch 2.1.x importa `pkg_resources` rimosso da setuptools≥70)
- tensorboard
- vedere `requirements_mstemba_v2.txt` per la lista completa (= `vim/vim_requirements.txt` upstream meno mmcv/mmsegmentation, più tensorboard)

### B.3 CUDA module
- `module load cuda/11.8.0_gcc-10.4.0` (modulo Grid5000 Sophia per torch 2.1.1+cu118)
- `TORCH_CUDA_ARCH_LIST` esportato automaticamente nello script setup. Per A40/A6000 (Ampere): `"8.6"`; per RTX 2080 Ti (Turing): `"7.5"`.

### B.4 Layout repo
```
MS-Temba-v2/
├── vim/                          # codice MS-Temba upstream (TAD core)
│   ├── MSTemba_main.py           # training entrypoint (modificato per checkpoint + logger + eval_only + multithumos branch)
│   ├── models_MSTemba.py         # architettura
│   ├── charades_dataloader.py    # dataloader (generico, riusato da multithumos)
│   ├── extensions/               # NUOVE estensioni progetto (non upstream)
│   │   ├── __init__.py
│   │   ├── checkpoint.py         # CheckpointManager + EarlyStopper + OARSignalHandler
│   │   └── metrics_logger.py     # per-class CSV + JSON
│   └── scripts/                  # script di run upstream
├── mamba-1p1p1/                  # mamba_ssm 1.1.1 (editable)
├── causal-conv1d/                # NON usato (versione 1.0.0 incompatibile)
├── seg/                          # codice segmentazione VIM (irrilevante per TAD)
├── data/
│   ├── charades.json             # annotation (upstream)
│   ├── smarthome.json            # annotation (upstream)
│   ├── multithumos.json          # annotation (upstream)
│   └── features/                 # symlink a feature .npy della vecchia repo
│       ├── tsu_i3d -> .../tsu_features_i3d
│       ├── tsu_clip_l14 -> .../tsu_features_clip_l14
│       ├── charades_i3d -> .../charades_features_i3d
│       ├── charades_clip -> .../charades_features_clip
│       ├── multithumos_i3d -> .../multithumos_features_i3d
│       └── multithumos_clip -> .../multithumos_features_clip
├── baselines_reference/          # snapshot baseline vecchia repo
│   ├── logs/                     # training.log + run_meta.txt
│   ├── checkpoints/              # best_model.pth, formato OrderedDict flat (GITIGNORED)
│   │   ├── charades_i3d_seed0/best_model.pth
│   │   ├── charades_clip_seed0/best_model.pth
│   │   ├── multithumos_i3d_seed0/best_model.pth
│   │   ├── multithumos_clip_seed0/best_model.pth
│   │   ├── tsu_i3d_seed0/best_model.pth
│   │   └── tsu_clip_seed0/best_model.pth
│   ├── scripts_historic/         # bash script vecchia repo (NON eseguibili upstream)
│   └── README.md
├── experiments/                  # output esperimenti correnti (GITIGNORED)
│   └── oar_logs/                 # OAR stdout/stderr per job batch
├── scripts/                      # infrastruttura progetto
│   ├── setup_env_mstemba_v2.sh   # build env da zero
│   ├── sanity_check_charades_i3d.sh   # smoke test OAR-batch-safe
│   └── check_env.sh              # env health check idempotente (5/5 verifications)
├── activate_mstemba.sh           # source-only env activation helper (repo root)
├── Timelines/                    # documentazione progetto (questo file e altri)
├── requirements_mstemba_v2.txt
└── README.md (upstream)
```

### B.5 Annotation paths
- Charades: `data/charades.json` (157 classi, 7985 train / 1863 test)
- TSU: `data/smarthome.json` (51 classi, splits CS_51, **185 test videos**, mean T=1598, max T=3147)
- MultiTHUMOS: `data/multithumos.json` (65 classi)

### B.6 Backbone dimensions
| Backbone | in_feat_dim | Status |
|---|---|---|
| I3D | 1024 | feature pre-estratte |
| CLIP-L/14 | 768 | feature pre-estratte (da HuggingFace) |
| DINOv3 ViT-L/16 | 1024 (CLS), 2048 (combined) | feature in vecchia repo |
| SCD-Net skeleton | 4096 | feature in vecchia repo |

### B.7 Fix applicati al codice upstream
| File | Cosa | Status |
|---|---|---|
| `vim/MSTemba_main.py` riga 548 | Path Charades JSON → repo locale | ✅ committed |
| `vim/MSTemba_main.py` riga 555 | Path TSU JSON → repo locale (`data/smarthome.json`) | ✅ committed (2026-05-21) |
| `vim/MSTemba_main.py` branch dataset | Aggiunto branch `elif args.dataset == 'multithumos'` (classes=65, riusa charades_dataloader) | ✅ committed (2026-05-21) |
| `vim/MSTemba_main.py` argparse | Aggiunti `-resume`, `-save_every_epoch`, `-early_stop_patience`, `-early_stop_min_delta`, `-eval_only` | ✅ committed |
| `vim/MSTemba_main.py` branch `-eval_only` | Carica checkpoint via `-resume`, fa una val_step, scrive per-class CSV+JSON, exit. Auto-detect formato (v2 payload vs legacy flat state_dict). | ✅ committed (2026-05-21) |
| `vim/MSTemba_main.py` top | `sys.path.insert` con `_THIS_DIR` per importare `extensions/` da qualsiasi cwd; `import sys` | ✅ committed |
| `vim/MSTemba_main.py` `val_step` | Restituisce 10-tuple con per-class AP tensors (full + sampled + 3 block × {full, sampled}) | ✅ committed |
| `vim/MSTemba_main.py` `run()` | Riscritta con `try/finally`, accetta `ckpt_manager`/`early_stopper`/`oar_signal`/`model_ema`/`start_epoch`/`metrics_logger` come kwargs opzionali | ✅ committed |
| `vim/extensions/checkpoint.py` | Nuovo modulo: atomic save, resume, SIGUSR2 handler, RNG state preservato, parametro `ema=None` opzionale | ✅ committed |
| `vim/extensions/metrics_logger.py` | Nuovo modulo: append-friendly per-epoch CSV + per-class CSV + finalize JSON ricostruito dal CSV | ✅ committed |
| `scripts/sanity_check_charades_i3d.sh` | Riscritto: direttive `#OAR` in header, env setup self-contained, `tee` invece di redirect (stderr non più nascosto) | ✅ committed |
| `scripts/check_env.sh` | Nuovo: 5/5 verifications (conda env, CUDA, deps, nvidia-smi, layout). Idempotente, exit 0/1 per uso in `&&`-chain | ✅ committed |
| `activate_mstemba.sh` (root) | Nuovo: source-only helper. Guard "must be sourced" via `BASH_SOURCE[0] == $0` | ✅ committed |
| `requirements_mstemba_v2.txt` | Skip mmcv, mmsegmentation; add tensorboard | ✅ committed |

### B.8 Bug noti (non bloccanti)
- **EMA mai aggiornato**: vedi A.4. Decisione supervisor: non usare. Codice morto presente ma inerte (`ema_state: None` nei checkpoint v2).
- **TSU sampled mAP gap in `-eval_only`**: vedi A.4. Causa sconosciuta dopo indagine completa. Workaround: usare `ap_full` per confronti TSU.
- **`val_dataloader(shuffle=True)`**: bug pre-esistente, non fixare ora (vedi A.4).
- **`lr` in formato scientifico nel CSV summary**: cosmetic. Fix con cast esplicito a `float` in `log_epoch_summary` se serve per analisi pandas.
- **Class names Charades non popolati**: `MetricsLogger(class_names=None)`. Per analisi "azione X vs Y" per nome occorre fornire mapping `class_id → action_name` (file `Charades_v1_classes.txt` esterno da agganciare).

---

## C. Storico esperimenti

### C.1 Baseline references (importate, seed=0, vecchia repo)
Eseguite il 2026-02-10 nella vecchia repo `MS-Temba/`. Importate in `baselines_reference/` come ground-truth per confronti futuri. Colonne **v2 (eval_only)** generate il 2026-05-20/21 caricando i `best_model.pth` legacy nella pipeline v2.

| Dataset | Backbone | Full C.1 | Full v2 | Δ Full | Sampled C.1 | Sampled v2 | Δ Sampled | Match |
|---|---|---|---|---|---|---|---|---|
| Charades | I3D | **24.74** | 24.7444 | 0.00 | 25.13 | 25.1262 | 0.00 | ✅ |
| Charades | CLIP-L | **32.40** | 32.4006 | 0.00 | 33.43 | 33.4276 | 0.00 | ✅ |
| MultiTHUMOS | I3D | **42.22** | 42.2151 | -0.01 | 44.05 | 44.0480 | 0.00 | ✅ |
| MultiTHUMOS | CLIP-L | **42.92** | 42.9213 | 0.00 | 44.08 | 44.0782 | 0.00 | ✅ |
| TSU | I3D | 32.57 | 31.8573 | -0.71 | **39.45** | 36.1347 | -3.31 | ⚠️ gap |
| TSU | CLIP-L | 42.71 | 39.8375 | -2.87 | **56.25** | 43.7059 | -12.54 | ⚠️ gap |

*Metrica primaria in bold. Vedi A.4 per gap TSU.*

### C.2 Validazioni pipeline
| Data | Esperimento | Risultato | Note |
|---|---|---|---|
| 2026-05-15 | Sanity Charades I3D 1ep (python diretto) | mAP 2.54 Full / 2.69 sampled | Pipeline OK. Modello 16,679,244 params (match paper 17M). |
| 2026-05-18 | Sanity Charades I3D 1ep in OAR batch (esterel35-2, A40) | mAP 2.54 / 2.69 (identico) | Validato workflow `oarsub -S` end-to-end. OAR stdout/stderr segregati in `experiments/oar_logs/`. |
| 2026-05-18 | Sanity resume da `checkpoint_last.pth` (epoca 1, `-epochs 2`) | Esegue solo l'epoca 1 (non rifà 0) | RNG state preservato. Best_val_map ricaricato correttamente. |
| 2026-05-19 | Sanity con metrics logger 1ep (esterel37-1, A40) | val_map 2.54, 157 classi tutte loggate | `metrics_per_class.csv` 158 righe + JSON ricostruito con 8 metriche per classe. |
| 2026-05-20 | `-eval_only` Charades I3D da legacy `best_model.pth` (esterel34-1, A40) | 24.7444 Full / 25.1262 sampled | Match C.1 entro <0.01 mAP. Pipeline `-eval_only` validata end-to-end. |
| 2026-05-20 | `-eval_only` Charades CLIP | 32.4006 / 33.4276 | Match C.1 entro <0.01. |
| 2026-05-21 | `-eval_only` TSU I3D (esterel21-1) | 31.8573 / 36.1347 | Full Δ -0.71, sampled Δ -3.31 vs C.1. Vedi A.4. |
| 2026-05-21 | `-eval_only` TSU CLIP | 39.8375 / 43.7059 | Full Δ -2.87, sampled Δ -12.54 vs C.1. Gap maggiore, deterministico (rerun identico). Vedi A.4. |
| 2026-05-21 | `-eval_only` MultiTHUMOS I3D (esterel22-1) | 42.2151 / 44.0480 | Match C.1 entro <0.01. |
| 2026-05-21 | `-eval_only` MultiTHUMOS CLIP | 42.9213 / 44.0782 | Match C.1 entro <0.01. |

### C.3 Esperimenti vecchia repo (riferimento storico)
Lavoro precedente nella vecchia repo `MS-Temba/`. Numeri di riferimento, non da rifare a meno di necessità specifica.

**Phase 1 — Baselines on Charades**:
- CLIP/seed0: 32.40 Full / 33.43 sampled (best ep 13, overfitting visibile post ep20)
- DINOv3/seed0 (orig): 25.44 / 25.94
- DINOv3 reg_v2/seed0: 25.21 / 25.82 (più stabile, curva pulita)

**Phase 2A — DINOv3 dense features**:
- CLS only: 25.21 (baseline)
- Mean patch / combined: pianificati ma non eseguiti / interrotti

**Phase 2B.1 — Skeleton single-stream**:
- SCD-Net/seed0: 9.46 Full / 9.73 sampled (best ep 21, ES ep 36)
- Bottleneck identificato: compressione Linear(4096→256) troppo aggressiva (16× vs CLIP 3×)
- Pattern per-classe complementare a CLIP (skel batte CLIP su 18/157 classi, 8 con margine ≥5 AP)
- Casi notevoli skeleton > CLIP: *closing closet* +28.8, *walking* +17.1, *sitting on bed* +6.3, *drinking* +5.8
- Casi sistematici skeleton < CLIP: object-discriminated throwing, fine-grained manipulation, minimal-body-motion actions
- **Decisione operativa**: ripartire da zero su MS-Temba-v2 perché nella vecchia repo skeleton funzionava solo con unidirezionalità (problema env/cuda)

**Phase 2B.2 — Skeleton interpolation investigation**:
- w=1 native (no interp): 8.52 mAP @ ep15
- w=1 + interp 1.5 FPS: 8.52 mAP (statisticamente identico)
- Conclusione: temporal alignment non è il bottleneck; il cap ~9.5 mAP è strutturale

**Anomalie Charades dataset** (sempre valide):
- c104/c105 mapping ambiguo
- c060 (46 train), c136 (35 train/143 test), c141 (506 train/20 test)
- Video anomalo `5UNDJ` per skeleton: gestire con zero-tensor fallback

### C.4 Documenti di riferimento
Tutti in `Timelines/`:
- `Fase_1_base_configuration/timeline_fase1_configurazione_base.md`
- `Fase_2_CLIP_vs_DINO/profonda_mstemba_clip_dino_comparazione.md`
- `Fase_3_DINO_experiments/dino_experiments_result_analysis.md`
- `Fase_4_skeleton_features/skeleton_features_timeline_pt2_investigation_and_tests.md`
- `Fase_5_fusion_of_features/timeline_fase5_fusion_of_features_literature_review.md`
- `Fase_6_model_improvements/MSTemba_roadmap.md` (themes T1–T6)
- `Papers/papers_bibliography.md`

---

## D. Conventions

### D.1 Naming
| Tipo | Pattern | Esempio |
|---|---|---|
| Branch | `feat/<short-name>` / `ablation/<theme>` | `feat/skeleton-bidir`, `ablation/T3-swa` |
| Experiment dir | `experiments/<exp>_<YYYYMMDD>_<HHMMSS>[_oar<jobid>]/` | `experiments/charades_i3d_baseline_perclass_20260520_165519_oar2530609/` |
| Run output | `<expdir>/training.log`, `<expdir>/checkpoint_{last,best}.pth`, `<expdir>/metrics_*.csv`, `<expdir>/metrics_per_class_final.json` | — |
| OAR logs | `experiments/oar_logs/<jobname>.<jobid>.{stdout,stderr}` | — |
| Config | `configs/experiments/<theme>_<variant>.yaml` | `configs/experiments/skeleton_gated.yaml` |

### D.2 Code style
- PyTorch ≥ 2.0, Python ≥ 3.10
- Type hints su tutte le nuove funzioni/metodi
- Tensor shapes nei docstring: `x: (B, T, D)`
- Nuovi moduli in `vim/extensions/`, NON modificare upstream direttamente — sottoclassare/estendere

### D.3 Discussion / code language
- Discussione in italiano
- Codice, commenti, equazioni in inglese

### D.4 Convenzioni naming nel paper
- **MS-Temba** o "the paper" = paper originale Pramanik et al. 2025 (arXiv:2501.06138)
- **MS-Temba v2** o "the extension" = mio lavoro
- **T1–T6** = sei temi da `Fase_6_model_improvements/MSTemba_roadmap.md`
- **TSU** = Toyota Smarthome Untrimmed; **Charades** = Charades multi-label

---

## E. Workflow tipico

### E.1 Ripresa lavoro
```bash
# Frontend → nodo GPU. Verificare prima su Monika quale esterel è libero
# (https://intranet.grid5000.fr/oar/Sophia/monika-prod.cgi) e PINNARLO.
ssh mdiiorio@fsophia.grid5000.fr
oarsub -I -q besteffort \
  -p "host='esterel-17.sophia.grid5000.fr'" \
  -l host=1/gpu=1,walltime=12

# Sul nodo: attiva env in un colpo solo
cd "$REPO"   # se $REPO è già esportato, altrimenti esportalo prima
source activate_mstemba.sh

# Verifica
bash scripts/check_env.sh
# Atteso: "== env check PASSED ==" con 5/5 verifiche
```

Se l'env non è ancora settato (shell nuova, $REPO non esportato):
```bash
export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
source $REPO/activate_mstemba.sh
```

### E.2 Lancio esperimento

**Smoke test** (interactive o batch):
```bash
# Interactive (output a video)
bash scripts/sanity_check_charades_i3d.sh

# Batch (script self-contained, esci dal nodo e lo lasci girare)
cd "$REPO"
oarsub -S scripts/sanity_check_charades_i3d.sh
# OAR stdout/stderr in experiments/oar_logs/sanity_charades_i3d.<jobid>.{stdout,stderr}
```

**Eval-only da checkpoint** (per generare per-class baseline o riprodurre numeri):
```bash
EXPDIR="$REPO/experiments/<name>_$(date +%Y%m%d_%H%M%S)_oar${OAR_JOB_ID:-manual}"
python vim/MSTemba_main.py \
  -dataset charades -mode rgb -backbone i3d \
  -model mstemba -num_clips 256 -skip 0 -comp_info False \
  -batch_size 5 -unisize True -alpha_l 1 -beta_l 0.05 \
  -rgb_root "$REPO/data/features/charades_i3d" \
  -train False -eval_only True \
  -resume "$REPO/baselines_reference/checkpoints/charades_i3d_seed0/best_model.pth" \
  -output_dir "$EXPDIR" -early_stop_patience 0 -epochs 1
# Output: $EXPDIR/metrics_per_class.csv + metrics_per_class_final.json
# Per altri dataset: cambiare -dataset, -backbone, -num_clips (2500 per TSU), -rgb_root, -resume.
# TSU richiede -batch_size 1 e GPU con ≥ 24 GiB VRAM (A40/A6000).
```

**Training reale** (con checkpoint + resume):
```bash
oarsub -S \
  --checkpoint 600 \
  -q besteffort \
  -p "host='esterel-17.sophia.grid5000.fr'" \
  -l host=1/gpu=1,walltime=12 \
  scripts/runs/<my_experiment>.sh

# In caso di kill / SIGUSR2 → checkpoint_last.pth è disponibile per resume:
python vim/MSTemba_main.py ... \
  -resume "$REPO/experiments/<expdir>/checkpoint_last.pth"
```

### E.3 Monitoraggio job in corso
```bash
# I tuoi job
oarstat -u $USER

# Dettagli specifici di un job
oarstat -j <jobid> -f | head -25

# Follow live
tail -f experiments/oar_logs/<jobname>.<jobid>.stdout
tail -f experiments/<expdir>/training.log

# Kill manuale
oardel <jobid>
```

### E.4 Aggiornamento contesto fine sessione
A fine sessione (prima di chiudere chat o nodo):
1. Aggiornare le sezioni **A.1–A.5** di questo file
2. Aggiungere nuove righe in **C.2** (validazioni pipeline) se hai fatto run di check
3. Aggiungere nuove righe in **C.3** se hai chiuso un esperimento
4. Commit + push: `git add Timelines/Context\ Document/Context_mstemba_project.md && git commit -m "Update context vN" && git push`
5. Ricaricare il file nel Project Claude (così la prossima chat lo legge aggiornato)

---

**Fine documento.** Per discussione su task specifici (skeleton fusion, ablation T1–T6, debugging), riferirsi ai documenti dedicati in `Timelines/`.