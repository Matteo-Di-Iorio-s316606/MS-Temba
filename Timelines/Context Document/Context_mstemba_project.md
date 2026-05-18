# Context — MS-Temba Research

**Owner**: Matteo Di Iorio (master's thesis, traineeship at Inria Sophia)  
**Project**: MS-Temba research extension — TAD on densely-labeled ADL benchmarks  
**Document version**: v3 (post repo-reset)  
**Last updated**: 2026-05-15

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
- **Convention**: `main` traccia upstream (immodificata), `reproduce-baseline` = setup + fix minori, `feat/<nome>` per macro-esperimenti

### A.2 Dove sono nel piano
- ✅ **Setup repo nuova** (MS-Temba-v2 da zero, fork pulito, env separato)
- ✅ **Baseline references importate** dalla vecchia repo (logs + checkpoint + script storici, 6 baseline)
- ✅ **Symlink feature dataset** dalla vecchia repo
- ✅ **Env conda nuovo `mstemba_v2`** funzionante, pipeline validata via sanity check
- ⏳ **Infrastruttura training**: checkpoint mgmt + metriche per-classe ordinate (PROSSIMO)
- ⏳ **Skeleton features bidirezionali**: riavvio esperimenti su repo pulita (DOPO infra)
- ⏳ **Ablation T1–T6**: roadmap esistente (vedi `Timelines/Fase_6_model_improvements/MSTemba_roadmap.md`)

### A.3 Prossimi task concreti
1. Sistemare `scripts/sanity_check_charades_i3d.sh` (in batch mode esce silenzioso, comando python diretto funziona)
2. Implementare gestione checkpoint robusta (atomic save, resume da `checkpoint_last.pth`, scheduler+optimizer+EMA state)
3. Logging metriche per-classe ordinate in CSV/JSON strutturato
4. Branch `feat/skeleton-bidir`: ripartire dagli esperimenti skeleton (rimosso vincolo unidirezionalità)

### A.4 Blockers / domande aperte
- Path TSU `smarthome_CS_51.json` ancora hardcoded in `vim/MSTemba_main.py` riga 555 → fix necessario al primo TSU run
- `causal-conv1d` cartella locale (versione 1.0.0) incompatibile con `mamba_ssm 1.1.1` → usata versione 1.1.1 da PyPI buildata da source
- `mmcv` e `mmsegmentation` rimossi (servivano solo a `seg/`, non a TAD)
- `tensorboard` aggiunto (mancava in upstream)

---

## B. Setup tecnico

### B.1 Compute environment
- **Cluster**: Grid5000 Sophia, principalmente `esterel`
- **GPU tipiche**: Quadro RTX 8000 (46 GB VRAM, compute capability 7.5), occasionali A100/H100
- **OAR command standard**: `oarsub -q besteffort -p "cluster='esterel'" -l host=1/gpu=1,walltime=12 -I`
- **Frontend**: `mdiiorio@fsophia.grid5000.fr` (no GPU, solo gestione)

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
- Su GPU diverse da RTX 8000, controllare `TORCH_CUDA_ARCH_LIST` (esportato automaticamente nello script setup, ma utile sapere: RTX 8000 = 7.5, A100 = 8.0, H100 = 9.0)

### B.4 Layout repo
MS-Temba-v2/
├── vim/                          # codice MS-Temba upstream (TAD core)
│   ├── MSTemba_main.py          # training entrypoint
│   ├── models_MSTemba.py        # architettura
│   ├── charades_dataloader.py   # dataloader
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
│   ├── checkpoints/              # best_model.pth (GITIGNORED)
│   ├── scripts_historic/         # bash script vecchia repo (NON eseguibili upstream)
│   └── README.md
├── experiments/                  # output esperimenti correnti (GITIGNORED)
├── scripts/                      # infrastruttura progetto (non runs di training)
│   ├── setup_env_mstemba_v2.sh   # build env da zero
│   └── sanity_check_charades_i3d.sh
├── Timelines/                    # documentazione progetto (questo file e altri)
├── requirements_mstemba_v2.txt
└── README.md (upstream)

### B.5 Annotation paths
- Charades: `data/charades.json` (157 classi, 7985 train / 1863 test)
- TSU: `data/smarthome.json` (51 classi, splits CS_51)
- MultiTHUMOS: `data/multithumos.json` (65 classi)

### B.6 Backbone dimensions
| Backbone | in_feat_dim | Status |
|---|---|---|
| I3D | 1024 | feature pre-estratte |
| CLIP-L/14 | 768 | feature pre-estratte (da HuggingFace) |
| DINOv3 ViT-L/16 | 1024 (CLS), 2048 (combined) | feature in vecchia repo |
| SCD-Net skeleton | 4096 | feature in vecchia repo |

### B.7 Fix applicati al codice upstream
| File | Riga | Fix | Status |
|---|---|---|---|
| `vim/MSTemba_main.py` | 548 | Path Charades JSON → repo locale | ✅ committed |
| `vim/MSTemba_main.py` | 555 | Path TSU JSON → repo locale | ⏳ da fare prima del primo TSU run |
| `requirements_mstemba_v2.txt` | — | Skip mmcv, mmsegmentation; add tensorboard | ✅ committed |

---

## C. Storico esperimenti

### C.1 Baseline references (importate, seed=0, vecchia repo)
Eseguite il 2026-02-10 nella vecchia repo `MS-Temba/`. Importate in `baselines_reference/` come ground-truth per confronti futuri.

| Dataset | Backbone | Full-val mAP | sampled-val mAP | Metrica riportata |
|---|---|---|---|---|
| TSU | I3D | 32.57 | **39.45** | sampled |
| TSU | CLIP-L | 42.71 | **56.25** | sampled |
| Charades | I3D | **24.74** | 25.13 | Full |
| Charades | CLIP-L | **32.40** | 33.43 | Full |
| MultiTHUMOS | I3D | **42.22** | 44.05 | Full |
| MultiTHUMOS | CLIP-L | **42.92** | 44.08 | Full |

### C.2 Validazioni pipeline
| Data | Esperimento | Risultato | Note |
|---|---|---|---|
| 2026-05-15 | Sanity Charades I3D 1ep | mAP 2.54 (Full) / 2.69 (sampled) | Pipeline OK. 1 epoca su 50, valore basso atteso. Modello 16,679,244 params (match paper 17M). |

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
- Bottleneck identificato: compressione Linear(4096→256) troppo aggressiva
- Pattern per-classe complementare a CLIP (skel batte CLIP su 18/157 classi)
- **Decisione operativa**: ripartire da zero su MS-Temba-v2 perché nella vecchia repo skeleton funzionava solo con unidirezionalità (problema env/cuda)

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
| Experiment dir | `experiments/<exp>_<YYYYMMDD>_<HHMMSS>/` | `experiments/skeleton_gated_20260520_143000/` |
| Run output | `<expdir>/training.log`, `<expdir>/checkpoint_best.pth`, `<expdir>/metrics.csv` | — |
| Config | `configs/experiments/<theme>_<variant>.yaml` | `configs/experiments/skeleton_gated.yaml` |

### D.2 Code style
- PyTorch ≥ 2.0, Python ≥ 3.10
- Type hints su tutte le nuove funzioni/metodi
- Tensor shapes nei docstring: `x: (B, T, D)`
- Nuovi moduli in `vim/extensions/`, NON modificare `models/temba_block.py` o `models/ms_fuser.py` direttamente — sottoclassare

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
# Frontend → nodo GPU
ssh mdiiorio@fsophia.grid5000.fr
oarsub -q besteffort -p "cluster='esterel'" -l host=1/gpu=1,walltime=12 -I

# Sul nodo
cd /srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mstemba_v2
module load cuda/11.8.0_gcc-10.4.0

# Verifica env attivo
python -c "import torch, mamba_ssm; print(torch.__version__, mamba_ssm.__file__)"
```

### E.2 Lancio esperimento
```bash
# Esempio (uno script per esperimento, in scripts/runs/)
bash scripts/runs/skeleton_gated_seed0.sh
```

### E.3 Aggiornamento contesto fine sessione
A fine sessione (prima di chiudere chat o nodo):
1. Aggiornare la sezione **A.1–A.4** di questo file
2. Aggiungere nuove righe in **C.2** (validazioni pipeline) se hai fatto run di check
3. Aggiungere nuove righe in **C.3** se hai chiuso un esperimento
4. Commit + push: `git add Timelines/Context\ Document/Context_mstemba_project.md && git commit -m "Update context" && git push`
5. Ricaricare il file nel Project Claude (così la prossima chat lo legge aggiornato)

---

**Fine documento.** Per discussione su task specifici (skeleton fusion, ablation T1–T6, debugging), riferirsi ai documenti dedicati in `Timelines/`.