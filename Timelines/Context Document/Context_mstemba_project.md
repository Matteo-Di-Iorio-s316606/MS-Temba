# Context — MS-Temba Research

**Owner**: Matteo Di Iorio (master's thesis, traineeship at Inria Sophia)
**Project**: MS-Temba research extension — TAD on densely-labeled ADL benchmarks
**Document version**: v6 (skeleton E1+E2 launched, awaiting results)
**Last updated**: 2026-05-24

> **Come usare questo documento**: è il briefing che ogni nuova chat di Claude deve leggere prima.
> Sezione A è lo stato corrente (aggiornare a fine sessione).
> Sezione B è il setup tecnico (cambia di rado).
> Sezione C è lo storico esperimenti (append-only, mai cancellare).

---

## A. Stato corrente

### A.1 Active repo & branch
- **Repo locale**: `/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2/`
- **GitHub**: `git@github.com:Matteo-Di-Iorio-s316606/MS-Temba.git` (fork di `thearkaprava/MS-Temba`)
- **Branch attiva**: `feat/skeleton` (creata da `reproduce-baseline` il 2026-05-23)
- **Convention**: `main` traccia upstream (immodificata), `reproduce-baseline` = setup + infra training + fix minori + per-class eval, `feat/<nome>` per macro-esperimenti, `exp/<id>-<variant>` per run specifici (futuro)

### A.2 Dove sono nel piano
- ✅ Setup repo nuova, env conda `mstemba_v2`, sanity check OAR batch
- ✅ Infrastruttura training completa: `CheckpointManager`, `EarlyStopper`, `OARSignalHandler`, `MetricsLogger` (per-class CSV + JSON)
- ✅ Decisione EMA (NON usare, codice morto presente ma inerte)
- ✅ Branch `-eval_only` in `MSTemba_main.py`, branch dataset `multithumos`
- ✅ Per-class baselines 6 dataset generate via `-eval_only` (4/6 match esatto, 2/6 gap TSU documentato)
- ✅ **Branch `feat/skeleton` creata + scdnet routing in `MSTemba_main.py` (2 punti, righe 680 e 743)**
- ✅ **Dataloader skeleton patch: zero-tensor fallback per file mancanti**
- ✅ **Pre-compute SCD-Net w=16 (9848 file, T_median=45) in `data/features_derived/charades_scdnet_w16/`**
- ✅ **Whitelist Sophia sm_86 verificata via Monika (2026-05-24): esterel-33, -34, -35, -39, -40**
- ⏳ **E1 RUNNING** (skel w=1 native, T=2400, Linear proj, bidir) — `oar2540667` su esterel33-1 (A40)
- ⏳ **E2 RUNNING** (skel w=16, T=256, Linear proj, bidir) — `oar2540668`
- ⏳ E3/E4 (MLP projection variants) — pianificati, decisione dopo risultati E1/E2
- ⏳ Ablation T1–T6: roadmap esistente

### A.3 Prossimi task concreti
1. **Aspettare completion di E1 + E2** (ETA E1 ~12-24h, E2 ~3-4h). Quando finiscono:
   - Estrarre best val mAP da `metrics_per_class.csv` (colonna `ap_full`)
   - Generare confronto per-classe E2 vs CLIP baseline Charades (replica Exp 5 in bidir)
   - Verificare gate go/no-go (vedi A.5)
2. **Decisione su E3/E4** basata su E1/E2:
   - Se E2 ≥ 10.5 mAP val_full → procedi con E3/E4 (MLP projection 4096→1024→256)
   - Se E2 < 10.0 → debug v2 setup prima di andare oltre
3. **Decisione fusion direction** dopo skeleton standalone:
   - Se complementarità per-classe confermata (18+ classi skel > CLIP) → gated fusion con coverage matched (R3 piano originale)
   - Se non confermata → pivot a T1-T6 ablation

### A.4 Blockers / known issues
- **EMA — decisione supervisor presa (2026-05-20)**: NON usare EMA. Stato: default argparse `model_ema=True` ma `update()` mai chiamato, `ema_state: None` nei checkpoint v2. Convivere col codice morto.
- **TSU sampled mAP gap in `-eval_only`**: TSU I3D Δ_sampled=-3.31, TSU CLIP Δ_sampled=-12.54. Causa non identificata. Workaround: usare `ap_full` per confronti TSU. Per skel-vs-RGB intra-v2 ininfluente.
- **`val_dataloader(shuffle=True)`**: bug pre-esistente identico vecchia repo. NON FIXARE ora (cambierebbe i numeri delle baseline matchate). Eventuale fix futuro richiede rigenerazione di tutti i 6 CSV per-class.
- **CUDA kernel sm_86 only**: `selective_scan_cuda.cpython-310-x86_64-linux-gnu.so` di `mamba-1p1p1/` compilato con `TORCH_CUDA_ARCH_LIST="8.6"`. Crash con `RuntimeError: no kernel image is available for execution on the device` su sm_80 (A100), sm_89 (Ada), sm_90 (Hopper). **Vedi B.3 per whitelist confermata.**
- **NFS atomic write rotto** (lesson from precompute): la sequenza `open+write+close` su `/srv/storage/...` ha lazy directory entry. `os.replace(tmp, final)` chiamato subito dopo trova `ENOENT`. Workaround: direct write + sanity-check resume (loadable + shape check). Mai usare tmp+rename su NFS Sophia.
- Solo coda `besteffort` accessibile (job killabile, mitigato da CheckpointManager + SIGUSR2 handler).

### A.5 Metriche di riferimento per confronto skeleton
- **Charades baseline RGB (per-class CSV già generato)**: usare colonna `ap_full` di `baselines_reference/charades_clip_perclass/metrics_per_class.csv` (CLIP) e `baselines_reference/charades_i3d_perclass/metrics_per_class.csv` (I3D).
- **Gate post-E2**:
  - E2 val_full ≥ 10.5 → proceed E3/E4
  - E2 val_full < 10.0 → debug
  - E1 val_full ≥ E2 → T=2400 utile anche in bidir (rilevante per fusion design)
  - E1 val_full < E2 → label-noise overfit dominante (replica Exp 10 in bidir)
- **Pattern atteso** (da Exp 5 vecchia repo): ~18 classi con `ap_skel - ap_clip > 0`, top 4-5: *closing closet* (+28.8), *walking* (+17.1), *sitting on bed* (+6.3), *drinking* (+5.8). In bidir mi aspetto deltas comparable o leggermente più grandi.

---

## B. Setup tecnico

### B.1 Grid5000 Sophia
- Frontend: `mdiiorio@fsophia.grid5000.fr` (no GPU, solo submission/management OAR)
- **Coda accessibile**: solo `besteffort` (job killabile, checkpoint + SIGUSR2 handler obbligatori)
- **Monika** (GPU live status): https://intranet.grid5000.fr/oar/Sophia/monika-prod.cgi
- **Comandi OAR vanno SOLO dal frontend**, mai da un nodo computing (esterelXX). Da nodo: `oarsub: command not found`.

### B.2 Conda env `mstemba_v2`
- Python 3.10.13, torch 2.1.1+cu118, torchvision 0.16.1, torchaudio 2.1.1
- mamba_ssm 1.1.1 (editable, source `mamba-1p1p1/` locale)
- causal-conv1d 1.1.1 (da PyPI source, NON la 1.0.0 della cartella locale — API mismatch)
- setuptools `<70` (pin obbligatorio: torch 2.1.x importa `pkg_resources` rimosso da setuptools≥70)
- tensorboard
- Vedere `requirements_mstemba_v2.txt` per la lista completa

### B.3 CUDA module e GPU compatibility
- `module load cuda/11.8.0_gcc-10.4.0` (modulo Grid5000 Sophia per torch 2.1.1+cu118)
- Kernel `selective_scan_cuda` (`mamba-1p1p1/`) compilato con `TORCH_CUDA_ARCH_LIST="8.6"`.
  - Richiede GPU con compute capability **8.6 ESATTO** (Ampere consumer: A40, A6000).
  - **NON compatibile**: 8.0 (A100), 8.9 (Ada/L40/RTX 4090), 9.0 (Hopper/H100). Crash con `RuntimeError: no kernel image is available for execution on the device`.

**Whitelist Sophia (cc=8.6) confermata via Monika (2026-05-24)**:
| Host | GPU model | cc | Status |
|---|---|---|---|
| esterel-33 | (A40 expected, da verificare GPU model) | 8.6 | ✓ |
| esterel-34 | A40 confermato | 8.6 | ✓ |
| esterel-35 | A40 confermato | 8.6 | ✓ |
| esterel-39 | (A40 expected) | 8.6 | ✓ |
| esterel-40 | (A40 expected) | 8.6 | ✓ |

**Da NON usare** (compute capability incompatibile, anche se `major=8`):
| Host | cc | Architettura |
|---|---|---|
| esterel-17, -36, -37, -38 | 8.0 | A100 datacenter |
| esterel-41, -43, -44 | 8.9 | Ada Lovelace |
| esterel-42 | 9.0 | Hopper H100 NVL |
| esterel-15 .. -33 (pre-17) | 7.x | Turing/Volta |

Per submission OAR usare nello script:
```
#OAR -p "host='esterel-33.sophia.grid5000.fr' OR host='esterel-34.sophia.grid5000.fr' OR host='esterel-35.sophia.grid5000.fr' OR host='esterel-39.sophia.grid5000.fr' OR host='esterel-40.sophia.grid5000.fr'"
```

Plus defense in depth nello script (hard fail se CC != 8.6):
```bash
CC=$(python3 -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))" 2>/dev/null)
if [[ "$CC" != "8.6" ]]; then
    echo "[FATAL] GPU compute capability is $CC, expected 8.6. Abort." >&2
    exit 2
fi
```

`TORCH_CUDA_ARCH_LIST` esportato automaticamente: per A40/A6000 (Ampere): `"8.6"`. Per ricompilare in futuro su arch più ampia (es. supportare A100/Hopper), usare `"8.0;8.6;8.9;9.0"` — non testato, ~30-40 min compile.

### B.4 Layout repo
```
MS-Temba-v2/
├── vim/                          # codice MS-Temba upstream (TAD core)
│   ├── MSTemba_main.py           # training entrypoint (eval_only + multithumos + scdnet branch)
│   ├── models_MSTemba.py         # architettura
│   ├── charades_dataloader.py    # dataloader (generico, riusato da multithumos)
│   ├── extensions/               # NUOVE estensioni progetto
│   │   ├── checkpoint.py         # CheckpointManager + EarlyStopper + OARSignalHandler
│   │   └── metrics_logger.py     # per-class CSV + JSON
│   └── scripts/                  # script di run upstream + nostri
│       ├── run_MSTemba_Charades.sh   # upstream
│       ├── run_MSTemba_TSU.sh        # upstream
│       ├── run_charades_scdnet_w1_E1.sh    # NEW: E1 launch (w=1 native, T=2400)
│       └── run_charades_scdnet_w16_E2.sh   # NEW: E2 launch (w=16, T=256)
├── mamba-1p1p1/                  # mamba_ssm 1.1.1 (editable)
├── causal-conv1d/                # NON usato (1.0.0 incompatibile)
├── seg/                          # codice segmentazione VIM (irrilevante)
├── data/
│   ├── charades.json
│   ├── smarthome.json
│   ├── multithumos.json
│   ├── features/                 # symlink a feature .npy
│   │   ├── tsu_i3d, tsu_clip_l14
│   │   ├── charades_i3d, charades_clip
│   │   ├── multithumos_i3d, multithumos_clip
│   │   ├── charades_scdnet_full -> /srv/.../MS-Temba/data/.../charades_scdnet_full  (w=1 native, T~765)
│   │   └── charades_scdnet_w16  -> ../features_derived/charades_scdnet_w16          (w=16 pre-computed, T~47)
│   └── features_derived/         # NEW: feature derivate locali a v2
│       └── charades_scdnet_w16/  # 9848 file, T_median=45, generated 2026-05-24
├── baselines_reference/          # snapshot vecchia repo (logs + checkpoints + scripts storici)
├── experiments/                  # output run correnti (GITIGNORED)
│   └── oar_logs/                 # OAR stdout/stderr per job batch
├── scripts/                      # infrastruttura progetto
│   ├── setup_env_mstemba_v2.sh
│   ├── sanity_check_charades_i3d.sh
│   ├── check_env.sh
│   └── precompute_scdnet_w16.py  # NEW: pre-compute w=16 from w=1
├── activate_mstemba.sh           # source-only env activation
└── Timelines/                    # documentazione progetto
```

### B.5 Annotation paths
- Charades: `data/charades.json` (157 classi, 7985 train / 1863 test)
- TSU: `data/smarthome.json` (51 classi, splits CS_51, 185 test videos, mean T=1598, max T=3147)
- MultiTHUMOS: `data/multithumos.json` (65 classi)

### B.6 Backbone dimensions
| Backbone | in_feat_dim | Status |
|---|---|---|
| I3D | 1024 | feature pre-estratte |
| CLIP-L/14 | 768 | feature pre-estratte (da HuggingFace) |
| DINOv3 ViT-L/16 | 1024 (CLS), 2048 (combined) | feature in vecchia repo |
| **SCD-Net skeleton w=1 native** | **4096** | **feature in vecchia repo, symlinked. T_median=765 @ 24 FPS** |
| **SCD-Net skeleton w=16** | **4096** | **pre-computed locale v2, T_median=45 @ ~1.5 FPS (CLIP-aligned)** |

### B.7 Fix applicati al codice upstream
| File | Cosa | Status |
|---|---|---|
| `vim/MSTemba_main.py` riga 548 | Path Charades JSON → repo locale | ✅ committed |
| `vim/MSTemba_main.py` riga 555 | Path TSU JSON → repo locale | ✅ committed |
| `vim/MSTemba_main.py` branch dataset | `elif args.dataset == 'multithumos'` (classes=65) | ✅ committed |
| `vim/MSTemba_main.py` argparse | `-resume`, `-save_every_epoch`, `-early_stop_patience`, `-eval_only` | ✅ committed |
| `vim/MSTemba_main.py` branch `-eval_only` | per-class CSV+JSON via legacy checkpoint | ✅ committed |
| `vim/MSTemba_main.py` riga 680 + 743 | **`elif args.backbone == 'scdnet': in_feat_dim = 4096`** | ✅ committed feat/skeleton |
| `vim/MSTemba_main.py` `val_step` | 10-tuple con per-class AP tensors | ✅ committed |
| `vim/MSTemba_main.py` `run()` | try/finally, accetta ckpt_manager/early_stopper/oar_signal | ✅ committed |
| `vim/charades_dataloader.py` `__getitem__` | **Zero-tensor fallback per file mancanti** | ✅ committed feat/skeleton |
| `vim/extensions/checkpoint.py` | NEW: atomic save, resume, SIGUSR2, RNG state | ✅ committed |
| `vim/extensions/metrics_logger.py` | NEW: append-friendly per-class CSV + finalize JSON | ✅ committed |
| `scripts/sanity_check_charades_i3d.sh` | OAR-batch-safe (header directives, self-contained env) | ✅ committed |
| `scripts/check_env.sh` | 5/5 verifications, idempotente | ✅ committed |
| `scripts/precompute_scdnet_w16.py` | **NEW: pre-compute w=16 from w=1 native** | ✅ committed feat/skeleton |
| `vim/scripts/run_charades_scdnet_w1_E1.sh` | **NEW: E1 launch (w=1, T=2400, bidir)** | ✅ committed feat/skeleton |
| `vim/scripts/run_charades_scdnet_w16_E2.sh` | **NEW: E2 launch (w=16, T=256, bidir)** | ✅ committed feat/skeleton |
| `activate_mstemba.sh` (root) | source-only env helper | ✅ committed |
| `requirements_mstemba_v2.txt` | Skip mmcv/mmsegmentation, add tensorboard | ✅ committed |

### B.8 Bug noti (non bloccanti)
- **EMA mai aggiornato**: vedi A.4. Inerte.
- **TSU sampled mAP gap in `-eval_only`**: vedi A.4. Workaround: usare `ap_full`.
- **`val_dataloader(shuffle=True)`**: pre-esistente, NON fixare ora.
- **NFS lazy direntry su tmp+rename**: documentato in A.4 (lesson from precompute). Use direct write + sanity check.
- **`lr` in formato scientifico nel CSV summary**: cosmetic.
- **Class names Charades non popolati**: per analisi per nome occorre fornire `Charades_v1_classes.txt` esterno.

---

## C. Storico esperimenti

### C.1 Baseline references (importate, seed=0, vecchia repo)
Eseguite il 2026-02-10 nella vecchia repo `MS-Temba/`. Importate in `baselines_reference/` come ground-truth.

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
| 2026-05-15 | Sanity Charades I3D 1ep | mAP 2.54 / 2.69 | Pipeline OK. Modello 16,679,244 params |
| 2026-05-18 | Sanity Charades I3D 1ep OAR batch (esterel35-2, A40) | mAP 2.54 / 2.69 (identico) | Validato workflow `oarsub -S` |
| 2026-05-18 | Sanity resume da checkpoint_last (epoca 1, `-epochs 2`) | Esegue solo epoca 1 | RNG state preservato |
| 2026-05-19 | Sanity con metrics logger 1ep (esterel37-1) | val_map 2.54, 157 classi loggate | per-class CSV+JSON ricostruito |
| 2026-05-20 | `-eval_only` Charades I3D legacy (esterel34-1) | 24.7444 / 25.1262 | Match C.1 entro <0.01 |
| 2026-05-20 | `-eval_only` Charades CLIP | 32.4006 / 33.4276 | Match C.1 entro <0.01 |
| 2026-05-21 | `-eval_only` TSU I3D | 31.8573 / 36.1347 | Δ -0.71/-3.31. Vedi A.4 |
| 2026-05-21 | `-eval_only` TSU CLIP | 39.8375 / 43.7059 | Δ -2.87/-12.54. Vedi A.4 |
| 2026-05-21 | `-eval_only` MultiTHUMOS I3D | 42.2151 / 44.0480 | Match C.1 entro <0.01 |
| 2026-05-21 | `-eval_only` MultiTHUMOS CLIP | 42.9213 / 44.0782 | Match C.1 entro <0.01 |
| **2026-05-24** | **Pre-compute SCD-Net w=16 (frontend CPU, ~20 min)** | **9848 file written, T_in median=734, T_out median=45** | **Charades-only. NFS atomic write issue identified e workaround applicato (direct write + sanity resume)** |
| **2026-05-24** | **E1 launch (oar2540667, esterel33-1, A40 sm_86)** | **RUNNING** | **Skel w=1 native, T=2400, Linear, bidir. ETA ~12-24h** |
| **2026-05-24** | **E2 launch (oar2540668)** | **RUNNING** | **Skel w=16, T=256, Linear, bidir. ETA ~3-4h** |

### C.3 Esperimenti vecchia repo (riferimento storico)

**Phase 1 — Baselines on Charades**:
- CLIP/seed0: 32.40 Full / 33.43 sampled (best ep 13, overfitting post ep20)
- DINOv3/seed0: 25.44 / 25.94
- DINOv3 reg_v2/seed0: 25.21 / 25.82 (più stabile)

**Phase 2A — DINOv3 dense features**:
- CLS only: 25.21 (baseline); altre varianti non eseguite

**Phase 2B.1 — Skeleton single-stream (Exp 5)**:
- SCD-Net/seed0, w=16, unidir: 9.46 Full / 9.73 sampled (best ep 21, ES ep 36)
- Bottleneck identificato: Linear(4096→256) compressione 16× troppo aggressiva
- **Pattern complementare a CLIP**: skel batte CLIP su 18/157 classi, 8 con margine ≥5 AP
- Casi notevoli skel > CLIP: *closing closet* +28.8, *walking* +17.1, *sitting on bed* +6.3, *drinking* +5.8
- Casi sistematici skel < CLIP: object-discriminated throwing, fine-grained manipulation, minimal-body-motion
- **Decisione operativa**: ripartire da zero in v2 (vecchia repo bloccata in unidirezionalità per env/cuda)

**Phase 2B.2 — Skeleton interpolation investigation (Exp 10)**:
- w=1 native (no interp): 8.52 mAP @ ep15 unidir, num_clips=2400
- w=1 + interp 1.5 FPS: 8.52 mAP (identico)
- Conclusione: temporal alignment NON è il bottleneck; cap ~9.5 mAP strutturale (label noise + bottleneck Linear)

**Anomalie Charades dataset**:
- c104/c105 mapping ambiguo, c060 (46 train), c136 (35/143), c141 (506/20)
- Video anomalo `5UNDJ`: usato in vecchia repo come edge case, in v2 coperto da zero-tensor fallback (ma 9848/9848 confermato presenti, fallback è defense-in-depth)

### C.4 Documenti di riferimento
Tutti in `Timelines/`:
- `Fase_1_base_configuration/timeline_fase1_configurazione_base.md`
- `Fase_2_CLIP_vs_DINO/profonda_mstemba_clip_dino_comparazione.md`
- `Fase_3_DINO_experiments/dino_experiments_result_analysis.md`
- `Fase_4_skeleton_features/skeleton_features_timeline_pt2_investigation_and_tests.md` ← *aggiungere parte 3 per il restart in v2*
- `Fase_5_fusion_of_features/timeline_fase5_fusion_of_features_literature_review.md`
- `Fase_6_model_improvements/MSTemba_roadmap.md` (T1–T6)
- `Papers/papers_bibliography.md`

---

## D. Conventions

### D.1 Naming
| Tipo | Pattern | Esempio |
|---|---|---|
| Branch | `feat/<short-name>` / `ablation/<theme>` / `exp/<id>-<variant>` | `feat/skeleton`, `exp/E1-w1-linear` |
| Experiment dir | `experiments/<exp>_<YYYYMMDD>_<HHMMSS>_oar<jobid>/` | `experiments/charades_scdnet_w1_E1_20260524_121156_oar2540667/` |
| Run output | `<expdir>/training.log`, `<expdir>/checkpoint_{last,best}.pth`, `<expdir>/metrics_*.csv`, `<expdir>/metrics_per_class_final.json`, `<expdir>/run_meta.txt` | — |
| OAR logs | `experiments/oar_logs/<jobname>.<jobid>.{stdout,stderr}` | — |
| Config | `configs/experiments/<theme>_<variant>.yaml` | `configs/experiments/skeleton_gated.yaml` |

### D.2 Code style
- PyTorch ≥ 2.0, Python ≥ 3.10
- Type hints su tutte le nuove funzioni/metodi
- Tensor shapes nei docstring: `x: (B, T, D)`
- Nuovi moduli in `vim/extensions/`, NON modificare upstream direttamente — sottoclassare/estendere

### D.3 Discussion / code language
- Discussione in italiano, codice/commenti/equazioni in inglese

### D.4 Convenzioni naming nel paper
- **MS-Temba** o "the paper" = paper originale Pramanik et al. 2025 (arXiv:2501.06138)
- **MS-Temba v2** o "the extension" = mio lavoro
- **T1–T6** = sei temi da `MSTemba_roadmap.md`
- **TSU** = Toyota Smarthome Untrimmed; **Charades** = Charades multi-label

---

## E. Workflow tipico

### E.1 Ripresa lavoro
```bash
ssh mdiiorio@fsophia.grid5000.fr   # alias .ssh/config, solo da rete G5K/VPN
export REPO="/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
cd "$REPO"
source activate_mstemba.sh
bash scripts/check_env.sh   # atteso: 5/5 verifications PASSED
```

### E.2 Lancio esperimento — IMPORTANT: sintassi OAR Sophia
**Sempre dal frontend, mai da nodo computing.**

```bash
# Batch submission (workflow A: con OAR directives nell'header dello script)
cd "$REPO"
oarsub -S vim/scripts/run_charades_scdnet_w1_E1.sh
oarsub -S vim/scripts/run_charades_scdnet_w16_E2.sh
oarstat -u $USER     # verifica stato job
```

**Sintassi `-l` con/senza host**:
- Senza `-p` (OAR sceglie): `-l host=1/gpu=1,walltime=24` ✓
- Con `-p host='...'`: `-l /gpu=1,walltime=24` ✓ (omettere `host=`)
- Combinazione `-p host='X' -l host=1/gpu=1` → ERRORE "Tie job resource request for GPU to resources with GPU"

**Constraint dinamico per Ampere `sm_86`**: NON funziona su Sophia con `gpu_compute_capability_major='8'` (include anche A100/Ada che crashano). Usare whitelist statica (vedi B.3).

### E.3 Monitoraggio
```bash
oarstat -u $USER                          # tuoi job
oarstat -j <jobid> -f                     # dettagli job
tail -f experiments/<expdir>/training.log
```

### E.4 Aggiornamento contesto fine sessione
1. Aggiornare A.1–A.5 di questo file
2. Aggiungere righe in C.2 (validazioni) o C.3 (esperimenti chiusi)
3. Commit + push: `git add Timelines/Context\ Document/Context_mstemba_project.md && git commit -m "Update context vN" && git push`
4. Ricaricare il file nel Project Claude (così la prossima chat lo legge aggiornato)

---

**Fine documento.**