# Baselines Reference

Snapshot delle 6 baseline (TSU / Charades / MultiTHUMOS x I3D / CLIP) prodotte nella
repo vecchia `MS-Temba/` con seed=0. Tutti i run sono stati eseguiti il 2026-02-10
con stesso ambiente conda (`mstemba_fresh`) e stessa versione del codice.

**Non rilanciare** questi run. Servono come ground-truth interno per confronti futuri.

## Layout
baselines_reference/
├── logs/                                  # log di training, leggeri (testo)
│   ├── tsu_i3d_seed0/
│   │   ├── training.log                   # mAP per epoca + Final Full/sampled-val-map
│   │   └── run_meta.txt                   # git hash, GPU, env del run originale
│   ├── tsu_clip_seed0/                    # idem
│   ├── charades_i3d_seed0/                # idem
│   ├── charades_clip_seed0/               # idem
│   ├── multithumos_i3d_seed0/             # idem
│   └── multithumos_clip_seed0/            # idem
│
├── checkpoints/                           # best model (.pth), GITIGNORED
│   ├── tsu_i3d_seed0/best_model.pth       # vivono solo sul filesystem,
│   ├── tsu_clip_seed0/best_model.pth      # non vanno su GitHub
│   └── ... (4 altri)
│
├── scripts_historic/                      # bash script originali della vecchia repo
│   ├── run_tsu_i3d_seed0.sh               # NON eseguibili con upstream main.py:
│   ├── run_tsu_clip_seed0.sh              # usano arg custom -seed, -resume, -save_every
│   └── ... (4 altri)                      # che non sono nell'upstream
│
└── README.md                              # questo file
## Numeri di riferimento (seed=0)

| Dataset      | Backbone | Full-val mAP | sampled-val mAP |
|--------------|----------|--------------|------------------|
| TSU          | I3D      | 32.57        | **39.45**        |
| TSU          | CLIP-L   | 42.71        | **56.25**        |
| Charades     | I3D      | **24.74**    | 25.13            |
| Charades     | CLIP-L   | **32.40**    | 33.43            |
| MultiTHUMOS  | I3D      | **42.22**    | 44.05            |
| MultiTHUMOS  | CLIP-L   | **42.92**    | 44.08            |

**Convenzione metriche** (segue MS-TCT / MS-Temba paper Tab. 1):
- TSU -> riportato `sampled-val-map`
- Charades -> riportato `Full-val-map`
- MultiTHUMOS -> riportato `Full-val-map`

I valori in **grassetto** sono la metrica riportata in tabella del paper.

## Estrazione mAP da log
```bash
tail -3 baselines_reference/logs/tsu_i3d_seed0/training.log
```

## Note
- Differenza Full vs sampled per TSU e grande (~14-16 pp) perche:
  - Full valuta su finestre fisse,
  - sampled valuta su clip campionati uniformemente (piu favorevole su video lunghi).
- Su Charades e MultiTHUMOS le due metriche sono molto vicine (< 2 pp).
- Confronti con paper: i numeri TSU/Charades I3D risultano lievemente sotto al paper
  (~2-4 punti). Da verificare se e dovuto a seed singolo (paper potrebbe riportare
  media su piu seed) o a differenze di env/feature extraction.
