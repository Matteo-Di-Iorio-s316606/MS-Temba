# Baselines Reference

Snapshot delle 6 baseline (TSU/Charades/MultiTHUMOS × I3D/CLIP) prodotte nella repo vecchia
`MS-Temba/` con seed=0.

**Non rilanciare** questi run. Servono come ground-truth per confronti futuri.

## Layout
- `logs/<dataset>_<backbone>_seed0/training.log` — mAP per epoca
- `logs/<dataset>_<backbone>_seed0/run_meta.txt`  — git hash, GPU, env del run originale
- `checkpoints/<dataset>_<backbone>_seed0/*.pth`  — best model (per re-eval con codice nuovo)
- `scripts_historic/`                              — bash script originali, **non eseguibili**
  con il `MSTemba_main.py` upstream (usano argomenti custom della vecchia repo)

## Estrarre il mAP finale
```bash
tail -3 baselines_reference/logs/<dataset>_<backbone>_seed0/training.log
```

## Numeri di riferimento (da popolare)
| Dataset      | Backbone | Paper mAP | Repro mAP |
|--------------|----------|-----------|-----------|
| TSU          | I3D      | ??        | ??        |
| TSU          | CLIP-L   | ??        | ??        |
| Charades     | I3D      | ??        | ??        |
| Charades     | CLIP-L   | ??        | ??        |
| MultiTHUMOS  | I3D      | ??        | ??        |
| MultiTHUMOS  | CLIP-L   | ??        | ??        |

Numeri "Paper mAP" da verificare nel paper MS-Temba (Tab. 1).
Numeri "Repro mAP" da estrarre con `tail` sui training.log.
