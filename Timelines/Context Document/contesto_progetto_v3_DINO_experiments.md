CONTESTO PROGETTO — MS-Temba / Charades Traineeship
Documento di contesto per nuove chat · Aggiornato: 27 marzo 2026
Istruzione d'uso: incolla questo documento all'inizio di ogni nuova chat per fornire tutto il contesto necessario. Aggiornalo dopo ogni sessione di lavoro significativa aggiungendo i nuovi risultati nella sezione "Risultati e stato corrente".

═══════════════════════════════════════════════════════════════
1. CHI SONO E COSA STO FACENDO
═══════════════════════════════════════════════════════════════
Nome: Matteo Di Iorio
Contesto: tirocinio magistrale, ricerca su Temporal Action Detection (TAD) multi-label
Dataset: Charades v1 — 157 classi di azioni domestiche, 7985 video train, 1863 test
Modello: MS-Temba (Multi-Scale Temporal Mamba), autore Pramanik et al. 2025, arXiv:2501.06138
Cluster: Grid5000 / ABACA, nodi esterel (Sophia Antipolis)
  Accesso GPU: oarsub -q besteffort -p esterelXX -l host=1/gpu=1,walltime=12 -I
Repository: /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/
Environment conda: mstemba_fresh
Features dir base: data/hf_features/Temporal_Action_Detection/
Frame video: data/charades_frames_24fps/ (subfolder per video, jpg a 24fps)

═══════════════════════════════════════════════════════════════
2. ARCHITETTURA MS-TEMBA (sintesi)
═══════════════════════════════════════════════════════════════
Pipeline: feature .npy → collation (pad a 256 finestre) → InputProjection (in_feat_dim→256, Sequential: Linear+LayerNorm+GELU+Dropout) → Block1 (1 SSM, 256-dim) → Block2 (2 SSM odd/even, 384-dim) → Block3 (3 SSM dilated, 576-dim) → interaction_block → classificatore (576→157). ~18M parametri totali.

Loss: BCE multi-label + diversity_loss (weight 100.0) + block auxiliary losses (beta=0.05)
Ottimizzatore: AdamW lr=5e-4, wd variabile
Scheduler: Cosine (warmup=5 ep, min_lr=1e-5)
EMA decay: 0.99996
Finestre temporali: window_size=16 frame per feature, pad a 256 finestre → ogni video = sequenza [256, D]
Dimensione interna: 256 (fissa, necessaria per confrontabilità con paper)

File chiave:
  vim/models_MSTemba.py — architettura modello
  vim/MSTemba_main.py — training loop, argparse
  vim/charades_dataloader.py — dataloader
  vim/dinov3_feature_extractor.py — estrazione feature DINOv3
  vim/extract_scdnet_features.py — allineamento skeleton features

═══════════════════════════════════════════════════════════════
3. FEATURE DISPONIBILI SU DISCO (tutte pronte)
═══════════════════════════════════════════════════════════════
Directory                                          | Shape      | Dtype   | Status
charades_features_clip/                            | [N, 768]   | float16 | ✅ pronte
charades_dinov3_vitl16_w16_24fps/                  | [N, 1024]  | float32 | ✅ pronte (CLS-only)
charades_dinov3_vitl16_w16_24fps_meanpatch/        | [N, 1024]  | float32 | ✅ pronte
charades_dinov3_vitl16_w16_24fps_combined/         | [N, 2048]  | float32 | ✅ pronte (CLS‖mean_patch)
charades_scdnet_w16/                               | [N, 4096]  | float32 | ✅ pronte (9848 file, allineate a CLIP)

NON SERVONO ALTRE ESTRAZIONI. Tutte le feature necessarie sono su disco.

Note tecniche:
- CLIP e DINOv3: estratte a ~1.5fps (window_size=16 su frame a 24fps)
- Skeleton SCD-Net: estratte a ~24fps nativi, poi allineate con average pooling window=16 → [N, 4096], N identico a CLIP
- Video anomalo: 5UNDJ (skel_windows=22 vs clip_T=292) — gestire con zero-tensor fallback nel dataloader
- Feature CLIP shape reale: [N, 768] non [N, 512] — CLIP ViT-B/16 ha dim interna 768

═══════════════════════════════════════════════════════════════
4. RISULTATI — PHASE 1 (Baselines Definitive) ✅ COMPLETATA
═══════════════════════════════════════════════════════════════
Config               | drop | dp   | wd   | Best ep | Full-val-MAP | Sampled-val-MAP | Stop ep
clip/seed0 ⭐         | 0.0  | 0.0  | 0.01 | 13      | 32.40        | 33.43           | 50
clip/seed1            | 0.0  | 0.0  | 0.01 | 13      | ~32.1        | ~33.2           | 50
clip/seed2            | 0.0  | 0.0  | 0.01 | 15      | ~31.8        | ~32.9           | 50
dinov3/seed0 (orig)   | 0.0  | 0.0  | 0.01 | 13      | 25.44        | 25.94           | 50
clip_reg v1/seed0     | 0.1  | 0.05 | 0.05 | 15      | 29.17        | 29.83           | 30
clip_reg v2/seed0     | 0.0  | 0.0  | 0.05 | 13      | 28.91        | 29.49           | 33
dinov3_reg v1/seed0   | 0.2  | 0.1  | 0.05 | 14      | 24.56        | 25.16           | 26
dinov3_reg v2/seed0 ⭐ | 0.05 | 0.05 | 0.05 | 13      | 25.21        | 25.82           | 28

Baseline definitive: CLIP orig (32.40 mAP), DINOv3 reg_v2 (25.21 mAP).

Conclusioni Phase 1:
- CLIP non migliora con regolarizzazione (prior semantico-linguistico già regolarizza)
- DINOv3 reg_v2 quasi pari all'originale ma con curva più stabile
- CLIP vince su classi semantiche (cooking 79.0, talking on phone 75.9, working on laptop 75.9)
- DINOv3 vince su classi di manipolazione oggetti visivamente specifici (opening refrigerator: CLIP 60.2→DINOv3 88.1)
- Training CLIP: best ep13 (MAP 32.40), poi decrescente. Train MAP 99.52 a ep49, gap 70.8pt

Anomalie dataset: c104/c105 (turn on/off light, same mapping), c010/c011, c043/c044. c060 (46 train), c136 (35 train vs 143 test).

═══════════════════════════════════════════════════════════════
5. RISULTATI — PHASE 2A (DINOv3 Dense Features) ✅ COMPLETATA — RISULTATO NULLO
═══════════════════════════════════════════════════════════════
Ipotesi: i patch token di DINOv3 (196 per frame, scartati dal CLS-only) contengono informazione complementare.
Letteratura: Jose et al. CVPR 2025 (CLS+patch_avg migliora su ImageNet/ADE20K), COMM, Talk2DINO.

ID   | Config                    | in_feat_dim | Best mAP | Δ vs CLS | Status
2A.0 | DINOv3 CLS reg_v2 (base) | 1024        | 25.21    | —        | ✅
2A.1 | DINOv3 mean_patch         | 1024        | <25.21   | negativo | ✅ peggiore
2A.2 | DINOv3 CLS+mean_patch     | 2048        | 25.20    | −0.01    | ✅ identico
2A.3 | DINOv3 attn pooling       | 1024        | —        | —        | ❌ CANCELLATO

CONCLUSIONE PHASE 2A: i patch tokens di DINOv3 NON aggiungono informazione discriminativa per Charades a ~1.5fps. Il CLS token cattura già tutto il segnale utile. L'attention pooling (2A.3) non è stato implementato perché se la concatenazione diretta non migliora, un pooling più sofisticato non cambierà il risultato. Phase 2A chiusa.

Training dynamics 2A.2 (combined): best ep15 (25.20 mAP), convergenza più rapida del CLS ma stesso picco. Overfitting: train 70.1 vs val 21.4 a ep30.

═══════════════════════════════════════════════════════════════
6. RISULTATI — PHASE 2B.1 (Skeleton Single-Stream) ✅ COMPLETATA
═══════════════════════════════════════════════════════════════
Config               | drop | dp  | wd   | Best ep | Full-val-MAP | Sampled-val-MAP | Stop ep
scdnet/seed0         | 0.1  | 0.1 | 0.05 | 21      | 9.46         | 9.73            | 36 (ES)

Confronto: CLIP 32.40 (−22.94), DINOv3 25.21 (−15.75).
Scenario C: mAP < 15, discriminatività standalone limitata ma COMPLEMENTARITÀ con CLIP confermata.

Training dynamics:
- Warmup lento: ~10 epoche per raggiungere zona del best
- Plateau stretto: val mAP 8.51–9.46 per 12 epoche (ep10–21) mentre train raddoppia
- Overfitting severo: train 52.45 vs val 7.26 allo stop (gap 45.2pt)
- Block-level: Block2 miglior val mAP (9.07), Block3 già in overfitting (8.92)
- Diversity loss: 0.0 tutto il training

Pattern per-classe — PUNTI DI FORZA skeleton (azioni posturali/gross motor):
  c151 Closing closet: 52.3 (vs CLIP 23.5, +28.8)
  c059 Drinking: 51.8 (vs CLIP 46.0, +5.8)
  c011 Sitting on bed: 43.0 (vs CLIP 36.7, +6.3)
  c123 Walking: 35.8 (vs CLIP 18.7, +17.1)
  c154 Sitting down: 31.6 (vs CLIP 30.5, +1.1)
  c118 Holding groceries: 15.9 (vs CLIP 7.1, +8.8)

Pattern per-classe — DEBOLEZZE skeleton (3 failure modes):
  1) Object-discriminated throwing: c045 book 0.17, c085 clothes 0.26, c064 pillow 0.64 (stessa traiettoria, diverso oggetto)
  2) Fine-grained manipulation: c060 opening box 0.37, c039 opening window 0.78
  3) Minimal body motion: c138 turning on TV 0.91, c095 playing phone 0.97

Complementarità quantificata:
  Skeleton batte CLIP su 18/157 classi (11.5%), 8 con margine ≥5 AP
  CLIP domina sul restante 88.5%
  Pattern IDEALE per gated fusion: skeleton cattura "come ti muovi", CLIP cattura "cosa tocchi"

Diagnosi: bottleneck nell'input projection Linear(4096→256) — compressione 16× vs 3× per CLIP. Train mAP raggiunge 52.5 (info presente) ma val si appiattisce a 9.5 (generalizzazione bloccata).

═══════════════════════════════════════════════════════════════
7. GERARCHIA INFORMATIVA STABILITA
═══════════════════════════════════════════════════════════════
CLIP (768-dim, 32.40 mAP)
  >> DINOv3 CLS (1024-dim, 25.21 mAP)
    = DINOv3 CLS+patch (2048-dim, 25.20 mAP)
      >> SCDNet skeleton (4096-dim, 9.46 mAP)

Questa gerarchia vale solo per performance standalone.
Il valore skeleton è nella complementarità per-classe, non nel mAP assoluto.

═══════════════════════════════════════════════════════════════
8. PIANO SPERIMENTALE — PROSSIMI PASSI (in ordine di priorità)
═══════════════════════════════════════════════════════════════

ID   | Config                        | Fusion  | Status      | Note
2B.2 | Score-level CLIP+skel         | score   | ⏳ NEXT     | Zero code changes, usa checkpoint esistenti
2B.5 | Gated fusion CLIP+skel        | gated   | ⏳ PRIORITÀ | Architettura raccomandata
2B.4 | Additive dual projector       | add     | ⏳          | Ablation per isolare contributo gate
2B.3 | Feature concatenation         | concat  | ⏳          | Lower bound (in_feat_dim=4864)
2B.6 | Skeleton-as-Query CA          | xattn   | ⏳ cond.    | Solo se gated ≥ +1.5 mAP
2B.7 | Visual-as-Query CA            | xattn   | ⏳ cond.    | Direzione preferita per l'asimmetria
2C.1 | Score-level CLIP+DINOv3       | score   | ⏳          | Fattibile subito
2C.3 | Gated CLIP+DINOv3             | gated   | ⏳          |
2D   | Three-stream gated            | gated   | ⏳ finale   | CLIP+DINOv3+Skeleton

Stime impatto:
  Score-level CLIP+skel (α=0.9): +0.3 a +0.6 mAP → 32.7–33.0
  Gated dual projector:           +1.1 a +2.6 mAP → 33.5–35.0
  Three-stream gated:             +2.6 a +6.6 mAP → 35.0–39.0

═══════════════════════════════════════════════════════════════
9. ARCHITETTURA GATED FUSION (già definita, da implementare)
═══════════════════════════════════════════════════════════════

class GatedDualProjection(nn.Module):
    def __init__(self, vis_dim, skel_dim, out_dim=256, drop_rate=0.1):
        super().__init__()
        self.proj_vis  = nn.Sequential(nn.Linear(vis_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate))
        self.proj_skel = nn.Sequential(nn.Linear(skel_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(drop_rate))
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
    
    def forward(self, f_vis, f_skel, skel_mask=None):
        h_v = self.proj_vis(f_vis)
        h_s = self.proj_skel(f_skel)
        g   = self.gate(torch.cat([h_v, h_s], dim=-1))
        if skel_mask is not None:
            g = g.masked_fill(skel_mask.unsqueeze(-1).unsqueeze(-1), 1.0)
        return g * h_v + (1.0 - g) * h_s

Design notes:
- Gate bias init=+0.5 → σ(0.5)≈0.62 favore CLIP (domina su 88.5% classi)
- proj_vis inizializzabile da checkpoint CLIP baseline (warm-start)
- proj_skel con LR ridotto (0.2× base) per evitare instabilità da 4096-dim
- Skeleton mask per video anomali (5UNDJ): forza g=1.0 (full visual weight)
- Parametri aggiuntivi: ~131K (+0.7%)

Implementazione richiede:
1. charades_dataloader.py: estendere per caricare 2 feature directory (CLIP+skeleton) in parallelo
2. models_MSTemba.py: sostituire InputProjection con GatedDualProjection
3. MSTemba_main.py: aggiungere CLI per secondary feature path, skeleton mask, gate bias init, per-module LR

═══════════════════════════════════════════════════════════════
10. CODICE — MODIFICHE GIÀ IMPLEMENTATE
═══════════════════════════════════════════════════════════════

10.1 models_MSTemba.py
  - LinearProjection esteso con drop_rate, Dropout dopo GELU
  - MSTemba.__init__ propagato drop_rate
  - self.proj: nn.Sequential(Linear, LayerNorm, GELU, Dropout)
  - Bug fix: permute(0,2,1) mantenuto (feature su disco [D,T], serve [T,D])

10.2 MSTemba_main.py
  - Early stopping: --early-stop-patience, --min-delta
  - run() riscritta: patience_counter, metrics_per_epoch.csv, run_config.json
  - save_checkpoint: extra=None per salvare patience_counter
  - Backbone mapping esteso: scdnet→4096, dinov3_combined→2048, dinov3_meanpatch→1024

10.3 charades_dataloader.py
  - Whitelist auto-transpose: aggiunto 4096 → (256, 512, 768, 1024, 2048, 4096)
  - BUG CRITICO FIXATO: senza 4096, video con N∈{256,512,768,1024} sarebbero transposti erroneamente

10.4 dinov3_feature_extractor.py
  - Pooling mode "combined": CLS‖mean_patch → [2*D]
  - CLI choices: --pooling {cls, pooler, mean_patch, combined}
  - Sharding: --shard_id, --num_shards
  - Skip existing: resume dopo interruzione

10.5 extract_scdnet_features.py
  - Lettura diretta da zip 93GB, average pooling window=16
  - Verifica allineamento vs CLIP, salvataggio atomico, skip_existing, sharding
  - Throughput: ~8.4 vid/s, ~20 min per 9848 video

10.6 Script in vim/scripts/ (tutti quelli esistenti)
  Training:
    run_charades_clip_reg_seed0.sh
    run_charades_clip_reg_v2_seed0.sh
    run_charades_dinov3_reg_seed0.sh
    run_charades_dinov3_reg_v2_seed0.sh
    run_charades_scdnet_seed0.sh               ← NUOVO
    run_charades_dinov3_meanpatch_seed0.sh      ← NUOVO
    run_charades_dinov3_combined_seed0.sh       ← NUOVO
  Estrazione:
    job_scdnet_align.oar.sh
    job_dinov3_meanpatch_extract.oar.sh         ← NUOVO (4 shard)
    job_dinov3_combined_extract.oar.sh          ← NUOVO (4 shard)

═══════════════════════════════════════════════════════════════
11. DOCUMENTI PRODOTTI
═══════════════════════════════════════════════════════════════
Documento                                        | Contenuto                                               | Status
deep_mstemba_clip_dino_comparison_updated.md      | Phase 1 CLIP vs DINOv3 reg_v2, 157 classi              | ✅
phase2b_skeleton_extraction_and_results.md        | Phase 2B unificato: estrazione, implementazione,        | ✅
                                                  | risultati, analisi per-classe, diagnosi, fusion         |
                                                  | roadmap, literature survey (13 sezioni)                 |
mstemba_comprehensive_experimental_analysis.md    | Analisi accademica completa: Phase 1 + 2A + 2B,        | ✅
                                                  | cross-phase comparison, 6 principal findings,           |
                                                  | fusion roadmap, references (9 sezioni + appendici)      |
mstemba_phase2a_dino_dense_features.md            | Piano tecnico DINOv3 dense features                     | ✅
mstemba_phase2a_dino_clip_literature_fusion.md    | Letteratura CLIP–DINOv3 fusion                         | ✅
mstemba_phase2b_skeleton_literature_fusion.md     | Letteratura skeleton fusion visual+skeleton              | ✅

═══════════════════════════════════════════════════════════════
12. WORKFLOW TIPICO SUL CLUSTER
═══════════════════════════════════════════════════════════════
# 1. Login e acquisizione nodo GPU
ssh mdiiorio@fsophia.grid5000.fr
oarsub -q besteffort -p esterel42 -l host=1/gpu=1,walltime=12 -I

# 2. Attivazione environment
cd /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba
source env_abaca.sh

# 3. Lancio training
bash vim/scripts/run_charades_NOME.sh
# oppure: python vim/MSTemba_main.py [args]

# 4. Monitoraggio
tail -f runs/charades/NOME_RUN/training.log
cat runs/charades/NOME_RUN/metrics_per_epoch.csv

# 5. Estrazione parallela con sharding (es. su 4 GPU, un terminale ciascuno)
python vim/dinov3_feature_extractor.py \
  --image_root_dir data/charades_frames_24fps \
  --save_dir data/hf_features/Temporal_Action_Detection/NOME_DIR \
  --window_size 16 --pooling TIPO --amp bf16 --save_dtype float32 \
  --shard_id {0,1,2,3} --num_shards 4

═══════════════════════════════════════════════════════════════
13. CONVENZIONI DI NAMING
═══════════════════════════════════════════════════════════════
Tipo             | Pattern                                        | Esempio
Run directory    | runs/charades/{backbone}_{config}/seed{N}      | runs/charades/scdnet/seed0
Feature dir      | charades_{backbone}_{params}/                  | charades_scdnet_w16/
Script training  | run_charades_{backbone}_{config}_seed{N}.sh    | run_charades_scdnet_seed0.sh
Checkpoint best  | runs/.../checkpoint_best.pth                   |
Metriche         | runs/.../metrics_per_epoch.csv                 |

═══════════════════════════════════════════════════════════════
14. LETTERATURA CHIAVE
═══════════════════════════════════════════════════════════════
Skeleton fusion:
  CLIP-MG (2025, arXiv:2506.16385): skeleton-as-Query per CLIP (+16.5pp NTU)
  SkeletonCLIP++ (2024): CLIP semantics per pesare frame skeleton
  Zhu et al. (ACM TOMM 2022, arXiv:2202.11374): two-stage fusion (+1.5–3.0 mAP)
  HCMFN (Hu et al., 2024): skeleton come spatial anchor (+2.1 mAP)

DINOv3 dense (risultato nullo per noi, ma utile come referenza):
  Jose et al. (CVPR 2025, arXiv:2412.16334): CLS+patch_avg migliora su ImageNet/ADE20K
  COMM (Jiang et al., arXiv:2310.08825): shallow=low-level, deep=semantics
  Talk2DINO (Barsellotti et al., arXiv:2411.19331): attention maps per spatial alignment

═══════════════════════════════════════════════════════════════
15. DOMANDE APERTE / DA CHIARIRE
═══════════════════════════════════════════════════════════════
- Verificare se SCD-Net su Charades_SCDNet_features2.zip è stato estratto con multi-stream o single-stream
- Chiarire questione unidirezionale vs bidirezionale in MS-Temba: il paper usa bidirezionale, verificare che la configurazione corrente sia bidirezionale per confrontabilità
- Per gated fusion: il dataloader deve caricare 2 feature directory in parallelo — serve modifica significativa
- Verificare se env_abaca.sh attiva già mstemba_fresh o serve conda activate separato
- Dimensione interna 256: NON cambiare, necessaria per confrontabilità con paper originale
- Le feature estratte (CLIP, DINOv3, skeleton) sono INDIPENDENTI dal modello (uni/bidirezionale): non serve riestrarre nulla se si cambia configurazione del modello
- Risposta alla domanda 5 del supervisor (temporal mismatch skeleton): le feature in charades_scdnet_w16/ sono GIÀ allineate a CLIP tramite average pooling window=16. Non c'è mismatch. Se il supervisor suggerisce di modificare la ground truth per skeleton raw a 24fps, è un approccio alternativo (training a risoluzione temporale diversa) ma le feature allineate che abbiamo sono corrette.