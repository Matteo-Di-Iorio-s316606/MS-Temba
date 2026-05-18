CONTESTO PROGETTO — MS-Temba / Charades Traineeship
Documento di contesto per nuove chat · Aggiornato: 27 marzo 2026
Istruzione d'uso: incolla questo documento all'inizio di ogni nuova chat per fornire tutto il contesto necessario. Aggiornalo dopo ogni sessione di lavoro significativa aggiungendo i nuovi risultati nella sezione "Risultati e stato corrente".

1. Chi sono e cosa sto facendo
Nome: Matteo Di Iorio
Contesto: tirocinio magistrale, ricerca su Temporal Action Detection (TAD) multi-label
Dataset: Charades v1 — 157 classi di azioni domestiche, 7985 video train, 1863 test
Modello: MS-Temba (Multi-Scale Temporal Mamba), autore Pramanik et al. 2025, arXiv:2501.06138
Cluster: Grid5000 / ABACA, nodi esterel (Sophia Antipolis). Accesso GPU via: oarsub -q besteffort -p esterelXX -l host=1/gpu=1,walltime=12 -I
Repository: /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba/
Environment conda: mstemba_fresh
Features dir base: data/hf_features/Temporal_Action_Detection/

2. Architettura MS-Temba (sintesi)
Pipeline: feature .npy → collation (pad a 256 finestre) → InputProjection (in_feat_dim→256, Sequential: Linear+LayerNorm+GELU+Dropout) → Block1 (1 SSM, 256-dim) → Block2 (2 SSM odd/even, 384-dim) → Block3 (3 SSM dilated, 576-dim) → interaction_block → classificatore (576→157). ~18M parametri totali.

Loss: BCE multi-label + diversity_loss (weight 100.0) + block auxiliary losses (beta=0.05)
Ottimizzatore: AdamW lr=5e-4, wd variabile
Scheduler: Cosine (warmup=5 ep, min_lr=1e-5)
EMA decay: 0.99996
Finestre temporali: window_size=16 frame per feature, pad a 256 finestre → ogni video = sequenza [256, D]

File chiave:

vim/models_MSTemba.py — architettura modello
vim/MSTemba_main.py — training loop, argparse
vim/charades_dataloader.py — dataloader
vim/dinov3_feature_extractor.py — estrazione feature DINOv3
vim/extract_scdnet_features.py — allineamento skeleton features
3. Feature disponibili su disco
Directory	Shape	Dtype	Status
charades_features_clip/	[N, 768]	float16	✅ pronte
charades_dinov3_vitl16_w16_24fps/	[N, 1024]	float32	✅ pronte (CLS-only)
charades_scdnet_w16/	[N, 4096]	float32	✅ pronte (9848 file, allineate a CLIP)
charades_dinov3_vitl16_w16_24fps_meanpatch/	[N, 1024]	float32	⏳ estrazione in corso (4 shard su 4 GPU)
charades_dinov3_vitl16_w16_24fps_combined/	[N, 2048]	float32	⏳ da estrarre (dopo mean_patch)
Note tecniche:

CLIP e DINOv3 sono estratte a ~1.5fps (window_size=16 su frame a 24fps), non a 24fps
Skeleton SCD-Net: estratte a ~24fps nativi, poi align con average pooling window=16 → [N, 4096], N identico a CLIP
Video anomalo: 5UNDJ (skel_windows=22 vs clip_T=292) — gestire con zero-tensor fallback nel dataloader
Feature CLIP shape reale: [N, 768] non [N, 512] — CLIP ViT-B/16 ha dim interna 768
Frame video per estrazione DINOv3 dense: data/charades_frames_24fps/ (cartella con subfolder per video, jpg a 24fps)
4. Risultati sperimentali — Phase 1 (Baselines Definitive)
4.1 Tabella risultati completa
Config	drop	dp	wd	Best ep	Full-val-MAP	Sampled-val-MAP	Stop ep
clip/seed0 ⭐	0.0	0.0	0.01	13	32.40	33.43	50
clip/seed1	0.0	0.0	0.01	13	~32.1	~33.2	50
clip/seed2	0.0	0.0	0.01	15	~31.8	~32.9	50
dinov3/seed0 (orig)	0.0	0.0	0.01	13	25.44	25.94	50
clip_reg v1/seed0	0.1	0.05	0.05	15	29.17	29.83	30
clip_reg v2/seed0	0.0	0.0	0.05	13	28.91	29.49	33
dinov3_reg v1/seed0	0.2	0.1	0.05	14	24.56	25.16	26
dinov3_reg v2/seed0 ⭐	0.05	0.05	0.05	13	25.21	25.82	28
Baseline definitive: CLIP orig (32.40 mAP), DINOv3 reg_v2 (25.21 mAP).

Conclusioni Phase 1:

CLIP non migliora con regolarizzazione (il prior semantico-linguistico già regolarizza implicita)
DINOv3 reg_v2 è quasi pari all'originale (-0.23 mAP) ma con curva più stabile
Il vantaggio DINOv3 su classi posturali del run originale era artefatto di overfitting — con reg_v2 CLIP vince su TUTTE le classi posturali
4.2 Pattern DINOv3 reg_v2 corretto (non le classi posturali!)
DINOv3 vince su classi di manipolazione oggetti visivamente specifici:

Opening a refrigerator (c143): CLIP 60.2 → DINOv3 88.1 (+27.9)
Putting a broom somewhere (c099): CLIP 27.4 → DINOv3 53.0 (+25.6)
Holding a vacuum (c137): CLIP 51.3 → DINOv3 74.2 (+22.9)
Putting a blanket somewhere (c071): CLIP 18.5 → DINOv3 38.8 (+20.3)
CLIP vince su classi semanticamente descrittive: cooking (79.0), talking on phone (75.9), working on laptop (75.9), holding a broom (75.4).

4.3 Curve training CLIP (epoche 0–49)
Best: ep13 (MAP 32.40), poi monotonicamente decrescente fino a 28.73 a ep49. Train MAP: 40.43 (ep13) → 99.52 (ep49). Gap train-val a ep49: ~70.8 pt.

4.4 Anomalie dataset rilevate
Classi strutturalmente irrisolvibili: c104/c105 (turn on/off light, same mapping), c010/c011 (sit on/at table), c043/c044 (take box from/from box). Dataset issues: c060 (46 train solo), c136 (35 train vs 143 test), c141 (506 train vs 20 test).

5. Risultati sperimentali — Phase 2B.1 (Skeleton Single-Stream) ✅ COMPLETATO
5.1 Risultati principali
Config	drop	dp	wd	Best ep	Full-val-MAP	Sampled-val-MAP	Stop ep
scdnet/seed0	0.1	0.1	0.05	21	9.46	9.73	36 (ES)

Confronto vs baselines: CLIP 32.40 → Skeleton 9.46 (−22.94), DINOv3 25.21 → Skeleton 9.46 (−15.75)
Scenario: C (mAP < 15) — skeleton features hanno discriminatività standalone limitata ma pattern per-classe complementare a CLIP.

5.2 Training dynamics
Warmup lento: ~10 epoche per raggiungere la zona del best (val mAP 8.51 a ep10 vs 9.46 a ep21)
Plateau stretto: val mAP oscilla tra 8.51–9.46 per 12 epoche (ep10–21) mentre train mAP raddoppia
Overfitting severo: train mAP 13.23 (ep21) → 52.45 (ep36), val mAP 9.46 → 7.26. Gap a stop: 45.2 pt
Diversity loss: 0.0 per tutto il training — nessun pattern degenerato nei SSM blocks
Block-level: Block2 ha il miglior val mAP (9.07), Block3 già in leggero overfitting (8.92)

5.3 Pattern per-classe skeleton (best epoch 21, sampled val)
Skeleton forte su azioni posturali/gross motor:
- c151 Closing closet: 52.3 (vs CLIP 23.5, +28.8)
- c059 Drinking: 51.8 (vs CLIP 46.0, +5.8)
- c011 Sitting on bed: 43.0 (vs CLIP 36.7, +6.3)
- c123 Walking: 35.8 (vs CLIP 18.7, +17.1)
- c154 Sitting down: 31.6 (vs CLIP 30.5, +1.1)
- c097 Watching TV: 30.5 (vs CLIP 30.5, +0.0)

Skeleton debole su azioni object-defined:
- c045 Throwing book: 0.17 (vs CLIP 8.2) — throwing trajectory identica per tutti gli oggetti
- c085 Throwing clothes: 0.26 (vs CLIP 7.1) — idem
- c060 Opening box: 0.37 (vs CLIP 14.1) — manipolazione fine
- c046 Opening refrigerator: 1.61 (vs CLIP 60.2) — minimal body motion
- c117 Talking on phone: 2.57 (vs CLIP 75.9) — identico a "sitting still"

Complementarità: skeleton batte CLIP su 18/157 classi (11.5%), di cui 8 con margine ≥5 AP. CLIP domina sul restante 88.5%. Pattern complementare ideale per gated fusion.

5.4 Diagnosi principale
Bottleneck: compressione Linear(4096→256) troppo aggressiva (16×) vs CLIP (768→256, 3×). Training mAP raggiunge 52.5 dimostrando che l'informazione è presente nelle features, ma la generalizzazione è bloccata dall'input projection.

Rimedi identificati (in ordine di priorità):
1. Gated fusion con CLIP (priorità massima — il valore skeleton è nella complementarità)
2. PCA pre-compressione 4096→1024 (zero parametri extra)
3. Proiezione a 2 stadi: Linear(4096→1024) + Linear(1024→256)
4. Regolarizzazione più forte (drop=0.2/0.3, wd=0.10)

6. Risultati sperimentali — Phase 2A (DINOv3 Dense Features) ⏳ IN CORSO
6.1 Stato attuale
Estrazione mean_patch: in corso su 4 GPU con sharding (4 shard), stimata ~14h/shard su alcune GPU
Estrazione combined: da lanciare dopo completamento mean_patch
Training 2A.1 (mean_patch): da lanciare dopo estrazione
Training 2A.2 (combined): da lanciare dopo estrazione

6.2 Esperimenti pianificati
ID	Config	in_feat_dim	Feature	Status
2A.0	DINOv3 CLS reg_v2 (baseline)	1024	CLS	✅ done (25.21 mAP)
2A.1	DINOv3 mean patch	1024	mean(patches)	⏳ estrazione in corso
2A.2	DINOv3 CLS + mean patch	2048	CLS‖mean(patch)	⏳ da estrarre
2A.3	DINOv3 attn pooling	1024	attn-weighted patches	⏳ conditional

Configurazione 2A.1: drop=0.05, dp=0.05, wd=0.05, patience=15, in_feat_dim=1024
Configurazione 2A.2: drop=0.1, dp=0.1, wd=0.05, patience=15, in_feat_dim=2048

7. Piano sperimentale — Phase 2B (Skeleton Fusion)
ID	Config	Fusion	Status	Note
2B.1	Skeleton single-stream	single (in_feat_dim=4096)	✅ 9.46 mAP	Completato
2B.2	Score-level fusion (CLIP+skel logit avg)	score	⏳ NEXT	Zero code changes
2B.3	Feature concatenation	concat (in_feat_dim=4864)	⏳	Lower bound
2B.4	Additive dual projector	add	⏳	
2B.5	Gated dual projector	gated	⏳ PRIORITÀ	Architettura raccomandata
2B.6	Skeleton-as-Query CA	xattn_sv	⏳ conditional	
2B.7	Visual-as-Query CA	xattn_vs	⏳ conditional	

Architettura gated fusion (2B.5) — già definita:
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

8. Piano sperimentale — Phase 2C (CLIP + DINOv3 Fusion, dipende da 2A)
ID	Config	Fusion	Lit.
2C.1	CLIP + DINOv3 concat	concat (in_feat_dim=2816)	COMM
2C.3	CLIP + DINOv3 gated	gated	GCTF VideoMamba
2C.4	CLIP + DINOv3 cross-attn	xattn	Talk2DINO

9. Piano sperimentale — Phase 2D (Three-Stream Fusion)
Target finale: gated fusion su tutti e tre gli stream (CLIP + DINOv3 + Skeleton). Atteso: 35–39 mAP (+3 a +7 vs CLIP baseline).

10. Codice — Modifiche già implementate
10.1 models_MSTemba.py
LinearProjection esteso con drop_rate=0.0, Dropout aggiunto dopo GELU
MSTemba.__init__ esteso con drop_rate=0.0, propagato a tutte le LinearProjection
self.proj cambiato da nn.Linear a nn.Sequential(Linear, LayerNorm, GELU, Dropout)
Bug fix: permute(0,2,1) mantenuto (feature su disco sono [D,T], serve [T,D] prima di Linear)
10.2 MSTemba_main.py
Aggiunto --early-stop-patience (default 15) e --min-delta (default 0.01) all'argparse
Funzione run() riscritta: early stopping con patience_counter, metrics_per_epoch.csv, run_config.json, fix bug double save
save_checkpoint esteso con extra=None per salvare patience_counter nel resume
Backbone mapping esteso con:
  - scdnet → 4096
  - dinov3_combined → 2048
  - dinov3_meanpatch → 1024
10.3 charades_dataloader.py
Whitelist auto-transpose aggiornata: aggiunto 4096 a (256, 512, 768, 1024, 2048) → (256, 512, 768, 1024, 2048, 4096)
Motivo: senza fix, video con esattamente 256/512/768/1024 finestre temporali con skeleton features [N, 4096] sarebbero transposti erroneamente
10.4 dinov3_feature_extractor.py
Aggiunto pooling mode "combined" in encode_image_global: CLS‖mean_patch → [2*D]
Aggiunto pooling mode "combined" alle CLI choices: --pooling {cls, pooler, mean_patch, combined}
Sharding supportato: --shard_id, --num_shards per parallelizzazione estrazione
Skip existing integrato: se output .npy esiste, viene saltato (resume dopo interruzione)
10.5 extract_scdnet_features.py
Script completo per allineamento skeleton features. Funzionalità: lettura diretta dallo zip senza decompressione, average pooling window=16, verifica allineamento vs CLIP, salvataggio atomico (.npy.tmp → .npy), skip_existing per resume, sharding. Throughput: ~8.4 vid/s, ~20 min per 9848 video.
10.6 Script di training in vim/scripts/
run_charades_clip_reg_seed0.sh (drop=0.1, dp=0.05, wd=0.05, patience=15)
run_charades_dinov3_reg_seed0.sh (drop=0.2, dp=0.1, wd=0.05, patience=12)
run_charades_clip_reg_v2_seed0.sh (drop=0.0, dp=0.0, wd=0.05, patience=20)
run_charades_dinov3_reg_v2_seed0.sh (drop=0.05, dp=0.05, wd=0.05, patience=15)
run_charades_scdnet_seed0.sh (drop=0.1, dp=0.1, wd=0.05, patience=15) ✅ NUOVO
run_charades_dinov3_meanpatch_seed0.sh (drop=0.05, dp=0.05, wd=0.05, patience=15) ✅ NUOVO
run_charades_dinov3_combined_seed0.sh (drop=0.1, dp=0.1, wd=0.05, patience=15) ✅ NUOVO
job_scdnet_align.oar.sh — job OAR per allineamento skeleton (già eseguito)
job_dinov3_meanpatch_extract.oar.sh — job OAR estrazione mean_patch (4 shard) ✅ NUOVO
job_dinov3_combined_extract.oar.sh — job OAR estrazione combined (4 shard) ✅ NUOVO

11. Documenti prodotti
Documento	Contenuto	Status
deep_mstemba_clip_dino_comparison_updated.md	Analisi completa Phase 1 CLIP vs DINOv3 reg_v2, tutte 157 classi	✅
mstemba_phase2b_skeleton_extraction.md	Estrazione e allineamento SCDNet, pipeline tecnica	✅ (superato)
phase2b_skeleton_extraction_and_results.md	Documento unificato Phase 2B: estrazione, implementazione, risultati, analisi per-classe, diagnosi, fusion roadmap, literature survey	✅ AGGIORNATO
mstemba_phase2a_dino_dense_features.md	Piano tecnico DINOv3 dense features	✅
mstemba_phase2a_dino_clip_literature_fusion.md	Letteratura CLIP–DINOv3 fusion, implementazioni	✅
mstemba_phase2b_skeleton_literature_fusion.md	Letteratura skeleton, implementazioni fusion visual+skeleton	✅

12. Workflow tipico sul cluster
# 1. Login e acquisizione nodo GPU
ssh mdiiorio@fsophia.grid5000.fr
oarsub -q besteffort -p esterel42 -l host=1/gpu=1,walltime=12 -I

# 2. Attivazione environment
cd /srv/storage/stars@storage3.sophia.grid5000.fr/mdiiorio/masters-thesis/Traineeship/MS-Temba
source env_abaca.sh   # oppure: conda activate mstemba_fresh

# 3. Lancio training
python vim/MSTemba_main.py [args]   # oppure via script .sh

# 4. Monitoraggio
tail -f runs/charades/NOME_RUN/training.log
cat runs/charades/NOME_RUN/metrics_per_epoch.csv

# 5. Job OAR (per task CPU senza GPU, es. estrazione feature)
oarsub -S vim/scripts/JOB.oar.sh
oarstat -u mdiiorio

# 6. Estrazione parallela con sharding (es. DINOv3 dense su 4 GPU)
# Terminale 1-4, ognuno con GPU diversa:
python vim/dinov3_feature_extractor.py \
  --image_root_dir data/charades_frames_24fps \
  --save_dir data/hf_features/Temporal_Action_Detection/charades_dinov3_vitl16_w16_24fps_meanpatch \
  --window_size 16 --pooling mean_patch --amp bf16 --save_dtype float32 \
  --shard_id {0,1,2,3} --num_shards 4
13. Convenzioni di naming
Tipo	Pattern	Esempio
Run directory	runs/charades/{backbone}_{config}/seed{N}	runs/charades/scdnet/seed0
Feature dir	charades_{backbone}_{params}/	charades_scdnet_w16/
Script training	run_charades_{backbone}_{config}_seed{N}.sh	run_charades_scdnet_seed0.sh
Checkpoint best	runs/.../checkpoint_best.pth	—
Metriche	runs/.../metrics_per_epoch.csv	—
14. Priorità task (ordine consigliato)
[IN CORSO] Completare estrazione DINOv3 mean_patch (4 shard, ~14h per shard su GPU correnti)
[DOPO ESTRAZIONE] Lanciare estrazione DINOv3 combined (stessi 4 shard, --pooling combined)
[DOPO ESTRAZIONE] Lanciare training 2A.1 (mean_patch) e 2A.2 (combined)
[PARALLELIZZABILE] Score-level fusion CLIP+skeleton (2B.2) — zero modifiche codice, usa checkpoint già esistenti
[NEXT MAJOR] Implementare gated fusion CLIP+skeleton (2B.5) — modifica models_MSTemba.py + charades_dataloader.py per dual-feature loading
Analizzare risultati 2A.1/2A.2 e 2B.2 → informano priorità fusion
Gated fusion CLIP+DINOv3 (2C.3)
Three-stream fusion (2D)
15. Letteratura chiave (referenze rapide)
Skeleton fusion:
- CLIP-MG (2025, arXiv:2506.16385): skeleton-as-Query per guidare attenzione CLIP (+16.5pp)
- SkeletonCLIP++ (2024): CLIP semantics per pesare frame skeleton → Weighted Frame Integration
- Zhu et al. (ACM TOMM 2022, arXiv:2202.11374): two-stage fusion (skeleton attention early + cross-attn late)
- HCMFN (Hu et al., 2024): skeleton come spatial anchor per visual features

DINOv3 dense:
- Jose et al. (CVPR 2025, arXiv:2412.16334): CLS+patch_avg migliora global (+1.8% ImageNet) e dense (+3.2% ADE20K)
- COMM (Jiang et al., arXiv:2310.08825): DINOv2 shallow layers = low-level detail, deep layers = semantics
- Talk2DINO (Barsellotti et al., arXiv:2411.19331): attention maps DINOv2 per spatial alignment

16. Domande aperte / da chiarire
Verificare se SCD-Net su Charades_SCDNet_features2.zip è stato estratto con multi-stream o single-stream
Verificare se charades_dataloader.py è già predisposto per caricare una seconda feature directory (per gated fusion) o se serve modifica significativa
Tempo reale estrazione DINOv3 dense: le GPU esterel variano molto in velocità (~14h su alcune, meno su altre). Lo script ha skip_existing per resume dopo interruzione walltime.
Per gated fusion: serve modifica del dataloader per caricare 2 feature directory in parallelo (CLIP + skeleton) — da implementare