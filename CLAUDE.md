# FedDualAtt Project Context

## Paper
- **Title:** "Dual Attention Heads for Personalized Federated Learning in Multi-Center ECG Classification"
- **Venue:** IEEE MWSCAS (submitted)
- **File:** `code/FedCVD/docs/feddualatt_paper.tex`
- **Bibliography:** `code/FedCVD/docs/refs.bib`

## Method Summary
FedDualAtt splits transformer attention heads into two parallel branches appended to a ResNet1D-34 backbone:
- **Global branch** (`θ^g`): FedAvg-aggregated, learns cross-site patterns
- **Local branch** (`φ_k`): Per-client, never aggregated, adapts to site-specific distributions

Architecture: ResNet1D-34 → 2 × DualAttentionTransformerBlock → classification head (FC + sigmoid, BCELoss)

Per block: global_proj_in → MHA(H_g heads) → global_proj_out → norm1, local_proj_in → MHA(H_l heads) → local_proj_out → norm2, combine Linear(2d→d) → FFN → norm3

Hyperparameters: d=512, H=8, d_h=64, 2 blocks, FFN hidden=2048

Parameter split (pattern matching in `algorithm/ecg/feddualatt.py`):
- **Local (`φ_k`):** params with `local_att` or `local_proj` in name
- **Global (`θ^g`):** everything else (ResNet, global attn, norm1/2/3, combine, FFN, FC head)

FL Protocol per round:
1. Server → client: θ^g_t + stored φ_{k,t}
2. Client trains jointly (SGD, E epochs) on D_k
3. Client → server: θ^g_{k,t+1} (local positions zeroed) + φ_{k,t+1} + n_k
4. Server: FedAvg on θ^g, stores φ_{k,t+1} per client

## Benchmark
- **Dataset:** FedCVD (`code/FedCVD/docs/refs.bib` key: `fedcvd`) — 4 clients: SPH, PTB-XL, SXPH, G12EC
- **Task:** 20-class multi-label ECG classification, metrics: Micro-F1 and mAP
- **Baselines:** FedAvg, FedProx, Scaffold, Ditto, FedALA (results taken from FedCVD paper)
- **Best result:** FedDualAtt 8Hg:0Hl → Global Micro-F1 72.68% (vs. Scaffold 70.1%)

## Key Files
| File | Purpose |
|---|---|
| `code/FedCVD/docs/feddualatt_paper.tex` | Main paper |
| `code/FedCVD/docs/refs.bib` | Bibliography (IEEEtran format, all entries verified) |
| `code/FedCVD/model/dual_attention_resnet.py` | DualAttentionTransformerBlock + DualAttentionResNet1D |
| `code/FedCVD/algorithm/ecg/feddualatt.py` | FL server/client protocol |
| `code/FedCVD/trainers/feddualatt_ecg.py` | Training entry point, hyperparameter config |

## Completed Experiments (8 Total Heads)

Ratio sweep across all 9 configs (0G:8L through 8G:0L), 5 seeds each [42, 123, 456, 789, 1024].

Key results (Global Micro-F1):
- 8G:0L: 72.68% ± 1.12% (best global performance)
- 5G:3L: 72.47% ± 0.70% (best stability, lowest variance)
- 3G:5L: 72.12% ± 1.25%
- 7G:1L: 71.74% ± 0.42%
- 4G:4L: 71.21% ± 2.70%
- 6G:2L, 2G:6L: unstable (high variance)
- 0G:8L: 58.50% (pure local fails for global metric)

Per-client observations: different clients prefer different ratios (SPH: 5G:3L, SXPH: 1G:7L, G12EC: 1G:7L). Increasing local heads improves per-client performance but hurts global (capacity is shared at 8 heads).

U-shaped performance curve: extreme and clear splits are stable; middle-ground splits (6:2, 2:6) are unstable.

Results files: `head_ratio_all_metrics.csv`, `head_ratio_results_summary.csv`, `baseline_comparison.csv`, `reevaluation_results.csv`
Experiment scripts: `scripts/ecg/run_head_ratio.py`, `scripts/ecg/run_head_count_experiment.py`

## Next Experiment: Head Count Scaling — Implementation Plan

### Goal
Test whether scaling total heads breaks the global/local zero-sum trade-off. At 8 heads, more local = less global capacity. At higher head counts, both branches can have sufficient capacity. The hypothesis is that balanced ratios can achieve good global AND per-client performance simultaneously when neither branch is capacity-starved.

### Experiment Design
- **Ratio sweep at 3 total head counts**: 10, 20, 30 (divisible by 10 for clean 10% steps)
- **11 ratio points per head count**: 0%, 10%, 20%, ..., 100% global
- **Overlay with existing 8-head data** (12.5% steps) as a 4th reference line
- **5 seeds each**: [42, 123, 456, 789, 1024]
- **Track both**: Global Micro-F1 and per-client Micro-F1
- **Total new runs**: 33 configs × 5 seeds = 165

Concrete configs:
```
10 heads: (10,0) (9,1) (8,2) (7,3) (6,4) (5,5) (4,6) (3,7) (2,8) (1,9) (0,10)
20 heads: (20,0) (18,2) (16,4) (14,6) (12,8) (10,10) (8,12) (6,14) (4,16) (2,18) (0,20)
30 heads: (30,0) (27,3) (24,6) (21,9) (18,12) (15,15) (12,18) (9,21) (6,24) (3,27) (0,30)
```

### Design Rationale
- d_model=512 is inherited from ResNet1D-34 backbone (layer4 output channels), not a free parameter
- Higher head counts (e.g., 30 heads → 1920 att dim) use the same expand-then-compress pattern as FFN (512→2048→512), so projection isn't a fundamental concern
- Real risks of scaling: overfitting on short sequences (~156 timesteps), compute cost, FL optimization difficulty
- 10% ratio steps give smooth curves; each head is only 5% at 20 heads (vs 12.5% at 8), reducing the coarseness that may have caused instability

### Implementation Steps

**Step 1: Rewrite `scripts/ecg/run_head_count_experiment.py`**
- Replace old `HEAD_CONFIGS` (fixed ratio, varying head count) with new design (fixed head count, varying ratio)
- Generate configs programmatically: for each `num_heads` in [10, 20, 30], create 11 ratio points at 10% steps
- CLI: `--num_heads 20` sweeps all ratios at 20 heads; `--all` runs all three head counts
- Keep existing structure: subprocess calls to `trainers/feddualatt_ecg.py`, seed management, summary output

**Step 2: No changes needed to model/algorithm/trainer**
- `model/dual_attention_resnet.py` already supports arbitrary head counts via `head_dim=64` projections
- `algorithm/ecg/feddualatt.py` parameter filtering uses name patterns (`local_att`, `local_proj`), works for any head count
- `trainers/feddualatt_ecg.py` accepts `--num_heads` and `--global_heads` args, output path `global{G}_local{L}/seed{S}/` already differentiates all configs
- `model/__init__.py` `get_model()` passes kwargs through to `dual_attention_resnet1d()`

**Step 3: Create `scripts/ecg/extract_head_count_metrics.py`**
- Scan `output/dual_attention_resnet1d/feddualatt/global*_local*/seed*/` directories
- Group by total head count (global + local from directory name)
- Compute `pct_global = global / (global + local) * 100`
- Reuse `extract_metrics()` and `aggregate_metrics()` logic from `extract_head_ratio_metrics.py`
- Output CSV: `total_heads, pct_global, global_heads, local_heads, global_micro_f1_mean, global_micro_f1_std, SPH_micro_f1_mean, SPH_micro_f1_std, PTB-XL_micro_f1_mean, ...`
- Include existing 8-head data automatically (same directory structure)

**Step 4: Create `scripts/ecg/plot_head_count_results.py`**
- Line plot: x = % global (0-100), y = Micro-F1
- 4 lines (8, 10, 20, 30 heads) with different colors, error bars or shaded std regions
- Two figures: (1) global Micro-F1, (2) per-client Micro-F1 (4 subplots or overlaid with legend)
- Reuse matplotlib style from `plot_head_ratio_results.py`
- Save to `docs/figures/`

**Step 5: Smoke test**
- Quick validation: `python run_head_count_experiment.py --num_heads 10 --pct_global 50 --num_seeds 1 --communication_round 2 --data_fraction 0.1`
- Verify: model instantiates, trains, saves metrics to correct output path

**Step 6: Run full experiments (165 runs)**

**Step 7: Extract results and generate plots**

### Files Summary
| Action | File | Purpose |
|--------|------|---------|
| Rewrite | `scripts/ecg/run_head_count_experiment.py` | Experiment runner |
| Create | `scripts/ecg/extract_head_count_metrics.py` | Results extraction |
| Create | `scripts/ecg/plot_head_count_results.py` | Visualization |
| Unchanged | `model/dual_attention_resnet.py` | Already supports arbitrary heads |
| Unchanged | `algorithm/ecg/feddualatt.py` | Name-based param filtering works |
| Unchanged | `trainers/feddualatt_ecg.py` | Already accepts head args |
| Unchanged | `model/__init__.py` | Already forwards kwargs |

### Future Experiment Ideas (Not Yet Started)
- Gated combination layer (learnable global/local weighting per sample)
- Differential learning rates for global vs local parameters
- Communication cost analysis (no training needed, just parameter counting)
- Convergence speed analysis (from existing checkpoint data)

## refs.bib Entry Types (verified against actual publications)
| Key | Type | Venue |
|---|---|---|
| `who_cvd_2025` | `@electronic` | WHO webpage |
| `sattar_electrocardiogram_2023` | `@inbook` | StatPearls, Jan 2026 |
| `DBLP:journals/corr/McMahanMRA16` | `@inproceedings` | AISTATS 2017, Fort Lauderdale FL |
| `fedcvd` | `@misc` | arXiv 2411.07050 |
| `natarajan2020wide` | `@inproceedings` | Computing in Cardiology 2020, Rimini Italy |
| `zhao2018federated` | `@misc` | arXiv 1806.00582 |
| `scaffold` | `@inproceedings` | ICML 2020 (Virtual) |
| `ditto` | `@inproceedings` | ICML 2021 (Virtual) |
| `fedala` | `@inproceedings` | AAAI 2023, Washington DC |
| `fedbn` | `@inproceedings` | ICLR 2021 (Virtual) |
| `fedprox` | `@inproceedings` | MLSys 2020, Austin TX |
