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

## Next Experiment: Head Count Scaling (Planned, Not Yet Executed)

**Goal:** Test whether scaling total heads breaks the global/local zero-sum trade-off. At 8 heads, more local = less global capacity. At higher head counts, both branches can have sufficient capacity.

**Design decisions made:**
1. Sweep global/local ratio at multiple total head counts, plot on the same axes (x = % global, y = Micro-F1) with one line per head count
2. Use 10% ratio steps (0%, 10%, 20%, ..., 100% global) for smooth curves and clean paper presentation
3. Head counts must be divisible by 10 for clean 10% steps → candidates: 10, 20, 30
4. d_model=512 is inherited from ResNet1D-34 backbone (layer4 output channels), not a free parameter
5. Higher head counts (e.g., 30 heads, 1920 att dim) are structurally similar to the FFN expand-then-compress (512→2048→512), so projection isn't a fundamental concern
6. Real risks of scaling: overfitting on short sequences (~156 timesteps), compute cost, optimization difficulty in limited FL local epochs

**Open questions:**
- Final head count set (10, 20, 30 is leading candidate)
- Number of seeds per config (5 seeds × 11 ratios × 3 head counts = 165 runs)
- Whether to run all upfront or start with a coarse sweep

**Two metrics to track:** Global Micro-F1 (aggregated performance) AND per-client Micro-F1 (personalization). The hypothesis is that at higher head counts, balanced ratios can achieve good global AND per-client performance simultaneously.

**Future experiment ideas (not yet started):**
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
