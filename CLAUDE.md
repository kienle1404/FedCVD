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

## Head Count Scaling Experiments

### Motivation
The 8-head ablation forces a zero-sum trade-off: adding local heads removes global heads.
It is unclear whether this trade-off is a **capacity constraint** (fixed budget) or an
**architectural constraint** (gradient competition in the shared encoder). The revised
design tests this with two sequential phases.

---

### Phase 1 — Decoupled Pre-check ✅ COMPLETE

**Hypothesis:** fixing `global_heads=8` and adding local heads on top (without removing any
global heads) should leave global F1 flat at ~72.7% while per-client F1 improves.
If global F1 drops, the trade-off is architectural, not a capacity artifact.

**Configs (global_heads fixed, local_heads swept):**
```
g=8: (8,0)[existing, 5 seeds] (8,2) (8,4) (8,8) (8,16)   [new configs: 1 seed]
g=4: (4,0)[existing, 5 seeds] (4,2) (4,4) (4,8) (4,16)   [new configs: 1 seed]
```

**Status:**
- [x] Scripts implemented: `scripts/ecg/run_decoupled_heads.py`, `scripts/ecg/plot_decoupled_heads.py`
- [x] Smoke test passed
- [x] Full run completed: `python run_decoupled_heads.py --all --num_seeds 1`
- [x] Extraction bug fixed: `find_head_ratio_runs()` now filters to completed runs only (requires `server/metric.json`)
- [x] Plots generated: `docs/figures/decoupled_heads_global_f1.png`, `decoupled_heads_global_mAP.png`, `decoupled_heads_client_f1.png`

---

**Results — Global Micro-F1:**

| Config (g-l) | Total Heads | Global F1 (%) | Δ from 8-0 baseline |
|---|---|---|---|
| 8-0 (baseline) | 8  | 72.68 ± 1.12 | — |
| 8-2            | 10 | ~64.5         | −8.2pp |
| 8-4            | 12 | ~58.7         | −14.0pp |
| 8-8            | 16 | ~56.0         | −16.7pp |
| 8-16           | 24 | ~60.0         | −12.7pp |
| 4-0 (baseline) | 4  | ~73.5 (1 seed)| +0.8pp (within variance) |
| 4-2            | 6  | ~72.0         | ~flat |
| 4-4            | 8  | ~65.0         | −7.7pp |
| 4-8            | 12 | ~35.0         | **COLLAPSED** (lr instability) |
| 4-16           | 20 | ~65.0         | −7.7pp |

*Non-baseline values are 1-seed estimates (no error bars). Baseline 8-0 is 5-seed.*

**Results — Global mAP:**

| Config | mAP (%) | Δ from baseline |
|---|---|---|
| 8-0 (baseline) | ~55.0 | — |
| 8-2            | ~45.0 | −10pp |
| 8-4 to 8-16    | ~43–46 | −9 to −12pp |
| 4-0            | ~55.0 | ~flat |
| 4-8            | ~40.0 | COLLAPSED |

**Per-client observations:**
- SPH (~82–88%): stable across all g=8 configs; barely affected by adding local heads
- PTB-XL (~65–77%): moderate decline as local heads increase
- SXPH (~65–75%): similar to PTB-XL
- G12EC (~65–75%): moderate decline
- All clients: g=4 collapses at l=8 (total 12 heads), same lr instability as 16-head configs

---

**Decision gate outcome: ARCHITECTURAL CONSTRAINT CONFIRMED → Phase 2 not viable as designed**

Global F1 drops monotonically from 72.7% to 56% as local heads increase from 0 to 8, even
with `global_heads=8` held constant. This rules out capacity as the explanation.

**Root cause:** Local head gradients flow backward through the shared ResNet backbone,
conflicting with global head gradients. This happens regardless of total head count —
scaling to N=16 or N=32 would not resolve it.

**Additional finding:** Configs with total heads ≥ 12–16 are unstable at default `lr=0.1` SGD.
The `4-8` collapse (12 total heads, 35% F1) and earlier `16-0` collapse (30% F1) confirm
that larger attention projections require lower learning rates.

---

**Pivot options (not yet decided):**
- **Option A:** Accept trade-off as finding; run 5 seeds for paper-quality error bars on g=8 line
- **Option B:** Lower lr test for N=16 (`--lr 0.01`); ~5 runs to check stability
- **Option C:** Gradient isolation via `detach()` on backbone before local attention branches

---

### Phase 2 — Ratio Sweep at Powers-of-2 Head Counts ⛔ NOT VIABLE

Phase 1 confirmed an architectural constraint. Scaling total heads cannot resolve gradient
interference through the shared backbone. Phase 2 as designed is blocked.

**Scripts remain available** if the direction changes (e.g., after gradient isolation fix):
- `scripts/ecg/run_head_count_experiment.py`
- `scripts/ecg/extract_head_count_metrics.py`
- `scripts/ecg/plot_head_count_results.py`

---

### Run Instructions

```bash
cd code/FedCVD/scripts/ecg

# Phase 1 full run
python run_decoupled_heads.py --all --num_seeds 5

# Phase 1 extract + plot
python extract_head_ratio_metrics.py --csv ../../docs/head_ratio_all_metrics.csv
python plot_decoupled_heads.py

# Phase 2 full run (only if Phase 1 passes gate)
python run_head_count_experiment.py --all --num_seeds 5

# Phase 2 extract + plot
python extract_head_count_metrics.py --csv ../../docs/head_count_all_metrics.csv
python plot_head_count_results.py
```

---

### Architecture Note
`head_dim=64` fixed. Projection adapts: `att_dim = n_heads × 64`, projected into/out-of
`d_model=512` via learned linear layers. No model/algorithm/trainer changes needed.
At N=32: att_dim=2048, same expand-then-compress ratio as FFN (512→2048→512).

### Scripts Summary
| File | Status | Purpose |
|------|--------|---------|
| `scripts/ecg/run_decoupled_heads.py` | ✅ Done | Phase 1 runner |
| `scripts/ecg/run_head_count_experiment.py` | ✅ Done | Phase 2 runner |
| `scripts/ecg/extract_head_count_metrics.py` | ✅ Done | Phase 2 extractor |
| `scripts/ecg/plot_decoupled_heads.py` | ✅ Done | Phase 1 visualisation |
| `scripts/ecg/plot_head_count_results.py` | ✅ Done | Phase 2 visualisation |

### Future Experiment Ideas (Not Yet Started)
- Gated combination layer (learnable global/local weighting per sample)
- Differential learning rates for global vs local parameters
- Communication cost analysis (no training needed, just parameter counting)
- Convergence speed analysis (from existing checkpoint data)

---

## Future Paper Direction — Gradient Isolation Fix

**Context:** The submitted paper reports the global/local trade-off as a finding. The future
paper diagnoses the root cause and proposes a targeted fix. The submitted paper's results are
unaffected; this is a forward-looking extension.

### Root Cause (precise)

In `DualAttentionTransformerBlock.forward()` (`model/dual_attention_resnet.py` lines 259–263),
both branches receive the same tensor `x` — the ResNet backbone output:

```
backbone → x ──→ global branch → x_global ──→ combine → FFN → loss
               └──→ local branch  → x_local  ──┘
```

During `loss.backward()`, PyTorch accumulates gradients at `x` from both paths:

```
backbone ← [∂loss/∂x_global + ∂loss/∂x_local]
```

`∂loss/∂x_local` is client-specific — it pulls the backbone toward client k's distribution.
Across 4 clients, these signals conflict with each other and with the global gradient.
The backbone is pulled in 4 competing directions every round, degrading shared features.

This happens regardless of total head count — more heads = more competing gradient paths.

### Proposed Fix — `detach()` at Local Branch Entry

Insert a gradient wall where `x` enters the local branch. `x.detach()` creates a tensor
with the same values but no connection to the computation graph, so gradients from the local
branch stop there and cannot reach the backbone.

**File:** `model/dual_attention_resnet.py`, lines 259–263

```python
# BEFORE:
if self.local_heads > 0:
    local_in = self.local_proj_in(x)
    local_att_out, _ = self.local_att(local_in, local_in, local_in)
    local_out = self.local_proj_out(local_att_out)
    x_local = self.norm2(x + local_out)

# AFTER:
if self.local_heads > 0:
    x_local_input = x.detach()          # gradient wall: local grads cannot reach backbone
    local_in = self.local_proj_in(x_local_input)
    local_att_out, _ = self.local_att(local_in, local_in, local_in)
    local_out = self.local_proj_out(local_att_out)
    x_local = self.norm2(x_local_input + local_out)
```

**Important:** Both `local_proj_in(x)` and the residual `x + local_out` must use
`x_local_input`. The residual connection is a second gradient path back to the backbone —
leaving it as `x` would partially defeat the fix.

**No other files change.** FL protocol, parameter naming, aggregation, and optimizer are
all unaffected.

### What Changes vs. What Stays the Same

| Component | Without fix | With fix |
|---|---|---|
| Backbone (ResNet) | Grads from global + local | Grads from global only |
| Global heads | Unchanged | Unchanged |
| Local heads | Unchanged | Unchanged |
| Combine, FFN, norm3 | Unchanged | Unchanged |
| norm1 | Unchanged | Unchanged |
| norm2 | Unchanged | Unchanged |
| FL aggregation | Unchanged | Unchanged |

### Possible Risk

Local heads can no longer steer the backbone toward client-specific features — they can only
re-weight existing backbone features. In practice this likely has minimal impact because the
ResNet backbone already achieves 72.7% global F1 with no local heads at all, suggesting it
extracts sufficiently general features.

### Experiment Plan for Future Paper

| Step | Command | Purpose |
|---|---|---|
| Smoke test | `python run_decoupled_heads.py --global_heads 8 --local_heads 4 --num_seeds 1` | Verify fix: expect ~72% (vs ~58.7% currently) |
| Phase 1 sweep | `python run_decoupled_heads.py --global_heads 8 --num_seeds 5` | Show g=8 line is flat after fix |
| 8-head ratio sweep | `python run_head_ratio.py` (with fixed model) | New Pareto curve for fixed architecture |
| Cross-eval | Run cross-eval script with 8G:2L or 8G:4L | Show per-client + global can both be high |

Gate: if smoke test recovers to ~72%, proceed with full Phase 1 and ratio sweep.
If it doesn't fully recover, the combine layer or norm layers may be a secondary interference
path requiring further investigation.

### Paper Narrative

> The submitted paper identifies a global/local performance trade-off in FedDualAtt.
> This paper diagnoses the root cause as gradient interference: local head gradients
> flow through the shared backbone, conflicting with global head gradients during joint
> training. We propose gradient isolation — a single `detach()` call at the local branch
> entry — which eliminates the trade-off. With the fix, global F1 remains flat at ~72.7%
> as local heads are added, while per-client F1 continues to improve.

### Status
- [ ] Apply detach() fix to `model/dual_attention_resnet.py`
- [ ] Smoke test: run `(8,4)` config, compare to current ~58.7%
- [ ] Phase 1 sweep with fix (5 seeds)
- [ ] 8-head ratio sweep with fix (5 seeds)
- [ ] Cross-evaluation with fix

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
