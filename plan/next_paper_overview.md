# Next Paper: Overview and Structure

**Build-on:** IEEE MWSCAS submitted paper — "Dual Attention Heads for Personalized Federated Learning in Multi-Center ECG Classification"

---

## Evolution of the Research Direction

| Phase | Focus | Key Finding | Status |
|-------|-------|-------------|--------|
| Submitted paper | FedDualAtt architecture, 8-head ratio sweep | Global/local trade-off exists | Published |
| Phase 1 | Diagnose trade-off (decoupled heads) | Architectural constraint (gradient interference) | Done |
| Phase 2 | Fix with `detach()`, head count scaling N=1→32 | Fix works, but dataset saturates at N=1 | Done |
| Phase 3 | Fusion ablation, differential LR | concat wins, marginal LR gains | Done |
| **Phase 4 (current)** | **Transfer learning with pre-trained ECG backbone** | **Break the dataset bottleneck** | Planning |

---

## Why Transfer Learning?

All experiments so far reveal the same bottleneck: **the FedCVD dataset limits performance, not the architecture.**

- N=1 head saturates at 72% F1 (same as N=32)
- Fusion modes don't help (concat best, others worse)
- Differential LR: marginal
- The ResNet backbone trained from scratch on ~20K ECGs cannot extract richer features

**Transfer learning breaks this ceiling** by replacing our from-scratch backbone with one pre-trained on 10.7M ECGs (ECGFounder, NEJM AI 2025). The hypothesis: with richer features, dual attention heads have more to work with, and head count scaling may finally show its value.

---

## Proposed Titles

1. *"Foundation Model-Powered Personalized Federated Learning with Dual Attention Heads for ECG Classification"*
2. *"Breaking the Dataset Bottleneck in Federated ECG Classification via Pre-trained Backbone Transfer"*
3. *"Dual Attention Personalization on Top of ECG Foundation Models in Federated Learning"*

---

## Narrative Arc

> Personalized federated learning for ECG classification is bottlenecked by the quality of
> shared feature representations. When training the backbone from scratch within FL, a small
> multi-center dataset like FedCVD (4 clients, ~20K ECGs) limits what the backbone can learn —
> additional attention heads saturate immediately (N=1 matches N=32). We address this by
> replacing the from-scratch backbone with ECGFounder, a foundation model pre-trained on 10.7M
> ECGs. The pre-trained backbone provides rich, general-purpose ECG features that the dual
> attention mechanism can leverage: global heads aggregate cross-site patterns while local heads
> personalize to each institution's distribution. With richer features, we investigate whether
> the architecture's scaling potential — previously bottlenecked by data — is unlocked.

---

## Architecture

### Current (from-scratch backbone)
```
Input (12, 5000) → ResNet1D-34 [random init, trained in FL] → (512, 157)
  → PosEnc → DualAttention ×2 → GAP → FC(512,20) → Sigmoid
```

### New (pre-trained backbone)
```
Input (12, 5000) → ECGFounder [pre-trained on 10.7M ECGs, frozen] → (1024, ~39)
  → transpose → Projection Linear(1024, 512)
  → PosEnc → DualAttention ×2 [global + local, with detach()] → GAP → FC(512,20) → Sigmoid
```

Key differences:
- Backbone: ECGFounder Net1D (7 stages, 22 blocks, SE attention) vs ResNet1D-34 (4 layers, 16 blocks)
- Feature quality: pre-trained on 10.7M ECGs vs trained from scratch on ~20K
- Feature shape: (1024, ~39) vs (512, 157) — fewer temporal positions but richer channels
- Preprocessing: z-score normalization required for ECGFounder

---

## Experiments

### Experiment 1: Pre-trained backbone baseline (no attention)
**Config:** ECGFounder (frozen) → FC(1024, 20) → FedAvg
**Question:** What does the pre-trained backbone give us out of the box in FL?
**Expected:** Likely beats 72% (our current best) — establishes the new ceiling

### Experiment 2: Pre-trained backbone + DualAttention
**Config:** ECGFounder (frozen) → Projection → DualAtt (4G:4L) → FC
**Question:** Does dual attention add value on top of strong features?
**Expected:** If > Exp 1, dual attention has meaningful contribution beyond the backbone

### Experiment 3: Head count sweep
**Config:** ECGFounder (frozen) → DualAtt with N=1, 4, 8 at 100% global
**Question:** Does head count matter now that features are richer?
**Expected:** If N=8 > N=1, the dataset bottleneck is broken — scaling works with better features

### Experiment 4: Ratio sweep
**Config:** ECGFounder (frozen) → DualAtt at best N, sweep 9 ratio configs
**Question:** Does the global/local ratio affect performance differently with richer features?
**Expected:** More nuanced personalization-generalization trade-off

### Experiment 5: Fine-tune backbone
**Config:** ECGFounder (fine-tuned with detach()) → DualAtt
**Question:** Can fine-tuning improve further? Does gradient interference reappear?
**Expected:** Fine-tuning may help; detach() ensures backbone doesn't degrade

### Comparison baselines
- Our existing results: ResNet1D-34 from scratch → DualAtt → 72% F1
- FedAvg, FedProx, SCAFFOLD, Ditto, FedALA results from FedCVD paper

---

## Implementation Steps

### Step 1: ECGFounder Integration
- Download ECGFounder code (`net1d.py`) and pre-trained weights from HuggingFace
- Create `ECGFounderBackbone` wrapper that returns pre-GAP features (batch, 1024, T)
- Add z-score normalization to data pipeline
- Add projection layer Linear(1024, 512) to match our d_model

### Step 2: Make backbone pluggable
- Modify `DualAttentionResNet1D` to accept `backbone` parameter ('resnet1d34' or 'ecgfounder')
- Add `--backbone`, `--pretrained_path`, `--freeze_backbone` CLI args
- Ensure frozen backbone params are excluded from optimizer and not transmitted in FL

### Step 3: Run experiments
- Experiment 1: baseline (no attention) — 1 seed, 50 rounds
- Experiment 2: DualAtt 4G:4L — 1 seed, 50 rounds
- Experiment 3: N=1, 4, 8 sweep — 1 seed each
- Gate: if Exp 2 > Exp 1 and Exp 3 shows scaling → proceed to Exp 4, 5
- Experiment 4: full ratio sweep — 1 seed, then 5 seeds for paper quality
- Experiment 5: fine-tuned backbone — 1 seed

### Step 4: Analysis and paper writing
- Compare all results against from-scratch baseline
- Generate plots: performance comparison, scaling curves, ratio sweep
- Write paper sections

---

## ECGFounder Details

| Aspect | Value |
|--------|-------|
| Paper | NEJM AI 2025 |
| Pre-training data | 10.7M 12-lead ECGs, 150 label categories |
| Architecture | Net1D: 7 stages, 22 BasicBlocks, SE attention, Swish activation |
| Input | (12, variable length), z-score normalized |
| Pre-GAP output | (batch, 1024, T) where T ≈ input_length / 128 |
| Post-GAP output | (batch, 1024) |
| Weights | HuggingFace: PKUDigitalHealth/ECGFounder |
| Code | GitHub: PKUDigitalHealth/ECGFounder |
| Preprocessing | Z-score normalization per signal, NaN→0, lead reordering |

---

## What This Paper Contributes (vs. submitted paper)

| Submitted Paper | This Paper |
|-----------------|-----------|
| FedDualAtt architecture with from-scratch backbone | FedDualAtt with pre-trained ECG foundation model backbone |
| Dataset-bottlenecked: N=1 saturates | Potentially unlocked: richer features → scaling matters |
| 72% global F1 ceiling | Higher ceiling with 10.7M-ECG backbone |
| Trade-off exists (before detach) | Trade-off managed (detach + strong backbone) |
| Single dataset, single backbone | Transfer learning paradigm: foundation model + FL personalization |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| ECGFounder doesn't improve over our backbone on FedCVD | Still a valid finding (FedCVD is too easy/small) |
| ECGFounder's architecture incompatible with our pipeline | Wrapper class isolates differences; pre-GAP extraction straightforward |
| Z-score normalization changes results for our baseline | Run from-scratch baseline with z-score too for fair comparison |
| Frozen backbone → attention heads do all the work → trivial result | Fine-tuning experiment (Exp 5) shows backbone adaptation still matters |
| ~39 temporal positions too few for meaningful attention | Still more than most NLP sentence lengths; attention can still learn patterns |

---

## Files to Modify/Create

| File | Action |
|------|--------|
| `model/ecgfounder_backbone.py` | CREATE — ECGFounder wrapper, pre-GAP extraction |
| `model/net1d.py` | CREATE — ECGFounder's Net1D architecture (from their repo) |
| `model/dual_attention_resnet.py` | MODIFY — accept pluggable backbone, add projection layer |
| `model/__init__.py` | MODIFY — register new backbone options in get_model() |
| `trainers/feddualatt_ecg.py` | MODIFY — add backbone/pretrained/freeze CLI args |
| `algorithm/ecg/feddualatt.py` | MODIFY — handle frozen backbone params in FL protocol |
| `utils/dataloader.py` | MODIFY — add z-score normalization option |
| `scripts/ecg/run_transfer_learning.py` | CREATE — experiment runner for transfer learning experiments |
