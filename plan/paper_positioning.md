# Paper Positioning: Transfer Learning + Dual Attention for Personalized Federated ECG

## Core Narrative

> Personalized federated learning for ECG classification faces a feature quality bottleneck:
> when training the backbone from scratch within FL on small multi-center datasets, the shared
> feature extractor cannot learn rich enough representations to benefit from sophisticated
> personalization mechanisms. We demonstrate this bottleneck empirically — a single attention
> head achieves the same performance as 32 heads (72% F1 on FedCVD). We address this by
> integrating a pre-trained ECG foundation model (ECGFounder, 10.7M ECGs) as the backbone in
> FedDualAtt, with dual attention heads providing personalization on top of rich, general-purpose
> features. With gradient isolation (`detach()`) ensuring the local branch doesn't corrupt the
> pre-trained backbone, this design combines the best of foundation models (feature quality) and
> federated learning (privacy-preserving personalization).

---

## Positioning Against Related Work

### Personalized FL Methods

| Method | Venue | Personal Component | Backbone | Our Advantage |
|--------|-------|-------------------|----------|---------------|
| FedAvg | AISTATS 2017 | None | Trained in FL | We add personalization without sacrificing global |
| FedProx | MLSys 2020 | Proximal regularization | Trained in FL | Architectural personalization > regularization |
| SCAFFOLD | ICML 2020 | Control variates | Trained in FL | No extra communication overhead |
| Ditto | ICML 2021 | Full local model copy | Trained in FL | More parameter-efficient (only local heads are personal) |
| FedPer | arXiv 2019 | FC head | Trained in FL | Attention-based personalization > FC head |
| FedRep | ICML 2021 | FC head, alternating opt | Trained in FL | Single-phase training (no alternating), stronger backbone |
| FedBN | ICLR 2021 | BatchNorm layers | Trained in FL | More expressive personalization than BN alone |
| FedALA | AAAI 2023 | Adaptive local aggregation | Trained in FL | Complementary; could be combined |
| pFedDB | AAAI 2026 | Dual branch from input | Trained in FL | Shared backbone enables feature transfer |

### ECG Foundation Models

| Model | Venue | Pre-training | Used in FL? | Our Contribution |
|-------|-------|-------------|-------------|-----------------|
| ECGFounder | NEJM AI 2025 | 10.7M ECGs, 150 classes | No | We integrate into FL with personalization |
| HuBERT-ECG | medRxiv 2024 | 9.1M ECGs | No | We show foundation models + FL personalization |
| ECG-FM | 2024 | 1M+ ECGs | No | We are first to combine ECG foundation model with pFL |
| ECGFM | Info Fusion 2025 | Multi-center, millions | No | We add per-client personalization |

### Transfer Learning in FL

| Method | Approach | Our Difference |
|--------|----------|---------------|
| Standard FL | Train from scratch | We start from pre-trained backbone |
| FedAvg + pre-trained | Fine-tune pre-trained model with FedAvg | We add dual attention for personalization |
| FedPer + pre-trained | Pre-trained backbone + personal FC head | We use attention (richer than FC) + gradient isolation |

**Key gap we fill:** No prior work combines ECG foundation models with personalized FL. Foundation model papers focus on centralized fine-tuning. FL papers train from scratch. We bridge the two.

---

## Contributions

1. **Transfer learning integration:** First to integrate an ECG foundation model (ECGFounder) as the backbone in a personalized FL framework, enabling richer feature representations than from-scratch training on limited multi-center data.

2. **Dataset bottleneck diagnosis:** Empirical demonstration that from-scratch backbone training saturates at N=1 attention head on FedCVD (72% F1 with 1 head = 32 heads), establishing that feature quality — not architecture capacity — is the performance bottleneck.

3. **Dual attention personalization on foundation features:** Global attention heads aggregate cross-site patterns in the rich feature space while local attention heads personalize to each institution, with gradient isolation (`detach()`) preventing local adaptations from corrupting the pre-trained backbone.

4. **Scaling study:** Investigation of whether richer foundation model features unlock the scaling potential of multi-head attention in FL, breaking the saturation observed with from-scratch training.

---

## Experiment Summary

| # | Experiment | Backbone | Attention | Question |
|---|-----------|----------|-----------|----------|
| 1 | Baseline | ECGFounder (frozen) | None (FC only) | What does pre-trained backbone give in FL? |
| 2 | Our method | ECGFounder (frozen) | DualAtt 4G:4L | Does dual attention add value? |
| 3 | Head scaling | ECGFounder (frozen) | N=1, 4, 8 | Is N=1 saturation broken? |
| 4 | Ratio sweep | ECGFounder (frozen) | Best N, all ratios | Does personalization improve? |
| 5 | Fine-tune | ECGFounder (fine-tuned) | DualAtt | Does backbone adaptation help? |
| Ref | From scratch | ResNet1D-34 | DualAtt | Our existing 72% baseline |

---

## Key Comparisons to Make in the Paper

### 1. From-scratch vs Pre-trained Backbone
Shows the feature quality bottleneck and how transfer learning addresses it.

### 2. Pre-trained + FC vs Pre-trained + DualAttention
Shows the value of attention-based personalization on top of strong features.

### 3. Head Count Scaling: From-scratch vs Pre-trained
If N=1 saturates with from-scratch but N=8 > N=1 with pre-trained → proves the bottleneck was feature quality.

### 4. FedDualAtt vs Existing FL Methods on Same Dataset
Shows our method outperforms FedAvg, FedProx, SCAFFOLD, Ditto, FedALA (numbers from FedCVD paper).

### 5. Frozen vs Fine-tuned Backbone
Shows whether FL-level backbone adaptation provides additional benefit.

---

## Story the Experiments Tell

**Best case scenario:**
1. Pre-trained backbone alone (Exp 1) already beats 72% → proves feature bottleneck
2. DualAttention adds further improvement (Exp 2 > Exp 1) → attention matters
3. N=8 > N=1 with pre-trained (Exp 3) → scaling unlocked
4. Local heads improve per-client F1 without hurting global (Exp 4) → personalization works

**Acceptable scenario:**
1. Pre-trained backbone matches 72% → FedCVD task is saturated regardless
2. DualAttention adds small improvement → attention helps marginally
3. Still get a paper about "foundation models in FL" with thorough analysis

**Worst case:**
1. ECGFounder doesn't help on FedCVD → incompatible preprocessing or FedCVD is too different from ECGFounder's pre-training distribution
2. Mitigation: this itself is a finding about transfer learning limitations in FL
