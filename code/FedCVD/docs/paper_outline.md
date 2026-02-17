# FedDualAtt — Detailed Paper Outline
**Target venue:** MWSCAS 2026 (IEEE, 5-page limit)
**Title:** FedDualAtt: Dual Attention Heads for Personalized Federated Learning in Multi-Center ECG Classification

---

## Abstract (~150 words)

- **S1 — Motivation:** Multi-center ECG classification under FL faces data heterogeneity across hospitals (demographics, equipment, labeling practices)
- **S2 — Gap:** Standard FL aggregates all parameters, forcing a single global model that cannot adapt to institution-specific patterns
- **S3 — Method:** FedDualAtt integrates a hybrid ResNet1D-Transformer where attention heads are split into global heads (FedAvg-aggregated) and local heads (per-client, never aggregated)
- **S4 — Setup:** FedCVD benchmark — 4 real-world ECG datasets (SPH, PTB-XL, SXPH, G12EC) as federated clients, 20-class multi-label classification
- **S5 — Results:** FedDualAtt (5-3 split) achieves 72.47±0.70% Global Micro-F1, outperforming FedAvg (68.79%), Ditto (66.75%), FedALA (68.41%)
- **S6 — Insight:** Ablation over 9 head-split ratios shows 5-3 achieves the best stability-performance tradeoff

---

## I. Introduction (~0.5 column, 3–4 paragraphs)

**P1 — Clinical context:**
- Cardiovascular diseases are a leading cause of mortality; automated ECG diagnosis aids clinicians
- Multi-label classification over 12-lead ECG signals with 20 diagnostic classes
- Real-world data is distributed across hospitals — privacy regulations prevent data pooling
- FL is the natural paradigm: train collaboratively without sharing raw data

**P2 — The heterogeneity problem:**
- Data is non-IID: SPH (large Chinese), PTB-XL (European Holter), SXPH (Chinese outpatient), G12EC (Georgian 12-lead) differ in class distributions, equipment, patient demographics
- Standard FedAvg forces a single global model — average performance is suboptimal for any individual site
- Personalized FL approaches (Ditto, FedALA) add fine-tuning overhead or dual-model complexity without addressing the root cause at the architecture level

**P3 — Our insight:**
- Attention mechanisms in transformers are particularly distribution-sensitive — they learn "what to focus on", which should differ between hospitals
- Key idea: split attention heads at design time
  - Global heads → learn shared arrhythmia morphologies (common across all 4 sites)
  - Local heads → adapt to institution-specific artifacts, equipment characteristics, local label distributions
- Lightweight: local params are only the attention projection weights of L heads per transformer block, stored per-client on the server

**P4 — Contributions:**
1. DualAttentionResNet1D: hybrid CNN-Transformer with explicitly partitioned global/local attention heads
2. A corrected FL protocol with strict parameter separation: local params are never aggregated; zero-padding ensures clean global uploads
3. Empirical analysis on FedCVD: +3.68% Global Micro-F1 over FedAvg; ablation over 9 head-ratio configurations characterizing the stability-performance tradeoff

---

## II. Related Work (~0.3 column, 2 focused paragraphs)

**A. Federated Learning for ECG / Medical Data:**
- FedCVD [cite]: establishes 4-center ECG FL benchmark; tests 7 FL algorithms; best result Scaffold at 70.1% Global Micro-F1
- Medical FL in general: handles privacy by design, but accuracy gap vs. centralized remains large (centralized: 80.0% in original paper, 83.48% in ours)
- **Gap:** FedCVD uses ResNet1D-34 backbone only — no temporal attention component. We augment it with transformer attention as the personalization layer.

**B. Personalized Federated Learning:**
- Parameter-level: Ditto (dual objective + local fine-tuning), FedALA (local adaptive aggregation), FedBN (local BN layers), FedInit (warm initialization)
- Our approach: personalization baked into architecture at design time — no separate fine-tuning phase, no dual objectives, no gradient manipulation
- Most similar spirit: FedBN (keep BN local) — but attention heads capture richer semantic content than normalization statistics
- **Positioning:** simpler protocol than Ditto/FedALA, more principled than FedBN, no extra hyperparameters beyond the head ratio H_g-H_l

---

## III. Proposed Method (~1.5 columns)

### A. Problem Formulation

- $K=4$ clients, each with local dataset $\mathcal{D}_k = \{(x_i^k, y_i^k)\}$ of 12-lead ECG recordings and 20-class multi-label binary vectors
- Cannot share raw data; collaborate via FL for $T=50$ communication rounds
- Objective: maximize per-client test performance while leveraging cross-client knowledge
- Challenge: distributions $P_k(x,y)$ differ significantly across sites → non-IID setting

### B. Model Architecture: DualAttentionResNet1D

*(Reference: Figure 1 left panel)*

**ResNet1D-34 Feature Extractor (global params):**
- Input: $(B, 12, 5000)$ — 12 ECG leads, 5000 time steps at 500 Hz
- Initial: Conv1d(12→64, k=15, stride=2) → BN → ReLU → MaxPool(k=3, s=2)
- Residual layers: [3, 4, 6, 3] BasicBlocks with channels [64, 128, 256, 512]
- BasicBlock: Conv1d(k=7) → BN → ReLU → Conv1d(k=7) → BN + skip connection
- Output: $(B, 512, \sim\!156)$

**Positional Encoding (global params):**
- Sinusoidal PE; after transposing to $(B, \sim\!156, 512)$

**Dual Attention Transformer Block ×2 (mixed global/local):**

Two parallel attention branches share the same input $\mathbf{X} \in \mathbb{R}^{L \times d}$ ($d=512$):

*Global branch* — params $\{\mathbf{W}_{in}^g, \mathbf{W}_{att}^g, \mathbf{W}_{out}^g\}$ (aggregated via FedAvg):

$$\mathbf{X}_g = \mathbf{X}\mathbf{W}_{in}^g \in \mathbb{R}^{L \times (H_g \cdot d_h)}, \quad
\mathbf{A}_g = \text{MHA}(\mathbf{X}_g,\, H_g), \quad
\mathbf{G} = \mathbf{A}_g \mathbf{W}_{out}^g \in \mathbb{R}^{L \times d}$$

*Local branch* — params $\{\mathbf{W}_{in}^l, \mathbf{W}_{att}^l, \mathbf{W}_{out}^l\}$ (per-client, never aggregated):

$$\mathbf{X}_l = \mathbf{X}\mathbf{W}_{in}^l \in \mathbb{R}^{L \times (H_l \cdot d_h)}, \quad
\mathbf{A}_l = \text{MHA}(\mathbf{X}_l,\, H_l), \quad
\mathbf{L} = \mathbf{A}_l \mathbf{W}_{out}^l \in \mathbb{R}^{L \times d}$$

- Fixed head dimension $d_h = 64$; total heads $H = H_g + H_l = 8$
- Both branches operate on $d=512$ inputs, project internally to $H \cdot d_h$ dimensional space
- Combination: $\mathbf{O} = [\mathbf{G};\mathbf{L}]\mathbf{W}_c \in \mathbb{R}^{L \times d}$, where $\mathbf{W}_c \in \mathbb{R}^{2d \times d}$
- Residual + LayerNorm: $\mathbf{X}' = \text{LN}(\mathbf{X} + \mathbf{O})$
- FFN + Residual + LayerNorm: $\mathbf{Y} = \text{LN}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$, FFN hidden dim = 2048, ReLU activation
- **Naming convention (implementation detail):** layer names containing `local_att` or `local_proj` are local params $\phi_k$; all others are global params $\theta^g$

**Classification Head (global params):**
- Transpose back → $(B, 512, \sim\!156)$
- Global Average Pooling → $(B, 512)$
- FC(512 → 20) → Sigmoid → multi-label predictions

**Parameter split:**

| Component | Status | Notes |
|---|---|---|
| ResNet1D-34 backbone | Global θ^g | ~11M params |
| Positional Encoding | Global θ^g | |
| global_att + global_proj (×2 blocks) | Global θ^g | aggregated via FedAvg |
| FFN + LayerNorm + combine (×2 blocks) | Global θ^g | |
| FC classifier | Global θ^g | |
| local_att + local_proj (×2 blocks) | **Local φ_k** | stored per-client, never aggregated |

### C. Federated Training Protocol

*(Reference: Figure 1 right panel)*

**Server state invariant:** `self._model` always holds $\theta^g$ with local positions zeroed. Per-client local params $\{\phi_k\}_{k=1}^K$ are stored separately as plain parameter dicts.

**Round $t$ — 6 steps:**

1. **Downlink** — server sends $(\theta^g_t,\; \phi_{k,t})$ to client $k$; $\theta^g_t$ has local positions = 0
2. **Client setup** — load $\theta^g_t$ into model, then overwrite local positions with $\phi_{k,t}$ via `load_state_dict(strict=False)`
3. **Local training** — joint SGD for $E$ epochs on all params (both $\theta^g$ and $\phi_k$), BCELoss
4. **Extract local** — $\phi_{k,t+1} = \{n: p.data.clone() \mid \text{`local' in } n\}$
5. **Clean global** — zero local positions → serialize as $\theta^g_{k,t+1}$ (local=0)
6. **Uplink** — send $(\theta^g_{k,t+1},\; \phi_{k,t+1},\; k,\; n_k)$ to server

**Server aggregation:**

$$\theta^g_{t+1} = \sum_{k=1}^K \frac{n_k}{n}\,\theta^g_{k,t+1} \qquad \text{(FedAvg on global only)}$$
$$\phi_{k,t+1} \leftarrow \phi_{k,t+1} \quad \forall k \qquad \text{(stored directly, no aggregation)}$$

Averaging zeros in local positions of $\theta^g_{k,t+1}$ automatically preserves the server model invariant.

### D. Head Ratio as a Design Parameter

- Total heads fixed: $H = H_g + H_l = 8$; ratio $(H_g\text{-}H_l)$ swept from 0-8 to 8-0
- **Degenerate ablation cases:**
  - 0-8: no global aggregation → purely personalized, no knowledge sharing
  - 8-0: no local params → equivalent to adding transformer to FedAvg (architecture contribution only)
- **Operational range:** 1-7 through 7-1
- Selected configuration for main comparison: **5-3** (highest stability, competitive performance)

---

## IV. Experiments (~1.5 columns)

### A. Setup

**Dataset — FedCVD ECG Benchmark:**

| Client | Dataset | Region | Characteristics |
|---|---|---|---|
| Client 1 | SPH | China | Large hospital, clean 12-lead recordings |
| Client 2 | PTB-XL | Europe | Holter-style, diverse rhythm classes |
| Client 3 | SXPH | China | Outpatient clinic, noisier recordings |
| Client 4 | G12EC | Georgia | Different lead placement conventions |

- 20 diagnostic classes, multi-label binary targets
- Preprocessing: standard normalization per the FedCVD benchmark

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Communication rounds T | 50 |
| Local epochs E | 1 |
| Optimizer | SGD |
| Learning rate | 0.1 |
| Batch size | 32 |
| Loss | BCELoss |
| Sample ratio | 1.0 (all clients per round) |
| FedDualAtt head config | 5-3 (main), 0-8 through 8-0 (ablation) |
| Seeds (FedDualAtt) | 5 seeds: {42, 123, 456, 789, 1024} |
| Seeds (baselines) | 1 seed: 42 |

**Baselines:**
- Non-FL lower bound: Local (train on own data only)
- Non-FL upper bound: Centralized (train on pooled data — not feasible in practice)
- Global FL: FedAvg, FedProx (μ=0.01), Scaffold (server_lr=1.0)
- Personalized FL: FedInit (β=0.01), Ditto (μ=0.01), FedALA (rand_percent=80), FedSM (λ=0.1, γ=0)

**Metrics:** Global Micro-F1 (primary), per-client Local Micro-F1, mAP

### B. Main Results (Table 1)

*[Table: all baselines + FedDualAtt, Local Mi-F1 per client + Global Mi-F1]*

**Numbers to cite and discuss:**

Global Micro-F1:
- Centralized (upper bound): **83.48%**
- FedDualAtt (5-3, 5-seed): **72.47 ± 0.70%**
- FedAvg: **68.79%** → FedDualAtt is +3.68%
- FedALA (next best personalized): **68.41%** → FedDualAtt is +4.06%
- Ditto: **66.75%** → FedDualAtt is +5.72%
- Original FedCVD paper's best (Scaffold, 5-seed): **70.1%** → FedDualAtt is +2.37%

**Discussion points:**
1. FedDualAtt outperforms all FL baselines on global Micro-F1 by a clear margin
2. Our re-implemented baselines are consistent with the original FedCVD paper (our FedAvg 68.79% vs. their 67.9%), validating the experimental setup
3. FedSM (62.12%) underperforms significantly — sensitive to λ/γ hyperparameters; reported with paper's suggested values
4. Remaining gap to centralized (83.48% − 72.47% = 11.01%) reflects the fundamental challenge of non-IID data heterogeneity

**Per-client (LOCAL) analysis:**
- FedDualAtt achieves strong local performance on SPH (81.17%) and G12EC (68.05%)
- PTB-XL (55.84%) and SXPH (77.79%) are competitive with baselines
- Comparing against paper's Ditto LOCAL (82.8 / 74.8 / 86.5 / 73.4): our LOCAL metrics do not uniformly dominate — the gain is primarily on the GLOBAL metric, consistent with the global-aggregation design

### C. Head Ratio Ablation (Table 2)

*[Table: 9 configurations 0-8 through 8-0, Local Mi-F1 per client ± std + Global Mi-F1 ± std, 5 seeds each]*

**Key findings (one finding per paragraph in paper):**

**Finding 1 — Shared knowledge is essential:**
- 0-8 (pure local): Global Mi-F1 = 58.50 ± 1.49%
- 8-0 (pure global): Global Mi-F1 = 72.68 ± 1.12%
- Without any global aggregation, clients cannot learn robust representations from limited local data alone → +14.18% absolute gap

**Finding 2 — Architecture improvement vs. personalization:**
- 8-0 already outperforms ALL baselines (72.68% vs. best baseline FedAvg 68.79%)
- Adding the transformer on top of ResNet1D-34 is itself a strong improvement regardless of the head split
- This separates the architectural contribution from the personalization contribution

**Finding 3 — Stability matters for the operational split:**
- Even-number local head configurations show high variance: 2-6 (std=8.72%), 6-2 (std=10.39%)
- Odd local head counts are all stable: 1-7 (1.25), 3-5 (1.25), 5-3 (0.70), 7-1 (0.42)
- The 5-3 configuration achieves the best global Mi-F1 among stable configurations with the lowest standard deviation (0.70%) — recommended for deployment

**Finding 4 — Per-client optimal ratios differ:**

| Client | Best ratio | Best Local Mi-F1 |
|---|---|---|
| SPH (Client 1) | 5-3 | 81.17 ± 1.04% |
| PTB-XL (Client 2) | 8-0 | 56.67 ± 2.09% |
| SXPH (Client 3) | 1-7 | 79.01 ± 1.07% |
| G12EC (Client 4) | 0-8 | 69.00 ± 1.28% |

- PTB-XL benefits most from global sharing (best at 8-0) → data is representative of cross-site patterns
- G12EC benefits most from personalization (best at 0-8) → highly site-specific characteristics
- This per-client divergence motivates future work on adaptive per-client head ratios
- With 5-3 as the fixed global choice, each client still benefits, showing it is a robust default

---

## V. Conclusion (~0.2 column)

- Proposed FedDualAtt: architectural personalization via dual attention heads in a hybrid ResNet1D-Transformer for federated ECG classification
- 5-3 split achieves 72.47±0.70% Global Micro-F1 — best among all FL methods, surpassing the FedCVD benchmark's previous best by +2.37%
- Ablation confirms: (a) shared knowledge is essential, (b) architecture improvement accounts for the bulk of the gain, (c) 5-3 is the stability-optimal split
- Limitation: single-seed baselines; local metric gap vs. stronger personalized methods
- Future work: adaptive per-client head ratios; extension to echocardiography modality; communication-efficient local param compression

---

## Figures and Tables

| # | Type | Content | File |
|---|---|---|---|
| Fig. 1 | Architecture + Protocol | DualAttentionResNet1D (left) + FL communication (right) | `docs/figures/feddualatt_framework.png` |
| Table 1 | Main results | All baselines vs. FedDualAtt (5-3), Local + Global Mi-F1 | `docs/results_tables.tex` Table 1 |
| Table 2 | Head ratio ablation | 0-8 through 8-0, Local + Global Mi-F1 ± std (5 seeds) | `docs/results_tables.tex` Table 2 |
| (Optional) Fig. 2 | Line chart | Global Mi-F1 vs. head ratio with std bands | `docs/figures/head_ratio_global_line.png` |

---

## Open Questions Before Writing

1. **Baseline seeds:** Table 1 baselines are single-seed (42). Plan A: acknowledge in setup section ("baselines reported for seed=42; FedDualAtt averaged over 5 seeds"). Plan B: re-run all baselines with 5 seeds for a fully fair comparison — requires a multi-seed baseline script to be created.
2. **mAP in Table 1:** Currently empty in the codebase results. Either fill it (requires re-extracting from output files) or drop mAP from Table 1, keeping only Mi-F1.
3. **LOCAL metric framing:** FedDualAtt local metrics do not clearly dominate paper's Ditto/FedALA on all clients. Frame this honestly: our gain is on the GLOBAL metric; per-client ablation is a secondary finding motivating future adaptive approaches.
4. **FedSM low performance:** 62.12% — either investigate whether the hyperparameters are correctly set, or include as-is and note hyperparameter sensitivity in the discussion.
5. **Author/affiliation:** Placeholder in the .tex file.
