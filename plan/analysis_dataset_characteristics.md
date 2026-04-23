---
name: Dataset Characteristics Analysis
description: PARTIALLY ADDRESSED. Dataset bottleneck identified (N=1 saturation). Transfer learning with ECGFounder is the solution — pre-trained on 10.7M ECGs vs FedCVD's ~20K. Per-client statistics analysis deferred.
type: project
---

# Analysis: Dataset Characteristics vs. Optimal Head Ratio

**Status: NOT IMPLEMENTED** — analysis only; no model training required
**Cost:** ~200 lines in a new `scripts/ecg/analyze_dataset_characteristics.py`

---

## What It Is

Compute per-client dataset statistics (size, label entropy, label distribution distance,
etc.) and correlate them with the optimal head ratio per client observed in the ratio sweep.
The goal is to explain **why** different clients prefer different global/local splits and
to provide a principled basis for choosing the ratio in new federated deployments.

---

## Motivation

The ratio sweep shows that different clients benefit from different ratios:
- SPH: best at 5G:3L (large, representative dataset → benefits from global sharing)
- SXPH: best at 1G:7L (smaller, distribution-shifted dataset → benefits from local adaptation)
- G12EC: best at 1G:7L (smaller, distribution-shifted → same pattern)
- PTB-XL: mixed response (largest dataset, but different label distribution)

**Question:** Can we predict which ratio a client will prefer from statistics we can
compute without training? If yes, this is practically useful: new federated deployments
can select the ratio analytically rather than by ablation sweep.

This analysis also motivates change_03 (adaptive head ratio): the correlation between
dataset statistics and optimal ratio is the signal that the adaptive algorithm should track.

---

## Dataset Statistics to Compute

### 1. Dataset Size (n_k)

Number of training samples per client. Already known approximately:
- SPH: largest (~10k+ samples)
- PTB-XL: second largest (~5k samples)
- SXPH: smaller (~2k samples)
- G12EC: smallest (~1k samples)

**Hypothesis:** larger datasets benefit more from global heads (more data → better
utilization of globally shared representations). Smaller datasets benefit more from local
heads (less data → risk of forgetting under global aggregation).

### 2. Label Entropy (H_k)

For each client k, compute:

```
H_k = -Σ_{c=1}^{20} p_k(c) × log p_k(c)
```

where `p_k(c)` = proportion of positive samples for class c in client k's dataset.

High entropy → diverse, balanced label distribution (representative of the global distribution).
Low entropy → skewed label distribution (few dominant classes → stronger distribution shift).

**Hypothesis:** clients with high label entropy benefit more from global heads (their
distribution matches the global more closely). Clients with low entropy benefit from local
heads (their skewed distribution diverges from other clients).

### 3. Earth Mover's Distance (EMD) Between Clients

For each pair of clients (j, k), compute:

```
EMD(p_j, p_k) = Σ_{c=1}^{20} |CDF_j(c) - CDF_k(c)|
```

This measures how different two clients' label distributions are. A client with high mean
EMD to other clients has the most distribution shift.

**Hypothesis:** clients with high mean EMD to others benefit more from local heads (their
distribution is most different from the rest of the federation). Clients close to the
federation center benefit from global heads.

### 4. Label Prevalence Vector (q_k)

For each client k:
```
q_k = [proportion of class c positive in k for c in 1..20]
```

Pairwise cosine similarity between q_j and q_k gives a simple measure of label
distribution alignment. High cosine similarity → clients can share more.

### 5. Intra-Client Class Co-occurrence

Compute the 20×20 label co-occurrence matrix per client. Clients with unique co-occurrence
patterns (relative to others) have more site-specific multi-label structure that local
heads can learn to exploit.

---

## Correlation Analysis

### Per-Client Optimal Ratio

From the ratio sweep results (CSV: `head_ratio_all_metrics.csv`), for each client:

1. Extract per-client Micro-F1 (or mAP) for each of the 9 head ratio configs.
2. Find the ratio config that maximizes per-client metric.
3. Record: `optimal_local_heads_k = argmax over configs of per_client_f1_k`

### Regression

For each dataset statistic x_k (n_k, H_k, EMD_k, etc.):
- Scatter plot: x_k vs. optimal_local_heads_k (4 points, one per client)
- Compute Pearson correlation coefficient

Given only 4 data points, formal statistical testing is underpowered. The analysis is
meant to be **descriptive and illustrative**, not inferential. Stronger claims would
require data from more sites.

### Visualization

1. **Scatter plots:** one per statistic, x = statistic value, y = optimal local head count
   - Color-code points by client name
   - Annotate each point with the client name
   - Add a linear regression line (OLS)

2. **Radar / spider chart:** per-client profile of all statistics, normalized to [0,1]
   - Visually shows which clients are outliers on each dimension

3. **Heatmap:** clients × statistics, values normalized per column
   - Highlights which clients cluster together on which statistics

---

## What to Implement

**New file:** `scripts/ecg/analyze_dataset_characteristics.py`

Structure:

```python
# 1. Load data for each client
#    Use the existing FedCVD data loader or load CSV/numpy files directly
#    from output/processed/ or wherever FedCVD stores per-client splits

# 2. Compute statistics per client
def compute_label_entropy(y_multilabel: np.ndarray) -> float: ...
def compute_prevalence_vector(y_multilabel: np.ndarray) -> np.ndarray: ...
def compute_emd(p1: np.ndarray, p2: np.ndarray) -> float: ...

# 3. Load ratio sweep results
#    Read head_ratio_all_metrics.csv
#    Compute per-client optimal ratio

# 4. Correlation analysis
#    For each stat, compute Pearson r with optimal local head count

# 5. Visualization
#    Save scatter plots to docs/figures/dataset_characteristics_*.png
```

---

## Expected Findings

Based on the ratio sweep results:

| Client | Dataset Size | Expected Label Entropy | Expected EMD to Others | Optimal Local Heads |
|---|---|---|---|---|
| SPH | Largest | High (many classes) | Low (most representative) | Few (2–4) |
| PTB-XL | Second largest | High | Low–medium | Few–moderate |
| SXPH | Smaller | Lower | High | Many (6–8) |
| G12EC | Smallest | Lower | High | Many (6–8) |

If this pattern holds, the analysis would support:

1. **Large, representative clients benefit from global heads** — they already represent
   the federation's distribution well, so shared representations are high quality for them.

2. **Small, distribution-shifted clients benefit from local heads** — the global model
   has insufficient capacity to specialize for their unique patterns; local heads fill this gap.

This would connect the architectural design (global/local head split) to the FL
heterogeneity literature, which consistently shows that clients with more distribution
shift need more personalization capacity.

---

## Literature Grounding

**FedAlign (CVPR 2022)** uses second-order statistics (covariance of feature maps) to
characterize heterogeneity and align representations. Our analysis uses label statistics
rather than feature statistics — computationally simpler and does not require a trained model.

**FedRep (Collins et al., ICML 2021)** assumes heterogeneity is concentrated in label
distributions rather than input features. Our analysis tests whether this assumption holds
for the FedCVD ECG dataset.

**Measuring Statistical Heterogeneity in FL:** Several papers (e.g., Zhao et al., 2018;
Li et al., 2020) use earth mover's distance between class distributions as the primary
measure of non-IID degree. Our analysis applies the same metric to a multi-label setting
(treating each class independently).

---

## Paper Use

**Section 6 (Analysis):**

> *"To understand why different clients prefer different head ratios, we compute per-client
> dataset statistics and correlate them with the optimal ratio observed in the ablation
> sweep. We find that [SXPH, G12EC] — which are smaller and have more distinct label
> distributions — benefit most from local heads, while [SPH, PTB-XL] — larger and more
> representative — benefit from global heads. Label entropy is the strongest predictor of
> optimal local head count (r = X.XX), suggesting that dataset representativeness is the
> primary driver of the personalization-generalization trade-off in heterogeneous ECG
> classification."*

This motivates change_03 (adaptive head ratio): if label entropy predicts the optimal ratio,
a system that estimates this online could set the ratio adaptively.
