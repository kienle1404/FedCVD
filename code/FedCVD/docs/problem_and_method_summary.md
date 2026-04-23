# FedDualAtt: Problem Statement and Proposed Method

## The Problem: Gradient Interference in Split-Architecture Federated Learning

### Context

Personalized federated learning (pFL) methods often split a model into **shared** and **personal** components. The shared part is aggregated across clients (e.g., via FedAvg) to learn generalizable features, while the personal part stays on each client to capture local patterns.

A common architecture pattern is: **shared backbone → personal head(s)**. Examples:

| Method | Shared | Personal | Architecture |
|--------|--------|----------|-------------|
| FedPer | Lower layers (feature extractor) | Upper layers (classifier head) | Sequential split |
| FedRep | Representation layers | Classification head | Sequential split |
| FedTP | Backbone | Per-client transformer heads | Parallel split |
| FedBN | Everything except BN | BatchNorm layers | Layer-type split |
| **FedDualAtt (ours)** | Backbone + global attention | Local attention heads | Parallel branches on shared backbone |

### The Undiagnosed Problem

When personal components are attached to a shared backbone and **trained jointly**, the personal branch creates a gradient path back to the backbone during backpropagation:

```
loss ← personal_head(backbone(x)) + shared_head(backbone(x))
                │                           │
                ▼                           ▼
        ∂L/∂backbone (client-specific)  +  ∂L/∂backbone (global)
```

The backbone receives the **sum** of both gradient signals. The personal gradient is client-specific — it optimizes the backbone for client k's data distribution. Across K clients, this produces K conflicting gradient directions for the shared backbone every round.

**This is NOT the same as the well-studied client drift / gradient divergence problem:**

| | Client drift (well-studied) | Gradient interference (our diagnosis) |
|---|---|---|
| Where | Between clients (inter-client) | Within a single model (intra-model) |
| What diverges | Full model updates across clients | Shared vs personal gradient signals within one client |
| Cause | Non-IID data across clients | Dual gradient paths through shared backbone |
| Prior solutions | FedProx, SCAFFOLD, FedNova | **None — this is the gap** |
| Affected methods | All FL methods | Only split-architecture pFL methods |

### Empirical Evidence (FedDualAtt on FedCVD)

**Experiment 1: Fixed global heads + add local heads (no fix)**

With 8 global attention heads fixed, we added local heads on top:

| Config | Total Heads | Global F1 | Change |
|--------|-------------|-----------|--------|
| 8G:0L | 8 | 72.68% | baseline |
| 8G:2L | 10 | 64.5% | -8.2pp |
| 8G:4L | 12 | 58.7% | -14.0pp |
| 8G:8L | 16 | 56.0% | -16.7pp |

Global F1 drops monotonically despite all 8 global heads being preserved. This rules out capacity constraints — the degradation is caused by the local branch interfering with backbone training.

**Experiment 2: Gradient norm measurement**

| Config | Backbone gradient (no fix) | Backbone gradient (with fix) | Reduction |
|--------|---------------------------|------------------------------|-----------|
| 6G:2L | 0.0692 | 0.0293 | **57.6%** |
| 4G:4L | 0.0719 | 0.0363 | **49.6%** |
| 2G:6L | 0.0675 | 0.0322 | **52.3%** |
| 0G:8L | 0.0657 | 0.0000 | **100.0%** |

Without the fix, local branches inject approximately **50% of the total backbone gradient norm** — a corruption signal as strong as the global learning signal itself.

---

## The Proposed Fix: Gradient Isolation via `detach()`

### Core Idea

Insert a gradient wall where the backbone output enters the personal (local) branch. The personal branch sees the same feature values (forward pass unchanged) but its gradients cannot flow back to the backbone (backward pass severed).

### Implementation

One line of code change:

```python
# Before (gradient interference):
local_in = self.local_proj_in(x)              # x links to backbone graph
x_local = self.norm(x + local_out)            # residual also links

# After (gradient isolation):
x_detached = x.detach()                       # sever computation graph
local_in = self.local_proj_in(x_detached)     # no gradient path to backbone
x_local = self.norm(x_detached + local_out)   # residual also severed
```

### Why It Works

```
Without detach():
  backbone ← ∂L/∂x_global (aggregated, consensus signal)
           + ∂L/∂x_local  (client-specific, conflicting signal)
  → backbone is pulled in K conflicting directions each round

With detach():
  backbone ← ∂L/∂x_global only
  local branch ← ∂L/∂x_local (stops at detach boundary)
  → backbone learns consensus features
  → local branch re-weights existing features for each client
```

### Properties

| Property | Evidence |
|----------|----------|
| **Eliminates interference** | Backbone gradient reduced 50-58% for mixed configs |
| **Zero computational overhead** | `detach()` is O(1), no extra memory or FLOPs |
| **Zero convergence penalty** | All configs converge at same speed (~9-12 rounds to 90% of final) |
| **Eliminates global-local trade-off** | Global F1 stays flat ~72-74% regardless of how many local heads are added |
| **Surgical** | Only backbone gradients are affected; local head, FFN, norm gradients unchanged |
| **Boundary condition holds** | 0G:8L + detach = exactly zero backbone gradient (complete isolation verified) |

### Results After Fix

**8-head ratio sweep with detach():**

| Config | Global F1 (no fix) | Global F1 (with fix) |
|--------|-------------------|---------------------|
| 8G:0L | 72.68% | ~72% |
| 6G:2L | unstable | ~73% |
| 4G:4L | 71.21% (high variance) | ~72% |
| 2G:6L | unstable | ~73% |
| 0G:8L | 58.50% | ~40% (expected: no global gradient to backbone) |

With the fix, global F1 remains flat across all configs (except 0G:8L where zero global heads means no backbone training signal). The trade-off between personalization and generalization is eliminated.

---

## Positioning: What's Novel

### Contribution 1: Diagnosis
We identify **intra-model gradient interference** as a previously undiagnosed failure mode in split-architecture pFL. All prior work on FL convergence (FedProx, SCAFFOLD, etc.) addresses inter-client gradient divergence. No prior work examines how personal branches corrupt shared representations through backward gradient flow.

### Contribution 2: Fix
We propose **gradient isolation** via `detach()` — a zero-overhead, single-line fix that eliminates the interference. The fix is:
- **General:** Applicable to any split-architecture pFL method where personal components attach to a shared backbone (FedPer, FedTP, FedRep, etc.)
- **Simple:** No architectural changes, no extra hyperparameters, no alternating optimization
- **Empirically validated:** Gradient norm analysis directly measures the interference and confirms the fix eliminates exactly the problematic gradient component

### Contribution 3: Analysis
We provide comprehensive empirical analysis:
- Gradient norm decomposition (quantifies interference)
- Convergence analysis (proves zero penalty)
- Head count scaling N=1→32 (dataset saturation study)
- Cross-evaluation (clean personalization-generalization trade-off)
- Communication cost analysis
- Fusion mode ablation

---

## Comparison Axes for Literature Review

When comparing with related work, key dimensions to evaluate:

1. **Does the method have a shared backbone + personal branch?** If yes, it's potentially affected by gradient interference.

2. **How does the method handle joint training?**
   - Alternating optimization (FedRep, FedDecomp): trains shared and personal in separate phases → avoids interference but doubles training cost
   - Joint training (FedPer, FedTP, FedDualAtt without fix): trains everything together → susceptible to interference
   - Separate inputs (pFedDB): personal branch doesn't share backbone → no interference path

3. **Does the method diagnose or address gradient flow between components?** This is the gap — most methods simply train jointly without examining the gradient dynamics.

4. **What is the overhead of the personalization mechanism?**
   - Extra training phases (FedRep: 2× epochs)
   - Extra communication (FedProx: proximal term requires extra gradient computation)
   - Extra hyperparameters (Ditto: λ regularization weight; SCAFFOLD: control variates)
   - **Our fix: zero overhead**

### Related Methods to Compare

| Method | Personal component | Training strategy | Gradient interference? | Overhead |
|--------|-------------------|-------------------|----------------------|----------|
| FedAvg | None | Joint | N/A | None |
| FedPer | Classifier head | Joint | **Yes** (if head gradients reach backbone) | None |
| FedRep | Classifier head | Alternating | Avoided (separate phases) | 2× local epochs |
| FedTP | Per-client transformer | Joint | **Yes** (same mechanism as ours) | Per-client heads |
| FedBN | BatchNorm layers | Joint | Minimal (BN has small gradients) | None |
| Ditto | Full local model copy | Regularized | N/A (full model is personal) | 2× model storage |
| SCAFFOLD | None (control variates) | Joint | N/A | 2× communication |
| FedProx | None (regularization) | Regularized | N/A | Proximal term |
| pFedDB | Dual branch from input | Two-phase | **No** (no shared intermediate) | 2× training phases |
| **FedDualAtt** | Local attention heads | Joint + detach() | **Eliminated** | **Zero** |

### Key Literature Questions

1. **Has anyone diagnosed intra-model gradient interference in FL?** (We believe not — all prior gradient analysis focuses on inter-client divergence)
2. **Has anyone used `detach()` or stop-gradient in FL?** (Check contrastive FL methods like FedCA, MOON — they use stop-gradient for different reasons)
3. **How do alternating optimization methods (FedRep, FedDecomp) compare?** (They avoid interference by design but at 2× training cost — our fix achieves the same effect at zero cost)
4. **Does pFedDB's implicit gradient separation relate to our explicit `detach()`?** (pFedDB's branches start from raw input, so no shared backbone = no interference. Our contribution is fixing the case where a shared backbone exists.)
