---
name: Head Importance Scoring
description: DEFERRED. N=1 saturation makes head importance moot with from-scratch backbone. May become relevant with pre-trained backbone if head count scaling shows differentiation.
type: project
---

# Change 05: Head Importance Scoring

**Status: NOT IMPLEMENTED** — architectural extension; eliminates the ratio ablation sweep
**Cost:** model changes + loss function update; moderate implementation effort

---

## What It Is

Add a learned scalar gate per attention head. During training, gates are regularized toward
0 (L0 or L1 penalty on global head gates). Heads with low global gate values contribute
less to the global aggregate output and effectively become local. The global/local split is
**soft** and **learned**, not hard-coded.

This eliminates the ablation sweep entirely: a single training run discovers the optimal
ratio from data.

---

## Motivation

The current ratio sweep (9 configs × 5 seeds) requires 45 runs to characterize the
global/local trade-off for a single dataset. Different federated environments may have
different optimal ratios. A head importance gate:

1. Finds the optimal ratio in a single run
2. Adapts automatically to a new federation without re-sweeping
3. Provides interpretable per-head importance scores as an analytical output

---

## How It Works

### Gated Attention Output

For head i with scalar gate `g_i`:

```
output_i = g_i × Attention_i(Q, K, V)
```

If `g_i → 0`, head i contributes nothing to the final output. Its parameters are still
updated during training, but their effect vanishes.

For the global branch, gates are shared across clients and regularized toward sparsity.
For the local branch, no regularization is needed — local heads can contribute as much
as they want.

### Regularization

Add L1 regularization on global head gates to encourage sparsity:

```
L_total = L_task + λ × Σ_i |g_i^global|
```

As λ increases, more global heads are pruned. The surviving global heads are those whose
consistent cross-client signal is strong enough to overcome the L1 penalty.

**Alternative:** L0 regularization (as in Voita et al.) directly penalizes the number of
active heads rather than gate magnitude. L0 is non-differentiable but can be approximated
using the hard-concrete distribution (Louizos et al., ICLR 2018).

---

## What to Implement

### 1. Gate parameters in `DualAttentionTransformerBlock`

```python
# In __init__:
self.global_head_gates = nn.Parameter(torch.ones(self.global_heads))
self.local_head_gates = nn.Parameter(torch.ones(self.local_heads))

# In forward() — global branch:
global_att_out, _ = self.global_att(global_in, global_in, global_in)
# global_att_out shape: (batch, seq, num_heads × head_dim)
# Reshape to (batch, seq, num_heads, head_dim), apply gates, reshape back
gates = torch.sigmoid(self.global_head_gates)  # keep in [0, 1]
global_att_out = global_att_out.view(B, S, self.global_heads, self.head_dim)
global_att_out = global_att_out * gates.view(1, 1, -1, 1)
global_att_out = global_att_out.view(B, S, self.global_heads * self.head_dim)
```

Note: reshaping requires that `nn.MultiheadAttention` returns the pre-projection output,
or that a custom attention module is used. PyTorch's built-in MHA returns post-projection
output. May need custom implementation to apply per-head gates before the output projection.

### 2. Regularization loss in training step

```python
# In FedDualAttSerialClientTrainer.local_process():
reg_loss = 0.0
for block in model.blocks:
    reg_loss += block.global_head_gates.abs().sum()
total_loss = task_loss + lambda_gates * reg_loss
total_loss.backward()
```

### 3. FL aggregation of gates

The global head gates (`global_head_gates`) are **global parameters** — they are aggregated
via FedAvg along with other global parameters. The existing `_is_local()` string filter
will classify them as global if they are named `global_head_gates` (no `local_att` or
`local_proj` in name).

Local head gates (`local_head_gates`) are local parameters — never aggregated.

### 4. Inference: which heads are "active"?

After training, heads with `sigmoid(gate_i) > threshold` (e.g., 0.5) are considered active.
The effective ratio can be read off from the gate values:
- Fraction of global heads with gate > 0.5 = effective global head ratio
- Compare to best fixed ratio from ablation sweep

---

## Expected Behavior

- **Early training:** all gates start at 1.0 (all heads active). L1 pressure gradually
  pushes low-importance global gates toward 0.
- **Convergence:** gates converge to a stable pattern where the most globally consistent
  heads survive and the rest are pruned.
- **Discovered ratio:** should approximately match the best fixed ratio from the ablation
  sweep (e.g., around 5G:3L based on current results).
- **Global F1:** similar to best fixed-ratio result (~72.5%).
- **Per-client F1:** similar to or better than best fixed-ratio result, since the gate
  discovers which heads are genuinely useful globally vs. locally.

---

## Hyperparameter: λ (gate regularization strength)

λ controls the sparsity level:
- λ too small: no heads are pruned, gates stay at 1.0 → identical to no gating
- λ too large: all global heads pruned → collapses to 0G:8L behavior

**Suggested sweep:** λ ∈ {1e-4, 1e-3, 1e-2, 1e-1}
Expect the optimal λ to be around 1e-3 to 1e-2 (small enough that task loss dominates
but large enough to prune redundant heads).

---

## Comparison to Adaptive Head Ratio (change_03)

| Property | Head Gates (this change) | Adaptive Ratio (change_03) |
|---|---|---|
| How ratio is found | Gradient descent on gate values | Threshold on cosine similarity |
| Implementation | Differentiable; standard PyTorch | Custom hooks; non-differentiable decision |
| Hyperparameter | λ (regularization strength) | τ (cosine similarity threshold) |
| Interpretability | Gate magnitude per head | Cosine similarity per head |
| Runtime overhead | Gate multiply in forward pass | Backward hooks + cross-client aggregation |
| Theoretical basis | L0/L1 sparsity (well-studied) | Gradient conflict literature |

Both approaches eliminate the ablation sweep. Head gates are more mathematically principled
(differentiable); adaptive ratio is more computationally direct (uses the causal signal).

---

## Literature Grounding

**Voita et al. (ACL 2019)** — "Analyzing Multi-Head Self-Attention: Specialized Heads Do
the Heavy Lifting, the Rest Can Be Pruned." Uses L0 regularization via hard-concrete
relaxation to identify and prune redundant attention heads. The surviving heads are shown
to have specialized, interpretable functions.

**Michel et al. (NeurIPS 2019)** — "Are Sixteen Heads Really Better than One?" Removes
heads one at a time and measures performance drop. Most heads can be removed with minimal
impact, suggesting strong redundancy. Provides motivation for why L1 gating can find a
sparse effective subset.

**Hard-Concrete Distribution (Louizos et al., ICLR 2018)** — provides a differentiable
relaxation of L0 regularization. Each gate uses a stretched sigmoid that can exactly reach
0 and 1, enabling true sparsity while remaining differentiable.

---

## Paper Framing

**Future work or extended contribution:**

> *"The ablation sweep over head ratios reveals an optimal configuration but requires
> prior knowledge of the federation's heterogeneity. We propose head importance gating —
> learned scalar gates on attention heads regularized toward sparsity — as an automated
> alternative. In a single training run, gates converge to a sparse pattern that
> approximately recovers the optimal ratio from the sweep, without requiring any
> hyperparameter search over the ratio itself."*

This would be a natural extension contribution if the gating approach matches or exceeds
the best fixed-ratio result on held-out seeds.
