---
name: Adaptive Head Ratio
description: DEFERRED. Fusion ablation showed concat > gate > scalar > add. Adaptive gating did not outperform fixed concat. May revisit with pre-trained backbone if richer features make gating more useful.
type: project
---

# Change 03: Adaptive Head Ratio

**Status: NOT IMPLEMENTED** — research direction; future work or extended paper
**Cost:** moderate — gradient hooks + per-head tracking + dynamic routing logic

---

## What It Is

Instead of a fixed H_g:H_l ratio chosen before training, compute per-round gradient cosine
similarity across clients for each attention head. Heads whose cross-client gradients are
conflicting (negative cosine similarity) → assigned to local branch. Heads with consistent
cross-client gradients (positive similarity) → kept in global branch. The ratio becomes
**dynamic and data-driven**.

---

## Motivation

The current approach requires a manual ablation sweep (9 configs × 5 seeds = 45 runs) to
find the optimal global/local split. The optimal ratio also varies by client (SPH prefers
5G:3L, SXPH prefers 1G:7L) — no single fixed ratio is universally optimal.

**Key insight from gradient isolation (change_01):** the reason some heads should be local
is that their gradients are client-specific and conflict across sites. Gradient cosine
similarity directly measures this property — it is the causal signal behind why a head
should be local vs. global.

If we measure this signal per head, per round, we can automate the global/local assignment
decision entirely.

---

## How It Works

### Algorithm

Each communication round:

1. **Forward + backward pass** on each client's local data.

2. **Extract per-head gradient vectors** for both global and local attention projection
   weights (`global_proj_in`, `local_proj_in`, `global_att`, `local_att`). This can be done
   via backward hooks on each head's projection parameters.

3. **Compute pairwise cosine similarity** across all 4 clients for each head i:

   ```
   sim_i = mean over all pairs (j,k): cos(∇_i^j, ∇_i^k)
   ```

4. **Threshold decision:**
   - If `sim_i < τ`: head i's gradient is conflicting across clients → assign to local
     branch next round (do not aggregate)
   - If `sim_i ≥ τ`: head i's gradient is consistent → assign to global branch next round
     (aggregate normally)

5. **Update routing flags** in `DualAttentionTransformerBlock` — which heads are "global"
   and which are "local" — before the next round's forward pass.

6. **Track convergence:** monitor how often heads switch branches per round. Expect the
   ratio to converge to a stable configuration within ~20-50 rounds.

### Threshold τ

`τ = 0` is the natural default (negative similarity = conflict). May need tuning per dataset.
A ramp schedule (`τ_t = τ_0 × decay^t`) could relax the constraint early in training when
gradients are noisy.

---

## What to Implement

### 1. Dynamic head routing in `DualAttentionTransformerBlock`

Instead of `self.local_heads` fixed at init, maintain a per-head assignment vector:

```python
# In __init__:
self.num_heads = total_heads
self.is_local = torch.zeros(total_heads, dtype=torch.bool)  # all global initially

# In forward():
# Route head i to global or local based on self.is_local[i]
# This requires splitting the multi-head attention into individual head computations
# (or using a masked approach if using PyTorch's nn.MultiheadAttention)
```

**Implementation note:** PyTorch's `nn.MultiheadAttention` does not easily support per-head
masking. May need to implement attention manually (split Q, K, V projections, compute
attention per head, concatenate, project out) to enable per-head routing.

### 2. Gradient hooks in `feddualatt.py`

Register backward hooks on each head's Q/K/V projection weight during local training.
After `loss.backward()`, collect per-head gradient vectors and store them.

### 3. Server-side routing update

After all clients report, the server computes per-head cosine similarities and sends
updated routing assignments along with the global parameters for the next round.

Alternatively, each client independently computes its own similarity signal and uses a
majority-vote rule (≥3 of 4 clients report conflict → assign local).

### 4. Parameter aggregation logic

The current `_is_local()` check in `algorithm/ecg/feddualatt.py` uses string matching.
With dynamic routing, this needs to be a per-round computed set of parameter names, not
a static string match.

---

## Expected Behavior

- **Convergence:** head assignments should stabilize within ~20-50 rounds. After convergence,
  the ratio should approximately match the best fixed ratio from the ablation sweep.
- **Per-client F1:** similar to or better than the best fixed-ratio result from the sweep.
- **Global F1:** stable at ~72.7% (gradient isolation is implicitly applied to local-assigned
  heads since they are not aggregated and their routing is based on conflict detection).
- **Early training:** assignments may be unstable early (noisy gradients). Using a burn-in
  period (all heads global for first R rounds) may help stability.
- **Client-specific assignments:** different clients may report different conflict patterns.
  The server needs a consensus rule. Options: majority vote, union of local assignments,
  or weighted by dataset size.

---

## Comparison to Fixed Ratio

| Property | Fixed ratio | Adaptive ratio |
|---|---|---|
| How ratio is chosen | Manual sweep | Gradient-driven |
| Runs needed | 9 configs × 5 seeds | 1 run |
| Adaptability to new datasets | Re-sweep required | Automatic |
| Per-client optimality | Single ratio for all clients | Potentially per-client |
| Implementation complexity | Simple | Moderate–high |
| Runtime overhead | None | Backward hooks (minor) |
| Paper contribution | Characterizes trade-off | Eliminates need for sweep |

---

## Literature Grounding

**FedLAG (Nguyen et al., arXiv 2024)** does this at the **layer** level: compute cross-client
gradient cosine similarity per layer; layers with high conflict → keep local; layers with low
conflict → aggregate. Our work does this at the **head** level, providing finer-grained
control within the attention mechanism.

**PCGrad (Yu et al., NeurIPS 2020)** defines conflict via negative cosine similarity and uses
it to modify gradients. We use the same metric to make a routing decision rather than a
gradient modification — conceptually simpler and more interpretable.

**FedRep (Collins et al., ICML 2021)** and **FedPerfix (Sun et al., ICCV 2023)** both
use static partitioning (backbone global, head local). Adaptive routing is strictly more
general: a static partition is a degenerate case where the routing never changes.

---

## Risks

1. **Noisy gradient similarity in early rounds** — may cause head assignments to oscillate.
   Mitigation: moving average of cosine similarity over multiple rounds, or burn-in period.

2. **Implementing per-head routing without native PyTorch support** — requires a custom
   attention module. Non-trivial, but straightforward given the existing codebase structure.

3. **Server-side consensus rule** — no single right answer. Majority vote is simple but
   may lose information. Weighted vote (by dataset size) may be more principled.

4. **τ hyperparameter** — adds one more degree of freedom. If τ = 0 does not work well,
   sweeping τ partially defeats the purpose of avoiding the ratio sweep.

---

## Paper Framing

**Future work section in submitted paper's follow-up:**

> *The head ratio was determined by an ablation sweep. A natural extension is to determine
> the ratio automatically from the training signal. Since gradient interference is the root
> cause of the trade-off, we can use cross-client gradient cosine similarity — computed per
> attention head during training — to dynamically route heads to global or local branches.
> This eliminates the ablation sweep and adapts the ratio to the specific statistical
> heterogeneity of the federation.*

**Or as a standalone contribution** if the adaptive ratio demonstrably outperforms the best
fixed ratio from the sweep.
