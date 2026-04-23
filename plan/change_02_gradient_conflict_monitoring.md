---
name: Gradient Conflict Monitoring
description: SUPERSEDED. Gradient norm analysis (analyze_gradient_norms.py) already provides the empirical evidence. Cosine similarity monitoring is no longer a priority — focus shifted to transfer learning.
type: project
---

# Change 02: Gradient Conflict Monitoring

**Status: NOT IMPLEMENTED** — diagnostic addition, no training changes
**Cost:** ~20 lines in `algorithm/ecg/feddualatt.py`; zero training overhead

---

## What It Is

Log the pairwise cosine similarity of backbone-output gradients across all 4 clients per
communication round. This provides **quantitative evidence** for the gradient interference
diagnosis: before the fix, cross-client gradients should be negatively correlated (conflicting);
after the fix, they should be near-zero or positively correlated.

This is a pure diagnostic — it does not change model training, FL protocol, or any outputs.
It runs on existing backward passes at zero additional cost.

---

## Motivation

The gradient isolation fix is motivated by the mechanistic claim:

> *Local head gradients flow backward through the shared backbone, creating conflicting
> cross-client gradient signals that degrade shared representations.*

The Phase 1 experiment provides **behavioral evidence** for this (global F1 drops with local
heads), but not direct mechanistic evidence. Gradient cosine similarity makes the conflict
directly visible.

**Paper use (Section 3 — Diagnosing the Trade-off):**
A figure showing gradient cosine similarity per round, with two curves (without fix / with
fix), directly supports the mechanistic claim. It answers: *how severe is the conflict, and
does the fix eliminate it?*

---

## How It Works

### Gradient Cosine Similarity

For clients j and k, the cosine similarity of their backbone-output gradients at round t is:

```
cos(∇_j, ∇_k) = (∇_j · ∇_k) / (||∇_j|| × ||∇_k||)
```

- `cos = +1`: gradients are aligned (same direction) — no conflict
- `cos = 0`: gradients are orthogonal — no direct conflict
- `cos = −1`: gradients are anti-aligned — maximum conflict

For C clients, we compute all C(C−1)/2 = 6 pairwise similarities and log their mean.

### When to Compute

After `loss.backward()` and before `optimizer.step()` in the local update, extract the
gradient at the backbone output (last ResNet feature map before the attention blocks).

### What to Extract

The backbone output is the input `x` to `DualAttentionTransformerBlock`. This is
`resnet_features` in `DualAttentionResNet1D.forward()` — the output of the ResNet backbone
before the transformer blocks.

In practice, register a backward hook on the first transformer block's input to capture the
gradient tensor: `∂loss/∂x` where `x` is the backbone output.

---

## What to Implement

### In `algorithm/ecg/feddualatt.py`

Add gradient cosine similarity logging inside `FedDualAttSerialClientTrainer.local_process()`.

Conceptual structure (not exact code — adapt to actual trainer interface):

```python
# After loss.backward(), before optimizer.step()

# 1. Extract backbone output gradient
# The backbone output is the input to transformer block 0
# Register a hook during forward to cache it, then read grad after backward

backbone_grad = self._get_backbone_output_grad()  # see hook below
if backbone_grad is not None:
    # Flatten to 1D vector per sample, then average across batch
    grad_vec = backbone_grad.detach().flatten()
    self._grad_buffer[client_id] = grad_vec  # store for cross-client comparison

# In server aggregation step, after all clients have run:
# 2. Compute pairwise cosine similarities
if len(self._grad_buffer) == num_clients:
    sims = []
    client_ids = sorted(self._grad_buffer.keys())
    for i in range(len(client_ids)):
        for j in range(i+1, len(client_ids)):
            g_i = self._grad_buffer[client_ids[i]]
            g_j = self._grad_buffer[client_ids[j]]
            cos_sim = F.cosine_similarity(g_i.unsqueeze(0), g_j.unsqueeze(0)).item()
            sims.append(cos_sim)
    mean_cos_sim = sum(sims) / len(sims)

    # 3. Log to client logger
    self.logger.info(f"Round {round_id} | backbone_grad_cosine_sim: {mean_cos_sim:.4f}")
    self._grad_buffer.clear()
```

### Hook Registration

To capture `∂loss/∂x` at the backbone output without modifying `forward()`:

```python
# Register in __init__ or at the start of local_process
grad_store = {}
def hook_fn(module, grad_input, grad_output):
    grad_store['backbone_output_grad'] = grad_output[0].detach()

# Attach to the first transformer block's input
handle = model.blocks[0].register_full_backward_hook(hook_fn)
# ... run forward + backward ...
handle.remove()
```

---

## Expected Behavior

### Without Gradient Isolation Fix

- Phase 1 configs with l > 0: mean cosine similarity **< 0** (conflicting gradients)
- Severity increases with l (more local heads → stronger conflicting signal)
- g=8, l=0: cosine similarity ≈ +0.2 to +0.5 (global-only gradients should be consistent
  across clients since they share the same objective)

### With Gradient Isolation Fix

- All configs: mean cosine similarity **≈ 0** or slightly positive
- l=0 case unchanged (no local branch → no change)
- Larger l configs: similarity near-zero (backbone isolated from local gradients)
- This directly demonstrates that the fix eliminates the interference

---

## Literature Grounding

**PCGrad (Yu et al., NeurIPS 2020)** uses gradient cosine similarity as the defining metric
for gradient conflict in multi-task learning. Two task gradients "conflict" if their cosine
similarity is negative. PCGrad surgically modifies conflicting gradients to reduce
interference. Our diagnostic uses the same metric but without modifying the gradients.

**FedLAG (Nguyen et al., arXiv 2024)** computes per-layer gradient cosine similarity across
FL clients to decide which layers to aggregate and which to keep local. Layers with high
cross-client gradient conflict are kept local; layers with consistent gradients are
aggregated. Our diagnostic is the head-level analog of FedLAG's per-layer analysis.

The connection to FedLAG also motivates **change_03** (adaptive head ratio): if we have
per-head cosine similarity, we can use it to dynamically assign heads to global or local.

---

## Paper Use

**Section 3 (Diagnosing the Trade-off):**
- Figure: gradient cosine similarity vs. round number, two panels (without fix / with fix)
- Expected shape without fix: starts negative, oscillates, stays negative as local heads
  create persistent cross-client conflict
- Expected shape with fix: starts near-zero, stays near-zero across all configs regardless
  of l
- This figure transforms the behavioral finding (Phase 1 F1 plots) into a mechanistic
  explanation that reviewers can directly verify

---

## Implementation Cost and Risk

**Cost:** ~20 lines in `feddualatt.py`; runs on existing backward passes
**No training changes:** hook is read-only; no gradient modification
**Risk:** if the hook captures a batched gradient (many samples averaged), the spatial
  structure may wash out cosine similarity. May need to log per-sample or per-batch with
  care about normalization. Verify that `grad_output[0]` from `register_full_backward_hook`
  is the gradient w.r.t. the module's input and not a module-internal gradient.
