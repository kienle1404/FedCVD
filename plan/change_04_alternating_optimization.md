---
name: Alternating Optimization
description: SUPERSEDED. detach() achieves gradient isolation with zero overhead and zero convergence penalty. Alternating optimization (FedRep-style) doubles training phases with no benefit over detach(). Not pursuing.
type: project
---

# Change 04: Alternating Optimization

**Status: NOT IMPLEMENTED** — alternative approach to gradient isolation; useful as a
comparison baseline in the future paper
**Cost:** ~30 lines in `algorithm/ecg/feddualatt.py`; no model changes

---

## What It Is

Instead of joint training of all parameters simultaneously (which causes gradient
interference), **alternate** between two training steps within each local epoch:

- **Step A:** Freeze local heads. Train backbone + global heads only.
- **Step B:** Freeze backbone + global heads. Train local heads only.
- Repeat A → B → A → B within each local epoch.

This prevents interference at the **optimizer level**: local heads only receive gradients
when the backbone is frozen, so they cannot affect backbone parameters.

---

## Motivation

The gradient isolation fix (change_01) stops interference at the **computation graph
level** (`detach()`). Alternating optimization achieves the same goal at the **optimizer
level** (parameter freezing). Both target the same root cause.

Alternating optimization is a more conservative approach — it makes no assumptions about
gradient propagation and is unambiguously correct. However, it is more complex to implement
and has convergence implications.

**Primary paper use:** present alternating optimization as a conceptual comparison to
justify why gradient isolation (change_01) is the preferred fix:

> *"The same goal can be achieved by alternating optimization (FedRep-style), but this adds
> convergence complexity and freezing overhead. Gradient isolation achieves the same result
> with a single line change and zero computational overhead."*

---

## How It Works

### Within Each Local Epoch

Current (joint training):
```
for each local epoch:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
```

With alternating optimization:
```
for each local epoch:
    # Step A: backbone + global heads only
    for p in local_params:   p.requires_grad = False
    for p in global_params:  p.requires_grad = True
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer_global.step()

    # Step B: local heads only
    for p in global_params:  p.requires_grad = False
    for p in local_params:   p.requires_grad = True
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer_local.step()

    # Restore all gradients for next iteration
    for p in model.parameters(): p.requires_grad = True
```

### Parameter Groups

Reusing the existing `_is_local()` string-match logic:
- **Global params:** everything except `local_att` and `local_proj` in name
- **Local params:** params with `local_att` or `local_proj` in name

These are already defined in `algorithm/ecg/feddualatt.py` for aggregation decisions — the
same split applies here.

---

## What to Implement

**File:** `algorithm/ecg/feddualatt.py`
**Method:** `FedDualAttSerialClientTrainer.local_process()` or `train()`

Changes needed:

1. **Separate optimizer groups:** split `optimizer.param_groups` into global and local groups,
   or maintain two separate optimizers (`optimizer_global`, `optimizer_local`).

2. **Freeze/unfreeze logic:** `requires_grad` toggling before each step. Use the existing
   `_is_local()` name filter to identify local params.

3. **Two forward+backward passes per step:** Step A and Step B each need a full forward
   pass. This doubles the number of forward passes per local step (2× computation per step).

4. **Gradient zeroing:** after each step, zero only the relevant param gradients to avoid
   accumulation across steps.

---

## Expected Behavior

- **Global F1:** stays flat at ~72.7% across all local head counts — backbone only receives
  gradients from Step A (global-only pass), same as with gradient isolation.
- **Per-client F1:** improves with local heads — Step B trains local heads on clean frozen
  backbone features.
- **Convergence:** may require more rounds to converge (two steps per epoch, and the
  backbone features used in Step B are always one step behind local head updates).
- **Stability:** training should be more stable than joint training for high head counts
  (e.g., N=16, N=32) since backbone and local heads are never optimized simultaneously.

---

## Comparison to Gradient Isolation Fix

| Property | Gradient Isolation (change_01) | Alternating Optimization |
|---|---|---|
| Mechanism | Computation graph wall | Optimizer-level freeze |
| Implementation | 1 line (`x.detach()`) | ~30 lines |
| Forward passes per step | 1 | 2 |
| Convergence | Same as joint training | May need more rounds |
| Gradient interaction | None (strict wall) | None (separate steps) |
| Local head access to backbone | Reads frozen backbone features | Reads frozen backbone features |
| Runtime overhead | Zero | ~2× per-step forward pass cost |
| Conceptual complexity | Low | Moderate |
| Risk of implementation error | Very low | Moderate (gradient zeroing bugs) |

**Conclusion:** gradient isolation achieves the same gradient flow guarantees at 1/30th the
implementation complexity and zero runtime cost. Alternating optimization is mainly useful
as a baseline to validate the claim that the two approaches produce equivalent results.

---

## Literature Grounding

**FedRep (Collins et al., ICML 2021)** is the direct precedent. FedRep explicitly uses
alternating optimization: several steps to optimize the shared representation (with the
local head frozen), then one step to optimize the local head (with the representation
frozen). The intuition is the same: the shared representation should not be steered by
client-specific objectives.

FedRep's alternating structure is more principled than joint training but requires careful
tuning of the step ratio (how many representation steps per head step). In our case,
1:1 alternation is the natural starting point.

**FedPer (Arivazhagan et al., arXiv 2019)** also keeps a local head that is never aggregated
but trains jointly — without alternation. The performance gap between FedPer and FedRep
(where FedRep is consistently better) provides indirect evidence that alternating
optimization benefits the representation quality.

---

## Paper Framing

In the future paper, alternating optimization serves as a **comparison baseline** for
gradient isolation:

- **Experiment:** Run 8-head ratio sweep with alternating optimization. Compare global F1
  and per-client F1 curves to gradient isolation.
- **Expected finding:** similar global F1 preservation, potentially slower convergence.
- **Conclusion:** gradient isolation achieves equivalent results with far lower implementation
  cost and zero computational overhead.

This strengthens the paper's argument that `detach()` is the preferred fix — not just
because it works, but because alternatives exist and are demonstrably more costly.

---

## Risk

**Double forward pass cost:** alternating optimization roughly doubles the computation per
round if implemented naively (two separate forward+backward passes per local step). This
can be mitigated by caching the backbone output in Step A for reuse in Step B — but this
requires storing intermediate activations, which increases memory usage.

**Step ratio tuning:** the 1:1 alternation (one global step per local step) may not be
optimal. More global steps per local step (as in FedRep) may improve backbone quality but
slow local head learning. This introduces a new hyperparameter.
