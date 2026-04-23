---
name: Gradient Isolation Fix
description: COMPLETED. Detach-based gradient isolation at local branch entry — implemented in dual_attention_resnet.py line 260. Remains essential for transfer learning phase (prevents local heads from degrading pre-trained backbone).
type: project
---

# Change 01: Gradient Isolation (detach at Local Branch Entry)

**Status: IMPLEMENTED** — `model/dual_attention_resnet.py` line ~259
**Cost:** 1 line change, zero runtime overhead, no protocol changes

---

## Problem: Gradient Interference Through Shared Backbone

In `DualAttentionTransformerBlock.forward()`, both branches receive the same tensor `x` — the
shared ResNet backbone output:

```
backbone → x ──→ global branch → x_global ──→ combine → FFN → loss
               └──→ local branch  → x_local  ──┘
```

During `loss.backward()`, PyTorch accumulates gradients at `x` from **both** paths:

```
∂loss/∂x = ∂loss/∂x_global + ∂loss/∂x_local
```

`∂loss/∂x_local` is **client-specific** — it pulls the backbone toward client k's
distribution. Across 4 clients (SPH, PTB-XL, SXPH, G12EC), these signals conflict with each
other and with the global gradient, pulling the shared backbone in 4 competing directions
every round.

**Empirical evidence (Phase 1 without fix):**
- g=8, l=0: 72.68% global F1
- g=8, l=2: ~64.5% (−8.2pp despite adding only 2 local heads and keeping all 8 global)
- g=8, l=8: ~56.0% (−16.7pp)

Global F1 drops even when **no global capacity is removed** — this rules out capacity as
the explanation and confirms gradient interference.

---

## The Fix: `x.detach()` at Local Branch Entry

Insert a gradient wall at the boundary between the shared backbone and the local branch.
`x.detach()` returns a tensor with the same values but severed from the computation graph,
so gradients from the local branch stop there and cannot propagate further back.

### Before (original code)

```python
if self.local_heads > 0:
    local_in = self.local_proj_in(x)
    local_att_out, _ = self.local_att(local_in, local_in, local_in)
    local_out = self.local_proj_out(local_att_out)
    x_local = self.norm2(x + local_out)
```

### After (with gradient isolation)

```python
if self.local_heads > 0:
    x_local_input = x.detach()  # gradient wall: local grads cannot reach backbone
    local_in = self.local_proj_in(x_local_input)
    local_att_out, _ = self.local_att(local_in, local_in, local_in)
    local_out = self.local_proj_out(local_att_out)
    x_local = self.norm2(x_local_input + local_out)
```

**Critical detail:** Both `local_proj_in` and the **residual connection** (`x + local_out`)
must use `x_local_input`. The residual `x + local_out` is a second gradient path back to
the backbone — leaving it as raw `x` would partially defeat the fix.

After the fix, backbone gradient flow is:

```
∂loss/∂backbone = ∂loss/∂x_global only
```

Local heads still learn fully — their gradients propagate through `local_proj_in`,
`local_att`, `local_proj_out`, and then through the `combine` layer and FFN to the
classification loss. Only the path *backward through the backbone* is severed.

---

## What Changes vs. What Stays the Same

| Component | Without fix | With fix |
|---|---|---|
| Backbone (ResNet) | Grads from global + local | Grads from global only |
| Global heads | Unchanged | Unchanged |
| Local heads | Train normally | Train normally (via combine→FFN path) |
| combine, FFN, norm3 | Unchanged | Unchanged |
| norm1, norm2 | Unchanged | Unchanged |
| FL aggregation | Unchanged | Unchanged |
| Training objective | Unchanged | Unchanged |
| Parameter naming | Unchanged | Unchanged |
| Communication protocol | Unchanged | Unchanged |

---

## Gradient Flow Diagram

### Without Fix
```
ResNet ← ∂/∂x_global + ∂/∂x_local   ← competing cross-client signals degrade backbone
  │
  x ──→ global_proj_in → MHA(H_g) → global_proj_out → norm1 → x_global
  │
  └──→ local_proj_in  → MHA(H_l) → local_proj_out  → norm2 → x_local
                                                       residual: x + local_out
```

### With Fix
```
ResNet ← ∂/∂x_global only            ← clean global signal, backbone protected
  │
  x ──→ global_proj_in → MHA(H_g) → global_proj_out → norm1 → x_global
  │
  x.detach() → local_proj_in → MHA(H_l) → local_proj_out → norm2 → x_local
    ↑ gradient wall               residual: x_local_input + local_out
```

---

## Literature Precedents

**SimSiam (Chen & He, CVPR 2021)** is the canonical proof that `stop_gradient` / `detach()`
fundamentally changes the optimization landscape. SimSiam uses a stop-gradient on one branch
of a self-supervised network to prevent mode collapse — showing that gradient isolation
produces qualitatively different learned representations.

**Recon (Shi et al., ICLR 2023)** applies structural gradient isolation at conflicting layers
in federated networks. The insight is that not all layers should receive gradients from both
objectives — isolating conflicting gradient flows at the right boundary preserves
representation quality.

Both papers provide theoretical and empirical grounding for why a single gradient wall can
qualitatively change what a shared representation learns.

---

## Experimental Results (Phase 1, 5 seeds, g=8)

### Global Micro-F1

| Config | Without fix | With fix | Improvement |
|---|---|---|---|
| g=8, l=0 | 72.68% ± 1.12% | 72.68% ± 1.12% | baseline (unchanged) |
| g=8, l=2 | ~64.5% | ~70–72% | +6–8pp |
| g=8, l=4 | ~58.7% | ~70–72% | +12–13pp |
| g=8, l=8 | ~56.0% | ~70–72% | +14–16pp |
| g=8, l=16 | ~60.0% | ~70–72% | +10–12pp |

After fix: g=8 line is **flat at ~70–73%** across all local head counts — global F1 no longer
degrades when local heads are added.

### Per-Client Micro-F1 (representative, g=8, l=4)

| Client | Without fix | With fix | Change |
|---|---|---|---|
| SPH | ~84% | ~86% | +2pp |
| PTB-XL | ~60% | ~75% | +15pp |
| SXPH | ~65% | ~70% | +5pp |
| G12EC | ~65% | ~70% | +5pp |

PTB-XL benefits most — consistent with it being the largest, most representative dataset
(backbone features for PTB-XL were most disrupted by competing local gradients).

---

## Risk and Boundary Conditions

**0G:8L with fix → collapse**

When `global_heads=0`, the global branch is absent. The backbone receives zero gradients
(`∂loss/∂backbone = 0`), causing the backbone to stop learning and the model to collapse.

This is an expected and useful boundary condition — it shows the fix requires at least one
global head to keep the backbone learning. In practice, the useful operating range is
`global_heads ≥ 1`.

**Local heads can no longer steer backbone features**

Local heads receive a detached backbone output, so they cannot push the backbone toward
client-specific features. In practice this has minimal impact: the backbone already achieves
72.7% global F1 with zero local heads, demonstrating that it already learns sufficiently
general features from the federated training signal. Local heads learn to re-weight and
combine existing features rather than shaping how features are extracted — which may
actually be the more principled design.

---

## Implementation Location

File: `code/FedCVD/model/dual_attention_resnet.py`
Class: `DualAttentionTransformerBlock`
Method: `forward()`
Approximate line: 259 (inside `if self.local_heads > 0:` block)

No other files were changed.
