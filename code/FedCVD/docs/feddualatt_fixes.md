# FedDualAtt Implementation Fixes

**File:** `algorithm/ecg/feddualatt.py`
**Status:** All 4 issues resolved (complete rewrite)

---

## Overview

The original implementation had four correctness and efficiency bugs in the
server–client communication protocol. Training appeared to work because clients
always loaded the correct global + local params before each local training step,
but global uploads were contaminated, server-side evaluation was wrong, and
local params were transmitted with unnecessary serialization overhead.

---

## Issue 1 — Global upload contaminated with local params

**Symptom:** After training, the client serialized the full model (including
`local_att`/`local_proj` weights) as the "global" upload. The server then ran
FedAvg over all clients' local attention weights, averaging them together and
pushing the result back to every client — destroying personalization.

**Root cause:** `model_parameters` (from FedLab's `SerializationTool`) captures
the entire parameter vector. Local positions were never zeroed before calling it.

**Fix:** Before serializing the global upload, the client **zeros all
`local_att` / `local_proj` positions** in-place:

```python
with torch.no_grad():
    for name, param in self._model.named_parameters():
        if _is_local(name):
            param.zero_()
global_updated = self.model_parameters   # local positions = 0
```

On the server side, `global_update()` now runs FedAvg on vectors that have zeros
in local positions. Averaging zeros stays zero, so the **server model invariant**
(`self._model` always has local positions = 0) is preserved automatically.

---

## Issue 2 — Server evaluation used wrong model state

**Symptom:** `local_test()` was inherited from `FedAvgServerHandler`, which
evaluates `self._model` directly. After `global_update()`, `self._model` holds
aggregated global params with **zeroed local positions** — so every client was
evaluated as if it had no local attention heads, producing metrics that did not
reflect personalization.

**Root cause:** The base class `local_test()` has no concept of per-client local
params and cannot load them.

**Fix:** `local_test()` is **overridden** in `FedDualAttServerHandler`. Before
evaluating client k, it loads that client's local params into the model:

```python
def local_test(self):
    for idx, item in enumerate(self.test_loaders):
        if self.local_attention_params[idx]:
            self._model.load_state_dict(self.local_attention_params[idx], strict=False)
        # ... evaluation loop ...
        self._zero_local_params()   # restore invariant after each client
```

`_zero_local_params()` is called after each client to restore the server model
invariant before moving to client k+1.

---

## Issue 3 — Full model deepcopy for local param transmission

**Symptom:** Local params were transmitted by deepcopying the entire model,
serializing it, and then deserializing it on the other side — roughly 20×
overhead for what is actually only 8 small tensors (the local attention weights).

**Root cause:** The original code reused the full-model serialization path for
local params instead of handling them separately.

**Fix:** Local params are passed as a plain **`dict {param_name: tensor}`**
directly. `downlink_package` bundles them alongside the serialized global model:

```python
@property
def downlink_package(self):
    global_serialized = self.model_parameters   # invariant: local = 0
    return [global_serialized] + [deepcopy(p) for p in self.local_attention_params]
```

On the client side, `load_state_dict(..., strict=False)` applies the dict
directly — no serialization round-trip.

---

## Issue 4 — FedAvg aggregated local attention heads

**Symptom:** `global_update()` called `Aggregators.fedavg_aggregate()` on the
full serialized model vector, which included local param positions. This averaged
every client's `local_att` weights into a single vector and wrote it back into
`self._model`, silently destroying all personalization every round.

**Root cause:** `global_update()` was not aware of the global/local param split
and treated the entire model as globally aggregatable.

**Fix:** The uplink payload is now structured as
`[global_params (local=0), local_dict, client_id, n_k]`. `global_update()` unpacks
these explicitly:

```python
def global_update(self, buffer):
    global_params_list = [ele[0] for ele in buffer]
    local_dicts        = [ele[1] for ele in buffer]
    client_ids         = [ele[2] for ele in buffer]
    weights            = [ele[3] for ele in buffer]

    # FedAvg on global params only (zeros in local positions → stays 0)
    global_aggregated = Aggregators.fedavg_aggregate(global_params_list, weights)
    SerializationTool.deserialize_model(self._model, global_aggregated)

    # Store each client's local params directly — no aggregation
    for idx, client_id in enumerate(client_ids):
        self.local_attention_params[client_id] = local_dicts[idx]
```

---

## Summary

| # | What was broken | Root cause | Fix |
|---|---|---|---|
| 1 | Local params leaked into global upload | Full model serialized before zeroing local positions | Zero `local_att`/`local_proj` in-place before `model_parameters` |
| 2 | Server eval used averaged/zeroed local params | Base `local_test()` not overridden | Override `local_test()` to load per-client local dict before each eval |
| 3 | ~20× serialization overhead for local params | Full model deepcopy for 8 small tensors | Pass local params as plain `dict {name: tensor}` |
| 4 | FedAvg averaged local attention heads every round | `global_update()` operated on full model vector | Separate uplink into `[θ_global, φ_k, id, n_k]`; FedAvg only on θ_global |

---

## Server Model Invariant (post-fix)

`self._model` in `FedDualAttServerHandler` **always** holds aggregated global
params with local positions zeroed. This invariant is established in `__init__`
and maintained by:

- `_zero_local_params()` — called after `__init__`, after each client eval in
  `local_test()`, and implicitly preserved by FedAvg on zero vectors.
- The uplink protocol — clients zero local positions before serializing, so
  FedAvg on the received vectors cannot introduce non-zero local values.

Per-client personalized params live exclusively in
`self.local_attention_params: list[dict]` and are never part of the aggregated
model state.
