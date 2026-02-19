# FedDualAtt: Detailed Mechanism Report

> **Files referenced**
> - `model/dual_attention_resnet.py` — model architecture
> - `algorithm/ecg/feddualatt.py` — FL protocol
> - `algorithm/ecg/fedavg.py` — base server/client classes
> - `utils/evaluation.py` — metrics

---

## 1. Overview

FedDualAtt is a personalized federated learning algorithm for multi-label ECG classification. Its core idea is to split transformer attention heads within each block into two disjoint branches:

- **Global heads** — parameters are aggregated via FedAvg each round (shared knowledge)
- **Local heads** — parameters are stored per-client on the server and never aggregated (personal adaptation)

Everything else in the model (ResNet backbone, positional encoding, FFN, LayerNorm, classification head) is treated as global.

---

## 2. Model Architecture: `DualAttentionResNet1D`

**File:** [`model/dual_attention_resnet.py`](../model/dual_attention_resnet.py)

### 2.1 Top-level forward pass

```
Input: (B, 12, 5000)   — batch of 12-lead ECG signals at 500 Hz, 10 s
  → ResNet1D-34 feature extractor
  → (B, 512, ~156)      — 512-channel feature map
  → transpose to (B, ~156, 512)
  → PositionalEncoding
  → DualAttentionTransformerBlock × 2
  → transpose back to (B, 512, ~156)
  → AdaptiveAvgPool1d(1) → (B, 512)
  → Linear(512, 20) + Sigmoid
Output: (B, 20)         — per-class probabilities for 20 ECG conditions
```

**Code:** [`dual_attention_resnet.py:358–391`](../model/dual_attention_resnet.py#L358)
```python
features = self.feature_extractor(x)    # (B, 512, ~156)
features = features.transpose(1, 2)     # (B, ~156, 512)
features = self.positional_encoding(features)
for block in self.transformer_blocks:
    features = block(features)
features = features.transpose(1, 2)     # (B, 512, ~156)
pooled   = self.global_pool(features).squeeze(-1)  # (B, 512)
logits   = self.fc(pooled)              # (B, 20)
output   = self.act(logits)             # Sigmoid → (B, 20)
```

### 2.2 ResNet1D-34 Feature Extractor

**Class:** `ResNet1DFeatureExtractor` — [`dual_attention_resnet.py:95–155`](../model/dual_attention_resnet.py#L95)

Standard ResNet-34 adapted for 1D temporal signals:

| Layer | Config | Output shape |
|-------|--------|-------------|
| conv1 | Conv1d(12→64, k=15, s=2, p=7) + BN + ReLU | (B, 64, 2500) |
| maxpool | MaxPool1d(k=3, s=2, p=1) | (B, 64, 1250) |
| layer1 | 3 × BasicBlock(64→64) | (B, 64, 1250) |
| layer2 | 4 × BasicBlock(64→128, s=2) | (B, 128, 625) |
| layer3 | 6 × BasicBlock(128→256, s=2) | (B, 256, 313) |
| layer4 | 3 × BasicBlock(256→512, s=2) | (B, 512, ~157) |

`BasicBlock` uses two Conv1d(k=7) with BN + ReLU and a skip connection. The `conv3x` function ([`dual_attention_resnet.py:27–29`](../model/dual_attention_resnet.py#L27)) uses kernel size 7 (not 3) for ECG's longer temporal dependencies:
```python
def conv3x(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
```

### 2.3 Positional Encoding

**Class:** `PositionalEncoding` — [`dual_attention_resnet.py:73–92`](../model/dual_attention_resnet.py#L73)

Standard sinusoidal positional encoding added to the (B, ~157, 512) feature sequence before the transformer blocks. The buffer is fixed (not trained):
```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

### 2.4 DualAttentionTransformerBlock — The Core Innovation

**Class:** `DualAttentionTransformerBlock` — [`dual_attention_resnet.py:158–283`](../model/dual_attention_resnet.py#L158)

Each block contains **two independent attention branches** operating on the same input `x`:

#### Global branch (aggregated via FedAvg)
```python
# dual_attention_resnet.py:250–256
global_in       = self.global_proj_in(x)             # Linear(512 → G*64)
global_att_out,_= self.global_att(global_in, ...)    # MHA(G heads, dim=G*64)
global_out      = self.global_proj_out(global_att_out)  # Linear(G*64 → 512)
x_global        = self.norm1(x + global_out)         # residual + LayerNorm
```

#### Local branch (stored per-client, never aggregated)
```python
# dual_attention_resnet.py:258–265
local_in        = self.local_proj_in(x)              # Linear(512 → L*64)
local_att_out,_ = self.local_att(local_in, ...)      # MHA(L heads, dim=L*64)
local_out       = self.local_proj_out(local_att_out) # Linear(L*64 → 512)
x_local         = self.norm2(x + local_out)          # residual + LayerNorm
```

**Key dimension**: each attention head always has `head_dim=64`, so `G` global heads use a `G×64`-dimensional attention space and `L` local heads use `L×64`. This allows any (G, L) split while keeping head capacity constant.

#### Branch fusion
```python
# dual_attention_resnet.py:268–277
if x_global is not None and x_local is not None:
    combined  = torch.cat([x_global, x_local], dim=-1)  # (B, T, 1024)
    x_combined = self.combine(combined)                  # Linear(1024→512)
elif x_global is not None:
    x_combined = x_global   # global-only (8G:0L edge case)
else:
    x_combined = x_local    # local-only  (0G:8L edge case)
```

#### Shared FFN + LayerNorm (always global)
```python
# dual_attention_resnet.py:279–281
ffn_out = self.ffn(x_combined)           # Linear(512→2048)→ReLU→Dropout→Linear(2048→512)→Dropout
x_out   = self.norm3(x_combined + ffn_out)
```

#### Layer name summary per block
| Sub-module | Parameter names | Partition |
|-----------|----------------|-----------|
| `global_proj_in` | `transformer_blocks.N.global_proj_in.{weight,bias}` | **GLOBAL** |
| `global_att` | `transformer_blocks.N.global_att.{in_proj,out_proj}.*` | **GLOBAL** |
| `global_proj_out` | `transformer_blocks.N.global_proj_out.{weight,bias}` | **GLOBAL** |
| `local_proj_in` | `transformer_blocks.N.local_proj_in.{weight,bias}` | **LOCAL** |
| `local_att` | `transformer_blocks.N.local_att.{in_proj,out_proj}.*` | **LOCAL** |
| `local_proj_out` | `transformer_blocks.N.local_proj_out.{weight,bias}` | **LOCAL** |
| `combine` | `transformer_blocks.N.combine.{weight,bias}` | **GLOBAL** |
| `ffn` | `transformer_blocks.N.ffn.*` | **GLOBAL** |
| `norm1/2/3` | `transformer_blocks.N.norm{1,2,3}.*` | **GLOBAL** |

### 2.5 Model Constructor

**File:** [`dual_attention_resnet.py:312–356`](../model/dual_attention_resnet.py#L312)

```python
DualAttentionResNet1D(
    input_channels=12,
    d_model=512,
    num_transformer_blocks=2,   # two stacked DualAttentionTransformerBlocks
    num_heads=8,                 # total heads per block
    global_heads=None,           # None → num_heads//2 = 4 (default 4G:4L)
    ff_dim=2048,
    dropout=0.1,
    num_classes=20,
    task='multilabel'
)
```

The `global_heads` argument controls the head-ratio ablation. `local_heads = num_heads - global_heads` is derived automatically.

---

## 3. Parameter Partition

**File:** [`algorithm/ecg/feddualatt.py:46–52`](../algorithm/ecg/feddualatt.py#L46)

```python
_LOCAL_PATTERNS = ('local_att', 'local_proj')

def _is_local(name: str) -> bool:
    return any(p in name for p in _LOCAL_PATTERNS)
```

Any parameter whose name contains `'local_att'` or `'local_proj'` is **local** (personalized). All others are **global** (aggregated). This catches:
- `local_proj_in.weight`, `local_proj_in.bias`
- `local_att.in_proj_weight`, `local_att.in_proj_bias`, `local_att.out_proj.weight`, `local_att.out_proj.bias`
- `local_proj_out.weight`, `local_proj_out.bias`

for both transformer blocks (indices 0 and 1).

**Parameter count** (default 5G:3L, `num_heads=8`, `d_model=512`, `head_dim=64`):
- Global attention (5 heads): `5×64=320`-dim space per block
- Local attention (3 heads): `3×64=192`-dim space per block
- Local params per block ≈ 3 × `(512×192 + 192 + 192×4×192 + 192 + 192 + 192×512 + 512)` ≈ ~1.1M
- Total local params ≈ ~2.2M across 2 blocks
- Total model params ≈ 22–25M (dominated by ResNet backbone)

---

## 4. Server State and Invariant

**Class:** `FedDualAttServerHandler` — [`algorithm/ecg/feddualatt.py:59–249`](../algorithm/ecg/feddualatt.py#L59)

The server maintains two disjoint stores:

| Store | Type | Content |
|-------|------|---------|
| `self._model` | `nn.Module` | Aggregated global params; **local positions always = 0** |
| `self.local_attention_params` | `list[dict]` | Per-client `{param_name: tensor}`, indexed by client id |

**Invariant**: local positions in `self._model` are always zero. This is established at construction:
```python
# feddualatt.py:87–92
initial_local = {n: p.data.clone()
                 for n, p in model.named_parameters() if _is_local(n)}
self.local_attention_params = [deepcopy(initial_local) for _ in range(num_clients)]
self._zero_local_params()   # enforce invariant immediately
```

**`_zero_local_params()`** — [`feddualatt.py:98–103`](../algorithm/ecg/feddualatt.py#L98):
```python
def _zero_local_params(self):
    with torch.no_grad():
        for name, param in self._model.named_parameters():
            if _is_local(name):
                param.zero_()
```

This means `self._model` is always a "clean" global model — safe to serialize and broadcast to all clients.

---

## 5. Federated Learning Protocol

### 5.1 Communication Round Overview

```
┌─────────────────────────────────────────────────────────┐
│  Server                                                  │
│  state:  _model (global, local=0)                        │
│           local_attention_params[0..N-1]                 │
└──────────────┬──────────────────────────────────────────┘
               │ DOWNLINK: global_serialized + all local dicts
               ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Client k            │  │  Client j            │
│  (serial sim)        │  │  ...                 │
└──────────────────────┘  └──────────────────────┘
               │ UPLINK: global_params (local=0) + local_dict_k + k + n_k
               ▼
┌─────────────────────────────────────────────────────────┐
│  Server aggregation                                      │
│  FedAvg(global_params) → deserialize into _model         │
│  local_params[k] ← local_dict_k  (no aggregation)       │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Downlink Package

**Method:** `downlink_package` property — [`feddualatt.py:109–123`](../algorithm/ecg/feddualatt.py#L109)

```python
@property
def downlink_package(self):
    global_serialized = self.model_parameters   # invariant: local = 0
    return [global_serialized] + [deepcopy(p) for p in self.local_attention_params]
```

Payload format: `[global_serialized, local_dict_0, local_dict_1, ..., local_dict_{N-1}]`

In a real deployment, client `k` would only receive `[global_serialized, local_dict_k]`. The full list is bundled here because the simulation is serial.

### 5.3 Client Training

**Class:** `FedDualAttSerialClientTrainer` — [`feddualatt.py:256–319`](../algorithm/ecg/feddualatt.py#L256)

**Method:** `local_process` — [`feddualatt.py:269–314`](../algorithm/ecg/feddualatt.py#L269)

Six-step procedure per client per round:

#### Step 1 — Load global model (local positions = 0)
```python
# feddualatt.py:281
self.set_model(global_params)   # deserializes self._model; local positions = 0
```

#### Step 2 — Overwrite local positions with client's own params
```python
# feddualatt.py:284–286
local_dict = payload[idx + 1]   # {param_name: tensor}
if local_dict:
    self._model.load_state_dict(local_dict, strict=False)
```
`strict=False` only updates the named keys in `local_dict`; global params are untouched. After this, the model has the correct global weights + this client's personal local weights.

#### Step 3 — Train jointly (single SGD pass on all params)
```python
# feddualatt.py:289–293
for epoch in range(self.max_epoch):
    pack = self.train(epoch, idx)   # SGD on all params together
    self.local_test(idx, epoch)
    self.global_test(idx, epoch)
```
Both global and local params are updated by the same SGD optimizer and BCELoss ([`fedavg.py:261`](../algorithm/ecg/fedavg.py#L261)). There is no separate loss term for personalization — the entire model is trained end-to-end.

#### Step 4 — Extract updated local params
```python
# feddualatt.py:296–300
local_updated = {
    n: p.data.clone()
    for n, p in self._model.named_parameters()
    if _is_local(n)
}
```

#### Step 5 — Zero local positions → serialize clean global upload
```python
# feddualatt.py:303–307
with torch.no_grad():
    for name, param in self._model.named_parameters():
        if _is_local(name):
            param.zero_()
global_updated = self.model_parameters   # local = 0, safe to aggregate
```

#### Step 6 — Restore local params for eval/save
```python
# feddualatt.py:310–311
if local_updated:
    self._model.load_state_dict(local_updated, strict=False)
```

#### Upload cache entry
```python
# feddualatt.py:314
self.cache.append([global_updated, local_updated, idx, pack[1]])
#                   ^global(local=0)  ^local dict   ^id  ^n_samples
```

### 5.4 Server Aggregation

**Method:** `global_update` — [`feddualatt.py:125–147`](../algorithm/ecg/feddualatt.py#L125)

```python
def global_update(self, buffer):
    global_params_list = [ele[0] for ele in buffer]
    local_dicts        = [ele[1] for ele in buffer]
    client_ids         = [ele[2] for ele in buffer]
    weights            = [ele[3] for ele in buffer]

    # FedAvg on global params
    # Local positions = 0 in all uploads → average to 0 → invariant preserved automatically
    global_aggregated = Aggregators.fedavg_aggregate(global_params_list, weights)
    SerializationTool.deserialize_model(self._model, global_aggregated)

    # Store each client's local params (no aggregation)
    for idx, client_id in enumerate(client_ids):
        self.local_attention_params[client_id] = local_dicts[idx]
```

Why the invariant is self-preserving: since every client zeroes its local positions before uploading, `global_params_list[k][local_position] = 0` for all k. Their FedAvg-weighted average is also 0. No explicit zeroing is needed after aggregation.

---

## 6. Evaluation Protocol

### 6.1 Local Test (Personalized Evaluation)

**Method:** `FedDualAttServerHandler.local_test` — [`feddualatt.py:153–238`](../algorithm/ecg/feddualatt.py#L153)

For each client `k`:
1. Load client k's local params into the model: `self._model.load_state_dict(local_attention_params[k], strict=False)` ([`feddualatt.py:168–169`](../algorithm/ecg/feddualatt.py#L168))
2. Evaluate on client k's test loader
3. Zero local params to restore invariant: `self._zero_local_params()` ([`feddualatt.py:199`](../algorithm/ecg/feddualatt.py#L199))

This means the "local test" score for client k reflects the model that client k actually trains — global backbone + k's local attention.

**Contrast with FedAvg baseline** ([`fedavg.py:52–112`](../algorithm/ecg/fedavg.py#L52)): FedAvg's `local_test` evaluates the same shared global model on each client's test set — no personalization.

### 6.2 Global Test

**Method:** `FedAvgServerHandler.global_test` — [`fedavg.py:114–175`](../algorithm/ecg/fedavg.py#L114) (inherited by FedDualAtt without override)

Concatenates all 4 clients' test sets and evaluates with the current global model (local=0). This measures the shared model's general performance across the federation. Note: for FedDualAtt, this is evaluated with zero local params — it reflects how well the global backbone alone generalizes.

### 6.3 Cross-Evaluation (Post-hoc, via `reevaluate_checkpoints.py`)

**Script:** [`scripts/ecg/reevaluate_checkpoints.py`](../scripts/ecg/reevaluate_checkpoints.py)

For each head-ratio config, after training is complete, a 4×4 cross-evaluation matrix is computed. For source client `k` and target client `j`:

1. Load saved checkpoint (`server/model.pth`): global model + all local dicts
2. Load local dict of source client `k` into the model (`load_state_dict(strict=False)`)
3. Run inference on target client `j`'s test set
4. Record micro-F1 and mAP for cell `(k, j)`
5. Zero local params before moving to next source client

Results saved to `server/cross_eval_corrected.json`:
```json
{
  "round": 50,
  "cross_eval": {
    "50": {
      "0": {"0": {"micro_f1": ..., "average_precision_score": [...], ...}, "1": {...}, ...},
      "1": {...},
      ...
    }
  }
}
```

**Diagonal** (`k=j`): identical to `local_test` scores (client k's personalized model on client k's data).
**Off-diagonal** (`k≠j`): measures cross-site portability — how well client k's specialized model generalizes to client j's ECG distribution.

### 6.4 Metrics

**File:** [`utils/evaluation.py:159–190`](../utils/evaluation.py#L159)

| Metric | Implementation | Notes |
|--------|---------------|-------|
| **Micro-F1** | `sklearn.f1_score(average="micro")` | Primary metric; counts TP/FP/FN across all 20 classes jointly |
| **mAP** | `np.mean(sklearn.average_precision_score(..., average=None))` | Mean over 20 per-class APs |
| **Threshold** | `pred_score >= 0.5` ([`evaluation.py:15`](../utils/evaluation.py#L15)) | Hard binary predictions from sigmoid scores |
| **Accuracy** | `sklearn.accuracy_score` (subset/exact-match) | All 20 labels must match; very strict |

---

## 7. Head-Ratio Ablation

The `global_heads` parameter in `DualAttentionResNet1D` controls the split. Nine configurations are tested:

| Config | G heads | L heads | `global_heads` arg |
|--------|---------|---------|-------------------|
| 8G:0L | 8 | 0 | 8 |
| 7G:1L | 7 | 1 | 7 |
| 6G:2L | 6 | 2 | 6 |
| 5G:3L | 5 | 3 | 5 |
| 4G:4L | 4 | 4 | 4 (default) |
| 3G:5L | 3 | 5 | 3 |
| 2G:6L | 2 | 6 | 2 |
| 1G:7L | 1 | 7 | 1 |
| 0G:8L | 0 | 8 | 0 |

Edge case handling in [`DualAttentionTransformerBlock.__init__`](../model/dual_attention_resnet.py#L187):
- `global_heads=0`: `global_proj_in`, `global_att`, `global_proj_out` are set to `None`; forward skips global branch entirely ([`dual_attention_resnet.py:199–201`](../model/dual_attention_resnet.py#L199))
- `local_heads=0`: same for local branch ([`dual_attention_resnet.py:214–217`](../model/dual_attention_resnet.py#L214))
- `combine` layer is only created when **both** branches exist ([`dual_attention_resnet.py:220–223`](../model/dual_attention_resnet.py#L220))

**8G:0L cross-eval sanity check**: when `local_heads=0`, `_LOCAL_PATTERNS` matches nothing → `local_attention_params[k]` is an empty dict `{}` → `if local_dict:` is `False` → no local params are loaded → all 4 source clients produce the same global model → all 4 rows of the 4×4 cross-eval matrix are identical. This is correct and expected.

---

## 8. Checkpoint Saving

**Server** — [`feddualatt.py:240–249`](../algorithm/ecg/feddualatt.py#L240):
```python
torch.save(
    {
        "global_model":           self._model.state_dict(),  # local = 0
        "local_attention_params": self.local_attention_params,  # list of N dicts
        "round":                  self.round,
    },
    path,
)
```

**Client** — [`feddualatt.py:316–319`](../algorithm/ecg/feddualatt.py#L316):
```python
torch.save(
    {"model": self._model.state_dict()},   # global + client's local params
    self.output_path + f"client{idx+1}/model.pth",
)
```

The server checkpoint contains the complete state needed to reconstruct any client's personalized model: load `global_model` then overlay `local_attention_params[k]` with `load_state_dict(strict=False)`.

---

## 9. Key Design Decisions

### Why zero-out local params before uploading?
Ensures the server only aggregates global knowledge. If local params were included in the upload, they would contaminate the FedAvg aggregate — mixing client A's personalized attention with client B's data distribution. By zeroing them, the upload carries only globally-useful gradient signal.

### Why `strict=False` when loading local params?
The local param dict is a subset of the full model's state dict. `strict=False` allows partial loading — only the keys present in the dict are updated, leaving all other params (global) untouched. This is the mechanism that lets the server overlay client-specific params on top of the shared global backbone.

### Why a single BCELoss (no auxiliary loss)?
The global and local branches interact through the `combine` layer and the shared FFN. A single end-to-end loss propagates gradients to both branches simultaneously. The separation of "global" vs "local" is purely a parameter management convention at the FL layer — not reflected in the training objective.

### Why two transformer blocks?
Two stacked blocks give the model two opportunities to apply the dual-attention mechanism. Each block has its own independent set of local heads. The cross-eval experiments show that local specialization is strong — diagonal F1 jumps ~7–17 pp above the global-only baseline regardless of how many local heads are used (1–8), suggesting the local attention learns to specialize rapidly even with a single head.

### Server invariant benefit
Keeping `self._model` with `local=0` at all times means:
1. `model_parameters` serialization is always a clean global state — no accidental local contamination in broadcasts
2. `global_test()` naturally evaluates the pure global model without needing special handling
3. The invariant is self-maintaining through FedAvg (zeroes average to zero), with no explicit cleanup needed post-aggregation

---

## 10. File Map

```
model/
  dual_attention_resnet.py   — DualAttentionResNet1D, DualAttentionTransformerBlock,
                               ResNet1DFeatureExtractor, PositionalEncoding

algorithm/ecg/
  feddualatt.py              — FedDualAttServerHandler, FedDualAttSerialClientTrainer
  fedavg.py                  — FedAvgServerHandler (base), FedAvgSerialClientTrainer (base)

utils/
  evaluation.py              — calculate_multilabel_metrics, micro_f1, mAP, threshold

scripts/ecg/
  reevaluate_checkpoints.py  — post-hoc cross-evaluation (evaluate_cross, save_cross_eval_metrics)
  extract_head_ratio_metrics.py — local/global metric extraction across head-ratio configs
  extract_benchmark_metrics.py  — benchmark table metric extraction
  extract_cross_eval_table.py   — cross-eval matrix extraction and summary

trainers/
  feddualatt_ecg.py          — argument parsing and experiment entry point
```
