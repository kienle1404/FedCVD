# Transfer Learning Implementation Plan

## Goal
Integrate ECGFounder (pre-trained on 10.7M ECGs) as the backbone in FedDualAtt to break
the dataset bottleneck that causes N=1 head saturation at 72% F1.

---

## Architecture

```
CURRENT:
  Input (12, 5000)
    → ResNet1D-34 [from scratch, 512 channels] → (batch, 512, 157)
    → transpose → PosEnc → DualAtt ×2 → GAP → FC(512,20) → Sigmoid

NEW:
  Input (12, 5000) [z-score normalized]
    → ECGFounder Net1D [pre-trained, frozen, 1024 channels] → (batch, 1024, ~39)
    → transpose → (batch, ~39, 1024)
    → Projection Linear(1024, 512) → (batch, ~39, 512)
    → PosEnc → DualAtt ×2 [global + local, with detach()]
    → transpose → GAP → FC(512,20) → Sigmoid
```

---

## Implementation Steps

### Step 1: Download ECGFounder

```bash
# Clone repo
git clone https://github.com/PKUDigitalHealth/ECGFounder.git

# Download weights from HuggingFace
# https://huggingface.co/PKUDigitalHealth/ECGFounder
# Place in: code/FedCVD/checkpoints/ecgfounder/
```

Files needed from their repo:
- `net1d.py` — Net1D model definition (copy to `model/net1d.py`)

### Step 2: Create ECGFounder backbone wrapper

**New file:** `model/ecgfounder_backbone.py`

```python
import torch
import torch.nn as nn
from model.net1d import Net1D

class ECGFounderBackbone(nn.Module):
    """
    ECGFounder as a feature extractor (pre-GAP).

    Returns temporal feature maps instead of classification output.
    Output: (batch, 1024, T) where T ≈ input_length / 128
    """
    def __init__(self, pretrained_path=None, freeze=True):
        super().__init__()

        # ECGFounder architecture config
        self.model = Net1D(
            in_channels=12,
            base_filters=64,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150  # will be ignored (we skip the dense layer)
        )

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Remove classification head weights
            state_dict = {k: v for k, v in state_dict.items()
                         if not k.startswith('dense.')}
            self.model.load_state_dict(state_dict, strict=False)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through ECGFounder stages only (skip GAP + Dense).

        Args:
            x: (batch, 12, 5000) z-score normalized ECG
        Returns:
            features: (batch, 1024, T) temporal feature map
        """
        out = x
        # Run through all CNN stages but skip GAP and dense
        # (Need to check net1d.py for exact stage access)
        for stage in self.model.stage_list:
            out = stage(out)
        return out  # (batch, 1024, ~39)
```

### Step 3: Add z-score normalization to data pipeline

**Modify:** `utils/dataloader.py`

Add optional z-score normalization in `ECGDataset.__getitem__()`:

```python
def __getitem__(self, idx):
    data, label = self.data[idx], self.label[idx]
    if self.normalize:
        mean = data.mean()
        std = data.std() + 1e-8
        data = (data - mean) / std
    return data, label
```

Or apply in `get_ecg_dataset()` with a `normalize=True` flag.

### Step 4: Make DualAttentionResNet1D backbone-agnostic

**Modify:** `model/dual_attention_resnet.py`

```python
class DualAttentionResNet1D(Module):
    def __init__(self, ..., backbone='resnet1d34', pretrained_path=None,
                 freeze_backbone=False):

        if backbone == 'resnet1d34':
            self.feature_extractor = ResNet1DFeatureExtractor(input_channels)
            self.backbone_proj = None  # output already 512
        elif backbone == 'ecgfounder':
            from model.ecgfounder_backbone import ECGFounderBackbone
            self.feature_extractor = ECGFounderBackbone(
                pretrained_path=pretrained_path,
                freeze=freeze_backbone
            )
            self.backbone_proj = nn.Linear(1024, d_model)  # 1024 → 512

        # ... rest unchanged (PosEnc, DualAtt blocks, FC)

    def forward(self, x):
        features = self.feature_extractor(x)         # (batch, C, T)
        features = features.transpose(1, 2)          # (batch, T, C)
        if self.backbone_proj is not None:
            features = self.backbone_proj(features)   # (batch, T, 512)
        features = self.positional_encoding(features)
        for block in self.transformer_blocks:
            features = block(features)
        features = features.transpose(1, 2)
        pooled = self.global_pool(features).squeeze(-1)
        logits = self.fc(pooled)
        output = self.act(logits)
        return output
```

### Step 5: Update CLI and FL protocol

**Modify:** `trainers/feddualatt_ecg.py`
```python
parser.add_argument("--backbone", type=str, default="resnet1d34",
                    choices=["resnet1d34", "ecgfounder"])
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--freeze_backbone", action="store_true")
parser.add_argument("--normalize", action="store_true",
                    help="Apply z-score normalization (required for ECGFounder)")
```

**Modify:** `algorithm/ecg/feddualatt.py`
- Frozen backbone params should NOT be transmitted in FL (saves communication)
- Add `'feature_extractor'` to local patterns when frozen? Or handle separately:
```python
# In FedDualAttSerialClientTrainer, skip frozen params in upload
if self.freeze_backbone:
    # Don't include backbone in global_params serialization
```

**Modify:** `model/__init__.py`
```python
elif name == "dual_attention_resnet1d":
    return dual_attention_resnet1d(**kwargs)
```
Already passes kwargs through, so no change needed if factory function accepts new args.

### Step 6: Create experiment runner

**New file:** `scripts/ecg/run_transfer_learning.py`

```python
EXPERIMENTS = {
    'baseline_fc': {
        'backbone': 'ecgfounder', 'freeze_backbone': True, 'normalize': True,
        'num_heads': 1, 'global_heads': 1,  # minimal attention, mostly FC
    },
    'dualatt_4g4l': {
        'backbone': 'ecgfounder', 'freeze_backbone': True, 'normalize': True,
        'num_heads': 8, 'global_heads': 4,
    },
    'scaling_n1': {
        'backbone': 'ecgfounder', 'freeze_backbone': True, 'normalize': True,
        'num_heads': 1, 'global_heads': 1,
    },
    'scaling_n4': {
        'backbone': 'ecgfounder', 'freeze_backbone': True, 'normalize': True,
        'num_heads': 4, 'global_heads': 4,
    },
    'scaling_n8': {
        'backbone': 'ecgfounder', 'freeze_backbone': True, 'normalize': True,
        'num_heads': 8, 'global_heads': 8,
    },
}
```

---

## FL Protocol Changes with Frozen Backbone

When backbone is frozen:
- Backbone params have `requires_grad=False`
- Backbone params are **identical** across all clients (never updated)
- No need to transmit backbone params each round → significant communication savings
- Only transmit: projection layer + attention heads + FFN + norms + FC
- ~3M params transmitted vs ~24M currently (87% communication reduction!)

This is actually a **major practical benefit** of transfer learning in FL.

---

## Preprocessing Compatibility

| | Current pipeline | ECGFounder requirement |
|---|---|---|
| Normalization | None (raw float32) | Z-score per signal: (x - mean) / (std + 1e-8) |
| NaN handling | Not needed | np.nan_to_num(x, nan=0) |
| Lead order | client1..4 directory names | I, II, III, aVR, aVL, aVF, V1-V6 |
| Input shape | (12, 5000) | (12, variable) — we use 5000 |

Need to verify FedCVD's lead ordering matches ECGFounder's expected order.

---

## Estimated Timeline

| Step | Effort | Dependencies |
|------|--------|-------------|
| Download + integrate ECGFounder code | 1 day | None |
| Z-score preprocessing | 0.5 day | None |
| Backbone wrapper + pluggable architecture | 1 day | Step 1 |
| CLI args + FL protocol updates | 0.5 day | Step 3 |
| Smoke test | 0.5 day | Steps 1-4 |
| Experiment 1 (baseline) | 2h GPU | Smoke test |
| Experiment 2 (DualAtt) | 2h GPU | Smoke test |
| Experiment 3 (head scaling) | 6h GPU | Smoke test |
| Experiment 4 (ratio sweep) | 18h GPU | Gate: Exp 2 > Exp 1 |
| Analysis + plots | 1 day | All experiments |
