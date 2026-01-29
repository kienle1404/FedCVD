"""
Dual Attention ResNet for Personalized Federated Learning.

This module implements a hybrid ResNet1D34 + Dual Attention Transformer architecture
for ECG classification in federated learning settings. The dual attention mechanism
splits transformer attention into:
- Global attention heads (4 heads): Shared across all clients, aggregated via FedAvg
- Local attention heads (4 heads): Personalized per client, not aggregated

Architecture:
    Input (batch, 12, 5000)
      → ResNet1D Feature Extractor (global)
      → Output (batch, 512, ~156)
      → Positional Encoding (global)
      → Dual Attention Transformer Blocks (mixed: global + local)
      → Global Average Pooling (global)
      → FC Layer + Sigmoid (global)
      → Output (batch, 20)
"""

import math
import torch
import torch.nn as nn
from torch.nn import Module


def conv3x(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)


def conv1x(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(Module):
    """
    Basic residual block for ResNet1D.
    Copied from resnet.py to maintain consistency.
    """
    expansion: int = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x(output_channels, output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            residual = self.downsample(x)

        y += residual
        y = self.relu(y)

        return y


class PositionalEncoding(Module):
    """
    Positional encoding for transformer.
    Copied from dtn.py.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResNet1DFeatureExtractor(Module):
    """
    ResNet1D34 feature extractor (without classification head).
    Extracts 512-dimensional features from 12-lead ECG signals.

    Input: (batch, 12, 5000)
    Output: (batch, 512, ~156)

    This follows the ResNet1D34 architecture from the FedCVD paper baseline.
    """
    def __init__(self, input_channels=12):
        super(ResNet1DFeatureExtractor, self).__init__()
        self.in_channels = 64

        # Initial convolution layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (3, 4, 6, 3 blocks for ResNet34)
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

    def _make_layer(self, block, channels, block_num, stride=1):
        """Create a ResNet layer with multiple blocks"""
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm1d(channels * block.expansion)
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through ResNet feature extractor.

        Args:
            x: Input tensor (batch, 12, 5000)

        Returns:
            features: Output tensor (batch, 512, ~156)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DualAttentionTransformerBlock(Module):
    """
    Transformer block with dual attention mechanism for personalized FL.

    This block splits multi-head attention into:
    - Global attention (4 heads): Shared/aggregated parameters
    - Local attention (4 heads): Personalized/non-aggregated parameters

    The outputs from both attention mechanisms are concatenated and combined.
    All other components (FFN, LayerNorm, combine layer) are shared/global.

    CRITICAL: Layer naming with 'global_att' and 'local_att' prefixes is essential
    for parameter filtering in the federated learning algorithm.
    """
    def __init__(self, d_model=512, num_heads=8, global_heads=4, local_heads=4,
                 ff_dim=2048, dropout=0.1):
        super(DualAttentionTransformerBlock, self).__init__()

        # Global attention (aggregated via FedAvg in FL)
        self.global_att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=global_heads,
            dropout=dropout,
            batch_first=True
        )

        # Local attention (personalized per client in FL)
        self.local_att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=local_heads,
            dropout=dropout,
            batch_first=True
        )

        # Combine layer to project concatenated outputs back to d_model
        self.combine = nn.Linear(d_model * 2, d_model)

        # Feed-forward network (shared/global)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization (shared/global)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass with dual attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            output: Output tensor (batch, seq_len, d_model)
        """
        # Global attention branch
        global_out, _ = self.global_att(x, x, x)
        x_global = self.norm1(x + global_out)

        # Local attention branch
        local_out, _ = self.local_att(x, x, x)
        x_local = self.norm2(x + local_out)

        # Concatenate both branches and combine
        combined = torch.cat([x_global, x_local], dim=-1)  # (batch, seq_len, 2*d_model)
        x_combined = self.combine(combined)  # (batch, seq_len, d_model)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x_combined)
        x_out = self.norm3(x_combined + ffn_out)

        return x_out


class DualAttentionResNet1D(Module):
    """
    Hybrid ResNet1D34 + Dual Attention Transformer for ECG classification.

    This model combines:
    1. ResNet1D34 feature extraction (proven baseline from FedCVD paper)
    2. Dual attention transformer blocks (novel personalization mechanism)
    3. Classification head for multi-label ECG diagnosis

    In federated learning:
    - Global parameters: ResNet, positional encoding, global attention, FFN, FC
    - Local parameters: Local attention heads only

    Architecture flow:
        Input (batch, 12, 5000) ECG signals
          → ResNet1D Feature Extractor
          → (batch, 512, ~156) features
          → Transpose to (batch, ~156, 512)
          → Positional Encoding
          → Dual Attention Transformer Blocks (×2)
          → Transpose to (batch, 512, ~156)
          → Global Average Pooling
          → FC Layer
          → Sigmoid
          → Output (batch, 20) multi-label predictions
    """
    def __init__(self, input_channels=12, d_model=512, num_transformer_blocks=2,
                 num_heads=8, global_heads=None, ff_dim=2048, dropout=0.1,
                 num_classes=20, task='multilabel'):
        super(DualAttentionResNet1D, self).__init__()

        # Handle global_heads configuration
        if global_heads is None:
            global_heads = num_heads // 2  # Default: 50-50 split (backward compatible)
        local_heads = num_heads - global_heads

        # Store head configuration
        self.global_heads = global_heads
        self.local_heads = local_heads
        self.num_heads = num_heads

        # ResNet feature extractor (GLOBAL - aggregated in FL)
        self.feature_extractor = ResNet1DFeatureExtractor(input_channels)

        # Positional encoding (GLOBAL)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=500)

        # Dual attention transformer blocks (MIXED: global + local components)
        self.transformer_blocks = nn.ModuleList([
            DualAttentionTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                global_heads=global_heads,
                local_heads=local_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_transformer_blocks)
        ])

        # Classification head (GLOBAL)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

        # Task-specific activation
        if task == 'multilabel':
            self.act = nn.Sigmoid()
        elif task == 'multiclass':
            self.act = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown task type: {task}")

    def forward(self, x):
        """
        Forward pass through the hybrid architecture.

        Args:
            x: Input tensor (batch, 12, 5000) - 12-lead ECG signals

        Returns:
            output: Output tensor (batch, 20) - multi-label predictions
        """
        # ResNet feature extraction
        features = self.feature_extractor(x)  # (batch, 512, ~156)

        # Transpose for transformer: (batch, seq_len, features)
        features = features.transpose(1, 2)  # (batch, ~156, 512)

        # Add positional encoding
        features = self.positional_encoding(features)

        # Apply dual attention transformer blocks
        for block in self.transformer_blocks:
            features = block(features)

        # Transpose back for pooling
        features = features.transpose(1, 2)  # (batch, 512, ~156)

        # Global average pooling
        pooled = self.global_pool(features).squeeze(-1)  # (batch, 512)

        # Classification
        logits = self.fc(pooled)  # (batch, 20)
        output = self.act(logits)

        return output


def dual_attention_resnet1d(input_channels=12, d_model=512, num_transformer_blocks=2,
                            num_heads=8, global_heads=None, ff_dim=2048, dropout=0.1,
                            num_classes=20, task='multilabel'):
    """
    Factory function to create a DualAttentionResNet1D model.

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        d_model: Transformer embedding dimension (default: 512)
        num_transformer_blocks: Number of dual attention blocks (default: 2)
        num_heads: Total attention heads (default: 8)
        global_heads: Number of global attention heads (default: None = num_heads // 2).
                      Local heads = num_heads - global_heads.
                      Valid configurations: 4-4, 5-3, 6-2, 7-1 (global-local)
        ff_dim: Feed-forward network dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
        num_classes: Number of output classes (default: 20)
        task: Task type - 'multilabel' or 'multiclass' (default: 'multilabel')

    Returns:
        model: DualAttentionResNet1D instance

    Example:
        >>> model = dual_attention_resnet1d()  # Default 4-4 split
        >>> model = dual_attention_resnet1d(global_heads=6)  # 6-2 split
        >>> x = torch.randn(8, 12, 5000)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([8, 20])
    """
    return DualAttentionResNet1D(
        input_channels=input_channels,
        d_model=d_model,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        global_heads=global_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        num_classes=num_classes,
        task=task
    )


if __name__ == "__main__":
    # Test the model
    print("Testing DualAttentionResNet1D...")
    model = dual_attention_resnet1d()

    # Test forward pass
    sample = torch.randn(8, 12, 5000)
    output = model(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output.shape}")

    # Verify parameter naming
    print("\nParameter names (checking for 'global_att' and 'local_att'):")
    global_count = 0
    local_count = 0
    for name, param in model.named_parameters():
        if 'global_att' in name:
            print(f"  [GLOBAL] {name}: {param.shape}")
            global_count += param.numel()
        elif 'local_att' in name:
            print(f"  [LOCAL]  {name}: {param.shape}")
            local_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter summary:")
    print(f"  Global attention params: {global_count:,}")
    print(f"  Local attention params: {local_count:,}")
    print(f"  Other params: {total_params - global_count - local_count:,}")
    print(f"  Total params: {total_params:,}")
