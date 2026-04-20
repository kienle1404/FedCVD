#!/usr/bin/env python3
"""
Gradient norm analysis for FedDualAtt.

Loads the model and data, runs a few forward/backward passes, and measures
gradient norms per component (backbone, global attention, local attention)
with and without the detach() fix.

This demonstrates how local branch gradients corrupt the backbone when
detach() is absent, and how detach() eliminates this interference.

Usage:
    python analyze_gradient_norms.py [--num_batches 10] [--seed 42]
    python analyze_gradient_norms.py --show

Requires: the ECG dataset at the standard data path.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# Add project root to path for imports
sys.path.insert(0, str(SCRIPT_DIR / "../../"))
from model.dual_attention_resnet import DualAttentionResNet1D

DOCS_DIR = SCRIPT_DIR / "../../docs"
EXPERIMENT_DIR = DOCS_DIR / "experiments" / "gradient_norms"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Component grouping for gradient analysis
COMPONENT_GROUPS = {
    "backbone": lambda n: "feature_extractor" in n,
    "positional_enc": lambda n: "positional_encoding" in n,
    "global_proj": lambda n: "global_proj" in n,
    "global_att": lambda n: "global_att" in n and "proj" not in n,
    "local_proj": lambda n: "local_proj" in n,
    "local_att": lambda n: "local_att" in n and "proj" not in n,
    "combine": lambda n: "combine" in n,
    "ffn": lambda n: "ffn" in n,
    "norms": lambda n: "norm" in n,
    "fc_head": lambda n: n.startswith("fc."),
}


def get_data_loader(client_idx=0, batch_size=32, data_fraction=0.1):
    """Load a small fraction of one client's training data."""
    from utils.dataloader import get_ecg_dataset
    from torch.utils.data import DataLoader

    clients = ["client1", "client2", "client3", "client4"]
    client = clients[client_idx]

    input_path = str(DATA_PATH) + "/"
    dataset = get_ecg_dataset(
        [f"{input_path}/ECG/preprocessed/{client}/train.csv"],
        base_path=f"{input_path}/ECG/preprocessed/",
        locations=clients,
        file_name="records.h5",
        n_classes=20,
        frac=data_fraction,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def measure_gradient_norms(model, data_loader, device, num_batches=10):
    """
    Run forward/backward passes and collect gradient norms per component.

    Returns: dict[component_name] -> list of per-batch gradient norms
    """
    criterion = nn.BCELoss()
    model.train()

    norms = defaultdict(list)

    for batch_idx, (data, label) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break

        data, label = data.to(device), label.to(device)

        model.zero_grad()
        pred = model(data)
        loss = criterion(pred, label)
        loss.backward()

        # Collect gradient norms per component
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.norm().item()
            # Classify into component groups
            for group_name, match_fn in COMPONENT_GROUPS.items():
                if match_fn(name):
                    norms[group_name].append(grad_norm)
                    break

    # Average per component
    avg_norms = {k: np.mean(v) if v else 0.0 for k, v in norms.items()}
    return avg_norms


def run_comparison(global_heads, local_heads, device, num_batches, data_fraction):
    """
    Compare gradient norms WITH vs WITHOUT detach() for a given config.
    """
    if local_heads == 0:
        print(f"  Skipping {global_heads}G:{local_heads}L (no local branch to test)")
        return None, None

    # Load data from one client
    loader = get_data_loader(client_idx=0, data_fraction=data_fraction)

    # --- WITHOUT detach ---
    model_no_detach = DualAttentionResNet1D(
        num_heads=global_heads + local_heads,
        global_heads=global_heads,
    ).to(device)

    # Temporarily remove detach by monkey-patching the forward method
    # We need to swap out detach() with identity
    for block in model_no_detach.transformer_blocks:
        original_forward = block.forward

        def make_no_detach_forward(blk):
            def forward_no_detach(x):
                # Global branch
                if blk.global_heads > 0:
                    global_in = blk.global_proj_in(x)
                    global_att_out, _ = blk.global_att(global_in, global_in, global_in)
                    global_out = blk.global_proj_out(global_att_out)
                    x_global = blk.norm1(x + global_out)
                else:
                    x_global = None

                # Local branch WITHOUT detach
                if blk.local_heads > 0:
                    local_in = blk.local_proj_in(x)  # no detach!
                    local_att_out, _ = blk.local_att(local_in, local_in, local_in)
                    local_out = blk.local_proj_out(local_att_out)
                    x_local = blk.norm2(x + local_out)  # no detach!
                else:
                    x_local = None

                # Combine
                if x_global is not None and x_local is not None:
                    combined = torch.cat([x_global, x_local], dim=-1)
                    x_combined = blk.combine(combined)
                elif x_global is not None:
                    x_combined = x_global
                else:
                    x_combined = x_local

                ffn_out = blk.ffn(x_combined)
                x_out = blk.norm3(x_combined + ffn_out)
                return x_out
            return forward_no_detach

        block.forward = make_no_detach_forward(block)

    norms_no_detach = measure_gradient_norms(model_no_detach, loader, device, num_batches)

    # --- WITH detach (default behavior) ---
    model_with_detach = DualAttentionResNet1D(
        num_heads=global_heads + local_heads,
        global_heads=global_heads,
    ).to(device)
    # Copy weights to make comparison fair
    model_with_detach.load_state_dict(model_no_detach.state_dict())

    norms_with_detach = measure_gradient_norms(model_with_detach, loader, device, num_batches)

    return norms_no_detach, norms_with_detach


def plot_comparison(all_results, out_path, show):
    """Bar chart comparing gradient norms with and without detach."""
    # Select components to show
    components = ["backbone", "global_proj", "global_att", "local_proj", "local_att",
                  "combine", "ffn", "norms", "fc_head"]

    configs = list(all_results.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 6), sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax, config in zip(axes, configs):
        norms_no, norms_with = all_results[config]
        if norms_no is None:
            ax.set_visible(False)
            continue

        present = [c for c in components if c in norms_no or c in norms_with]
        x = np.arange(len(present))
        width = 0.35

        vals_no = [norms_no.get(c, 0) for c in present]
        vals_with = [norms_with.get(c, 0) for c in present]

        ax.barh(x - width / 2, vals_no, width, label="Without detach()", color="#d62728", alpha=0.8)
        ax.barh(x + width / 2, vals_with, width, label="With detach()", color="#1f77b4", alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(present, fontsize=9)
        ax.set_xlabel("Mean Gradient Norm", fontsize=10)
        ax.set_title(config, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    fig.suptitle("Gradient Norms per Component: With vs Without detach()\n"
                 "(backbone norms should drop significantly with detach)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_backbone_focus(all_results, out_path, show):
    """
    Focused bar chart: backbone gradient norm with vs without detach,
    across multiple configs.
    """
    configs = []
    vals_no = []
    vals_with = []

    for config, (norms_no, norms_with) in all_results.items():
        if norms_no is None:
            continue
        configs.append(config)
        vals_no.append(norms_no.get("backbone", 0))
        vals_with.append(norms_with.get("backbone", 0))

    if not configs:
        return

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, vals_no, width, label="Without detach()", color="#d62728")
    ax.bar(x + width / 2, vals_with, width, label="With detach()", color="#1f77b4")

    # Annotate reduction %
    for i in range(len(configs)):
        if vals_no[i] > 0:
            reduction = (1 - vals_with[i] / vals_no[i]) * 100
            ax.text(i, max(vals_no[i], vals_with[i]) * 1.05,
                    f"-{reduction:.0f}%", ha="center", fontsize=9, color="#2ca02c")

    ax.set_xlabel("Head Ratio Config", fontsize=11)
    ax.set_ylabel("Mean Backbone Gradient Norm", fontsize=11)
    ax.set_title("Backbone Gradient Norm: Local Branch Interference", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gradient norm analysis")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches for gradient measurement")
    parser.add_argument("--data_fraction", type=float, default=0.1,
                        help="Fraction of training data to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test representative configs
    test_configs = [
        (8, 0, "8G:0L"),   # pure global (no local branch — baseline)
        (6, 2, "6G:2L"),
        (4, 4, "4G:4L"),
        (2, 6, "2G:6L"),
        (0, 8, "0G:8L"),   # pure local
    ]

    all_results = {}
    for g, l, label in test_configs:
        print(f"\nMeasuring {label}...")
        norms_no, norms_with = run_comparison(
            g, l, device, args.num_batches, args.data_fraction
        )
        all_results[label] = (norms_no, norms_with)

        if norms_no is not None:
            print(f"  Backbone grad norm: {norms_no.get('backbone', 0):.6f} (no detach) "
                  f"-> {norms_with.get('backbone', 0):.6f} (with detach)")
            bb_no = norms_no.get("backbone", 0)
            bb_with = norms_with.get("backbone", 0)
            if bb_no > 0:
                reduction = (1 - bb_with / bb_no) * 100
                print(f"  Backbone gradient reduction: {reduction:.1f}%")

    # Filter out configs with no data
    valid = {k: v for k, v in all_results.items() if v[0] is not None}

    if valid:
        plot_comparison(valid, FIGURES_DIR / "gradient_norms_comparison.png", args.show)
        plot_backbone_focus(valid, FIGURES_DIR / "gradient_norms_backbone.png", args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
