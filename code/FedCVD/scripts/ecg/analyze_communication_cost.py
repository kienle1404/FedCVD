#!/usr/bin/env python3
"""
Communication cost analysis for FedDualAtt.

Computes parameter counts for each head ratio config, split into:
- Global (transmitted each round): ResNet backbone, global attention, combine, FFN, norms, FC
- Local (never transmitted): local attention heads + local projections

Outputs a table and bar chart showing communication savings vs FedAvg (full model).

Usage:
    python analyze_communication_cost.py [--total_heads 8] [--show]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR / "../../docs"
EXPERIMENT_DIR = DOCS_DIR / "experiments" / "communication_cost"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Architecture constants
D_MODEL = 512
HEAD_DIM = 64
FF_DIM = 2048
NUM_BLOCKS = 2
NUM_CLASSES = 20
INPUT_CHANNELS = 12


def resnet1d34_params():
    """Count ResNet1D-34 feature extractor parameters."""
    # conv1: Conv1d(12, 64, kernel_size=16, stride=2, padding=7)
    params = 12 * 64 * 16  # no bias
    # bn1: BatchNorm1d(64)
    params += 64 * 2  # weight + bias

    # ResNet-34 layers: [3, 4, 6, 3] blocks
    # Each BasicBlock has 2 conv layers (kernel=7) + 2 BN layers
    channels = [64, 128, 256, 512]
    block_counts = [3, 4, 6, 3]

    for i, (ch_out, n_blocks) in enumerate(zip(channels, block_counts)):
        ch_in = channels[i - 1] if i > 0 else 64
        for b in range(n_blocks):
            c_in = ch_in if b == 0 else ch_out
            stride = 2 if (b == 0 and i > 0) else 1

            # conv1: Conv1d(c_in, ch_out, 7, stride, padding=3, bias=False)
            params += c_in * ch_out * 7
            # bn1: BatchNorm1d(ch_out)
            params += ch_out * 2
            # conv2: Conv1d(ch_out, ch_out, 7, 1, padding=3, bias=False)
            params += ch_out * ch_out * 7
            # bn2: BatchNorm1d(ch_out)
            params += ch_out * 2

            # Downsample: if stride != 1 or c_in != ch_out
            if stride != 1 or c_in != ch_out:
                # conv1x1: Conv1d(c_in, ch_out, 1, stride, bias=False)
                params += c_in * ch_out * 1
                # bn: BatchNorm1d(ch_out)
                params += ch_out * 2

    return params


def attention_branch_params(n_heads):
    """Count params for one attention branch (proj_in + MHA + proj_out)."""
    if n_heads == 0:
        return 0
    att_dim = n_heads * HEAD_DIM

    # proj_in: Linear(D_MODEL, att_dim) — weight + bias
    params = D_MODEL * att_dim + att_dim
    # MHA: in_proj (3 * att_dim * att_dim + 3 * att_dim) + out_proj (att_dim * att_dim + att_dim)
    params += 3 * att_dim * att_dim + 3 * att_dim  # in_proj weight + bias
    params += att_dim * att_dim + att_dim            # out_proj weight + bias
    # proj_out: Linear(att_dim, D_MODEL) — weight + bias
    params += att_dim * D_MODEL + D_MODEL

    return params


def combine_params(has_both_branches):
    """Combine layer: Linear(2*D_MODEL, D_MODEL) if both branches exist."""
    if not has_both_branches:
        return 0
    return 2 * D_MODEL * D_MODEL + D_MODEL  # weight + bias


def ffn_params():
    """FFN: Linear(512, 2048) + Linear(2048, 512)."""
    return (D_MODEL * FF_DIM + FF_DIM) + (FF_DIM * D_MODEL + D_MODEL)


def norm_params():
    """3 LayerNorms per block, each LayerNorm(512) has weight + bias."""
    return 3 * (D_MODEL * 2)


def fc_params():
    """Classification head: Linear(512, 20)."""
    return D_MODEL * NUM_CLASSES + NUM_CLASSES


def positional_encoding_params():
    """PositionalEncoding has no learnable parameters (uses sin/cos)."""
    return 0


def count_params(global_heads, local_heads):
    """
    Count total, global (transmitted), and local (kept) parameters.

    Returns: (total, global_params, local_params)
    """
    has_both = global_heads > 0 and local_heads > 0

    # Components that are always GLOBAL
    resnet = resnet1d34_params()
    pos_enc = positional_encoding_params()
    fc = fc_params()

    # Per-block components
    global_att_per_block = attention_branch_params(global_heads)
    local_att_per_block = attention_branch_params(local_heads)
    comb_per_block = combine_params(has_both)
    ffn_per_block = ffn_params()
    norm_per_block = norm_params()

    # Total across blocks
    global_per_round = (
        resnet + pos_enc + fc +
        NUM_BLOCKS * (global_att_per_block + comb_per_block + ffn_per_block + norm_per_block)
    )
    local_total = NUM_BLOCKS * local_att_per_block

    total = global_per_round + local_total

    return total, global_per_round, local_total


def analyze(total_heads_list):
    """Run analysis for given total head counts."""
    results = []

    for N in total_heads_list:
        if N <= 4:
            ratios = [(g, N - g) for g in range(N + 1)]
        else:
            # Standard 5 ratio points
            pcts = [0, 25, 50, 75, 100]
            ratios = [(round(N * p / 100), N - round(N * p / 100)) for p in pcts]

        for g, l in ratios:
            total, glob, loc = count_params(g, l)
            results.append({
                "total_heads": N,
                "global_heads": g,
                "local_heads": l,
                "pct_global": g / N * 100 if N > 0 else 0,
                "total_params": total,
                "transmitted_params": glob,
                "local_params": loc,
                "comm_ratio": glob / total,
            })

    return results


def print_table(results):
    """Print a formatted table."""
    # FedAvg baseline: full model = total params of 8G:0L
    baseline_total = next(r["total_params"] for r in results
                         if r["global_heads"] == 8 and r["local_heads"] == 0
                         and r["total_heads"] == 8)

    print(f"\n{'=' * 90}")
    print(f"{'Config':>10s} | {'Total':>12s} | {'Transmitted':>12s} | {'Local':>12s} | {'% Transmitted':>14s} | {'vs FedAvg':>10s}")
    print(f"{'-' * 90}")

    current_n = None
    for r in results:
        if r["total_heads"] != current_n:
            current_n = r["total_heads"]
            print(f"  N={current_n}")

        config = f"{r['global_heads']}G:{r['local_heads']}L"
        vs_fedavg = r["transmitted_params"] / baseline_total
        print(f"  {config:>8s} | {r['total_params']:>12,d} | {r['transmitted_params']:>12,d} | "
              f"{r['local_params']:>12,d} | {r['comm_ratio']*100:>13.1f}% | {vs_fedavg:>9.3f}x")

    print(f"{'=' * 90}")
    print(f"\nFedAvg baseline (full model, 8G:0L): {baseline_total:,d} params")


def plot_comm_cost(results, out_path, show):
    """Bar chart: transmitted vs local params for N=8 configs."""
    n8 = [r for r in results if r["total_heads"] == 8]
    n8.sort(key=lambda r: r["global_heads"], reverse=True)

    labels = [f"{r['global_heads']}G:{r['local_heads']}L" for r in n8]
    transmitted = [r["transmitted_params"] / 1e6 for r in n8]
    local = [r["local_params"] / 1e6 for r in n8]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x, transmitted, width, label="Transmitted (global)", color="#1f77b4")
    bars2 = ax.bar(x, local, width, bottom=transmitted, label="Local (not transmitted)", color="#ff7f0e")

    ax.set_xlabel("Head Ratio Config", fontsize=11)
    ax.set_ylabel("Parameters (millions)", fontsize=11)
    ax.set_title("FedDualAtt: Communication Cost per Round (N=8)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    # Annotate savings
    baseline = n8[0]["total_params"]  # 8G:0L
    for i, r in enumerate(n8):
        saving = (1 - r["transmitted_params"] / baseline) * 100
        if saving > 0.5:
            ax.text(i, (transmitted[i] + local[i]) * 1.01, f"-{saving:.1f}%",
                    ha="center", va="bottom", fontsize=8, color="#2ca02c")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_scaling_comm(results, out_path, show):
    """Line plot: transmitted params vs % global heads for N=1,2,4,8,16,32."""
    head_counts = sorted(set(r["total_heads"] for r in results))
    colors = {1: "#9467bd", 2: "#8c564b", 4: "#e377c2",
              8: "#2ca02c", 16: "#1f77b4", 32: "#d62728"}
    markers = {1: "D", 2: "v", 4: "P", 8: "s", 16: "o", 32: "^"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for n in head_counts:
        subset = sorted([r for r in results if r["total_heads"] == n],
                        key=lambda r: r["pct_global"])
        xs = [r["pct_global"] for r in subset]
        ys = [r["transmitted_params"] / 1e6 for r in subset]
        ax.plot(xs, ys, marker=markers.get(n, "o"), color=colors.get(n, "gray"),
                label=f"N={n}", linewidth=2)

    ax.set_xlabel("% Global Heads", fontsize=11)
    ax.set_ylabel("Transmitted Parameters (millions)", fontsize=11)
    ax.set_title("Communication Cost vs Head Ratio at Different Total Heads", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Communication cost analysis")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    total_heads_list = [1, 2, 4, 8, 16, 32]
    results = analyze(total_heads_list)

    print_table(results)

    # Save CSV
    csv_path = EXPERIMENT_DIR / "communication_cost.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    plot_comm_cost(results, FIGURES_DIR / "comm_cost_n8.png", args.show)
    plot_scaling_comm(results, FIGURES_DIR / "comm_cost_scaling.png", args.show)


if __name__ == "__main__":
    main()
