#!/usr/bin/env python3
"""
Plot head ratio experiment results with error bars for all metrics.

Usage:
    python plot_head_ratio_results.py [--show]
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR.parent.parent / "docs"
CSV_PATH = DOCS_DIR / "head_ratio_all_metrics.csv"
OUTPUT_DIR = DOCS_DIR / "figures"

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors for each client/metric
COLORS = {
    'SPH': '#1f77b4',
    'PTB-XL': '#ff7f0e',
    'SXPH': '#2ca02c',
    'G12EC': '#d62728',
    'Global': '#9467bd',
}

CLIENT_NAMES = ['SPH', 'PTB-XL', 'SXPH', 'G12EC']


def load_data():
    """Load the head ratio results CSV."""
    df = pd.read_csv(CSV_PATH)
    return df


def plot_metric_bars(df, metric_name: str, ylabel: str, title: str, ylim: tuple = None):
    """Plot bar chart with error bars for a specific metric."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(df))
    width = 0.15

    # Build columns based on metric name
    if metric_name == 'micro_f1':
        metrics = [
            (f'SPH_{metric_name}_mean', f'SPH_{metric_name}_std', 'SPH'),
            (f'PTB-XL_{metric_name}_mean', f'PTB-XL_{metric_name}_std', 'PTB-XL'),
            (f'SXPH_{metric_name}_mean', f'SXPH_{metric_name}_std', 'SXPH'),
            (f'G12EC_{metric_name}_mean', f'G12EC_{metric_name}_std', 'G12EC'),
            (f'global_{metric_name}_mean', f'global_{metric_name}_std', 'Global'),
        ]
    else:
        metrics = [
            (f'SPH_{metric_name}_mean', f'SPH_{metric_name}_std', 'SPH'),
            (f'PTB-XL_{metric_name}_mean', f'PTB-XL_{metric_name}_std', 'PTB-XL'),
            (f'SXPH_{metric_name}_mean', f'SXPH_{metric_name}_std', 'SXPH'),
            (f'G12EC_{metric_name}_mean', f'G12EC_{metric_name}_std', 'G12EC'),
            (f'global_{metric_name}_mean', f'global_{metric_name}_std', 'Global'),
        ]

    for i, (mean_col, std_col, label) in enumerate(metrics):
        offset = (i - 2) * width
        ax.bar(
            x + offset,
            df[mean_col],
            width,
            yerr=df[std_col],
            label=label,
            color=COLORS[label],
            capsize=3,
            error_kw={'elinewidth': 1, 'capthick': 1}
        )

    ax.set_xlabel('Head Ratio (Global : Local)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ratio'])
    ax.legend(loc='lower right')
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


def plot_global_only(df, metric_name: str, ylabel: str, title: str, ylim: tuple = None):
    """Plot only Global metric with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    mean_col = f'global_{metric_name}_mean'
    std_col = f'global_{metric_name}_std'

    ax.bar(
        x,
        df[mean_col],
        yerr=df[std_col],
        color=COLORS['Global'],
        capsize=5,
        error_kw={'elinewidth': 2, 'capthick': 2},
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )

    # Add value labels on bars
    for i, (val, std) in enumerate(zip(df[mean_col], df[std_col])):
        ax.annotate(
            f'{val:.1f}',
            xy=(i, val + std + 1),
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Highlight best configuration
    best_idx = df[mean_col].idxmax()
    ax.bar(
        best_idx,
        df.loc[best_idx, mean_col],
        yerr=df.loc[best_idx, std_col],
        color='#2ca02c',
        capsize=5,
        error_kw={'elinewidth': 2, 'capthick': 2},
        alpha=0.9,
        edgecolor='black',
        linewidth=2
    )

    ax.set_xlabel('Head Ratio (Global : Local)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ratio'])
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


def plot_line_with_errorband(df, metric_name: str, ylabel: str, title: str, ylim: tuple = None):
    """Plot line chart with error bands for a specific metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))

    metrics = [
        (f'SPH_{metric_name}_mean', f'SPH_{metric_name}_std', 'SPH'),
        (f'PTB-XL_{metric_name}_mean', f'PTB-XL_{metric_name}_std', 'PTB-XL'),
        (f'SXPH_{metric_name}_mean', f'SXPH_{metric_name}_std', 'SXPH'),
        (f'G12EC_{metric_name}_mean', f'G12EC_{metric_name}_std', 'G12EC'),
        (f'global_{metric_name}_mean', f'global_{metric_name}_std', 'Global'),
    ]

    for mean_col, std_col, label in metrics:
        means = df[mean_col].values
        stds = df[std_col].values
        color = COLORS[label]

        linewidth = 2.5 if label == 'Global' else 1.5
        markersize = 8 if label == 'Global' else 6
        ax.plot(x, means, 'o-', label=label, color=color,
                linewidth=linewidth, markersize=markersize)
        ax.fill_between(x, means - stds, means + stds,
                        alpha=0.2, color=color)

    ax.set_xlabel('Head Ratio (Global : Local)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ratio'])
    ax.legend(loc='lower right')
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


def plot_global_line_with_errorbar(df, metric_name: str, ylabel: str, title: str, ylim: tuple = None):
    """Plot Global metric as line with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    mean_col = f'global_{metric_name}_mean'
    std_col = f'global_{metric_name}_std'
    means = df[mean_col].values
    stds = df[std_col].values

    ax.errorbar(
        x, means, yerr=stds,
        fmt='o-',
        color=COLORS['Global'],
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
        markeredgecolor='black',
        markeredgewidth=1
    )

    # Highlight best
    best_idx = df[mean_col].idxmax()
    ax.plot(best_idx, means[best_idx], 'o',
            color='#2ca02c', markersize=12,
            markeredgecolor='black', markeredgewidth=2,
            zorder=10, label=f'Best: {df.loc[best_idx, "ratio"]}')

    ax.set_xlabel('Head Ratio (Global : Local)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ratio'])
    ax.legend(loc='lower right')
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    return fig


def plot_combined_global_metrics(df):
    """Plot all global metrics (Micro-F1, Accuracy, mAP) in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(df))

    metrics = [
        ('micro_f1', 'Micro-F1 (%)', (55, 78)),
        ('acc', 'Accuracy (%)', (38, 68)),
        ('mAP', 'mAP (%)', (48, 60)),
    ]

    for ax, (metric_name, ylabel, ylim) in zip(axes, metrics):
        mean_col = f'global_{metric_name}_mean'
        std_col = f'global_{metric_name}_std'
        means = df[mean_col].values
        stds = df[std_col].values

        ax.errorbar(
            x, means, yerr=stds,
            fmt='o-',
            color=COLORS['Global'],
            linewidth=2,
            markersize=8,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            markeredgecolor='black',
            markeredgewidth=1
        )

        # Highlight best
        best_idx = df[mean_col].idxmax()
        ax.plot(best_idx, means[best_idx], 'o',
                color='#2ca02c', markersize=12,
                markeredgecolor='black', markeredgewidth=2,
                zorder=10)

        ax.set_xlabel('Head Ratio')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Global {ylabel.split()[0]}')
        ax.set_xticks(x)
        ax.set_xticklabels(df['ratio'], rotation=45, ha='right')
        ax.set_ylim(ylim)

    plt.suptitle('FedDualAtt: Global Metrics vs Head Ratio (n=5 seeds)', y=1.02)
    plt.tight_layout()
    return fig


def plot_personalization_benefit(df, metric_name: str, ylabel: str, title: str):
    """
    Line graph showing per-client delta vs the 8G:0L (no-personalization) baseline.

    For each head ratio configuration, compute for each of the 4 clients:
        delta_i = mean_metric(ratio, client_i) - mean_metric(8G:0L, client_i)

    Then aggregate across clients:
        high = max(delta_i)   — best-benefiting client
        mean = mean(delta_i)  — average benefit
        low  = min(delta_i)   — worst-affected client

    A shaded band between high and low shows the spread.
    """
    clients = ['SPH', 'PTB-XL', 'SXPH', 'G12EC']

    # Locate the 8G:0L baseline row
    baseline_mask = df['ratio'] == '8-0'
    if not baseline_mask.any():
        raise ValueError("Could not find ratio '8-0' (8G:0L) in dataframe.")
    baseline = df[baseline_mask].iloc[0]

    # Sort descending by global heads so x-axis reads 8G:0L → 0G:8L
    # (increasing personalization left to right)
    df = df.copy()
    df['_g'] = df['ratio'].apply(lambda r: int(r.split('-')[0]))
    df = df.sort_values('_g', ascending=False).reset_index(drop=True)

    # Format x-axis labels: "8-0" → "8G:0L"
    x_labels = [f"{r.split('-')[0]}G:{r.split('-')[1]}L" for r in df['ratio']]
    x = np.arange(len(df))

    highs, means, lows = [], [], []
    for _, row in df.iterrows():
        deltas = [
            row[f'{c}_{metric_name}_mean'] - baseline[f'{c}_{metric_name}_mean']
            for c in clients
        ]
        highs.append(max(deltas))
        means.append(float(np.mean(deltas)))
        lows.append(min(deltas))

    highs = np.array(highs)
    means = np.array(means)
    lows  = np.array(lows)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference line at zero (= 8G:0L performance)
    ax.axhline(0, color='#555555', linewidth=1.2, linestyle='--', alpha=0.7,
               label='8G:0L baseline (Δ = 0)')

    # Shaded band between high and low
    ax.fill_between(x, lows, highs, alpha=0.12, color='#1f77b4')

    # Three lines
    ax.plot(x, highs, 'o-', color='#2ca02c', linewidth=2,   markersize=7,
            label='High (best client)')
    ax.plot(x, means, 's-', color='#1f77b4', linewidth=2.5, markersize=8,
            label='Mean (avg. clients)', zorder=5)
    ax.plot(x, lows,  '^-', color='#d62728', linewidth=2,   markersize=7,
            label='Low (worst client)')

    # Always anchor y-axis so zero baseline is visible; extend into negative
    # territory when any client's delta is negative (worse than 8G:0L).
    data_min = float(lows.min())
    data_max = float(highs.max())
    margin = max((data_max - data_min) * 0.12, 0.5)
    ax.set_ylim(min(data_min, 0) - margin, max(data_max, 0) + margin)

    ax.set_xlabel('Head Ratio (Global : Local)', fontsize=13)
    ax.set_ylabel(f'Δ {ylabel} vs 8G:0L', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot head ratio results with error bars")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()
    print(f"Loaded data from: {CSV_PATH}")
    print(f"\nData columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    # Generate plots
    print(f"\nGenerating plots...")

    # === MICRO-F1 PLOTS ===
    fig1 = plot_metric_bars(df, 'micro_f1', 'Micro-F1 (%)',
                            'FedDualAtt: Micro-F1 vs Head Ratio', ylim=(30, 90))
    fig1.savefig(OUTPUT_DIR / 'head_ratio_micro_f1_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_micro_f1_bar.png")

    fig2 = plot_global_only(df, 'micro_f1', 'Global Micro-F1 (%)',
                            'FedDualAtt: Global Micro-F1 vs Head Ratio (n=5 seeds)', ylim=(55, 78))
    fig2.savefig(OUTPUT_DIR / 'head_ratio_micro_f1_global_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_micro_f1_global_bar.png")

    fig3 = plot_line_with_errorband(df, 'micro_f1', 'Micro-F1 (%)',
                                    'FedDualAtt: Micro-F1 vs Head Ratio (±1 std)', ylim=(30, 90))
    fig3.savefig(OUTPUT_DIR / 'head_ratio_micro_f1_line.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_micro_f1_line.png")

    # === ACCURACY PLOTS ===
    fig4 = plot_metric_bars(df, 'acc', 'Accuracy (%)',
                            'FedDualAtt: Accuracy vs Head Ratio', ylim=(5, 80))
    fig4.savefig(OUTPUT_DIR / 'head_ratio_accuracy_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_accuracy_bar.png")

    fig5 = plot_global_only(df, 'acc', 'Global Accuracy (%)',
                            'FedDualAtt: Global Accuracy vs Head Ratio (n=5 seeds)', ylim=(38, 68))
    fig5.savefig(OUTPUT_DIR / 'head_ratio_accuracy_global_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_accuracy_global_bar.png")

    fig6 = plot_line_with_errorband(df, 'acc', 'Accuracy (%)',
                                    'FedDualAtt: Accuracy vs Head Ratio (±1 std)', ylim=(5, 80))
    fig6.savefig(OUTPUT_DIR / 'head_ratio_accuracy_line.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_accuracy_line.png")

    # === mAP PLOTS ===
    fig7 = plot_metric_bars(df, 'mAP', 'mAP (%)',
                            'FedDualAtt: mAP vs Head Ratio', ylim=(30, 65))
    fig7.savefig(OUTPUT_DIR / 'head_ratio_mAP_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_mAP_bar.png")

    fig8 = plot_global_only(df, 'mAP', 'Global mAP (%)',
                            'FedDualAtt: Global mAP vs Head Ratio (n=5 seeds)', ylim=(48, 60))
    fig8.savefig(OUTPUT_DIR / 'head_ratio_mAP_global_bar.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_mAP_global_bar.png")

    fig9 = plot_line_with_errorband(df, 'mAP', 'mAP (%)',
                                    'FedDualAtt: mAP vs Head Ratio (±1 std)', ylim=(30, 65))
    fig9.savefig(OUTPUT_DIR / 'head_ratio_mAP_line.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_mAP_line.png")

    # === COMBINED GLOBAL PLOT ===
    fig10 = plot_combined_global_metrics(df)
    fig10.savefig(OUTPUT_DIR / 'head_ratio_global_combined.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_global_combined.png")

    # === PERSONALIZATION BENEFIT PLOTS ===
    for metric_name, ylabel in [
        ('micro_f1', 'Micro-F1 (%)'),
        ('acc',      'Accuracy (%)'),
        ('mAP',      'mAP (%)'),
    ]:
        fig = plot_personalization_benefit(
            df, metric_name, ylabel,
            f'Personalization Benefit vs 8G:0L — {ylabel.split()[0]}',
        )
        fname = f'head_ratio_personalization_{metric_name}.png'
        fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
