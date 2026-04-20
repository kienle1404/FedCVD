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
EXPERIMENT_DIR = DOCS_DIR / "experiments" / "ratio_sweep"
CSV_PATH = EXPERIMENT_DIR / "metrics.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "figures"

# Style settings — serif/Computer Modern to match LaTeX paper fonts
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',          # Computer Modern for math symbols
    'font.size': 18,
    'axes.labelsize': 19,
    'axes.titlesize': 19,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 17,
    'figure.figsize': (10, 7),
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


def load_data(total_heads: int = 8):
    """Load the head ratio results CSV, filtered to configs where global+local==total_heads.
    Sorted descending by global_heads (8G:0L → 0G:8L) so all plots share the same x-axis direction.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[df['global_heads'] + df['local_heads'] == total_heads]
    df = df.sort_values('global_heads', ascending=False).reset_index(drop=True)
    return df


def plot_metric_bars(df, metric_name: str, ylabel: str, title: str, ylim: tuple = None):
    """Plot bar chart with error bars for a specific metric."""
    fig, ax = plt.subplots(figsize=(12, 8))

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
    fig, ax = plt.subplots(figsize=(10, 7))

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

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
    Line graph showing per-client delta vs the 8Hg:0Hl (no-personalization) baseline.

    For each head ratio configuration, compute for each of the 4 clients:
        delta_i = mean_metric(ratio, client_i) - mean_metric(8Hg:0Hl, client_i)

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
        raise ValueError("Could not find ratio '8-0' (8Hg:0Hl) in dataframe.")
    baseline = df[baseline_mask].iloc[0]

    # Format x-axis labels: "8-0" → "8Hg:0Hl"  (df already sorted 8G:0L → 0G:8L by load_data)
    x_labels = [f"${r.split('-')[0]}H_g$:${r.split('-')[1]}H_l$" for r in df['ratio']]
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

    # Reference line at zero (= 8Hg:0Hl performance)
    ax.axhline(0, color='#555555', linewidth=1.2, linestyle='--', alpha=0.7,
               label=r'$8H_g$:$0H_l$ baseline ($\Delta = 0$)')

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
    ax.set_ylabel(f'$\\Delta$ {ylabel} vs $8H_g$:$0H_l$', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def plot_delta_per_client(df):
    """
    Two-panel figure (Micro-F1 | mAP) showing the delta from the 8G:0L baseline
    for each client individually and for the global metric.

    X-axis runs left-to-right from 8G:0L (Δ=0, fully global) to 0G:8L (fully local),
    so the trade-off curve reads naturally as "adding more local heads".
    """
    # df already sorted 8G:0L → 0G:8L by load_data
    x_labels = [f"${r.split('-')[0]}H_g$:${r.split('-')[1]}H_l$" for r in df['ratio']]
    x = np.arange(len(df))

    baseline = df.iloc[0]  # 8Hg:0Hl row

    clients = ['SPH', 'PTB-XL', 'SXPH', 'G12EC']
    markers = {'SPH': 'o', 'PTB-XL': 's', 'SXPH': '^', 'G12EC': 'D', 'Global': 'o'}
    linestyles = {'SPH': '-', 'PTB-XL': '-', 'SXPH': '-', 'G12EC': '-', 'Global': '--'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

    for ax, metric_name, ylabel in zip(axes,
                                        ['micro_f1', 'mAP'],
                                        ['Micro-F1', 'mAP']):
        # Per-client delta lines with error bands
        for client in clients:
            mc = f'{client}_{metric_name}_mean'
            sc = f'{client}_{metric_name}_std'
            deltas = df[mc].values - baseline[mc]
            stds   = df[sc].values
            ax.plot(x, deltas,
                    linestyles[client] + markers[client],
                    color=COLORS[client], linewidth=2, markersize=6,
                    label=client, zorder=3)
            # ax.fill_between(x, deltas - stds, deltas + stds,
            #                 alpha=0.10, color=COLORS[client])

        # Global delta line (dashed, heavier, drawn last so it sits on top)
        mc_g = f'global_{metric_name}_mean'
        sc_g = f'global_{metric_name}_std'
        deltas_g = df[mc_g].values - baseline[mc_g]
        stds_g   = df[sc_g].values
        ax.plot(x, deltas_g, '--o',
                color=COLORS['Global'], linewidth=2.5, markersize=8,
                label='Global', zorder=5, markeredgecolor='white',
                markeredgewidth=0.8)
        # ax.fill_between(x, deltas_g - stds_g, deltas_g + stds_g,
        #                 alpha=0.12, color=COLORS['Global'])

        # Reference at Δ = 0 (= 8G:0L performance)
        ax.axhline(0, color='#444444', linewidth=1.0, linestyle=':', alpha=0.7,
                   label=r'$8H_g$:$0H_l$ ($\Delta = 0$)')

        # Axis labels and decoration
        ax.set_xlabel('Head Ratio (Global : Local)')
        ax.set_ylabel(f'$\\Delta$ {ylabel} vs. $8H_g$:$0H_l$ (pp)')
        ax.set_title(f'Personalization Gain — {ylabel}')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right')

    # Shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot head ratio results with error bars")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--total_heads", type=int, default=8,
                        help="Only plot configs where global+local==this value (default: 8)")
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(total_heads=args.total_heads)
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
            f'Personalization Benefit vs $8H_g$:$0H_l$ — {ylabel.split()[0]}',
        )
        fname = f'head_ratio_personalization_{metric_name}.png'
        fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.close(fig)

    # === DELTA FROM 8G:0L PLOT ===
    fig_delta = plot_delta_per_client(df)
    fig_delta.savefig(OUTPUT_DIR / 'head_ratio_delta_from_baseline.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: head_ratio_delta_from_baseline.png")
    plt.close(fig_delta)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
