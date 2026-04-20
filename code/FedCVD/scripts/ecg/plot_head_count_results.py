#!/usr/bin/env python3
"""
Phase 2 plot — ratio sweep at multiple total head counts.

Reads from extract_head_count_metrics.py (or its CSV output) and plots:
  1. Global Micro-F1 vs. % global heads — one line per N ∈ {8, 16, 32}
  2. Per-client Micro-F1 — same axes layout

N=8 data comes from existing head-ratio experiments.

Usage:
    python plot_head_count_results.py [--csv PATH] [--output-dir DIR] [--show]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"
DOCS_DIR = SCRIPT_DIR / "../../docs"
EXPERIMENT_DIR = DOCS_DIR / "experiments" / "head_count"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

sys.path.insert(0, str(SCRIPT_DIR))
from extract_head_count_metrics import build_head_count_results
from extract_head_ratio_metrics import CLIENT_NAMES

# Style matching existing plot scripts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TOTAL_HEADS_LEVELS = [1, 2, 4, 8, 16, 32]
COLORS = {1: "#9467bd", 2: "#8c564b", 4: "#e377c2",
          8: "#2ca02c", 16: "#1f77b4", 32: "#d62728"}   # purple, brown, pink, green, blue, red
MARKERS = {1: "D", 2: "v", 4: "P", 8: "s", 16: "o", 32: "^"}
PCT_TICKS = [0, 25, 50, 75, 100]


def load_data(output_path: Path) -> dict:
    """Returns {total_heads: list of row dicts sorted by pct_global}."""
    rows = build_head_count_results(output_path)
    data = {n: [] for n in TOTAL_HEADS_LEVELS}
    for row in rows:
        n = row["total_heads"]
        if n in data:
            data[n].append(row)
    for n in data:
        data[n].sort(key=lambda r: r["pct_global"])
    return data


def _curve(rows: list, metric_key: str):
    """Extract (xs, ys_mean, ys_std) from a list of row dicts."""
    xs, ys_mean, ys_std = [], [], []
    for r in rows:
        m = r.get(metric_key)
        s = r.get(metric_key.replace("_mean", "_std"))
        if m is not None:
            xs.append(r["pct_global"])
            ys_mean.append(m)
            ys_std.append(s if s is not None else 0.0)
    return np.array(xs), np.array(ys_mean), np.array(ys_std)


def plot_global(data: dict, metric_key: str, ylabel: str, title: str, out_path: Path, show: bool):
    fig, ax = plt.subplots(figsize=(9, 5))

    for n in TOTAL_HEADS_LEVELS:
        rows = data.get(n, [])
        if not rows:
            continue
        xs, ys, stds = _curve(rows, f"global_{metric_key}_mean")
        if len(xs) == 0:
            continue
        color = COLORS[n]
        ax.plot(xs, ys, marker=MARKERS[n], color=color, label=f"N={n} heads",
                linewidth=2, zorder=3)
        ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

    ax.set_xlabel("% Global Heads", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(PCT_TICKS)
    ax.set_xticklabels([f"{p}%" for p in PCT_TICKS])
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_client(data: dict, metric_key: str, ylabel: str, out_path: Path, show: bool):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes_flat = axes.flatten()

    for idx, (ax, name) in enumerate(zip(axes_flat, CLIENT_NAMES)):
        for n in TOTAL_HEADS_LEVELS:
            rows = data.get(n, [])
            if not rows:
                continue
            xs, ys, stds = _curve(rows, f"{name}_{metric_key}_mean")
            if len(xs) == 0:
                continue
            color = COLORS[n]
            ax.plot(xs, ys, marker=MARKERS[n], color=color, label=f"N={n}",
                    linewidth=2, zorder=3)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

        ax.set_title(name, fontsize=11)
        ax.set_xticks(PCT_TICKS)
        ax.set_xticklabels([f"{p}%" for p in PCT_TICKS])
        ax.set_xlabel("% Global Heads", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TOTAL_HEADS_LEVELS),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Phase 2: Per-Client {ylabel} vs. % Global Heads", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_combined_global(data: dict, out_path: Path, show: bool):
    """1×3 subplot: Micro-F1 | Accuracy | mAP (global only)."""
    metrics = [
        ("micro_f1", "Global Micro-F1 (%)"),
        ("acc", "Global Accuracy (%)"),
        ("mAP", "Global mAP (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        for n in TOTAL_HEADS_LEVELS:
            rows = data.get(n, [])
            if not rows:
                continue
            xs, ys, stds = _curve(rows, f"global_{metric_key}_mean")
            if len(xs) == 0:
                continue
            color = COLORS[n]
            ax.plot(xs, ys, marker=MARKERS[n], color=color, label=f"N={n}",
                    linewidth=2, zorder=3)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

        ax.set_xlabel("% Global Heads", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(PCT_TICKS)
        ax.set_xticklabels([f"{p}%" for p in PCT_TICKS])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TOTAL_HEADS_LEVELS),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Phase 2: Global Metrics vs. Head Ratio at Different Total Heads",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_saturation_curve(data: dict, out_path: Path, show: bool):
    """
    Saturation curve: Global Micro-F1 vs total head count N, one line per ratio.

    Ratio points shown: 100% global, 75%, 50% (the non-collapsing configs).
    x-axis: N on log2 scale (1, 2, 4, 8, 16, 32).
    """
    # (pct_global, label, color, marker)
    ratio_styles = [
        (100, "100% global", "#1f77b4", "o"),
        (75,  "75% global",  "#2ca02c", "s"),
        (50,  "50% global",  "#ff7f0e", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (metric_key, ylabel) in zip(axes, [
        ("micro_f1", "Global Micro-F1 (%)"),
        ("mAP",      "Global mAP (%)"),
    ]):
        for pct, label, color, marker in ratio_styles:
            xs, ys, stds = [], [], []
            for n in TOTAL_HEADS_LEVELS:
                rows = data.get(n, [])
                # find the row closest to this pct_global
                target_g = round(n * pct / 100)
                match = next((r for r in rows if r["global_heads"] == target_g), None)
                if match is None:
                    continue
                m = match.get(f"global_{metric_key}_mean")
                s = match.get(f"global_{metric_key}_std", 0.0)
                if m is not None:
                    xs.append(n)
                    ys.append(m)
                    stds.append(s if s is not None else 0.0)

            if xs:
                xs, ys, stds = np.array(xs), np.array(ys), np.array(stds)
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=2, zorder=3)
                ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xticks(TOTAL_HEADS_LEVELS)
        ax.set_xticklabels([str(n) for n in TOTAL_HEADS_LEVELS])
        ax.set_xlabel("Total Attention Heads (N)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10)

    fig.suptitle("Saturation Curve: Performance vs. Total Head Count",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 head count ratio sweep plots")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.output_dir))

    present = [n for n in TOTAL_HEADS_LEVELS if data[n]]
    if not present:
        print("[!] No Phase 2 results found. Run run_head_count_experiment.py first.")
        return
    print(f"Found data for N in {present}")

    # Global Micro-F1
    plot_global(
        data, "micro_f1", "Global Micro-F1 (%)",
        "Phase 2: Global Micro-F1 vs. Head Ratio",
        FIGURES_DIR / "head_count_global_f1.png", args.show,
    )

    # Global mAP
    plot_global(
        data, "mAP", "Global mAP (%)",
        "Phase 2: Global mAP vs. Head Ratio",
        FIGURES_DIR / "head_count_global_mAP.png", args.show,
    )

    # Combined global (3-panel)
    plot_combined_global(data, FIGURES_DIR / "head_count_global_combined.png", args.show)

    # Per-client Micro-F1
    plot_per_client(
        data, "micro_f1", "Micro-F1 (%)",
        FIGURES_DIR / "head_count_client_f1.png", args.show,
    )

    # Saturation curve (key figure: Global F1 vs N at fixed ratios)
    plot_saturation_curve(data, FIGURES_DIR / "head_count_saturation_curve.png", args.show)


if __name__ == "__main__":
    main()
