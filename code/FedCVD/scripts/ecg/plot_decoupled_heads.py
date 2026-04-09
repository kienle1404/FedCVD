#!/usr/bin/env python3
"""
Phase 1 plot — decoupled head scaling.

Reads from extract_head_ratio_metrics.py output (or a CSV produced by it),
filters to configs where global_heads ∈ {4, 8} and local_heads varies,
and plots global + per-client Micro-F1 vs. local_heads.

Decision reference:
  - If the g=8 line stays flat as local_heads increases → capacity-driven trade-off → proceed to Phase 2.
  - If the g=8 line drops → architectural trade-off → pivot.

Usage:
    python plot_decoupled_heads.py [--csv PATH] [--show]
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
FIGURES_DIR = DOCS_DIR / "figures"

sys.path.insert(0, str(SCRIPT_DIR))
from extract_head_ratio_metrics import (
    find_head_ratio_runs,
    extract_metrics,
    aggregate_metrics,
    CLIENT_NAMES,
)

# Matplotlib style matching the existing plot scripts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Two fixed-global configurations form the two lines
GLOBAL_HEADS_LEVELS = [8]
COLORS = {8: "#1f77b4", 4: "#ff7f0e"}   # blue, orange
LOCAL_HEADS_ORDER = [0, 2, 4, 8, 16]    # expected x-axis values


def load_data(output_path: Path) -> dict:
    """
    Returns {global_heads: {local_heads: agg_dict}}.
    Only keeps configs where global_heads ∈ GLOBAL_HEADS_LEVELS.
    """
    runs = find_head_ratio_runs(output_path)
    data = {g: {} for g in GLOBAL_HEADS_LEVELS}

    for ratio_key, seed_runs in runs.items():
        g, l = (int(x) for x in ratio_key.split("-"))
        if g not in GLOBAL_HEADS_LEVELS:
            continue
        if l not in LOCAL_HEADS_ORDER:
            continue
        seed_metrics = {seed: extract_metrics(rd) for seed, rd in seed_runs.items()}
        agg = aggregate_metrics(seed_metrics)
        if agg:
            data[g][l] = agg

    return data


def _get_curve(data_for_g: dict, metric_key: str, section: str = "global", client_idx: int = None):
    """
    Extract x (local_heads), y_mean, y_std arrays for one line.

    section: "global" or "clients"
    """
    xs, ys_mean, ys_std = [], [], []
    for l in sorted(data_for_g.keys()):
        agg = data_for_g[l]
        if section == "global" and agg.get("global"):
            xs.append(l)
            ys_mean.append(agg["global"][f"{metric_key}_mean"] * 100)
            ys_std.append(agg["global"][f"{metric_key}_std"] * 100)
        elif section == "clients" and client_idx in agg.get("clients", {}):
            xs.append(l)
            ys_mean.append(agg["clients"][client_idx][f"{metric_key}_mean"] * 100)
            ys_std.append(agg["clients"][client_idx][f"{metric_key}_std"] * 100)
    return np.array(xs), np.array(ys_mean), np.array(ys_std)


def plot_global_metric(data: dict, metric_key: str, ylabel: str, title: str, out_path: Path, show: bool):
    fig, ax = plt.subplots(figsize=(8, 5))

    for g in GLOBAL_HEADS_LEVELS:
        if not data[g]:
            continue
        xs, ys, stds = _get_curve(data[g], metric_key, section="global")
        if len(xs) == 0:
            continue
        color = COLORS[g]
        ax.plot(xs, ys, marker="o", color=color, label=f"global_heads={g}", linewidth=2, zorder=3)
        ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

    # Dashed reference: g=8, l=0 baseline
    if 0 in data[8]:
        agg = data[8][0]
        if agg.get("global"):
            ref = agg["global"][f"{metric_key}_mean"] * 100
            ax.axhline(ref, linestyle="--", color=COLORS[8], alpha=0.5, linewidth=1,
                       label=f"baseline (g=8, l=0): {ref:.1f}%")

    ax.set_xlabel("local_heads (global_heads fixed)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(LOCAL_HEADS_ORDER)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_client_metric(data: dict, metric_key: str, ylabel: str, out_path: Path, show: bool):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for idx, (ax, name) in enumerate(zip(axes_flat, CLIENT_NAMES)):
        for g in GLOBAL_HEADS_LEVELS:
            if not data[g]:
                continue
            xs, ys, stds = _get_curve(data[g], metric_key, section="clients", client_idx=idx)
            if len(xs) == 0:
                continue
            color = COLORS[g]
            ax.plot(xs, ys, marker="o", color=color, label=f"global_heads={g}", linewidth=2, zorder=3)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

        ax.set_title(name, fontsize=11)
        ax.set_xticks(LOCAL_HEADS_ORDER)
        ax.set_xlabel("local_heads", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 decoupled heads plot")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.output_dir))

    total = sum(len(v) for v in data.values())
    if total == 0:
        print("[!] No Phase 1 results found. Run run_decoupled_heads.py first.")
        return

    for g in GLOBAL_HEADS_LEVELS:
        if data[g]:
            ls = sorted(data[g].keys())
            print(f"  global_heads={g}: local_heads found = {ls}")

    # Global Micro-F1
    plot_global_metric(
        data, "micro_f1", "Global Micro-F1 (%)",
        "Phase 1: Global Micro-F1 vs. Added Local Heads (global_heads fixed)",
        FIGURES_DIR / "decoupled_heads_global_f1.png", args.show,
    )

    # Global mAP
    plot_global_metric(
        data, "mAP", "Global mAP (%)",
        "Phase 1: Global mAP vs. Added Local Heads",
        FIGURES_DIR / "decoupled_heads_global_mAP.png", args.show,
    )

    # Per-client Micro-F1
    plot_per_client_metric(
        data, "micro_f1", "Micro-F1 (%)",
        FIGURES_DIR / "decoupled_heads_client_f1.png", args.show,
    )


if __name__ == "__main__":
    main()
