#!/usr/bin/env python3
"""
Aggregate and visualise FedDualAtt cross-evaluation results.

Cross-eval tests each client k's personalized model against all 4 test sets,
producing a 4x4 matrix per head-ratio config (averaged over seeds).

Output figures:
  head_ratio_cross_eval_f1.png    — 4x4 heatmaps of Micro-F1, one panel per config
  head_ratio_cross_eval_map.png   — same for mAP
  head_ratio_cross_eval_diff.png  — off-diagonal minus diagonal (generalisation gap)

Usage:
    python plot_cross_eval.py [--metric f1|map] [--configs 8g0l 7g1l ...]
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_PATH = SCRIPT_DIR / "../../../../output"
FIGURES_PATH = SCRIPT_DIR / "../../docs/experiments/cross_eval/figures"

CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]
N_CLIENTS = 4

# Ordered list of head-ratio configs (most global → most local)
ALL_CONFIGS = [
    ("global8_local0", "8G:0L"),
    ("global7_local1", "7G:1L"),
    ("global6_local2", "6G:2L"),
    ("global5_local3", "5G:3L"),
    ("global4_local4", "4G:4L"),
    ("global3_local5", "3G:5L"),
    ("global2_local6", "2G:6L"),
    ("global1_local7", "1G:7L"),
    ("global0_local8", "0G:8L"),
]


def load_cross_eval(base_dir: Path, config_dir_name: str):
    """
    Load all cross_eval_corrected.json files for a given config, average over seeds.

    Returns: np.ndarray (N_CLIENTS, N_CLIENTS, 2)  — [src, tgt, (f1, mAP)]
             or None if no data found.
    """
    config_path = base_dir / config_dir_name
    if not config_path.exists():
        return None

    matrices = []  # list of (4, 4, 2) arrays, one per seed

    for seed_dir in sorted(config_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        # Only use the latest valid timestamp (requires server/metric.json)
        valid_ts = [
            ts_dir for ts_dir in sorted(seed_dir.iterdir())
            if ts_dir.is_dir()
            and (ts_dir / "server" / "metric.json").exists()
        ]
        if not valid_ts:
            continue
        ts_dir = valid_ts[-1]
        ce_path = ts_dir / "server" / "cross_eval_corrected.json"
        if not ce_path.exists():
            continue

        with open(ce_path) as f:
            data = json.load(f)

        # Get the last round's data
        rounds = data.get("cross_eval", {})
        if not rounds:
            continue
        last_round = rounds[str(max(int(k) for k in rounds.keys()))]

        mat = np.full((N_CLIENTS, N_CLIENTS, 2), np.nan)
        for k in range(N_CLIENTS):
            src = last_round.get(str(k), {})
            for j in range(N_CLIENTS):
                tgt = src.get(str(j), {})
                if tgt:
                    f1 = tgt.get("micro_f1", np.nan)
                    mAP = float(np.mean(tgt.get("average_precision_score", [np.nan])))
                    mat[k, j, 0] = f1
                    mat[k, j, 1] = mAP

        matrices.append(mat)

    if not matrices:
        return None

    return np.nanmean(np.stack(matrices, axis=0), axis=0)  # (4, 4, 2)


def plot_heatmap_grid(matrices: list, labels: list, metric_idx: int,
                      metric_name: str, save_path: Path, vmin=None, vmax=None):
    """
    Plot a grid of 4x4 heatmaps (one per config), annotated with values.
    """
    n_configs = len(matrices)
    ncols = 3
    nrows = (n_configs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # Compute global vmin/vmax across all configs for consistent colorbar
    all_vals = [m[:, :, metric_idx] for m in matrices if m is not None]
    if vmin is None and all_vals:
        vmin = np.nanmin(np.stack(all_vals))
    if vmax is None and all_vals:
        vmax = np.nanmax(np.stack(all_vals))

    im = None
    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if mat is None:
            ax.set_visible(False)
            continue

        data = mat[:, :, metric_idx] * 100  # → percentage

        im = ax.imshow(data, cmap="YlOrRd", vmin=vmin * 100, vmax=vmax * 100,
                       aspect="auto")

        # Annotate cells
        for k in range(N_CLIENTS):
            for j in range(N_CLIENTS):
                val = data[k, j]
                if not np.isnan(val):
                    color = "black" if val < (vmin + vmax) * 50 else "white"
                    border = " [diag]" if k == j else ""
                    ax.text(j, k, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color=color,
                            fontweight="bold" if k == j else "normal")

        ax.set_xticks(range(N_CLIENTS))
        ax.set_yticks(range(N_CLIENTS))
        ax.set_xticklabels(CLIENT_NAMES, fontsize=8)
        ax.set_yticklabels(CLIENT_NAMES, fontsize=8)
        ax.set_xlabel("Target test set", fontsize=9)
        ax.set_ylabel("Source model", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")

        # Highlight diagonal
        for k in range(N_CLIENTS):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                       fill=False, edgecolor="blue",
                                       linewidth=2))

    # Hide unused subplots
    for idx in range(n_configs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared colorbar
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     label=f"{metric_name} (%)")

    fig.suptitle(
        f"FedDualAtt Cross-Evaluation — {metric_name}\n"
        "Rows = source client model, Cols = target test set\n"
        "Diagonal (blue border) = personalized model on its own test set",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_generalisation_gap(matrices: list, labels: list, save_path: Path):
    """
    For each config, compute mean off-diagonal minus mean diagonal for F1 and mAP.
    Plot as lines vs head ratio.
    """
    n = len(matrices)
    gap_f1 = np.full(n, np.nan)
    gap_mAP = np.full(n, np.nan)
    off_diag_f1 = np.full(n, np.nan)
    off_diag_mAP = np.full(n, np.nan)
    diag_f1 = np.full(n, np.nan)
    diag_mAP = np.full(n, np.nan)

    for i, mat in enumerate(matrices):
        if mat is None:
            continue
        mask_diag = np.eye(N_CLIENTS, dtype=bool)
        mask_off = ~mask_diag

        diag_f1[i] = np.nanmean(mat[:, :, 0][mask_diag]) * 100
        diag_mAP[i] = np.nanmean(mat[:, :, 1][mask_diag]) * 100
        off_diag_f1[i] = np.nanmean(mat[:, :, 0][mask_off]) * 100
        off_diag_mAP[i] = np.nanmean(mat[:, :, 1][mask_off]) * 100
        gap_f1[i] = off_diag_f1[i] - diag_f1[i]
        gap_mAP[i] = off_diag_mAP[i] - diag_mAP[i]

    x = np.arange(n)
    short_labels = [lb for lb in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, gap, diag, off, title in zip(
        axes,
        ["Micro-F1", "mAP"],
        [gap_f1, gap_mAP],
        [diag_f1, diag_mAP],
        [off_diag_f1, off_diag_mAP],
        ["Micro-F1 (%)", "mAP (%)"],
    ):
        ax.plot(x, diag, "o-", color="#1f77b4", label="Diagonal (own test set)", linewidth=2)
        ax.plot(x, off, "s--", color="#ff7f0e", label="Off-diagonal (other test sets)", linewidth=2)
        ax.plot(x, gap, "^:", color="#2ca02c", label="Gap (off − diag)", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
        ax.set_xlabel("Head ratio config (8G:0L → 0G:8L)")
        ax.set_ylabel(title)
        ax.set_title(f"Personalisation vs Generalisation — {metric}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FedDualAtt cross-evaluation heatmaps")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Subset of config dir names (e.g. global5_local3). "
                             "Default: all.")
    args = parser.parse_args()

    base = OUTPUT_PATH / "dual_attention_resnet1d" / "feddualatt"
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    configs_to_use = ALL_CONFIGS
    if args.configs:
        configs_to_use = [(d, l) for d, l in ALL_CONFIGS if d in args.configs]

    print("Loading cross-evaluation results...")
    matrices = []
    labels = []
    for dir_name, label in configs_to_use:
        mat = load_cross_eval(base, dir_name)
        matrices.append(mat)
        labels.append(label)
        status = "OK" if mat is not None else "MISSING"
        print(f"  {label:8s} ({dir_name}): {status}")

    valid = [(m, l) for m, l in zip(matrices, labels) if m is not None]
    if not valid:
        print("No cross-eval data found. Run reevaluate_checkpoints.py --cross-eval first.")
        return

    # F1 heatmaps
    plot_heatmap_grid(
        matrices, labels, metric_idx=0, metric_name="Micro-F1",
        save_path=FIGURES_PATH / "head_ratio_cross_eval_f1.png",
    )

    # mAP heatmaps
    plot_heatmap_grid(
        matrices, labels, metric_idx=1, metric_name="mAP",
        save_path=FIGURES_PATH / "head_ratio_cross_eval_map.png",
    )

    # Generalisation gap summary
    plot_generalisation_gap(
        matrices, labels,
        save_path=FIGURES_PATH / "head_ratio_cross_eval_gap.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
