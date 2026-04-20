#!/usr/bin/env python3
"""
Convergence speed analysis for FedDualAtt.

Extracts per-round metrics from existing metric.json files and plots:
  1. Global Micro-F1 vs round (learning curves) for all head ratios
  2. Per-client Micro-F1 vs round
  3. Rounds-to-threshold analysis (rounds to reach 90%/95% of final F1)
  4. Training loss vs round

No re-training needed — uses existing experiment data.

Usage:
    python analyze_convergence.py [--configs global8_local0 global4_local4 ...]
    python analyze_convergence.py --all  # All 9 ratio configs at N=8
    python analyze_convergence.py --show
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"
DOCS_DIR = SCRIPT_DIR / "../../docs"
EXPERIMENT_DIR = DOCS_DIR / "experiments" / "convergence"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Configs ordered from most global to most local
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

# Distinct colors for 9 configs
COLORS = plt.cm.viridis(np.linspace(0, 0.95, 9))


def load_convergence_data(base_dir: Path, config_dir_name: str):
    """
    Load per-round metrics for a config, averaged over seeds.

    Returns dict with:
        rounds: list of int
        global_micro_f1: (n_rounds,) array — mean across seeds
        global_micro_f1_std: (n_rounds,) array
        global_loss: (n_rounds,) array
        global_mAP: (n_rounds,) array
        client_micro_f1: (n_clients, n_rounds) array
        n_seeds: int
    Or None if no data found.
    """
    config_path = base_dir / config_dir_name
    if not config_path.exists():
        return None

    all_global_f1 = []   # list of per-seed arrays
    all_global_loss = []
    all_global_mAP = []
    all_client_f1 = []   # list of (4, n_rounds) arrays

    for seed_dir in sorted(config_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue

        # Find latest valid timestamp
        valid_ts = [
            ts_dir for ts_dir in sorted(seed_dir.iterdir())
            if ts_dir.is_dir()
            and (ts_dir / "server" / "metric.json").exists()
        ]
        if not valid_ts:
            continue
        ts_dir = valid_ts[-1]

        metric_path = ts_dir / "server" / "metric.json"
        with open(metric_path) as f:
            data = json.load(f)

        gt = data.get("global_test", {})
        lt = data.get("local_test", {})

        if not gt:
            continue

        rounds = sorted(gt.keys(), key=int)
        n_rounds = len(rounds)

        # Global metrics per round
        g_f1 = np.array([gt[r].get("micro_f1", np.nan) for r in rounds]) * 100
        g_loss = np.array([gt[r].get("loss", np.nan) for r in rounds])
        g_mAP = np.array([
            float(np.mean(gt[r].get("average_precision_score", [np.nan]))) * 100
            for r in rounds
        ])

        all_global_f1.append(g_f1)
        all_global_loss.append(g_loss)
        all_global_mAP.append(g_mAP)

        # Per-client F1
        c_f1 = np.full((4, n_rounds), np.nan)
        for ri, r in enumerate(rounds):
            round_lt = lt.get(r, {})
            for ci in range(4):
                client_data = round_lt.get(str(ci), {})
                if client_data:
                    c_f1[ci, ri] = client_data.get("micro_f1", np.nan) * 100
        all_client_f1.append(c_f1)

    if not all_global_f1:
        return None

    # Stack and average
    n_rounds = min(len(a) for a in all_global_f1)
    g_f1_stack = np.stack([a[:n_rounds] for a in all_global_f1])
    g_loss_stack = np.stack([a[:n_rounds] for a in all_global_loss])
    g_mAP_stack = np.stack([a[:n_rounds] for a in all_global_mAP])
    c_f1_stack = np.stack([a[:, :n_rounds] for a in all_client_f1])

    return {
        "rounds": list(range(n_rounds)),
        "global_micro_f1": np.nanmean(g_f1_stack, axis=0),
        "global_micro_f1_std": np.nanstd(g_f1_stack, axis=0),
        "global_loss": np.nanmean(g_loss_stack, axis=0),
        "global_loss_std": np.nanstd(g_loss_stack, axis=0),
        "global_mAP": np.nanmean(g_mAP_stack, axis=0),
        "client_micro_f1": np.nanmean(c_f1_stack, axis=0),  # (4, n_rounds)
        "n_seeds": len(all_global_f1),
    }


def plot_global_convergence(all_data, out_path, show):
    """Plot global Micro-F1 vs round for all configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (dir_name, label) in enumerate(ALL_CONFIGS):
        data = all_data.get(dir_name)
        if data is None:
            continue
        rounds = data["rounds"]
        f1 = data["global_micro_f1"]
        std = data["global_micro_f1_std"]
        color = COLORS[i]

        ax.plot(rounds, f1, color=color, label=f"{label} ({data['n_seeds']}s)",
                linewidth=1.5, alpha=0.9)
        if data["n_seeds"] > 1:
            ax.fill_between(rounds, f1 - std, f1 + std, alpha=0.1, color=color)

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Global Micro-F1 (%)", fontsize=11)
    ax.set_title("Convergence: Global Micro-F1 vs Round (N=8, with detach fix)", fontsize=13)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_convergence(all_data, out_path, show):
    """Plot training loss vs round."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (dir_name, label) in enumerate(ALL_CONFIGS):
        data = all_data.get(dir_name)
        if data is None:
            continue
        rounds = data["rounds"]
        loss = data["global_loss"]
        color = COLORS[i]
        ax.plot(rounds, loss, color=color, label=label, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Global Test Loss", fontsize=11)
    ax.set_title("Convergence: Global Test Loss vs Round (N=8, with detach fix)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_client_convergence(all_data, out_path, show):
    """2x2 subplot: per-client F1 convergence for selected configs."""
    selected = ["global8_local0", "global5_local3", "global4_local4", "global0_local8"]
    selected_labels = ["8G:0L", "5G:3L", "4G:4L", "0G:8L"]
    sel_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for ci, (ax, client_name) in enumerate(zip(axes.flatten(), CLIENT_NAMES)):
        for si, (dir_name, label, color) in enumerate(zip(selected, selected_labels, sel_colors)):
            data = all_data.get(dir_name)
            if data is None:
                continue
            rounds = data["rounds"]
            f1 = data["client_micro_f1"][ci]
            ax.plot(rounds, f1, color=color, label=label, linewidth=1.5)

        ax.set_title(client_name, fontsize=11)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel("Micro-F1 (%)", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Per-Client Convergence (selected configs)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_rounds_to_threshold(all_data, out_path, show):
    """
    Bar chart: rounds to reach 90% and 95% of final Micro-F1.
    """
    thresholds = [0.90, 0.95]
    th_labels = ["90%", "95%"]
    th_colors = ["#1f77b4", "#ff7f0e"]

    configs_found = []
    rounds_data = {th: [] for th in thresholds}

    for dir_name, label in ALL_CONFIGS:
        data = all_data.get(dir_name)
        if data is None:
            continue
        configs_found.append(label)
        final_f1 = data["global_micro_f1"][-1]

        for th in thresholds:
            target = final_f1 * th
            reached = np.where(data["global_micro_f1"] >= target)[0]
            if len(reached) > 0:
                rounds_data[th].append(reached[0])
            else:
                rounds_data[th].append(len(data["rounds"]))  # never reached

    if not configs_found:
        print("No data for rounds-to-threshold plot.")
        return

    x = np.arange(len(configs_found))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (th, th_label, color) in enumerate(zip(thresholds, th_labels, th_colors)):
        offset = (i - 0.5) * width
        ax.bar(x + offset, rounds_data[th], width, label=f"{th_label} of final F1", color=color)

    ax.set_xlabel("Head Ratio Config", fontsize=11)
    ax.set_ylabel("Rounds to Threshold", fontsize=11)
    ax.set_title("Convergence Speed: Rounds to Reach % of Final Global Micro-F1", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_found, fontsize=9)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Convergence speed analysis")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base = Path(args.output_dir) / "dual_attention_resnet1d" / "feddualatt"

    configs_to_use = ALL_CONFIGS
    if args.configs:
        configs_to_use = [(d, l) for d, l in ALL_CONFIGS if d in args.configs]

    print("Loading per-round metrics...")
    all_data = {}
    for dir_name, label in configs_to_use:
        data = load_convergence_data(base, dir_name)
        if data:
            all_data[dir_name] = data
            print(f"  {label:8s}: {len(data['rounds'])} rounds, {data['n_seeds']} seed(s)")
        else:
            print(f"  {label:8s}: MISSING")

    if not all_data:
        print("No convergence data found.")
        return

    plot_global_convergence(all_data, FIGURES_DIR / "convergence_global_f1.png", args.show)
    plot_loss_convergence(all_data, FIGURES_DIR / "convergence_loss.png", args.show)
    plot_client_convergence(all_data, FIGURES_DIR / "convergence_client_f1.png", args.show)
    plot_rounds_to_threshold(all_data, FIGURES_DIR / "convergence_rounds_to_threshold.png", args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
