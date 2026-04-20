#!/usr/bin/env python3
"""
Phase 2 extractor — group results by total head count for the ratio sweep.

Reuses find_head_ratio_runs / extract_metrics / aggregate_metrics from
extract_head_ratio_metrics.py (which already scans global*_local*/seed*/).
Adds total_heads and pct_global columns so results from N=8, N=16, and N=32
can be plotted on the same axes.

Usage:
    python extract_head_count_metrics.py [--output-dir OUTPUT_DIR] [--csv OUTPUT.csv]

Options:
    --output-dir: Path to output directory (default: PROJECT_ROOT/output)
    --csv:        Save results to CSV file
    --n:          Filter to specific total head count (e.g. --n 16)
"""

import csv
import sys
import argparse
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"
DOCS_DIR = SCRIPT_DIR / "../../docs"
DEFAULT_CSV_PATH = DOCS_DIR / "experiments" / "head_count" / "metrics.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from extract_head_ratio_metrics import (
    find_head_ratio_runs,
    extract_metrics,
    aggregate_metrics,
    CLIENT_NAMES,
    format_mean_std,
)


def build_head_count_results(output_path: Path, filter_n: int = None) -> list[dict]:
    """
    Return a list of result dicts, one per (global_heads, local_heads) config.

    Each dict contains:
        total_heads, pct_global, global_heads, local_heads,
        num_seeds, seeds,
        global_micro_f1_mean/std, global_acc_mean/std, global_mAP_mean/std,
        {CLIENT}_micro_f1_mean/std, {CLIENT}_acc_mean/std, {CLIENT}_mAP_mean/std
    """
    runs = find_head_ratio_runs(output_path)

    rows = []
    for ratio_key, seed_runs in runs.items():
        g, l = (int(x) for x in ratio_key.split("-"))
        n = g + l
        if filter_n is not None and n != filter_n:
            continue

        seed_metrics = {seed: extract_metrics(run_dir) for seed, run_dir in seed_runs.items()}
        agg = aggregate_metrics(seed_metrics)
        if agg is None:
            continue

        row = {
            "total_heads": n,
            "pct_global": round(g / n * 100) if n > 0 else 0,
            "global_heads": g,
            "local_heads": l,
            "num_seeds": agg["num_seeds"],
            "seeds": agg["seeds"],
        }

        # Global metrics
        if agg.get("global"):
            for metric in ("acc", "micro_f1", "mAP"):
                row[f"global_{metric}_mean"] = round(agg["global"][f"{metric}_mean"] * 100, 4)
                row[f"global_{metric}_std"] = round(agg["global"][f"{metric}_std"] * 100, 4)
        else:
            for metric in ("acc", "micro_f1", "mAP"):
                row[f"global_{metric}_mean"] = None
                row[f"global_{metric}_std"] = None

        # Per-client metrics
        for i, name in enumerate(CLIENT_NAMES):
            if i in agg.get("clients", {}):
                for metric in ("acc", "micro_f1", "mAP"):
                    row[f"{name}_{metric}_mean"] = round(agg["clients"][i][f"{metric}_mean"] * 100, 4)
                    row[f"{name}_{metric}_std"] = round(agg["clients"][i][f"{metric}_std"] * 100, 4)
            else:
                for metric in ("acc", "micro_f1", "mAP"):
                    row[f"{name}_{metric}_mean"] = None
                    row[f"{name}_{metric}_std"] = None

        rows.append(row)

    # Sort by (total_heads, pct_global)
    rows.sort(key=lambda r: (r["total_heads"], r["pct_global"]))
    return rows


def print_tables(rows: list[dict]):
    """Print Micro-F1 comparison grouped by total_heads."""
    if not rows:
        print("No results found.")
        return

    ns = sorted(set(r["total_heads"] for r in rows))

    for n in ns:
        n_rows = [r for r in rows if r["total_heads"] == n]
        print(f"\n{'=' * 80}")
        print(f"N={n} total heads — Global Micro-F1 and Per-Client Micro-F1 (%)")
        print(f"{'=' * 80}")
        header = f"  {'%global':>8} | {'global':>14}"
        for name in CLIENT_NAMES:
            header += f" | {name:>14}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in n_rows:
            gf = format_mean_std(r["global_micro_f1_mean"] / 100, r["global_micro_f1_std"] / 100) \
                if r.get("global_micro_f1_mean") is not None else "--"
            row_str = f"  {r['pct_global']:>7}% | {gf:>14}"
            for name in CLIENT_NAMES:
                val = r.get(f"{name}_micro_f1_mean")
                std = r.get(f"{name}_micro_f1_std")
                if val is not None:
                    row_str += f" | {format_mean_std(val/100, std/100):>14}"
                else:
                    row_str += f" | {'--':>14}"
            print(row_str)


def save_csv(rows: list[dict], csv_path: str):
    if not rows:
        print("No data to save.")
        return

    fieldnames = ["total_heads", "pct_global", "global_heads", "local_heads", "num_seeds", "seeds"]
    for metric in ("acc", "micro_f1", "mAP"):
        fieldnames += [f"global_{metric}_mean", f"global_{metric}_std"]
    for name in CLIENT_NAMES:
        for metric in ("acc", "micro_f1", "mAP"):
            fieldnames += [f"{name}_{metric}_mean", f"{name}_{metric}_std"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Phase 2 head count metrics grouped by total heads"
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH),
                        help="Save results to CSV (default: docs/experiments/head_count/metrics.csv)")
    parser.add_argument("--n", type=int, default=None, help="Filter to specific total head count")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    print(f"Scanning: {output_path / 'dual_attention_resnet1d/feddualatt/global*_local*/seed*/'}")

    rows = build_head_count_results(output_path, filter_n=args.n)

    if not rows:
        print("\n[!] No results found. Run experiments first.")
        return

    ns = sorted(set(r["total_heads"] for r in rows))
    print(f"Found results for N in {ns} ({len(rows)} configs total)")

    print_tables(rows)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(rows, str(csv_path))


if __name__ == "__main__":
    main()
