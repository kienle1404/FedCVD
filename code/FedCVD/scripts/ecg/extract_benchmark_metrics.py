#!/usr/bin/env python3
"""
Extract and display metrics from ECG benchmark results.

Usage:
    python extract_benchmark_metrics.py [--output-dir OUTPUT_DIR]

Options:
    --output-dir: Path to output directory (default: PROJECT_ROOT/output)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Get project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"


def find_latest_run(base_path: Path, algorithm: str) -> Path | None:
    """Find the latest run directory for an algorithm."""
    if algorithm == "feddualatt":
        search_path = base_path / "dual_attention_resnet1d" / "feddualatt"
    elif algorithm in ["centralized", "local"]:
        search_path = base_path / "resnet1d34" / algorithm
    else:
        search_path = base_path / "resnet1d34" / algorithm

    if not search_path.exists():
        return None

    # Find timestamp directories
    timestamp_dirs = [d for d in search_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not timestamp_dirs:
        return None

    return max(timestamp_dirs, key=lambda d: d.name)


def extract_metrics(run_dir: Path, algorithm: str) -> dict | None:
    """Extract metrics from a run directory."""
    # Determine metric file location
    if algorithm in ["centralized"]:
        metric_file = run_dir / "metric.json"
    elif algorithm == "local":
        # Local has per-client metrics, aggregate them
        client_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("client")]
        if not client_dirs:
            return None
        # Use first client for now
        metric_file = client_dirs[0] / "metric.json"
    else:
        metric_file = run_dir / "server" / "metric.json"

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)

        # Get global_test metrics from the last round
        if 'global_test' in data:
            rounds = list(data['global_test'].keys())
            last_round = max(rounds, key=int)
            metrics = data['global_test'][last_round]

            micro_f1 = metrics.get('micro_f1', 0)
            accuracy = metrics.get('accuracy', 0)
            mAP = metrics.get('average_precision_score', [])
            if isinstance(mAP, list):
                mAP = np.mean(mAP) if mAP else 0

            return {
                'round': last_round,
                'micro_f1': micro_f1,
                'accuracy': accuracy,
                'mAP': mAP,
                'run_dir': str(run_dir)
            }
    except Exception as e:
        print(f"Error reading {metric_file}: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Extract ECG benchmark metrics")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to output directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    # All algorithms to check
    algorithms = [
        "centralized",
        "local",
        "fedavg",
        "fedprox",
        "scaffold",
        "ditto",
        "fedbn",
        "fedinit",
        "fedala",
        "fedfa",
        "fedsm",
        "feddualatt",
    ]

    print("=" * 90)
    print("ECG BENCHMARK RESULTS - Global Test Metrics (Final Round)")
    print("=" * 90)
    print(f"Output directory: {output_path}")
    print("-" * 90)
    print(f"{'Algorithm':<15} | {'Round':>5} | {'Micro-F1':>10} | {'Accuracy':>10} | {'mAP':>10} | {'Status':<10}")
    print("-" * 90)

    results = []
    for algo in algorithms:
        run_dir = find_latest_run(output_path, algo)
        if run_dir is None:
            print(f"{algo:<15} | {'--':>5} | {'--':>10} | {'--':>10} | {'--':>10} | {'Not found':<10}")
            continue

        metrics = extract_metrics(run_dir, algo)
        if metrics is None:
            print(f"{algo:<15} | {'--':>5} | {'--':>10} | {'--':>10} | {'--':>10} | {'No metrics':<10}")
            continue

        results.append({
            'algorithm': algo,
            **metrics
        })

        print(f"{algo:<15} | {metrics['round']:>5} | {metrics['micro_f1']*100:>9.2f}% | {metrics['accuracy']*100:>9.2f}% | {metrics['mAP']*100:>9.2f}% | {'OK':<10}")

    print("=" * 90)

    # Find best FL method (excluding centralized which is upper bound)
    fl_results = [r for r in results if r['algorithm'] not in ['centralized', 'local']]
    if fl_results:
        best_f1 = max(fl_results, key=lambda x: x['micro_f1'])
        best_acc = max(fl_results, key=lambda x: x['accuracy'])
        best_map = max(fl_results, key=lambda x: x['mAP'])

        print("\nBest FL Methods:")
        print(f"  Micro-F1:  {best_f1['algorithm']} ({best_f1['micro_f1']*100:.2f}%)")
        print(f"  Accuracy:  {best_acc['algorithm']} ({best_acc['accuracy']*100:.2f}%)")
        print(f"  mAP:       {best_map['algorithm']} ({best_map['mAP']*100:.2f}%)")

    # Save to CSV if requested
    if args.csv and results:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['algorithm', 'round', 'micro_f1', 'accuracy', 'mAP', 'run_dir'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()
