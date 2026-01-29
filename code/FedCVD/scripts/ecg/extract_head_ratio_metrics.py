#!/usr/bin/env python3
"""
Extract and compare metrics from head ratio experiments.

This script extracts metrics from dual attention experiments with different
global/local head ratios (4-4, 5-3, 6-2, 7-1) and displays them in a
comparison table.

Usage:
    python extract_head_ratio_metrics.py [--output-dir OUTPUT_DIR] [--csv OUTPUT.csv]

Options:
    --output-dir: Path to output directory (default: PROJECT_ROOT/output)
    --csv: Save results to CSV file
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path


# Get project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"

# Client names matching the paper
CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]


def find_head_ratio_runs(base_path: Path) -> dict:
    """
    Find all head ratio experiment runs.

    Returns dict mapping ratio string (e.g., "4-4") to run directory path.
    """
    search_path = base_path / "dual_attention_resnet1d" / "feddualatt"

    if not search_path.exists():
        return {}

    runs = {}

    # Look for global*_local* directories
    pattern = re.compile(r'global(\d+)_local(\d+)')

    for item in search_path.iterdir():
        if not item.is_dir():
            continue

        match = pattern.match(item.name)
        if match:
            global_heads = int(match.group(1))
            local_heads = int(match.group(2))
            ratio_key = f"{global_heads}-{local_heads}"

            # Find latest timestamp directory
            timestamp_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.isdigit()]
            if timestamp_dirs:
                latest = max(timestamp_dirs, key=lambda d: d.name)
                runs[ratio_key] = latest

    return runs


def extract_metrics(run_dir: Path) -> dict | None:
    """
    Extract per-client and global metrics from a run directory.

    Returns a dict with structure:
    {
        'round': last_round,
        'clients': {
            0: {'acc': X, 'micro_f1': X, 'mAP': X},
            1: {'acc': X, 'micro_f1': X, 'mAP': X},
            ...
        },
        'global': {'acc': X, 'micro_f1': X, 'mAP': X},
        'settings': {...}
    }
    """
    metric_file = run_dir / "server" / "metric.json"
    settings_file = run_dir / "setting.json"

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)

        result = {
            'round': None,
            'clients': {},
            'global': None,
            'settings': None,
            'run_dir': str(run_dir)
        }

        # Load settings if available
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                result['settings'] = json.load(f)

        # Extract per-client LOCAL metrics
        if 'local_test' in data:
            rounds = list(data['local_test'].keys())
            last_round = max(rounds, key=int)
            result['round'] = last_round

            client_metrics = data['local_test'][last_round]

            for client_id, metrics in client_metrics.items():
                idx = int(client_id)
                # Handle nested structure (may have epoch level)
                if isinstance(metrics, dict):
                    if '0' in metrics:  # Epoch level exists
                        metrics = metrics['0']

                    mAP = metrics.get('average_precision_score', [])
                    if isinstance(mAP, list):
                        mAP = np.mean(mAP) if mAP else 0

                    result['clients'][idx] = {
                        'acc': metrics.get('accuracy', metrics.get('acc', 0)),
                        'micro_f1': metrics.get('micro_f1', 0),
                        'mAP': mAP
                    }

        # Extract GLOBAL metrics
        if 'global_test' in data:
            rounds = list(data['global_test'].keys())
            last_round = max(rounds, key=int)
            result['round'] = last_round

            metrics = data['global_test'][last_round]
            mAP = metrics.get('average_precision_score', [])
            if isinstance(mAP, list):
                mAP = np.mean(mAP) if mAP else 0

            result['global'] = {
                'acc': metrics.get('accuracy', metrics.get('acc', 0)),
                'micro_f1': metrics.get('micro_f1', 0),
                'mAP': mAP
            }

        return result

    except Exception as e:
        print(f"Error reading {metric_file}: {e}")
        return None


def print_comparison_tables(results: dict):
    """Print comparison tables for all head ratios."""

    # Sort ratios by global heads (descending local personalization)
    sorted_ratios = sorted(results.keys(), key=lambda x: int(x.split('-')[0]))

    # Table header
    header = f"{'Ratio':<10}"
    for name in CLIENT_NAMES:
        header += f" | {name:>10}"
    header += f" | {'Global':>10}"

    # Table 1: Micro-F1
    print("\n" + "=" * 90)
    print("Head Ratio Comparison: Per-Client Micro-F1 (%)")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<10}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{ratio:<10}"
        for i in range(4):
            if i in data['clients']:
                val = data['clients'][i].get('micro_f1', 0)
                row += f" | {val*100:>10.2f}"
            else:
                row += f" | {'--':>10}"

        if data['global']:
            row += f" | {data['global']['micro_f1']*100:>10.2f}"
        else:
            row += f" | {'--':>10}"
        print(row)

    # Table 2: Accuracy
    print("\n" + "=" * 90)
    print("Head Ratio Comparison: Per-Client Accuracy (%)")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<10}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{ratio:<10}"
        for i in range(4):
            if i in data['clients']:
                val = data['clients'][i].get('acc', 0)
                row += f" | {val*100:>10.2f}"
            else:
                row += f" | {'--':>10}"

        if data['global']:
            row += f" | {data['global']['acc']*100:>10.2f}"
        else:
            row += f" | {'--':>10}"
        print(row)

    # Table 3: mAP
    print("\n" + "=" * 90)
    print("Head Ratio Comparison: Per-Client mAP (%)")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<10}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{ratio:<10}"
        for i in range(4):
            if i in data['clients']:
                val = data['clients'][i].get('mAP', 0)
                row += f" | {val*100:>10.2f}"
            else:
                row += f" | {'--':>10}"

        if data['global']:
            row += f" | {data['global']['mAP']*100:>10.2f}"
        else:
            row += f" | {'--':>10}"
        print(row)

    print("=" * 90)


def print_summary(results: dict):
    """Print summary with best configuration."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v is not None and v.get('global')}

    if not valid_results:
        print("No valid results found.")
        return

    # Find best configuration for each metric
    best_f1 = max(valid_results.items(), key=lambda x: x[1]['global']['micro_f1'])
    best_acc = max(valid_results.items(), key=lambda x: x[1]['global']['acc'])
    best_map = max(valid_results.items(), key=lambda x: x[1]['global']['mAP'])

    print("\nBest Head Ratio Configuration (Global Metrics):")
    print(f"  Micro-F1:  {best_f1[0]} ({best_f1[1]['global']['micro_f1']*100:.2f}%)")
    print(f"  Accuracy:  {best_acc[0]} ({best_acc[1]['global']['acc']*100:.2f}%)")
    print(f"  mAP:       {best_map[0]} ({best_map[1]['global']['mAP']*100:.2f}%)")

    # Per-client best
    print("\nBest Head Ratio per Client (Micro-F1):")
    for i, name in enumerate(CLIENT_NAMES):
        client_results = {k: v for k, v in valid_results.items() if i in v.get('clients', {})}
        if client_results:
            best = max(client_results.items(), key=lambda x: x[1]['clients'][i]['micro_f1'])
            print(f"  {name}: {best[0]} ({best[1]['clients'][i]['micro_f1']*100:.2f}%)")

    print(f"\nConfigurations found: {len(valid_results)}")


def save_to_csv(results: dict, csv_path: str):
    """Save results to CSV file."""
    import csv

    fieldnames = ['ratio', 'global_heads', 'local_heads']
    for name in CLIENT_NAMES:
        fieldnames.extend([f'{name}_acc', f'{name}_micro_f1', f'{name}_mAP'])
    fieldnames.extend(['global_acc', 'global_micro_f1', 'global_mAP', 'run_dir'])

    rows = []
    sorted_ratios = sorted(results.keys(), key=lambda x: int(x.split('-')[0]))

    for ratio in sorted_ratios:
        data = results[ratio]
        global_h, local_h = ratio.split('-')

        row = {
            'ratio': ratio,
            'global_heads': global_h,
            'local_heads': local_h
        }

        if data is None:
            for name in CLIENT_NAMES:
                row[f'{name}_acc'] = ''
                row[f'{name}_micro_f1'] = ''
                row[f'{name}_mAP'] = ''
            row['global_acc'] = ''
            row['global_micro_f1'] = ''
            row['global_mAP'] = ''
            row['run_dir'] = ''
        else:
            for i, name in enumerate(CLIENT_NAMES):
                if i in data['clients']:
                    row[f'{name}_acc'] = f"{data['clients'][i]['acc']*100:.2f}"
                    row[f'{name}_micro_f1'] = f"{data['clients'][i]['micro_f1']*100:.2f}"
                    row[f'{name}_mAP'] = f"{data['clients'][i]['mAP']*100:.2f}"
                else:
                    row[f'{name}_acc'] = ''
                    row[f'{name}_micro_f1'] = ''
                    row[f'{name}_mAP'] = ''

            if data['global']:
                row['global_acc'] = f"{data['global']['acc']*100:.2f}"
                row['global_micro_f1'] = f"{data['global']['micro_f1']*100:.2f}"
                row['global_mAP'] = f"{data['global']['mAP']*100:.2f}"
            else:
                row['global_acc'] = ''
                row['global_micro_f1'] = ''
                row['global_mAP'] = ''

            row['run_dir'] = data.get('run_dir', '')

        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract head ratio experiment metrics")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to output directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    print("=" * 90)
    print("Head Ratio Experiment Results")
    print("=" * 90)
    print(f"Output directory: {output_path}")
    print(f"Looking for: dual_attention_resnet1d/feddualatt/global*_local*/")
    print("-" * 90)

    # Find all head ratio runs
    runs = find_head_ratio_runs(output_path)

    if not runs:
        print("\n[!] No head ratio experiments found.")
        print("    Expected folder structure: output/dual_attention_resnet1d/feddualatt/global4_local4/TIMESTAMP/")
        print("\n    Run experiments with:")
        print("    python trainers/run_head_ratio_experiment.py --input_path /data --output_path /output --global_heads 4")
        return

    print(f"\nFound {len(runs)} head ratio configuration(s):")
    for ratio, path in sorted(runs.items()):
        print(f"  {ratio}: {path}")

    # Extract metrics for each ratio
    results = {}
    print("\nExtracting metrics...")
    for ratio, run_dir in runs.items():
        metrics = extract_metrics(run_dir)
        if metrics is None:
            print(f"  [!] {ratio}: No metrics found")
        else:
            print(f"  [OK] {ratio}: Round {metrics.get('round', 'N/A')}")
        results[ratio] = metrics

    # Print comparison tables
    print_comparison_tables(results)
    print_summary(results)

    # Save to CSV if requested
    if args.csv:
        save_to_csv(results, args.csv)


if __name__ == "__main__":
    main()
