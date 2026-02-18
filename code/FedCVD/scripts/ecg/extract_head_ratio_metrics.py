#!/usr/bin/env python3
"""
Extract and compare metrics from head ratio experiments with multiple seeds.

This script extracts metrics from dual attention experiments with different
global/local head ratios and computes mean ± std across seeds.

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
    Find all head ratio experiment runs organized by ratio and seed.

    Returns dict:
    {
        "4-4": {
            42: Path(...),    # seed 42 run
            123: Path(...),   # seed 123 run
            ...
        },
        "5-3": {...},
        ...
    }
    """
    search_path = base_path / "dual_attention_resnet1d" / "feddualatt"

    if not search_path.exists():
        return {}

    runs = {}

    # Look for global*_local* directories
    ratio_pattern = re.compile(r'global(\d+)_local(\d+)')
    seed_pattern = re.compile(r'seed(\d+)')

    for ratio_dir in search_path.iterdir():
        if not ratio_dir.is_dir():
            continue

        match = ratio_pattern.match(ratio_dir.name)
        if not match:
            continue

        global_heads = int(match.group(1))
        local_heads = int(match.group(2))
        ratio_key = f"{global_heads}-{local_heads}"

        runs[ratio_key] = {}

        # Look for seed directories
        for seed_dir in ratio_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            seed_match = seed_pattern.match(seed_dir.name)
            if seed_match:
                seed = int(seed_match.group(1))
                # Find latest timestamp directory
                timestamp_dirs = [d for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if timestamp_dirs:
                    latest = max(timestamp_dirs, key=lambda d: d.name)
                    runs[ratio_key][seed] = latest
            elif seed_dir.name.isdigit():
                # Legacy format: direct timestamp without seed folder
                # Treat as seed 42 (default)
                runs[ratio_key][42] = seed_dir

    # Remove empty ratios
    runs = {k: v for k, v in runs.items() if v}

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
    # Prefer corrected metrics (from reevaluate_checkpoints.py) over original
    metric_corrected = run_dir / "server" / "metric_corrected.json"
    metric_file = metric_corrected if metric_corrected.exists() else run_dir / "server" / "metric.json"
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


def aggregate_metrics(seed_results: dict) -> dict:
    """
    Aggregate metrics across multiple seeds, computing mean and std.

    Args:
        seed_results: {seed: metrics_dict, ...}

    Returns:
        {
            'num_seeds': N,
            'seeds': [42, 123, ...],
            'clients': {
                0: {'acc_mean': X, 'acc_std': X, 'micro_f1_mean': X, ...},
                ...
            },
            'global': {'acc_mean': X, 'acc_std': X, ...}
        }
    """
    valid_results = {k: v for k, v in seed_results.items() if v is not None}

    if not valid_results:
        return None

    result = {
        'num_seeds': len(valid_results),
        'seeds': list(valid_results.keys()),
        'clients': {},
        'global': {}
    }

    # Aggregate per-client metrics
    for client_idx in range(4):
        client_data = {
            'acc': [],
            'micro_f1': [],
            'mAP': []
        }

        for metrics in valid_results.values():
            if client_idx in metrics.get('clients', {}):
                client_data['acc'].append(metrics['clients'][client_idx]['acc'])
                client_data['micro_f1'].append(metrics['clients'][client_idx]['micro_f1'])
                client_data['mAP'].append(metrics['clients'][client_idx]['mAP'])

        if client_data['acc']:
            result['clients'][client_idx] = {
                'acc_mean': np.mean(client_data['acc']),
                'acc_std': np.std(client_data['acc']),
                'micro_f1_mean': np.mean(client_data['micro_f1']),
                'micro_f1_std': np.std(client_data['micro_f1']),
                'mAP_mean': np.mean(client_data['mAP']),
                'mAP_std': np.std(client_data['mAP']),
            }

    # Aggregate global metrics
    global_data = {
        'acc': [],
        'micro_f1': [],
        'mAP': []
    }

    for metrics in valid_results.values():
        if metrics.get('global'):
            global_data['acc'].append(metrics['global']['acc'])
            global_data['micro_f1'].append(metrics['global']['micro_f1'])
            global_data['mAP'].append(metrics['global']['mAP'])

    if global_data['acc']:
        result['global'] = {
            'acc_mean': np.mean(global_data['acc']),
            'acc_std': np.std(global_data['acc']),
            'micro_f1_mean': np.mean(global_data['micro_f1']),
            'micro_f1_std': np.std(global_data['micro_f1']),
            'mAP_mean': np.mean(global_data['mAP']),
            'mAP_std': np.std(global_data['mAP']),
        }

    return result


def format_mean_std(mean: float, std: float, multiplier: float = 100) -> str:
    """Format mean ± std as string."""
    if std > 0:
        return f"{mean*multiplier:.2f}±{std*multiplier:.2f}"
    else:
        return f"{mean*multiplier:.2f}"


def print_comparison_tables(results: dict, show_std: bool = True):
    """Print comparison tables for all head ratios with mean ± std."""

    # Sort ratios by global heads
    sorted_ratios = sorted(results.keys(), key=lambda x: int(x.split('-')[0]))

    # Determine column width based on whether we're showing std
    col_width = 14 if show_std else 10

    # Table header
    header = f"{'Ratio':<8} | {'N':>3}"
    for name in CLIENT_NAMES:
        header += f" | {name:>{col_width}}"
    header += f" | {'Global':>{col_width}}"

    line_width = len(header) + 5

    # Table 1: Micro-F1
    print("\n" + "=" * line_width)
    print("Head Ratio Comparison: Per-Client Micro-F1 (%)")
    print("=" * line_width)
    print(header)
    print("-" * line_width)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<8} | {'--':>3}"
            for _ in range(5):
                row += f" | {'--':>{col_width}}"
            print(row)
            continue

        row = f"{ratio:<8} | {data['num_seeds']:>3}"
        for i in range(4):
            if i in data['clients']:
                val = format_mean_std(data['clients'][i]['micro_f1_mean'],
                                      data['clients'][i]['micro_f1_std']) if show_std else \
                      f"{data['clients'][i]['micro_f1_mean']*100:.2f}"
                row += f" | {val:>{col_width}}"
            else:
                row += f" | {'--':>{col_width}}"

        if data['global']:
            val = format_mean_std(data['global']['micro_f1_mean'],
                                  data['global']['micro_f1_std']) if show_std else \
                  f"{data['global']['micro_f1_mean']*100:.2f}"
            row += f" | {val:>{col_width}}"
        else:
            row += f" | {'--':>{col_width}}"
        print(row)

    # Table 2: Accuracy
    print("\n" + "=" * line_width)
    print("Head Ratio Comparison: Per-Client Accuracy (%)")
    print("=" * line_width)
    print(header)
    print("-" * line_width)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<8} | {'--':>3}"
            for _ in range(5):
                row += f" | {'--':>{col_width}}"
            print(row)
            continue

        row = f"{ratio:<8} | {data['num_seeds']:>3}"
        for i in range(4):
            if i in data['clients']:
                val = format_mean_std(data['clients'][i]['acc_mean'],
                                      data['clients'][i]['acc_std']) if show_std else \
                      f"{data['clients'][i]['acc_mean']*100:.2f}"
                row += f" | {val:>{col_width}}"
            else:
                row += f" | {'--':>{col_width}}"

        if data['global']:
            val = format_mean_std(data['global']['acc_mean'],
                                  data['global']['acc_std']) if show_std else \
                  f"{data['global']['acc_mean']*100:.2f}"
            row += f" | {val:>{col_width}}"
        else:
            row += f" | {'--':>{col_width}}"
        print(row)

    # Table 3: mAP
    print("\n" + "=" * line_width)
    print("Head Ratio Comparison: Per-Client mAP (%)")
    print("=" * line_width)
    print(header)
    print("-" * line_width)

    for ratio in sorted_ratios:
        data = results[ratio]
        if data is None:
            row = f"{ratio:<8} | {'--':>3}"
            for _ in range(5):
                row += f" | {'--':>{col_width}}"
            print(row)
            continue

        row = f"{ratio:<8} | {data['num_seeds']:>3}"
        for i in range(4):
            if i in data['clients']:
                val = format_mean_std(data['clients'][i]['mAP_mean'],
                                      data['clients'][i]['mAP_std']) if show_std else \
                      f"{data['clients'][i]['mAP_mean']*100:.2f}"
                row += f" | {val:>{col_width}}"
            else:
                row += f" | {'--':>{col_width}}"

        if data['global']:
            val = format_mean_std(data['global']['mAP_mean'],
                                  data['global']['mAP_std']) if show_std else \
                  f"{data['global']['mAP_mean']*100:.2f}"
            row += f" | {val:>{col_width}}"
        else:
            row += f" | {'--':>{col_width}}"
        print(row)

    print("=" * line_width)


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

    # Find best configuration for each metric (by mean)
    best_f1 = max(valid_results.items(), key=lambda x: x[1]['global']['micro_f1_mean'])
    best_acc = max(valid_results.items(), key=lambda x: x[1]['global']['acc_mean'])
    best_map = max(valid_results.items(), key=lambda x: x[1]['global']['mAP_mean'])

    print("\nBest Head Ratio Configuration (Global Metrics):")
    print(f"  Micro-F1:  {best_f1[0]} ({format_mean_std(best_f1[1]['global']['micro_f1_mean'], best_f1[1]['global']['micro_f1_std'])}%)")
    print(f"  Accuracy:  {best_acc[0]} ({format_mean_std(best_acc[1]['global']['acc_mean'], best_acc[1]['global']['acc_std'])}%)")
    print(f"  mAP:       {best_map[0]} ({format_mean_std(best_map[1]['global']['mAP_mean'], best_map[1]['global']['mAP_std'])}%)")

    # Per-client best
    print("\nBest Head Ratio per Client (Micro-F1):")
    for i, name in enumerate(CLIENT_NAMES):
        client_results = {k: v for k, v in valid_results.items() if i in v.get('clients', {})}
        if client_results:
            best = max(client_results.items(), key=lambda x: x[1]['clients'][i]['micro_f1_mean'])
            val = format_mean_std(best[1]['clients'][i]['micro_f1_mean'],
                                  best[1]['clients'][i]['micro_f1_std'])
            print(f"  {name}: {best[0]} ({val}%)")

    # Print seed counts
    print("\nSeeds per configuration:")
    for ratio in sorted(valid_results.keys(), key=lambda x: int(x.split('-')[0])):
        data = valid_results[ratio]
        print(f"  {ratio}: {data['num_seeds']} seeds {data['seeds']}")


def save_to_csv(results: dict, csv_path: str):
    """Save results to CSV file with mean ± std."""
    import csv

    fieldnames = ['ratio', 'global_heads', 'local_heads', 'num_seeds', 'seeds']
    for name in CLIENT_NAMES:
        fieldnames.extend([
            f'{name}_acc_mean', f'{name}_acc_std',
            f'{name}_micro_f1_mean', f'{name}_micro_f1_std',
            f'{name}_mAP_mean', f'{name}_mAP_std'
        ])
    fieldnames.extend([
        'global_acc_mean', 'global_acc_std',
        'global_micro_f1_mean', 'global_micro_f1_std',
        'global_mAP_mean', 'global_mAP_std'
    ])

    rows = []
    sorted_ratios = sorted(results.keys(), key=lambda x: int(x.split('-')[0]))

    for ratio in sorted_ratios:
        data = results[ratio]
        global_h, local_h = ratio.split('-')

        row = {
            'ratio': ratio,
            'global_heads': global_h,
            'local_heads': local_h,
        }

        if data is None:
            row['num_seeds'] = 0
            row['seeds'] = ''
            for name in CLIENT_NAMES:
                for suffix in ['_mean', '_std']:
                    row[f'{name}_acc{suffix}'] = ''
                    row[f'{name}_micro_f1{suffix}'] = ''
                    row[f'{name}_mAP{suffix}'] = ''
            for suffix in ['_mean', '_std']:
                row[f'global_acc{suffix}'] = ''
                row[f'global_micro_f1{suffix}'] = ''
                row[f'global_mAP{suffix}'] = ''
        else:
            row['num_seeds'] = data['num_seeds']
            row['seeds'] = str(data['seeds'])

            for i, name in enumerate(CLIENT_NAMES):
                if i in data['clients']:
                    row[f'{name}_acc_mean'] = f"{data['clients'][i]['acc_mean']*100:.2f}"
                    row[f'{name}_acc_std'] = f"{data['clients'][i]['acc_std']*100:.2f}"
                    row[f'{name}_micro_f1_mean'] = f"{data['clients'][i]['micro_f1_mean']*100:.2f}"
                    row[f'{name}_micro_f1_std'] = f"{data['clients'][i]['micro_f1_std']*100:.2f}"
                    row[f'{name}_mAP_mean'] = f"{data['clients'][i]['mAP_mean']*100:.2f}"
                    row[f'{name}_mAP_std'] = f"{data['clients'][i]['mAP_std']*100:.2f}"
                else:
                    for suffix in ['_mean', '_std']:
                        row[f'{name}_acc{suffix}'] = ''
                        row[f'{name}_micro_f1{suffix}'] = ''
                        row[f'{name}_mAP{suffix}'] = ''

            if data['global']:
                row['global_acc_mean'] = f"{data['global']['acc_mean']*100:.2f}"
                row['global_acc_std'] = f"{data['global']['acc_std']*100:.2f}"
                row['global_micro_f1_mean'] = f"{data['global']['micro_f1_mean']*100:.2f}"
                row['global_micro_f1_std'] = f"{data['global']['micro_f1_std']*100:.2f}"
                row['global_mAP_mean'] = f"{data['global']['mAP_mean']*100:.2f}"
                row['global_mAP_std'] = f"{data['global']['mAP_std']*100:.2f}"
            else:
                for suffix in ['_mean', '_std']:
                    row[f'global_acc{suffix}'] = ''
                    row[f'global_micro_f1{suffix}'] = ''
                    row[f'global_mAP{suffix}'] = ''

        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract head ratio experiment metrics with mean ± std")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to output directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--no-std", action="store_true",
                        help="Don't show standard deviation (just mean)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    print("=" * 100)
    print("Head Ratio Experiment Results (with Multiple Seeds)")
    print("=" * 100)
    print(f"Output directory: {output_path}")
    print(f"Looking for: dual_attention_resnet1d/feddualatt/global*_local*/seed*/")
    print("-" * 100)

    # Find all head ratio runs
    runs = find_head_ratio_runs(output_path)

    if not runs:
        print("\n[!] No head ratio experiments found.")
        print("    Expected folder structure: output/dual_attention_resnet1d/feddualatt/global4_local4/seed42/TIMESTAMP/")
        print("\n    Run experiments with:")
        print("    python scripts/ecg/run_head_ratio.py --all --num_seeds 5")
        return

    # Count total runs
    total_runs = sum(len(seeds) for seeds in runs.values())
    print(f"\nFound {len(runs)} head ratio configuration(s) with {total_runs} total runs:")
    for ratio in sorted(runs.keys(), key=lambda x: int(x.split('-')[0])):
        seeds = sorted(runs[ratio].keys())
        print(f"  {ratio}: {len(seeds)} seeds {seeds}")

    # Extract and aggregate metrics for each ratio
    aggregated_results = {}
    print("\nExtracting and aggregating metrics...")

    for ratio, seed_runs in runs.items():
        seed_metrics = {}
        for seed, run_dir in seed_runs.items():
            metrics = extract_metrics(run_dir)
            seed_metrics[seed] = metrics

        # Aggregate across seeds
        aggregated = aggregate_metrics(seed_metrics)
        if aggregated:
            print(f"  [OK] {ratio}: {aggregated['num_seeds']} seeds aggregated")
        else:
            print(f"  [!] {ratio}: No valid metrics found")
        aggregated_results[ratio] = aggregated

    # Print comparison tables
    print_comparison_tables(aggregated_results, show_std=not args.no_std)
    print_summary(aggregated_results)

    # Save to CSV if requested
    if args.csv:
        save_to_csv(aggregated_results, args.csv)


if __name__ == "__main__":
    main()
