#!/usr/bin/env python3
"""
Extract and display metrics from ECG benchmark results in Table 2 format.

This script extracts per-client metrics and cross-evaluation results to match
the original FedCVD paper Table 2 format.

Usage:
    python extract_benchmark_metrics.py [--output-dir OUTPUT_DIR] [--csv OUTPUT.csv]

Options:
    --output-dir: Path to output directory (default: PROJECT_ROOT/output)
    --csv: Save results to CSV file
"""

import json
import argparse
import numpy as np
from pathlib import Path


# Get project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"

# Client names matching the paper
CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]

# Display names for algorithms (maps internal names to paper format)
DISPLAY_NAMES = {
    "centralized": "Centralized",
    "local_client1": "Client1 (SPH)",
    "local_client2": "Client2 (PTB-XL)",
    "local_client3": "Client3 (SXPH)",
    "local_client4": "Client4 (G12EC)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "Scaffold",
    "ditto": "Ditto",
    "fedinit": "FedInit",
    "fedala": "FedALA",
    "fedsm": "FedSM",
    "feddualatt": "FedDualAtt",
}


def find_latest_run(base_path: Path, algorithm: str) -> Path | None:
    """Find the latest run directory for an algorithm."""
    if algorithm == "feddualatt":
        search_path = base_path / "dual_attention_resnet1d" / "feddualatt"
    elif algorithm.startswith("local_client"):
        # Local client training: output/resnet1d34/local/clientN/TIMESTAMP/
        client_num = algorithm.replace("local_client", "")  # e.g., "1", "2", "3", "4"
        search_path = base_path / "resnet1d34" / "local" / f"client{client_num}"
    elif algorithm == "centralized":
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


def parse_cross_eval_from_metric_files(run_dir: Path, algorithm: str) -> dict | None:
    """
    Parse cross-evaluation results from per-client metric.json files.

    Returns a dict with structure:
    {
        'accuracy': {i: {j: value}},  # 4x4 matrix
        'micro_f1': {i: {j: value}},  # 4x4 matrix
        'mAP': {i: {j: value}},       # 4x4 matrix
    }
    Where matrix[i][j] = metric of model trained on client i, tested on client j's data.
    """
    if algorithm in ["centralized", "local"]:
        return None  # No cross-evaluation for these

    # Initialize 4x4 matrices for each metric
    cross_eval = {
        'accuracy': {i: {j: None for j in range(4)} for i in range(4)},
        'micro_f1': {i: {j: None for j in range(4)} for i in range(4)},
        'mAP': {i: {j: None for j in range(4)} for i in range(4)},
    }

    # Parse each client's metric.json file
    for train_client in range(4):
        client_dir = run_dir / f"client{train_client + 1}"
        metric_file = client_dir / "metric.json"

        if not metric_file.exists():
            continue

        try:
            with open(metric_file, 'r') as f:
                data = json.load(f)

            if 'local_test' not in data:
                continue

            # Get last round
            rounds = list(data['local_test'].keys())
            last_round = max(rounds, key=int)

            # The structure is local_test[round]['0'][test_client_id]
            round_data = data['local_test'][last_round]

            # Handle nested epoch structure
            if '0' in round_data and isinstance(round_data['0'], dict):
                # Check if this is epoch level or test client level
                first_key = list(round_data['0'].keys())[0]
                if first_key in ['0', '1', '2', '3']:
                    # This is epoch level, get epoch 0
                    test_data = round_data['0']
                else:
                    # Direct metrics for one client
                    test_data = round_data
            else:
                test_data = round_data

            # Extract metrics for each test client
            for test_client_str, metrics in test_data.items():
                if not test_client_str.isdigit():
                    continue
                test_client = int(test_client_str)
                if test_client >= 4:
                    continue

                # Get accuracy
                acc = metrics.get('accuracy', metrics.get('acc', None))
                cross_eval['accuracy'][train_client][test_client] = acc

                # Get micro_f1
                micro_f1 = metrics.get('micro_f1', None)
                cross_eval['micro_f1'][train_client][test_client] = micro_f1

                # Get mAP (average of per-class average precision)
                avg_prec = metrics.get('average_precision_score', [])
                if isinstance(avg_prec, list) and avg_prec:
                    mAP = np.mean(avg_prec)
                else:
                    mAP = None
                cross_eval['mAP'][train_client][test_client] = mAP

        except Exception as e:
            print(f"Error parsing {metric_file}: {e}")

    return cross_eval


def extract_local_client_metrics(run_dir: Path) -> dict | None:
    """
    Extract metrics for a locally-trained client model tested on all clients.

    For local training, each client trains independently and is evaluated on:
    - local_test[epoch][test_client_idx]: this model tested on each client's data
    - global_test[epoch]: this model tested on merged data from all clients

    Returns a dict matching the FL metrics format for consistency.
    """
    metric_file = run_dir / "metric.json"

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)

        result = {
            'round': None,
            'clients': {},
            'global': None,
            'cross_eval': None,
            'run_dir': str(run_dir)
        }

        # Extract per-client LOCAL metrics (this model tested on each client's data)
        if 'local_test' in data:
            epochs = list(data['local_test'].keys())
            last_epoch = max(epochs, key=int)
            result['round'] = last_epoch

            test_data = data['local_test'][last_epoch]

            # Extract metrics for each test client
            for test_client_str, metrics in test_data.items():
                if not test_client_str.isdigit():
                    continue
                test_client = int(test_client_str)
                if test_client >= 4:
                    continue

                mAP = metrics.get('average_precision_score', [])
                if isinstance(mAP, list):
                    mAP = np.mean(mAP) if mAP else 0

                result['clients'][test_client] = {
                    'acc': metrics.get('accuracy', metrics.get('acc', 0)),
                    'micro_f1': metrics.get('micro_f1', 0),
                    'mAP': mAP
                }

        # Extract GLOBAL metrics (this model tested on merged data)
        if 'global_test' in data:
            epochs = list(data['global_test'].keys())
            last_epoch = max(epochs, key=int)
            result['round'] = last_epoch

            metrics = data['global_test'][last_epoch]
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


def extract_per_client_metrics(run_dir: Path, algorithm: str) -> dict | None:
    """
    Extract per-client metrics from a run directory.

    Returns a dict with structure:
    {
        'round': last_round,
        'clients': {
            0: {'acc': X, 'micro_f1': X, 'mAP': X},
            1: {'acc': X, 'micro_f1': X, 'mAP': X},
            ...
        },
        'global': {'acc': X, 'micro_f1': X, 'mAP': X},
        'cross_eval': 4x4 matrix of accuracies,
        'run_dir': str
    }
    """
    # Determine metric file location and extraction method
    if algorithm.startswith("local_client"):
        # Local client training: delegate to specialized function
        return extract_local_client_metrics(run_dir)
    elif algorithm == "centralized":
        metric_file = run_dir / "metric.json"
    else:
        metric_file = run_dir / "server" / "metric.json"

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)

        result = {
            'round': None,
            'clients': {},
            'global': None,
            'cross_eval': None,
            'run_dir': str(run_dir)
        }

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

        # For centralized, only global metrics exist
        if algorithm == "centralized" and 'test' in data:
            rounds = list(data['test'].keys())
            last_round = max(rounds, key=int)
            result['round'] = last_round

            metrics = data['test'][last_round]
            if isinstance(metrics, dict) and '0' in metrics:
                metrics = metrics['0']

            mAP = metrics.get('average_precision_score', [])
            if isinstance(mAP, list):
                mAP = np.mean(mAP) if mAP else 0

            result['global'] = {
                'acc': metrics.get('accuracy', metrics.get('acc', 0)),
                'micro_f1': metrics.get('micro_f1', 0),
                'mAP': mAP
            }

        # Get cross-evaluation from per-client metric files
        cross_eval = parse_cross_eval_from_metric_files(run_dir, algorithm)
        result['cross_eval'] = cross_eval

        return result

    except Exception as e:
        print(f"Error reading {metric_file}: {e}")
        return None


def print_table_2_format(results: dict):
    """Print results in Table 2 format from the paper."""

    # Table 2a: Per-client Micro-F1
    print("\n" + "=" * 105)
    print("Table 2a: Per-Client Micro-F1 (%)")
    print("=" * 105)
    header = f"{'Method':<18}"
    for name in CLIENT_NAMES:
        header += f" | {name:>10}"
    header += f" | {'Global':>10}"
    print(header)
    print("-" * 105)

    for algo, data in results.items():
        display_name = DISPLAY_NAMES.get(algo, algo)
        if data is None:
            row = f"{display_name:<18}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{display_name:<18}"
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

    # Table 2b: Per-client Accuracy
    print("\n" + "=" * 105)
    print("Table 2b: Per-Client Accuracy (%)")
    print("=" * 105)
    print(header)
    print("-" * 105)

    for algo, data in results.items():
        display_name = DISPLAY_NAMES.get(algo, algo)
        if data is None:
            row = f"{display_name:<18}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{display_name:<18}"
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

    # Table 2c: Per-client mAP
    print("\n" + "=" * 105)
    print("Table 2c: Per-Client mAP (%)")
    print("=" * 105)
    print(header)
    print("-" * 105)

    for algo, data in results.items():
        display_name = DISPLAY_NAMES.get(algo, algo)
        if data is None:
            row = f"{display_name:<18}"
            for _ in range(5):
                row += f" | {'--':>10}"
            print(row)
            continue

        row = f"{display_name:<18}"
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

    print("=" * 105)


def print_cross_evaluation(results: dict):
    """Print cross-evaluation matrices for each algorithm."""

    # Print matrices for each metric type
    for metric_name, metric_label in [('accuracy', 'Accuracy'), ('micro_f1', 'Micro-F1'), ('mAP', 'mAP')]:
        print("\n" + "=" * 100)
        print(f"Cross-Evaluation {metric_label} Matrices (%)")
        print("(Row = Training Client, Column = Test Client)")
        print("=" * 100)

        for algo, data in results.items():
            if data is None or data.get('cross_eval') is None:
                continue

            cross_eval = data['cross_eval']

            # Check if this metric exists in cross_eval
            if metric_name not in cross_eval:
                continue

            metric_matrix = cross_eval[metric_name]

            # Check if we have any data for this metric
            has_data = any(
                metric_matrix[i][j] is not None
                for i in range(4)
                for j in range(4)
            )

            if not has_data:
                continue

            print(f"\n{algo.upper()}:")
            print("-" * 60)

            # Header
            header = f"{'Train\\Test':<12}"
            for name in CLIENT_NAMES:
                header += f" | {name:>10}"
            print(header)
            print("-" * 60)

            # Rows
            for i in range(4):
                row = f"{CLIENT_NAMES[i]:<12}"
                for j in range(4):
                    val = metric_matrix[i][j]
                    if val is not None:
                        row += f" | {val*100:>10.2f}"
                    else:
                        row += f" | {'--':>10}"
                print(row)

            print("-" * 60)


def print_summary(results: dict):
    """Print a summary with best methods."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find best FL method (excluding centralized and local)
    fl_results = {k: v for k, v in results.items()
                  if v is not None and k not in ['centralized', 'local'] and v.get('global')}

    if fl_results:
        best_f1 = max(fl_results.items(), key=lambda x: x[1]['global']['micro_f1'])
        best_acc = max(fl_results.items(), key=lambda x: x[1]['global']['acc'])
        best_map = max(fl_results.items(), key=lambda x: x[1]['global']['mAP'])

        print("\nBest FL Methods (Global):")
        print(f"  Micro-F1:  {best_f1[0]} ({best_f1[1]['global']['micro_f1']*100:.2f}%)")
        print(f"  Accuracy:  {best_acc[0]} ({best_acc[1]['global']['acc']*100:.2f}%)")
        print(f"  mAP:       {best_map[0]} ({best_map[1]['global']['mAP']*100:.2f}%)")

    # Count successful runs
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"\nCompleted: {success_count}/{len(results)} algorithms")


def save_to_csv(results: dict, csv_path: str):
    """Save results to CSV file."""
    import csv

    fieldnames = ['algorithm']
    for name in CLIENT_NAMES:
        fieldnames.extend([f'{name}_acc', f'{name}_micro_f1', f'{name}_mAP'])
    fieldnames.extend(['global_acc', 'global_micro_f1', 'global_mAP'])

    # Add cross-eval columns for each metric
    for metric in ['acc', 'f1', 'mAP']:
        for i, train_name in enumerate(CLIENT_NAMES):
            for j, test_name in enumerate(CLIENT_NAMES):
                fieldnames.append(f'cross_{metric}_{train_name}_to_{test_name}')

    fieldnames.append('run_dir')

    rows = []
    for algo, data in results.items():
        row = {'algorithm': algo}

        if data is None:
            for name in CLIENT_NAMES:
                row[f'{name}_acc'] = ''
                row[f'{name}_micro_f1'] = ''
                row[f'{name}_mAP'] = ''
            row['global_acc'] = ''
            row['global_micro_f1'] = ''
            row['global_mAP'] = ''
            for metric in ['acc', 'f1', 'mAP']:
                for i, train_name in enumerate(CLIENT_NAMES):
                    for j, test_name in enumerate(CLIENT_NAMES):
                        row[f'cross_{metric}_{train_name}_to_{test_name}'] = ''
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

            # Cross-evaluation (new structure with accuracy, micro_f1, mAP)
            cross_eval = data.get('cross_eval')
            metric_map = {'acc': 'accuracy', 'f1': 'micro_f1', 'mAP': 'mAP'}
            for metric_short, metric_key in metric_map.items():
                for i, train_name in enumerate(CLIENT_NAMES):
                    for j, test_name in enumerate(CLIENT_NAMES):
                        val = None
                        if cross_eval and metric_key in cross_eval:
                            if i in cross_eval[metric_key] and j in cross_eval[metric_key][i]:
                                val = cross_eval[metric_key][i][j]
                        if val is not None:
                            row[f'cross_{metric_short}_{train_name}_to_{test_name}'] = f"{val*100:.2f}"
                        else:
                            row[f'cross_{metric_short}_{train_name}_to_{test_name}'] = ''

            row['run_dir'] = data.get('run_dir', '')

        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {csv_path}")


def print_latex_tables(results: dict):
    """Print LaTeX-formatted tables for paper."""
    print("\n" + "=" * 100)
    print("LaTeX Table Format (Micro-F1)")
    print("=" * 100)
    print("\\begin{tabular}{l|cccc|c}")
    print("\\hline")
    print("Method & SPH & PTB-XL & SXPH & G12EC & Global \\\\")
    print("\\hline")

    for algo, data in results.items():
        if data is None:
            print(f"{algo} & -- & -- & -- & -- & -- \\\\")
            continue

        row = algo
        for i in range(4):
            if i in data['clients']:
                row += f" & {data['clients'][i]['micro_f1']*100:.2f}"
            else:
                row += " & --"

        if data['global']:
            row += f" & {data['global']['micro_f1']*100:.2f}"
        else:
            row += " & --"

        print(row + " \\\\")

    print("\\hline")
    print("\\end{tabular}")

    print("\n" + "=" * 100)
    print("LaTeX Table Format (mAP)")
    print("=" * 100)
    print("\\begin{tabular}{l|cccc|c}")
    print("\\hline")
    print("Method & SPH & PTB-XL & SXPH & G12EC & Global \\\\")
    print("\\hline")

    for algo, data in results.items():
        if data is None:
            print(f"{algo} & -- & -- & -- & -- & -- \\\\")
            continue

        row = algo
        for i in range(4):
            if i in data['clients']:
                row += f" & {data['clients'][i]['mAP']*100:.2f}"
            else:
                row += " & --"

        if data['global']:
            row += f" & {data['global']['mAP']*100:.2f}"
        else:
            row += " & --"

        print(row + " \\\\")

    print("\\hline")
    print("\\end{tabular}")

    # Cross-evaluation LaTeX table for one algorithm (example)
    print("\n" + "=" * 100)
    print("LaTeX Table Format (Cross-Evaluation Accuracy)")
    print("=" * 100)
    print("% Example for one algorithm - repeat for each")
    print("\\begin{tabular}{l|cccc}")
    print("\\hline")
    print("Train $\\backslash$ Test & SPH & PTB-XL & SXPH & G12EC \\\\")
    print("\\hline")

    # Find first algorithm with cross-eval data
    for algo, data in results.items():
        if data is None or data.get('cross_eval') is None:
            continue
        cross_eval = data['cross_eval']
        has_data = any(cross_eval[i][j] is not None for i in range(4) for j in range(4))
        if not has_data:
            continue

        print(f"% {algo}")
        for i in range(4):
            row = CLIENT_NAMES[i]
            for j in range(4):
                val = cross_eval[i][j]
                if val is not None:
                    row += f" & {val*100:.2f}"
                else:
                    row += " & --"
            print(row + " \\\\")
        break

    print("\\hline")
    print("\\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Extract ECG benchmark metrics (Table 2 format)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to output directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX-formatted tables")
    parser.add_argument("--cross-eval", action="store_true",
                        help="Print cross-evaluation matrices")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    # Algorithms from Table 2 (excluding FedBN and FedFA per user request)
    algorithms = [
        "centralized",
        "local_client1",  # Client-only baselines (separate rows per paper format)
        "local_client2",
        "local_client3",
        "local_client4",
        "fedavg",
        "fedprox",
        "scaffold",
        "ditto",
        "fedinit",
        "fedala",
        "fedsm",
        "feddualatt",  # Our proposed method
    ]

    print("=" * 100)
    print("Fed-ECG Benchmark Results (Table 2 Format)")
    print("=" * 100)
    print(f"Output directory: {output_path}")
    print(f"Client mapping: 0=SPH, 1=PTB-XL, 2=SXPH, 3=G12EC")
    print("-" * 100)

    # Collect results
    results = {}
    for algo in algorithms:
        run_dir = find_latest_run(output_path, algo)
        if run_dir is None:
            print(f"[!] {algo}: Not found")
            results[algo] = None
            continue

        metrics = extract_per_client_metrics(run_dir, algo)
        if metrics is None:
            print(f"[!] {algo}: No metrics found")
            results[algo] = None
            continue

        print(f"[OK] {algo}: Round {metrics.get('round', 'N/A')}")
        results[algo] = metrics

    # Print tables
    print_table_2_format(results)

    # Print cross-evaluation if requested or by default
    if args.cross_eval or True:  # Always print cross-eval
        print_cross_evaluation(results)

    print_summary(results)

    # Save to CSV if requested
    if args.csv:
        save_to_csv(results, args.csv)

    # Print LaTeX if requested
    if args.latex:
        print_latex_tables(results)


if __name__ == "__main__":
    main()
