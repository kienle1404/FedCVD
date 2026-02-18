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
import re
import argparse
import numpy as np
from pathlib import Path


# Get project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output"

# Client names matching the paper
CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]

# Algorithms where LOCAL metrics should use the personal/personalized model
# (from client-side metric files) instead of the server's global model evaluation.
# For these methods, the paper evaluates each client's personalized model on its own test set.
PERSONALIZED_ALGORITHMS = {"ditto", "fedala"}

# Display names for algorithms (maps internal names to paper format)
DISPLAY_NAMES = {
    "centralized": "Centralized",
    "local_client1": "Client1",
    "local_client2": "Client2",
    "local_client3": "Client3",
    "local_client4": "Client4",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "Scaffold",
    "ditto": "Ditto",
    "fedinit": "FedInit",
    "fedala": "FedALA",
    "fedsm": "FedSM",
    "feddualatt": "FedDualAtt",
    "feddualatt_8g0l": "FedDualAtt (8G:0L)",
    "feddualatt_7g1l": "FedDualAtt (7G:1L)",
    "feddualatt_6g2l": "FedDualAtt (6G:2L)",
    "feddualatt_5g3l": "FedDualAtt (5G:3L)",
    "feddualatt_4g4l": "FedDualAtt (4G:4L)",
    "feddualatt_3g5l": "FedDualAtt (3G:5L)",
    "feddualatt_2g6l": "FedDualAtt (2G:6L)",
    "feddualatt_1g7l": "FedDualAtt (1G:7L)",
    "feddualatt_0g8l": "FedDualAtt (0G:8L)",
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


def find_head_ratio_runs(base_path: Path) -> dict:
    """
    Find all head ratio experiment runs organised by ratio and seed.

    Returns:
        {"8-0": {42: Path, 456: Path, ...}, "7-1": {...}, ...}
    """
    search_path = base_path / "dual_attention_resnet1d" / "feddualatt"
    if not search_path.exists():
        return {}

    ratio_pattern = re.compile(r'global(\d+)_local(\d+)')
    seed_pattern = re.compile(r'seed(\d+)')
    runs = {}

    for ratio_dir in search_path.iterdir():
        if not ratio_dir.is_dir():
            continue
        m = ratio_pattern.match(ratio_dir.name)
        if not m:
            continue

        g, l = int(m.group(1)), int(m.group(2))
        key = f"{g}-{l}"
        runs[key] = {}

        for seed_dir in ratio_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            sm = seed_pattern.match(seed_dir.name)
            if sm:
                seed = int(sm.group(1))
                ts_dirs = [d for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if ts_dirs:
                    runs[key][seed] = max(ts_dirs, key=lambda d: d.name)
            elif seed_dir.name.isdigit():
                # Legacy: timestamp directly under ratio dir
                runs[key][42] = seed_dir

    return {k: v for k, v in runs.items() if v}


def aggregate_seed_metrics(seed_results: dict) -> dict | None:
    """
    Aggregate per-seed metrics into mean ± std.

    Args:
        seed_results: {seed: metrics_dict, ...}  (metrics_dict from extract_metrics helper)

    Returns dict with 'clients', 'clients_std', 'global', 'global_std', 'num_seeds', 'seeds'.
    """
    valid = {k: v for k, v in seed_results.items() if v is not None}
    if not valid:
        return None

    result = {
        'num_seeds': len(valid),
        'seeds': sorted(valid.keys()),
        'clients': {},
        'clients_std': {},
        'global': None,
        'global_std': None,
        'cross_eval': None,
        'run_dir': '',
        'round': None,
    }

    for ci in range(4):
        vals = {m: [] for m in ('acc', 'micro_f1', 'mAP')}
        for mets in valid.values():
            if ci in mets.get('clients', {}):
                for m in vals:
                    vals[m].append(mets['clients'][ci][m])
        if vals['micro_f1']:
            result['clients'][ci] = {m: float(np.mean(vals[m])) for m in vals}
            result['clients_std'][ci] = {m: float(np.std(vals[m])) for m in vals}

    g_vals = {m: [] for m in ('acc', 'micro_f1', 'mAP')}
    for mets in valid.values():
        if mets.get('global'):
            for m in g_vals:
                g_vals[m].append(mets['global'][m])
    if g_vals['micro_f1']:
        result['global'] = {m: float(np.mean(g_vals[m])) for m in g_vals}
        result['global_std'] = {m: float(np.std(g_vals[m])) for m in g_vals}

    return result


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
        # Prefer corrected metrics (produced by reevaluate_checkpoints.py) when available.
        # For FedDualAtt head ratio runs the original metric.json evaluated with zeroed
        # local params (Bug #2); metric_corrected.json loads per-client local heads first.
        _mc = run_dir / "server" / "metric_corrected.json"
        metric_file = _mc if _mc.exists() else run_dir / "server" / "metric.json"

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

        # For personalized algorithms, override LOCAL per-client metrics with
        # the diagonal of the cross-eval matrix (personal model on own data).
        # The server evaluates the global model, but the paper reports the
        # personalized model's performance for these methods.
        if algorithm in PERSONALIZED_ALGORITHMS and cross_eval:
            for i in range(4):
                acc = cross_eval['accuracy'][i][i]
                micro_f1 = cross_eval['micro_f1'][i][i]
                mAP = cross_eval['mAP'][i][i]
                if micro_f1 is not None:
                    result['clients'][i] = {
                        'acc': acc if acc is not None else 0,
                        'micro_f1': micro_f1,
                        'mAP': mAP if mAP is not None else 0,
                    }

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
    """Print cross-evaluation matrices for local training only.

    Cross-evaluation only makes sense for local (client-only) training where
    each client trains its own model independently. For FL methods, there's
    only one global model, so cross-evaluation is not meaningful.
    """

    # Print matrices for each metric type
    for metric_name, metric_label in [('accuracy', 'Accuracy'), ('micro_f1', 'Micro-F1'), ('mAP', 'mAP')]:
        print("\n" + "=" * 100)
        print(f"Cross-Evaluation {metric_label} Matrices - Local Training Only (%)")
        print("(Row = Training Client, Column = Test Client)")
        print("=" * 100)

        for algo, data in results.items():
            # Only show cross-eval for local training (client-only baselines)
            # FL methods have one global model, so cross-eval doesn't apply
            if not algo.startswith("local_client"):
                continue

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

            display_name = DISPLAY_NAMES.get(algo, algo)
            print(f"\n{display_name}:")
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

    # Std columns (populated for multi-seed head ratio entries, empty for baselines)
    for name in CLIENT_NAMES:
        fieldnames.extend([f'{name}_acc_std', f'{name}_micro_f1_std', f'{name}_mAP_std'])
    fieldnames.extend(['global_acc_std', 'global_micro_f1_std', 'global_mAP_std'])
    fieldnames.append('num_seeds')

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
            for name in CLIENT_NAMES:
                row[f'{name}_acc_std'] = ''
                row[f'{name}_micro_f1_std'] = ''
                row[f'{name}_mAP_std'] = ''
            row['global_acc_std'] = ''
            row['global_micro_f1_std'] = ''
            row['global_mAP_std'] = ''
            row['num_seeds'] = ''
            for metric in ['acc', 'f1', 'mAP']:
                for i, train_name in enumerate(CLIENT_NAMES):
                    for j, test_name in enumerate(CLIENT_NAMES):
                        row[f'cross_{metric}_{train_name}_to_{test_name}'] = ''
            row['run_dir'] = ''
        else:
            clients_std = data.get('clients_std', {})
            global_std = data.get('global_std') or {}

            for i, name in enumerate(CLIENT_NAMES):
                if i in data['clients']:
                    row[f'{name}_acc'] = f"{data['clients'][i]['acc']*100:.2f}"
                    row[f'{name}_micro_f1'] = f"{data['clients'][i]['micro_f1']*100:.2f}"
                    row[f'{name}_mAP'] = f"{data['clients'][i]['mAP']*100:.2f}"
                    std = clients_std.get(i, {})
                    row[f'{name}_acc_std'] = f"{std['acc']*100:.2f}" if std else ''
                    row[f'{name}_micro_f1_std'] = f"{std['micro_f1']*100:.2f}" if std else ''
                    row[f'{name}_mAP_std'] = f"{std['mAP']*100:.2f}" if std else ''
                else:
                    row[f'{name}_acc'] = ''
                    row[f'{name}_micro_f1'] = ''
                    row[f'{name}_mAP'] = ''
                    row[f'{name}_acc_std'] = ''
                    row[f'{name}_micro_f1_std'] = ''
                    row[f'{name}_mAP_std'] = ''

            if data['global']:
                row['global_acc'] = f"{data['global']['acc']*100:.2f}"
                row['global_micro_f1'] = f"{data['global']['micro_f1']*100:.2f}"
                row['global_mAP'] = f"{data['global']['mAP']*100:.2f}"
                row['global_acc_std'] = f"{global_std['acc']*100:.2f}" if global_std else ''
                row['global_micro_f1_std'] = f"{global_std['micro_f1']*100:.2f}" if global_std else ''
                row['global_mAP_std'] = f"{global_std['mAP']*100:.2f}" if global_std else ''
            else:
                row['global_acc'] = ''
                row['global_micro_f1'] = ''
                row['global_mAP'] = ''
                row['global_acc_std'] = ''
                row['global_micro_f1_std'] = ''
                row['global_mAP_std'] = ''

            row['num_seeds'] = data.get('num_seeds', '')

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


def _latex_cell(data: dict, client_idx_or_none, metric: str) -> str:
    """
    Format a single LaTeX cell value.
    - For entries with std (head ratio), produces  mean$\\pm$std.
    - For plain entries, produces the mean only.
    - client_idx_or_none=None means global metric.
    """
    if client_idx_or_none is None:
        src = data.get('global')
        std_src = data.get('global_std') or {}
    else:
        src = data['clients'].get(client_idx_or_none)
        std_src = (data.get('clients_std') or {}).get(client_idx_or_none, {})

    if src is None:
        return "--"

    val = src[metric] * 100
    std = std_src.get(metric, 0) * 100 if std_src else 0
    if std > 0:
        return f"{val:.2f}$\\pm${std:.2f}"
    return f"{val:.2f}"


def print_latex_tables(results: dict):
    """Print LaTeX-formatted tables for paper."""

    for metric, label in [('micro_f1', 'Micro-F1'), ('mAP', 'mAP'), ('acc', 'Accuracy')]:
        print("\n" + "=" * 100)
        print(f"LaTeX Table Format ({label})")
        print("=" * 100)
        print("\\begin{tabular}{l|cccc|c}")
        print("\\hline")
        print("Method & SPH & PTB-XL & SXPH & G12EC & Global \\\\")
        print("\\hline")

        for algo, data in results.items():
            display = DISPLAY_NAMES.get(algo, algo)
            if data is None:
                print(f"{display} & -- & -- & -- & -- & -- \\\\")
                continue

            row = display
            for i in range(4):
                row += f" & {_latex_cell(data, i, metric)}"
            row += f" & {_latex_cell(data, None, metric)}"
            print(row + " \\\\")

        print("\\hline")
        print("\\end{tabular}")

    # Cross-evaluation LaTeX table (local clients only)
    print("\n" + "=" * 100)
    print("LaTeX Table Format (Cross-Evaluation Accuracy)")
    print("=" * 100)
    print("% One block per local client — repeat pattern for other algorithms")
    print("\\begin{tabular}{l|cccc}")
    print("\\hline")
    print("Train $\\backslash$ Test & SPH & PTB-XL & SXPH & G12EC \\\\")
    print("\\hline")

    for algo, data in results.items():
        if data is None or data.get('cross_eval') is None:
            continue
        cross_eval = data['cross_eval']
        acc_matrix = cross_eval.get('accuracy', {})
        has_data = any(
            acc_matrix.get(i, {}).get(j) is not None
            for i in range(4) for j in range(4)
        )
        if not has_data:
            continue

        display = DISPLAY_NAMES.get(algo, algo)
        print(f"% {display}")
        for i in range(4):
            row = CLIENT_NAMES[i]
            for j in range(4):
                val = acc_matrix.get(i, {}).get(j)
                row += f" & {val*100:.2f}" if val is not None else " & --"
            print(row + " \\\\")
        break

    print("\\hline")
    print("\\end{tabular}")


def save_latex_file(results: dict, tex_path: str):
    """Write a self-contained, compilable LaTeX document with all benchmark tables."""

    # Groups: draw \midrule before first algorithm of each new group
    GROUP_BREAKS = {"fedavg", "feddualatt_8g0l"}

    def get_val(data, client_idx_or_none, metric):
        """Return the raw (0-1) mean value for a cell, or None."""
        if client_idx_or_none is None:
            src = data.get('global')
        else:
            src = data['clients'].get(client_idx_or_none)
        return src[metric] if src else None

    def doc_cell(data, client_idx_or_none, metric, best_val=None):
        """Format one table cell; bold if value equals best_val."""
        val = get_val(data, client_idx_or_none, metric)
        if val is None:
            return '--'
        if client_idx_or_none is None:
            std_src = data.get('global_std') or {}
        else:
            std_src = (data.get('clients_std') or {}).get(client_idx_or_none, {})
        std = (std_src.get(metric, 0) * 100) if std_src else 0
        val_pct = val * 100
        cell = f'${val_pct:.2f} \\pm {std:.2f}$' if std > 0 else f'{val_pct:.2f}'
        if best_val is not None and abs(val - best_val) < 1e-9:
            cell = f'\\textbf{{{cell}}}'
        return cell

    lines = [
        r'\documentclass[a4paper]{article}',
        r'\usepackage{booktabs}',
        r'\usepackage{geometry}',
        r'\geometry{a4paper, landscape, left=1.5cm, right=1.5cm, top=2cm, bottom=2cm}',
        r'',
        r'\begin{document}',
        r'',
    ]

    for metric, label in [('micro_f1', 'Micro-F1'), ('acc', 'Accuracy'), ('mAP', 'mAP')]:
        # First pass: find best value per column across all rows
        best = {}
        for col in list(range(4)) + [None]:
            vals = [get_val(d, col, metric) for d in results.values()
                    if d is not None and get_val(d, col, metric) is not None]
            best[col] = max(vals) if vals else None

        lines += [
            r'\begin{table}[htbp]',
            r'  \centering',
            f'  \\caption{{Per-Client {label} (\\%) on ECG Benchmark}}',
            r'  \begin{tabular}{l|cccc|c}',
            r'    \toprule',
            '    Method & SPH & PTB-XL & SXPH & G12EC & Global \\\\',
            r'    \midrule',
        ]

        for algo, data in results.items():
            if algo in GROUP_BREAKS:
                lines.append(r'    \midrule')

            display = DISPLAY_NAMES.get(algo, algo)
            display = display.replace('_', r'\_')

            if data is None:
                lines.append(f'    {display} & -- & -- & -- & -- & -- \\\\')
                continue

            cells = [display]
            for i in range(4):
                cells.append(doc_cell(data, i, metric, best[i]))
            cells.append(doc_cell(data, None, metric, best[None]))
            lines.append('    ' + ' & '.join(cells) + r' \\')

        lines += [
            r'    \bottomrule',
            r'  \end{tabular}',
            r'\end{table}',
            r'\clearpage',
            r'',
        ]

    lines.append(r'\end{document}')
    lines.append('')

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nLaTeX file saved to: {tex_path}')


def save_combined_latex_file(results: dict, tex_path: str):
    """Write a single wide table (all 3 metrics per client/global) as a compilable .tex document.

    Bold = best FL method per column; underline = second-best.
    Centralized and local clients are shown but excluded from ranking.
    """

    NON_FL = {'centralized', 'local_client1', 'local_client2', 'local_client3', 'local_client4'}
    GROUP_BREAKS = {'fedavg', 'feddualatt_8g0l'}
    METRICS = ['micro_f1', 'acc', 'mAP']
    METRIC_LABELS = ['F1', 'Acc', 'mAP']
    CLIENT_COLS = list(range(4))
    CLIENT_GROUP_LABELS = ['SPH', 'PTB-XL', 'SXPH', 'G12EC']

    def get_val(data, col, metric):
        src = data.get('global') if col is None else data['clients'].get(col)
        return src[metric] if src else None

    def get_std(data, col, metric):
        src = data.get('global_std') if col is None else (data.get('clients_std') or {}).get(col, {})
        return (src or {}).get(metric, 0)

    # Find best & second-best among FL methods only for each (col, metric)
    fl_algos = [k for k in results if k not in NON_FL and results[k] is not None]
    ranks = {}
    for col in CLIENT_COLS + [None]:
        for metric in METRICS:
            fl_vals = [get_val(results[a], col, metric) for a in fl_algos]
            fl_vals = [v for v in fl_vals if v is not None]
            unique_sorted = sorted(set(fl_vals), reverse=True)
            ranks[(col, metric)] = (
                unique_sorted[0] if unique_sorted else None,
                unique_sorted[1] if len(unique_sorted) > 1 else None,
            )

    def fmt_cell(data, col, metric, is_fl):
        val = get_val(data, col, metric)
        if val is None:
            return '--'
        std = get_std(data, col, metric)
        val_pct, std_pct = val * 100, std * 100
        cell = f'${val_pct:.2f} \\pm {std_pct:.2f}$' if std_pct > 0 else f'{val_pct:.2f}'
        if is_fl:
            best, second = ranks[(col, metric)]
            if best is not None and abs(val - best) < 1e-9:
                cell = f'\\textbf{{{cell}}}'
            elif second is not None and abs(val - second) < 1e-9:
                cell = f'\\underline{{{cell}}}'
        return cell

    # 16-column table: Method + 5 groups × 3 metrics
    col_spec = 'l|' + '|'.join(['ccc'] * 5)

    lines = [
        r'\documentclass[a4paper]{article}',
        r'\usepackage{booktabs}',
        r'\usepackage{multirow}',
        r'\usepackage{adjustbox}',
        r'\usepackage{geometry}',
        r'\geometry{a4paper, landscape, left=1cm, right=1cm, top=1.5cm, bottom=1.5cm}',
        r'',
        r'\begin{document}',
        r'',
        r'\begin{table}[htbp]',
        r'  \centering',
        r'  \caption{ECG Benchmark: Per-Client and Global Metrics (\%)}',
        r'  \begin{adjustbox}{max width=\linewidth}',
        f'  \\begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
    ]

    # Header row 1: group labels spanning 3 columns each
    h1_cells = [r'    \multirow{2}{*}{Method}']
    for lbl in CLIENT_GROUP_LABELS:
        h1_cells.append(f'\\multicolumn{{3}}{{c|}}{{{lbl}}}')
    h1_cells.append(r'\multicolumn{3}{c}{Global}')
    lines.append(' & '.join(h1_cells) + r' \\')

    # Header row 2: individual metric labels (15 columns)
    lines.append('    & ' + ' & '.join(METRIC_LABELS * 5) + r' \\')
    lines.append(r'    \midrule')

    # Data rows
    for algo, data in results.items():
        if algo in GROUP_BREAKS:
            lines.append(r'    \midrule')

        display = DISPLAY_NAMES.get(algo, algo).replace('_', r'\_')
        is_fl = algo not in NON_FL

        if data is None:
            lines.append(f'    {display} & ' + ' & '.join(['--'] * 15) + r' \\')
            continue

        cells = [display]
        for col in CLIENT_COLS + [None]:
            for metric in METRICS:
                cells.append(fmt_cell(data, col, metric, is_fl))
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines += [
        r'    \bottomrule',
        f'  \\end{{tabular}}',
        r'  \end{adjustbox}',
        r'\end{table}',
        r'',
        r'\end{document}',
        r'',
    ]

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nCombined LaTeX table saved to: {tex_path}')


def main():
    parser = argparse.ArgumentParser(description="Extract ECG benchmark metrics (Table 2 format)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to output directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX-formatted tables to console")
    parser.add_argument("--latex-file", type=str, default=None,
                        help="Save compilable LaTeX document to .tex file")
    parser.add_argument("--latex-combined", type=str, default=None,
                        help="Save single combined-metric LaTeX table to .tex file")
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

    # Collect baseline results
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

    # Collect head ratio results (multi-seed, aggregated)
    head_ratio_runs = find_head_ratio_runs(output_path)
    if head_ratio_runs:
        print(f"\nFound {len(head_ratio_runs)} head ratio configuration(s):")
        for ratio_key in sorted(head_ratio_runs.keys(), key=lambda x: int(x.split('-')[0]), reverse=True):
            g, l = ratio_key.split('-')
            algo_key = f"feddualatt_{g}g{l}l"
            seed_runs = head_ratio_runs[ratio_key]

            # Extract per-seed metrics
            seed_metrics = {}
            for seed, run_dir in seed_runs.items():
                # Reuse extract_per_client_metrics but via the head-ratio metric file
                metric_corrected = run_dir / "server" / "metric_corrected.json"
                metric_file = metric_corrected if metric_corrected.exists() else run_dir / "server" / "metric.json"
                if metric_file.exists():
                    seed_metrics[seed] = extract_per_client_metrics(run_dir, "feddualatt")
                else:
                    seed_metrics[seed] = None

            aggregated = aggregate_seed_metrics(seed_metrics)
            if aggregated:
                print(f"  [OK] {algo_key}: {aggregated['num_seeds']} seeds {aggregated['seeds']}")
            else:
                print(f"  [!] {algo_key}: No valid metrics found")
            results[algo_key] = aggregated
    else:
        print("\n[!] No head ratio experiments found.")

    # Print tables
    print_table_2_format(results)

    # Print cross-evaluation if requested or by default
    if args.cross_eval or True:  # Always print cross-eval
        print_cross_evaluation(results)

    print_summary(results)

    # Save to CSV if requested
    if args.csv:
        save_to_csv(results, args.csv)

    # Print LaTeX to console if requested
    if args.latex:
        print_latex_tables(results)

    # Save compilable LaTeX file if requested
    if args.latex_file:
        save_latex_file(results, args.latex_file)

    # Save single combined-metric LaTeX table if requested
    if args.latex_combined:
        save_combined_latex_file(results, args.latex_combined)


if __name__ == "__main__":
    main()
