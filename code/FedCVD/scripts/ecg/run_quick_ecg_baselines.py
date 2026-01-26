#!/usr/bin/env python3
"""
Quick benchmark script for ECG baselines - runs ~100x faster than full benchmark.
Results are stored separately in output/quick_benchmark/ to avoid mixing with main results.

Defaults:
- 5 communication rounds (vs 50 in full)
- 10% of data (vs 100% in full)
- 10 epochs for centralized (vs 50 in full)

Usage:
    python run_quick_ecg_baselines.py
    python run_quick_ecg_baselines.py --algorithms fedavg,scaffold
    python run_quick_ecg_baselines.py --data-fraction 0.2
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Disable wandb prompts
os.environ["WANDB_MODE"] = "disabled"

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
TRAINERS_DIR = SCRIPT_DIR.parent.parent / "trainers"
INPUT_PATH = PROJECT_ROOT / "data"
# Use separate output directory for quick benchmarks
OUTPUT_PATH = PROJECT_ROOT / "output" / "quick_benchmark"

# Common parameters
COMMON_PARAMS = {
    "batch_size": 32,
    "lr": 0.1,
    "seed": 42,
}

# Quick benchmark configurations (reduced from full benchmark)
ALGORITHMS = {
    "centralized": {
        "trainer": "centralized_ecg.py",
        "params": {
            "max_epoch": 10,  # Reduced from 50
        },
        "description": "Centralized training (upper bound)",
    },
    "local": {
        "trainer": "local_ecg.py",
        "params": {
            "max_epoch": 10,  # Reduced from 50
        },
        "description": "Local training only (no FL)",
    },
    "fedavg": {
        "trainer": "fedavg_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,  # Reduced from 50
        },
        "description": "FedAvg - McMahan et al. 2017",
    },
    "fedprox": {
        "trainer": "fedprox_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
            "mu": 0.01,
        },
        "description": "FedProx - Li et al. 2020",
    },
    "scaffold": {
        "trainer": "scaffold_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
            "server_lr": 1.0,
        },
        "description": "SCAFFOLD - Karimireddy et al. 2020",
    },
    "ditto": {
        "trainer": "ditto_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
            "mu": 0.01,
        },
        "description": "Ditto - Li et al. 2021",
    },
    "fedbn": {
        "trainer": "fedbn_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedBN - Li et al. 2021",
    },
    "fedinit": {
        "trainer": "fedinit_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedInit",
    },
    "fedala": {
        "trainer": "fedala_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedALA",
    },
    "fedfa": {
        "trainer": "fedfa_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedFA",
    },
    "fedsm": {
        "trainer": "fedsm_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedSM",
    },
    "feddualatt": {
        "trainer": "feddualatt_ecg.py",
        "params": {
            "max_epoch": 1,
            "communication_round": 5,
        },
        "description": "FedDualAtt - Dual Attention Personalization (Ours)",
    },
}


def get_latest_output_dir(algorithm: str) -> Path | None:
    """Find the latest output directory for an algorithm."""
    if algorithm == "centralized":
        search_path = OUTPUT_PATH / "resnet1d34" / "centralized"
    elif algorithm == "local":
        search_path = OUTPUT_PATH / "resnet1d34" / "local"
    elif algorithm == "feddualatt":
        search_path = OUTPUT_PATH / "dual_attention_resnet1d" / "feddualatt"
    else:
        search_path = OUTPUT_PATH / "resnet1d34" / algorithm

    if not search_path.exists():
        return None

    timestamp_dirs = [d for d in search_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not timestamp_dirs:
        return None

    return max(timestamp_dirs, key=lambda d: d.name)


def extract_metrics(output_dir: Path) -> dict | None:
    """Extract metrics from an algorithm's output directory."""
    server_metric = output_dir / "server" / "metric.json"
    direct_metric = output_dir / "metric.json"

    metric_file = server_metric if server_metric.exists() else direct_metric

    if not metric_file.exists():
        return None

    try:
        with open(metric_file, 'r') as f:
            metrics = json.load(f)

        if isinstance(metrics, list):
            last_round = metrics[-1]
        else:
            last_round = metrics

        result = {
            "round": last_round.get("round", "N/A"),
        }

        global_metrics = last_round.get("global", last_round.get("Global", {}))
        if global_metrics:
            result["accuracy"] = global_metrics.get("acc", global_metrics.get("Acc", "N/A"))
            result["micro_f1"] = global_metrics.get("micro_f1", global_metrics.get("Micro_F1", "N/A"))
            result["map"] = global_metrics.get("mAP", "N/A")

        return result
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return None


def run_algorithm(algorithm: str, config: dict, data_fraction: float, skip_existing: bool = False) -> dict | None:
    """Run a single algorithm and return its results."""
    print(f"\n{'='*60}")
    print(f"[QUICK] Running: {algorithm.upper()}")
    print(f"Description: {config['description']}")
    print(f"Data fraction: {data_fraction*100:.0f}%")
    print(f"{'='*60}")

    if skip_existing:
        existing_dir = get_latest_output_dir(algorithm)
        if existing_dir:
            print(f"Skipping (output exists): {existing_dir}")
            metrics = extract_metrics(existing_dir)
            if metrics:
                return {"status": "skipped", "metrics": metrics, "output_dir": str(existing_dir)}
            return {"status": "skipped_no_metrics", "output_dir": str(existing_dir)}

    trainer_path = TRAINERS_DIR / config["trainer"]
    if not trainer_path.exists():
        print(f"ERROR: Trainer not found: {trainer_path}")
        return {"status": "error", "error": "Trainer not found"}

    cmd = [sys.executable, str(trainer_path)]

    # Add common parameters
    for key, value in COMMON_PARAMS.items():
        cmd.extend([f"--{key}", str(value)])

    # Add algorithm-specific parameters
    for key, value in config["params"].items():
        cmd.extend([f"--{key}", str(value)])

    # Add paths
    cmd.extend(["--input_path", str(INPUT_PATH)])
    cmd.extend(["--output_path", str(OUTPUT_PATH)])

    # Add data fraction
    cmd.extend(["--data_fraction", str(data_fraction)])

    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = datetime.now()
    try:
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"

        result = subprocess.run(
            cmd,
            cwd=str(TRAINERS_DIR),
            capture_output=False,
            env=env,
        )

        end_time = datetime.now()
        duration = end_time - start_time

        if result.returncode != 0:
            print(f"ERROR: Algorithm failed with return code {result.returncode}")
            return {"status": "failed", "return_code": result.returncode, "duration": str(duration)}

        output_dir = get_latest_output_dir(algorithm)
        if output_dir:
            metrics = extract_metrics(output_dir)
            return {
                "status": "success",
                "metrics": metrics,
                "output_dir": str(output_dir),
                "duration": str(duration),
            }
        else:
            return {"status": "success_no_output", "duration": str(duration)}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "error", "error": str(e)}


def print_summary(results: dict):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("QUICK BENCHMARK RESULTS SUMMARY")
    print("="*80)

    print(f"{'Algorithm':<15} {'Status':<12} {'Accuracy':<12} {'Micro-F1':<12} {'mAP':<12}")
    print("-"*80)

    for alg, result in results.items():
        status = result.get("status", "unknown")
        metrics = result.get("metrics", {})

        if metrics:
            acc = metrics.get("accuracy", "N/A")
            f1 = metrics.get("micro_f1", "N/A")
            mAP = metrics.get("map", "N/A")

            if isinstance(acc, (int, float)):
                acc = f"{acc*100:.2f}%" if acc <= 1 else f"{acc:.2f}%"
            if isinstance(f1, (int, float)):
                f1 = f"{f1*100:.2f}%" if f1 <= 1 else f"{f1:.2f}%"
            if isinstance(mAP, (int, float)):
                mAP = f"{mAP*100:.2f}%" if mAP <= 1 else f"{mAP:.2f}%"
        else:
            acc = f1 = mAP = "N/A"

        print(f"{alg:<15} {status:<12} {acc:<12} {f1:<12} {mAP:<12}")

    print("="*80)


def save_results(results: dict, output_file: Path):
    """Save results to a JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Quick ECG baseline benchmark (~100x faster)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip algorithms that already have output")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated list of algorithms to run (default: all)")
    parser.add_argument("--data-fraction", type=float, default=0.1,
                        help="Fraction of data to use (default: 0.1 = 10%%)")
    parser.add_argument("--list", action="store_true",
                        help="List available algorithms and exit")
    args = parser.parse_args()

    if args.list:
        print("Available algorithms:")
        for alg, config in ALGORITHMS.items():
            print(f"  {alg:<15} - {config['description']}")
        return

    if args.algorithms:
        selected = [a.strip().lower() for a in args.algorithms.split(",")]
        algorithms_to_run = {k: v for k, v in ALGORITHMS.items() if k in selected}

        invalid = set(selected) - set(ALGORITHMS.keys())
        if invalid:
            print(f"ERROR: Unknown algorithms: {invalid}")
            print("Use --list to see available algorithms")
            return
    else:
        algorithms_to_run = ALGORITHMS

    print("="*80)
    print("QUICK ECG Baseline Benchmark (~100x faster)")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Data fraction: {args.data_fraction*100:.0f}%")
    print(f"Algorithms to run: {list(algorithms_to_run.keys())}")
    print(f"Skip existing: {args.skip_existing}")
    print()

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    results = {}
    for alg, config in algorithms_to_run.items():
        results[alg] = run_algorithm(alg, config, args.data_fraction, args.skip_existing)

    print_summary(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_PATH / f"ecg_quick_results_{timestamp}.json"
    save_results(results, results_file)


if __name__ == "__main__":
    main()
