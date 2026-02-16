#!/usr/bin/env python3
"""
Run FedDualAtt experiments with different total head counts.

This script tests the saturation point where adding more heads no longer improves performance.
Uses the best split ratio (~62.5% global) for all configurations.

Usage:
    python run_head_count_experiment.py                    # All configs, single seed
    python run_head_count_experiment.py --num_seeds 5      # All configs, 5 seeds
    python run_head_count_experiment.py --config 8         # Single config (8 heads), single seed
    python run_head_count_experiment.py --config 16 --num_seeds 5  # Single config, 5 seeds
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
TRAINER_PATH = SCRIPT_DIR / "../../trainers/feddualatt_ecg.py"
DATA_PATH = SCRIPT_DIR / "../../../../data"
OUTPUT_PATH = SCRIPT_DIR / "../../../../output"

# Default seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Head count configurations: (num_heads, global_heads, local_heads)
# Using ~62.5% global ratio (rounded to nearest for non-divisible cases)
HEAD_CONFIGS = {
    4: (4, 2, 2),     # 4 total: 50% global (rounded from 62.5%)
    8: (8, 5, 3),     # 8 total: 62.5% global (baseline best)
    16: (16, 10, 6),  # 16 total: 62.5% global
    24: (24, 15, 9),  # 24 total: 62.5% global
    32: (32, 20, 12), # 32 total: 62.5% global
}


def run_single_experiment(num_heads: int, global_heads: int, seed: int, args) -> bool:
    """Run a single head count experiment. Returns True if successful."""
    local_heads = num_heads - global_heads

    print(f"\n{'=' * 70}")
    print(f"Running FedDualAtt: {num_heads} total heads ({global_heads} global, {local_heads} local) | Seed: {seed}")
    print(f"{'=' * 70}")
    print(f"Rounds:       {args.communication_round}")
    print(f"Data path:    {DATA_PATH}")
    print(f"Output path:  {OUTPUT_PATH}")
    print(f"Started at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    cmd = [
        sys.executable, str(TRAINER_PATH),
        "--input_path", str(DATA_PATH),
        "--output_path", str(OUTPUT_PATH),
        "--num_heads", str(num_heads),
        "--global_heads", str(global_heads),
        "--communication_round", str(args.communication_round),
        "--max_epoch", str(args.max_epoch),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(seed),
        "--data_fraction", str(args.data_fraction),
        "--model", "dual_attention_resnet1d",
        "--case_name", f"feddualatt_h{num_heads}_g{global_heads}_l{local_heads}",
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run FedDualAtt with different total head counts (saturation analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All head count configurations, single seed
  python run_head_count_experiment.py

  # All configurations, 5 seeds for statistical significance
  python run_head_count_experiment.py --num_seeds 5

  # Single configuration (16 heads), single seed
  python run_head_count_experiment.py --config 16

  # Single configuration, 5 seeds
  python run_head_count_experiment.py --config 16 --num_seeds 5

  # Quick test with fewer rounds
  python run_head_count_experiment.py --communication_round 5 --data_fraction 0.1

Available configurations (num_heads -> global-local split):
  4  -> 2-2  (50% global, rounded from 62.5%)
  8  -> 5-3  (62.5% global, baseline)
  16 -> 10-6 (62.5% global)
  24 -> 15-9 (62.5% global)
  32 -> 20-12 (62.5% global)
        """
    )
    parser.add_argument("--config", type=int, choices=list(HEAD_CONFIGS.keys()),
                        help="Specific head count to test (4, 8, 16, 24, or 32)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Specific seeds to use")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds from default list [42,123,456,789,1024]")
    parser.add_argument("--communication_round", type=int, default=50, help="Number of FL rounds")
    parser.add_argument("--max_epoch", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    args = parser.parse_args()

    # Determine seeds to use
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS[:args.num_seeds]

    # Determine configurations to run
    if args.config:
        configs_to_run = [HEAD_CONFIGS[args.config]]
    else:
        configs_to_run = list(HEAD_CONFIGS.values())

    total_experiments = len(configs_to_run) * len(seeds)

    # Print experiment plan
    print("=" * 70)
    print("HEAD COUNT EXPERIMENT PLAN")
    print("=" * 70)
    print(f"Configurations:")
    for num_heads, global_heads, local_heads in configs_to_run:
        ratio = global_heads / num_heads * 100
        print(f"  {num_heads} heads -> {global_heads}-{local_heads} ({ratio:.1f}% global)")
    print(f"Seeds:            {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"Rounds per exp:   {args.communication_round}")
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Run experiments
    results = {}
    experiment_num = 0

    for num_heads, global_heads, local_heads in configs_to_run:
        config_key = f"h{num_heads}"
        results[config_key] = {}

        for seed in seeds:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Running {num_heads} heads with seed {seed}")

            success = run_single_experiment(num_heads, global_heads, seed, args)
            results[config_key][seed] = "Success" if success else "Failed"

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for config_key in sorted(results.keys(), key=lambda x: int(x[1:])):
        seed_results = results[config_key]
        successes = sum(1 for s in seed_results.values() if s == "Success")
        failures = len(seed_results) - successes
        success_count += successes
        fail_count += failures

        status_str = ", ".join([f"seed{s}:{r[0]}" for s, r in seed_results.items()])
        print(f"  {config_key}: {successes}/{len(seed_results)} succeeded | {status_str}")

    print("-" * 70)
    print(f"Total: {success_count}/{total_experiments} succeeded, {fail_count} failed")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
