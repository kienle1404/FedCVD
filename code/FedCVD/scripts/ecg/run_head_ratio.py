#!/usr/bin/env python3
"""
Run FedDualAtt training with configurable head ratio and multiple seeds.

Usage:
    python run_head_ratio.py --global_heads 4                    # Single config, single seed
    python run_head_ratio.py --global_heads 4 --num_seeds 5      # Single config, 5 seeds
    python run_head_ratio.py --all                               # All 9 configs (0-8), single seed
    python run_head_ratio.py --all --num_seeds 5                 # All 9 configs, 5 seeds (45 runs)
    python run_head_ratio.py --all --seeds 42 123 456            # All 9 configs, custom seeds
    python run_head_ratio.py --global_heads 8                    # Global-only (ablation)
    python run_head_ratio.py --global_heads 0                    # Local-only (ablation)
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


def run_single_experiment(global_heads: int, seed: int, args) -> bool:
    """Run a single head ratio experiment with specified seed. Returns True if successful."""
    num_heads = args.num_heads
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
        description="Run FedDualAtt with configurable head ratio and multiple seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single configuration, single seed (default: 42)
  python run_head_ratio.py --global_heads 4

  # Single configuration, 5 seeds for statistical significance
  python run_head_ratio.py --global_heads 4 --num_seeds 5

  # All 7 head ratios, single seed
  python run_head_ratio.py --all

  # All 7 head ratios, 5 seeds each (35 total experiments)
  python run_head_ratio.py --all --num_seeds 5

  # Custom seeds
  python run_head_ratio.py --all --seeds 42 123 456

  # Quick test with fewer rounds
  python run_head_ratio.py --all --num_seeds 3 --communication_round 10
        """
    )
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Total number of attention heads (default: 8)")
    parser.add_argument("--global_heads", type=int, help="Number of global heads (0 to num_heads)")
    parser.add_argument("--all", action="store_true", help="Run all configurations (0 to num_heads)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Specific seeds to use (e.g., --seeds 42 123 456)")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds to use from default list [42,123,456,789,1024] (default: 1)")
    parser.add_argument("--communication_round", type=int, default=50, help="Number of FL rounds")
    parser.add_argument("--max_epoch", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    args = parser.parse_args()

    # Validate arguments
    if not args.all and args.global_heads is None:
        print("Error: Must specify either --global_heads or --all")
        sys.exit(1)

    if args.global_heads is not None and (args.global_heads < 0 or args.global_heads > args.num_heads):
        print(f"Error: global_heads must be between 0 and {args.num_heads}, got {args.global_heads}")
        sys.exit(1)

    # Determine seeds to use
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS[:args.num_seeds]

    # Determine head ratios to run
    if args.all:
        head_ratios = list(range(0, args.num_heads + 1))  # 0 to num_heads (includes global-only and local-only)
    else:
        head_ratios = [args.global_heads]

    total_experiments = len(head_ratios) * len(seeds)
    num_heads = args.num_heads

    # Print experiment plan
    print("=" * 70)
    print("EXPERIMENT PLAN")
    print("=" * 70)
    print(f"Total heads:      {num_heads}")
    print(f"Head ratios:      {[f'{h}-{num_heads-h}' for h in head_ratios]}")
    print(f"Seeds:            {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"Rounds per exp:   {args.communication_round}")
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Run experiments
    results = {}
    experiment_num = 0

    for global_heads in head_ratios:
        local_heads = num_heads - global_heads
        ratio_key = f"{global_heads}-{local_heads}"
        results[ratio_key] = {}

        for seed in seeds:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Running {ratio_key} with seed {seed}")

            success = run_single_experiment(global_heads, seed, args)
            results[ratio_key][seed] = "Success" if success else "Failed"

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for ratio_key in sorted(results.keys(), key=lambda x: int(x.split('-')[0])):
        seed_results = results[ratio_key]
        successes = sum(1 for s in seed_results.values() if s == "Success")
        failures = len(seed_results) - successes
        success_count += successes
        fail_count += failures

        status_str = ", ".join([f"seed{s}:{r[0]}" for s, r in seed_results.items()])
        print(f"  {ratio_key}: {successes}/{len(seed_results)} succeeded | {status_str}")

    print("-" * 70)
    print(f"Total: {success_count}/{total_experiments} succeeded, {fail_count} failed")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
