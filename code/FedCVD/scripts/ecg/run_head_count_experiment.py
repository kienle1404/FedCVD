#!/usr/bin/env python3
"""
Head count scaling — Ratio sweep at multiple total head counts.

Tests how performance scales with total attention heads (N=1,2,4,8,16,32).
Uses representative ratio points per head count (0%, 25%, 50%, 75%, 100% global).
All N=8 points already exist from the head-ratio sweep.

Usage:
    python run_head_count_experiment.py --all --num_seeds 1    # All new configs (N=1,2,4,16,32)
    python run_head_count_experiment.py --total_heads 4        # N=4 only, 1 seed
    python run_head_count_experiment.py --total_heads 16 --pct_global 50  # Single config
    python run_head_count_experiment.py --total_heads 1 --num_seeds 1 \\
        --communication_round 2 --data_fraction 0.1            # Smoke test
    python run_head_count_experiment.py --small --num_seeds 1  # N=1,2,4 only
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

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Phase 2 configs: (total_heads -> list of (global_heads, local_heads))
# Ratio points per head count: all integer splits that correspond to 0/25/50/75/100% global.
# For N<4 some ratios collapse: N=2 has 3 points, N=1 has 2 points.
# N=8 row is provided for reference only — all those runs already exist.
PHASE2_CONFIGS = {
    1: [
        (0, 1), (1, 0),                             # 0%, 100%
    ],
    2: [
        (0, 2), (1, 1), (2, 0),                     # 0%, 50%, 100%
    ],
    4: [
        (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),    # 0%, 25%, 50%, 75%, 100%
    ],
    8: [
        (0, 8), (2, 6), (4, 4), (6, 2), (8, 0),    # existing — skip by default
    ],
    16: [
        (0, 16), (4, 12), (8, 8), (12, 4), (16, 0),
    ],
    32: [
        (0, 32), (8, 24), (16, 16), (24, 8), (32, 0),
    ],
}

# N=8 configs that already exist — skipped unless --include_n8 is passed
EXISTING_N8 = {(0, 8), (2, 6), (4, 4), (6, 2), (8, 0)}


def run_single_experiment(global_heads: int, local_heads: int, seed: int, args) -> bool:
    """Run a single head count experiment. Returns True if successful."""
    num_heads = global_heads + local_heads

    print(f"\n{'=' * 70}")
    print(f"FedDualAtt Phase 2: global={global_heads}, local={local_heads} (N={num_heads}) | seed={seed}")
    print(f"{'=' * 70}")
    print(f"Rounds:     {args.communication_round}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 head count scaling — ratio sweep at N=16 and N=32",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Phase 2 (50 new runs, 5 seeds each)
  python run_head_count_experiment.py --all --num_seeds 5

  # N=16 only, all 5 ratio points, 5 seeds
  python run_head_count_experiment.py --total_heads 16 --num_seeds 5

  # Single config (N=16, 50% global), 1 seed for smoke test
  python run_head_count_experiment.py --total_heads 16 --pct_global 50 \\
      --num_seeds 1 --communication_round 2 --data_fraction 0.1

Phase 2 configs (0%, 25%, 50%, 75%, 100% global):
  N=16: (0,16) (4,12) (8,8) (12,4) (16,0)
  N=32: (0,32) (8,24) (16,16) (24,8) (32,0)
  N=8:  all existing (skipped unless --include_n8)
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all new configs (N=1,2,4,16,32)")
    group.add_argument("--small", action="store_true", help="Run N=1,2,4 only (10 new runs)")
    group.add_argument("--total_heads", type=int, choices=[1, 2, 4, 8, 16, 32],
                       help="Total head count to sweep")
    parser.add_argument("--pct_global", type=int, choices=[0, 25, 50, 75, 100],
                        help="Run only this ratio point (requires --total_heads)")
    parser.add_argument("--include_n8", action="store_true",
                        help="Include N=8 configs (already exist, skip by default)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Specific seeds to use")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds from default list [42,123,456,789,1024]")
    parser.add_argument("--communication_round", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    args = parser.parse_args()

    if args.pct_global is not None and args.total_heads is None:
        parser.error("--pct_global requires --total_heads")

    # Determine seeds
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[:args.num_seeds]

    # Determine configs
    def pct_to_config(n, pct):
        g = round(n * pct / 100)
        return (g, n - g)

    if args.all:
        configs = [(g, l) for n in [1, 2, 4, 16, 32] for g, l in PHASE2_CONFIGS[n]]
    elif args.small:
        configs = [(g, l) for n in [1, 2, 4] for g, l in PHASE2_CONFIGS[n]]
    elif args.total_heads is not None:
        n = args.total_heads
        all_for_n = PHASE2_CONFIGS[n]
        if args.pct_global is not None:
            target = pct_to_config(n, args.pct_global)
            configs = [c for c in all_for_n if c == target]
            if not configs:
                parser.error(f"No config found for N={n}, pct_global={args.pct_global}%")
        else:
            configs = all_for_n
    else:
        parser.error("Specify --all, --small, or --total_heads {1,2,4,8,16,32}")

    # Filter out existing N=8 configs unless explicitly included
    if not args.include_n8:
        configs = [(g, l) for g, l in configs if (g, l) not in EXISTING_N8]

    if not configs:
        print("No configs to run (all were existing N=8 configs). Pass --include_n8 to run them.")
        return

    total = len(configs) * len(seeds)

    print("=" * 70)
    print("PHASE 2 — HEAD COUNT RATIO SWEEP")
    print("=" * 70)
    print(f"Configs ({len(configs)}):")
    for g, l in configs:
        n = g + l
        pct = g / n * 100 if n > 0 else 0
        print(f"  N={n:2d}: global={g:2d}, local={l:2d} ({pct:.0f}% global)")
    print(f"Seeds:             {seeds}")
    print(f"Total experiments: {total}")
    print(f"Rounds per exp:    {args.communication_round}")
    print(f"Started at:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    exp_num = 0

    for g, l in configs:
        key = f"g{g}_l{l}"
        results[key] = {}
        for seed in seeds:
            exp_num += 1
            print(f"\n[{exp_num}/{total}] global={g}, local={l}, N={g+l}, seed={seed}")
            success = run_single_experiment(g, l, seed, args)
            results[key][seed] = "Success" if success else "Failed"

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    success_count = fail_count = 0
    for key in results:
        sr = results[key]
        s = sum(1 for v in sr.values() if v == "Success")
        f = len(sr) - s
        success_count += s
        fail_count += f
        status = ", ".join(f"seed{k}:{v[0]}" for k, v in sr.items())
        print(f"  {key}: {s}/{len(sr)} succeeded | {status}")
    print("-" * 70)
    print(f"Total: {success_count}/{total} succeeded, {fail_count} failed")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
