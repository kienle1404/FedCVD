#!/usr/bin/env python3
"""
Phase 1 — Decoupled head scaling pre-check.

Fixes global_heads at one or two levels (4 and 8) and sweeps local_heads
independently. The key question: does global F1 stay flat when local heads
are added without removing any global heads?

If yes  → the 8-head zero-sum trade-off was a capacity constraint; proceed to Phase 2.
If no   → the trade-off is architectural (gradient competition); pivot experiment.

Usage:
    python run_decoupled_heads.py --all --num_seeds 5        # Full Phase 1 (40 new runs)
    python run_decoupled_heads.py --global_heads 8           # Only the g=8 sweep
    python run_decoupled_heads.py --global_heads 8 --local_heads 8  # Single config
    python run_decoupled_heads.py --all --num_seeds 1 --communication_round 2 --data_fraction 0.1  # Smoke test
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

# Phase 1 configs: (global_heads, local_heads)
# A0 (8, 0) is the existing 8-head all-global baseline — already run.
# B0 (4, 0) is the existing 4-head all-global config — already run.
PHASE1_CONFIGS = {
    8: [(8, 0), (8, 2), (8, 4), (8, 8), (8, 16)],
    4: [(4, 0), (4, 2), (4, 4), (4, 8), (4, 16)],
}


def run_single_experiment(global_heads: int, local_heads: int, seed: int, args) -> bool:
    """Run a single decoupled-heads experiment. Returns True if successful."""
    num_heads = global_heads + local_heads

    print(f"\n{'=' * 70}")
    print(f"FedDualAtt decoupled: global={global_heads}, local={local_heads} (total={num_heads}) | seed={seed}")
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
        "--case_name", f"feddualatt_g{global_heads}_l{local_heads}",
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 decoupled head scaling — fix global_heads, vary local_heads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Phase 1: both g=8 and g=4 lines, 5 seeds (40 new runs)
  python run_decoupled_heads.py --all --num_seeds 5

  # Only g=8 line, 5 seeds
  python run_decoupled_heads.py --global_heads 8 --num_seeds 5

  # Single config for smoke test
  python run_decoupled_heads.py --global_heads 8 --local_heads 8 --num_seeds 1 \\
      --communication_round 2 --data_fraction 0.1

Phase 1 configs (global_heads fixed, local_heads swept):
  g=8: (8,0) (8,2) (8,4) (8,8) (8,16)   [8,0] is existing baseline
  g=4: (4,0) (4,2) (4,4) (4,8) (4,16)   [4,0] is existing baseline
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run both g=8 and g=4 sweeps")
    group.add_argument("--global_heads", type=int, choices=[4, 8],
                       help="Fixed global head count (4 or 8)")
    parser.add_argument("--local_heads", type=int,
                        help="Run only this local_heads value (requires --global_heads)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Specific seeds to use")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds from default list [42,123,456,789,1024]")
    parser.add_argument("--communication_round", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    args = parser.parse_args()

    if args.local_heads is not None and args.global_heads is None:
        parser.error("--local_heads requires --global_heads")

    # Determine seeds
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[:args.num_seeds]

    # Determine configs
    if args.all:
        configs = PHASE1_CONFIGS[8] + PHASE1_CONFIGS[4]
    elif args.global_heads:
        all_for_g = PHASE1_CONFIGS[args.global_heads]
        if args.local_heads is not None:
            configs = [(g, l) for g, l in all_for_g if l == args.local_heads]
            if not configs:
                parser.error(f"local_heads={args.local_heads} not in Phase 1 config for global_heads={args.global_heads}")
        else:
            configs = all_for_g
    else:
        parser.error("Specify --all or --global_heads {4,8}")

    total = len(configs) * len(seeds)

    print("=" * 70)
    print("PHASE 1 — DECOUPLED HEADS EXPERIMENT")
    print("=" * 70)
    print(f"Configs ({len(configs)}):")
    for g, l in configs:
        tag = " [existing baseline]" if l == 0 else ""
        print(f"  global={g}, local={l}, total={g+l}{tag}")
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
            print(f"\n[{exp_num}/{total}] global={g}, local={l}, seed={seed}")
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
