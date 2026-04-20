#!/usr/bin/env python3
"""
Re-evaluate FedDualAtt checkpoints with corrected evaluation logic.

The original training code had a bug where server-side evaluation did NOT load
per-client local attention params before evaluating each client. This script
re-evaluates all saved checkpoints using the corrected logic:

LOCAL TEST:  Load client k's local params -> evaluate on client k's test set
GLOBAL TEST: Use global model (local positions = 0) -> evaluate on ALL test sets

The training itself was correct (clients loaded the right params before SGD),
so only the recorded metrics were wrong. This script produces corrected
metric.json files without needing to re-train (~112 hours saved).

Usage:
    # Re-evaluate all head ratio checkpoints
    python reevaluate_checkpoints.py

    # Re-evaluate specific configs
    python reevaluate_checkpoints.py --configs global5_local3 global4_local4

    # Re-evaluate the baseline (non-head-ratio) FedDualAtt run
    python reevaluate_checkpoints.py --baseline

    # Dry run (list what would be evaluated)
    python reevaluate_checkpoints.py --dry-run
"""

import sys
import os
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR / "../.."
sys.path.insert(0, str(PROJECT_ROOT))

from model import get_model
from utils.dataloader import get_ecg_dataset
from utils.evaluation import (
    Accumulator, transfer_tensor_to_numpy, calculate_accuracy,
    get_pred_label, calculate_multilabel_metrics,
    FedServerMultiLabelEvaluator,
)

# Paths
DATA_PATH = SCRIPT_DIR / "../../../../data"
OUTPUT_PATH = SCRIPT_DIR / "../../../../output"

# Client names and dataset mapping
CLIENTS = ["client1", "client2", "client3", "client4"]
CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]

# Naming patterns that identify local (personalized) parameters
_LOCAL_PATTERNS = ('local_att', 'local_proj')


def _is_local(name: str) -> bool:
    """Return True if this parameter belongs to the local (personalized) branch."""
    return any(p in name for p in _LOCAL_PATTERNS)


def _zero_local_params(model: nn.Module):
    """Zero all local param positions in the model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if _is_local(name):
                param.zero_()


def load_test_data(batch_size: int = 32, data_fraction: float = 1.0):
    """Load test datasets for all 4 clients."""
    input_path = str(DATA_PATH)
    if not input_path.endswith("/"):
        input_path += "/"

    print("Loading test datasets...")
    test_datasets = [
        get_ecg_dataset(
            [f"{input_path}ECG/preprocessed/{client}/test.csv"],
            base_path=f"{input_path}ECG/preprocessed/",
            locations=CLIENTS,
            file_name="records.h5",
            n_classes=20,
            frac=data_fraction,
        )
        for client in CLIENTS
    ]

    test_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in test_datasets
    ]

    print(f"  Loaded {len(test_loaders)} client test sets: "
          f"{[len(ds) for ds in test_datasets]} samples")
    return test_loaders


def parse_head_config(dirname: str):
    """Parse global/local head counts from directory name like 'global5_local3'."""
    parts = dirname.split("_")
    global_heads = None
    local_heads = None
    for p in parts:
        if p.startswith("global"):
            global_heads = int(p.replace("global", ""))
        elif p.startswith("local"):
            local_heads = int(p.replace("local", ""))
    return global_heads, local_heads


def evaluate_local(model, test_loaders, local_attention_params, criterion, device):
    """
    Evaluate each client using their personalized local params.

    For each client k:
      1. Load client k's local params into model
      2. Evaluate on client k's test set
      3. Zero local positions (restore invariant)

    Returns dict: {str(client_idx): metric_dict}
    """
    model.eval()
    l_metric_dict = {}

    for idx, loader in enumerate(test_loaders):
        # Load this client's local params
        if local_attention_params[idx]:
            model.load_state_dict(local_attention_params[idx], strict=False)

        metric = Accumulator(3)
        pred_score_list, pred_label_list, true_label_list = [], [], []

        for data, label in loader:
            data, label = data.to(device), label.to(device)
            with torch.no_grad():
                pred_score = model(data)
                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)

                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

                loss = criterion(pred_score, label)
                metric.add(
                    float(loss) * len(label),
                    calculate_accuracy(pred_label_np, true_label_np),
                    len(label),
                )

        # Restore invariant
        _zero_local_params(model)

        all_pred_score = np.concatenate(pred_score_list, axis=0)
        all_pred_label = np.concatenate(pred_label_list, axis=0)
        all_true_label = np.concatenate(true_label_list, axis=0)

        metric_dict = calculate_multilabel_metrics(
            all_pred_score, all_pred_label, all_true_label
        )
        metric_dict["loss"] = metric[0] / metric[2]
        l_metric_dict[str(idx)] = metric_dict

        mAP = float(np.mean(metric_dict["average_precision_score"]))
        print(f"    Client {idx} ({CLIENT_NAMES[idx]}): "
              f"Acc={metric[1]/metric[2]:.4f}  "
              f"F1={metric_dict['micro_f1']:.4f}  "
              f"mAP={mAP:.4f}")

    return l_metric_dict


def evaluate_global(model, test_loaders, criterion, device):
    """
    Evaluate global model (local positions = 0) on ALL test sets combined.

    Returns metric_dict with aggregated metrics.
    """
    model.eval()
    metric = Accumulator(3)
    pred_score_list, pred_label_list, true_label_list = [], [], []

    for loader in test_loaders:
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            with torch.no_grad():
                pred_score = model(data)
                pred_score_np = transfer_tensor_to_numpy(pred_score)
                pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                true_label_np = transfer_tensor_to_numpy(label)

                pred_score_list.append(pred_score_np)
                pred_label_list.append(pred_label_np)
                true_label_list.append(true_label_np)

                loss = criterion(pred_score, label)
                metric.add(
                    float(loss) * len(label),
                    calculate_accuracy(pred_label_np, true_label_np),
                    len(label),
                )

    all_pred_score = np.concatenate(pred_score_list, axis=0)
    all_pred_label = np.concatenate(pred_label_list, axis=0)
    all_true_label = np.concatenate(true_label_list, axis=0)

    metric_dict = calculate_multilabel_metrics(
        all_pred_score, all_pred_label, all_true_label
    )
    metric_dict["loss"] = metric[0] / metric[2]

    mAP = float(np.mean(metric_dict["average_precision_score"]))
    print(f"    Global: Acc={metric[1]/metric[2]:.4f}  "
          f"F1={metric_dict['micro_f1']:.4f}  mAP={mAP:.4f}")

    return metric_dict


def evaluate_cross(model, test_loaders, local_attention_params, criterion, device):
    """
    Cross-evaluate each client's personalized model on ALL other clients' test sets.

    For each source client k:
      1. Load client k's local params into model
      2. Evaluate on EVERY client j's test set (including j == k)
      3. Zero local positions (restore invariant)

    Returns dict: {str(source_k): {str(target_j): metric_dict}}
    The diagonal (k == j) matches evaluate_local() results.
    """
    model.eval()
    cross_metric_dict = {}

    for k in range(len(test_loaders)):
        # Load source client k's local params
        if local_attention_params[k]:
            model.load_state_dict(local_attention_params[k], strict=False)

        cross_metric_dict[str(k)] = {}

        for j, loader in enumerate(test_loaders):
            pred_score_list, pred_label_list, true_label_list = [], [], []
            metric = Accumulator(3)

            for data, label in loader:
                data, label = data.to(device), label.to(device)
                with torch.no_grad():
                    pred_score = model(data)
                    pred_score_np = transfer_tensor_to_numpy(pred_score)
                    pred_label_np = transfer_tensor_to_numpy(get_pred_label(pred_score))
                    true_label_np = transfer_tensor_to_numpy(label)

                    pred_score_list.append(pred_score_np)
                    pred_label_list.append(pred_label_np)
                    true_label_list.append(true_label_np)

                    loss = criterion(pred_score, label)
                    metric.add(
                        float(loss) * len(label),
                        calculate_accuracy(pred_label_np, true_label_np),
                        len(label),
                    )

            all_pred_score = np.concatenate(pred_score_list, axis=0)
            all_pred_label = np.concatenate(pred_label_list, axis=0)
            all_true_label = np.concatenate(true_label_list, axis=0)

            metric_dict = calculate_multilabel_metrics(
                all_pred_score, all_pred_label, all_true_label
            )
            metric_dict["loss"] = metric[0] / metric[2]
            cross_metric_dict[str(k)][str(j)] = metric_dict

            diag = " (diag)" if k == j else ""
            mAP = float(np.mean(metric_dict["average_precision_score"]))
            print(f"    Src={CLIENT_NAMES[k]} -> Tgt={CLIENT_NAMES[j]}{diag}: "
                  f"F1={metric_dict['micro_f1']:.4f}  mAP={mAP:.4f}")

        # Restore invariant
        _zero_local_params(model)

    return cross_metric_dict


def save_cross_eval_metrics(run_dir: Path, round_num, cross_metrics):
    """Save cross-evaluation metrics to cross_eval_corrected.json."""
    data = {
        "cross_eval": {str(round_num): cross_metrics},
    }
    out_path = run_dir / "server" / "cross_eval_corrected.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved cross-eval: {out_path}")
    return out_path


def evaluate_checkpoint(ckpt_path: Path, model, test_loaders, criterion, device,
                        cross_eval: bool = False):
    """
    Re-evaluate a single checkpoint with corrected logic.

    Returns (local_metric_dict, global_metric_dict, round_num[, cross_metric_dict])
    or None if checkpoint is invalid.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "global_model" not in ckpt or "local_attention_params" not in ckpt:
        print(f"  SKIP: Missing keys in {ckpt_path}")
        return None

    # Load global model state
    try:
        model.load_state_dict(ckpt["global_model"], strict=True)
    except RuntimeError as e:
        print(f"  SKIP: Incompatible checkpoint (architecture mismatch): {e!s:.100s}...")
        return None
    model.to(device)

    # Zero local positions to establish invariant
    # (old checkpoints have contaminated local positions in global_model)
    _zero_local_params(model)

    local_attention_params = ckpt["local_attention_params"]
    round_num = ckpt.get("round", "?")

    print(f"  Round: {round_num}, Clients: {len(local_attention_params)}")

    # LOCAL test: per-client evaluation with personalized params
    print("  [LOCAL TEST]")
    local_metrics = evaluate_local(
        model, test_loaders, local_attention_params, criterion, device
    )

    # Ensure local positions are zeroed before global test
    _zero_local_params(model)

    # GLOBAL test: global-only model on all data combined
    print("  [GLOBAL TEST]")
    global_metrics = evaluate_global(model, test_loaders, criterion, device)

    if cross_eval:
        # Ensure local positions zeroed before cross-eval
        _zero_local_params(model)
        print("  [CROSS EVAL]")
        cross_metrics = evaluate_cross(
            model, test_loaders, local_attention_params, criterion, device
        )
        return local_metrics, global_metrics, round_num, cross_metrics

    return local_metrics, global_metrics, round_num


def find_head_ratio_runs():
    """Find all head ratio experiment directories with checkpoints."""
    base = OUTPUT_PATH / "dual_attention_resnet1d" / "feddualatt"
    runs = []

    if not base.exists():
        print(f"ERROR: Output directory not found: {base}")
        return runs

    for config_dir in sorted(base.iterdir()):
        if not config_dir.is_dir():
            continue
        dirname = config_dir.name

        # Match globalX_localY directories
        global_heads, local_heads = parse_head_config(dirname)
        if global_heads is None:
            continue

        # Find seed subdirectories (skip old-format timestamp dirs without seed prefix)
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue

            seed = seed_dir.name.replace("seed", "")

            # Find the latest valid timestamp subdirectory (requires server/metric.json)
            valid_ts = [
                ts_dir for ts_dir in sorted(seed_dir.iterdir())
                if ts_dir.is_dir()
                and (ts_dir / "server" / "model.pth").exists()
                and (ts_dir / "server" / "metric.json").exists()
            ]
            if not valid_ts:
                continue
            ts_dir = valid_ts[-1]  # latest timestamp
            ckpt = ts_dir / "server" / "model.pth"
            runs.append({
                "config": dirname,
                "global_heads": global_heads,
                "local_heads": local_heads,
                "seed": seed,
                "run_dir": ts_dir,
                "ckpt_path": ckpt,
            })

    return runs


def find_baseline_runs():
    """Find baseline FedDualAtt runs (non-head-ratio)."""
    base = OUTPUT_PATH / "dual_attention_resnet1d" / "feddualatt"
    runs = []

    if not base.exists():
        return runs

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        # Baseline runs have timestamp directories directly (no globalX_localY prefix)
        if item.name.startswith("global") or item.name.startswith("seed"):
            continue
        # Check if it's a timestamp directory
        if len(item.name) == 14 and item.name.isdigit():
            ckpt = item / "server" / "model.pth"
            if ckpt.exists():
                # Read setting.json to get head config
                setting_file = item / "setting.json"
                global_heads, local_heads = 5, 3  # default
                if setting_file.exists():
                    with open(setting_file) as f:
                        setting = json.load(f)
                    global_heads = setting.get("global_heads", 5)
                    local_heads = setting.get("local_heads", 3)

                runs.append({
                    "config": "baseline",
                    "global_heads": global_heads,
                    "local_heads": local_heads,
                    "seed": "default",
                    "run_dir": item,
                    "ckpt_path": ckpt,
                })

    return runs


def save_corrected_metrics(run_dir: Path, round_num, local_metrics, global_metrics):
    """Save corrected metrics to metric_corrected.json alongside original metric.json."""
    # Build metric structure matching the original format
    corrected = {
        "local_test": {str(round_num): local_metrics},
        "global_test": {str(round_num): global_metrics},
    }

    out_path = run_dir / "server" / "metric_corrected.json"
    with open(out_path, "w") as f:
        json.dump(corrected, f, indent=2)

    print(f"  Saved: {out_path}")
    return out_path


def compare_metrics(run_dir: Path, round_num, local_metrics, global_metrics):
    """Compare corrected metrics with original and print differences."""
    orig_path = run_dir / "server" / "metric.json"
    if not orig_path.exists():
        print("  (No original metric.json to compare)")
        return

    with open(orig_path) as f:
        orig = json.load(f)

    # Compare global test
    orig_global = orig.get("global_test", {})
    orig_round = orig_global.get(str(round_num), orig_global.get(str(max(int(k) for k in orig_global.keys())), {})) if orig_global else {}

    if orig_round:
        orig_f1 = orig_round.get("micro_f1", 0)
        new_f1 = global_metrics.get("micro_f1", 0)
        orig_mAP = float(np.mean(orig_round.get("average_precision_score", [0])))
        new_mAP = float(np.mean(global_metrics.get("average_precision_score", [0])))

        print(f"  GLOBAL: F1 {orig_f1:.4f} -> {new_f1:.4f} (delta={new_f1-orig_f1:+.4f})  "
              f"mAP {orig_mAP:.4f} -> {new_mAP:.4f} (delta={new_mAP-orig_mAP:+.4f})")

    # Compare local test
    orig_local = orig.get("local_test", {})
    orig_local_round = orig_local.get(str(round_num), {})
    if not orig_local_round:
        # Try last round
        if orig_local:
            last_key = str(max(int(k) for k in orig_local.keys()))
            orig_local_round = orig_local.get(last_key, {})

    if orig_local_round:
        for cid in ["0", "1", "2", "3"]:
            if cid in orig_local_round and cid in local_metrics:
                orig_cf1 = orig_local_round[cid].get("micro_f1", 0)
                new_cf1 = local_metrics[cid].get("micro_f1", 0)
                print(f"  LOCAL client {cid} ({CLIENT_NAMES[int(cid)]}): "
                      f"F1 {orig_cf1:.4f} -> {new_cf1:.4f} (delta={new_cf1-orig_cf1:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate FedDualAtt checkpoints")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific configs to re-evaluate (e.g., global5_local3)")
    parser.add_argument("--baseline", action="store_true",
                        help="Include baseline (non-head-ratio) runs")
    parser.add_argument("--all", action="store_true",
                        help="Re-evaluate all head ratio + baseline runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="List runs without evaluating")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not set.")
    parser.add_argument("--cross-eval", action="store_true",
                        help="Also run cross-evaluation (source client k model on all "
                             "target test sets j), saving cross_eval_corrected.json")
    args = parser.parse_args()

    # Default: all head ratio runs
    if args.configs is None and not args.baseline:
        args.all = True

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Find runs
    runs = []
    if args.all or args.configs is not None:
        all_runs = find_head_ratio_runs()
        if args.configs:
            runs = [r for r in all_runs if r["config"] in args.configs]
        else:
            runs = all_runs

    if args.all or args.baseline:
        runs += find_baseline_runs()

    if not runs:
        print("No checkpoint runs found!")
        return

    print(f"\nFound {len(runs)} checkpoint(s) to re-evaluate:")
    for r in runs:
        print(f"  {r['config']} seed={r['seed']}  ->  {r['ckpt_path']}")

    if args.dry_run:
        return

    # Load test data once (shared across all evaluations)
    test_loaders = load_test_data(args.batch_size, args.data_fraction)

    criterion = nn.BCELoss()

    # Group by head config to reuse model instances
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in runs:
        key = (r["global_heads"], r["local_heads"])
        by_config[key].append(r)

    results_summary = []

    for (gh, lh), config_runs in sorted(by_config.items()):
        num_heads = gh + lh
        print(f"\n{'=' * 70}")
        print(f"Config: {gh} global + {lh} local heads (total={num_heads})")
        print(f"{'=' * 70}")

        # Create model with this head configuration
        model = get_model("dual_attention_resnet1d",
                          global_heads=gh, num_heads=num_heads)
        model.to(device)

        for run in config_runs:
            print(f"\n--- Seed: {run['seed']} ---")
            print(f"  Checkpoint: {run['ckpt_path']}")

            result = evaluate_checkpoint(
                run["ckpt_path"], model, test_loaders, criterion, device,
                cross_eval=args.cross_eval,
            )
            if result is None:
                continue

            if args.cross_eval:
                local_metrics, global_metrics, round_num, cross_metrics = result
            else:
                local_metrics, global_metrics, round_num = result
                cross_metrics = None

            # Compare with original
            compare_metrics(run["run_dir"], round_num, local_metrics, global_metrics)

            # Save corrected metrics
            save_corrected_metrics(
                run["run_dir"], round_num, local_metrics, global_metrics
            )

            # Save cross-eval metrics if requested
            if cross_metrics is not None:
                save_cross_eval_metrics(run["run_dir"], round_num, cross_metrics)

            # Collect summary
            global_f1 = global_metrics["micro_f1"]
            global_mAP = float(np.mean(global_metrics["average_precision_score"]))
            global_acc = global_metrics["accuracy"]

            client_f1s = [local_metrics[str(i)]["micro_f1"] for i in range(4)]
            client_mAPs = [
                float(np.mean(local_metrics[str(i)]["average_precision_score"]))
                for i in range(4)
            ]
            client_accs = [local_metrics[str(i)]["accuracy"] for i in range(4)]

            results_summary.append({
                "config": run["config"],
                "global_heads": gh,
                "local_heads": lh,
                "seed": run["seed"],
                "global_f1": global_f1,
                "global_mAP": global_mAP,
                "global_acc": global_acc,
                **{f"{CLIENT_NAMES[i]}_f1": client_f1s[i] for i in range(4)},
                **{f"{CLIENT_NAMES[i]}_mAP": client_mAPs[i] for i in range(4)},
                **{f"{CLIENT_NAMES[i]}_acc": client_accs[i] for i in range(4)},
            })

    # Save summary CSV
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_path = SCRIPT_DIR / "../../docs/reevaluation_results.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"Summary saved to: {summary_path}")
        print(f"Total runs re-evaluated: {len(results_summary)}")
        print(f"{'=' * 70}")

        # Print summary table
        print(f"\n{'Config':<20} {'Seed':<8} {'Global F1':>10} {'Global mAP':>10} "
              f"{'SPH F1':>8} {'PTB F1':>8} {'SXPH F1':>8} {'G12 F1':>8}")
        print("-" * 90)
        for r in sorted(results_summary, key=lambda x: (x["global_heads"], x["seed"])):
            print(f"{r['config']:<20} {r['seed']:<8} "
                  f"{r['global_f1']:>10.4f} {r['global_mAP']:>10.4f} "
                  f"{r['SPH_f1']:>8.4f} {r['PTB-XL_f1']:>8.4f} "
                  f"{r['SXPH_f1']:>8.4f} {r['G12EC_f1']:>8.4f}")


if __name__ == "__main__":
    main()
