#!/usr/bin/env python3
"""
Generate a Table 2-style comparison table matching the FedCVD paper format.

Structure (per metric):
         | LOCAL                                          | GLOBAL
Methods  | Client1 (SPH) | Client2 (PTB-XL) | Client3 (SXPH) | Client4 (G12EC) |
         | Mi-F1  Acc mAP | Mi-F1  Acc  mAP  | Mi-F1  Acc mAP | Mi-F1  Acc  mAP | Mi-F1  Acc mAP

Usage:
    python generate_table2.py [--csv output.csv] [--latex]
"""

import csv
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR.parent.parent / "docs"

BASELINE_CSV = SCRIPT_DIR / "results1.csv"
HEAD_RATIO_CSV = DOCS_DIR / "head_ratio_all_metrics.csv"

CLIENT_NAMES = ["SPH", "PTB-XL", "SXPH", "G12EC"]
CLIENT_LABELS = ["Client1 (SPH)", "Client2 (PTB-XL)", "Client3 (SXPH)", "Client4 (G12EC)"]
METRICS = ["micro_f1", "acc", "mAP"]
METRIC_LABELS = {"micro_f1": "Mi-F1", "acc": "Acc", "mAP": "mAP"}

BASELINE_ORDER = [
    ("centralized", "Centralized"),
    ("fedavg", "FedAvg"),
    ("fedprox", "FedProx"),
    ("scaffold", "Scaffold"),
    ("ditto", "Ditto"),
    ("fedinit", "FedInit"),
    ("fedsm", "FedSM"),
    ("fedala", "FedALA"),
]

HEAD_RATIO_ORDER = [
    "8-0", "7-1", "6-2", "5-3", "4-4", "3-5", "2-6", "1-7", "0-8"
]


def load_baselines(csv_path: Path) -> dict:
    """Load per-client + global metrics from results1.csv."""
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            if not row.get("global_micro_f1"):
                results[algo] = None
                continue

            data = {"clients": {}, "global": {}, "has_std": False}

            for i, name in enumerate(CLIENT_NAMES):
                client = {}
                for m in METRICS:
                    key = f"{name}_{m}" if m != "acc" else f"{name}_acc"
                    val = row.get(key, "")
                    client[m] = float(val) if val else None
                data["clients"][i] = client

            for m in METRICS:
                key = f"global_{m}" if m != "acc" else "global_acc"
                val = row.get(key, "")
                data["global"][m] = float(val) if val else None

            results[algo] = data
    return results


def load_head_ratios(csv_path: Path) -> dict:
    """Load per-client + global metrics with mean/std from head_ratio_all_metrics.csv."""
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio = row["ratio"]
            data = {"clients": {}, "global": {}, "has_std": True,
                    "num_seeds": int(row["num_seeds"])}

            for i, name in enumerate(CLIENT_NAMES):
                client = {}
                for m in METRICS:
                    csv_m = m if m != "acc" else "acc"
                    mean_key = f"{name}_{csv_m}_mean"
                    std_key = f"{name}_{csv_m}_std"
                    client[f"{m}_mean"] = float(row[mean_key])
                    client[f"{m}_std"] = float(row[std_key])
                data["clients"][i] = client

            for m in METRICS:
                csv_m = m if m != "acc" else "acc"
                data["global"][f"{m}_mean"] = float(row[f"global_{csv_m}_mean"])
                data["global"][f"{m}_std"] = float(row[f"global_{csv_m}_std"])

            results[ratio] = data
    return results


def fmt_val(val, std=None):
    """Format value with optional std."""
    if val is None:
        return "--"
    if std is not None and std > 0:
        return f"{val:.1f}+-{std:.1f}"
    return f"{val:.1f}"


def fmt_latex_val(val, std=None, bold=False):
    """Format for LaTeX."""
    if val is None:
        return "--"
    if std is not None and std > 0:
        s = f"{val:.1f}$\\pm${std:.1f}"
    else:
        s = f"{val:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def get_client_val(data, client_idx, metric):
    """Get client metric value (handles both baseline and head_ratio formats)."""
    if data is None:
        return None, None
    client = data["clients"].get(client_idx, {})
    if data["has_std"]:
        return client.get(f"{metric}_mean"), client.get(f"{metric}_std")
    else:
        return client.get(metric), None


def get_global_val(data, metric):
    """Get global metric value."""
    if data is None:
        return None, None
    g = data["global"]
    if data["has_std"]:
        return g.get(f"{metric}_mean"), g.get(f"{metric}_std")
    else:
        return g.get(metric), None


def print_metric_table(metric, metric_label, all_rows):
    """Print one table for a specific metric."""
    col_w = 13
    name_w = 22

    header1 = f"{'':<{name_w}} |{'LOCAL':^{4*(col_w+3)-1}}| {'GLOBAL':^{col_w}}"
    header2 = f"{'Methods':<{name_w}}"
    for label in CLIENT_LABELS:
        short = label.split("(")[1].rstrip(")")
        header2 += f" | {short:>{col_w}}"
    header2 += f" | {'Global':>{col_w}}"

    line_w = len(header2)
    print("\n" + "=" * line_w)
    print(f"{metric_label} (%)")
    print("=" * line_w)
    print(header1)
    print(header2)
    print("-" * line_w)

    for name, data, is_separator in all_rows:
        if is_separator:
            print("-" * line_w)
            continue

        row = f"{name:<{name_w}}"
        for i in range(4):
            val, std = get_client_val(data, i, metric)
            row += f" | {fmt_val(val, std):>{col_w}}"
        gval, gstd = get_global_val(data, metric)
        row += f" | {fmt_val(gval, gstd):>{col_w}}"
        print(row)

    print("=" * line_w)


def print_tables(baselines, head_ratios):
    """Print all metric tables."""
    # Build row list
    all_rows = []

    for algo_key, display_name in BASELINE_ORDER:
        all_rows.append((display_name, baselines.get(algo_key), False))

    all_rows.append(("---", None, True))  # separator

    for ratio in HEAD_RATIO_ORDER:
        data = head_ratios.get(ratio)
        if data is None:
            continue
        g, l = ratio.split("-")
        name = f"FedDualAtt ({g}G:{l}L)"
        all_rows.append((name, data, False))

    for metric, label in [("micro_f1", "Micro-F1"), ("acc", "Accuracy"), ("mAP", "mAP")]:
        print_metric_table(metric, label, all_rows)

    # Summary: best per metric
    print("\n" + "=" * 80)
    print("Best per column (excluding Centralized)")
    print("=" * 80)

    fl_rows = [(n, d) for n, d, s in all_rows if not s and d is not None and n != "Centralized"]

    for metric, label in [("micro_f1", "Mi-F1"), ("acc", "Acc"), ("mAP", "mAP")]:
        print(f"\n  {label}:")
        for i, cname in enumerate(CLIENT_NAMES):
            best_name, best_val = None, -1
            for name, data in fl_rows:
                val, _ = get_client_val(data, i, metric)
                if val is not None and val > best_val:
                    best_val = val
                    best_name = name
            if best_name:
                print(f"    {cname:<8}: {best_name} ({best_val:.1f}%)")

        best_name, best_val = None, -1
        for name, data in fl_rows:
            val, _ = get_global_val(data, metric)
            if val is not None and val > best_val:
                best_val = val
                best_name = name
        if best_name:
            print(f"    {'GLOBAL':<8}: {best_name} ({best_val:.1f}%)")


def find_top2_per_col(all_rows, metric):
    """Find best and second-best value per column (excludes Centralized)."""
    vals = {i: [] for i in range(4)}
    vals["global"] = []

    for name, data, is_sep in all_rows:
        if is_sep or data is None or name in ("Centralized", "Central."):
            continue
        for i in range(4):
            val, _ = get_client_val(data, i, metric)
            if val is not None:
                vals[i].append(val)
        gval, _ = get_global_val(data, metric)
        if gval is not None:
            vals["global"].append(gval)

    result = {}
    for key in list(range(4)) + ["global"]:
        sorted_v = sorted(set(vals[key]), reverse=True)
        result[key] = {
            "best": sorted_v[0] if len(sorted_v) > 0 else -1,
            "second": sorted_v[1] if len(sorted_v) > 1 else -1,
        }
    return result


def fmt_latex_ranked(val, std, best, second):
    """Format LaTeX value with bold (best) or underline (second-best)."""
    if val is None:
        return "--"
    if std is not None and std > 0:
        s = f"{val:.1f}$\\pm${std:.1f}"
    else:
        s = f"{val:.1f}"
    if abs(val - best) < 0.05:
        s = f"\\textbf{{{s}}}"
    elif abs(val - second) < 0.05:
        s = f"{{\\ul {s}}}"
    return s


def print_latex(baselines, head_ratios):
    """Print LaTeX table matching original FedCVD paper Table 2 format exactly."""
    all_rows = []
    for algo_key, display_name in BASELINE_ORDER:
        all_rows.append((display_name, baselines.get(algo_key), False))
    all_rows.append(("---", None, True))
    for ratio in HEAD_RATIO_ORDER:
        data = head_ratios.get(ratio)
        if data is None:
            continue
        g, l = ratio.split("-")
        all_rows.append((f"FedDualAtt ({g}G:{l}L)", data, False))

    # Find best/second-best per column
    top2 = {}
    for m in METRICS:
        top2[m] = find_top2_per_col(all_rows, m)

    n_metrics = len(METRICS)
    n_data_cols = 4 * n_metrics + n_metrics  # 15 columns

    # Column spec: 4 clients x 3 metrics each, then 3 global
    col_spec = "l|" + "c" * (4 * n_metrics) + "|" + "c" * n_metrics

    print("\n% LaTeX Table 2 - Matching original FedCVD paper format + Accuracy")
    print("% Requires: \\usepackage{colortbl}, \\usepackage{multirow}, \\usepackage{soul}")
    print("\\begin{table*}[t]")
    print("\\caption{The performance of different FL methods on Fed-ECG is reported using Micro F1-Score (Mi-F1), Accuracy (Acc), and Mean Average Precision (mAP), all in \\%. Best results in \\textbf{bold}, second-best \\underline{underlined}. FedDualAtt results: mean$\\pm$std over 5 seeds.}")
    print("\\label{tab:r-perf-ecg}")
    print("\\resizebox{\\textwidth}{!}{%")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")

    # Header row 1: LOCAL / GLOBAL
    local_span = 4 * n_metrics
    global_span = n_metrics
    print(f"            & \\multicolumn{{{local_span}}}{{c|}}{{\\textbf{{LOCAL}}}}"
          f" & \\multicolumn{{{global_span}}}{{c}}{{\\textbf{{GLOBAL}}}}"
          f" \\\\ \\cline{{2-{1 + n_data_cols}}}")

    # Header row 2: Client names
    print("Methods     ", end="")
    for i, label in enumerate(["Client1", "Client2", "Client3", "Client4"]):
        sep = "|" if i == 3 else ""
        print(f"& \\multicolumn{{{n_metrics}}}{{c{sep}}}{{{label}}} ", end="")
    print(f"& \\multicolumn{{{n_metrics}}}{{c}}{{}} \\\\")

    # Header row 3: Metric labels with green shading on GLOBAL
    print("            ", end="")
    for _ in range(4):
        for m in METRICS:
            print(f"& {METRIC_LABELS[m]}$\\uparrow$ ", end="")
    for m in METRICS:
        print(f"& \\cellcolor[HTML]{{E6EFE6}}{METRIC_LABELS[m]}$\\uparrow$ ", end="")
    print("\\\\ \\hline")

    def make_row(name, data):
        if data is None:
            parts = ["--"] * n_data_cols
        else:
            parts = []
            # Per-client LOCAL
            for i in range(4):
                for m in METRICS:
                    val, std = get_client_val(data, i, m)
                    b = top2[m][i]["best"]
                    s = top2[m][i]["second"]
                    parts.append(fmt_latex_ranked(val, std, b, s))
            # GLOBAL (with green shading)
            for m in METRICS:
                gval, gstd = get_global_val(data, m)
                b = top2[m]["global"]["best"]
                s = top2[m]["global"]["second"]
                cell = fmt_latex_ranked(gval, gstd, b, s)
                parts.append(f"\\cellcolor[HTML]{{E6EFE6}}{cell}")

        # Pad name for alignment
        padded = f"{name:<12}"
        return f"{padded} & {' & '.join(parts)} \\\\"

    for name, data, is_sep in all_rows:
        if is_sep:
            print("\\hline")
        else:
            print(make_row(name, data))

    print("\\hline")
    print("\\end{tabular}")
    print("}")
    print("\\end{table*}")


def save_csv(baselines, head_ratios, csv_path: str):
    """Save combined per-client + global table to CSV."""
    fieldnames = ["method", "type"]
    for loc in CLIENT_NAMES + ["global"]:
        for m in METRICS:
            fieldnames.append(f"{loc}_{m}_mean")
            fieldnames.append(f"{loc}_{m}_std")

    rows = []

    for algo_key, display_name in BASELINE_ORDER:
        data = baselines.get(algo_key)
        row = {"method": display_name, "type": "baseline"}
        if data:
            for i, name in enumerate(CLIENT_NAMES):
                for m in METRICS:
                    val, _ = get_client_val(data, i, m)
                    row[f"{name}_{m}_mean"] = f"{val:.2f}" if val else ""
                    row[f"{name}_{m}_std"] = ""
            for m in METRICS:
                gval, _ = get_global_val(data, m)
                row[f"global_{m}_mean"] = f"{gval:.2f}" if gval else ""
                row[f"global_{m}_std"] = ""
        rows.append(row)

    for ratio in HEAD_RATIO_ORDER:
        data = head_ratios.get(ratio)
        if data is None:
            continue
        g, l = ratio.split("-")
        row = {"method": f"FedDualAtt ({g}G:{l}L)", "type": "feddualatt"}
        for i, name in enumerate(CLIENT_NAMES):
            for m in METRICS:
                val, std = get_client_val(data, i, m)
                row[f"{name}_{m}_mean"] = f"{val:.2f}" if val else ""
                row[f"{name}_{m}_std"] = f"{std:.2f}" if std else ""
        for m in METRICS:
            gval, gstd = get_global_val(data, m)
            row[f"global_{m}_mean"] = f"{gval:.2f}" if gval else ""
            row[f"global_{m}_std"] = f"{gstd:.2f}" if gstd else ""
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Table 2-style comparison")
    parser.add_argument("--csv", type=str, default=None, help="Save to CSV")
    parser.add_argument("--latex", action="store_true", help="Also output LaTeX table")
    args = parser.parse_args()

    baselines = load_baselines(BASELINE_CSV)
    head_ratios = load_head_ratios(HEAD_RATIO_CSV)

    n_valid = len([v for v in baselines.values() if v])
    print(f"Loaded {n_valid} baselines, {len(head_ratios)} head ratio configs\n")

    print_tables(baselines, head_ratios)

    if args.latex:
        print_latex(baselines, head_ratios)

    if args.csv:
        save_csv(baselines, head_ratios, args.csv)


if __name__ == "__main__":
    main()
