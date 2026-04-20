#!/usr/bin/env python3
"""Extract and print averaged cross-eval 4x4 F1 matrices per head ratio config."""
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR / "../../../../output/dual_attention_resnet1d/feddualatt"
CLIENTS = ['SPH', 'PTB-XL', 'SXPH', 'G12EC']
CONFIGS = [
    ('global8_local0', '8G:0L'),
    ('global7_local1', '7G:1L'),
    ('global6_local2', '6G:2L'),
    ('global5_local3', '5G:3L'),
    ('global4_local4', '4G:4L'),
    ('global3_local5', '3G:5L'),
    ('global2_local6', '2G:6L'),
    ('global1_local7', '1G:7L'),
    ('global0_local8', '0G:8L'),
]

results = {}
for dir_name, label in CONFIGS:
    config_path = BASE / dir_name
    if not config_path.exists():
        continue
    matrices = []
    for seed_dir in sorted(config_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
            continue
        for ts_dir in sorted(seed_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            ce_path = ts_dir / 'server' / 'cross_eval_corrected.json'
            if not ce_path.exists():
                continue
            with open(ce_path) as f:
                data = json.load(f)
            rounds = data.get('cross_eval', {})
            if not rounds:
                continue
            last_round = rounds[str(max(int(k) for k in rounds.keys()))]
            mat = np.full((4, 4), np.nan)
            for k in range(4):
                src = last_round.get(str(k), {})
                for j in range(4):
                    tgt = src.get(str(j), {})
                    if tgt:
                        mat[k, j] = tgt.get('micro_f1', np.nan) * 100
            matrices.append(mat)

    if not matrices:
        print(f'{label}: no seeded data')
        continue

    mean_mat = np.nanmean(np.stack(matrices), axis=0)
    std_mat  = np.nanstd(np.stack(matrices), axis=0)
    results[label] = (mean_mat, std_mat, len(matrices))

    diag    = np.nanmean([mean_mat[k, k] for k in range(4)])
    mask    = ~np.eye(4, dtype=bool)
    offdiag = np.nanmean(mean_mat[mask])

    print(f'\n=== {label} (n={len(matrices)} seeds) ===')
    print('         ' + '  '.join(f'{c:>8}' for c in CLIENTS))
    for k in range(4):
        vals = '  '.join(f'{mean_mat[k,j]:5.1f}{"*" if k==j else " "}' for j in range(4))
        print(f'  {CLIENTS[k]:>8}: {vals}')
    print(f'  >> Diag mean: {diag:.1f}  Off-diag mean: {offdiag:.1f}  Gap: {offdiag-diag:+.1f}')

# Print summary table
print('\n\n=== SUMMARY (mean Micro-F1 %) ===')
print(f'{"Config":>8}  {"Diag":>6}  {"Off-diag":>9}  {"Gap":>6}')
for dir_name, label in CONFIGS:
    if label not in results:
        continue
    mean_mat, _, n = results[label]
    diag    = np.nanmean([mean_mat[k, k] for k in range(4)])
    mask    = ~np.eye(4, dtype=bool)
    offdiag = np.nanmean(mean_mat[mask])
    print(f'{label:>8}  {diag:6.1f}  {offdiag:9.1f}  {offdiag-diag:+6.1f}')
