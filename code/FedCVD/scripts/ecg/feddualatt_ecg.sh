#!/bin/bash
#
# Federated Learning with Dual Attention Heads - ECG Classification
#
# This script launches federated learning with dual attention personalization.
# - Model: DualAttentionResNet1D (Hybrid ResNet1D34 + Dual Attention Transformer)
# - Algorithm: FedDualAtt (4 global + 4 local attention heads)
# - Task: Multi-label ECG classification (20 classes)
# - Clients: 4 institutions (SPH, PTB-XL, SXPH, G12EC)
#

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi

# Paths
input_path="$current_dir"/../../../../data
output_path="$current_dir"/../../../../output

# Hyperparameters
seed=42
batch_size=32
lr=0.1
max_epoch=1
communication_round=50
model="dual_attention_resnet1d"

# Launch training
python "$current_dir"/../../trainers/feddualatt_ecg.py \
    --batch_size $batch_size \
    --lr $lr \
    --seed $seed \
    --input_path "$input_path" \
    --output_path "$output_path" \
    --max_epoch $max_epoch \
    --communication_round $communication_round \
    --model $model \
    --case_name "feddualatt_ecg_resnet1d"
