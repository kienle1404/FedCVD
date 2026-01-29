#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ];then
    current_dir=$(dirname "$DIRNAME")
else
    current_dir="$(pwd)"/"$(dirname "$DIRNAME")"
fi
input_path=/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/data/CVD/dataset
output_path="$current_dir"/../../../../output
seed=42
batch_sizes=(32 64 128)
lrs=(0.1)
server_lrs=(1 0.1 0.01)
gammas=(0 0.1 0.5 0.9 1)
alpha=1
beta=0
max_epoch=1
communication_round=50
optimizer_name=Adam
momentum_rounds=(1 3 5)
models=(resnet1d50)

for batch_size in "${batch_sizes[@]}"; do
  for model in "${models[@]}"; do
    for lr in "${lrs[@]}"; do
      for server_lr in "${server_lrs[@]}"; do
          for gamma in "${gammas[@]}"; do
            for momentum_round in "${momentum_rounds[@]}"; do
              case_name="mf1-fedfa-model=${model}-batch_size=${batch_size}-lr=${lr}-server_lr=${server_lr}-gamma=${gamma}-optim=${optimizer_name}-alpha=${alpha}-beta=${beta}-momentum_round=${momentum_round}-seed=${seed}"
              python "$current_dir"/../../trainers/fedfa_ecg.py  	 --batch_size "$batch_size" \
                                                                   --case_name "$case_name" \
                                                                   --alpha $alpha \
                                                                   --beta $beta \
                                                                   --gamma "$gamma" \
                                                                   --momentum_round "$momentum_round" \
                                                                   --server_lr "$server_lr" \
                                                                   --lr "$lr" \
                                                                   --seed $seed \
                                                                   --input_path "$input_path" \
                                                                   --output_path "$output_path" \
                                                                   --max_epoch $max_epoch \
                                                                   --model "$model" \
                                                                   --communication_round $communication_round
            done
          done
      done
    done
  done
done
