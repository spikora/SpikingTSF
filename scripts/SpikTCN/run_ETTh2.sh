#!/bin/bash
# SpikTCN — ETTh2 benchmark  (config: configs/SpikTCN/ETTh2.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikTCN

export CUDA_VISIBLE_DEVICES=3

for pred_len in 96 192 336 720
do
    echo "SpikTCN ETTh2 pl=${pred_len}"
    python run_long.py \
        --config configs/SpikTCN/ETTh2.yaml \
        --pred_len ${pred_len} \
        | tee "Output/SpikTCN/ETTh2_pl${pred_len}.txt"
done
