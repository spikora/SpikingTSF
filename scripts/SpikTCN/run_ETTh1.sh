#!/bin/bash
# SpikTCN — ETTh1 benchmark  (config: configs/SpikTCN/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikTCN

export CUDA_VISIBLE_DEVICES=3

for pred_len in 336 720
do
    echo "SpikTCN ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/SpikTCN/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/SpikTCN/ETTh1_pl${pred_len}.txt"
done
