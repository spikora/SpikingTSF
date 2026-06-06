#!/bin/bash
# SpikF — ETTh1 benchmark  (config: configs/SpikF/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikF

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
    echo "SpikF ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/SpikF/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/SpikF/ETTh1_pl${pred_len}.txt"
done
