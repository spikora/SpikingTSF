#!/bin/bash
# Spikingformer — ETTh2 benchmark  (config: configs/Spikingformer/ETTh2.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/Spikingformer

export CUDA_VISIBLE_DEVICES=2

for pred_len in 96 192 336 720
do
    echo "Spikingformer ETTh2 pl=${pred_len}"
    python run_long.py \
        --config configs/Spikingformer/ETTh2.yaml \
        --pred_len ${pred_len} \
        | tee "Output/Spikingformer/ETTh2_pl${pred_len}.txt"
done
