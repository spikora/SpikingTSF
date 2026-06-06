#!/bin/bash
# Spikingformer — ETTh1 benchmark  (config: configs/Spikingformer/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/Spikingformer

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
    echo "Spikingformer ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/Spikingformer/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/Spikingformer/ETTh1_pl${pred_len}.txt"
done
