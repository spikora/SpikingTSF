#!/bin/bash
# SpikeRNN — ETTh2 benchmark  (config: configs/SpikeRNN/ETTh2.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikeRNN

export CUDA_VISIBLE_DEVICES=1

for pred_len in 96 192 336 720
do
    echo "SpikeRNN ETTh2 pl=${pred_len}"
    python run_long.py \
        --config configs/SpikeRNN/ETTh2.yaml \
        --pred_len ${pred_len} \
        | tee "Output/SpikeRNN/ETTh2_pl${pred_len}.txt"
done
