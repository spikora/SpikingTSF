#!/bin/bash
# SpikeRNN — ETTh1 benchmark  (config: configs/SpikeRNN/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikeRNN

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
    echo "SpikeRNN ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/SpikeRNN/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/SpikeRNN/ETTh1_pl${pred_len}.txt"
done
