#!/bin/bash
# TSGRU — ETTh2 benchmark  (config: configs/TSGRU/ETTh2.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/TSGRU

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
    echo "TSGRU ETTh2 pl=${pred_len}"
    python run_long.py \
        --config configs/TSGRU/ETTh2.yaml \
        --pred_len ${pred_len} \
        | tee "Output/TSGRU/ETTh2_pl${pred_len}.txt"
done
