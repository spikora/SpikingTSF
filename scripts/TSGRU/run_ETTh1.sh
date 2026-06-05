#!/bin/bash
# TSGRU — ETTh1 benchmark  (config: configs/TSGRU/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/TSGRU

export CUDA_VISIBLE_DEVICES=1

for pred_len in 720
do
    echo "TSGRU ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/TSGRU/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/TSGRU/ETTh1_pl${pred_len}.txt"
done
