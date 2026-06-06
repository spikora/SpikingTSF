#!/bin/bash
# TSFormer — ETTh1 benchmark  (config: configs/TSFormer/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/TSFormer

export CUDA_VISIBLE_DEVICES=2

for pred_len in 96 192 336 720
do
    echo "TSFormer ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/TSFormer/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/TSFormer/ETTh1_pl${pred_len}.txt"
done
