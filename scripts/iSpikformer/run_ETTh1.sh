#!/bin/bash
# iSpikformer — ETTh1 benchmark  (config: configs/iSpikformer/ETTh1.yaml)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/iSpikformer

export CUDA_VISIBLE_DEVICES=3

for pred_len in 96 192 336 720
do
    echo "iSpikformer ETTh1 pl=${pred_len}"
    python run_long.py \
        --config configs/iSpikformer/ETTh1.yaml \
        --pred_len ${pred_len} \
        | tee "Output/iSpikformer/ETTh1_pl${pred_len}.txt"
done
