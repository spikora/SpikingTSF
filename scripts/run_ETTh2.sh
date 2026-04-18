#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

CONFIG="${CONFIG:-configs/SpikF/ETTh2.yaml}"
GPU="${GPU:-0}"
PRED_LENS=(${PRED_LENS:-96 192 336 720})

OUT_DIR="${OUT_DIR:-Output/ETTh2/SpikF}"
mkdir -p "${OUT_DIR}"

for pred_len in "${PRED_LENS[@]}"; do
    echo "ETTh2 pl=${pred_len}"
    python run_long.py \
        --config "${CONFIG}" \
        --pred_len "${pred_len}" \
        --gpu "${GPU}" \
        | tee "${OUT_DIR}/pl${pred_len}.txt"
done
