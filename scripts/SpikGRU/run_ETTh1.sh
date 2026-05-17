#!/bin/bash
# SpikGRU — ETTh1 benchmark
# Best hyperparameters from Optuna search (50 trials, best_val_mae=0.91866)
# Trial 48: tau=3.15, levels=1, alpha=128, encoder=delta, pe_type=conv, scheduler=cosine
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikGRU

export CUDA_VISIBLE_DEVICES=0

for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "SpikGRU ETTh1 pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model SpikGRU \
            --data ETTh1 --data_path ETTh1.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --T 4 --tau 3.149666943742141 --levels 1 --alpha 128 \
            --encoder_type delta \
            --pe_type conv --pe_mode add \
            --lr 0.004841718080373416 \
            --batch_size 32 \
            --weight_decay 0.00025005659370460736 \
            --scheduler cosine \
            --grad_clip 1.0 \
            --train_epochs 300 --patience 15 \
            --random_seed ${seed} \
            | tee "Output/SpikGRU/ETTh1_96-${pred_len}_seed${seed}.txt"
    done
done
