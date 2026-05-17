#!/bin/bash
# SpikGRU — ETTh2 benchmark
# Best hyperparameters from Optuna search (50 trials, best_val_mae=0.43767)
# Trial 20: tau=1.28, levels=2, alpha=256, encoder=conv, pe_type=none, scheduler=none
cd "$(dirname "$0")/../.." || exit 1
mkdir -p Output/SpikGRU
export CUDA_VISIBLE_DEVICES=1

for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "SpikGRU ETTh2 pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model SpikGRU \
            --data ETTh2 --data_path ETTh2.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --T 4 --tau 1.2791551285336131 --levels 2 --alpha 256 \
            --encoder_type conv \
            --pe_type none --pe_mode add \
            --lr 0.004687551265274829 \
            --batch_size 32 \
            --weight_decay 0.00033236764537473783 \
            --scheduler none \
            --grad_clip 0.0 \
            --train_epochs 300 --patience 15 \
            --random_seed ${seed} \
            | tee "Output/SpikGRU/ETTh2_96-${pred_len}_seed${seed}.txt"
    done
done
