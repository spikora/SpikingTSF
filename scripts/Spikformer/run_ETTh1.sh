#!/bin/bash
# Spikformer — ETTh1 benchmark
# Adapted from SeqSNN (Microsoft, MIT License)
for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "Spikformer ETTh1 pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model Spikformer \
            --data ETTh1 --data_path ETTh1.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --T 4 --tau 2.0 --levels 2 \
            --d_model 256 --n_heads 8 --d_ff 1024 \
            --common_thr 1.0 --qk_scale 0.125 \
            --encoder_type conv \
            --dropout 0.1 \
            --train_epochs 10 --batch_size 32 --lr 5e-4 \
            --random_seed ${seed} \
            | tee "Output/Spikformer/ETTh1_96-${pred_len}_seed${seed}.txt"
    done
done
