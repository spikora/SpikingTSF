#!/bin/bash
# Spikingformer — ETTh1 benchmark (with ConvPE)
# Adapted from SeqSNN (Microsoft, MIT License)
for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "Spikingformer ETTh1 pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model Spikingformer \
            --data ETTh1 --data_path ETTh1.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --T 4 --tau 2.0 --levels 2 \
            --d_model 256 --n_heads 8 --d_ff 1024 \
            --common_thr 1.0 --qk_scale 0.125 \
            --dropout 0.1 \
            --train_epochs 10 --batch_size 32 --lr 5e-4 \
            --random_seed ${seed} \
            | tee "Output/Spikingformer/ETTh1_96-${pred_len}_seed${seed}.txt"
    done
done
