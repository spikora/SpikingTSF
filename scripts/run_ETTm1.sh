#!/bin/bash
for seed in 0 42 79
do
    for seq_len in 96
    do
        for pred_len in 96 192 336 720
        do
            echo ETTm1_${seq_len}-${pred_len}_seed${seed}
            python run_long.py \
                --data ETTm1 \
                --data_path ETTm1.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --batch_size 32 \
                --lr 5e-4 \
                --levels 1 \
                --patch_dim 32 \
                --T 16 \
                --hidden_dim 720 \
                --train_epochs 5 \
                --patience 3 \
                --model_name SpikF_ETTm1_input${seq_len}_output${pred_len} \
                --model SpikF \
                | tee "Output/ETTm1/SpikF/${seq_len}-${pred_len}-seed${seed}.txt"
        done
    done
done
