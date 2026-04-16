#!/bin/bash
# ITransformer — ETTh1 benchmark (ANN baseline)
# Adapted from SeqSNN (Microsoft) / iTransformer (THUML), MIT License
for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "ITransformer ETTh1 pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model ITransformer \
            --data ETTh1 --data_path ETTh1.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --levels 2 \
            --d_model 512 --n_heads 8 --d_ff 2048 \
            --dropout 0.1 \
            --train_epochs 10 --batch_size 32 --lr 1e-4 \
            --random_seed ${seed} \
            | tee "Output/ITransformer/ETTh1_96-${pred_len}_seed${seed}.txt"
    done
done
