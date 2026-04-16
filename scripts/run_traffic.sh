#!/bin/bash
for seed in 0 42 79
do
    for seq_len in 96
    do
        for pred_len in 96 192 336 720
        do
            echo traffic_${seq_len}-${pred_len}_seed${seed}
            python run_long.py \
                --data traffic \
                --data_path traffic.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --batch_size 8 \
                --lr 5e-4 \
                --levels 2 \
                --patch_dim 16 \
                --T 16 \
                --hidden_dim 720 \
                --train_epochs 15 \
                --patience 3 \
                --model_name SpikF_traffic_input${seq_len}_output${pred_len} \
                --model SpikF \
                | tee "Output/traffic/SpikF/${seq_len}-${pred_len}-seed${seed}.txt"
        done
    done
done
