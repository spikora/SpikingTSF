#!/bin/bash
# ITransformer — Weather benchmark
for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "ITransformer Weather pl=${pred_len} seed=${seed}"
        python run_long.py \
            --model ITransformer \
            --data weather --data_path weather.csv \
            --features M --seq_len 96 --pred_len ${pred_len} \
            --levels 3 \
            --d_model 512 --n_heads 8 --d_ff 2048 \
            --dropout 0.1 \
            --train_epochs 10 --batch_size 32 --lr 1e-4 \
            --random_seed ${seed} \
            | tee "Output/ITransformer/weather_96-${pred_len}_seed${seed}.txt"
    done
done
