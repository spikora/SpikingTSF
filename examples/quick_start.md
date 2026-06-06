# Quick Start

This example shows how to run a single model on ETTh1 using the provided scripts.

## Prerequisites

1. Install dependencies (see README Installation section).
2. Download the ETT datasets and place them under `datasets/long/` (see [DATASETS.md](../DATASETS.md)).

## Running SpikF on ETTh1

```bash
# Clone the repository
git clone https://github.com/spikora/SpikingTSF.git
cd SpikingTSF

# Activate your environment
conda activate spikingtsf
# or: source .venv/bin/activate

# Run SpikF on ETTh1 with the best known hyperparameters
bash scripts/SpikF/run_ETTh1.sh
```

This script runs 3 independent seeds (42, 1234, 3407) for prediction horizons 96, 192, 336, and 720.
Results are written to `Output/SpikF/ETTh1_pl<horizon>.txt`.

## Running a Single Configuration Manually

```bash
python run_long.py \
    --model SpikF \
    --data ETTh1 \
    --root_path datasets/long/ \
    --data_path ETTh1.csv \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --seed 42
```

## Output Format

Each output file contains per-run metrics and a summary:

```
============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407])
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.3917      0.0007
MSE           0.3779      0.0016
RMSE          0.6147      0.0013
R2            0.4562      0.0008
============================================================
```

## Next Steps

- See `scripts/` for scripts for other models and datasets.
- See [PROTOCOL.md](../PROTOCOL.md) for the full experimental protocol.
- See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for a complete reproduction guide.
