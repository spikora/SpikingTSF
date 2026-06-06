# Reproducibility Guide

This document explains how to reproduce the experiments in SpikingTSF from scratch.

---

## 1. Environment Setup

```bash
git clone https://github.com/spikora/SpikingTSF.git
cd SpikingTSF

# Option A: pip
pip install torch>=1.13.0 torchvision
pip install spikingjelly==0.0.0.0.14
pip install numpy pandas scikit-learn scipy matplotlib tqdm pyyaml

# Option B: conda
conda env create -f environment.yml
conda activate spikingtsf
```

Tested with Python 3.9 and 3.10. GPU is required for most models; the codebase uses CUDA via PyTorch.

---

## 2. Dataset Placement

Download the ETT CSV files and place them at:

```
SpikingTSF/
└── datasets/
    └── long/
        ├── ETTh1.csv
        ├── ETTh2.csv
        ├── ETTm1.csv
        └── ETTm2.csv
```

See [DATASETS.md](DATASETS.md) for download sources.

---

## 3. Running Experiments

Each model has shell scripts under `scripts/<ModelName>/`. For example:

```bash
# SpikF on ETTh1 (all 4 horizons, 3 seeds)
bash scripts/SpikF/run_ETTh1.sh

# TSGRU on ETTh2
bash scripts/TSGRU/run_ETTh2.sh

# iSpikformer on ETTh1
bash scripts/iSpikformer/run_ETTh1.sh
```

Scripts call `python run_long.py` with the hyperparameters recorded in `configs/<ModelName>/`.

---

## 4. Seeds

All final evaluations use seeds **42, 1234, and 3407** — three independent runs. Results are reported as mean ± std across these seeds.

Do not use only a single seed for final reported numbers.

---

## 5. Output Format

Results are written to:

```
Output/<ModelName>/<dataset>_pl<horizon>.txt
```

Each file contains per-run metrics (MAE, MSE, RMSE, R², etc.) and a summary block:

```
============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407])
============================================================
Metric          Mean         Std
--------------------------------
MAE           X.XXXX      X.XXXX
MSE           X.XXXX      X.XXXX
...
```

---

## 6. Hyperparameter Tuning

Tuning is performed with Optuna. Scripts are in `scripts/hparam/` and the tuning code is in `hparam/tune_<ModelName>.py`. The best trial per model/dataset is recorded in `scripts/<ModelName>/BEST_HPARAMS.md` where available. The tuned hyperparameters are incorporated into the run scripts.

---

## 7. Submitting Reproduced Results

If you independently reproduce results for a model/dataset/horizon combination, please submit them via the [Result Submission issue template](https://github.com/spikora/SpikingTSF/issues/new?template=result_submission.yml). Include:

- The exact script or command used
- Seed(s) used
- Hardware (GPU model, CUDA version)
- Full output log or the aggregated metrics
- Environment (Python + PyTorch + SpikingJelly versions)

See [docs/leaderboard_policy.md](docs/leaderboard_policy.md) for the acceptance criteria.
