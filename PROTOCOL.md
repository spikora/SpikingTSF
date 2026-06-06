# Experimental Protocol

This document describes the evaluation protocol used in SpikingTSF for long-term time-series forecasting.

---

## Datasets

| Dataset | Domain | Frequency | Variables |
|---------|--------|-----------|-----------|
| ETTh1 | Electricity (hourly) | 1h | 7 |
| ETTh2 | Electricity (hourly) | 1h | 7 |
| ETTm1 | Electricity (15-min) | 15min | 7 |
| ETTm2 | Electricity (15-min) | 15min | 7 |

Datasets are sourced from the [Autoformer repository](https://github.com/thuml/Autoformer). See [DATASETS.md](DATASETS.md) for placement instructions.

---

## Forecasting Setting

- **Task:** Multivariate long-term forecasting (`-M` flag)
- **Look-back window (input length):** 96
- **Prediction horizons:** 96, 192, 336, 720
- **Train / Validation / Test split:** Standard ETT protocol as implemented in `data_provider/ETT_data_loader.py`. Refer to the data loader for exact split boundaries.

---

## Evaluation Metrics

All metrics are computed on the test set:

| Metric | Description | Direction |
|--------|-------------|-----------|
| MSE | Mean Squared Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| RMSE | Root Mean Squared Error | Lower is better |
| R² | Coefficient of Determination | Higher is better |

Implementation: `utils/metrics.py`.

---

## Normalization

Per-sample instance normalization is applied in the data loader. This is consistent across all models.

---

## Reproducibility

- **Seeds:** 42, 1234, 3407 (3 independent runs per configuration)
- **Reported values:** Mean ± standard deviation across 3 seeds
- Hyperparameters are tuned per model per dataset using Optuna (see `hparam/`). The best hyperparameter configuration is then used for the 3-seed evaluation.
- All models use the **same look-back window, prediction horizon, and dataset split**.

---

## Hyperparameter Policy

- Hyperparameter tuning is performed independently per model and per dataset.
- Tuned hyperparameters include: learning rate, weight decay, batch size, number of layers, attention dimension, scheduler, and other model-specific parameters.
- The best trial from the Optuna study is used for the final 3-seed evaluation.


---

## Reporting

- Numbers are reported as **mean ± std** across 3 runs.
- Results are logged to `Output/<ModelName>/<dataset>_pl<horizon>.txt`.
- Do not report results from partial runs or single seeds as final results.
- Results from this library should not be used in publications without verifying them independently.
