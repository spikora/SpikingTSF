# Results

Long-term forecasting results on the ETT benchmark (ETTh1, ETTh2, ETTm1, ETTm2).

**Protocol:** look-back window = 96, multivariate (`-M`), 3 independent runs with seeds {42, 1234, 3407}, per-sample instance normalisation. Numbers are **mean ± std** across the 3 runs. Lower is better for MSE and MAE.

Results are added here as experiments complete. The [README leaderboard](./README.md) is updated once a full sweep (all 4 horizons × 4 datasets) is finished for a model.

---

## SpikF

> SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting, ICML 2025  
> Config: `T=16`, `levels=1`, `alpha=2`, `lr=5e-4`, `bs=32`, `optimizer=Adam`, `scheduler=cosine`

### ETTh1

| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.3813 ± 0.0013 | 0.3899 ± 0.0008 |
| 192 | 0.4381 ± 0.0012 | 0.4223 ± 0.0001 |
| 336 | 0.4744 ± 0.0019 | 0.4408 ± 0.0010 |
| 720 | 0.4742 ± 0.0045 | 0.4584 ± 0.0025 |
| **Avg** | **0.4420** | **0.4279** |

### ETTh2

| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.2894 ± 0.0057 | 0.3356 ± 0.0033 |
| 192 | 0.3681 ± 0.0013 | 0.3844 ± 0.0004 |
| 336 | 0.4168 ± 0.0037 | 0.4211 ± 0.0006 |
| 720 | 0.4173 ± 0.0023 | 0.4354 ± 0.0014 |
| **Avg** | **0.3729** | **0.3941** |

### ETTm1

| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.3171 ± 0.0015 | 0.3444 ± 0.0004 |
| 192 | 0.3719 ± 0.0033 | 0.3735 ± 0.0011 |
| 336 | — | — |
| 720 | — | — |
| **Avg** | — | — |

### ETTm2

| Horizon | MSE | MAE |
|---------|-----|-----|
| 96 | 0.1746 ± 0.0013 | 0.2522 ± 0.0014 |
| 192 | 0.2405 ± 0.0008 | 0.2962 ± 0.0006 |
| 336 | 0.3033 ± 0.0017 | 0.3373 ± 0.0006 |
| 720 | — | — |
| **Avg** | — | — |

---

## iSpikformer

> Inverted Spiking Transformer (from SpikF codebase), ICML 2025

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## SpikeRNN

> Spiking Recurrent Network (from SpikF codebase), ICML 2025

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## SpikTCN

> Spiking Temporal Convolutional Network, SeqSNN (ICML 2024)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## SpikGRU

============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 96
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.5531      0.0003
MSE           0.7005      0.0000
RMSE          0.8370      0.0000
RSE           0.9648      0.0000
R2            0.1438      0.0002
MAPE         13.6839      0.0288
MSPE      55666.2383    286.7966
CORR         16.5343      0.0016
============================================================



============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 192
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.5661      0.0003
MSE           0.7177      0.0001
RMSE          0.8472      0.0000
RSE           0.9815      0.0000
R2            0.0789      0.0003
MAPE         14.2969      0.0313
MSPE      59043.5312    298.1634
CORR         13.3525      0.0018
============================================================


============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 336
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.5787      0.0003
MSE           0.7220      0.0002
RMSE          0.8497      0.0001
RSE           0.9962      0.0002
R2            0.0198      0.0002
MAPE         13.9223      0.0366
MSPE      55090.4336    323.3087
CORR         10.8420      0.0038
============================================================



============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 720
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.5954      0.0009
MSE           0.7123      0.0012
RMSE          0.8440      0.0007
RSE           1.0145      0.0009
R2           -0.0766      0.0003
MAPE         14.0335      0.0823
MSPE      55077.8906    741.3928
CORR          9.2597      0.0020
============================================================



# Etth2
============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 96
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.3871      0.0000
MSE           0.3529      0.0000
RMSE          0.5940      0.0000
RSE           0.9437      0.0000
R2            0.1428      0.0000
MAPE          1.5310      0.0006
MSPE        349.1496      0.6078
CORR         18.7766      0.0014
============================================================



============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 192
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.4290      0.0001
MSE           0.4275      0.0001
RMSE          0.6539      0.0001
RSE           1.0443      0.0001
R2           -0.0325      0.0001
MAPE          1.6531      0.0026
MSPE        391.0710      3.1973
CORR         13.2461      0.0084
============================================================




============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 336
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.4545      0.0000
MSE           0.4544      0.0000
RMSE          0.6741      0.0000
RSE           1.0820      0.0000
R2           -0.1227      0.0001
MAPE          1.8030      0.0018
MSPE        452.7217      1.8606
CORR         10.0051      0.0059
============================================================




============================================================
  Final Results  (3 runs, seeds=[42, 1234, 3407]) - pred 720
============================================================
Metric          Mean         Std
--------------------------------
MAE           0.4608      0.0000
MSE           0.4515      0.0000
RMSE          0.6719      0.0000
RSE           1.0896      0.0001
R2           -0.1681      0.0001
MAPE          2.0295      0.0006
MSPE        557.6428      0.7223
CORR          8.7751      0.0019
============================================================




> Spiking Gated Recurrent Unit, SeqSNN (ICML 2024)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## Spikformer

> Spiking Transformer with Spike-driven Self-Attention, SeqSNN (ICML 2024)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## Spikingformer

> Pre-LIF Spiking Transformer, SeqSNN (ICML 2024)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## QKFormer

> Token-Level Q/K Spiking Transformer, SeqSNN (ICML 2024)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## TSGRU

> Two-Compartment TS-LIF Gated Recurrent Unit, TS-LIF (ICLR 2025)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## TSTCN

> Two-Compartment TS-LIF Temporal Convolutional Network, TS-LIF (ICLR 2025)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## TSFormer

> Two-Compartment TS-LIF Inverted Transformer, TS-LIF (ICLR 2025)

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## ITransformer

> iTransformer: Inverted Transformers Are Effective for Time Series Forecasting, ICLR 2024

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |

---

## DLinear

> Are Transformers Effective for Time Series Forecasting?, AAAI 2023

| Dataset | 96 | 192 | 336 | 720 | Avg |
|---------|----|-----|-----|-----|-----|
| ETTh1 MSE | — | — | — | — | — |
| ETTh2 MSE | — | — | — | — | — |
| ETTm1 MSE | — | — | — | — | — |
| ETTm2 MSE | — | — | — | — | — |
