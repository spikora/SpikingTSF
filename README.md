# SpikingTSF

SpikingTSF is an open-source benchmark library for **spiking neural network (SNN) time series forecasting**. It brings together state-of-the-art SNN architectures from multiple papers under one consistent training and evaluation framework, making it easy to compare methods fairly on the ETT benchmark suite.

> **Evaluation protocol:** All models use **per-sample instance normalisation** (subtract mean, divide by std per sample, reverse after prediction), a fixed look-back window of 96, and three independent runs with seeds {42, 1234, 3407}. Reported numbers are mean ± std across the three runs.

---

🚩 **News** (2026.04) SpikingTSF is released as a unified SNN time-series forecasting library. Implementations cover five papers spanning 2023–2025, eleven SNN architectures, and two ANN baselines, all evaluated under a single consistent protocol on ETTh1, ETTh2, ETTm1, and ETTm2.

🚩 **News** (2025) [[SpikF]](https://github.com/WWJ-creator/SpikF) introduces a spiking frequency-domain transformer for long-term forecasting. SpikF, iSpikformer, and SpikeRNN from this codebase are now included.

🚩 **News** (2025) [[TS-LIF]](https://github.com/kkking-kk/TS-LIF) (ICLR 2025) introduces two-compartment dendritic-somatic LIF neurons for richer temporal dynamics. TSGRU, TSTCN, and TSFormer are adapted from this work.

🚩 **News** (2024) [[SeqSNN]](https://github.com/microsoft/SeqSNN) (ICML 2024, Microsoft Research) provides a systematic sequential SNN framework for time series. SpikTCN, SpikGRU, Spikformer, Spikingformer, and QKFormer are adapted from this work.

🚩 **News** (2024) [[iTransformer]](https://github.com/thuml/iTransformer) (ICLR 2024) is included as an ANN upper-bound baseline, alongside [[DLinear]](https://github.com/thuml/Time-Series-Library) (AAAI 2023) as a lightweight ANN baseline.

---

## Leaderboard — Long-term Forecasting (ETT Average, Look-Back-96)

Rankings are determined by average MSE across ETTh1, ETTh2, ETTm1, and ETTm2 (horizons 96 / 192 / 336 / 720). Results are being populated as experiments complete — see [RESULTS.md](./RESULTS.md) for per-dataset per-horizon breakdowns.

| Ranking | Model | Paper | Pure Spike? | Avg MSE | Avg MAE |
|---------|-------|-------|:-----------:|---------|---------|
| 🥇 1st | — | — | — | — | — |
| 🥈 2nd | — | — | — | — | — |
| 🥉 3rd | — | — | — | — | — |

**Note:** Leaderboard will be updated once all models finish their ETT runs. The table above will be replaced with final ranked numbers. Partial results (SpikF complete) are available in [RESULTS.md](./RESULTS.md).

**All models in this library.** ☑ means the code is included and runnable.

- [x] **SpikF** — SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting [[ICML 2025]](https://github.com/WWJ-creator/SpikF) | Pure Spike ✅
- [x] **iSpikformer** — Inverted Spiking Transformer (from SpikF codebase) | Pure Spike ✅
- [x] **SpikeRNN** — Spiking Recurrent Network (from SpikF codebase) | Pure Spike ✅
- [x] **SpikTCN** — Spiking Temporal Convolutional Network [[SeqSNN, ICML 2024]](https://github.com/microsoft/SeqSNN) | Pure Spike ✅
- [x] **SpikGRU** — Spiking Gated Recurrent Unit [[SeqSNN, ICML 2024]](https://github.com/microsoft/SeqSNN) | Pure Spike ✅
- [x] **Spikformer** — Spiking Transformer with Spike-driven Self-Attention [[SeqSNN, ICML 2024]](https://github.com/microsoft/SeqSNN) | Pure Spike ✅
- [x] **Spikingformer** — Pre-LIF Spiking Transformer [[SeqSNN, ICML 2024]](https://github.com/microsoft/SeqSNN) | Pure Spike ✅
- [x] **QKFormer** — Token-Level Q/K Attention Spiking Transformer [[SeqSNN, ICML 2024]](https://github.com/microsoft/SeqSNN) | Pure Spike ✅
- [x] **TSGRU** — Two-Compartment TS-LIF Gated Recurrent Unit [[TS-LIF, ICLR 2025]](https://github.com/kkking-kk/TS-LIF) | Pure Spike ✅
- [x] **TSTCN** — Two-Compartment TS-LIF Temporal Convolutional Network [[TS-LIF, ICLR 2025]](https://github.com/kkking-kk/TS-LIF) | Pure Spike ✅
- [x] **TSFormer** — Two-Compartment TS-LIF Inverted Transformer [[TS-LIF, ICLR 2025]](https://github.com/kkking-kk/TS-LIF) | Pure Spike ✅
- [x] **ITransformer** — iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) | Pure Spike ❌
- [x] **DLinear** — Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/abs/2205.13504) | Pure Spike ❌

---

## Implemented Papers

Papers are listed in chronological order of publication. Each entry links to the original repository and shows which models in this library are derived from it.

### 2023

- [x] **DLinear** — Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/abs/2205.13504) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
  - Models: `DLinear`
  - Lightweight ANN decomposition baseline (trend + seasonal linear layers). Included as a strong non-spiking reference point.

### 2024

- [x] **SeqSNN** — Sequential Spiking Neural Networks for Time Series Forecasting [[ICML 2024]](https://github.com/microsoft/SeqSNN)
  - Models: `SpikTCN`, `SpikGRU`, `Spikformer`, `Spikingformer`, `QKFormer`
  - Microsoft Research framework providing a systematic comparison of SNN architectures (TCN, GRU, and three Transformer variants) on time-series tasks. Uses the `activation_based` SpikingJelly backend for Transformer variants and clock-driven backend for recurrent/convolutional models.

- [x] **iTransformer** — iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/iTransformer)
  - Models: `ITransformer`
  - ANN baseline that inverts the attention axis to treat variates as tokens. Included to provide an ANN upper bound for the same channel-as-token design adopted by SNN counterparts.

### 2025

- [x] **TS-LIF** — TS-LIF: Two-Compartment Dendritic-Somatic Spiking Neuron for Time Series Forecasting [[ICLR 2025]](https://github.com/kkking-kk/TS-LIF) [[arXiv]](https://arxiv.org/abs/2503.05108)
  - Models: `TSGRU`, `TSTCN`, `TSFormer`
  - Proposes a biologically-inspired two-compartment neuron (dendritic `v1` + somatic `v2`) with learnable temporal dynamics. Drop-in replacement for standard LIF nodes inside any SNN architecture.

- [x] **SpikF** — SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting [[ICML 2025]](https://github.com/WWJ-creator/SpikF)
  - Models: `SpikF`, `iSpikformer`, `SpikeRNN`
  - Introduces Spiking Patch Embedding (SPE) and a Spiking Fourier Selection (SFS) block that processes time-series patches in the frequency domain with binary spike activations throughout.

---

## Getting Started

### Installation

```bash
# Python 3.9+
pip install torch torchvision
pip install spikingjelly
pip install numpy pandas scikit-learn optuna
```

### Prepare Data

Download the ETT datasets and place them under `datasets/long/`:

```
datasets/long/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
└── ETTm2.csv
```

The ETT datasets are available from the [Autoformer repository](https://github.com/thuml/Autoformer) or [Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library).

### Run Experiments

All experiments are run through `run_long.py`. Configuration can be supplied via a YAML file under `configs/` or directly as command-line arguments. Provided shell scripts under `scripts/` reproduce all paper configurations.

```bash
# SpikF on all ETT datasets (all horizons, 3 seeds each)
bash scripts/run_ETTh1.sh
bash scripts/run_ETTh2.sh
bash scripts/run_ETTm1.sh
bash scripts/run_ETTm2.sh
```

Outputs (training logs and final MAE/MSE/RMSE/R² per run plus mean ± std) are written to `Output/<dataset>/<model>/pl<horizon>.txt`.

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--T` | SNN simulation time steps | 16 |
| `--tau` | LIF membrane time constant | 2.0 |
| `--levels` | Number of blocks / encoder layers | 2 |
| `--alpha` | Capacity knob (maps to d_model or hidden_dim) | 2.0 |
| `--seq_len` | Look-back window length | 96 |
| `--pred_len` | Forecast horizon | 336 |
| `--itr` | Independent runs (seeds: 42, 1234, 3407) | 3 |
| `--encoder_type` | Spike encoding scheme: `conv`, `delta`, `repeat` | `conv` |
| `--pe_type` | Positional encoding: `none`, `learn`, `static`, `conv`, `neuron`, `random` | `none` |
| `--loss` | Training loss: `mae` or `mse` | `mae` |
| `--scheduler` | LR schedule: `cosine`, `step`, `none` | `cosine` |

A full parameter reference is in [Parameters.md](./Paramters.md).

---

## Results

Detailed per-dataset, per-horizon results (mean ± std across 3 seeds) are in **[RESULTS.md](./RESULTS.md)**.

Summary of completed runs (averaged across horizons 96 / 192 / 336 / 720):

| Model | ETTh1 MSE | ETTh2 MSE | ETTm1 MSE | ETTm2 MSE | Status |
|-------|-----------|-----------|-----------|-----------|--------|
| **SpikF** | 0.4420 | 0.3729 | — | — | Partial (ETTm1 pl336+ and ETTm2 pl720 in progress) |
| iSpikformer | — | — | — | — | Pending |
| SpikeRNN | — | — | — | — | Pending |
| SpikTCN | — | — | — | — | Pending |
| SpikGRU | — | — | — | — | Pending |
| Spikformer | — | — | — | — | Pending |
| Spikingformer | — | — | — | — | Pending |
| QKFormer | — | — | — | — | Pending |
| TSGRU | — | — | — | — | Pending |
| TSTCN | — | — | — | — | Pending |
| TSFormer | — | — | — | — | Pending |
| ITransformer | — | — | — | — | Pending |
| DLinear | — | — | — | — | Pending |

---

## Citation

If you use SpikingTSF in your research, please cite the papers corresponding to the models you use:

```bibtex
@inproceedings{spikf2025,
  title     = {SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}

@inproceedings{seqsnn2024,
  title     = {Sequential Spiking Neural Networks for Time Series Forecasting},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}

@inproceedings{tslif2025,
  title     = {TS-LIF: Two-Compartment Dendritic-Somatic Spiking Neuron for Time Series Forecasting},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}

@inproceedings{itransformer2024,
  title     = {iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@inproceedings{dlinear2023,
  title     = {Are Transformers Effective for Time Series Forecasting?},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2023}
}
```

---

## Acknowledgement

SpikingTSF is built on top of the following excellent open-source projects:

- SNN backend: [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- SpikF family: [WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF)
- SeqSNN family: [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN)
- TS-LIF family: [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF)
- ANN baselines: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), [thuml/iTransformer](https://github.com/thuml/iTransformer)
- ETT datasets: [thuml/Autoformer](https://github.com/thuml/Autoformer)
