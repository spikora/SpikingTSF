# SpikingTSF

SpikingTSF is an open-source benchmark library for **third-generation neural network (SNN) time series forecasting**. It is the **first work to bring together all SNN architectures for time series forecasting under one unified** training and evaluation framework, making it easy to compare methods fairly on the ETT benchmark suite.

---

🚩 **News** (2026.04) SpikingTSF is released as the first unified SNN time-series forecasting library. Implementations span eleven SNN architectures and two ANN baselines, all evaluated under a single consistent protocol on ETTh1 and ETTh2.

🚩 **News** (2026.05) [[SpikF]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf) (ICML 2025) introduces a spiking frequency-domain transformer for long-term forecasting.

🚩 **News** (2025.10) [[SeqSNN]](https://arxiv.org/abs/2501.16745) (NeurIPS 2025 (spotlight)) provides QKFormer, Spikingformer, and Spikformer — three spiking transformer variants with approximate RPE techniques such as Gray-PE and Log-PE

🚩 **News** (2025.03) [[TS-LIF]](https://arxiv.org/abs/2503.05108) (ICLR 2025) introduces two-compartment dendritic-somatic LIF neurons for richer temporal dynamics. TSFormer, TSGRU, and TSTCN are adapted from this work.

🚩 **News** (2024.05) [[SeqSNN]](https://arxiv.org/abs/2402.01533) (ICML 2024, Microsoft Research) provides a systematic sequential SNN framework for time series. SpikTCN, SpikeRNN, iSpikformer, and SpikGRU are adapted from this work.

🚩 **News** (2024.03) [[iTransformer]](https://arxiv.org/abs/2310.06625) (ICLR 2024) is included as an ANN upper-bound baseline.

🚩 **News** (2022.08) [[DLinear]](https://arxiv.org/abs/2205.13504) (AAAI 2023) is included as a lightweight ANN baseline.

---

## Leaderboard — Long-term Forecasting

Rankings are determined by average MSE across ETTh1 and ETTh2 (horizons 96 / 192 / 336 / 720).

| Ranking | Model | Paper | Spike? | ETTh1 MSE | ETTh2 MSE |
|---------|-------|-------|:------:|-----------|-----------|
| 🥇 1st | — | — | — | — | — |
| 🥈 2nd | — | — | — | — | — |
| 🥉 3rd | — | — | — | — | — |

> **Note:** ETTm1 and ETTm2 results will be added as we complete hyperparameter tuning. For full results with all metrics, see [Results.md](./RESULTS.md). For the experimental protocol, see [Protocol.md](./Protocol.md).

**All models in this library.** ☑ means the code is included and runnable.

- ☑ **SpikF** — SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting [[ICML 2025]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf) [[Code]](https://github.com/WWJ-creator/SpikF) | Spike ✅
- ☑ **Spikformer** — Spiking Transformer with Spike-driven Self-Attention [[SeqSNN, NeurIPS 2025 (spotlight)]](https://arxiv.org/abs/2501.16745) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **Spikingformer** — Pre-LIF Spiking Transformer [[SeqSNN, NeurIPS 2025 (spotlight)]](https://arxiv.org/abs/2501.16745) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **QKFormer** — Token-Level Q/K Attention Spiking Transformer [[SeqSNN, NeurIPS 2025 (spotlight)]](https://arxiv.org/abs/2501.16745) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **TSGRU** — Two-Compartment TS-LIF Gated Recurrent Unit [[TS-LIF, ICLR 2025]](https://arxiv.org/abs/2503.05108) [[Code]](https://github.com/kkking-kk/TS-LIF) | Spike ✅
- ☑ **TSTCN** — Two-Compartment TS-LIF Temporal Convolutional Network [[TS-LIF, ICLR 2025]](https://arxiv.org/abs/2503.05108) [[Code]](https://github.com/kkking-kk/TS-LIF) | Spike ✅
- ☑ **TSFormer** — Two-Compartment TS-LIF Inverted Transformer [[TS-LIF, ICLR 2025]](https://arxiv.org/abs/2503.05108) [[Code]](https://github.com/kkking-kk/TS-LIF) | Spike ✅
- ☑ **iSpikformer** — Inverted Spiking Transformer [[SeqSNN, ICML 2024]](https://arxiv.org/abs/2402.01533) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **SpikeRNN** — Spiking Recurrent Network [[SeqSNN, ICML 2024]](https://arxiv.org/abs/2402.01533) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **SpikTCN** — Spiking Temporal Convolutional Network [[SeqSNN, ICML 2024]](https://arxiv.org/abs/2402.01533) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **SpikGRU** — Spiking Gated Recurrent Unit [[SeqSNN, ICML 2024]](https://arxiv.org/abs/2402.01533) [[Code]](https://github.com/microsoft/SeqSNN) | Spike ✅
- ☑ **ITransformer** — iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/iTransformer) | Spike ❌
- ☑ **DLinear** — Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/abs/2205.13504) [[Code]](https://github.com/thuml/Time-Series-Library) | Spike ❌

---

## Getting Started

### Installation

1. Clone this repository.
   ```bash
   git clone https://github.com/jafarbakhshaliyev/SpikingTSF.git
   cd SpikingTSF
   ```

2. Install dependencies.
   ```bash
   # Python 3.9+
   pip install torch torchvision
   pip install spikingjelly
   pip install numpy pandas scikit-learn
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

All experiments are run through `run_long.py`. Provided shell scripts under `scripts/` reproduce all configurations.

```bash
# Run all models on ETTh1 and ETTh2
bash scripts/run_ETTh1.sh
bash scripts/run_ETTh2.sh
```

Outputs (training logs and final MAE/MSE/RMSE/R² per run plus mean ± std) are written to `Output/<dataset>/<model>/pl<horizon>.txt`.



## Project Architecture

```
SpikingTSF/
├── README.md                        # This file
├── run_long.py                      # Unified entry point — parses args and runs experiments
├── requirements.txt                 # pip dependency list
├── exp/                             # Experiment pipelines
│   ├── exp_basic.py                 # Base experiment class
│   └── exp_ETT.py                   # ETT forecasting logic (train / val / test)
├── models/                          # All model implementations
│   ├── SpikF.py                     # SpikF (ICML 2025)
│   ├── iSpikformer.py               # Inverted Spiking Transformer
│   ├── SpikeRNN.py                  # Spiking RNN
│   ├── SpikTCN.py                   # Spiking TCN
│   ├── SpikGRU.py                   # Spiking GRU
│   ├── Spikformer.py                # Spikformer
│   ├── Spikingformer.py             # Spikingformer
│   ├── QKFormer.py                  # QKFormer
│   ├── TSGRU.py                     # TS-LIF GRU
│   ├── TSTCN.py                     # TS-LIF TCN
│   ├── TSFormer.py                  # TS-LIF Transformer
│   ├── ITransformer.py              # iTransformer (ANN baseline)
│   ├── DLinear.py                   # DLinear (ANN baseline)
│   └── layers/                      # Shared SNN primitives and attention blocks
├── data_provider/                   # Dataset loaders
│   └── ETT_data_loader.py           # ETT sliding-window loader
├── utils/                           # Utility toolbox
│   ├── metrics.py                   # MSE / MAE / RMSE / R²
│   └── tools.py                     # EarlyStopping, LR scheduling
├── scripts/                         # Bash scripts for reproducible experiments
│   ├── run_ETTh1.sh
│   ├── run_ETTh2.sh
│   ├── run_ETTm1.sh
│   └── run_ETTm2.sh
├── configs/                         # YAML configuration files
├── datasets/long/                   # ETT CSV datasets
└── Output/                          # Experiment results
    └── <dataset>/<model>/pl<horizon>.txt
```

**E2E flow:** configure experiments via `scripts/*.sh` → run `python run_long.py ...` → `run_long.py` parses arguments and selects the model → `exp/exp_ETT.py` builds the dataset through `data_provider`, instantiates the network from `models`, and drives train/val/test with utilities in `utils` → metrics and results are written to `Output/`.

---

## Citation

If you use SpikingTSF in your research, please cite:

```bibtex
@inproceedings{wu25m,
  title     = {SpikF: Spiking Frequency-Domain Transformer for Time Series Forecasting},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025}
}
```

---

## Contact

If you have any questions or suggestions, feel free to reach out:

- **Jafar Bakhshaliyev** — [jafar.bakhshaliyev@gmail.com](mailto:bakhshaliyevj@uni-hildesheim.de)
- GitHub: [jafarbakhshaliyev](https://github.com/jafarbakhshaliyev)
- Website: [jafarbakhshaliyev.github.io](https://jafarbakhshaliyev.github.io/)

Or open an issue in this repository.

---

## Acknowledgement

SpikingTSF is built on top of the following excellent open-source projects:

- SNN backend: [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- SpikF family: [WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF)
- SeqSNN family: [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN)
- TS-LIF family: [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF)
- ANN baselines: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), [thuml/iTransformer](https://github.com/thuml/iTransformer)
- ETT datasets: [thuml/Autoformer](https://github.com/thuml/Autoformer)
