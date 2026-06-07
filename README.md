# SpikingTSF

**A unified benchmark library for spiking neural network based time-series forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange.svg)](https://pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

SpikingTSF is an open-source benchmark library for **spiking neural network (SNN) based time-series forecasting**. It is the first work to bring together all major SNN forecasting architectures under one unified training and evaluation framework, enabling fair and reproducible comparison on standard benchmark datasets.

---

## Why SpikingTSF?

SNN-based time-series forecasting is an active but fragmented research area. Existing works are scattered across incompatible codebases, use different dataset splits, normalization strategies, evaluation metrics, and hyperparameter protocols — making direct comparison unreliable or impossible.

**SpikingTSF addresses this by providing:**

- A **single entry point** (`run_long.py`) for training and evaluating all models
- **Unified data loaders, metrics, and train/val/test splits** across all models and datasets
- **Reproducible scripts** under `scripts/` for every model and dataset combination
- A **shared experimental protocol** documented in [PROTOCOL.md](PROTOCOL.md)
- **Multi-seed evaluation** (seeds 42, 1234, 3407) with mean ± std reporting
- **Hyperparameter tuning infrastructure** (Optuna) applied consistently

SpikingTSF supports both SNN models and ANN baselines, enabling honest comparison of energy-efficient spiking architectures against conventional deep learning methods.

---

## Key Features

- **13 models integrated:** 11 SNN architectures + 2 ANN baselines, all runnable from the same interface
- **4 ETT datasets:** ETTh1, ETTh2, ETTm1, ETTm2 — with more datasets planned
- **4 forecasting horizons:** 96, 192, 336, 720
- **4 evaluation metrics:** MSE, MAE, RMSE, R²
- **Consistent protocol:** same input length, split boundaries, normalization, and seeds for all models
- **Optuna hyperparameter tuning** per model per dataset, included in the repository
- **Clean output logging** with per-run and aggregated metrics in `Output/`
- **Extensible design** — adding a new model requires only a model file, config, and script (see [docs/adding_a_model.md](docs/adding_a_model.md))

---

## News

🚩 **(2026.06)** [[SpikF-GO]](https://github.com/jafarbakhshaliyev/SpikF-GO) (accepted to ECML PKDD 2026) introduces Spiking Fourier Graph Operators for multivariate time-series forecasting, with an optional spike-domain CPG positional encoding. 

🚩 **(2026.04)** SpikingTSF is released as the first unified SNN time-series forecasting library. Implementations span eleven SNN architectures and two ANN baselines, all evaluated under a single consistent protocol on ETTh1 and ETTh2.

🚩 **(2026.05)** [[SpikF]](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf) (ICML 2025) introduces a spiking frequency-domain transformer for long-term forecasting.

🚩 **(2025.10)** [[SeqSNN]](https://arxiv.org/abs/2501.16745) (NeurIPS 2025 spotlight) provides QKFormer, Spikingformer, and Spikformer — three spiking transformer variants with approximate RPE techniques such as Gray-PE and Log-PE.

🚩 **(2025.03)** [[TS-LIF]](https://arxiv.org/abs/2503.05108) (ICLR 2025) introduces two-compartment dendritic-somatic LIF neurons for richer temporal dynamics. TSFormer, TSGRU, and TSTCN are adapted from this work.

🚩 **(2024.05)** [[SeqSNN]](https://arxiv.org/abs/2402.01533) (ICML 2024, Microsoft Research) provides a systematic sequential SNN framework for time series. SpikTCN, SpikeRNN, iSpikformer, and SpikGRU are adapted from this work.

🚩 **(2024.03)** [[iTransformer]](https://arxiv.org/abs/2310.06625) (ICLR 2024) is included as an ANN upper-bound baseline.

🚩 **(2022.08)** [[DLinear]](https://arxiv.org/abs/2205.13504) (AAAI 2023) is included as a lightweight ANN baseline.

---

## Leaderboard — Long-term Forecasting

Rankings are determined by average MSE across ETTh1 and ETTh2 (horizons 96 / 192 / 336 / 720).

| Ranking | Model | Paper | Spike? | ETTh1 avg MSE | ETTh2 avg MSE |
|---------|-------|-------|:------:|:-------------:|:-------------:|
| 🥇 1st | — | — | — | — | — |
| 🥈 2nd | — | — | — | — | — |
| 🥉 3rd | — | — | — | — | — |

> **Note:** Leaderboard rankings will be filled in as verified results become available for all models across all horizons. For partial and in-progress results, see [RESULTS.md](RESULTS.md). For the experimental protocol, see [PROTOCOL.md](PROTOCOL.md).

---

## Model Zoo

Full details — source papers, source repositories, file paths, and verification status — are in [MODEL_ZOO.md](MODEL_ZOO.md).

| Model | Type | Spike? | Source | Status |
|-------|------|:------:|--------|--------|
| **SpikF-GO** | Spiking Fourier Graph Operator (freq. domain, optional CPG-PE) | ✅ | Bakhshaliyev & Landwehr, *accepted to ECML PKDD 2026* — [jafarbakhshaliyev/SpikF-GO](https://github.com/jafarbakhshaliyev/SpikF-GO) | Runnable |
| **SpikF** | Transformer (freq. domain) | ✅ | [Wu et al., ICML 2025](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf) | Runnable |
| **Spikformer** | Transformer (spike-driven SA) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | Runnable |
| **Spikingformer** | Transformer (pre-LIF) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | Runnable, ETTh1+ETTh2 verified |
| **QKFormer** | Transformer (token-level Q/K) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | Runnable |
| **TSGRU** | GRU (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | Runnable, ETTh1+ETTh2 verified |
| **TSTCN** | TCN (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | Runnable, ETTh1 verified |
| **TSFormer** | Transformer (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | Runnable, ETTh1 verified |
| **iSpikformer** | Inverted Spiking Transformer | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | Runnable, ETTh1 verified |
| **SpikeRNN** | Spiking Recurrent Network | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | Runnable, ETTh1+ETTh2 verified |
| **SpikTCN** | Spiking TCN | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | Runnable, ETTh1 partial |
| **SpikGRU** | Spiking GRU | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | Runnable, ETTh1+ETTh2 verified |
| **iTransformer** | Inverted Transformer (ANN) | ❌ | [Liu et al., ICLR 2024](https://arxiv.org/abs/2310.06625) | Runnable |
| **DLinear** | Decomposition Linear (ANN) | ❌ | [Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504) | Runnable |

---

## Installation

1. Clone this repository.
   ```bash
   git clone https://github.com/spikora/SpikingTSF.git
   cd SpikingTSF
   ```

2. Install dependencies.

   **Option A — pip:**
   ```bash
   pip install torch>=1.13.0 torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install spikingjelly==0.0.0.0.14
   pip install numpy pandas scikit-learn scipy matplotlib tqdm pyyaml timm optuna
   ```

   **Option B — conda:**
   ```bash
   conda env create -f environment.yml
   conda activate spikingtsf
   ```

   Tested with Python 3.9 and 3.10. A GPU is required for most models.

---

## Prepare Data

Download the ETT datasets and place them under `datasets/long/`:

```
SpikingTSF/
└── datasets/
    └── long/
        ├── ETTh1.csv
        ├── ETTh2.csv
        ├── ETTm1.csv
        └── ETTm2.csv
```

ETT datasets are available from the [Autoformer repository](https://github.com/thuml/Autoformer) or [Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library). See [DATASETS.md](DATASETS.md) for details.

---

## Run Experiments

All experiments are run through `run_long.py`. Shell scripts under `scripts/` reproduce all configurations.

```bash
# Run SpikF on ETTh1 (all horizons, 3 seeds)
bash scripts/SpikF/run_ETTh1.sh

# Run TSGRU on ETTh2
bash scripts/TSGRU/run_ETTh2.sh

# Run all ETTh1 models
bash scripts/run_ETTh1.sh
```

Outputs (per-run metrics and mean ± std across 3 seeds) are written to `Output/<ModelName>/<dataset>_pl<horizon>.txt`.

You can also run a single configuration manually:

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

---

## Project Structure

```
SpikingTSF/
├── run_long.py                      # Unified entry point — parses args and runs experiments
├── requirements.txt                 # pip dependency list
├── environment.yml                  # conda environment
├── exp/                             # Experiment pipelines
│   ├── exp_basic.py                 # Base experiment class
│   └── exp_ETT.py                   # ETT forecasting logic (train / val / test)
├── models/                          # All model implementations
│   ├── SpikF_GO.py                  # SpikF-GO (ours, accepted ECML PKDD 2026)
│   ├── SpikF.py                     # SpikF (ICML 2025)
│   ├── Spikformer.py                # Spikformer
│   ├── Spikingformer.py             # Spikingformer
│   ├── QKFormer.py                  # QKFormer
│   ├── TSGRU.py                     # TS-LIF GRU
│   ├── TSTCN.py                     # TS-LIF TCN
│   ├── TSFormer.py                  # TS-LIF Transformer
│   ├── iSpikformer.py               # Inverted Spiking Transformer
│   ├── SpikeRNN.py                  # Spiking RNN
│   ├── SpikTCN.py                   # Spiking TCN
│   ├── SpikGRU.py                   # Spiking GRU
│   ├── ITransformer.py              # iTransformer (ANN baseline)
│   ├── DLinear.py                   # DLinear (ANN baseline)
│   └── layers/                      # Shared SNN primitives and attention blocks
├── data_provider/                   # Dataset loaders
│   └── ETT_data_loader.py           # ETT sliding-window loader
├── utils/                           # Utility toolbox
│   ├── metrics.py                   # MSE / MAE / RMSE / R²
│   └── tools.py                     # EarlyStopping, LR scheduling
├── configs/                         # YAML configuration files per model
├── scripts/                         # Bash scripts for reproducible experiments
├── hparam/                          # Optuna hyperparameter tuning
├── tests/                           # Basic import tests
├── examples/                        # Quick-start examples
├── docs/                            # Extended documentation
├── datasets/long/                   # ETT CSV datasets (not committed)
└── Output/                          # Experiment results (not committed)
```

**E2E flow:** `scripts/*.sh` → `python run_long.py` → parses args and selects model → `exp/exp_ETT.py` builds the dataset through `data_provider`, instantiates the network from `models`, and drives train/val/test with utilities in `utils` → metrics written to `Output/`.

---

## Reproducibility

SpikingTSF is designed for reproducibility. Key guarantees:

- All models use the **same input length (96), dataset splits, normalization, and seeds (42, 1234, 3407)**
- Hyperparameters are tuned with Optuna and recorded in `configs/` and `hparam/`
- Results are reported as **mean ± std** across 3 runs — no cherry-picking
- Scripts in `scripts/` reproduce every reported configuration

For a complete step-by-step reproduction guide, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Contributing

Contributions are welcome. You can help by:

- Reporting bugs via [GitHub Issues](https://github.com/spikora/SpikingTSF/issues)
- Submitting reproduced results via the [Result Submission template](https://github.com/spikora/SpikingTSF/issues/new?template=result_submission.yml)
- Adding new model implementations (see [docs/adding_a_model.md](docs/adding_a_model.md))
- Improving documentation

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request. Do not commit datasets, checkpoints, or output files.

---

## Citation

If you use SpikingTSF as a benchmark framework, codebase, or experimental platform, please cite our SpikF-GO / SpikingTSF paper. If you use a specific model implementation, please also cite the corresponding original paper.

**SpikF-GO (SpikingTSF paper) — accepted to ECML PKDD 2026:**
```bibtex
@inproceedings{bakhshaliyev2026spikfgo,
  title     = {{SpikF-GO}: Spiking Fourier Graph Operators for Multivariate Time Series Forecasting},
  author    = {Bakhshaliyev, Jafar and Landwehr, Niels},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles
               and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2026},
  note      = {To appear; full proceedings citation and DOI will be added once available}
}
```

**SpikingTSF software:**
```bibtex
@misc{spikingtsf2026,
  title        = {{SpikingTSF}: A Unified Benchmark Library for Spiking Neural Network Based Time-Series Forecasting},
  author       = {Bakhshaliyev, Jafar},
  year         = {2026},
  howpublished = {\url{https://github.com/spikora/SpikingTSF}},
  note         = {Open-source software library}
}
```

For model-specific citations, see [docs/citation_policy.md](docs/citation_policy.md) and [MODEL_ZOO.md](MODEL_ZOO.md).

---

## Acknowledgements

SpikingTSF builds upon the following excellent open-source projects:

- **SNN backend:** [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- **SpikF-GO:** [jafarbakhshaliyev/SpikF-GO](https://github.com/jafarbakhshaliyev/SpikF-GO) (Bakhshaliyev & Landwehr, accepted to ECML PKDD 2026 — our own paper)
- **SpikF family:** [WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF)
- **SeqSNN family:** [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN)
- **TS-LIF family:** [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF)
- **ANN baselines:** [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), [thuml/iTransformer](https://github.com/thuml/iTransformer)
- **ETT datasets:** [thuml/Autoformer](https://github.com/thuml/Autoformer)

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full details and [NOTICE](NOTICE) for license information.

---

## Contact

- **Jafar Bakhshaliyev** — [jafar.bakhshaliyev@gmail.com](mailto:jafar.bakhshaliyev@gmail.com)
- GitHub: [jafarbakhshaliyev](https://github.com/jafarbakhshaliyev)
- Website: [jafarbakhshaliyev.github.io](https://jafarbakhshaliyev.github.io/)

For bugs and feature requests, open an issue. For research collaboration, use the email above. See [SUPPORT.md](SUPPORT.md) for support channels.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Some model implementations are adapted from prior open-source research projects with their own license and citation requirements. See [NOTICE](NOTICE) and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).
