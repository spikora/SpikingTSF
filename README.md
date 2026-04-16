# SpikingTSF

A unified library for **spiking neural network (SNN) time series forecasting**, combining state-of-the-art SNN architectures with an ANN baseline under one consistent training and evaluation framework.

---

## Models

| Model | Paper / Origin | Venue | Backend | Tokens |
|-------|---------------|-------|---------|--------|
| **SpikF** | [SpikF: Spiking Frequency-Domain Transformer](https://github.com/WWJ-creator/SpikF) | — | clock_driven | patch |
| **iSpikformer** | Inverted Spike Transformer (SpikF repo) | — | clock_driven | patch |
| **SpikeRNN** | Spike Recurrent Network (SpikF repo) | — | clock_driven | patch |
| **SpikTCN** | Adapted from [SeqSNN](https://github.com/microsoft/SeqSNN) | ICML 2024 | clock_driven | channel-ind. |
| **SpikGRU** | Adapted from [SeqSNN](https://github.com/microsoft/SeqSNN) | ICML 2024 | clock_driven | channel-ind. |
| **TSLIF** | Inspired by [TS-LIF](https://github.com/kkking-kk/TS-LIF) | ICLR 2025 | clock_driven | patch |
| **Spikformer** | Adapted from [SeqSNN](https://github.com/microsoft/SeqSNN) | ICML 2024 | activation_based | time-step |
| **Spikingformer** | Adapted from [SeqSNN](https://github.com/microsoft/SeqSNN) | ICML 2024 | activation_based | time-step |
| **DLinear** | [Are Transformers Effective for TSF?](https://github.com/thuml/Time-Series-Library) | AAAI 2023 | ANN | — |
| **ITransformer** | Adapted from [SeqSNN](https://github.com/microsoft/SeqSNN) / [iTransformer](https://github.com/thuml/iTransformer) | ICLR 2024 | ANN | variate |

SNN models use [SpikingJelly](https://github.com/fangwei123456/spikingjelly). SpikF-family uses the `clock_driven` API; Spikformer/Spikingformer use the newer `activation_based` API. No snntorch dependency.

All models use **per-sample instance normalisation** (TSLib unified evaluation protocol): subtract per-sample mean, divide by per-sample std, then reverse after prediction. This ensures fair comparison across models and datasets.

---

## Datasets

| Dataset | Variables | Granularity | Train / Val / Test |
|---------|-----------|-------------|-------------------|
| ETTh1, ETTh2 | 7 | 1 hour | 8545 / 2881 / 2881 |
| ETTm1, ETTm2 | 7 | 15 min | 34465 / 11521 / 11521 |
| Weather | 21 | 10 min | 70% / 10% / 20% |
| ECL (Electricity) | 321 | 1 hour | 70% / 10% / 20% |
| Traffic | 862 | 1 hour | 70% / 10% / 20% |
| Exchange | 8 | 1 day | 70% / 10% / 20% |
| Solar-Energy | 137 | 10 min | 70% / 10% / 20% |

Place all dataset `.csv` files under `datasets/long/`.

---

## Installation

```bash
# Python 3.9+
pip install torch torchvision
pip install spikingjelly
pip install numpy pandas scikit-learn
```

---

## Quick Start

**SpikF on ETTh1 (multivariate, horizon 336):**
```bash
python run_long.py \
  --model SpikF \
  --data ETTh1 --data_path ETTh1.csv \
  --features M --seq_len 96 --pred_len 336 \
  --T 16 --tau 2.0 --levels 2 \
  --patch_num 48 --patch_dim 32 \
  --hidden_dim 720 --alpha 2.0 \
  --train_epochs 10 --batch_size 32 --lr 5e-4
```

**DLinear baseline:**
```bash
python run_long.py \
  --model DLinear \
  --data ETTh1 --data_path ETTh1.csv \
  --features M --seq_len 96 --pred_len 336 \
  --moving_avg 25 \
  --train_epochs 10 --batch_size 32 --lr 5e-4
```

**TS-LIF with multi-scale temporal memory:**
```bash
python run_long.py \
  --model TSLIF \
  --data weather --data_path weather.csv \
  --features M --seq_len 96 --pred_len 336 \
  --T 16 --tau 2.0 --levels 3 \
  --patch_num 48 --patch_dim 32 --hidden_dim 720 \
  --train_epochs 10 --batch_size 32 --lr 5e-4
```

**Spikformer (SSA spike transformer):**
```bash
python run_long.py \
  --model Spikformer \
  --data ETTh1 --data_path ETTh1.csv \
  --features M --seq_len 96 --pred_len 336 \
  --T 4 --tau 2.0 --levels 2 \
  --d_model 256 --n_heads 8 --d_ff 1024 \
  --common_thr 1.0 --qk_scale 0.125 --encoder_type conv \
  --train_epochs 10 --batch_size 32 --lr 5e-4
```

**ITransformer (inverted ANN transformer):**
```bash
python run_long.py \
  --model ITransformer \
  --data weather --data_path weather.csv \
  --features M --seq_len 96 --pred_len 336 \
  --levels 3 --d_model 512 --n_heads 8 --d_ff 2048 \
  --train_epochs 10 --batch_size 32 --lr 1e-4
```

**Run pre-written scripts:**
```bash
bash scripts/SpikF_ETTh1.sh
bash scripts/SpikF_weather.sh
# etc.
```

---

## Repository Structure

```
SpikingTSF/
├── run_long.py                  # main entry point
├── exp/
│   ├── exp_basic.py             # Exp_Basic base class
│   └── exp_ETT.py               # training / validation / test loop
├── models/
│   ├── SpikF.py                 # SPE + SFS blocks; returns (T,B,H,D)
│   ├── iSpikformer.py           # inverted spike transformer (SpikF repo)
│   ├── SpikeRNN.py              # spike RNN (SpikF repo)
│   ├── SpikTCN.py               # dilated causal TCN (adapted from SeqSNN)
│   ├── SpikGRU.py               # spiking GRU (adapted from SeqSNN)
│   ├── TSLIF.py                 # multi-scale fast/slow LIF (inspired by TS-LIF)
│   ├── Spikformer.py            # SSA spike transformer — temporal tokens (SeqSNN)
│   ├── Spikingformer.py         # Spikformer + ConvPE (SeqSNN)
│   ├── DLinear.py               # decomposition linear baseline (TSLib)
│   ├── ITransformer.py          # inverted ANN transformer baseline (SeqSNN/TSLib)
│   └── layers/
│       ├── spike_attention.py   # SSA, MLP, Block — shared by Spikformer models
│       └── spike_encoder.py     # ConvEncoder, DeltaEncoder, RepeatEncoder
├── data_provider/
│   └── ETT_data_loader.py       # Dataset_ETT_hour/minute, Dataset_Custom, Solar
├── utils/
│   ├── metrics.py               # MAE, MSE, RSE, R²
│   └── tools.py                 # EarlyStopping, StandardScaler
├── scripts/
│   ├── *.sh                     # SpikF scripts (existing)
│   ├── Spikformer/              # Spikformer benchmark scripts
│   ├── Spikingformer/           # Spikingformer benchmark scripts
│   └── ITransformer/            # ITransformer benchmark scripts
├── SeqSNN/                      # original SeqSNN repo (reference, not imported)
└── datasets/long/               # place CSV files here
```

---

## Model Interface

All models follow the TSLib `Model(configs)` convention:

```python
class Model(nn.Module):
    def __init__(self, configs):
        # configs has: seq_len, pred_len, enc_in, T, tau, levels,
        #              patch_num, patch_dim, hidden_dim, dropout, ...
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (B, seq_len, D)
        # out: (B, pred_len, D)
        ...
```

SpikF returns `(T, B, pred_len, D)` (4D) — `exp_ETT.py` averages over the T dimension automatically before computing loss and metrics.

The experiment runner injects `enc_in` and `c_out` into `args` before model construction, so models can always rely on `configs.enc_in`.

---

## Adding a New Model

1. Create `models/MyModel.py` with `class Model(nn.Module)`.
2. Add `from models.MyModel import Model as MyModel` to `exp/exp_ETT.py`.
3. Add `'MyModel'` to `_NEW_STYLE_MODELS` and `_NEW_STYLE_CLASS` in `exp_ETT.py`.
4. Add `'MyModel'` to the `--model` choices in `run_long.py`.
5. If it is a pure ANN (no spikingjelly neurons), add it to `_ANN_MODELS`.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `SpikF` | Model name |
| `--data` | `ETTh1` | Dataset name |
| `--root_path` | `./datasets/long/` | Dataset directory |
| `--data_path` | `ETTh1.csv` | CSV filename |
| `--features` | `M` | `M` multivariate, `S` univariate, `MS` multi→single |
| `--seq_len` | `96` | Lookback window length |
| `--pred_len` | `336` | Forecast horizon |
| `--label_len` | `48` | Decoder overlap length |
| `--T` | `16` | SNN time steps |
| `--tau` | `2.0` | LIF membrane time constant |
| `--levels` | `2` | Number of blocks / layers |
| `--patch_num` | `48` | Number of patches |
| `--patch_dim` | `32` | Patch embedding / TCN hidden dim |
| `--alpha` | `2.0` | Alpha parameter (SpikF / iSpikformer) |
| `--hidden_dim` | `720` | Dense hidden size |
| `--d_model` | `256` | Attention / embedding dim (Spikformer / Spikingformer / ITransformer) |
| `--n_heads` | `8` | Number of attention heads |
| `--d_ff` | `1024` | Feedforward hidden dim (default 4 × d_model) |
| `--common_thr` | `1.0` | LIF spike threshold (Spikformer / Spikingformer) |
| `--qk_scale` | `0.125` | Attention score scale (Spikformer / Spikingformer) |
| `--encoder_type` | `conv` | Spike encoder: `conv` \| `delta` \| `repeat` |
| `--kernel_size` | `3` | Conv kernel size (SpikTCN) |
| `--dropout` | `0.1` | Dropout rate |
| `--moving_avg` | `25` | Trend kernel size (DLinear) |
| `--individual` | `False` | Per-variate linear layers (DLinear) |
| `--train_epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--patience` | `3` | Early stopping patience |
| `--lr` | `5e-4` | Learning rate |
| `--loss` | `mae` | Loss function: `mae` or `mse` |
| `--gpu` | `0` | CUDA device index |
| `--save` | `0` | Save predictions (1 = yes) |
| `--random_seed` | `0` | Random seed |

---

## Acknowledgements

This library builds on and adapts code from:

- **[SpikF](https://github.com/WWJ-creator/SpikF)** — core SpikF, iSpikformer, SpikeRNN architectures
- **[SeqSNN](https://github.com/microsoft/SeqSNN)** (Microsoft Research, MIT License) — SpikeTCN and SpikeGRU adapted from snntorch to spikingjelly
- **[TS-LIF](https://github.com/kkking-kk/TS-LIF)** — multi-scale temporal LIF design (ICLR 2025)
- **[Time-Series-Library](https://github.com/thuml/Time-Series-Library)** (THUML, MIT License) — DLinear model, `Model(configs)` API convention, data loaders

---

## Citation

If you use SpikingTSF in your research, please cite the relevant upstream works:

```bibtex
@inproceedings{lv2024seqsnn,
  title     = {Efficient and Effective Time-Series Forecasting with Spiking Neural Networks},
  author    = {Lv, Changze and others},
  booktitle = {ICML},
  year      = {2024}
}

@inproceedings{zeng2023dlinear,
  title     = {Are Transformers Effective for Time Series Forecasting?},
  author    = {Zeng, Ailing and others},
  booktitle = {AAAI},
  year      = {2023}
}
```
