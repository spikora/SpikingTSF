# Model Zoo

All models integrated into SpikingTSF are listed below. Model implementations are adapted from the source repositories indicated. This library does not claim ownership of any original architecture.

When using a specific model, please cite both SpikingTSF and the corresponding source paper.

---

| Model | Type | Spike? | Source Paper | Source Repository | File in Repo | Status |
|-------|------|:------:|-------------|-------------------|-------------|--------|
| SpikF | Transformer (freq. domain) | ✅ | [Wu et al., ICML 2025](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf) | [WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF) | `models/SpikF.py` | Runnable, ETTh1+ETTh2 |
| Spikformer | Transformer (spike-driven SA) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/Spikformer.py` | Runnable, ETTh1 |
| Spikingformer | Transformer (pre-LIF) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/Spikingformer.py` | Runnable, ETTh1+ETTh2 verified |
| QKFormer | Transformer (token-level Q/K attn) | ✅ | [SeqSNN, NeurIPS 2025](https://arxiv.org/abs/2501.16745) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/QKFormer.py` | Runnable, results pending |
| TSGRU | GRU (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) | `models/TSGRU.py` | Runnable, ETTh1+ETTh2 verified |
| TSTCN | TCN (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) | `models/TSTCN.py` | Runnable, ETTh1 verified |
| TSFormer | Transformer (two-compartment TS-LIF) | ✅ | [TS-LIF, ICLR 2025](https://arxiv.org/abs/2503.05108) | [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) | `models/TSFormer.py` | Runnable, ETTh1 verified |
| iSpikformer | Inverted Spiking Transformer | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/iSpikformer.py` | Runnable, ETTh1 verified |
| SpikeRNN | Spiking Recurrent Network | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/SpikeRNN.py` | Runnable, ETTh1+ETTh2 verified |
| SpikTCN | Spiking Temporal Convolutional Network | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/SpikTCN.py` | Runnable, ETTh1 partial |
| SpikGRU | Spiking Gated Recurrent Unit | ✅ | [SeqSNN, ICML 2024](https://arxiv.org/abs/2402.01533) | [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN) | `models/SpikGRU.py` | Runnable, ETTh1+ETTh2 verified |
| iTransformer | Inverted Transformer (ANN) | ❌ | [Liu et al., ICLR 2024](https://arxiv.org/abs/2310.06625) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | `models/ITransformer.py` | Runnable, results pending |
| DLinear | Decomposition Linear (ANN) | ❌ | [Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504) | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `models/DLinear.py` | Runnable, results pending |

---

## Status Definitions

- **Runnable**: Code is integrated and can be executed with the provided scripts.
- **Verified**: Results have been produced and logged under `Output/`. Numbers are in RESULTS.md.
- **Partial**: Some horizons or datasets have been evaluated but not all.
- **Pending**: Code is present but results have not yet been computed.

---

## Adding a New Model

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/adding_a_model.md](docs/adding_a_model.md).
