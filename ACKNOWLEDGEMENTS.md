# Acknowledgements

SpikingTSF integrates, adapts, and builds upon code and ideas from the following open-source research projects. We are grateful to their authors.

---

## SNN Backend

**SpikingJelly** — [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)
The primary SNN simulation framework used throughout this library for LIF neuron dynamics and spike encoding.

---

## SNN Forecasting Models

**SpikF-GO** — [jafarbakhshaliyev/SpikF-GO](https://github.com/jafarbakhshaliyev/SpikF-GO)
Source of the SpikF-GO model (Spiking Fourier Graph Operators, Bakhshaliyev & Landwehr, accepted to ECML PKDD 2026). The `models/SpikF_GO.py` implementation, including its optional spike-domain CPG positional encoding variant, is adapted from this repository.

**SpikF** — [WWJ-creator/SpikF](https://github.com/WWJ-creator/SpikF)
Source of the SpikF model (Spiking Frequency-Domain Transformer, ICML 2025). The `models/SpikF.py` implementation is adapted from this repository.

**SeqSNN** — [microsoft/SeqSNN](https://github.com/microsoft/SeqSNN)
Source of Spikformer, Spikingformer, QKFormer (NeurIPS 2025 spotlight) and iSpikformer, SpikeRNN, SpikTCN, SpikGRU (ICML 2024). Implementations of these models in this library are adapted from SeqSNN.

**TS-LIF** — [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF)
Source of the two-compartment TS-LIF neuron and the TSFormer, TSGRU, TSTCN models (ICLR 2025). Implementations are adapted from this repository.

---

## ANN Baselines

**Time-Series-Library** — [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
Source of the DLinear baseline and general forecasting framework conventions.

**iTransformer** — [thuml/iTransformer](https://github.com/thuml/iTransformer)
Source of the iTransformer ANN baseline (ICLR 2024).

---

## Datasets

**Autoformer / THUML** — [thuml/Autoformer](https://github.com/thuml/Autoformer)
Original source of the ETT (Electricity Transformer Temperature) dataset benchmark suite used throughout this library.

---

## Notes

This library does not claim ownership of any of the original architectures listed above. We provide a unified implementation, benchmarking infrastructure, and reproducibility framework. When using any specific model, please cite both this library and the corresponding original paper.

See [docs/citation_policy.md](docs/citation_policy.md) for details.
