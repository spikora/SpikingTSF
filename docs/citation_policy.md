# Citation Policy

SpikingTSF is an open-source benchmark library for spiking neural network based time-series forecasting. It provides model implementations, training scripts, dataset loaders, evaluation utilities, and a unified experimental framework for comparing SNN and ANN forecasting models.

---

## If You Use This Library

If you use any substantial part of SpikingTSF in academic work — including the codebase, training loop, model interface, data loading pipeline, evaluation protocol, scripts, or benchmark structure — please cite the related SpikF-GO paper (accepted to ECML PKDD 2026):

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

Please also cite the software repository:

```bibtex
@misc{spikingtsf2026,
  title        = {{SpikingTSF}: A Unified Benchmark Library for Spiking Neural Network Based Time-Series Forecasting},
  author       = {Bakhshaliyev, Jafar},
  year         = {2026},
  howpublished = {\url{https://github.com/spikora/SpikingTSF}},
  note         = {Open-source software library}
}
```

---

## Original Model Citations

SpikingTSF includes implementations or adaptations of several existing SNN and ANN forecasting architectures. If you use a specific model, please also cite the corresponding original paper:

| Model(s) | Paper to Cite |
|----------|--------------|
| SpikF-GO | Bakhshaliyev & Landwehr, accepted to ECML PKDD 2026 (see above) |
| SpikF | Wu et al., ICML 2025 |
| Spikformer, Spikingformer, QKFormer | SeqSNN, NeurIPS 2025 (spotlight) |
| iSpikformer, SpikeRNN, SpikTCN, SpikGRU | SeqSNN, ICML 2024 |
| TSFormer, TSGRU, TSTCN | TS-LIF, ICLR 2025 |
| iTransformer | Liu et al., ICLR 2024 |
| DLinear | Zeng et al., AAAI 2023 |

BibTeX entries for each are available in [MODEL_ZOO.md](../MODEL_ZOO.md) and in the respective model files.

---

## General Rule

Please cite:

1. The **SpikF-GO paper** if you use SpikingTSF code, framework, protocol, scripts, or benchmark infrastructure.
2. The **SpikingTSF software repository** if you use the library as software.
3. The **original model papers** for any specific architectures adapted from prior work.

SpikingTSF does not claim ownership of the original architectures included in the library. The goal of this project is to provide a unified, reproducible, and extensible benchmark framework for spiking neural network based time-series forecasting.
