# Changelog

All notable changes to **SpikingTSF** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- `SpikF_GO` model (`models/SpikF_GO.py`) — Spiking Fourier Graph Operators (Bakhshaliyev & Landwehr, accepted to ECML PKDD 2026), with an optional spike-domain CPG positional encoding toggled via `--pe_type cpg`
- Documentation files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `GOVERNANCE.md`, `ROADMAP.md`, `ACKNOWLEDGEMENTS.md`, `NOTICE`, `PROTOCOL.md`, `DATASETS.md`, `REPRODUCIBILITY.md`, and `MODEL_ZOO.md`
- `CITATION.cff` for software citation
- GitHub issue templates and pull request template
- Lightweight CI workflow for basic import checks
- `environment.yml` for conda-based setup
- `tests/test_imports.py` for basic import testing
- `examples/quick_start.md`
- `docs/` directory with citation policy, leaderboard policy, and model addition guide

### Changed

- Restructured `README.md` with clearer project framing, motivation, key features, model zoo, reproducibility information, and citation instructions
- Transferred the repository to the Spikora Neural Research organization

---

## [0.1.0] — 2026-06-06

### Added

- Initial release of the **SpikingTSF** benchmark library
- Unified training and evaluation framework for spiking neural network based time-series forecasting
- 11 SNN model implementations:
  - `SpikF`
  - `Spikformer`
  - `Spikingformer`
  - `QKFormer`
  - `TSGRU`
  - `TSTCN`
  - `TSFormer`
  - `iSpikformer`
  - `SpikeRNN`
  - `SpikTCN`
  - `SpikGRU`
- 2 ANN baseline implementations:
  - `iTransformer`
  - `DLinear`
- Unified experiment entry point through `run_long.py`
- ETT forecasting experiment pipeline through `exp/exp_ETT.py`
- Dataset loading support for ETT-style long-term forecasting datasets
- Shell scripts under `scripts/` for reproducible experiment execution
- YAML configuration files under `configs/`
- Support for multiple forecasting metrics: MSE, MAE, RMSE, R²
- Multi-seed evaluation (seeds 42, 1234, 3407) with mean ± standard deviation reporting
- Optuna-based hyperparameter tuning infrastructure (`hparam/`)
- Output logging structure for storing training logs and evaluation results

### Notes

- Raw datasets, checkpoints, logs, Optuna study databases, and generated output files are not included in the repository.
- Verified leaderboard results will be added progressively after reproducibility checks.
