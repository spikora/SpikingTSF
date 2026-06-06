# Roadmap

This document outlines the planned development milestones for SpikingTSF.

---

## v0.1.0 — Initial Public Release *(2026-06-06)*

- [x] Unified training and evaluation framework
- [x] 11 SNN models + 2 ANN baselines integrated
- [x] ETTh1 and ETTh2 datasets supported
- [x] Hyperparameter tuning infrastructure (Optuna)
- [x] Multi-seed evaluation with mean ± std reporting
- [x] Output logging (per-run + aggregated)
- [x] Documentation files (README, CONTRIBUTING, PROTOCOL, DATASETS, REPRODUCIBILITY, MODEL_ZOO, etc.)
- [x] CITATION.cff and citation policy
- [x] GitHub repository transferred to Spikora organization

## v0.2.0 — Complete ETTh1 / ETTh2 Leaderboard

- [x] CI workflow for import checks
- [x] `tests/` directory with basic sanity checks
- [ ] Verified leaderboard results for all 13 models on ETTh1 and ETTh2
- [ ] Full RESULTS.md with results for all models at all horizons (96 / 192 / 336 / 720)

## v0.3.0 — ETTm1 / ETTm2 Results

- [ ] Hyperparameter tuning complete for ETTm1 and ETTm2
- [ ] Full results for all 4 ETT datasets and all 4 horizons
- [ ] Updated leaderboard in README

## v0.4.0 — Additional Datasets

- [ ] Weather dataset support
- [ ] Electricity dataset support
- [ ] Traffic and/or PEMS datasets (if within scope)
- [ ] Results on additional datasets

## Future Work

- Efficiency metrics: spike rate, synaptic operations (SOPs), parameter count, and theoretical energy estimation.
- Potential support for neuromorphic hardware evaluation if infrastructure becomes available.
- Additional SNN architectures from newer publications.

---

Items without checkboxes are planned but not yet in progress. The roadmap may shift as research priorities evolve.
