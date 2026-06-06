# Contributing to SpikingTSF

Contributions are welcome and appreciated. This document explains how to contribute effectively.

---

## Types of Contributions

### Bug Reports

If you encounter a bug, please open a [GitHub Issue](https://github.com/spikora/SpikingTSF/issues) using the bug report template. Include:
- Your environment (OS, Python version, PyTorch version, CUDA version)
- The exact command you ran
- The full error output
- Expected vs actual behavior

### Documentation Improvements

If something in the documentation is unclear, incorrect, or missing, open an issue or submit a pull request with the fix.

### New Model Implementations

To add a new SNN or ANN forecasting model, follow these steps:

1. **Add model file** under `models/`. Follow the existing model structure (e.g., `models/SpikF.py`).
2. **Add config** under `configs/<ModelName>/` with at least an `ETTh1.yaml` file.
3. **Add scripts** under `scripts/<ModelName>/` to reproduce experiments.
4. **Add citation** for the original paper in the model file (as a comment) and in `MODEL_ZOO.md`.
5. **Update `MODEL_ZOO.md`** with the new model entry.
6. **Add reproducible results** to `RESULTS.md` if available, following the protocol in `PROTOCOL.md`.
7. **Update `README.md`** model list and news section if appropriate.

### Dataset Loaders

If you add support for a new dataset, add the loader under `data_provider/` and document the expected file structure in `DATASETS.md`.

### Reproducibility Checks

If you independently reproduce results for an existing model, please submit them via the [Result Submission issue template](https://github.com/spikora/SpikingTSF/issues/new?template=result_submission.yml). Include your script, seed, hardware, and output logs.

### Leaderboard Submissions

Results submitted to the leaderboard must follow the policy in [docs/leaderboard_policy.md](docs/leaderboard_policy.md). Unverified or incomplete submissions will not be added.

---

## Pull Request Workflow

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-model-name`.
3. Make your changes. Keep commits focused and descriptive.
4. Ensure the code runs locally and produces expected output.
5. Do **not** commit datasets, checkpoints, trained weights, or large output files. These are gitignored for a reason.
6. Open a pull request against the `main` branch. Fill in the PR template.
7. A maintainer will review and may request changes before merging.

---

## Coding Style

- Follow the coding conventions already present in the codebase.
- Model files should be self-contained where possible.
- Avoid unnecessary dependencies beyond what is already in `requirements.txt`.
- Do not add inline comments that simply restate what the code does.

---

## What Not to Commit

- Dataset CSV files (`datasets/`)
- Model checkpoints or weights (`*.pt`, `*.pth`, `*.ckpt`)
- Optuna study databases (`hparam/studies/`)
- Output files (`Output/`, `test_results/`)
- Virtual environments

These are all listed in `.gitignore`.

---

## Acknowledgement of Contributors

Significant contributors may be acknowledged in `ACKNOWLEDGEMENTS.md` and release notes. Please ensure your contributions are clearly described in the pull request.
