# Datasets

SpikingTSF uses standard long-term forecasting benchmark datasets. **Datasets are not included in this repository** and must be downloaded separately.

---

## Required Datasets

| Dataset | Description | Frequency | Variables |
|---------|-------------|-----------|-----------|
| ETTh1 | Electricity Transformer Temperature (hourly) | 1h | 7 |
| ETTh2 | Electricity Transformer Temperature (hourly) | 1h | 7 |
| ETTm1 | Electricity Transformer Temperature (15-min) | 15min | 7 |
| ETTm2 | Electricity Transformer Temperature (15-min) | 15min | 7 |

---

## Where to Download

The ETT datasets are available from the following sources. Use whichever is accessible:

- **Autoformer repository (THUML):** [github.com/thuml/Autoformer](https://github.com/thuml/Autoformer) — look in the `dataset/` directory or follow the download instructions in their README.
- **Hugging Face:** [huggingface.co/datasets/thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library)

---

## File Placement

Place the downloaded CSV files under `datasets/long/`:

```
SpikingTSF/
└── datasets/
    └── long/
        ├── ETTh1.csv
        ├── ETTh2.csv
        ├── ETTm1.csv
        └── ETTm2.csv
```

The data loader (`data_provider/ETT_data_loader.py`) expects files at this path. Do not rename the files.

---

## Notes

- The `datasets/` directory is listed in `.gitignore` and will not be committed.
- If running on a cluster (e.g., SLURM), place the datasets in shared storage and symlink or set the path accordingly. The data root path can be set via the `--root_path` argument in `run_long.py`.
