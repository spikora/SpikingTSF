"""
Traffic and sensor-grid dataset loaders for long-term forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Copyright (c) Microsoft Corporation.
  https://github.com/microsoft/SeqSNN  (SeqSNN/dataset/tsforecast.py)

Handles datasets that differ from the ETT/weather/ECL CSV family:

  Dataset_H5   — PEMS-BAY, METR-LA (.h5 format, 60/20/20 split)
  Dataset_TXT  — solar-energy, electricity-SeqSNN (.txt comma-delimited, 60/20/20 split)

Both classes expose the same interface as ETT_data_loader classes:
  __getitem__ → (seq_x, seq_y)
  seq_x : (seq_len, D)          input window
  seq_y : (pred_len, D)         future horizon without overlap

The 60/20/20 split matches SeqSNN's convention for these benchmarks.
Window splitting is performed over the full set of valid forecast windows,
matching SeqSNN's protocol more closely than the default ETT border split.
Global per-sensor max-abs normalisation is applied before windowing to
match SeqSNN's `normalize=2`; targets use raw future values without
decoder overlap. Time features are intentionally not added here so the
input dimensionality remains compatible with exp_ETT.py and run_long.py.
"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PerSensorMaxAbsScaler:
    """Match SeqSNN normalize=2: divide each sensor by its max absolute value."""

    def __init__(self):
        self.scale_ = None

    def fit(self, data):
        scale = np.max(np.abs(data), axis=0).astype(np.float32)
        # Avoid division by zero for constant-zero sensors.
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, data):
        return data / self.scale_

    def inverse_transform(self, data):
        return data * self.scale_


# ---------------------------------------------------------------------------
# H5 loader — PEMS-BAY, METR-LA
# ---------------------------------------------------------------------------

class Dataset_H5(Dataset):
    """
    Loads HDF5 traffic files (PEMS-BAY, METR-LA).

    File layout expected:
      • pandas HDF5 written as a DataFrame
      • Index is a DatetimeIndex (not stored in .values)
      • Columns are sensor IDs; shape (T, N_sensors)

    Split: 60 % train / 20 % val / 20 % test over valid forecast windows
    (SeqSNN convention). Targets contain only the raw prediction horizon,
    without overlap.

    Args:
        root_path : directory containing the .h5 file
        data_path : filename (e.g. 'metr-la.h5' or 'pems-bay.h5')
        flag      : 'train' | 'val' | 'test'
        size      : [seq_len, label_len, pred_len]
        features  : 'M' multivariate | 'S' univariate | 'MS' multi→single
        target    : column to predict (only used when features='S')
        scale     : apply SeqSNN-style per-sensor max-abs scaling (default True)
        train_ratio : fraction for training   (default 0.6)
        test_ratio  : fraction for test       (default 0.2)
    """

    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='M',
        data_path='metr-la.h5',
        target='OT',
        scale=True,
        inverse=False,
        cols=None,
        train_ratio=0.6,
        test_ratio=0.2,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ('train', 'val', 'test')
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = PerSensorMaxAbsScaler()
        filepath = os.path.join(self.root_path, self.data_path)

        # Read h5 — DataFrame with DatetimeIndex; .values gives (T, N_sensors)
        df = pd.read_hdf(filepath)
        if isinstance(df.index, pd.DatetimeIndex):
            raw = df.values.astype(np.float32)          # (T, N)
        else:
            # Fallback: try reset_index and drop first column (timestamp)
            df = df.reset_index()
            raw = df.iloc[:, 1:].values.astype(np.float32)

        if self.features == 'S':
            raw = raw[:, [0]]                            # univariate: first sensor

        if self.scale:
            # Match SeqSNN normalize=2 behavior: use each sensor's max abs value.
            self.scaler.fit(raw)
            scaled = self.scaler.transform(raw)
        else:
            scaled = raw

        self.data_x = scaled
        self.data_y = raw
        self._set_window_split(len(raw))

    def _set_window_split(self, series_length):
        total_size = series_length - self.seq_len - self.pred_len + 1
        train_size = int(total_size * self.train_ratio)
        test_size = int(total_size * self.test_ratio)
        val_size = total_size - train_size - test_size

        start_idxs = [0, train_size, train_size + val_size]
        lengths = [train_size, val_size, test_size]
        self.start_idx = start_idxs[self.set_type]
        self.length = lengths[self.set_type]

    def __getitem__(self, index):
        s_begin = index + self.start_idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len
        return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]

    def __len__(self):
        return self.length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ---------------------------------------------------------------------------
# TXT loader — solar-energy, electricity (SeqSNN .txt format)
# ---------------------------------------------------------------------------

class Dataset_TXT(Dataset):
    """
    Loads plain comma-delimited .txt sensor files.

    File layout expected:
      • No header row
      • Comma-separated float values
      • Shape (T, N_sensors) after loading

    Examples:
      solar_AL.txt  — 137 solar-panel sensors, Alabama
      electricity.txt — 321 electricity consumption sensors (LSTNET format)

    Split: 60 % train / 20 % val / 20 % test over valid forecast windows
    (SeqSNN convention). Targets contain only the raw prediction horizon,
    without overlap.

    Args identical to Dataset_H5 except data_path is a .txt file.
    """

    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        features='M',
        data_path='solar_AL.txt',
        target='OT',
        scale=True,
        inverse=False,
        cols=None,
        train_ratio=0.6,
        test_ratio=0.2,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ('train', 'val', 'test')
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.__read_data__()

    def __read_data__(self):
        self.scaler = PerSensorMaxAbsScaler()
        filepath = os.path.join(self.root_path, self.data_path)

        # Headerless comma-delimited; try comma first, fall back to tab
        try:
            raw = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        except ValueError:
            raw = np.loadtxt(filepath, delimiter='\t', dtype=np.float32)

        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)                    # univariate edge-case

        if self.features == 'S':
            raw = raw[:, [0]]

        if self.scale:
            # Match SeqSNN normalize=2 behavior: use each sensor's max abs value.
            self.scaler.fit(raw)
            scaled = self.scaler.transform(raw)
        else:
            scaled = raw

        self.data_x = scaled
        self.data_y = raw
        self._set_window_split(len(raw))

    def _set_window_split(self, series_length):
        total_size = series_length - self.seq_len - self.pred_len + 1
        train_size = int(total_size * self.train_ratio)
        test_size = int(total_size * self.test_ratio)
        val_size = total_size - train_size - test_size

        start_idxs = [0, train_size, train_size + val_size]
        lengths = [train_size, val_size, test_size]
        self.start_idx = start_idxs[self.set_type]
        self.length = lengths[self.set_type]

    def __getitem__(self, index):
        s_begin = index + self.start_idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len
        return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]

    def __len__(self):
        return self.length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
