"""
DLinear: Decomposition Linear — ANN baseline for long-term forecasting.

Taken from Time-Series-Library (THUML, MIT License):
  Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
  https://github.com/thuml/Time-Series-Library

"""

import torch
from torch import nn



class MovingAvg(nn.Module):
    """Centered moving average with edge-padding to preserve length."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L, D)
        half = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, half, 1)
        end = x[:, -1:, :].repeat(1, half, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend



class Model(nn.Module):
    """
    DLinear: one linear layer per component (seasonal + trend), channel-independent.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', False)
        self.channels = configs.enc_in
        kernel = getattr(configs, 'moving_avg', 25)

        self.decomp = SeriesDecomp(kernel)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            for lin in self.linear_seasonal + self.linear_trend:
                nn.init.constant_(lin.weight,
                                  1.0 / self.seq_len)
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)

    def forward(self, x):
        # x: (B, L, D)
        seasonal, trend = self.decomp(x)  # each: (B, L, D)

        seasonal = seasonal.permute(0, 2, 1)  # (B, D, L)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            out_s = torch.stack(
                [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(self.channels)],
                dim=1,
            )  # (B, D, pred_len)
            out_t = torch.stack(
                [self.linear_trend[i](trend[:, i, :]) for i in range(self.channels)],
                dim=1,
            )
        else:
            out_s = self.linear_seasonal(seasonal)  # (B, D, pred_len)
            out_t = self.linear_trend(trend)

        out = (out_s + out_t).permute(0, 2, 1)  # (B, pred_len, D)
        return out
