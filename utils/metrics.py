import numpy as np


def RSE(pred, true):
    """Root Relative Squared Error."""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean(0)) ** 2))


# Alias used in SeqSNN-RPE literature
RRSE = RSE


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean()


def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """Mean Absolute Percentage Error. Skips entries where true == 0."""
    mask = true != 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))


def MSPE(pred, true):
    """Mean Squared Percentage Error. Skips entries where true == 0."""
    mask = true != 0
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask]))


def R2(pred, true, eps: float = 1e-12):
    """Element-wise R² averaged over all (channel, time-step) positions.

    For each output position computes 1 - SS_res / (SS_tot + eps) where
    SS is summed over the batch axis, then returns the mean across all
    positions.  Matches the paper-style evaluation used in TS-LIF / SeqSNN
    where each output dimension is treated independently.

    Args:
        pred: array (B, pred_len, D)
        true: array (B, pred_len, D)
        eps:  small constant to avoid division by zero
    """
    y_bar = true.mean(axis=0)           # (pred_len, D)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - y_bar) ** 2).sum(axis=0)
    r2_elem = 1.0 - ss_res / (ss_tot + eps)
    return float(r2_elem.mean())


def metric(pred, true):
    """Fast validation metric — returns (mae, mse)."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse


def metric_(pred, true):
    """Standard test metric — returns (mae, mse, rse, r2)."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rse = RSE(pred, true)
    r2 = R2(pred, true)
    return mae, mse, rse, r2


def metric_full(pred, true):
    """Extended metric — returns an OrderedDict with all scalar metrics."""
    return {
        'mae':  MAE(pred, true),
        'mse':  MSE(pred, true),
        'rmse': RMSE(pred, true),
        'rse':  RSE(pred, true),
        'r2':   R2(pred, true),
        'mape': MAPE(pred, true),
        'mspe': MSPE(pred, true),
        'corr': CORR(pred, true),
    }
