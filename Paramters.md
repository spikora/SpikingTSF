# SpikF Model Parameters

## Parameters

- `input_len`: `seq_len`
- `patch_num`
- `patch_dim`
- `T`
- `blocks`: layers / levels
- `D`: channels
- `pred_len`
- `tau`: LIF
- `hidden_dim`: dense layer
- `normalize`: RevIN-style normalization

## Passed Through `__init__`

```python
def __init__(
    self,
    input_len,
    patch_num,
    patch_dim,
    T,
    blocks,
    D,
    pred_len,
    tau,
    hidden_dim,
    normalize=True,
):
```

# SpikeRNN Model Parameters

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: SNN time steps
- `blocks`: number of SpikeRNNCell layers (`levels`)
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `hidden_dim`: recurrent cell and decoder hidden size (mapped from `alpha` in args)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding type — `'none'` (default), `'learn'`, `'static'`, `'conv'`, `'neuron'`, `'random'`
- `pe_mode`: how PE is combined — `'add'` (default) or `'concat'`; `'concat'` is only meaningful for `pe_type` in `{'neuron', 'random'}` and widens the input projection
- `num_pe_neuron`: number of PE neurons for `neuron`/`random` PE in concat mode (default `10`)
- `neuron_pe_scale`: frequency scale for `neuron` PE (default `1000.0`)
- `normalize`: RevIN-style instance normalization (default `True`)

## Passed Through `__init__`

```python
def __init__(
    self,
    input_len,
    T,
    blocks,
    D,
    pred_len,
    tau,
    hidden_dim,
    encoder_type='conv',
    pe_type='none',
    pe_mode='add',
    num_pe_neuron=10,
    neuron_pe_scale=1000.0,
    normalize=True,
):
```

## Notes

- PE is applied over the flattened `T × L` dimension (each SNN step + sequence position gets a unique encoding) for `neuron` and `random` types.
- `pe_type='conv'` and `pe_type='static'` are applied over the `L` dimension only.
- `encoder_type='conv'`: `Conv2D((1, kernel_size)) → BN → LIF` — captures local shape features.
- `encoder_type='delta'`: first-order temporal difference `→ Linear → BN → LIF` — captures rate-of-change.

---

# iSpikformer Model Parameters

## Overview

Inverted spiking transformer following the iTransformer paradigm: each channel
is treated as a token, embedded over its full temporal history (`L → d_model`),
then spiking self-attention (SSA) models multivariate correlations across channels.

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps
- `blocks`: number of transformer blocks (`levels`)
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `d_model`: per-channel embedding dimension — each channel's `L` time steps are projected to `d_model` (mapped from `alpha` in args)
- `d_ff`: feedforward hidden size in MLP sub-block (default: `d_model × 4`)
- `heads`: number of attention heads in SSA (must divide `d_model`, default `8`)
- `common_thr`: LIF firing threshold for all neurons (default `1.0`)
- `qk_scale`: Q·K scaling factor in SSA (default `0.125`)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `normalize`: RevIN-style instance normalization (default `True`)

## Passed Through `__init__`

```python
def __init__(
    self,
    input_len,
    T,
    blocks,
    D,
    pred_len,
    tau,
    d_model,
    d_ff=None,
    heads=8,
    common_thr=1.0,
    qk_scale=0.125,
    encoder_type='conv',
    normalize=True,
):
```

## Architecture

```
(B, L, D)
  → SpikeEncoder            (T, B, D, L)        conv/delta encoder
  → transpose               (T, B, L, D)
  → DataEmbeddingInverted   (T, B, D, d_model)  Linear(L→d_model) + BN + LIF per channel
  → Block × blocks          (T, B, D, d_model)  SSA(D tokens, d_model dim) + MLP
  → mean(T)                 (B, D, d_model)      average over SNN time steps
  → Linear(d_model→pred_len)(B, D, pred_len)
  → transpose               (B, pred_len, D)
```

## Notes

- SSA attention runs over the `D` channel tokens (length=D, dim=d_model); it models **multivariate correlations** rather than temporal dependencies.
- Temporal information is encoded in the inverted embedding (`L → d_model`) — no positional encoding is needed over the channel dimension.
- `d_model` must be divisible by `heads`.
- `encoder_type='conv'` and `encoder_type='delta'` are shared with SpikeRNN (see above).
