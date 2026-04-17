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

---

# SpikeTCN Model Parameters

## Overview

Spiking Temporal Convolutional Network with dilated causal convolutions.
Processes each of the D variates independently (channel-independent), using
the spike encoder to convert the raw input into T-step spike trains.
SEW (spike-element-wise) residual connections stabilise training.

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps (also equals encoder output channels)
- `blocks`: number of dilated TCN blocks — block `i` has dilation `2^i`
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `hidden_dim`: TCN hidden channel size (mapped from `alpha` in args)
- `kernel_size`: dilated causal conv kernel size (default `3`)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding type — `'none'` (default), `'learn'`, `'static'`, `'conv'`, `'neuron'`, `'random'`. **Note:** only `add` mode is supported (concat would change the D dimension in the channel-independent pipeline)
- `num_pe_neuron`: PE neurons for `neuron`/`random` PE (add mode, default `10`)
- `neuron_pe_scale`: frequency scale for neuron PE (default `1000.0`)
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
    kernel_size=3,
    encoder_type='conv',
    pe_type='none',
    num_pe_neuron=10,
    neuron_pe_scale=1000.0,
    normalize=True,
):
```

## Architecture

```
(B, L, D)
  → SpikeEncoder             (T, B, D, L)        conv/delta encoder
  → Optional PE              (T, B, D, L)        transposed for PE, then back
  → flatten + unsqueeze      (T*B*D, 1, L)       channel-independent batch
  → Conv1d(1→hidden_dim)     (T*B*D, hidden_dim, L)
  → SpikeTCNBlock × blocks   (T*B*D, hidden_dim, L)
      dilated causal Conv1d → BN → single-step LIF (per-sample, no T-unfolding), SEW residual
  → last causal position     (T*B*D, hidden_dim)
  → Linear(hidden_dim→pred_len)(T*B*D, pred_len)
  → reshape → mean T         (B, pred_len, D)
```

## Notes

- Dilation doubles per block: block 0 has dilation 1, block 1 has dilation 2, etc.
- `T` is folded into the batch dimension (`T*B*D`) for all Conv1d operations. Single-step LIF (`step_mode='s'`) is applied directly to the `(T*B*D, C, L)` tensor — each sample (including each T replicate) is independent, so membrane state does **not** persist across SNN time steps. This matches the paper's rule that TCN membrane potentials reset at each time-series step, enabling fully parallel training.
- `pe_mode` is always `'add'` for SpikeTCN regardless of what is passed.

---

# SpikeGRU Model Parameters

## Overview

Gated spiking RNN with GRU-style gating. Follows the same processing pipeline
as SpikeRNN (same encoder, PE, and decoder structure) but replaces each
`SpikeRNNCell` with a `SpikeGRUCell` that adds reset and update gates computed
via the ATan surrogate function (binary {0, 1} spike-like activations).

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps
- `blocks`: number of SpikeGRUCell layers
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `hidden_dim`: recurrent cell and decoder hidden size (mapped from `alpha` in args)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding type — `'none'` (default), `'learn'`, `'static'`, `'conv'`, `'neuron'`, `'random'`
- `pe_mode`: how PE is combined — `'add'` (default) or `'concat'`; `'concat'` widens the input projection for `neuron`/`random` PE
- `num_pe_neuron`: PE neurons for `neuron`/`random` PE in concat mode (default `10`)
- `neuron_pe_scale`: frequency scale for neuron PE (default `1000.0`)
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

## Architecture

```
(B, L, D)
  → SpikeEncoder             (T, B, D, L)        conv/delta encoder
  → transpose                (T, B, L, D)
  → Optional PE              (T, B, L, D')
  → Linear(D'→hidden_dim) + LIF  (T, B, L, hidden_dim)
  → SpikeGRUCell × blocks   (T, B, L, hidden_dim)   sequential T-step loop
      y_ih = linear_ih(x_t),  y_hh = linear_hh(h)
      r = ATan(y_ih[0] + y_hh[0])                reset gate  ∈ {0, 1}
      z = ATan(y_ih[1] + y_hh[1])                update gate ∈ {0, 1}
      n = ATan(y_ih[2] + r·y_hh[2])              candidate
      h = LIF((1−z)·n + z·h)                     spike-quantized hidden state
  → Linear(hidden_dim→D) → BN → LIF  (T, B, L, D)
  → Linear(L→pred_len)     (T, B, D, pred_len)
  → permute → mean T        (B, pred_len, D)
```

## Notes

- `SpikeGRUCell` has two linear paths: `linear_ih` (input-to-hidden) and `linear_hh` (hidden-to-hidden), matching SeqSNN's `GRUCell`. Gates use `ATan` surrogate, yielding binary {0, 1} spike-like values.
- T steps are processed **sequentially** inside each cell, with the explicit hidden state `h` passed from step to step. Single-step LIF (`step_mode='s'`) spike-quantizes `h` at every step, so the LIF membrane is implicitly encoded in the spiked hidden state rather than the LIF membrane register.
- Decoder structure is identical to SpikeRNN: `dense1 → BN → LIF → dense2(L→pred_len)`.
- All PE types from `models/layers/positional_encoding.py` are supported.

---

# Spikformer Model Parameters

## Overview

Unified spiking transformer (formerly SpikformerV) that consolidates five SeqSNN-RPE
Spikformer variants into a single class. The active variant is selected via `attn_type`.
All variants share the same encoder → embedding → blocks → decoder pipeline; they differ
in how positional information is injected and how attention similarity is computed.

| `attn_type`  | Attention           | PE                        | Notes                                              |
|--------------|---------------------|---------------------------|----------------------------------------------------|
| `standard`   | SSA (dot-product)   | any `pe_type`/`pe_mode`   | Baseline Spikformer; no init_bn                    |
| `cpg`        | SSA (dot-product)   | none (built into embed)   | CPGLinear input + CPGLinear MLP; no init_bn        |
| `xnor`       | XNOR (global min)   | `conv` (required)         | scale init −2, global min-shift; has init_bn       |
| `xnor_gray`  | XNOR + Gray-code    | `conv` (required)         | Gray bits appended to Q/K, scale init −4; init_bn  |
| `xnor_log`   | XNOR + int log-dist | `conv` (required)         | Integer symmetric bias, scale init −4; init_bn     |

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps
- `blocks`: number of transformer blocks (`levels`)
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `d_model`: embedding and attention dimension — mapped from `alpha` in args (must be divisible by `heads`)
- `d_ff`: feedforward hidden dim in MLP sub-block (default: `d_model × 4`)
- `heads`: attention heads in SSA/SSA_XNOR (default `8`)
- `common_thr`: LIF firing threshold (default `1.0`)
- `qk_scale`: Q·K scaling factor for standard SSA (default `0.125`; ignored for XNOR)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding — `'none'` (default), `'learn'`, `'static'`, `'conv'`, `'neuron'`, `'random'`
- `pe_mode`: how PE is combined — `'add'` (default) or `'concat'`; `'concat'` widens the input Linear for `standard` + `pe_type in {'neuron','random'}`; ignored for XNOR (forced `'add'`) and CPG
- `attn_type`: attention variant — `'standard'` (default), `'cpg'`, `'xnor'`, `'xnor_gray'`, `'xnor_log'`
- `gray_bits`: Gray-code bits appended to Q/K for `xnor_gray` (default `10` → 1024 positions)
- `num_pe_neuron`: CPG oscillator neurons / PE neurons for `neuron`/`random` PE (default `10`)
- `neuron_pe_scale`: frequency scale for CPG / neuron PE (default `1000.0`)
- `dropout`: dropout before output projection (default `0.1`)
- `normalize`: RevIN-style instance normalization (default `True`)

## Assertions

```python
# XNOR variants require convolutional relative PE
if attn_type in ('xnor', 'xnor_gray', 'xnor_log'):
    assert pe_type == 'conv'

# CPG encodes position inside CPGLinear — no external PE allowed
if attn_type == 'cpg':
    assert pe_type == 'none'
```

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
    pe_type='none',
    pe_mode='add',
    attn_type='standard',
    gray_bits=10,
    num_pe_neuron=10,
    neuron_pe_scale=1000.0,
    dropout=0.1,
    normalize=True,
):
```

## Architecture

```
(B, L, D)
  → RevIN norm
  → SpikeEncoder                      (T, B, D, L)   conv/delta encoder
  → transpose                         (T, B, L, D)
  → [PE if pe_type != 'none']          (T, B, L, D')  D'=D+num_pe_neuron for concat mode
  → InputEmbed [+ BN for XNOR] + LIF (T, B, L, d_model)
      standard: Linear(D' → d_model)             no init_bn
      cpg:      CPGLinear(D → d_model)            no init_bn; position fused inside
      xnor*:    Linear(D → d_model) + init_bn
  → Block × blocks                    (T, B, L, d_model)
      standard: SSA (fixed qk_scale) + MLP
      cpg:      SSA (fixed qk_scale) + MLPCPG    CPGLinear for fc1/fc2
      xnor:     SSA_XNOR(attn_pe='none',  scale_init=-2) + MLP
      xnor_gray:SSA_XNOR(attn_pe='gray',  scale_init=-4) + MLP
      xnor_log: SSA_XNOR(attn_pe='log',   scale_init=-4) + MLP
  → mean(T)                           (B, L, d_model)
  → dropout
  → mean(L)                           (B, d_model)
  → Linear(d_model → pred_len×D)      (B, pred_len×D)
  → reshape                           (B, pred_len, D)
  → denorm
```

## Notes

- `standard` and `cpg` skip `init_bn` (matching SeqSNN-RPE `spikformer.py` and `spikformer_CPG.py`); XNOR variants include `init_bn` (matching `spikformer_xnor*.py`).
- `cpg` variant is the most faithful to the reference: `CPGLinear` for the input embedding **and** `MLPCPG` (CPGLinear fc1/fc2) inside every transformer block. Position is encoded at every layer via the binarised sinusoidal CPG table.
- `SSA_XNOR` uses a learnable scalar `scale` (`sigmoid(scale)` replaces fixed `qk_scale`). `xnor` base normalises by the global minimum scalar (matching `spikformer_xnor.py`); `xnor_gray`/`xnor_log` do not subtract a minimum.
- `xnor_log` bias matrix uses **integer** values (`torch.linspace(...).round().int()`), matching the reference `spikformer_xnor_log.py`.
- XNOR similarity: `sim(q, k) = D_new − Σq − Σk + 2·q·kᵀ`; for `xnor_gray`, `D_new = head_dim + gray_bits`.
- `standard` with `pe_mode='concat'` and `pe_type in {'neuron','random'}` widens the input Linear to `D + num_pe_neuron → d_model`, matching `spikformer.py`'s widened encoder.
- `d_model` must be divisible by `heads`.
- In `exp_ETT.py`, `d_model` is mapped from `args.alpha`; `attn_type` from `args.attn_type` (default `'standard'`); `pe_mode` from `args.pe_mode` (default `'add'`).

---

# Spikingformer Model Parameters

## Overview

Pre-LIF spiking transformer that consolidates four SeqSNN-RPE Spikingformer variants
into a single class. The key distinction from Spikformer is the **pre-LIF** design:
each SpikingBlock gates the **input** `x` with a LIF neuron before QKV projection
(instead of post-LIF on the output). There is no `init_lif` after `init_bn`; the first
block's `pre_lif` handles the initial gate.

| `attn_type`  | Attention                  | PE required   | Notes                                        |
|--------------|----------------------------|---------------|----------------------------------------------|
| `standard`   | SpikingSSA (pre-LIF, dot)  | any           | Pre-LIF gate on input; no output LIF in SSA  |
| `xnor`       | SpikingSSA_XNOR (none)     | `conv`        | XNOR, global min-shift, scale init −2        |
| `xnor_gray`  | SpikingSSA_XNOR (gray)     | `conv`        | XNOR + Gray-code bits, scale init −4         |
| `xnor_log`   | SpikingSSA_XNOR (log)      | `conv`        | XNOR + integer log-distance bias, scale −4   |

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps
- `blocks`: number of SpikingBlock layers
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `d_model`: embedding and attention dimension — mapped from `alpha` in args (must be divisible by `heads`)
- `d_ff`: feedforward hidden dim in SpikingMLP (default: `d_model × 4`)
- `heads`: attention heads (default `8`)
- `common_thr`: LIF firing threshold (default `1.0`)
- `qk_scale`: Q·K scaling factor for standard SpikingSSA (default `0.125`; ignored for XNOR)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding — `'none'` (default), `'learn'`, `'static'`, `'conv'`, `'neuron'`, `'random'`; XNOR requires `'conv'`
- `attn_type`: attention variant — `'standard'` (default), `'xnor'`, `'xnor_gray'`, `'xnor_log'`
- `gray_bits`: Gray-code bits for `xnor_gray` (default `10`)
- `dropout`: dropout before output projection (default `0.1`)
- `normalize`: RevIN-style instance normalization (default `True`)

## Assertions

```python
# XNOR variants require convolutional relative PE
if attn_type in ('xnor', 'xnor_gray', 'xnor_log'):
    assert pe_type == 'conv'
```

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
    pe_type='conv',
    attn_type='standard',
    gray_bits=10,
    dropout=0.1,
    normalize=True,
):
```

## Architecture

```
(B, L, D)
  → RevIN norm
  → SpikeEncoder                     (T, B, D, L)   conv/delta encoder
  → transpose                        (T, B, L, D)
  → [PE if pe_type != 'none']         (T, B, L, D)
  → Linear(D → d_model) + init_bn   (T, B, L, d_model)   no init_lif
  → SpikingBlock × blocks            (T, B, L, d_model)
      standard:  SpikingSSA(pre_lif → QKV → attn_lif → proj_bn) + SpikingMLP
      xnor:      SpikingSSA_XNOR(attn_pe='none',  scale_init=-2) + SpikingMLP
      xnor_gray: SpikingSSA_XNOR(attn_pe='gray',  scale_init=-4) + SpikingMLP
      xnor_log:  SpikingSSA_XNOR(attn_pe='log',   scale_init=-4) + SpikingMLP
  → mean(T)                          (B, L, d_model)
  → dropout
  → mean(L)                          (B, d_model)
  → Linear(d_model → pred_len×D)     (B, pred_len×D)
  → reshape                          (B, pred_len, D)
  → denorm
```

## Notes

- **Pre-LIF design**: `SpikingSSA` gates the input `x` with `pre_lif` first, then projects to Q/K/V. The output has no final LIF — the next block's `pre_lif` handles the gate. This contrasts with Spikformer's `init_lif` after embedding and post-LIF inside SSA.
- **SpikingMLP** follows the same pre-LIF pattern: `lif1 → fc1 → bn1 → lif2 → fc2 → bn2`. No post-LIF at the end of the MLP.
- `init_bn` is present (after `input_embed`) but there is no `init_lif` — each block's `pre_lif` serves as the first gate for all variants.
- XNOR similarity is identical to Spikformer XNOR: `sim(q, k) = D_new − Σq − Σk + 2·q·kᵀ` with learnable scalar `scale`.
- `attn_lif` in SpikingSSA uses `v_threshold=0.5` (lower threshold to allow more spike activity on the attention map).
- In `exp_ETT.py`, `d_model` is mapped from `args.alpha`; `pe_type` from `args.pe_type` (default `'conv'`); `attn_type` from `args.attn_type` (default `'standard'`).

---

# QKFormer Model Parameters

## Overview

Three-stage spiking transformer with a token-level Q/K attention mechanism.
Stages 1 and 2 use `TokenQKBlock` (Q/K attention with no V matrix, O(L·D) complexity);
stage 3 uses `SpikingBlock` (full SpikingSSA or SpikingSSA_XNOR). At least 3 blocks
are required — the first two are always TokenQKBlocks and the remaining `blocks−2`
are SpikingBlocks.

| `attn_type`  | Stage 3 Attention              | PE required   | Notes                                       |
|--------------|-------------------------------|---------------|---------------------------------------------|
| `standard`   | SpikingSSA (pre-LIF, dot)     | any           | Standard dot-product in stage 3             |
| `xnor_gray`  | SpikingSSA_XNOR (gray)        | `conv`        | XNOR + Gray-code bits in stage 3            |
| `xnor_log`   | SpikingSSA_XNOR (log)         | `conv`        | XNOR + integer log-distance bias in stage 3 |

## Parameters

- `input_len`: sequence length (`seq_len`)
- `T`: number of SNN time steps
- `blocks`: total transformer blocks — must be ≥ 3 (2 TokenQKBlocks + ≥1 SpikingBlocks)
- `D`: number of input/output channels (`enc_in`)
- `pred_len`: prediction horizon
- `tau`: LIF membrane time constant
- `d_model`: embedding and attention dimension — mapped from `alpha` in args (must be divisible by `heads`)
- `d_ff`: feedforward hidden dim in SpikingMLP (default: `d_model × 4`)
- `heads`: attention heads in TokenQKAttention and stage-3 SpikingSSA (default `8`)
- `common_thr`: LIF firing threshold (default `1.0`)
- `qk_scale`: Q·K scaling factor for standard SpikingSSA in stage 3 (default `0.125`; ignored for XNOR)
- `encoder_type`: spike encoder — `'conv'` (default) or `'delta'`
- `pe_type`: positional encoding — `'none'`, `'conv'` (default); XNOR variants require `'conv'`
- `attn_type`: stage-3 attention variant — `'standard'` (default), `'xnor_gray'`, `'xnor_log'`
- `gray_bits`: Gray-code bits for `xnor_gray` (default `10`)
- `dropout`: dropout before output projection (default `0.1`)
- `normalize`: RevIN-style instance normalization (default `True`)

## Assertions

```python
assert blocks >= 3   # 2 token-QK stages + at least 1 spiking stage

# XNOR variants in stage 3 require convolutional relative PE
if attn_type in ('xnor_gray', 'xnor_log'):
    assert pe_type == 'conv'
```

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
    pe_type='conv',
    attn_type='standard',
    gray_bits=10,
    dropout=0.1,
    normalize=True,
):
```

## Architecture

```
(B, L, D)
  → RevIN norm
  → SpikeEncoder                       (T, B, D, L)    conv/delta encoder
  → transpose                          (T, B, L, D)
  → [ConvPE if pe_type='conv']          (T, B, L, D)
  → Linear(D → d_model) + init_bn     (T, B, L, d_model)   no init_lif
  → Stage 1: TokenQKBlock              (T, B, L, d_model)
  → Stage 2: TokenQKBlock              (T, B, L, d_model)
  → Stage 3: SpikingBlock × (blocks-2) (T, B, L, d_model)
      standard:  SpikingSSA + SpikingMLP
      xnor_gray: SpikingSSA_XNOR(attn_pe='gray') + SpikingMLP
      xnor_log:  SpikingSSA_XNOR(attn_pe='log')  + SpikingMLP
  → mean(T)                            (B, L, d_model)
  → dropout
  → mean(L)                            (B, d_model)
  → Linear(d_model → pred_len×D)       (B, pred_len×D)
  → reshape                            (B, pred_len, D)
  → denorm
```

### TokenQKAttention (Stages 1 & 2)

```
(T, B, L, D)
  → Q path: Linear(D→D) → BN(D-axis) → LIF → reshape (T,B,H,D/H,L)
  → K path: Linear(D→D) → BN(D-axis) → LIF → reshape (T,B,H,D/H,L)
  → sum Q over feature dim → (T,B,H,1,L)   [scalar token per head per position]
  → attn_lif (v_threshold=0.5) → binary spike gate
  → elementwise multiply with K → (T,B,H,D/H,L)
  → flatten heads → project (Linear + BN + LIF) → (T,B,L,D)
```

## Notes

- **TokenQKAttention complexity**: O(L·D) per layer — no L² attention matrix is formed. Q is collapsed to a scalar token per position (summed over feature dimension) and used as a binary gate on K.
- **No V matrix**: TokenQKAttention has only Q and K projections. The gated K output serves as the attended value.
- **`attn_lif` threshold = 0.5**: lower than the default 1.0 to allow more gates to fire on the compressed Q-sum signal.
- Stages 1 and 2 are always `TokenQKBlock`s regardless of `attn_type`; `attn_type` only affects stage 3.
- `xnor` is not a valid `attn_type` for QKFormer (only `xnor_gray` and `xnor_log` are supported, matching the SeqSNN-RPE reference variants).
- `_init_weights` uses `trunc_normal_(std=0.02)` (same as QKFormer reference), unlike Spikformer/Spikingformer which use `normal_(std=0.02)`.
- In `exp_ETT.py`, `d_model` is mapped from `args.alpha`; `pe_type` from `args.pe_type` (default `'conv'`); `attn_type` from `args.attn_type` (default `'standard'`); `blocks` must be set ≥ 3.
