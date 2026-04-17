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
