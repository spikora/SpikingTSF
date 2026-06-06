# SpikeRNN — Best Hyperparameters (ETTh1 & ETTh2)

Hyperparameters were selected via Optuna with **100 trials**, **300 epochs** per trial,
early stopping patience of 15 epochs, optimising validation MAE on the 96-step horizon.

Fixed across both datasets: `T=4`, `pe_mode=add`.

---

## ETTh1

| Hyperparameter   | Value                      |
|------------------|----------------------------|
| **tau**          | 3.2032                     |
| **levels**       | 3                          |
| **alpha**        | 128                        |
| **encoder_type** | `conv`                     |
| **pe_type**      | `static`                   |
| **lr**           | 4.538 × 10⁻³               |
| **batch_size**   | 32                         |
| **weight_decay** | 4.606 × 10⁻⁵               |
| **scheduler**    | `none`                     |
| **grad_clip**    | 1.0                        |

**Best trial:** 93 &nbsp;|&nbsp; **Best val MAE:** 0.91796

### Key characteristics
- Three RNN levels (`levels=3`) — deeper hierarchy helps capture multi-scale temporal patterns.
- Convolutional encoder with static sinusoidal PE; the fixed frequency basis provides stable
  position information without adding learnable parameters.
- Very low weight decay (4.6 × 10⁻⁵) suggests the model relies on gradient clipping
  (`grad_clip=1.0`) rather than L2 regularisation for stability.
- High tau (~3.20) means slow membrane decay, retaining longer spike history across time steps.

---

## ETTh2

| Hyperparameter   | Value                      |
|------------------|----------------------------|
| **tau**          | 2.9406                     |
| **levels**       | 3                          |
| **alpha**        | 128                        |
| **encoder_type** | `conv`                     |
| **pe_type**      | `static`                   |
| **lr**           | 4.983 × 10⁻³               |
| **batch_size**   | 32                         |
| **weight_decay** | 9.223 × 10⁻⁴               |
| **scheduler**    | `none`                     |
| **grad_clip**    | 5.0                        |

**Best trial:** 23 &nbsp;|&nbsp; **Best val MAE:** 0.43723

### Key characteristics
- Identical topology to ETTh1 (`levels=3`, `alpha=128`, `encoder=conv`, `pe_type=static`)
  — the RNN architecture generalises well across both datasets.
- Higher weight decay (9.2 × 10⁻⁴) and looser gradient clipping (5.0) vs ETTh1; ETTh2's
  cleaner signal needs less sharp regularisation.
- Slightly lower tau (~2.94 vs 3.20) allows marginally faster membrane decay on ETTh2.

---

## Comparison summary

| Parameter        | ETTh1           | ETTh2           |
|------------------|-----------------|-----------------|
| tau              | 3.20 (slow)     | 2.94 (slow)     |
| levels           | 3               | 3               |
| alpha            | 128             | 128             |
| encoder_type     | conv            | conv            |
| pe_type          | static          | static          |
| scheduler        | none            | none            |
| grad_clip        | 1.0             | 5.0             |
| weight_decay     | 4.6 × 10⁻⁵      | 9.2 × 10⁻⁴      |
| best val MAE     | 0.91796         | 0.43723         |

Unusually consistent: both datasets prefer the same architecture (3 levels, alpha=128,
conv encoder, static PE, no scheduler). The main difference is regularisation strength —
ETTh2's lower MAE target is achieved with higher weight decay and relaxed gradient clipping
rather than a different topology.
