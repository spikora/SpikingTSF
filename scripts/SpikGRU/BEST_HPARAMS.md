# SpikGRU — Best Hyperparameters (ETTh1 & ETTh2)

Hyperparameters were selected via Optuna with **50 trials**, **300 epochs** per trial,
early stopping patience of 15 epochs, optimising validation MAE on the 96-step horizon.

Fixed across both datasets: `T=4`, `pe_mode=add`.

---

## ETTh1

| Hyperparameter  | Value                      |
|-----------------|----------------------------|
| **tau**         | 3.1497                     |
| **levels**      | 1                          |
| **alpha**       | 128                        |
| **encoder_type**| `delta`                    |
| **pe_type**     | `conv`                     |
| **lr**          | 4.842 × 10⁻³               |
| **batch_size**  | 32                         |
| **weight_decay**| 2.501 × 10⁻⁴               |
| **scheduler**   | `cosine`                   |
| **grad_clip**   | 1.0                        |

**Best trial:** 48 &nbsp;|&nbsp; **Best val MAE:** 0.91866

### Key characteristics
- Single GRU level (`levels=1`) — shallow architecture outperformed deeper variants on ETTh1.
- Delta encoder captures temporal differences effectively on this noisier dataset.
- Convolutional positional encoding (`pe_type=conv`) adds local temporal context.
- Cosine LR schedule with gradient clipping stabilises training.

---

## ETTh2

| Hyperparameter  | Value                      |
|-----------------|----------------------------|
| **tau**         | 1.2792                     |
| **levels**      | 2                          |
| **alpha**       | 256                        |
| **encoder_type**| `conv`                     |
| **pe_type**     | `none`                     |
| **lr**          | 4.688 × 10⁻³               |
| **batch_size**  | 32                         |
| **weight_decay**| 3.324 × 10⁻⁴               |
| **scheduler**   | `none`                     |
| **grad_clip**   | 0.0 (disabled)             |

**Best trial:** 20 &nbsp;|&nbsp; **Best val MAE:** 0.43767

### Key characteristics
- Two GRU levels (`levels=2`) with larger hidden size (`alpha=256`) vs ETTh1.
- Convolutional encoder with no positional encoding — the temporal convolution itself
  provides sufficient inductive bias for this smoother dataset.
- Constant LR (no scheduler) and no gradient clipping; the higher `weight_decay` acts
  as the sole regulariser.
- Low membrane time constant (`tau≈1.28`) causes faster spike-rate decay, matching the
  shorter-range dependencies in ETTh2.

---

## Comparison summary

| Parameter       | ETTh1        | ETTh2        |
|-----------------|--------------|--------------|
| tau             | 3.15 (slow)  | 1.28 (fast)  |
| levels          | 1            | 2            |
| alpha           | 128          | 256          |
| encoder_type    | delta        | conv         |
| pe_type         | conv         | none         |
| scheduler       | cosine       | none         |
| grad_clip       | 1.0          | 0.0          |
| best val MAE    | 0.91866      | 0.43767      |

ETTh2 is a significantly easier target (lower MAE), allowing a wider, deeper model
with minimal regularisation. ETTh1 benefits from the delta encoder's explicit
difference representation and a lighter capacity.
