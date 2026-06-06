# SpikTCN — Best Hyperparameters (ETTh1)

Hyperparameters were selected via Optuna with **100 trials**, **300 epochs** per trial,
early stopping patience of 15 epochs, optimising validation MAE on the 96-step horizon.

Fixed across all runs: `T=4`, `pe_mode=add`.

---

## ETTh1

| Hyperparameter   | Value                      |
|------------------|----------------------------|
| **tau**          | 1.5261                     |
| **levels**       | 2                          |
| **alpha**        | 128                        |
| **kernel_size**  | 7                          |
| **encoder_type** | `conv`                     |
| **pe_type**      | `none`                     |
| **lr**           | 2.752 × 10⁻³               |
| **batch_size**   | 64                         |
| **weight_decay** | 1.395 × 10⁻⁵               |
| **scheduler**    | `cosine`                   |
| **grad_clip**    | 5.0                        |

**Best trial:** 62 &nbsp;|&nbsp; **Best val MAE:** 0.77443

### Key characteristics
- Two TCN levels (`levels=2`) with dilation 1→2 gives receptive field ≈ 25 timesteps —
  sufficient for the 96-step input without over-expanding the field.
- Large kernel (`kernel_size=7`) with dilation maximises the causal receptive field while
  keeping parameter count low (`alpha=128` → 128 hidden channels).
- No positional encoding (`pe_type=none`) — the dilated causal convolutions already encode
  relative position implicitly; adding explicit PE did not help.
- Low tau (~1.53) causes fast membrane decay, matching TCN's feed-forward (non-recurrent)
  nature where spike history from distant steps is less important.
- Cosine LR schedule with gradient clipping (5.0) stabilises training with weight-normalised
  convolutions.
