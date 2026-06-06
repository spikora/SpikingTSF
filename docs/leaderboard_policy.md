# Leaderboard Policy

This document defines the criteria for results to be included in the SpikingTSF leaderboard.

---

## Acceptance Criteria

A result submission must include:

1. **Model name** — must match an existing model in the repository.
2. **Dataset and horizon** — must follow the protocol in [PROTOCOL.md](../PROTOCOL.md).
3. **Script or command** — the exact script used to produce the result. A reader must be able to reproduce it.
4. **Seeds** — results must use seeds 42, 1234, and 3407 (3 runs). Single-seed results are noted separately and not included in the main leaderboard.
5. **Metrics** — MSE and MAE are required. RMSE and R² are encouraged.
6. **Hardware** — GPU model and CUDA version help identify hardware-dependent variation.
7. **Software versions** — Python, PyTorch, SpikingJelly.

---

## Verification

Results are marked as:

- **Verified** — reproduced by the maintainer or confirmed by a second independent run.
- **Submitted** — received and plausible but not yet independently verified.
- **Pending** — not yet evaluated.

Unverified results may appear in RESULTS.md with a status annotation but will not be listed in the README leaderboard until verified.

---

## Disqualification

Results may be rejected or removed if:

- The submitted numbers cannot be reproduced with the provided script.
- The protocol deviates from [PROTOCOL.md](../PROTOCOL.md) without clear justification.
- Seeds, dataset splits, or normalization differ from the standard.

---

## Submitting Results

Use the [Result Submission issue template](https://github.com/spikora/SpikingTSF/issues/new?template=result_submission.yml).
