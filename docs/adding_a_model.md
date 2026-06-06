# Adding a New Model

This guide explains how to add a new SNN or ANN forecasting model to SpikingTSF.

---

## Step 1: Add the Model File

Create `models/<YourModel>.py`. The model class should:

- Accept configuration arguments (via the `args` namespace passed from `run_long.py`)
- Implement a `forward(x)` method taking `(batch, seq_len, n_vars)` and returning `(batch, pred_len, n_vars)`
- Follow the interface of existing models (e.g., `models/SpikF.py`, `models/TSGRU.py`)

Include a comment at the top citing the original paper.

---

## Step 2: Register in `exp/exp_ETT.py`

This is where models are actually imported and instantiated. You need to make three additions:

**a) Add the import at the top:**
```python
from models.YourModel import YourModel
```

**b) Add it to the `_MODEL_REGISTRY` dict** (or the equivalent mapping used to build the model):
```python
'YourModel': YourModel,
```

**c) Classify it** by adding its name to the appropriate set:
- `_ANN_MODELS` — if it is an ANN (no spikes)
- `_CD_MODELS` — if it uses `spikingjelly.clock_driven` and needs `cd_functional.reset_net()`
- `_AB_MODELS` — if it uses `spikingjelly.activation_based`
- `_SELF_RESET_MODELS` — if the model resets itself internally during `forward()`

Check which SpikingJelly backend your model uses and pick accordingly.

---

## Step 3: Register in `run_long.py`

Add the model name to the `choices` list for the `--model` argument:

```python
parser.add_argument('--model', ..., choices=[
    ...
    'YourModel',
])
```

---

## Step 4: Add a Config File

Create `configs/<YourModel>/ETTh1.yaml` with default hyperparameters. Use existing YAML configs as templates.

---

## Step 5: Add Scripts

Create `scripts/<YourModel>/run_ETTh1.sh` with the standard 3-seed pattern. Add scripts for other datasets as available.

---

## Step 6: Update `MODEL_ZOO.md`

Add a row with model name, type, Spike?, source paper, source repo, file path, and status.

---

## Step 7: Add Citation

In `MODEL_ZOO.md` and in the model file, provide the BibTeX citation for the original paper.

---

## Step 8: Add Results (if available)

If you have results following the protocol in [PROTOCOL.md](../PROTOCOL.md), add them to [RESULTS.md](../RESULTS.md).

---

## Step 9: Open a Pull Request

Follow the checklist in [CONTRIBUTING.md](../CONTRIBUTING.md) and the PR template.
