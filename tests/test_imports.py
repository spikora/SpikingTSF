"""
Basic import tests for SpikingTSF.
These tests do not require datasets or GPU.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_utils_imports():
    metrics = importlib.import_module("utils.metrics")
    assert hasattr(metrics, "metric"), "utils.metrics should expose a metric function"


def test_data_provider_imports():
    importlib.import_module("data_provider")


def test_exp_imports():
    importlib.import_module("exp")


def test_model_imports():
    model_modules = [
        "models.SpikF",
        "models.Spikformer",
        "models.Spikingformer",
        "models.QKFormer",
        "models.TSGRU",
        "models.TSTCN",
        "models.TSFormer",
        "models.iSpikformer",
        "models.SpikeRNN",
        "models.SpikTCN",
        "models.SpikGRU",
        "models.ITransformer",
        "models.DLinear",
    ]
    failed = []
    for mod_name in model_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failed.append(f"{mod_name}: {exc}")

    if failed:
        raise ImportError(
            "The following model modules failed to import:\n" + "\n".join(failed)
        )


def test_layers_imports():
    importlib.import_module("models.layers")


if __name__ == "__main__":
    test_utils_imports()
    test_data_provider_imports()
    test_exp_imports()
    test_model_imports()
    test_layers_imports()
    print("All import tests passed.")
