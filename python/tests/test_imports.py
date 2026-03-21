import importlib


def test_import_jetson_utils_core_symbols():
    ju = importlib.import_module("jetson_utils")
    # Core API presence
    assert hasattr(ju, "Log")
    assert hasattr(ju, "videoSource")
    assert hasattr(ju, "videoOutput")
    assert hasattr(ju, "cudaMemory")
    assert hasattr(ju, "cudaImage")
    # Optional version
    assert isinstance(getattr(ju, "VERSION", "0.0.0"), str)

