import importlib


def test_logging_api_basic_calls():
    ju = importlib.import_module("jetson_utils")
    # exercise class methods without side effects
    level = ju.Log.GetLevel()
    assert isinstance(level, str)
    ju.Log.Debug("debug")
    ju.Log.Verbose("verbose")
    ju.Log.Info("info")
    ju.Log.Success("success")
    ju.Log.Warning("warning")
    ju.Log.Error("error")
    # usage string sanity
    usage = ju.Log.Usage()
    assert isinstance(usage, str) and len(usage) > 0

