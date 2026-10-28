from app import run

def test_run_is_callable():
    assert callable(run)

def test_runtime_has_device_detection():
    from runtime import device_name
    assert device_name() in {"cpu", "cuda", "mps"}
