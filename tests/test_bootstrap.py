from prompt_experiment import bootstrap_mean_ci

def test_bootstrap_mean_interval_is_reproducible():
    a = bootstrap_mean_ci([1, 2, 3, 4], iterations=1000, seed=42)
    b = bootstrap_mean_ci([1, 2, 3, 4], iterations=1000, seed=42)
    assert a == b
    assert a["mean"] == 2.5
    assert a["lower"] <= a["mean"] <= a["upper"]

def test_bootstrap_single_value_reports_no_interval():
    result = bootstrap_mean_ci([5])
    assert result["mean"] == 5
    assert result["lower"] is None
    assert result["upper"] is None
